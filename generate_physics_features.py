"""Physics-based electrostatic and solvation feature generation.

Computes per-residue physics features from PDB crystal structures using
established analytical models. These features explicitly encode ionic
strength, electrostatic environment, and solvation effects that
sequence-only and PSSM-based models cannot capture.

Physics models implemented
--------------------------
1. Debye-Hückel electrostatics (Debye & Hückel, 1923; Gilson & Honig, 1988)
   Models the screened Coulombic interaction between charged residues in
   solution at a given ionic strength I (mol/L):

       ΔΔG_elec [kcal/mol] = (C / ε_r) · Δq_i · Σ_j [q_j · exp(−r_ij/λD) / r_ij]

   where
     C   = 332.06 kcal·Å/(mol·e²)  [Coulomb constant in molecular units]
     ε_r = 80.0                      [relative permittivity of water at 25°C]
     Δq_i = q_mut − q_wt             [charge change at mutation site]
     q_j  = charge on residue j (Henderson-Hasselbalch at assay pH)
     r_ij = Cα−Cα distance (Å) between sites i and j
     λD   = 3.04 / √I  (Å)          [Debye screening length at 25°C]

   Ionic strength directly modulates λD:
     I = 0.05 M → λD = 13.6 Å  (low salt, long-range interactions)
     I = 0.15 M → λD =  7.8 Å  (physiological)
     I = 0.50 M → λD =  4.3 Å  (high salt, short-range only)

2. Born solvation penalty (Born, 1920; Roux & Simonson, 1999)
   Energy cost of transferring a charged group between aqueous solvent
   (ε_w = 80) and protein interior (ε_p = 4), modulated by RSA:

       ΔΔG_Born [kcal/mol] = (C / 2r_Born) · (q_mut² − q_wt²) · (1/ε_eff − 1/ε_w)

   where ε_eff = ε_p + (ε_w − ε_p) · RSA  (interpolated by burial)
         r_Born = 3.5 Å  (effective Born radius for amino acid side chains)

   A buried charge-introducing mutation pays a large solvation penalty;
   a surface-exposed one pays almost none.

3. Local electrostatic potential (Gilson & Honig, 1988)
   Net electrostatic potential at the mutation site from all neighbouring
   charges within a cutoff of 4·λD:

       Φ_site = (C / ε_r) · Σ_{r<4λD} q_j · exp(−r_ij/λD) / r_ij

4. Local charge density
   Sum of Henderson-Hasselbalch charges within a sphere of radius 2·λD
   around the mutation site. Distinguishes acidic, basic, and neutral
   microenvironments.

5. Nearest-charge Debye factor
   exp(−r_nearest / λD) for the closest charged residue. Indicates how
   strongly ionic strength modulates the dominant electrostatic interaction.

6. Electrostatic burial coupling
   Φ_site · (1 − RSA): the electrostatic environment weighted by burial.
   Buried sites in regions of high electrostatic potential are particularly
   sensitive to charge mutations.

Output
------
backend/app/trained_models/physics_features_cache.pkl
  { protein_id (str) : { resnum (int) : np.ndarray of shape (8,) } }

Feature vector layout (8 features per residue)
  [0] ddg_elec_dh        : ΔΔG_elec at physiological I=0.15 M (kcal/mol)
  [1] ddg_born           : ΔΔG_Born solvation penalty (kcal/mol; requires RSA)
                           Set to 0.0 if RSA unavailable — filled later at
                           training time using the model's RSA feature.
  [2] phi_site           : electrostatic potential at mutation site (kcal/mol/e)
  [3] q_local            : net charge density within 2·λD sphere
  [4] debye_factor_near  : exp(−r_nearest_charge/λD)  [0-1, ionic sensitivity]
  [5] elec_burial        : Φ_site · (1−RSA) burial coupling  [0 if RSA unknown]
  [6] contact_norm       : Cα contacts within 8Å / 20 (packing density proxy)
  [7] bfactor_z          : B-factor z-score within chain, clamped [-3,3]
                           (crystallographic thermal motion / local flexibility)

Note on ionic strength: all features are computed at I = 0.15 M (physiological
default) during cache generation. At training/inference time, the ionic_strength
feature (added in v17) allows the model to learn condition-dependent scaling.
For a future database that records assay ionic strength, the cache can be
regenerated with per-record I values.

References
----------
Debye P, Hückel E (1923) Phys Z 24:185-206.
Born M (1920) Z Phys 1:45-48.
Gilson MK, Honig B (1988) Proteins 4:7-18.
Roux B, Simonson T (1999) Biophys Chem 78:1-20.
Yuan Z et al. (2005) Proteins 58:905-912.
Radivojac P et al. (2004) Biophys J 86:1-10.
"""

import os
import io
import json
import time
import pickle
import urllib.request
import urllib.error
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREPROT_PATH   = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
THERMOMUTDB_PATH = os.path.join(BASE_DIR, "thermomutdb.json")
PDB_CACHE_DIR   = os.path.join(BASE_DIR, "pdb_structures")   # local PDB file cache
OUTPUT_PATH     = os.path.join(BASE_DIR, "backend/app/trained_models/physics_features_cache.pkl")

os.makedirs(PDB_CACHE_DIR, exist_ok=True)

# ── Physical constants (molecular units) ──────────────────────────────────────
COULOMB_CONST = 332.06      # kcal·Å / (mol · e²)
EPSILON_WATER = 80.0        # relative permittivity of water at 25°C
EPSILON_PROT  = 4.0         # effective permittivity of protein interior
BORN_RADIUS   = 3.5         # Å  — effective Born radius for amino acid side chains
ION_STRENGTH  = 0.15        # M  — physiological default (NaCl equivalent)
TEMP_K        = 298.15      # K

# Debye screening length at 25°C: λD = 3.04 / sqrt(I)  (Å, I in mol/L)
def debye_length(ionic_strength_M):
    return 3.04 / np.sqrt(max(ionic_strength_M, 1e-6))

LAMBDA_D = debye_length(ION_STRENGTH)    # ≈ 7.84 Å at I=0.15 M

# ── Henderson-Hasselbalch charges at pH 7.0 ───────────────────────────────────
# Fraction ionised at pH 7.0, multiplied by the formal charge of the ionised form.
# For acids (D,E,C,Y): pKa < pH → mostly deprotonated (negative)
# For bases (K,R,H):   pKa > pH → mostly protonated (positive)
AA_CHARGE_PH7 = {
    'D': -0.99,   # pKa 3.9  → fully deprotonated
    'E': -0.99,   # pKa 4.1  → fully deprotonated
    'K': +0.99,   # pKa 10.5 → fully protonated
    'R': +1.00,   # pKa 12.5 → fully protonated
    'H': +0.09,   # pKa 6.0  → ~9% protonated at pH 7
    'C': -0.01,   # pKa 8.3  → ~1% deprotonated at pH 7
    'Y': -0.01,   # pKa 10.1 → ~1% deprotonated at pH 7
}

def residue_charge(aa, ph=7.0):
    """Henderson-Hasselbalch net charge at given pH."""
    pka_map = {'D': 3.9, 'E': 4.1, 'C': 8.3, 'Y': 10.1,
               'H': 6.0, 'K': 10.5, 'R': 12.5}
    acid_set = {'D', 'E', 'C', 'Y'}
    pka = pka_map.get(aa, 0.0)
    if pka == 0.0:
        return 0.0
    if aa in acid_set:
        # acid: deprotonated form has charge -1; Henderson-Hasselbalch: f_deprot = 10^(pH-pKa)/(1+10^(pH-pKa)) = 1/(1+10^(pKa-pH))
        return -1.0 / (1.0 + 10.0 ** (pka - ph))
    else:
        # base: protonated form has charge +1; f_prot = 1/(1+10^(pH-pKa))
        return +1.0 / (1.0 + 10.0 ** (ph - pka))

# ── PDB downloading and parsing ────────────────────────────────────────────────

def fetch_pdb(pdb_id, max_retries=3):
    """Return PDB text, using local cache first."""
    local = os.path.join(PDB_CACHE_DIR, f"{pdb_id.upper()}.pdb")
    if os.path.exists(local):
        with open(local) as f:
            return f.read()
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                text = r.read().decode('utf-8', errors='ignore')
            with open(local, 'w') as f:
                f.write(text)
            return text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def parse_ca_coordinates(pdb_text):
    """Extract Cα coordinates and residue identities from PDB ATOM records.

    Returns:
        list of (resnum, resname, chain, x, y, z) for all Cα atoms,
        one entry per residue position (first chain only).
    """
    if not pdb_text:
        return []

    ca_atoms = {}   # resnum → (resname, chain, xyz, bfactor)
    first_chain = None

    for line in pdb_text.splitlines():
        if not line.startswith('ATOM'):
            continue
        atom_name = line[12:16].strip()
        if atom_name != 'CA':
            continue
        try:
            chain   = line[21]
            resnum  = int(line[22:26])
            resname = line[17:20].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bfactor = float(line[60:66]) if len(line) >= 66 else 0.0
        except (ValueError, IndexError):
            continue

        if first_chain is None:
            first_chain = chain
        if chain != first_chain:
            continue   # only first chain

        if resnum not in ca_atoms:
            ca_atoms[resnum] = (resname, chain, np.array([x, y, z]), bfactor)

    return [(resnum, data[0], data[1], data[2], data[3])
            for resnum, data in sorted(ca_atoms.items())]


def three_to_one(resname):
    """Convert 3-letter amino acid code to 1-letter."""
    table = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'MSE': 'M',   # selenomethionine
        'HSD': 'H', 'HSE': 'H', 'HSP': 'H',   # CHARMM histidine variants
    }
    return table.get(resname.upper(), None)


# ── Physics feature computation ────────────────────────────────────────────────

def compute_physics_features(ca_list, ph=7.0, ionic_strength_M=ION_STRENGTH):
    """Compute physics-based features for every residue position.

    Parameters
    ----------
    ca_list : list of (resnum, resname_3, chain, xyz_array, bfactor)
    ph      : assay pH (default 7.0)
    ionic_strength_M : assay ionic strength in mol/L (default 0.15)

    Returns
    -------
    dict { resnum (int) : np.ndarray of shape (8,) }
    """
    lam = debye_length(ionic_strength_M)   # Debye length (Å)
    cutoff_elec = 4.0 * lam                # interaction cutoff

    # Build arrays for vectorised computation
    resnums  = np.array([r[0] for r in ca_list])
    xyz      = np.array([r[3] for r in ca_list], dtype=np.float64)  # (N, 3)
    aas      = [three_to_one(r[1]) for r in ca_list]
    charges  = np.array([residue_charge(aa, ph) if aa else 0.0
                         for aa in aas], dtype=np.float64)
    # B-factors: z-score within chain (clamped to [-3, 3])
    bfactors = np.array([r[4] if len(r) > 4 else 0.0 for r in ca_list],
                        dtype=np.float64)
    bf_mean  = float(np.mean(bfactors))
    bf_std   = float(np.std(bfactors)) if np.std(bfactors) > 0.1 else 1.0
    bfactor_z = np.clip((bfactors - bf_mean) / bf_std, -3.0, 3.0)

    N = len(resnums)
    features = {}

    for i in range(N):
        resnum_i = int(resnums[i])
        aa_i = aas[i]
        if aa_i is None:
            continue   # non-standard residue

        # Pairwise distances to all other Cα atoms
        diff = xyz - xyz[i]                          # (N, 3)
        dists = np.sqrt((diff**2).sum(axis=1))       # (N,)
        dists[i] = np.inf                            # exclude self

        # Mask to residues within electrostatic cutoff and charged
        mask = (dists < cutoff_elec) & (dists > 0.1) & (np.abs(charges) > 0.01)
        r_near    = dists[mask]
        q_near    = charges[mask]

        # ── Feature 0: ΔΔG_elec_DH ─────────────────────────────────────────
        # Change in Debye-Hückel electrostatic energy upon mutation.
        # Δq_i will be filled per-mutation at training time; here we store the
        # precomputed interaction sum Σ[q_j·exp(−r/λD)/r] for this site.
        # At training time: ΔΔG_elec = Δq_i * interaction_sum * (C/ε_r)
        if len(r_near) > 0:
            dh_weights = np.exp(-r_near / lam) / r_near   # Debye-Hückel kernel
            interaction_sum = float(np.sum(q_near * dh_weights))
            # Store as a potential; multiply by Δq at training time
            ddg_elec_scale = COULOMB_CONST / EPSILON_WATER * interaction_sum
        else:
            ddg_elec_scale = 0.0

        # ── Feature 1: ΔΔG_Born solvation (charge-squared term only) ───────
        # Full expression requires RSA; store the geometry-independent prefactor.
        # At training time: ΔΔG_Born = ddg_born_coeff * (q_mut² − q_wt²)
        # Born coefficient = -(C / 2*r_Born) * (1/ε_w - 1/ε_eff(RSA))
        # Here we store the ε_eff denominator factor; RSA applied at training.
        # Simplified: assuming RSA = 0.5 (average) for the cache value.
        rsa_approx = 0.5
        epsilon_eff = EPSILON_PROT + (EPSILON_WATER - EPSILON_PROT) * rsa_approx
        ddg_born_coeff = -(COULOMB_CONST / (2 * BORN_RADIUS)) * (
            1.0 / epsilon_eff - 1.0 / EPSILON_WATER)
        # Per-residue value: charge-environment factor only (dimensionless)
        ddg_born_scale = float(ddg_born_coeff)

        # ── Feature 2: Electrostatic potential Φ_site ──────────────────────
        # Absolute potential at position i from all neighbouring charges.
        # Positive = dominated by nearby positive residues; negative = vice versa.
        if len(r_near) > 0:
            phi_site = float(
                COULOMB_CONST / EPSILON_WATER *
                np.sum(q_near * np.exp(-r_near / lam) / r_near))
        else:
            phi_site = 0.0

        # ── Feature 3: Local charge density within 2·λD ────────────────────
        mask_local = (dists < 2 * lam) & (dists > 0.1)
        q_local = float(np.sum(charges[mask_local]))

        # ── Feature 4: Debye factor for nearest charged residue ─────────────
        # Measures how strongly ionic screening suppresses the dominant
        # pairwise electrostatic interaction. Ranges 0 (far) to 1 (very near).
        charged_dists = dists.copy()
        charged_dists[np.abs(charges) < 0.01] = np.inf
        r_near_charge = float(np.min(charged_dists)) if np.any(np.isfinite(charged_dists)) else 30.0
        debye_factor = float(np.exp(-r_near_charge / lam))

        # ── Feature 5: Electrostatic burial coupling ─────────────────────────
        # Φ_site * (1 − RSA_approx): buried sites with high electrostatic
        # potential are most sensitive to charge mutations.
        # RSA will be overridden at training time; here use 0.5 average.
        elec_burial = phi_site * (1.0 - rsa_approx)

        # ── Feature 6: Contact number (Cα within 8Å) ─────────────────────────
        # Well-established structural measure of burial/packing density.
        # Correlates with stability: buried residues (high contact number) have
        # larger stability contributions than surface residues.
        # Normalised by dividing by 20 (typical max ~17–18 for buried residues).
        # Reference: Selvaraj & Gromiha (2003) Proteins 53, 546–557.
        CONTACT_CUTOFF_8A = 8.0
        n_contacts = float(np.sum((dists < CONTACT_CUTOFF_8A) & (dists > 0.1)))
        contact_norm = n_contacts / 20.0   # normalise to ~[0,1] range

        # ── Feature 7: B-factor z-score (crystallographic thermal motion) ────
        # Debye-Waller / temperature factor: measures mean-square atomic
        # displacement. High B-factor → flexible loop/hinge → mutations often
        # have smaller ΔΔG in magnitude. Z-scored within the chain to remove
        # inter-structure resolution bias. Clamped to [-3, 3].
        # Reference: Yuan et al. (2005) Proteins 58, 905–912;
        #            Radivojac et al. (2004) Biophys J 86, 1–10.
        bfz = float(bfactor_z[i])

        features[resnum_i] = np.array([
            ddg_elec_scale,    # [0] DH interaction sum (multiply by Δq at train time)
            ddg_born_scale,    # [1] Born solvation coefficient (multiply by Δq² at train)
            phi_site,          # [2] absolute electrostatic potential (kcal/mol/e)
            q_local,           # [3] local net charge (e)
            debye_factor,      # [4] Debye screening factor [0-1]
            elec_burial,       # [5] Φ_site × (1-RSA_approx)
            contact_norm,      # [6] Cα contact number / 20 (burial/packing proxy)
            bfz,               # [7] B-factor z-score (relative flexibility, [-3,3])
        ], dtype=np.float32)

    return features


def get_training_pdb_ids():
    pdb_ids = set()
    df = pd.read_csv(FIREPROT_PATH)
    for v in df['pdb_id'].dropna():
        pid = str(v).split('|')[0].strip().upper()[:4]
        if len(pid) == 4:
            pdb_ids.add(pid)
    with open(THERMOMUTDB_PATH) as f:
        data = json.load(f)
    for entry in data:
        pid = str(entry.get('PDB_wild', '')).strip().upper()[:4]
        if len(pid) == 4:
            pdb_ids.add(pid)
    return sorted(pdb_ids)


def main():
    print("=" * 70)
    print("PHYSICS FEATURE GENERATION — Debye-Hückel + Born Solvation")
    print("I = 0.15 M (physiological), pH = 7.0, T = 25°C")
    print("Reference: Debye & Hückel (1923); Born (1920); Gilson & Honig (1988)")
    print("=" * 70)

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'rb') as f:
            existing = pickle.load(f)
        print(f"\nExisting cache: {len(existing)} proteins.")
        total = sum(len(v) for v in existing.values())
        print(f"Total residue positions with physics features: {total}")
        return existing

    pdb_ids = get_training_pdb_ids()
    print(f"\nProcessing {len(pdb_ids)} PDB structures...")
    print(f"Debye length at I={ION_STRENGTH} M: {LAMBDA_D:.2f} Å\n")

    cache = {}
    n_failed = 0
    n_residues_total = 0
    t0 = time.time()

    for i, pdb_id in enumerate(pdb_ids):
        pdb_text = fetch_pdb(pdb_id)

        if pdb_text is None:
            n_failed += 1
            cache[pdb_id] = {}
        else:
            ca_list = parse_ca_coordinates(pdb_text)
            if ca_list:
                feats = compute_physics_features(ca_list)
                cache[pdb_id] = feats
                n_residues_total += len(feats)
            else:
                cache[pdb_id] = {}

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pdb_ids) - i - 1) / rate
            print(f"  [{i+1:4d}/{len(pdb_ids)}]  "
                  f"residues_cached={n_residues_total:6d}  "
                  f"failed={n_failed}  "
                  f"ETA={eta/60:.1f} min")

        # Polite rate limiting
        if i % 5 == 0:
            time.sleep(0.05)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"  Structures processed: {len(pdb_ids)}")
    print(f"  Failed (no PDB):      {n_failed}")
    print(f"  Residue positions:    {n_residues_total}")
    print(f"  Mean coverage:        "
          f"{n_residues_total/max(len(pdb_ids)-n_failed,1):.0f} res/protein")

    # Sample feature statistics
    all_feats = np.vstack([v for vs in cache.values()
                           for v in vs.values() if len(vs) > 0])
    feat_names = ['DH_interaction', 'Born_coeff', 'Phi_site',
                  'Q_local', 'Debye_factor', 'Elec_burial', 'Contact_norm',
                  'Bfactor_z']
    print("\nPhysics feature statistics across all residue positions:")
    print(f"  {'Feature':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    for j, name in enumerate(feat_names):
        col = all_feats[:, j]
        print(f"  {name:<20} {col.mean():>8.3f} {col.std():>8.3f} "
              f"{col.min():>8.3f} {col.max():>8.3f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(cache, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"\nSaved → {OUTPUT_PATH} ({size_mb:.1f} MB)")
    return cache


if __name__ == '__main__':
    main()
