"""Generate metal coordination cache from PDB structures.

For each training protein, downloads the PDB structure from RCSB and identifies
residues that coordinate metal ions (Ca²⁺, Zn²⁺, Mg²⁺, Fe, Mn, Cu, Ni, Co)
using crystallographic distances.

Coordination distance cutoffs (Å):
  Ca²⁺ :  3.0  (outer-sphere coordination common)
  Zn²⁺ :  2.5
  Mg²⁺ :  2.5
  Fe   :  2.5
  Mn²⁺ :  2.5
  Cu²⁺ :  2.5
  Ni²⁺ :  2.5
  Co²⁺ :  2.5

Output: metal_coord_cache.pkl
  { protein_id (str) : { resnum (int) : set of metal symbols } }

Used by train_v17.py to add features:
  is_metal_coordinating     — 1 if any metal within cutoff
  is_ca2_coordinating       — 1 if Ca²⁺ specifically
  is_zn_coordinating        — 1 if Zn²⁺ specifically
  n_coordinated_metal_types — count of distinct metal types at site
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREPROT_PATH = os.path.join(BASE_DIR,
    "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
THERMOMUTDB_PATH = os.path.join(BASE_DIR, "thermomutdb.json")
OUTPUT_PATH = os.path.join(BASE_DIR,
    "backend/app/trained_models/metal_coord_cache.pkl")

# Metal HETATM residue names (PDB 3-letter codes) and their common symbols
METAL_IDS = {
    'CA':  'CA2',   # Calcium
    'ZN':  'ZN2',   # Zinc
    'MG':  'MG2',   # Magnesium
    'FE':  'FE',    # Iron
    'MN':  'MN2',   # Manganese
    'CU':  'CU2',   # Copper
    'NI':  'NI2',   # Nickel
    'CO':  'CO2',   # Cobalt
    'CD':  'CD2',   # Cadmium
    'HG':  'HG2',   # Mercury
    'FE2': 'FE2',   # Ferrous iron
    'FE3': 'FE3',   # Ferric iron
}

# Per-metal coordination distance cutoffs (Å)
# Slightly generous to capture outer-sphere coordination
COORD_CUTOFF = {
    'CA2': 3.0,
    'ZN2': 2.8,
    'MG2': 2.8,
    'FE':  2.8,
    'FE2': 2.8,
    'FE3': 2.8,
    'MN2': 2.8,
    'CU2': 2.8,
    'NI2': 2.8,
    'CO2': 2.8,
    'CD2': 2.8,
    'HG2': 2.8,
}
DEFAULT_CUTOFF = 3.0


def fetch_pdb_structure(pdb_id, max_retries=3):
    """Download PDB file as text from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return r.read().decode('utf-8', errors='ignore')
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # Structure doesn't exist in PDB format (use mmCIF)
            if attempt < max_retries - 1:
                time.sleep(1.5)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.5)
    return None


def parse_metal_coordination(pdb_text):
    """Parse PDB file to find residues within coordination distance of metals.

    Returns:
        dict: { auth_seq_id (int) : set of metal_symbol strings }
    """
    if not pdb_text:
        return {}

    # Collect all ATOM/HETATM records
    atom_records = []   # (resname, chain, resnum, x, y, z, is_hetatm)
    metal_records = []  # (metal_symbol, x, y, z)

    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec not in ('ATOM', 'HETATM'):
            continue
        try:
            resname = line[17:20].strip()
            chain   = line[21:22].strip()
            resnum  = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        if rec == 'HETATM' and resname in METAL_IDS:
            metal_records.append((METAL_IDS[resname], np.array([x, y, z])))
        else:
            atom_records.append((resname, chain, resnum, np.array([x, y, z])))

    if not metal_records:
        return {}

    # For each metal, find all residues within cutoff
    coord_map = defaultdict(set)  # resnum → set of metal symbols
    for metal_sym, metal_xyz in metal_records:
        cutoff = COORD_CUTOFF.get(metal_sym, DEFAULT_CUTOFF)
        for resname, chain, resnum, atom_xyz in atom_records:
            dist = float(np.linalg.norm(atom_xyz - metal_xyz))
            if dist <= cutoff:
                coord_map[resnum].add(metal_sym)

    return dict(coord_map)


def get_training_pdb_ids():
    """Collect all unique PDB IDs from FireProtDB and ThermoMutDB."""
    import pandas as pd
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
    print("=" * 60)
    print("METAL COORDINATION CACHE GENERATION")
    print("Source: RCSB PDB crystallographic structures")
    print("Metals: Ca²⁺, Zn²⁺, Mg²⁺, Fe, Mn, Cu, Ni, Co, Cd, Hg")
    print("=" * 60)

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'rb') as f:
            existing = pickle.load(f)
        print(f"Existing cache found: {len(existing)} proteins")
        metals = {k: v for k, v in existing.items() if v}
        print(f"  Proteins with ≥1 metal site: {len(metals)}")
        coord_residues = sum(len(v) for v in existing.values())
        print(f"  Total metal-coordinating residues: {coord_residues}")
        return existing

    pdb_ids = get_training_pdb_ids()
    print(f"\nProcessing {len(pdb_ids)} unique PDB structures...\n")

    cache = {}       # protein_id → {resnum: set of metal symbols}
    n_with_metals = 0
    n_failed = 0
    t0 = time.time()

    for i, pdb_id in enumerate(pdb_ids):
        pdb_text = fetch_pdb_structure(pdb_id)
        if pdb_text is None:
            n_failed += 1
            cache[pdb_id] = {}
        else:
            coord_map = parse_metal_coordination(pdb_text)
            cache[pdb_id] = coord_map
            if coord_map:
                n_with_metals += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(pdb_ids) - i - 1) / rate
            print(f"  [{i+1}/{len(pdb_ids)}] "
                  f"with_metals={n_with_metals}  "
                  f"failed={n_failed}  "
                  f"ETA={remaining/60:.1f}min")

        time.sleep(0.03)  # polite RCSB rate limit

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Structures processed: {len(pdb_ids)}")
    print(f"  With ≥1 metal site:   {n_with_metals} "
          f"({100*n_with_metals/len(pdb_ids):.1f}%)")
    print(f"  Failed (no PDB file): {n_failed}")

    all_metals = set()
    for v in cache.values():
        for metals in v.values():
            all_metals.update(metals)
    print(f"  Distinct metal types found: {sorted(all_metals)}")

    # Summary by metal type
    metal_counts = defaultdict(int)
    for v in cache.values():
        for metals in v.values():
            for m in metals:
                metal_counts[m] += 1
    print("\nMetal-coordinating residue counts by ion:")
    for metal, count in sorted(metal_counts.items(), key=lambda x: -x[1]):
        print(f"  {metal}: {count} residues")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(cache, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"\nSaved → {OUTPUT_PATH} ({size_mb:.1f} MB)")
    return cache


if __name__ == '__main__':
    main()
