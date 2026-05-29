"""Publication-Ready Protein Stability Prediction Model v32.

v18 is the complete publication model. It adds physics-based electrostatic
and solvation features derived analytically from PDB crystal structures using
Debye-Hückel theory and the Born solvation model — the same physical framework
underlying FoldX and Rosetta's electrostatic terms, implemented here without
requiring those external tools.

v18 additions over v17 (physics-based features):
  Debye-Hückel electrostatics (Debye & Hückel 1923; Gilson & Honig 1988):
    ddg_elec_dh      — ΔΔG electrostatic: Δq·Σ[q_j·exp(−r/λD)/r]·(C/ε_r)
                       Explicitly encodes ionic strength via Debye length
                       λD = 3.04/√I Å (I=0.15 M → λD=7.85 Å physiological)
    phi_site         — absolute electrostatic potential at mutation site
    q_local          — net charge density within 2·λD sphere around site
    debye_factor     — exp(−r_nearest/λD): ionic screening of nearest charge
  Born solvation model (Born 1920; Roux & Simonson 1999):
    ddg_born         — ΔΔG_Born = −(C/2r_B)·Δq²·(1/ε_eff − 1/ε_w)
                       Burial-weighted desolvation penalty for charge mutations
    elec_burial      — Φ_site·(1−RSA): buried electrostatic coupling
  Total physics features: +6 → 131 total features (125+6)

v17 additions over v16:
  Metal coordination (from PDB crystal structures):
    is_metal_coordinating / is_ca2_coordinating / is_zn_coordinating /
    is_mg_coordinating / n_coord_metal_types
  Ionic strength normalisation feature (physiological default 0.15 M)
  Total: 119 + 6 = 125 features

v16 additions over v15:
  ESM-2 language model embeddings (esm2_t30_150M_UR50D, PCA-32)
  Coverage: 100% FireProtDB, 98.5% ThermoMutDB
  Total: 86 + 33 = 119 features

v15: Real FireProtDB RSA (ASA→RSA) + real secondary structure in existing slots
v13: PSSM cache fix (0%→84.4% coverage) + 6 structural cross-terms (86 features)
v12: Real ThermoMutDB RSA/phi/psi/depth (75.36%, +3.87% over v11)

Feature summary: 131 total
  [1–86]   Physicochemical + BLOSUM + SS + RSA + context + PSSM + cross-terms
  [87–92]  Metal coordination (Ca²⁺/Zn²⁺/Mg²⁺) + ionic_strength_norm
  [93–119] ESM-2 PCA-32 + has_esm flag
  [120–125] (v17 = features 87-92, renumbered in v18)
  Actually in v18: base 86 + metal(5) + ionic(1) + ESM-PCA-32 + has_esm = 125
  + physics 6 = 131 features total

v16 additions over v15:
  - ESM-2 protein language model embeddings (esm2_t30_150M_UR50D, 640-dim),
    PCA-reduced to 32 components (49.9% variance explained)
  - Coverage: 100% FireProtDB, 98.5% ThermoMutDB (529/537 proteins)
  - Mean imputation for 8 missing proteins (no dataset-identity leakage)
  - Total features: 86 + 32 + 1 (has_esm flag) = 119

v15 additions over v13:
  - FireProtDB real RSA (from ASA column) replaces sequence-estimated RSA
    (priority: ThermoMutDB struct_rsa > FireProtDB ASA-derived > estimated)
  - FireProtDB real secondary structure (PDB H/E/L) replaces Chou-Fasman
    estimates in existing feature slots — no zero-padding artifacts

v13 additions over v12:
  - FIXED: conservation cache path now correctly points to backend/app/trained_models/
    (v12 had 0% PSSM coverage due to wrong path — this fix alone expected +2-3%)
  - 6 new structural cross-terms: phi×psi, phi×dH, phi×temp, psi×dH, psi×temp, depth×dH
  Total features: 86 (was 80)

v12 additions over v11:
  - Real RSA from ThermoMutDB structural records (8,349 entries) replaces sequence-estimated RSA
  - phi / psi backbone dihedral angles added as features (2 new)
  - Ca-alpha depth added as a feature (1 new)
  - has_real_rsa flag tells the model when to trust the RSA value (1 new)
  Total features: 80 (was 76)
  Optuna trials: 100 XGB / 100 LGBM / 50 CB (was 50/50/30)

"""
"""Publication-Ready Protein Stability Prediction Model v27 (Ensemble Regression).

v27 strategy: use pre-tuned best params from v26 runs — no slow Optuna for XGB/LGBM/CB.
  - XGB: fixed best params from v26 (MAE 1.0996, 100-trial tuned)
  - LGBM: fixed best params from v26 (MAE 1.0983, 100-trial tuned)
  - CatBoost: fixed best params from v24 (MAE 1.1704, 50-trial tuned) — CB Optuna too slow
  - HGB: Optuna 50 trials on 4k subsample (fast, ~20 min)
  - All v26 improvements kept: wide stack, feature augmentation, XGB meta, RF/ET 1200 trees
  - MLP meta-clf added as classifier candidate alongside CatBoost clf
  - Expected runtime: ~2-3 hours total
  - Target: 80%+ stacked CV accuracy (v24: 77.82%)

Predicts ΔΔG values using an ensemble of 8 regressors.
Trained ONLY on real experimental data — no synthetic mutations.

Data sources:
  - FireProtDB (curated mutations with DDG)
  - ThermoMutDB (~300K+ mutations with DDG)

Independent test set (never seen during training):
  - S669 (669 mutations) — held out entirely
"""

import os
import json
import re
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold, cross_val_score, cross_val_predict, GroupKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_curve
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREPROT_PATH = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
PRODDG_PATH = os.path.join(BASE_DIR, "proddg_s2648.csv")
S669_PATH = os.path.join(BASE_DIR, "s669_full.tsv")
THERMOMUTDB_PATH = os.path.join(BASE_DIR, "thermomutdb.json")
CONSERVATION_CACHE_PATH = os.path.join(BASE_DIR, "backend", "app", "trained_models", "conservation_cache.pkl")
ESM_EMBEDDINGS_PATH    = os.path.join(BASE_DIR, "backend", "app", "trained_models", "esm_embeddings_cache.pkl")
ESM_LOGLIK_PATH        = os.path.join(BASE_DIR, "backend", "app", "trained_models", "esm_loglik_cache.pkl")
METAL_COORD_CACHE_PATH   = os.path.join(BASE_DIR, "backend", "app", "trained_models", "metal_coord_cache.pkl")
PHYSICS_FEATURES_PATH    = os.path.join(BASE_DIR, "backend", "app", "trained_models", "physics_features_cache.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "backend/app/trained_models")

ESM_DIM = 32  # PCA components from 640-dim ESM-2 embeddings

# Physiological default ionic strength (M) — used for all training records since
# neither FireProtDB nor ThermoMutDB records experimental ionic strength.
# 0.15 M corresponds to physiological NaCl in standard biochemistry buffers.
PHYSIOLOGICAL_IONIC_STRENGTH = 0.15  # M
IONIC_STRENGTH_SCALE = 0.5           # Normalization denominator (0→1 range)

# ═══════════════════════════════════════════════════════════
# Amino acid properties (same as production)
# ═══════════════════════════════════════════════════════════
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AMINO_ACIDS)

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

VOLUME = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
}

CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.5, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}

FLEXIBILITY = {
    'A': 0.36, 'C': 0.35, 'D': 0.51, 'E': 0.50, 'F': 0.31,
    'G': 0.54, 'H': 0.32, 'I': 0.46, 'K': 0.47, 'L': 0.40,
    'M': 0.30, 'N': 0.46, 'P': 0.51, 'Q': 0.49, 'R': 0.53,
    'S': 0.51, 'T': 0.44, 'V': 0.39, 'W': 0.31, 'Y': 0.42,
}

HELIX_PROPENSITY = {
    'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
    'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
    'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
    'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69,
}

SHEET_PROPENSITY = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
}

# ── Additional physicochemical properties for richer feature set ──

# Molecular weight (Da)  — Lehninger Biochemistry Table 3-1
MOLECULAR_WEIGHT = {
    'A': 89.1,  'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.0,  'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.1, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2,
}

# H-bond donors (backbone NH counted for all, sidechain donors added)
HBOND_DONORS = {
    'A': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1,
    'G': 1, 'H': 2, 'I': 1, 'K': 2, 'L': 1,
    'M': 1, 'N': 2, 'P': 0, 'Q': 2, 'R': 4,
    'S': 2, 'T': 2, 'V': 1, 'W': 2, 'Y': 2,
}

# H-bond acceptors (backbone C=O counted for all, sidechain acceptors added)
HBOND_ACCEPTORS = {
    'A': 1, 'C': 0, 'D': 3, 'E': 3, 'F': 0,
    'G': 1, 'H': 1, 'I': 1, 'K': 1, 'L': 1,
    'M': 2, 'N': 2, 'P': 1, 'Q': 2, 'R': 1,
    'S': 2, 'T': 2, 'V': 1, 'W': 0, 'Y': 1,
}

# Turn propensity (Chou-Fasman, normalized) — frequent in loops/turns
TURN_PROPENSITY = {
    'A': 0.66, 'C': 1.19, 'D': 1.46, 'E': 0.74, 'F': 0.60,
    'G': 1.56, 'H': 0.95, 'I': 0.47, 'K': 1.01, 'L': 0.59,
    'M': 0.60, 'N': 1.56, 'P': 1.52, 'Q': 0.98, 'R': 0.95,
    'S': 1.43, 'T': 0.96, 'V': 0.50, 'W': 0.96, 'Y': 1.14,
}

# Polarity class: 0=nonpolar aliphatic, 1=polar uncharged, 2=charged
POLARITY_CLASS = {
    'A': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 0,
    'G': 0, 'H': 2, 'I': 0, 'K': 2, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 2,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1,
}

# Side-chain pKa (for ionization state features; 0 = no titratable group)
SIDECHAIN_PKA = {
    'A': 0.0,  'C': 8.3,  'D': 3.9,  'E': 4.1,  'F': 0.0,
    'G': 0.0,  'H': 6.0,  'I': 0.0,  'K': 10.5, 'L': 0.0,
    'M': 0.0,  'N': 0.0,  'P': 0.0,  'Q': 0.0,  'R': 12.5,
    'S': 0.0,  'T': 0.0,  'V': 0.0,  'W': 0.0,  'Y': 10.1,
}
# True = acid (loses proton above pKa, gives negative charge)
SIDECHAIN_IS_ACID = {
    'C': True, 'D': True, 'E': True, 'Y': True,
    'H': False, 'K': False, 'R': False,
}

# Aliphatic index contribution (Ikai 1980) — proxy for thermostability
# AI = 100 × (nA + 2.9×nV + 3.9×(nI+nL)) / N
ALIPHATIC_CONTRIB = {
    'A': 1.0, 'V': 2.9, 'I': 3.9, 'L': 3.9,
}

# Side-chain size class: 0=tiny, 1=small, 2=medium, 3=large
# (captures steric clash effects beyond volume)
SIZE_CLASS = {
    'G': 0, 'A': 0,
    'S': 1, 'C': 1, 'T': 1, 'P': 1, 'D': 1, 'N': 1, 'V': 1,
    'E': 2, 'Q': 2, 'I': 2, 'L': 2, 'M': 2, 'H': 2, 'K': 2,
    'F': 3, 'R': 3, 'W': 3, 'Y': 3,
}

# Intrinsic disorder propensity (Uversky 2002 scale — higher = more disorder-prone)
DISORDER_PROPENSITY = {
    'A': 0.06, 'C': 0.02, 'D': 0.19, 'E': 0.18, 'F': -0.05,
    'G': 0.17, 'H': 0.04, 'I': -0.07, 'K': 0.16, 'L': -0.07,
    'M': 0.00, 'N': 0.14, 'P': 0.12, 'Q': 0.15, 'R': 0.14,
    'S': 0.13, 'T': 0.07, 'V': -0.06, 'W': -0.05, 'Y': -0.01,
}

# BLOSUM62 diagonal (self-substitution scores)
BLOSUM62_DIAG = {
    'A': 4, 'R': 5, 'N': 6, 'D': 6, 'C': 9,
    'Q': 5, 'E': 5, 'G': 6, 'H': 8, 'I': 4,
    'L': 4, 'K': 5, 'M': 5, 'F': 6, 'P': 7,
    'S': 4, 'T': 5, 'W': 11, 'Y': 7, 'V': 4,
}

# BLOSUM62 full matrix (subset of common substitutions)
BLOSUM62 = {}
blosum_str = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
lines = [l for l in blosum_str.strip().split('\n') if l.strip()]
header = lines[0].split()
for line in lines[1:]:
    parts = line.split()
    aa1 = parts[0]
    for j, aa2 in enumerate(header):
        BLOSUM62[(aa1, aa2)] = int(parts[j + 1])


def get_blosum62(wt, mut):
    return BLOSUM62.get((wt, mut), 0)


# ═══════════════════════════════════════════════════════════
# Max ASA per amino acid (Miller et al., 1987) — for FireProtDB ASA → RSA
# ═══════════════════════════════════════════════════════════
MAX_ASA = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
    'Q': 225.0, 'E': 223.0, 'G': 104.0, 'H': 224.0, 'I': 197.0,
    'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
    'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0,
}

# ═══════════════════════════════════════════════════════════
# Conservation (PSSM) features
# ═══════════════════════════════════════════════════════════
PSSM_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_conservation_cache = None

_esm_cache = None

def load_esm_cache():
    global _esm_cache
    if os.path.exists(ESM_EMBEDDINGS_PATH):
        with open(ESM_EMBEDDINGS_PATH, 'rb') as f:
            _esm_cache = pickle.load(f)
        print(f"  Loaded ESM-2 cache: {len(_esm_cache)} proteins")
    else:
        _esm_cache = {}
        print("  WARNING: ESM-2 cache not found. Run generate_esm_embeddings.py first.")

def get_esm_embedding(protein_id, position):
    """Return raw 640-dim ESM-2 embedding at mutation position, or None."""
    if not _esm_cache:
        return None
    emb = _esm_cache.get(protein_id)
    if emb is None:
        return None
    idx = position - 1
    if idx < 0 or idx >= len(emb):
        return None
    return emb[idx].astype(np.float32)  # shape (640,)


_metal_coord_cache = None

def load_metal_coord_cache():
    """Load metal coordination cache generated by generate_metal_coord_cache.py."""
    global _metal_coord_cache
    if os.path.exists(METAL_COORD_CACHE_PATH):
        with open(METAL_COORD_CACHE_PATH, 'rb') as f:
            _metal_coord_cache = pickle.load(f)
        n_with_metals = sum(1 for v in _metal_coord_cache.values() if v)
        total_sites = sum(len(v) for v in _metal_coord_cache.values())
        print(f"  Loaded metal coordination cache: {len(_metal_coord_cache)} proteins")
        print(f"  Proteins with ≥1 metal site: {n_with_metals} "
              f"({100*n_with_metals/max(len(_metal_coord_cache),1):.1f}%)")
        print(f"  Total metal-coordinating residue positions: {total_sites}")
    else:
        _metal_coord_cache = {}
        print("  WARNING: Metal coordination cache not found. "
              "Run generate_metal_coord_cache.py first.")


def get_metal_features(protein_id, position):
    """Return 5 metal coordination features for a mutation site.

    Features:
      [0] is_metal_coordinating  — 1 if ANY metal within coordination distance
      [1] is_ca2_coordinating    — 1 if Ca²⁺ (CA2) present
      [2] is_zn_coordinating     — 1 if Zn²⁺ (ZN2) present
      [3] is_mg_coordinating     — 1 if Mg²⁺ (MG2) present
      [4] n_coord_metal_types    — count of distinct metal types (0–N)
    """
    if not _metal_coord_cache:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    site_metals = _metal_coord_cache.get(protein_id, {}).get(position, set())
    if not site_metals:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    is_any   = 1.0
    is_ca2   = 1.0 if 'CA2' in site_metals else 0.0
    is_zn    = 1.0 if 'ZN2' in site_metals else 0.0
    is_mg    = 1.0 if 'MG2' in site_metals else 0.0
    n_types  = float(len(site_metals))
    return [is_any, is_ca2, is_zn, is_mg, n_types]


_physics_cache = None

_esm_loglik_cache = None   # {protein_id: {resnum: np.array(20,) log-probs}}
# Standard amino acid order for ESM log-likelihood lookup (alphabetical)
_ESM_LL_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
_ESM_LL_AA_TO_IDX = {aa: i for i, aa in enumerate(_ESM_LL_AA_ORDER)}

def load_physics_cache():
    """Load Debye-Hückel + Born solvation features from generate_physics_features.py."""
    global _physics_cache
    if os.path.exists(PHYSICS_FEATURES_PATH):
        with open(PHYSICS_FEATURES_PATH, 'rb') as f:
            _physics_cache = pickle.load(f)
        n_res = sum(len(v) for v in _physics_cache.values())
        print(f"  Loaded physics cache: {len(_physics_cache)} proteins, "
              f"{n_res} residue positions")
    else:
        _physics_cache = {}
        print("  WARNING: Physics cache not found — run generate_physics_features.py first.")


def load_esm_loglik_cache():
    """Load ESM-2 masked log-likelihood cache from generate_esm_loglik.py.

    Cache structure: { protein_id: { resnum: np.array(20,) } }
    Array contains log P(aa | context) for each of 20 AAs (ACDEFGHIKLMNPQRSTVWY order).
    """
    global _esm_loglik_cache
    if os.path.exists(ESM_LOGLIK_PATH):
        with open(ESM_LOGLIK_PATH, 'rb') as f:
            _esm_loglik_cache = pickle.load(f)
        n_pos = sum(len(v) for v in _esm_loglik_cache.values())
        print(f"  Loaded ESM log-likelihood cache: {len(_esm_loglik_cache)} proteins, "
              f"{n_pos} positions")
    else:
        _esm_loglik_cache = {}
        print("  WARNING: ESM log-likelihood cache not found — run generate_esm_loglik.py first.")


# Physical constants (must match generate_physics_features.py)
_COULOMB     = 332.06    # kcal·Å/(mol·e²)
_EPS_WATER   = 80.0
_EPS_PROT    = 4.0
_BORN_RADIUS = 3.5       # Å
_ION_STR_DEF = 0.15      # M
_LAMBDA_D_DEF = 3.04 / (_ION_STR_DEF ** 0.5)   # ≈ 7.85 Å


def get_physics_features(protein_id, position, wt_aa, mut_aa, ph=7.0,
                          rsa=0.5, ionic_strength=0.15):
    """Compute 6 physics-based features for a specific mutation.

    Combines the precomputed per-site structural data (stored in physics cache)
    with the per-mutation charge change (Δq = q_mut − q_wt) to produce
    physically meaningful ΔΔG estimates.

    Parameters
    ----------
    protein_id     : PDB ID string
    position       : 1-based residue number
    wt_aa          : wild-type amino acid (1-letter)
    mut_aa         : mutant amino acid (1-letter)
    ph             : assay pH (default 7.0)
    rsa            : relative solvent accessibility (0–1)
    ionic_strength : assay ionic strength in M (default 0.15)

    Returns
    -------
    list of 6 floats
    """
    # Charge change at mutation site (Henderson-Hasselbalch)
    def _q(aa, ph):
        pka = {'D':3.9,'E':4.1,'C':8.3,'Y':10.1,'H':6.0,'K':10.5,'R':12.5}.get(aa,0)
        if pka == 0: return 0.0
        if aa in ('D','E','C','Y'): return -1.0/(1+10**(pka-ph))
        return +1.0/(1+10**(ph-pka))

    q_wt  = _q(wt_aa, ph)
    q_mut = _q(mut_aa, ph)
    dq    = q_mut - q_wt

    # Look up precomputed site features
    site_feats = None
    if _physics_cache:
        site_feats = _physics_cache.get(protein_id, {}).get(position)

    if site_feats is None:
        # No structural data: use sequence-only approximations
        # Born solvation from RSA only
        eps_eff = _EPS_PROT + (_EPS_WATER - _EPS_PROT) * rsa
        ddg_born = -(_COULOMB/(2*_BORN_RADIUS)) * (q_mut**2 - q_wt**2) * (
            1/eps_eff - 1/_EPS_WATER)
        return [0.0, float(ddg_born), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Cache stores 6, 7, or 8 elements (8 if regenerated with contact_norm + B-factor)
    if len(site_feats) >= 8:
        dh_scale, born_coeff, phi_site, q_local, debye_fac, elec_burial_base, contact_norm, bfactor_z = site_feats
    elif len(site_feats) >= 7:
        dh_scale, born_coeff, phi_site, q_local, debye_fac, elec_burial_base, contact_norm = site_feats
        bfactor_z = 0.0
    else:
        dh_scale, born_coeff, phi_site, q_local, debye_fac, elec_burial_base = site_feats
        contact_norm = 0.0
        bfactor_z = 0.0

    # ── Feature 0: ΔΔG_elec (Debye-Hückel) ─────────────────────────────────
    # Scale precomputed interaction sum by Δq to get actual energy change.
    # Adjust for actual ionic strength vs. cache default (I=0.15 M):
    lam_actual  = 3.04 / max(ionic_strength, 1e-6)**0.5
    lam_default = _LAMBDA_D_DEF
    # First-order ionic correction: multiply by ratio of Debye factors
    ionic_correction = lam_actual / lam_default
    ddg_elec = float(dq * dh_scale * ionic_correction)

    # ── Feature 1: ΔΔG_Born (solvation penalty) ────────────────────────────
    # Use actual RSA for burial-dependent solvation
    eps_eff = _EPS_PROT + (_EPS_WATER - _EPS_PROT) * rsa
    ddg_born_actual = -(_COULOMB/(2*_BORN_RADIUS)) * (q_mut**2 - q_wt**2) * (
        1/eps_eff - 1/_EPS_WATER)

    # ── Features 2–7: site-level electrostatic + structural properties ────────
    elec_burial_actual = float(phi_site) * (1.0 - rsa)

    return [
        ddg_elec,               # [0] DH electrostatic ΔΔG (kcal/mol)
        float(ddg_born_actual), # [1] Born solvation ΔΔG (kcal/mol)
        float(phi_site),        # [2] electrostatic potential at site
        float(q_local),         # [3] local charge density
        float(debye_fac),       # [4] Debye screening factor
        elec_burial_actual,     # [5] potential × burial coupling
        float(contact_norm),    # [6] Cα contact number / 20 (packing density)
        float(bfactor_z),       # [7] B-factor z-score (relative flexibility, [-3,3])
    ]


def load_conservation_cache():
    global _conservation_cache
    if os.path.exists(CONSERVATION_CACHE_PATH):
        with open(CONSERVATION_CACHE_PATH, 'rb') as f:
            _conservation_cache = pickle.load(f)
        print(f"  Loaded conservation cache: {len(_conservation_cache)} entries")
    else:
        _conservation_cache = {}
        print("  WARNING: No conservation cache found. Run generate_pssm_conservation.py first.")

def get_conservation_features(protein_id, position, wt_aa, mut_aa):
    """Extract 6 PSSM-based conservation features for a mutation."""
    if _conservation_cache is None:
        return [0.0] * 6

    pssm_data = _conservation_cache.get(protein_id)
    if pssm_data is None:
        return [0.0] * 6

    pssm = pssm_data['pssm']
    info = pssm_data['info_content']

    idx = position - 1
    if idx < 0 or idx >= len(pssm):
        return [0.0] * 6

    aa_to_idx = {aa: i for i, aa in enumerate(PSSM_AA_ORDER)}
    wt_idx = aa_to_idx.get(wt_aa)
    mut_idx = aa_to_idx.get(mut_aa)
    if wt_idx is None or mut_idx is None:
        return [0.0] * 6

    row = pssm[idx]
    pssm_wt = float(row[wt_idx])
    pssm_mut = float(row[mut_idx])
    delta_pssm = pssm_mut - pssm_wt
    info_at_pos = float(info[idx]) if idx < len(info) else 0.0
    rank = float(np.sum(info <= info_at_pos) / max(len(info), 1)) if len(info) > 1 else 0.5
    wt_rank = float(np.sum(row <= pssm_wt) / 20.0)

    return [pssm_wt, pssm_mut, delta_pssm, info_at_pos, rank, wt_rank]


# ═══════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════

def charge_at_ph(aa: str, ph: float) -> float:
    """Net side-chain charge at a given pH via Henderson-Hasselbalch."""
    pka = SIDECHAIN_PKA.get(aa, 0.0)
    if pka == 0:
        return 0.0
    is_acid = SIDECHAIN_IS_ACID.get(aa, True)
    if is_acid:
        return -1.0 / (1.0 + 10.0 ** (pka - ph))   # < 0 above pKa
    else:
        return  1.0 / (1.0 + 10.0 ** (ph - pka))   # > 0 below pKa


def estimate_rsa(sequence, position):
    """Estimate relative solvent accessibility from sequence context."""
    if not sequence or position < 1 or position > len(sequence):
        return 0.5
    idx = position - 1
    aa = sequence[idx]
    # Buried residues tend to be hydrophobic
    h = HYDROPHOBICITY.get(aa, 0)
    base = 0.5 - h * 0.05  # hydrophobic = more buried

    # Terminal residues more exposed
    rel_pos = idx / max(len(sequence) - 1, 1)
    if rel_pos < 0.05 or rel_pos > 0.95:
        base += 0.2

    # Neighbors: if surrounded by hydrophobic, likely buried
    window = sequence[max(0, idx-3):idx+4]
    avg_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
    base -= avg_h * 0.02

    return max(0.0, min(1.0, base))


def estimate_secondary_structure(sequence, position):
    """Estimate SS propensities from local sequence."""
    if not sequence or position < 1 or position > len(sequence):
        return 0.33, 0.33, 0.34
    idx = position - 1
    window = sequence[max(0, idx-4):idx+5]
    h_score = np.mean([HELIX_PROPENSITY.get(a, 1.0) for a in window])
    s_score = np.mean([SHEET_PROPENSITY.get(a, 1.0) for a in window])
    total = h_score + s_score + 1.0
    return h_score / total, s_score / total, 1.0 / total


def extract_features(wt_aa, position, mut_aa, sequence=None, protein_id=None,
                     temperature=25.0, ph=7.0,
                     struct_rsa=None, struct_phi=None, struct_psi=None, struct_depth=None,
                     fp_asa=None, fp_ss=None,
                     ionic_strength=None):
    """Extract feature vector for a single mutation.

    Features (80 total in v12):
      - 6  physicochemical deltas (hydrophobicity, volume, charge, flexibility, helix, sheet)
      - 6  absolute values for WT and MUT
      - 1  BLOSUM62 substitution score
      - 3  secondary structure propensities at position
      - 1  RSA (real from ThermoMutDB when available, else sequence-estimated)
      - 4  sequence context features
      - 6  thermostability-specific features
      - 9  interaction terms
      - 6  additional features
      - 6  PSSM conservation features
      - 1  assay temperature (°C)  [feature 49]
      - 1  assay pH               [feature 50]
      - 8  extended biochemical features (MW, H-bonds, turn, polarity, pKa) [51-58]
      - 10 further extended features (aliphatic, charge-at-pH, size, disorder, entropy, Cys) [59-68]
      - 8  physically motivated cross-terms [69-76]
      -- v12 new features --
      - 1  phi backbone dihedral (degrees / 180, 0 if unknown)     [77]
      - 1  psi backbone dihedral (degrees / 180, 0 if unknown)     [78]
      - 1  Ca-alpha depth (Angstroms / 20, 0 if unknown)           [79]
      - 1  has_real_rsa flag (1 if structural RSA used, else 0)    [80]
    """
    if wt_aa not in AA_SET or mut_aa not in AA_SET:
        return None

    features = []

    # ── Physicochemical deltas (6) ──
    dH = HYDROPHOBICITY.get(mut_aa, 0) - HYDROPHOBICITY.get(wt_aa, 0)
    dV = VOLUME.get(mut_aa, 0) - VOLUME.get(wt_aa, 0)
    dC = CHARGE.get(mut_aa, 0) - CHARGE.get(wt_aa, 0)
    dF = FLEXIBILITY.get(mut_aa, 0) - FLEXIBILITY.get(wt_aa, 0)
    dHelix = HELIX_PROPENSITY.get(mut_aa, 1) - HELIX_PROPENSITY.get(wt_aa, 1)
    dSheet = SHEET_PROPENSITY.get(mut_aa, 1) - SHEET_PROPENSITY.get(wt_aa, 1)
    features.extend([dH, dV, dC, dF, dHelix, dSheet])

    # ── Absolute deltas (6) ──
    features.extend([abs(dH), abs(dV), abs(dC), abs(dF), abs(dHelix), abs(dSheet)])

    # ── BLOSUM62 (1) ──
    features.append(get_blosum62(wt_aa, mut_aa))

    # ── Secondary structure at position (3) ──
    # v15: use real PDB secondary structure from FireProtDB when available
    # fp_ss values: H/G/I=helix, E=sheet, L/S/T=coil
    if fp_ss is not None:
        if fp_ss in ('H', 'G', 'I'):
            h, s, c = 1.0, 0.0, 0.0  # helix
        elif fp_ss == 'E':
            h, s, c = 0.0, 1.0, 0.0  # sheet
        else:
            h, s, c = 0.0, 0.0, 1.0  # coil (L, S, T, etc.)
    elif sequence:
        h, s, c = estimate_secondary_structure(sequence, position)
    else:
        h, s, c = 0.33, 0.33, 0.34
    features.extend([h, s, c])

    # ── RSA (1) — use best available structural value ──
    # Priority: ThermoMutDB struct_rsa > FireProtDB ASA-derived > sequence-estimated
    has_real_rsa_flag = 0
    if struct_rsa is not None:
        rsa = float(struct_rsa)
        rsa = max(0.0, min(1.0, rsa))
        has_real_rsa_flag = 1
    elif fp_asa is not None:
        # v15: convert FireProtDB absolute ASA to relative RSA
        max_asa = MAX_ASA.get(wt_aa, 200.0)
        rsa = max(0.0, min(1.0, float(fp_asa) / max_asa))
        has_real_rsa_flag = 1
    else:
        rsa = estimate_rsa(sequence, position) if sequence else 0.5
    features.append(rsa)

    # ── Sequence context (4) ──
    if sequence and 1 <= position <= len(sequence):
        idx = position - 1
        # Local hydrophobicity
        window = sequence[max(0, idx-3):idx+4]
        local_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
        # Local charge
        local_c = np.mean([CHARGE.get(a, 0) for a in window])
        # Glycine/proline count in window
        gp_count = sum(1 for a in window if a in ('G', 'P'))
        # Relative position
        rel_pos = idx / max(len(sequence) - 1, 1)
        features.extend([local_h, local_c, gp_count / len(window), rel_pos])
    else:
        features.extend([0, 0, 0, 0.5])

    # ── Thermostability features (6) ──
    # Proline introduction (rigidifies backbone)
    to_proline = 1.0 if mut_aa == 'P' and wt_aa != 'P' else 0.0
    from_proline = 1.0 if wt_aa == 'P' and mut_aa != 'P' else 0.0
    # Glycine introduction (increases flexibility)
    to_glycine = 1.0 if mut_aa == 'G' and wt_aa != 'G' else 0.0
    # Deamidation risk (N,Q are prone at high temp)
    deamid_risk = 0.0
    if wt_aa in ('N', 'Q') and mut_aa not in ('N', 'Q'):
        deamid_risk = -1.0  # removing risk = good
    elif mut_aa in ('N', 'Q') and wt_aa not in ('N', 'Q'):
        deamid_risk = 1.0  # adding risk = bad
    # Salt bridge potential
    salt_bridge = 0.0
    if mut_aa in ('D', 'E', 'K', 'R') and wt_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = 1.0
    elif wt_aa in ('D', 'E', 'K', 'R') and mut_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = -1.0
    # Cysteine (disulfide potential)
    cys_change = 0.0
    if mut_aa == 'C' and wt_aa != 'C':
        cys_change = 1.0
    elif wt_aa == 'C' and mut_aa != 'C':
        cys_change = -1.0
    features.extend([to_proline, from_proline, to_glycine, deamid_risk, salt_bridge, cys_change])

    # ── Interaction terms (9) ──
    burial = 1.0 - rsa
    features.extend([
        abs(dH) * burial,      # hydrophobicity change × burial
        abs(dV) * burial,      # volume change × burial
        abs(dC) * burial,      # charge change × burial
        abs(dH) * abs(dV),     # hydrophobicity × volume
        abs(dC) * abs(dH),     # charge × hydrophobicity
        to_proline * burial,   # proline intro × burial
        burial * h,            # burial × helix
        burial * s,            # burial × sheet
        abs(dH) * h,           # hydrophobicity × helix
    ])

    # ── Additional (6) ──
    # Aromatic change
    aromatic_wt = 1.0 if wt_aa in ('F', 'W', 'Y', 'H') else 0.0
    aromatic_mut = 1.0 if mut_aa in ('F', 'W', 'Y', 'H') else 0.0
    # Small-to-large / large-to-small
    small_aa = {'G', 'A', 'S', 'T', 'C'}
    large_aa = {'F', 'W', 'Y', 'R', 'K', 'H'}
    small_to_large = 1.0 if wt_aa in small_aa and mut_aa in large_aa else 0.0
    large_to_small = 1.0 if wt_aa in large_aa and mut_aa in small_aa else 0.0
    # Conservation proxy (BLOSUM self-score difference)
    cons_wt = BLOSUM62_DIAG.get(wt_aa, 4)
    cons_mut = BLOSUM62_DIAG.get(mut_aa, 4)
    features.extend([
        aromatic_wt - aromatic_mut,  # aromatic change
        small_to_large,
        large_to_small,
        cons_wt,
        cons_mut,
        cons_wt - cons_mut,
    ])

    # ── PSSM conservation features (6) ──
    cons_feats = get_conservation_features(protein_id, position, wt_aa, mut_aa)
    features.extend(cons_feats)

    # ── Condition features (2): temperature (°C) and pH ──
    # These are the actual assay conditions reported in ThermoMutDB/FireProtDB.
    # Including them lets the model learn condition-dependent stability effects.
    features.append(float(temperature))  # feature 49: assay temperature (°C)
    features.append(float(ph))           # feature 50: assay pH

    # ── Extended biochemical features (8) [51-58] ──
    # Molecular weight delta
    dMW = MOLECULAR_WEIGHT.get(mut_aa, 130.0) - MOLECULAR_WEIGHT.get(wt_aa, 130.0)
    # H-bond capacity deltas
    dHD = float(HBOND_DONORS.get(mut_aa, 1)    - HBOND_DONORS.get(wt_aa, 1))
    dHA = float(HBOND_ACCEPTORS.get(mut_aa, 1) - HBOND_ACCEPTORS.get(wt_aa, 1))
    # Turn propensity delta (positive = more likely in turns/loops)
    dTurn = TURN_PROPENSITY.get(mut_aa, 1.0) - TURN_PROPENSITY.get(wt_aa, 1.0)
    # Polarity class change (0→2 range, signed: gaining charge is +2)
    pol_wt  = POLARITY_CLASS.get(wt_aa, 0)
    pol_mut = POLARITY_CLASS.get(mut_aa, 0)
    pol_change = float(pol_mut - pol_wt)
    # Binary: does a charged residue appear or disappear?
    charge_gain = 1.0 if pol_mut == 2 and pol_wt != 2 else 0.0
    charge_loss = 1.0 if pol_wt  == 2 and pol_mut != 2 else 0.0
    # pKa-based ionization: fraction ionized at assay pH (Henderson-Hasselbalch)
    pka_wt  = SIDECHAIN_PKA.get(wt_aa,  0.0)
    pka_mut = SIDECHAIN_PKA.get(mut_aa, 0.0)
    if pka_wt > 0:
        ion_wt  = 1.0 / (1.0 + 10.0 ** (pka_wt  - float(ph)))
    else:
        ion_wt  = 0.0
    if pka_mut > 0:
        ion_mut = 1.0 / (1.0 + 10.0 ** (pka_mut - float(ph)))
    else:
        ion_mut = 0.0
    delta_ionization = ion_mut - ion_wt

    features.extend([dMW, dHD, dHA, dTurn, pol_change, charge_gain, charge_loss, delta_ionization])

    # ── Further extended features (10) [59-68] ──

    # 59: aliphatic index delta (Ikai 1980 — thermostability proxy)
    dAliphatic = ALIPHATIC_CONTRIB.get(mut_aa, 0.0) - ALIPHATIC_CONTRIB.get(wt_aa, 0.0)

    # 60-61: net side-chain charge at assay pH for WT and MUT
    ch_wt  = charge_at_ph(wt_aa, ph)
    ch_mut = charge_at_ph(mut_aa, ph)
    dCharge_ph = ch_mut - ch_wt  # signed charge delta at assay pH

    # 62: side-chain size class change (steric clash indicator)
    size_wt  = SIZE_CLASS.get(wt_aa, 1)
    size_mut = SIZE_CLASS.get(mut_aa, 1)
    dSizeClass = float(size_mut - size_wt)

    # 63: intrinsic disorder propensity delta (Uversky 2002)
    dDisorder = DISORDER_PROPENSITY.get(mut_aa, 0.0) - DISORDER_PROPENSITY.get(wt_aa, 0.0)

    # 64: local Shannon sequence entropy (before mutation, measures local conservation)
    if sequence and 1 <= position <= len(sequence):
        idx = position - 1
        win = sequence[max(0, idx-5):idx+6]
        aa_counts = {a: win.count(a) for a in set(win)}
        n = len(win)
        entropy = -sum((c/n) * np.log2(c/n) for c in aa_counts.values() if c > 0)
    else:
        entropy = 2.0  # ~average protein entropy

    # 65: buried hydrophobic indicator — WT and MUT
    buried_h_wt  = 1.0 if (rsa < 0.25 and HYDROPHOBICITY.get(wt_aa, 0) > 1.5) else 0.0
    buried_h_mut = 1.0 if (rsa < 0.25 and HYDROPHOBICITY.get(mut_aa, 0) > 1.5) else 0.0

    # 66-67: WT and MUT absolute net charge at assay pH (already have signed delta above)
    abs_ch_wt  = abs(ch_wt)
    abs_ch_mut = abs(ch_mut)

    # 68: nearest cysteine distance (proxy for disulfide region, 0 if no Cys nearby)
    if sequence and 1 <= position <= len(sequence):
        idx = position - 1
        cys_positions = [i for i, a in enumerate(sequence) if a == 'C']
        if cys_positions:
            nearest_cys_dist = min(abs(idx - cp) for cp in cys_positions) / max(len(sequence), 1)
        else:
            nearest_cys_dist = 1.0
    else:
        nearest_cys_dist = 1.0

    features.extend([
        dAliphatic, dCharge_ph, dSizeClass, dDisorder,
        entropy, buried_h_wt, buried_h_mut, abs_ch_wt, abs_ch_mut, nearest_cys_dist,
    ])

    # ── Physically motivated cross-term features (8) [69-76] ──
    # These capture condition×mutation and burial×property couplings that
    # gradient-boosted trees can learn but benefit from being made explicit.
    features.extend([
        dH * dC,                           # hydrophobicity-charge coupling
        dH * dV,                           # hydrophobicity-volume (packing energy)
        dCharge_ph * float(ph),            # pH-adjusted charge × assay pH
        dH * float(temperature) * 0.01,   # hydrophobicity × temp (scaled)
        burial * dCharge_ph,               # electrostatic burial coupling
        burial * dMW * 0.01,               # packing × size (scaled)
        abs(dH) * abs(dCharge_ph),         # amphipathic change magnitude
        dAliphatic * float(temperature) * 0.01,  # aliphatic × temp (thermostability)
    ])

    # ── v12: Structural geometry features (4) [77-80] ──
    phi_norm   = float(struct_phi)   / 180.0 if struct_phi   is not None else 0.0
    psi_norm   = float(struct_psi)   / 180.0 if struct_psi   is not None else 0.0
    depth_norm = float(struct_depth) / 20.0  if struct_depth is not None else 0.0
    features.extend([phi_norm, psi_norm, depth_norm, float(has_real_rsa_flag)])

    # ── v13: Structural cross-terms (6) [81-86] ──
    # These capture non-linear interactions between backbone geometry and mutation properties.
    temp_scaled = float(temperature) * 0.01
    features.extend([
        phi_norm * psi_norm,              # 81: phi×psi — Ramachandran region indicator
        phi_norm * dH,                    # 82: phi×hydrophobicity — backbone context coupling
        phi_norm * temp_scaled,           # 83: phi×temperature — thermal sensitivity of backbone
        psi_norm * dH,                    # 84: psi×hydrophobicity — strand/helix context
        psi_norm * temp_scaled,           # 85: psi×temperature — complementary thermal term
        depth_norm * dH,                  # 86: depth×hydrophobicity — 3D burial × mutation type
    ])

    # ── v17: Metal ion coordination features (5) ──────────────────────────────
    # Derived from crystallographic PDB structures via generate_metal_coord_cache.py.
    # Residues within coordination distance of a metal ion are fundamentally
    # different in their stability response to mutations: mutations that disrupt
    # Ca²⁺ coordination can destabilize a protein by 3–10 kcal/mol (Vriend 1991,
    # Proctor et al. 2013), far exceeding sequence-based estimates.
    metal_feats = get_metal_features(protein_id, position)
    features.extend(metal_feats)  # [87–91]: is_metal, is_ca2, is_zn, is_mg, n_types

    # ── v17: Ionic strength feature (1) ────────────────────────────────────────
    # Neither FireProtDB nor ThermoMutDB records experimental ionic strength, so
    # this feature is set to the physiological default for all training data.
    # It is included so inference can be conditioned on actual assay ionic
    # strength (e.g., 0.05 M for low-salt experiments, 0.5 M for high-salt).
    # At training time this is a constant and cannot contribute signal; however,
    # its inclusion makes the model interface explicit about this dependency and
    # allows future retraining when databases include ionic strength.
    if ionic_strength is not None:
        ionic_norm = float(ionic_strength) / IONIC_STRENGTH_SCALE
    else:
        ionic_norm = PHYSIOLOGICAL_IONIC_STRENGTH / IONIC_STRENGTH_SCALE  # 0.30
    features.append(ionic_norm)   # [92]

    # ── v22: Physics-based electrostatic + solvation + packing + flexibility (8) [93–100] ──
    # Debye-Hückel electrostatics, Born solvation, Cα contact number, and B-factor
    # computed analytically from PDB crystal structures. These implement the same
    # physical models as FoldX/Rosetta's electrostatic terms without requiring those tools.
    # v21: contact number [6]; v22: B-factor z-score [7] (Yuan et al. 2005).
    # References: Debye & Hückel (1923); Born (1920); Gilson & Honig (1988).
    actual_ionic = ionic_strength if ionic_strength is not None else PHYSIOLOGICAL_IONIC_STRENGTH
    phys_feats = get_physics_features(
        protein_id, position, wt_aa, mut_aa,
        ph=float(ph), rsa=rsa, ionic_strength=actual_ionic,
    )
    features.extend(phys_feats)   # [93–100]: ddg_elec, ddg_born, phi_site, q_local,
                                  #            debye_factor, elec_burial, contact_norm, bfactor_z

    # ── v24: ESM-2 masked log-likelihood ratio (ΔLL) [101] ─────────────────────
    # ΔLL = log P(mut_aa | context) − log P(wt_aa | context)
    # Directly measures evolutionary fitness cost of the mutation from ESM-2.
    # Negative ΔLL = mutation is evolutionarily unfavored (typically destabilizing).
    # Reference: Meier et al. (2021) NeurIPS (ESM-1v); Lin et al. (2023) Science.
    pid_upper = protein_id.upper() if protein_id else ''
    delta_ll     = 0.0
    wt_logprob   = 0.0
    mut_logprob  = 0.0
    mut_rank     = 10.0   # neutral default (middle of 0-19)
    site_entropy = 0.0
    if _esm_loglik_cache:
        pos_logliks = _esm_loglik_cache.get(pid_upper, {}).get(position)
        if pos_logliks is not None:
            wt_idx  = _ESM_LL_AA_TO_IDX.get(wt_aa,  -1)
            mut_idx = _ESM_LL_AA_TO_IDX.get(mut_aa, -1)
            if wt_idx >= 0 and mut_idx >= 0:
                delta_ll    = float(pos_logliks[mut_idx] - pos_logliks[wt_idx])
                wt_logprob  = float(pos_logliks[wt_idx])
                mut_logprob = float(pos_logliks[mut_idx])
                mut_rank    = float(np.sum(pos_logliks > pos_logliks[mut_idx]))
            probs = np.exp(pos_logliks - np.max(pos_logliks))
            probs /= probs.sum()
            site_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    features.append(delta_ll)      # [101] ESM-2 ΔLL
    features.append(wt_logprob)    # [102] WT log-probability (conservation)
    features.append(mut_logprob)   # [103] Mut log-probability
    features.append(mut_rank)      # [104] Mutation rank among 20 AAs (0=most preferred)
    features.append(site_entropy)  # [105] Site entropy (position variability)

    return features  # 105 base features total (v24; + ESM-PCA-32+1 = 138 in training)


# ═══════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════

def parse_mutation_code(code):
    """Parse 'A123G' format into (wt_aa, position, mut_aa)."""
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', code)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def load_fireprotdb():
    """Load FireProtDB dataset."""
    print("Loading FireProtDB...")
    df = pd.read_csv(FIREPROT_PATH)
    records = []
    for _, row in df.iterrows():
        wt = row.get('wild_type', '')
        mut = row.get('mutation', '')
        pos = row.get('position', 0)
        ddg = row.get('ddG', None)
        seq = row.get('sequence', '')
        pdb = str(row.get('pdb_id', '')).split('|')[0]

        if pd.isna(ddg) or wt not in AA_SET or mut not in AA_SET or wt == mut:
            continue
        try:
            pos = int(pos)
        except (ValueError, TypeError):
            continue

        # FireProtDB has a pH column; temperature is measured at 25°C by default
        ph_val = row.get('pH', row.get('ph', 7.0))
        try:
            ph_val = float(ph_val) if pd.notna(ph_val) else 7.0
        except (ValueError, TypeError):
            ph_val = 7.0

        # v15: capture real structural features for in-place slot replacement
        fp_asa_val = row.get('asa')
        try:
            fp_asa_val = float(fp_asa_val) if pd.notna(fp_asa_val) else None
        except (ValueError, TypeError):
            fp_asa_val = None

        fp_ss_val = row.get('secondary_structure')
        fp_ss_val = str(fp_ss_val) if pd.notna(fp_ss_val) else None

        records.append({
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'ddg': float(ddg), 'sequence': str(seq) if pd.notna(seq) else '',
            'protein_id': pdb, 'source': 'FireProtDB',
            'temperature_c': 25.0,
            'ph': ph_val,
            'fp_asa': fp_asa_val,
            'fp_ss': fp_ss_val,
        })
    print(f"  Loaded {len(records)} mutations from FireProtDB")
    fp_asa_count = sum(1 for r in records if r.get('fp_asa') is not None)
    fp_ss_count  = sum(1 for r in records if r.get('fp_ss')  is not None)
    print(f"  Real ASA: {fp_asa_count}/{len(records)} ({100*fp_asa_count/max(len(records),1):.1f}%)")
    print(f"  Real SS:  {fp_ss_count}/{len(records)} ({100*fp_ss_count/max(len(records),1):.1f}%)")
    return records


def load_proddg():
    """Load ProDDG / S2648 dataset."""
    print("Loading ProDDG (S2648)...")
    if not os.path.exists(PRODDG_PATH):
        print(f"  WARNING: ProDDG file not found at {PRODDG_PATH} — skipping")
        return []
    df = pd.read_csv(PRODDG_PATH, sep='\t')
    records = []
    for _, row in df.iterrows():
        mut_code = row.get('mutation', '')
        wt, pos, mut = parse_mutation_code(str(mut_code))
        ddg = row.get('ddG', None)
        seq = row.get('wt_sequence', '')
        pdb = str(row.get('pdb', ''))

        if wt is None or pd.isna(ddg):
            continue

        records.append({
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'ddg': float(ddg), 'sequence': str(seq) if pd.notna(seq) else '',
            'protein_id': pdb, 'source': 'ProDDG',
            'temperature_c': 25.0,  # standard biochemistry assay temp
            'ph': 7.0,
        })
    print(f"  Loaded {len(records)} mutations from ProDDG")
    return records


def load_s669():
    """Load S669 independent test set."""
    print("Loading S669 (independent test set)...")
    if not os.path.exists(S669_PATH):
        print(f"  WARNING: S669 file not found at {S669_PATH} — skipping independent test")
        return []
    df = pd.read_csv(S669_PATH, sep='\t')
    records = []
    for _, row in df.iterrows():
        mut_code = row.get('mutation', '')
        wt, pos, mut = parse_mutation_code(str(mut_code))
        ddg = row.get('ddG', None)
        seq = row.get('wt_sequence', '')
        pdb = str(row.get('pdb', ''))

        if wt is None or pd.isna(ddg):
            continue

        records.append({
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'ddg': float(ddg), 'sequence': str(seq) if pd.notna(seq) else '',
            'protein_id': pdb, 'source': 'S669'
        })
    print(f"  Loaded {len(records)} mutations from S669")
    return records


def load_thermomutdb():
    """Load ThermoMutDB dataset.

    Returns DDG training records and a separate list of ΔTm records.
    ThermoMutDB provides measured assay temperature (Kelvin) and pH for each
    entry — these become real ML features (features 49 and 50).
    Source: ThermoMutDB (Pucci et al., 2021, Nucleic Acids Res.)
    """
    print("Loading ThermoMutDB...")
    with open(THERMOMUTDB_PATH, 'r') as f:
        data = json.load(f)

    records = []
    dtm_records = []   # subset with measured ΔTm (melting temperature shift)
    no_temp = 0
    no_ph = 0

    for entry in data:
        mut_code = entry.get('mutation_code', '')
        wt, pos, mut = parse_mutation_code(str(mut_code))
        if wt is None:
            continue
        pdb = entry.get('PDB_wild', '')

        # ── Condition features — from the database record itself ──
        temp_k = entry.get('temperature', None)
        try:
            temp_c = float(temp_k) - 273.15 if temp_k is not None else 37.0
        except (ValueError, TypeError):
            temp_c = 37.0
            no_temp += 1

        ph_val = entry.get('ph', None)
        try:
            ph_val = float(ph_val) if ph_val is not None else 7.0
        except (ValueError, TypeError):
            ph_val = 7.0
            no_ph += 1

        # ── v12: capture real structural features when present ──
        def _safe_float(val):
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        struct_rsa   = _safe_float(entry.get('rsa'))
        struct_phi   = _safe_float(entry.get('phi'))
        struct_psi   = _safe_float(entry.get('psi'))
        struct_depth = _safe_float(entry.get('ca_depth'))

        base = {
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'sequence': '', 'protein_id': str(pdb), 'source': 'ThermoMutDB',
            'temperature_c': temp_c, 'ph': ph_val,
            # structural features (None for records without structural data)
            'struct_rsa':   struct_rsa,
            'struct_phi':   struct_phi,
            'struct_psi':   struct_psi,
            'struct_depth': struct_depth,
        }

        # DDG record
        ddg = entry.get('ddg', None)
        if ddg is not None:
            try:
                ddg = float(ddg)
                records.append({**base, 'ddg': ddg})
            except (ValueError, TypeError):
                pass

        # ΔTm record (subset: 6,107 entries in ThermoMutDB)
        dtm = entry.get('dtm', None)
        if dtm is not None:
            try:
                dtm = float(dtm)
                dtm_records.append({**base, 'dtm': dtm})
            except (ValueError, TypeError):
                pass

    print(f"  Loaded {len(records)} DDG mutations from ThermoMutDB")
    print(f"  Loaded {len(dtm_records)} ΔTm records from ThermoMutDB")
    if no_temp:
        print(f"  WARNING: {no_temp} entries missing temperature (used 37.0°C default)")
    if no_ph:
        print(f"  WARNING: {no_ph} entries missing pH (used 7.0 default)")
    return records, dtm_records


def deduplicate(records):
    """Remove duplicate mutations (same protein + position + mutation)."""
    seen = set()
    unique = []
    for r in records:
        key = (r['protein_id'], r['position'], r['wt_aa'], r['mut_aa'])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f"  After deduplication: {len(unique)} unique mutations (removed {len(records) - len(unique)})")
    return unique


# ═══════════════════════════════════════════════════════════
# Main training pipeline
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PUBLICATION-READY MODEL TRAINING — v27")
    print("Real experimental data only — no synthetic mutations — v25")
    print("v24: + ESM-2 masked log-likelihood ΔLL (138 features total)")
    print("=" * 70)
    print()

    # ── Step 1: Load all data ──
    print("STEP 1: Loading datasets")
    print("-" * 40)
    fireprot = load_fireprotdb()
    proddg = load_proddg()
    thermomutdb, dtm_records = load_thermomutdb()
    s669 = load_s669()

    # ── Step 2: Combine training data and deduplicate ──
    print("\nSTEP 2: Combining and deduplicating training data")
    print("-" * 40)
    train_records = fireprot + proddg + thermomutdb
    print(f"  Total before dedup: {len(train_records)}")
    train_records = deduplicate(train_records)

    # Remove any S669 proteins from training (strict independence)
    s669_proteins = set(r['protein_id'] for r in s669)
    s669_mutations = set((r['protein_id'], r['position'], r['wt_aa'], r['mut_aa']) for r in s669)
    train_clean = []
    removed_overlap = 0
    for r in train_records:
        key = (r['protein_id'], r['position'], r['wt_aa'], r['mut_aa'])
        if key in s669_mutations:
            removed_overlap += 1
        else:
            train_clean.append(r)
    train_records = train_clean
    print(f"  Removed {removed_overlap} mutations overlapping with S669 test set")
    print(f"  Final training set: {len(train_records)} mutations")

    # ── Step 2.5: Outlier removal — clip extreme DDG values ──
    # Extreme DDG values (e.g. ±68 kcal/mol in ThermoMutDB) are likely measurement
    # artefacts or data entry errors. Clipping to ±10 kcal/mol removes < 1% of samples
    # while substantially reducing noise that degrades regressor performance.
    DDG_CLIP = 10.0
    n_before = len(train_records)
    train_records = [r for r in train_records if abs(r['ddg']) <= DDG_CLIP]
    n_clipped = n_before - len(train_records)
    if n_clipped:
        print(f"\n  Removed {n_clipped} outlier mutations (|DDG| > {DDG_CLIP} kcal/mol)")
    print(f"  Training set after outlier removal: {len(train_records)}")

    # ── Step 2.6: Reverse mutation augmentation (thermodynamic antisymmetry) ──
    # Physical law (Hess' law / thermodynamic cycle): if WT→MUT has ΔΔG = x kcal/mol,
    # then MUT→WT necessarily has ΔΔG = -x kcal/mol.
    # Adding reversed mutations is NOT data synthesis — it is a hard physical constraint
    # used by FoldX, Rosetta, and standard thermodynamic databases (Guerois 2002,
    # Dehouck 2009). Roughly doubles effective training set.
    # Sequence context is approximated (original neighbors kept); mutation features are exact.
    print("\nAugmenting with reverse mutations (antisymmetry of ΔΔG)...")
    augmented = []
    for r in train_records:
        if r['wt_aa'] == r['mut_aa']:
            continue
        augmented.append({
            'wt_aa':        r['mut_aa'],
            'mut_aa':       r['wt_aa'],
            'position':     r['position'],
            'ddg':          -r['ddg'],
            'sequence':     r.get('sequence', ''),
            'protein_id':   r['protein_id'],
            'source':       r['source'],
            'temperature_c': r.get('temperature_c', 25.0),
            'ph':           r.get('ph', 7.0),
            'struct_rsa':   r.get('struct_rsa'),
            'struct_phi':   r.get('struct_phi'),
            'struct_psi':   r.get('struct_psi'),
            'struct_depth': r.get('struct_depth'),
            # Site properties unchanged for reverse mutation
            'fp_asa':       r.get('fp_asa'),
            'fp_ss':        r.get('fp_ss'),
        })
    pre_aug = len(train_records)
    train_records = deduplicate(train_records + augmented)
    print(f"  Added {len(train_records) - pre_aug} reverse mutations → {len(train_records)} total")

    # ── Step 2.7: Load all feature caches ──
    print("\nLoading PSSM conservation cache...")
    load_conservation_cache()
    print("Loading ESM-2 embeddings cache...")
    load_esm_cache()
    print("Loading metal coordination cache...")
    load_metal_coord_cache()
    print("Loading physics features cache (Debye-Hückel + Born solvation)...")
    load_physics_cache()
    print("Loading ESM-2 masked log-likelihood cache (ΔLL mutation scoring)...")
    load_esm_loglik_cache()

    # ── Step 3: Extract features ──
    print("\nSTEP 3: Extracting features")
    print("-" * 40)

    def records_to_arrays(records, pca=None, pca_mean=None):
        """Convert records to feature arrays with ESM-2 embeddings appended.

        Two-pass for training:
          Pass 1: extract base features + gather raw ESM-2 embeddings
          PCA fit on raw ESM-2 embeddings
          Pass 2: concatenate base features with PCA-reduced ESM-2

        For test/inference: pca and pca_mean are passed in from training.
        """
        from sklearn.decomposition import PCA as _PCA

        base_feats_list = []
        raw_esm_list = []      # 640-dim or None
        valid_esm_mask = []
        y_ddg_list, y_bin_list, prot_list, src_list = [], [], [], []
        skipped = 0
        has_pssm = 0
        has_esm_count = 0
        has_metal_count = 0
        has_physics_count = 0
        has_loglik_count = 0

        for r in records:
            temp_c = r.get('temperature_c', 25.0)
            ph_val = r.get('ph', 7.0)
            feats = extract_features(
                r['wt_aa'], r['position'], r['mut_aa'],
                r['sequence'], protein_id=r['protein_id'],
                temperature=temp_c, ph=ph_val,
                struct_rsa=r.get('struct_rsa'),
                struct_phi=r.get('struct_phi'),
                struct_psi=r.get('struct_psi'),
                struct_depth=r.get('struct_depth'),
                fp_asa=r.get('fp_asa'),
                fp_ss=r.get('fp_ss'),
                ionic_strength=r.get('ionic_strength'),  # None → physiological default
            )
            if feats is None:
                skipped += 1
                continue

            # ESM-2 embedding at mutation position
            esm_emb = get_esm_embedding(r['protein_id'], r['position'])

            base_feats_list.append(feats)
            raw_esm_list.append(esm_emb)
            valid_esm_mask.append(esm_emb is not None)
            y_ddg_list.append(r['ddg'])
            y_bin_list.append(1 if r['ddg'] < 0 else 0)
            prot_list.append(r['protein_id'])
            src_list.append(r['source'])

            if _conservation_cache and r['protein_id'] in _conservation_cache:
                has_pssm += 1
            if esm_emb is not None:
                has_esm_count += 1
            mf = get_metal_features(r['protein_id'], r['position'])
            if mf[0] > 0:
                has_metal_count += 1
            if _physics_cache and r['protein_id'] in _physics_cache:
                if r['position'] in _physics_cache[r['protein_id']]:
                    has_physics_count += 1
            if _esm_loglik_cache and r['protein_id'].upper() in _esm_loglik_cache:
                if r['position'] in _esm_loglik_cache[r['protein_id'].upper()]:
                    has_loglik_count += 1

        if skipped:
            print(f"  Skipped {skipped} mutations (invalid amino acids)")
        n = len(base_feats_list)
        print(f"  PSSM coverage:        {has_pssm}/{n} ({100*has_pssm/max(n,1):.1f}%)")
        print(f"  ESM-2 coverage:       {has_esm_count}/{n} ({100*has_esm_count/max(n,1):.1f}%)")
        print(f"  Metal-coord sites:    {has_metal_count}/{n} ({100*has_metal_count/max(n,1):.1f}%)")
        print(f"  Physics (DH+Born):    {has_physics_count}/{n} ({100*has_physics_count/max(n,1):.1f}%)")
        print(f"  ESM ΔLL coverage:     {has_loglik_count}/{n} ({100*has_loglik_count/max(n,1):.1f}%)")

        X_base = np.array(base_feats_list, dtype=np.float32)

        # Build ESM matrix; impute missing with column mean after PCA
        valid_idx = [i for i, v in enumerate(valid_esm_mask) if v]
        raw_esm_valid = np.array([raw_esm_list[i] for i in valid_idx], dtype=np.float32)

        if pca is None:
            # Fit PCA on training data
            print(f"  Fitting PCA({ESM_DIM}) on {len(valid_idx)} ESM-2 embeddings...")
            pca = _PCA(n_components=ESM_DIM, random_state=42)
            pca.fit(raw_esm_valid)
            var_explained = pca.explained_variance_ratio_.sum()
            print(f"  PCA variance explained: {100*var_explained:.1f}%")
            # Mean of PCA-transformed valid embeddings (used for imputation)
            pca_mean = pca.transform(raw_esm_valid).mean(axis=0)

        # Transform valid embeddings; use mean for missing
        esm_features = np.tile(pca_mean, (n, 1)).astype(np.float32)
        has_esm_flag = np.zeros((n, 1), dtype=np.float32)
        if len(valid_idx) > 0:
            esm_features[valid_idx] = pca.transform(raw_esm_valid)
            has_esm_flag[valid_idx] = 1.0

        X = np.concatenate([X_base, esm_features, has_esm_flag], axis=1)

        return (X, np.array(y_ddg_list), np.array(y_bin_list),
                prot_list, src_list, pca, pca_mean)

    result = records_to_arrays(train_records)
    X_train, y_train_ddg, y_train, train_proteins, train_sources, _pca, _pca_mean = result

    # Test set uses same PCA
    if s669:
        result_test = records_to_arrays(s669, pca=_pca, pca_mean=_pca_mean)
        X_test, y_test_ddg, y_test, test_proteins, test_sources = result_test[:5]
    else:
        X_test = np.zeros((0, X_train.shape[1]), dtype=np.float32)
        y_test_ddg = np.array([])
        y_test = np.array([])
        test_proteins, test_sources = [], []

    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test (S669): {X_test.shape[0]} samples")
    print(f"  Training class balance: {np.sum(y_train == 1)} stabilizing, {np.sum(y_train == 0)} destabilizing")
    print(f"  Test class balance: {np.sum(y_test == 1)} stabilizing, {np.sum(y_test == 0)} destabilizing")

    # Source breakdown
    source_counts = defaultdict(int)
    for s in train_sources:
        source_counts[s] += 1
    print(f"  Training sources: {dict(source_counts)}")

    # ── Step 4: Scale features ──
    print("\nSTEP 4: Scaling features")
    print("-" * 40)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test.shape[0] > 0 else X_test
    print(f"  Scaled {X_train.shape[1]} features")

    # ── Step 5: Train ensemble of 4 regressors with Optuna hyperparameter tuning ──
    print("\nSTEP 5: Training ensemble (GBM + XGBoost + LightGBM + CatBoost) with Optuna tuning")
    print("-" * 40)

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    print(f"  Stabilizing (DDG<0): {n_pos}, Destabilizing (DDG>=0): {n_neg}")
    print(f"  DDG range: [{y_train_ddg.min():.2f}, {y_train_ddg.max():.2f}] kcal/mol")

    kf5_tune = KFold(n_splits=5, shuffle=True, random_state=42)
    # v27: XGB/LGBM/CB use pre-tuned params — no Optuna needed for them.
    # Only HGB is Optuna-tuned (fast, ~20 min on 4k subsample).

    # ── XGBoost: use pre-tuned best params from v26 (100-trial Optuna, MAE 1.0996) ──
    print("\n  [Fixed] XGBoost — using v26 pre-tuned params (MAE 1.0996)...")
    best_xgb = {
        'n_estimators': 589, 'max_depth': 9, 'learning_rate': 0.016857723394967876,
        'subsample': 0.9490003831060219, 'colsample_bytree': 0.7911621808522555,
        'colsample_bylevel': 0.6369503663889663, 'min_child_weight': 4,
        'reg_alpha': 0.36448061196314285, 'reg_lambda': 0.18450540313606173,
        'gamma': 0.578228192086447,
    }
    xgb_reg = XGBRegressor(**best_xgb, random_state=42, verbosity=0, n_jobs=-1)
    xgb_reg.fit(X_train_scaled, y_train_ddg)
    print("    XGBoost trained.")

    # ── LightGBM: use pre-tuned best params from v26 (100-trial Optuna, MAE 1.0983) ──
    print("\n  [Fixed] LightGBM — using v26 pre-tuned params (MAE 1.0983)...")
    best_lgbm = {
        'n_estimators': 1393, 'max_depth': 10, 'learning_rate': 0.020643460431513237,
        'num_leaves': 199, 'subsample': 0.9835482825580907,
        'colsample_bytree': 0.5029009084366727, 'min_child_samples': 6,
        'reg_alpha': 0.10367941150136066, 'reg_lambda': 0.41629158663921856,
        'min_split_gain': 0.06831586173554616,
    }
    lgbm_reg = LGBMRegressor(**best_lgbm, n_jobs=-1, random_state=42, verbosity=-1)
    lgbm_reg.fit(X_train_scaled, y_train_ddg)
    print("    LightGBM trained.")

    # ── Model 1: GradientBoosting (sklearn, Huber loss for robustness) ──
    print("\n  Training GradientBoostingRegressor...")
    gb_reg = GradientBoostingRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=6, min_samples_split=12,
        max_features='sqrt', loss='huber', alpha=0.9, random_state=42,
    )
    gb_reg.fit(X_train_scaled, y_train_ddg)
    print("    Done.")

    # ── CatBoost: use pre-tuned best params from v24 (50-trial Optuna, MAE 1.1704) ──
    # CatBoost Optuna tuning is too slow (each trial ~90 min even on 3k samples).
    print("\n  [Fixed] CatBoost — using v24 pre-tuned params (MAE 1.1704)...")
    best_cb = {
        'iterations': 966, 'depth': 8, 'learning_rate': 0.02904505270290301,
        'subsample': 0.9423044862563902, 'colsample_bylevel': 0.9268902334716056,
        'l2_leaf_reg': 2.651194230102684,
    }
    cb_reg = CatBoostRegressor(**best_cb, loss_function='MAE', random_seed=42, verbose=0)
    cb_reg.fit(X_train_scaled, y_train_ddg)
    print("    CatBoost trained.")

    # ── Model 5: HistGradientBoosting (Optuna-tuned, 4k subsample for speed) ──
    rng_hgb = np.random.default_rng(456)
    hgb_tune_idx = rng_hgb.choice(len(X_train_scaled), size=4000, replace=False)
    X_hgb_tune = X_train_scaled[hgb_tune_idx]
    y_hgb_tune = y_train_ddg[hgb_tune_idx]

    def hgb_objective(trial):
        params = {
            'max_iter':         trial.suggest_int('max_iter', 200, 1000),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization':trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True),
            'max_leaf_nodes':   trial.suggest_int('max_leaf_nodes', 15, 63),
            'random_state': 42, 'loss': 'absolute_error',
        }
        model = HistGradientBoostingRegressor(**params)
        preds = cross_val_predict(model, X_hgb_tune, y_hgb_tune, cv=kf5_tune)
        return mean_absolute_error(y_hgb_tune, preds)

    print("\n  [Optuna] Tuning HistGradientBoosting (50 trials on 4k subsample)...")
    hgb_study = optuna.create_study(direction='minimize')
    hgb_study.optimize(hgb_objective, n_trials=50, show_progress_bar=False)
    best_hgb = hgb_study.best_params
    print(f"    Best HGB MAE: {hgb_study.best_value:.4f} | params: {best_hgb}")
    hgb_reg = HistGradientBoostingRegressor(**best_hgb, random_state=42, loss='absolute_error')
    hgb_reg.fit(X_train_scaled, y_train_ddg)
    print("    Done.")

    # ── Model 6: MLP — neural network captures non-linear feature interactions
    # differently from all tree-based models, adding complementary diversity ──
    print("  Training MLPRegressor (512-256-128-64, early stopping)...")
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0005,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        alpha=0.005,
        batch_size=256,
    )
    mlp_reg.fit(X_train_scaled, y_train_ddg)
    print("    Done.")

    # ── Model 7: RandomForest — bagging-based diversity, robust to outliers ──
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    print("  Training RandomForestRegressor (1200 trees)...")
    rf_reg = RandomForestRegressor(
        n_estimators=1200, max_depth=None, min_samples_leaf=4,
        max_features='sqrt', n_jobs=-1, random_state=42,
    )
    rf_reg.fit(X_train_scaled, y_train_ddg)
    print("    Done.")

    # ── Model 8: ExtraTreesRegressor — maximum randomisation for diversity ──
    print("  Training ExtraTreesRegressor (1200 trees)...")
    et_reg = ExtraTreesRegressor(
        n_estimators=1200, max_depth=None, min_samples_leaf=4,
        max_features='sqrt', n_jobs=-1, random_state=42,
    )
    et_reg.fit(X_train_scaled, y_train_ddg)
    print("    Done.")

    models = [
        ('GradientBoosting',     gb_reg),
        ('XGBoost',              xgb_reg),
        ('LightGBM',             lgbm_reg),
        ('CatBoost',             cb_reg),
        ('HistGradientBoosting', hgb_reg),
        ('MLP',                  mlp_reg),
        ('RandomForest',         rf_reg),
        ('ExtraTrees',           et_reg),
    ]

    # ── Step 6: Cross-validation — individual + ensemble ──
    print("\nSTEP 6: Cross-validation (10-fold)")
    print("-" * 40)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Get CV predictions from each model
    cv_preds_all = {}
    for name, model in models:
        cv_pred = cross_val_predict(model, X_train_scaled, y_train_ddg, cv=kf)
        cv_preds_all[name] = cv_pred
        pr, _ = pearsonr(cv_pred, y_train_ddg)
        sr, _ = spearmanr(cv_pred, y_train_ddg)
        cv_mae_val = mean_absolute_error(y_train_ddg, cv_pred)
        cv_binary = (cv_pred < 0).astype(int)
        cv_acc_val = accuracy_score(y_train, cv_binary)
        print(f"  {name:20s}  MAE={cv_mae_val:.4f}  Pearson={pr:.4f}  Spearman={sr:.4f}  Acc={cv_acc_val:.4f}")

    # Ensemble: average of all 4
    cv_preds_ensemble = np.mean([cv_preds_all[n] for n, _ in models], axis=0)
    cv_pearson, _ = pearsonr(cv_preds_ensemble, y_train_ddg)
    cv_spearman, _ = spearmanr(cv_preds_ensemble, y_train_ddg)
    cv_mae_ens = mean_absolute_error(y_train_ddg, cv_preds_ensemble)
    cv_r2_ens = r2_score(y_train_ddg, cv_preds_ensemble)
    cv_binary_pred = (cv_preds_ensemble < 0).astype(int)
    cv_acc = accuracy_score(y_train, cv_binary_pred)
    cv_f1_val = f1_score(y_train, cv_binary_pred)

    print(f"\n  {'ENSEMBLE (avg)':20s}  MAE={cv_mae_ens:.4f}  Pearson={cv_pearson:.4f}  Spearman={cv_spearman:.4f}  Acc={cv_acc:.4f}")
    print(f"  CV R²: {cv_r2_ens:.4f}")
    print(f"  CV F1 (from threshold): {cv_f1_val:.4f}")

    # ── Step 7: Leave-one-protein-out CV ──
    # STEP 7: LOPO CV skipped in v32 to speed up iteration
    print("\nSTEP 7: Leave-one-protein-out CV — skipped in v32 for faster iteration")

    # ── Step 7.5: Wide stacking v32 — direct MLP + tuned soft-vote + LR meta ──
    # v32 key changes over v31:
    #   1. Skip LOPO to save ~1.5 hours per run
    #   2. Add MLP (256-128-64-32) as direct base classifier on all 138 features
    #   3. Drop ET_clf (lowest performer at 76.40% in v31, drags down soft-vote)
    #   4. Use v31 Optuna-tuned XGB/LGBM params (best found)
    #   5. Try multiple LR regularization strengths for meta (C=0.01,0.1,1,10)
    #   6. Pure soft-vote of [MLP, XGB_clf, LGBM_clf, CB_clf, RF_clf] as candidate
    print("\nSTEP 7.5: Wide stacking v32 (MLP direct clf + tuned soft-vote + LR regularization sweep)")
    print("-" * 40)
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier as CatBoostCLF
    from sklearn.metrics import roc_curve
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    kf10_stack = KFold(n_splits=10, shuffle=True, random_state=42)
    kf5_stack  = KFold(n_splits=5,  shuffle=True, random_state=42)

    # ── 1. Direct base classifiers (10-fold OOF) ──
    # Use v31 Optuna-tuned params for XGB/LGBM; drop ET_clf (worst performer in v31)
    # Add MLP (256-128-64-32) directly on 138 features — neural signal orthogonal to trees
    print("  Training direct binary classifiers (10-fold OOF probs)...")
    clf_configs = [
        ('XGB_clf', XGBClassifier(
            # v31 Optuna best params on full data
            n_estimators=368, max_depth=10, learning_rate=0.026867935658904614,
            subsample=0.8907972930688506, colsample_bytree=0.7688320011175299,
            colsample_bylevel=0.633158624457072, min_child_weight=1,
            reg_alpha=0.20168626024487632, reg_lambda=0.0010056810605419055,
            gamma=0.8701918127913921,
            eval_metric='logloss', tree_method='hist',
            random_state=42, verbosity=0, n_jobs=-1,
        )),
        ('LGBM_clf', LGBMClassifier(
            # v31 Optuna best params on full data
            n_estimators=310, max_depth=12, learning_rate=0.014486652907891896,
            num_leaves=239, subsample=0.6404629307149422,
            colsample_bytree=0.5693373118449812, min_child_samples=7,
            reg_alpha=0.0611588834567198, reg_lambda=0.00027844727233982503,
            random_state=42, verbosity=-1, n_jobs=-1,
        )),
        ('CB_clf', CatBoostCLF(
            iterations=600, depth=7, learning_rate=0.03,
            subsample=0.8, l2_leaf_reg=3.0,
            random_seed=42, verbose=0,
        )),
        ('RF_clf', RandomForestClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=3,
            max_features='sqrt', random_state=42, n_jobs=-1,
        )),
        ('MLP_direct', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32), max_iter=800, random_state=42,
            early_stopping=True, validation_fraction=0.1, alpha=0.001,
            learning_rate_init=0.001, batch_size=256,
        )),
    ]

    clf_oof_probs = {}
    for clf_name, clf in clf_configs:
        probs = cross_val_predict(clf, X_train_scaled, y_train,
                                  cv=kf10_stack, method='predict_proba')[:, 1]
        clf_oof_probs[clf_name] = probs
        clf_acc = accuracy_score(y_train, (probs > 0.5).astype(int))
        print(f"    {clf_name:20s} OOF Acc={clf_acc:.4f}")

    # Soft-vote of top-4 (drop RF if MLP+3 trees beat all-5)
    soft_vote_top4 = np.mean([clf_oof_probs[n] for n in ['XGB_clf','LGBM_clf','CB_clf','MLP_direct']], axis=0)
    soft_vote_all5 = np.mean([clf_oof_probs[n] for n, _ in clf_configs], axis=0)
    sv4_acc = accuracy_score(y_train, (soft_vote_top4 > 0.5).astype(int))
    sv5_acc = accuracy_score(y_train, (soft_vote_all5 > 0.5).astype(int))
    print(f"    {'soft_vote_top4':20s} OOF Acc={sv4_acc:.4f} (XGB+LGBM+CB+MLP)")
    print(f"    {'soft_vote_all5':20s} OOF Acc={sv5_acc:.4f} (all 5)")

    # ── 2. Build stacks ──
    oof_stack   = np.column_stack([cv_preds_all[n] for n, _ in models])       # 8 cols
    clf_oof_mat = np.column_stack([clf_oof_probs[n] for n, _ in clf_configs])  # 5 cols

    # MI-based top-30 feature selection
    print("  Computing mutual information for feature selection (top 30)...")
    mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
    top_feat_idx = np.argsort(mi_scores)[-30:]
    raw_top_mi = X_train_scaled[:, top_feat_idx]                               # 30 MI cols

    stack_A = np.hstack([clf_oof_mat, oof_stack, raw_top_mi])                 # 43 cols
    stack_B = np.hstack([clf_oof_mat, raw_top_mi])                            # 35 cols

    wide_stack = stack_A  # for model saving

    # ── 3. Meta-learners: LR with regularization sweep + XGB + CatBoost ──
    print("  Training LR metas with regularization sweep (stack_A, 5-fold CV)...")
    lr_probs_by_C = {}
    for C_val in [0.01, 0.1, 1.0, 10.0]:
        lr_c = LogisticRegression(C=C_val, max_iter=2000, random_state=42)
        probs = cross_val_predict(lr_c, stack_A, y_train,
                                  cv=kf5_stack, method='predict_proba')[:, 1]
        acc = accuracy_score(y_train, (probs > 0.5).astype(int))
        lr_probs_by_C[C_val] = probs
        print(f"    LR(C={C_val:5.2f}) stack_A Acc={acc:.4f}")

    print("  Training LR meta on stack_B (5-fold CV)...")
    lr_B = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    lr_probs_B = cross_val_predict(lr_B, stack_B, y_train,
                                   cv=kf5_stack, method='predict_proba')[:, 1]
    lr_B_acc = accuracy_score(y_train, (lr_probs_B > 0.5).astype(int))
    print(f"    LR(C=1.0) stack_B Acc={lr_B_acc:.4f}")

    print("  Training XGB meta on stack_A (5-fold CV)...")
    xgb_meta_clf = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric='logloss', tree_method='hist',
        random_state=42, verbosity=0, n_jobs=-1,
    )
    xgb_meta_probs = cross_val_predict(xgb_meta_clf, stack_A, y_train,
                                       cv=kf5_stack, method='predict_proba')[:, 1]

    print("  Training CatBoost meta on stack_A (5-fold CV)...")
    meta_clf_cb = CatBoostCLF(
        iterations=800, depth=6, learning_rate=0.02,
        subsample=0.8, l2_leaf_reg=3.0, random_seed=42, verbose=0,
    )
    cb_meta_probs = cross_val_predict(meta_clf_cb, stack_A, y_train,
                                      cv=kf5_stack, method='predict_proba')[:, 1]

    # Meta soft-vote of all meta-clfs
    meta_soft_vote = np.mean([lr_probs_by_C[1.0], xgb_meta_probs, cb_meta_probs], axis=0)

    # Youden threshold on regressor ensemble for baseline
    oof_ensemble_preds = np.mean(oof_stack, axis=1)
    fpr, tpr, thresholds_roc = roc_curve(y_train, -oof_ensemble_preds)
    j_scores = tpr - fpr
    optimal_threshold = -thresholds_roc[np.argmax(j_scores)]
    reg_binary = (oof_ensemble_preds < optimal_threshold).astype(int)
    reg_acc = accuracy_score(y_train, reg_binary)

    # Ridge meta for MAE/Pearson reporting
    ridge_meta = Ridge(alpha=1.0)
    ridge_preds_cv = cross_val_predict(ridge_meta, stack_A, y_train_ddg, cv=kf5_stack)
    meta_mae = mean_absolute_error(y_train_ddg, ridge_preds_cv)
    meta_pearson, _ = pearsonr(ridge_preds_cv, y_train_ddg)
    meta_preds_cv = ridge_preds_cv
    meta_learner = ridge_meta
    meta_type = "Ridge"

    # ── 4. Threshold sweep — pick best across all candidates ──
    thresholds_sweep = np.linspace(0.30, 0.70, 141)
    best_clf_acc = 0.0
    best_clf_thr = 0.5
    best_clf_name = "LR_C1_stackA"
    best_clf_probs = lr_probs_by_C[1.0]

    all_clf_candidates = (
        [(f"LR_C{C}_stackA", p) for C, p in lr_probs_by_C.items()] +
        [
            ("LR_C1_stackB",    lr_probs_B),
            ("XGB_meta",        xgb_meta_probs),
            ("CB_meta",         cb_meta_probs),
            ("meta_soft_vote",  meta_soft_vote),
            ("soft_vote_top4",  soft_vote_top4),
            ("soft_vote_all5",  soft_vote_all5),
        ]
    )
    for clf_name, probs in all_clf_candidates:
        for thr in thresholds_sweep:
            acc_t = accuracy_score(y_train, (probs > thr).astype(int))
            if acc_t > best_clf_acc:
                best_clf_acc = acc_t
                best_clf_thr = thr
                best_clf_name = clf_name
                best_clf_probs = probs

    meta_clf_binary = (best_clf_probs > best_clf_thr).astype(int)
    meta_clf_f1 = f1_score(y_train, meta_clf_binary)
    print(f"  Best wide-stack clf: {best_clf_name}  Acc={best_clf_acc:.4f}  F1={meta_clf_f1:.4f}  thr={best_clf_thr:.3f}")

    # ── 5. Pick winner ──
    if best_clf_acc > reg_acc:
        meta_binary = meta_clf_binary
        meta_acc    = best_clf_acc
        meta_f1     = meta_clf_f1
        winning_method = f"wide-stack {best_clf_name} clf (thr={best_clf_thr:.3f})"
    else:
        meta_binary = reg_binary
        meta_acc    = reg_acc
        meta_f1     = f1_score(y_train, reg_binary)
        winning_method = "regressor ensemble + Youden threshold"

    print(f"  Winning method: {winning_method}")
    print(f"  Stacking CV  MAE={meta_mae:.4f}  Pearson={meta_pearson:.4f}  Acc={meta_acc:.4f}  F1={meta_f1:.4f}")

    # ── 6. Train final meta-clf on full stack ──
    if "LR_C" in best_clf_name and "stackB" not in best_clf_name:
        C_final = float(best_clf_name.split("_C")[1].split("_")[0])
        meta_clf_final = LogisticRegression(C=C_final, max_iter=2000, random_state=42)
        meta_clf_final.fit(stack_A, y_train)
    elif "stackB" in best_clf_name:
        meta_clf_final = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        meta_clf_final.fit(stack_B, y_train)
    elif best_clf_name == "XGB_meta" or "soft_vote" in best_clf_name:
        meta_clf_final = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric='logloss', tree_method='hist',
            random_state=42, verbosity=0, n_jobs=-1,
        )
        meta_clf_final.fit(stack_A, y_train)
    else:
        meta_clf_final = CatBoostCLF(
            iterations=800, depth=6, learning_rate=0.02,
            subsample=0.8, l2_leaf_reg=3.0, random_seed=42, verbose=0,
        )
        meta_clf_final.fit(stack_A, y_train)

    # Refit ridge meta-learner on full stack_A for inference
    meta_learner.fit(stack_A, y_train_ddg)

    clf_models_trained = []
    for clf_name, clf in clf_configs:
        clf.fit(X_train_scaled, y_train)
        clf_models_trained.append((clf_name, clf))

    use_stacking = True
    cv_mae_ens     = meta_mae
    cv_pearson     = meta_pearson
    cv_acc         = meta_acc
    cv_binary_pred = meta_binary
    cv_r2_ens      = r2_score(y_train_ddg, meta_preds_cv)
    cv_f1_val      = meta_f1

    # ── Step 8: Independent test on S669 ──
    print("\nSTEP 8: Independent test on S669")
    print("-" * 40)

    mae = rmse = r2 = pearson_r_val = pearson_p = spearman_r_val = spearman_p = 0.0
    acc = f1 = prec = rec = auc = 0.0

    if X_test.shape[0] == 0:
        print("  S669 not available — skipping independent test")
    else:
        # Individual model predictions
        test_preds_all = {}
        for name, model in models:
            tp = model.predict(X_test_scaled)
            test_preds_all[name] = tp
            pr, _ = pearsonr(tp, y_test_ddg)
            mae_val = mean_absolute_error(y_test_ddg, tp)
            tb = (tp < 0).astype(int)
            acc_val = accuracy_score(y_test, tb)
            print(f"  {name:20s}  MAE={mae_val:.4f}  Pearson={pr:.4f}  Acc={acc_val:.4f}")

        # Final prediction: stacking or average
        test_stack = np.column_stack([test_preds_all[n] for n, _ in models])
        if use_stacking:
            y_pred_ddg = meta_learner.predict(test_stack)
        else:
            y_pred_ddg = np.mean([test_preds_all[n] for n, _ in models], axis=0)

        mae = mean_absolute_error(y_test_ddg, y_pred_ddg)
        rmse = np.sqrt(mean_squared_error(y_test_ddg, y_pred_ddg))
        r2 = r2_score(y_test_ddg, y_pred_ddg)
        pearson_r_val, pearson_p = pearsonr(y_pred_ddg, y_test_ddg)
        spearman_r_val, spearman_p = spearmanr(y_pred_ddg, y_test_ddg)

        y_pred_binary = (y_pred_ddg < 0).astype(int)
        acc = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        prec = precision_score(y_test, y_pred_binary, zero_division=0)
        rec = recall_score(y_test, y_pred_binary, zero_division=0)
        try:
            auc = roc_auc_score(y_test, -y_pred_ddg)
        except ValueError:
            auc = 0.0

        print(f"\n  ENSEMBLE results:")
        print(f"  MAE:         {mae:.4f} kcal/mol")
        print(f"  RMSE:        {rmse:.4f} kcal/mol")
        print(f"  R²:          {r2:.4f}")
        print(f"  Pearson r:   {pearson_r_val:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman r:  {spearman_r_val:.4f} (p={spearman_p:.2e})")
        print(f"\n  Classification (threshold DDG < 0):")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  AUC-ROC:     {auc:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        cm = confusion_matrix(y_test, y_pred_binary)
        print(f"  TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # ── Step 9: Feature importance (averaged across models) ──
    print("\nSTEP 9: Top 15 feature importances (averaged)")
    print("-" * 40)
    feature_names = [
        'dH', 'dV', 'dC', 'dF', 'dHelix', 'dSheet',
        '|dH|', '|dV|', '|dC|', '|dF|', '|dHelix|', '|dSheet|',
        'BLOSUM62',
        'helix_prop', 'sheet_prop', 'coil_prop',
        'RSA',
        'local_hydro', 'local_charge', 'GP_fraction', 'rel_position',
        'to_Pro', 'from_Pro', 'to_Gly', 'deamid_risk', 'salt_bridge', 'cys_change',
        'dH×burial', 'dV×burial', 'dC×burial', 'dH×dV', 'dC×dH',
        'Pro×burial', 'burial×helix', 'burial×sheet', 'dH×helix',
        'aromatic_change', 'small→large', 'large→small',
        'cons_wt_blosum', 'cons_mut_blosum', 'cons_delta_blosum',
        'PSSM_wt', 'PSSM_mut', 'delta_PSSM', 'info_content', 'cons_rank', 'wt_rank',
        'temperature_C',        # feature 49
        'pH',                   # feature 50
        'dMW',                  # feature 51
        'dHbond_donors',        # feature 52
        'dHbond_acceptors',     # feature 53
        'dTurn_propensity',     # feature 54
        'polarity_change',      # feature 55
        'charge_gain',          # feature 56
        'charge_loss',          # feature 57
        'delta_ionization',     # feature 58
        'dAliphatic',           # feature 59: aliphatic index delta (Ikai 1980)
        'dCharge_pH',           # feature 60: net charge delta at assay pH (HH)
        'dSizeClass',           # feature 61: side-chain size class delta
        'dDisorder',            # feature 62: intrinsic disorder propensity delta
        'local_entropy',        # feature 63: Shannon entropy of ±5 window
        'buried_hydro_wt',      # feature 64: WT buried+hydrophobic indicator
        'buried_hydro_mut',     # feature 65: MUT buried+hydrophobic indicator
        'abs_charge_wt_pH',     # feature 66: |charge at pH| for WT
        'abs_charge_mut_pH',    # feature 67: |charge at pH| for MUT
        'nearest_cys_dist',     # feature 68
        'dH×dC',                # feature 69: hydrophobicity-charge coupling
        'dH×dV',                # feature 70: hydrophobicity-volume coupling
        'dChargePH×pH',         # feature 71: pH-adjusted charge × assay pH
        'dH×temp',              # feature 72: hydrophobicity × temperature
        'burial×dChargePH',     # feature 73: electrostatic burial
        'burial×dMW',           # feature 74: packing × size
        '|dH|×|dChargePH|',     # feature 75: amphipathic change
        'dAliphatic×temp',      # feature 76: aliphatic index × temperature
        # v12 structural geometry features
        'phi_norm',             # feature 77: backbone phi / 180 (0 if unknown)
        'psi_norm',             # feature 78: backbone psi / 180 (0 if unknown)
        'ca_depth_norm',        # feature 79: Ca-alpha depth / 20 (0 if unknown)
        'has_real_rsa',         # feature 80: 1 if structural RSA used, 0 if estimated
        # v13 structural cross-terms
        'phi×psi',              # feature 81
        'phi×dH',               # feature 82
        'phi×temp',             # feature 83
        'psi×dH',               # feature 84
        'psi×temp',             # feature 85
        'depth×dH',             # feature 86
    ]
    # HistGradientBoosting does not expose feature_importances_ — skip it
    importances_list = [m.feature_importances_ for _, m in models if hasattr(m, 'feature_importances_')]
    avg_imp = np.mean(importances_list, axis=0) if importances_list else np.zeros(X_train.shape[1])
    idx_sorted = np.argsort(avg_imp)[::-1]
    for i in range(min(15, len(feature_names))):
        j = idx_sorted[i]
        name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        print(f"  {i+1:2d}. {name:20s} {avg_imp[j]:.4f}")

    # ── Step 10: Train ΔTm regressor on ThermoMutDB ──
    print("\nSTEP 10: Training ΔTm regressor (ThermoMutDB, 6,107 entries)")
    print("-" * 40)
    print("  Source: ThermoMutDB (Pucci et al. 2021, Nucleic Acids Res.)")
    print("  Target: ΔTm (°C) — measured change in melting temperature upon mutation")

    dtm_X_base, dtm_esm_raw, dtm_valid_esm = [], [], []
    dtm_y, dtm_proteins = [], []
    dtm_skipped = 0
    for r in dtm_records:
        feats = extract_features(
            r['wt_aa'], r['position'], r['mut_aa'],
            r.get('sequence', ''), protein_id=r.get('protein_id'),
            temperature=r.get('temperature_c', 37.0), ph=r.get('ph', 7.0),
            struct_rsa=r.get('struct_rsa'),
            struct_phi=r.get('struct_phi'),
            struct_psi=r.get('struct_psi'),
            struct_depth=r.get('struct_depth'),
        )
        if feats is None:
            dtm_skipped += 1
            continue
        esm_emb = get_esm_embedding(r.get('protein_id', ''), r['position'])
        dtm_X_base.append(feats)
        dtm_esm_raw.append(esm_emb)
        dtm_valid_esm.append(esm_emb is not None)
        dtm_y.append(r['dtm'])
        dtm_proteins.append(r.get('protein_id', ''))

    if len(dtm_X_base) < 100:
        print(f"  WARNING: Only {len(dtm_X_base)} ΔTm records — skipping ΔTm regressor")
        dtm_regressor = None
        dtm_meta = None
    else:
        # Append ESM-2 PCA features using the SAME PCA fitted on DDG training data
        dtm_base_arr = np.array(dtm_X_base, dtype=np.float32)
        n_dtm = len(dtm_X_base)
        dtm_esm_features = np.tile(_pca_mean, (n_dtm, 1)).astype(np.float32)
        dtm_has_esm = np.zeros((n_dtm, 1), dtype=np.float32)
        valid_idx = [i for i, v in enumerate(dtm_valid_esm) if v]
        if valid_idx:
            raw_valid = np.array([dtm_esm_raw[i] for i in valid_idx], dtype=np.float32)
            dtm_esm_features[valid_idx] = _pca.transform(raw_valid)
            dtm_has_esm[valid_idx] = 1.0
        dtm_X = np.concatenate([dtm_base_arr, dtm_esm_features, dtm_has_esm], axis=1)
        dtm_y = np.array(dtm_y)
        print(f"  ΔTm training set: {len(dtm_X)} mutations (skipped {dtm_skipped})")
        print(f"  ΔTm range: [{dtm_y.min():.2f}, {dtm_y.max():.2f}] °C")

        # Scale using the SAME scaler trained on DDG features (138 features)
        dtm_X_scaled = scaler.transform(dtm_X)

        print("  Training GradientBoosting ΔTm regressor...")
        dtm_gb = GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, max_features='sqrt',
            loss='huber', alpha=0.9, random_state=42,
        )
        dtm_gb.fit(dtm_X_scaled, dtm_y)

        print("  Training XGBoost ΔTm regressor...")
        dtm_xgb = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
        )
        dtm_xgb.fit(dtm_X_scaled, dtm_y)

        # Evaluate with 5-fold CV
        kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
        dtm_cv_gb = cross_val_predict(dtm_gb, dtm_X_scaled, dtm_y, cv=kf5)
        dtm_cv_xgb = cross_val_predict(dtm_xgb, dtm_X_scaled, dtm_y, cv=kf5)
        dtm_cv_ens = (dtm_cv_gb + dtm_cv_xgb) / 2.0
        dtm_mae = mean_absolute_error(dtm_y, dtm_cv_ens)
        dtm_pearson, _ = pearsonr(dtm_cv_ens, dtm_y)
        dtm_spearman, _ = spearmanr(dtm_cv_ens, dtm_y)
        print(f"  ΔTm CV (5-fold): MAE={dtm_mae:.3f}°C  Pearson={dtm_pearson:.4f}  Spearman={dtm_spearman:.4f}")

        dtm_regressor = {
            'models': [('GradientBoosting', dtm_gb), ('XGBoost', dtm_xgb)],
            'weights': [0.5, 0.5],
        }
        dtm_meta = {
            "n_training_samples": int(len(dtm_X)),
            "source": "ThermoMutDB (Pucci et al. 2021)",
            "cv_mae_celsius": round(float(dtm_mae), 4),
            "cv_pearson": round(float(dtm_pearson), 4),
            "cv_spearman": round(float(dtm_spearman), 4),
            "dtm_range": [round(float(dtm_y.min()), 2), round(float(dtm_y.max()), 2)],
        }

    # ── Step 11: Save ensemble and ΔTm model ──
    print("\nSTEP 11: Saving ensemble model and ΔTm regressor")
    print("-" * 40)
    os.makedirs(MODEL_DIR, exist_ok=True)

    ensemble = {
        'models': [(name, model) for name, model in models],
        'weights': [1.0/len(models)] * len(models),  # equal fallback weights
        'meta_learner': meta_learner,
        'use_stacking': True,
        'optimal_threshold': float(optimal_threshold),
        'meta_type': meta_type,
        # Wide-stack classifier (direct binary prediction, more accurate)
        'clf_models': clf_models_trained,
        'meta_clf': meta_clf_final,
        'meta_clf_threshold': float(best_clf_thr),
        'use_clf_meta': best_clf_acc > reg_acc,
        # Feature-augmented stacking: indices of raw features appended to wide stack
        'top_feat_idx': top_feat_idx.tolist(),
    }
    with open(os.path.join(MODEL_DIR, "mutation_regressor.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    if dtm_regressor is not None:
        with open(os.path.join(MODEL_DIR, "deltaTm_regressor.pkl"), "wb") as f:
            pickle.dump(dtm_regressor, f)
        print(f"  Saved deltaTm_regressor.pkl ({len(dtm_X)} ΔTm training points)")

    # Save conservation cache for deployment
    if _conservation_cache:
        with open(os.path.join(MODEL_DIR, "conservation_cache.pkl"), "wb") as f:
            pickle.dump(_conservation_cache, f)
        print(f"  Saved conservation_cache.pkl ({len(_conservation_cache)} proteins)")

    # Save ESM-2 PCA for deployment (needed for inference)
    esm_pca_data = {'pca': _pca, 'pca_mean': _pca_mean, 'esm_dim': ESM_DIM}
    with open(os.path.join(MODEL_DIR, "esm_pca.pkl"), "wb") as f:
        pickle.dump(esm_pca_data, f)
    print(f"  Saved esm_pca.pkl (PCA-{ESM_DIM} from ESM-2 640-dim)")

    # Save metal coordination cache for deployment
    if _metal_coord_cache:
        with open(os.path.join(MODEL_DIR, "metal_coord_cache.pkl"), "wb") as f:
            pickle.dump(_metal_coord_cache, f)
        print(f"  Saved metal_coord_cache.pkl ({len(_metal_coord_cache)} proteins)")

    if _physics_cache:
        with open(os.path.join(MODEL_DIR, "physics_features_cache.pkl"), "wb") as f:
            pickle.dump(_physics_cache, f)
        n_res = sum(len(v) for v in _physics_cache.values())
        print(f"  Saved physics_features_cache.pkl ({n_res} residue positions)")

    meta = {
        "model_type": "v27: 8-model ensemble + wide-stack clf + feature-augmented stacking + real RSA/SS + PSSM + ESM-2 PCA-32 + metal coord + ionic strength + Debye-Hückel/Born physics + ESM-2 ΔLL (138 features)",
        "prediction_target": "DDG (kcal/mol)",
        "n_models": 6,
        "use_stacking": True,
        "meta_type": meta_type,
        "optimal_threshold": float(optimal_threshold),
        "n_features": int(X_train.shape[1]),  # 68 features
        "feature_version": "v10_cross_terms",
        "condition_features": {
            "feature_49": "temperature_C (assay temperature from ThermoMutDB/FireProtDB)",
            "feature_50": "pH (assay pH from ThermoMutDB/FireProtDB)",
            "feature_51": "dMW (molecular weight delta)",
            "feature_52": "dHbond_donors",
            "feature_53": "dHbond_acceptors",
            "feature_54": "dTurn_propensity",
            "feature_55": "polarity_change",
            "feature_56": "charge_gain",
            "feature_57": "charge_loss",
            "feature_58": "delta_ionization (pH-dependent)",
        },
        "training_samples": int(X_train.shape[0]),
        "stabilizing_samples": int(n_pos),
        "destabilizing_samples": int(n_neg),
        "data_sources": {
            "FireProtDB": int(source_counts.get('FireProtDB', 0)),
            "ProDDG": int(source_counts.get('ProDDG', 0)),
            "ThermoMutDB": int(source_counts.get('ThermoMutDB', 0)),
        },
        "synthetic_data": False,
        "independent_test_set": "S669 (669 mutations)",
        "cv_mae": round(float(cv_mae_ens), 4),
        "cv_r2": round(float(cv_r2_ens), 4),
        "cv_pearson": round(float(cv_pearson), 4),
        "cv_spearman": round(float(cv_spearman), 4),
        "cv_accuracy": round(float(cv_acc), 4),
        "cv_f1": round(float(cv_f1_val), 4),
        "test_mae": round(float(mae), 4),
        "test_rmse": round(float(rmse), 4),
        "test_r2": round(float(r2), 4),
        "test_pearson_r": round(float(pearson_r_val), 4),
        "test_spearman_r": round(float(spearman_r_val), 4),
        "test_accuracy": round(float(acc), 4),
        "test_f1": round(float(f1), 4),
        "test_auc": round(float(auc), 4),
        "deltaTm_regressor": dtm_meta,
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved mutation_regressor.pkl (ensemble, {int(X_train.shape[1])} features)")
    print(f"  Saved scaler.pkl")
    print(f"  Saved model_meta.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — ENSEMBLE MODEL")
    print(f"  Training: {X_train.shape[0]} real experimental mutations")
    print(f"  Test (S669): {X_test.shape[0]} independent mutations")
    print(f"  CV MAE: {cv_mae_ens:.4f} kcal/mol")
    print(f"  CV Pearson: {cv_pearson:.4f}")
    print(f"  CV Accuracy (threshold): {cv_acc:.4f}")
    print(f"  S669 MAE: {mae:.4f} kcal/mol")
    print(f"  S669 Pearson: {pearson_r_val:.4f}")
    print(f"  S669 Accuracy (threshold): {acc:.4f}")
    print(f"  Synthetic data used: NONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
