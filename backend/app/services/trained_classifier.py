"""Publication-ready ensemble regressor for mutation thermostability prediction.

Predicts DDG (kcal/mol) using an ensemble of 3 models:
  - GradientBoostingRegressor (sklearn)
  - XGBRegressor (xgboost)
  - RandomForestRegressor (sklearn)
Final prediction = average of all 3.

Trained on real experimental data only (no synthetic mutations):
- FireProtDB: ~3,400 curated mutations with DDG values
- ProDDG (S2648): ~2,300 mutations with DDG values
- ThermoMutDB: ~4,000 mutations with DDG values

42 features.
"""

import numpy as np
import pickle
import os
import json

# Path to pre-trained model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "mutation_regressor.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
CONSERVATION_PATH = os.path.join(MODEL_DIR, "conservation_cache.pkl")
DTM_REGRESSOR_PATH = os.path.join(MODEL_DIR, "deltaTm_regressor.pkl")

_ensemble = None  # dict with 'models' list and 'weights'
_scaler = None
_training_metrics = None
_conservation_cache = None
_dtm_ensemble = None   # ΔTm regressor (GradientBoosting + XGBoost, ThermoMutDB)
_n_features = 48       # updated from model_meta.json after load

# ═══════════════════════════════════════════════════════════
# Amino acid property tables
# ═══════════════════════════════════════════════════════════
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

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

MOLECULAR_WEIGHT = {
    'A': 89.1,  'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.0,  'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.1, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2,
}
HBOND_DONORS = {
    'A': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1,
    'G': 1, 'H': 2, 'I': 1, 'K': 2, 'L': 1,
    'M': 1, 'N': 2, 'P': 0, 'Q': 2, 'R': 4,
    'S': 2, 'T': 2, 'V': 1, 'W': 2, 'Y': 2,
}
HBOND_ACCEPTORS = {
    'A': 1, 'C': 0, 'D': 3, 'E': 3, 'F': 0,
    'G': 1, 'H': 1, 'I': 1, 'K': 1, 'L': 1,
    'M': 2, 'N': 2, 'P': 1, 'Q': 2, 'R': 1,
    'S': 2, 'T': 2, 'V': 1, 'W': 0, 'Y': 1,
}
TURN_PROPENSITY = {
    'A': 0.66, 'C': 1.19, 'D': 1.46, 'E': 0.74, 'F': 0.60,
    'G': 1.56, 'H': 0.95, 'I': 0.47, 'K': 1.01, 'L': 0.59,
    'M': 0.60, 'N': 1.56, 'P': 1.52, 'Q': 0.98, 'R': 0.95,
    'S': 1.43, 'T': 0.96, 'V': 0.50, 'W': 0.96, 'Y': 1.14,
}
POLARITY_CLASS = {
    'A': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 0,
    'G': 0, 'H': 2, 'I': 0, 'K': 2, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 2,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1,
}
SIDECHAIN_PKA = {
    'A': 0.0,  'C': 8.3,  'D': 3.9,  'E': 4.1,  'F': 0.0,
    'G': 0.0,  'H': 6.0,  'I': 0.0,  'K': 10.5, 'L': 0.0,
    'M': 0.0,  'N': 0.0,  'P': 0.0,  'Q': 0.0,  'R': 12.5,
    'S': 0.0,  'T': 0.0,  'V': 0.0,  'W': 0.0,  'Y': 10.1,
}
SIDECHAIN_IS_ACID = {
    'C': True, 'D': True, 'E': True, 'Y': True,
    'H': False, 'K': False, 'R': False,
}
ALIPHATIC_CONTRIB = {'A': 1.0, 'V': 2.9, 'I': 3.9, 'L': 3.9}
SIZE_CLASS = {
    'G': 0, 'A': 0,
    'S': 1, 'C': 1, 'T': 1, 'P': 1, 'D': 1, 'N': 1, 'V': 1,
    'E': 2, 'Q': 2, 'I': 2, 'L': 2, 'M': 2, 'H': 2, 'K': 2,
    'F': 3, 'R': 3, 'W': 3, 'Y': 3,
}
DISORDER_PROPENSITY = {
    'A': 0.06, 'C': 0.02, 'D': 0.19, 'E': 0.18, 'F': -0.05,
    'G': 0.17, 'H': 0.04, 'I': -0.07, 'K': 0.16, 'L': -0.07,
    'M': 0.00, 'N': 0.14, 'P': 0.12, 'Q': 0.15, 'R': 0.14,
    'S': 0.13, 'T': 0.07, 'V': -0.06, 'W': -0.05, 'Y': -0.01,
}

def _charge_at_ph(aa: str, ph: float) -> float:
    pka = SIDECHAIN_PKA.get(aa, 0.0)
    if pka == 0:
        return 0.0
    is_acid = SIDECHAIN_IS_ACID.get(aa, True)
    if is_acid:
        return -1.0 / (1.0 + 10.0 ** (ph - pka))
    else:
        return  1.0 / (1.0 + 10.0 ** (pka - ph))

BLOSUM62_DIAG = {
    'A': 4, 'R': 5, 'N': 6, 'D': 6, 'C': 9,
    'Q': 5, 'E': 5, 'G': 6, 'H': 8, 'I': 4,
    'L': 4, 'K': 5, 'M': 5, 'F': 6, 'P': 7,
    'S': 4, 'T': 5, 'W': 11, 'Y': 7, 'V': 4,
}

_BLOSUM62 = {}
_blosum_str = """
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
_lines = [l for l in _blosum_str.strip().split('\n') if l.strip()]
_header = _lines[0].split()
for _line in _lines[1:]:
    _parts = _line.split()
    _aa1 = _parts[0]
    for _j, _aa2 in enumerate(_header):
        _BLOSUM62[(_aa1, _aa2)] = int(_parts[_j + 1])


# ═══════════════════════════════════════════════════════════
# Feature extraction (42 features — matches training script)
# ═══════════════════════════════════════════════════════════

def _estimate_rsa(sequence, position):
    if not sequence or position < 1 or position > len(sequence):
        return 0.5
    idx = position - 1
    aa = sequence[idx]
    h = HYDROPHOBICITY.get(aa, 0)
    base = 0.5 - h * 0.05
    rel_pos = idx / max(len(sequence) - 1, 1)
    if rel_pos < 0.05 or rel_pos > 0.95:
        base += 0.2
    window = sequence[max(0, idx-3):idx+4]
    avg_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
    base -= avg_h * 0.02
    return max(0.0, min(1.0, base))


def _estimate_secondary_structure(sequence, position):
    if not sequence or position < 1 or position > len(sequence):
        return 0.33, 0.33, 0.34
    idx = position - 1
    window = sequence[max(0, idx-4):idx+5]
    h_score = np.mean([HELIX_PROPENSITY.get(a, 1.0) for a in window])
    s_score = np.mean([SHEET_PROPENSITY.get(a, 1.0) for a in window])
    total = h_score + s_score + 1.0
    return h_score / total, s_score / total, 1.0 / total


PSSM_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")

def _get_conservation_features(protein_id, position, wt_aa, mut_aa):
    """Extract 6 PSSM conservation features."""
    if not _conservation_cache or not protein_id:
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


def _extract_features(wt_aa: str, position: int, mut_aa: str,
                      sequence: str = None, protein_id: str = None,
                      temperature: float = 25.0, ph: float = 7.0, **kwargs) -> list[float]:
    """Extract features for a single mutation.

    Returns 50 features when the model was trained with condition features (v7+),
    or 48 features for backward compatibility with older model files.
    Features 49-50 are assay temperature (°C) and pH.
    """
    if wt_aa not in AA_SET or mut_aa not in AA_SET:
        return [0.0] * _n_features

    features = []

    # Physicochemical deltas (6)
    dH = HYDROPHOBICITY.get(mut_aa, 0) - HYDROPHOBICITY.get(wt_aa, 0)
    dV = VOLUME.get(mut_aa, 0) - VOLUME.get(wt_aa, 0)
    dC = CHARGE.get(mut_aa, 0) - CHARGE.get(wt_aa, 0)
    dF = FLEXIBILITY.get(mut_aa, 0) - FLEXIBILITY.get(wt_aa, 0)
    dHelix = HELIX_PROPENSITY.get(mut_aa, 1) - HELIX_PROPENSITY.get(wt_aa, 1)
    dSheet = SHEET_PROPENSITY.get(mut_aa, 1) - SHEET_PROPENSITY.get(wt_aa, 1)
    features.extend([dH, dV, dC, dF, dHelix, dSheet])

    # Absolute deltas (6)
    features.extend([abs(dH), abs(dV), abs(dC), abs(dF), abs(dHelix), abs(dSheet)])

    # BLOSUM62 (1)
    features.append(_BLOSUM62.get((wt_aa, mut_aa), 0))

    # Secondary structure at position (3)
    if sequence:
        h, s, c = _estimate_secondary_structure(sequence, position)
    else:
        h, s, c = 0.33, 0.33, 0.34
    features.extend([h, s, c])

    # RSA (1)
    rsa = _estimate_rsa(sequence, position) if sequence else 0.5
    features.append(rsa)

    # Sequence context (4)
    if sequence and 1 <= position <= len(sequence):
        idx = position - 1
        window = sequence[max(0, idx-3):idx+4]
        local_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
        local_c = np.mean([CHARGE.get(a, 0) for a in window])
        gp_count = sum(1 for a in window if a in ('G', 'P'))
        rel_pos = idx / max(len(sequence) - 1, 1)
        features.extend([local_h, local_c, gp_count / len(window), rel_pos])
    else:
        features.extend([0, 0, 0, 0.5])

    # Thermostability features (6)
    to_proline = 1.0 if mut_aa == 'P' and wt_aa != 'P' else 0.0
    from_proline = 1.0 if wt_aa == 'P' and mut_aa != 'P' else 0.0
    to_glycine = 1.0 if mut_aa == 'G' and wt_aa != 'G' else 0.0
    deamid_risk = 0.0
    if wt_aa in ('N', 'Q') and mut_aa not in ('N', 'Q'):
        deamid_risk = -1.0
    elif mut_aa in ('N', 'Q') and wt_aa not in ('N', 'Q'):
        deamid_risk = 1.0
    salt_bridge = 0.0
    if mut_aa in ('D', 'E', 'K', 'R') and wt_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = 1.0
    elif wt_aa in ('D', 'E', 'K', 'R') and mut_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = -1.0
    cys_change = 0.0
    if mut_aa == 'C' and wt_aa != 'C':
        cys_change = 1.0
    elif wt_aa == 'C' and mut_aa != 'C':
        cys_change = -1.0
    features.extend([to_proline, from_proline, to_glycine, deamid_risk, salt_bridge, cys_change])

    # Interaction terms (9)
    burial = 1.0 - rsa
    features.extend([
        abs(dH) * burial, abs(dV) * burial, abs(dC) * burial,
        abs(dH) * abs(dV), abs(dC) * abs(dH),
        to_proline * burial, burial * h, burial * s, abs(dH) * h,
    ])

    # Additional (6)
    aromatic_wt = 1.0 if wt_aa in ('F', 'W', 'Y', 'H') else 0.0
    aromatic_mut = 1.0 if mut_aa in ('F', 'W', 'Y', 'H') else 0.0
    small_aa = {'G', 'A', 'S', 'T', 'C'}
    large_aa = {'F', 'W', 'Y', 'R', 'K', 'H'}
    small_to_large = 1.0 if wt_aa in small_aa and mut_aa in large_aa else 0.0
    large_to_small = 1.0 if wt_aa in large_aa and mut_aa in small_aa else 0.0
    cons_wt = BLOSUM62_DIAG.get(wt_aa, 4)
    cons_mut = BLOSUM62_DIAG.get(mut_aa, 4)
    features.extend([
        aromatic_wt - aromatic_mut, small_to_large, large_to_small,
        cons_wt, cons_mut, cons_wt - cons_mut,
    ])

    # PSSM conservation features (6)
    cons_feats = _get_conservation_features(protein_id, position, wt_aa, mut_aa)
    features.extend(cons_feats)

    # Condition features (2) — only appended when model expects 50+ features
    if _n_features >= 50:
        features.append(float(temperature))  # feature 49
        features.append(float(ph))           # feature 50

    # Extended biochemical features (8) — features 51-58 for v8+ model
    if _n_features >= 58:
        import math
        dMW = MOLECULAR_WEIGHT.get(mut_aa, 130.0) - MOLECULAR_WEIGHT.get(wt_aa, 130.0)
        dHD = float(HBOND_DONORS.get(mut_aa, 1)    - HBOND_DONORS.get(wt_aa, 1))
        dHA = float(HBOND_ACCEPTORS.get(mut_aa, 1) - HBOND_ACCEPTORS.get(wt_aa, 1))
        dTurn = TURN_PROPENSITY.get(mut_aa, 1.0) - TURN_PROPENSITY.get(wt_aa, 1.0)
        pol_wt  = POLARITY_CLASS.get(wt_aa, 0)
        pol_mut = POLARITY_CLASS.get(mut_aa, 0)
        pol_change  = float(pol_mut - pol_wt)
        charge_gain = 1.0 if pol_mut == 2 and pol_wt != 2 else 0.0
        charge_loss = 1.0 if pol_wt  == 2 and pol_mut != 2 else 0.0
        pka_wt  = SIDECHAIN_PKA.get(wt_aa,  0.0)
        pka_mut = SIDECHAIN_PKA.get(mut_aa, 0.0)
        ion_wt  = 1.0 / (1.0 + 10.0 ** (pka_wt  - float(ph))) if pka_wt  > 0 else 0.0
        ion_mut = 1.0 / (1.0 + 10.0 ** (pka_mut - float(ph))) if pka_mut > 0 else 0.0
        features.extend([dMW, dHD, dHA, dTurn, pol_change, charge_gain, charge_loss, ion_mut - ion_wt])

    # Further extended features (10) — features 59-68 for v9 model
    if _n_features >= 68:
        dAliphatic = ALIPHATIC_CONTRIB.get(mut_aa, 0.0) - ALIPHATIC_CONTRIB.get(wt_aa, 0.0)
        ch_wt  = _charge_at_ph(wt_aa,  ph)
        ch_mut = _charge_at_ph(mut_aa, ph)
        dCharge_ph = ch_mut - ch_wt
        dSizeClass = float(SIZE_CLASS.get(mut_aa, 1) - SIZE_CLASS.get(wt_aa, 1))
        dDisorder  = DISORDER_PROPENSITY.get(mut_aa, 0.0) - DISORDER_PROPENSITY.get(wt_aa, 0.0)
        if sequence and 1 <= position <= len(sequence):
            idx = position - 1
            win = sequence[max(0, idx-5):idx+6]
            aa_counts = {a: win.count(a) for a in set(win)}
            n = len(win)
            entropy = -sum((c/n) * np.log2(c/n) for c in aa_counts.values() if c > 0)
        else:
            entropy = 2.0
        rsa_here = _estimate_rsa(sequence, position) if sequence else 0.5
        buried_h_wt  = 1.0 if (rsa_here < 0.25 and HYDROPHOBICITY.get(wt_aa, 0)  > 1.5) else 0.0
        buried_h_mut = 1.0 if (rsa_here < 0.25 and HYDROPHOBICITY.get(mut_aa, 0) > 1.5) else 0.0
        abs_ch_wt  = abs(ch_wt)
        abs_ch_mut = abs(ch_mut)
        if sequence and 1 <= position <= len(sequence):
            idx = position - 1
            cys_positions = [i for i, a in enumerate(sequence) if a == 'C']
            nearest_cys_dist = (min(abs(idx - cp) for cp in cys_positions) / max(len(sequence), 1)
                                if cys_positions else 1.0)
        else:
            nearest_cys_dist = 1.0
        features.extend([
            dAliphatic, dCharge_ph, dSizeClass, dDisorder,
            entropy, buried_h_wt, buried_h_mut, abs_ch_wt, abs_ch_mut, nearest_cys_dist,
        ])

    return features


# ═══════════════════════════════════════════════════════════
# Model loading and prediction
# ═══════════════════════════════════════════════════════════

def train_model(force_retrain: bool = False) -> dict:
    """Load pre-trained ensemble from disk."""
    global _ensemble, _scaler, _training_metrics, _conservation_cache, _dtm_ensemble, _n_features

    # Already loaded — skip disk I/O
    if not force_retrain and _ensemble is not None and _scaler is not None:
        return _training_metrics or {}

    if not force_retrain and os.path.exists(REGRESSOR_PATH) and os.path.exists(SCALER_PATH):
        import sys
        print(f"[trained_classifier] Loading ensemble from {REGRESSOR_PATH}...", file=sys.stderr)
        with open(REGRESSOR_PATH, "rb") as f:
            _ensemble = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)

        # Detect feature count from model metadata (backward-compatible)
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                _training_metrics = json.load(f)
            _n_features = int(_training_metrics.get("n_features", 48))
            _training_metrics["loaded_from_cache"] = True
        else:
            _training_metrics = {
                "model_type": "Ensemble (GradientBoosting + XGBoost + RandomForest)",
                "loaded_from_cache": True,
            }
            _n_features = 48  # assume legacy model

        print(f"[trained_classifier] Model loaded: {_n_features} features", file=sys.stderr)

        # Load ΔTm regressor (optional — only present after v7 retrain)
        if os.path.exists(DTM_REGRESSOR_PATH):
            try:
                with open(DTM_REGRESSOR_PATH, "rb") as f:
                    _dtm_ensemble = pickle.load(f)
                print(f"[trained_classifier] ΔTm regressor loaded", file=sys.stderr)
            except Exception as e:
                print(f"[trained_classifier] WARNING: Could not load ΔTm regressor: {e}", file=sys.stderr)
                _dtm_ensemble = None
        else:
            _dtm_ensemble = None
            print(f"[trained_classifier] No ΔTm regressor found (run retrain to enable)", file=sys.stderr)

        # Load conservation cache if available
        if os.path.exists(CONSERVATION_PATH):
            try:
                with open(CONSERVATION_PATH, "rb") as f:
                    _conservation_cache = pickle.load(f)
                print(f"[trained_classifier] Conservation cache loaded ({len(_conservation_cache)} entries)", file=sys.stderr)
            except Exception as e:
                print(f"[trained_classifier] WARNING: Could not load conservation cache: {e}", file=sys.stderr)
                _conservation_cache = None

        print(f"[trained_classifier] Model loaded successfully", file=sys.stderr)
        return _training_metrics

    raise FileNotFoundError(
        f"Pre-trained model not found at {REGRESSOR_PATH}. "
        "Run train_publication_model.py first."
    )


def _ensemble_predict(features_scaled):
    """Get ensemble DDG prediction — stacking meta-learner if available, else weighted average."""
    base_preds = [model.predict(features_scaled) for (_, model) in _ensemble['models']]

    if _ensemble.get('use_stacking') and _ensemble.get('meta_learner') is not None:
        stack = np.column_stack(base_preds)
        return _ensemble['meta_learner'].predict(stack)

    weights = _ensemble.get('weights', [1.0 / len(base_preds)] * len(base_preds))
    return np.sum([p * w for p, w in zip(base_preds, weights)], axis=0) / sum(weights)


def get_optimal_threshold() -> float:
    """Return the Youden-optimal DDG classification threshold (default 0.0 if not set)."""
    if _ensemble is None:
        return 0.0
    return _ensemble.get('optimal_threshold', 0.0)


def predict_mutation(wt_aa: str, position: int, mut_aa: str,
                     sequence: str = None, protein_id: str = None, **kwargs) -> dict:
    """Predict DDG for a mutation using ensemble and derive stability classification."""
    global _ensemble, _scaler

    if _ensemble is None:
        train_model()

    features = np.array([_extract_features(wt_aa, position, mut_aa,
                                           sequence=sequence, protein_id=protein_id)])
    features_scaled = _scaler.transform(features)

    predicted_ddg = float(_ensemble_predict(features_scaled)[0])

    # DDG < 0 means stabilizing (beneficial)
    is_beneficial = predicted_ddg < 0

    # Convert DDG to a probability-like confidence score
    confidence = 1.0 / (1.0 + np.exp(predicted_ddg))  # sigmoid(-ddg)

    return {
        "predicted_beneficial": is_beneficial,
        "predicted_ddg": round(predicted_ddg, 4),
        "confidence": round(float(confidence), 4),
        "probability_beneficial": round(float(confidence), 4),
    }


def _extract_features_batch(mutation_tuples: list[tuple], sequence: str = None,
                            protein_id: str = None,
                            temperature: float = 25.0, ph: float = 7.0) -> np.ndarray:
    """Vectorized feature extraction for many mutations at once.

    Much faster than calling _extract_features in a loop because it
    pre-computes shared sequence-level data and caches position-dependent
    features (RSA, secondary structure, context, PSSM) that are identical
    across all mutations at the same residue position.

    temperature: assay temperature in °C (user-selected, applied to all mutations)
    ph: assay pH (user-selected, applied to all mutations)
    """
    n = len(mutation_tuples)
    features = np.zeros((n, _n_features), dtype=np.float64)

    # Pre-compute sequence-level data once (shared across all mutations)
    seq_len = len(sequence) if sequence else 0
    if sequence:
        seq_hydro = np.array([HYDROPHOBICITY.get(a, 0.0) for a in sequence])
        seq_charge = np.array([CHARGE.get(a, 0.0) for a in sequence])
        seq_helix = np.array([HELIX_PROPENSITY.get(a, 1.0) for a in sequence])
        seq_sheet = np.array([SHEET_PROPENSITY.get(a, 1.0) for a in sequence])

    # Local references for tight loop
    _h = HYDROPHOBICITY
    _v = VOLUME
    _c = CHARGE
    _f = FLEXIBILITY
    _hp = HELIX_PROPENSITY
    _sp = SHEET_PROPENSITY
    _b62 = _BLOSUM62
    _bd = BLOSUM62_DIAG
    _aromatic = frozenset('FWYH')
    _small = frozenset('GASTC')
    _large = frozenset('FWYRKH')
    _charged = frozenset('DEKR')
    _deamid = frozenset('NQ')
    _gp = frozenset('GP')

    # Cache position-dependent features (RSA, secondary structure, context, PSSM)
    # These are identical for all mutations at the same position (~19x savings)
    _pos_cache = {}

    for i, (wt_aa, position, mut_aa) in enumerate(mutation_tuples):
        if wt_aa not in AA_SET or mut_aa not in AA_SET:
            continue

        # Physicochemical deltas (6)
        dH = _h.get(mut_aa, 0) - _h.get(wt_aa, 0)
        dV = _v.get(mut_aa, 0) - _v.get(wt_aa, 0)
        dC = _c.get(mut_aa, 0) - _c.get(wt_aa, 0)
        dF = _f.get(mut_aa, 0) - _f.get(wt_aa, 0)
        dHelix = _hp.get(mut_aa, 1) - _hp.get(wt_aa, 1)
        dSheet = _sp.get(mut_aa, 1) - _sp.get(wt_aa, 1)
        features[i, 0:6] = [dH, dV, dC, dF, dHelix, dSheet]

        # Absolute deltas (6)
        adH, adV, adC, adF = abs(dH), abs(dV), abs(dC), abs(dF)
        adHelix, adSheet = abs(dHelix), abs(dSheet)
        features[i, 6:12] = [adH, adV, adC, adF, adHelix, adSheet]

        # BLOSUM62 (1)
        features[i, 12] = _b62.get((wt_aa, mut_aa), 0)

        # Position-dependent features — compute once per position, reuse ~19x
        if position in _pos_cache:
            h_norm, s_norm, c_norm, rsa, local_h, local_c, gp_frac, rel_pos, cons_wt_feats = _pos_cache[position]
        else:
            idx = position - 1
            if sequence and 0 <= idx < seq_len:
                w_start = max(0, idx - 4)
                w_end = min(seq_len, idx + 5)
                h = float(seq_helix[w_start:w_end].mean())
                s = float(seq_sheet[w_start:w_end].mean())
                total = h + s + 1.0
                h_norm, s_norm, c_norm = h / total, s / total, 1.0 / total

                aa_h = _h.get(sequence[idx], 0)
                base_rsa = 0.5 - aa_h * 0.05
                rel_pos = idx / max(seq_len - 1, 1)
                if rel_pos < 0.05 or rel_pos > 0.95:
                    base_rsa += 0.2
                ctx_start = max(0, idx - 3)
                ctx_end = min(seq_len, idx + 4)
                avg_h = float(seq_hydro[ctx_start:ctx_end].mean())
                base_rsa -= avg_h * 0.02
                rsa = max(0.0, min(1.0, base_rsa))

                local_h = float(seq_hydro[ctx_start:ctx_end].mean())
                local_c = float(seq_charge[ctx_start:ctx_end].mean())
                window_str = sequence[ctx_start:ctx_end]
                gp_frac = sum(1 for a in window_str if a in _gp) / len(window_str)
            else:
                h_norm, s_norm, c_norm = 0.33, 0.33, 0.34
                rsa = 0.5
                local_h, local_c, gp_frac, rel_pos = 0, 0, 0, 0.5

            # PSSM features for wt at this position (position-dependent part)
            cons_wt_feats = _get_conservation_features(protein_id, position, wt_aa, wt_aa)
            _pos_cache[position] = (h_norm, s_norm, c_norm, rsa, local_h, local_c, gp_frac, rel_pos, cons_wt_feats)

        features[i, 13:16] = [h_norm, s_norm, c_norm]
        features[i, 16] = rsa
        features[i, 17:21] = [local_h, local_c, gp_frac, rel_pos]

        # Thermostability features (6, indices 21-26)
        features[i, 21] = 1.0 if mut_aa == 'P' and wt_aa != 'P' else 0.0
        features[i, 22] = 1.0 if wt_aa == 'P' and mut_aa != 'P' else 0.0
        features[i, 23] = 1.0 if mut_aa == 'G' and wt_aa != 'G' else 0.0
        if wt_aa in _deamid and mut_aa not in _deamid:
            features[i, 24] = -1.0
        elif mut_aa in _deamid and wt_aa not in _deamid:
            features[i, 24] = 1.0
        if mut_aa in _charged and wt_aa not in _charged:
            features[i, 25] = 1.0
        elif wt_aa in _charged and mut_aa not in _charged:
            features[i, 25] = -1.0
        if mut_aa == 'C' and wt_aa != 'C':
            features[i, 26] = 1.0
        elif wt_aa == 'C' and mut_aa != 'C':
            features[i, 26] = -1.0

        # Interaction terms (9, indices 27-35)
        burial = 1.0 - rsa
        to_proline = features[i, 21]
        features[i, 27:36] = [
            adH * burial, adV * burial, adC * burial,
            adH * adV, adC * adH,
            to_proline * burial, burial * h_norm, burial * s_norm, adH * h_norm,
        ]

        # Additional (6, indices 36-41)
        aromatic_wt = 1.0 if wt_aa in _aromatic else 0.0
        aromatic_mut = 1.0 if mut_aa in _aromatic else 0.0
        cons_wt = _bd.get(wt_aa, 4)
        cons_mut = _bd.get(mut_aa, 4)
        features[i, 36:42] = [
            aromatic_wt - aromatic_mut,
            1.0 if wt_aa in _small and mut_aa in _large else 0.0,
            1.0 if wt_aa in _large and mut_aa in _small else 0.0,
            cons_wt, cons_mut, cons_wt - cons_mut,
        ]

        # PSSM conservation features (6, indices 42-47)
        cons_feats = _get_conservation_features(protein_id, position, wt_aa, mut_aa)
        features[i, 42:48] = cons_feats

        # Condition features (indices 48-49) — only when model expects 50 features
        if _n_features >= 50:
            features[i, 48] = float(temperature)  # assay temperature (°C)
            features[i, 49] = float(ph)            # assay pH

    return features


def predict_mutations_batch_raw(mutation_tuples: list[tuple], sequence: str = None,
                               protein_id: str = None,
                               temperature: float = 25.0, ph: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
    """Batch prediction returning raw numpy arrays (ddg, probability).

    Much faster than predict_mutations_batch when you only need the arrays
    and will filter before building dicts (e.g. scanning ~6000 mutations).

    temperature: assay temperature in °C — used as feature 49 in v7+ models
    ph: assay pH — used as feature 50 in v7+ models
    """
    if _ensemble is None:
        train_model()

    if not mutation_tuples:
        return np.array([]), np.array([])

    all_features = _extract_features_batch(
        mutation_tuples, sequence=sequence, protein_id=protein_id,
        temperature=temperature, ph=ph,
    )
    all_scaled = _scaler.transform(all_features)
    all_ddg = _ensemble_predict(all_scaled)
    all_prob = 1.0 / (1.0 + np.exp(all_ddg))
    return all_ddg, all_prob


def predict_dtm_batch(mutation_tuples: list[tuple], sequence: str = None,
                      protein_id: str = None,
                      temperature: float = 65.0, ph: float = 8.0) -> np.ndarray:
    """Predict ΔTm (°C) for each mutation using the ThermoMutDB-trained regressor.

    Returns array of ΔTm predictions (°C). Positive = more thermostable.
    Falls back to NaN array if the ΔTm model was not loaded (pre-v7 models).

    temperature: target assay temperature (°C) — passed as feature 49
    ph: target assay pH — passed as feature 50
    """
    if _ensemble is None:
        train_model()

    if not mutation_tuples:
        return np.array([])

    if _dtm_ensemble is None:
        # ΔTm model not available (old model file) — return zeros
        return np.zeros(len(mutation_tuples))

    all_features = _extract_features_batch(
        mutation_tuples, sequence=sequence, protein_id=protein_id,
        temperature=temperature, ph=ph,
    )
    all_scaled = _scaler.transform(all_features)

    preds = []
    weights = _dtm_ensemble.get('weights', [0.5, 0.5])
    total_w = sum(weights)
    for (name, model), w in zip(_dtm_ensemble['models'], weights):
        preds.append(model.predict(all_scaled) * w)
    return np.sum(preds, axis=0) / total_w


def predict_mutations_batch(mutation_tuples: list[tuple], sequence: str = None, protein_id: str = None) -> list[dict]:
    """Vectorized batch prediction for many mutations at once.

    mutation_tuples: list of (wt_aa, position, mut_aa)
    Returns list of prediction dicts in same order.
    """
    all_ddg, all_prob = predict_mutations_batch_raw(mutation_tuples, sequence=sequence, protein_id=protein_id)

    if len(all_ddg) == 0:
        return []

    results = []
    for i in range(len(mutation_tuples)):
        ddg = float(all_ddg[i])
        conf = float(all_prob[i])
        results.append({
            "predicted_beneficial": ddg < 0,
            "predicted_ddg": round(ddg, 4),
            "confidence": round(conf, 4),
            "probability_beneficial": round(conf, 4),
        })
    return results


def predict_candidate_mutations(mutations: list[str], sequence: str = None) -> dict:
    """Predict all mutations in a candidate and return aggregate assessment."""
    if _ensemble is None:
        train_model()

    # Parse all mutations and batch-predict
    tuples = []
    for mut_str in mutations:
        tuples.append((mut_str[0], int(mut_str[1:-1]), mut_str[-1]))

    batch_results = predict_mutations_batch(tuples, sequence=sequence)

    predictions = []
    total_ddg = 0.0
    for mut_str, pred in zip(mutations, batch_results):
        pred["mutation"] = mut_str
        predictions.append(pred)
        total_ddg += pred["predicted_ddg"]

    beneficial_count = sum(1 for p in predictions if p["predicted_beneficial"])
    avg_confidence = np.mean([p["confidence"] for p in predictions])

    return {
        "predictions": predictions,
        "all_beneficial": beneficial_count == len(predictions),
        "beneficial_count": beneficial_count,
        "total": len(predictions),
        "total_predicted_ddg": round(total_ddg, 4),
        "average_confidence": round(float(avg_confidence), 4),
    }


def get_training_metrics() -> dict:
    """Return training metrics for display."""
    if _training_metrics is None:
        train_model()
    return _training_metrics
