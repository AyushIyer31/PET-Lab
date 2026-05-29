"""Compute ESM-1v ensemble scores (all 5 models averaged) for all training proteins.

ESM-1v was released as an ensemble of 5 models trained on different random seeds.
Averaging across all 5 gives more robust evolutionary fitness scores than any
single model alone — the standard practice in the literature (Meier et al. 2021).

Output: backend/app/trained_models/esm1v_ensemble_cache.pkl
Format: {protein_id: {position: np.array(33,)}}  — same as esm1v_cache.pkl
        but values are the mean log-probs across all 5 ESM-1v models.

Runtime: ~5× longer than single model (~15-25 hours on CPU).
"""

import os, json, time, pickle
import numpy as np
import pandas as pd
import torch
import esm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "backend/app/trained_models/esm1v_ensemble_cache.pkl")
FIREPROT = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
THERMO   = os.path.join(BASE_DIR, "thermomutdb.json")

ESM1V_MODELS = [
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
]

# ── 1. Collect proteins and positions from training data ──────────────────────
print("Loading training data...")
fp = pd.read_csv(FIREPROT)
with open(THERMO) as f:
    thermo = json.load(f)

needed = {}
for _, row in fp.iterrows():
    if pd.isna(row.get('ddG')) or pd.isna(row.get('pdb_id')):
        continue
    pid = str(row['pdb_id']).split('|')[0].strip()
    seq = str(row.get('sequence', '') or '').strip()
    pos = row.get('position')
    if not seq or seq == 'nan' or pd.isna(pos):
        continue
    pos = int(pos)
    if pid not in needed:
        needed[pid] = {'seq': seq, 'positions': set()}
    needed[pid]['positions'].add(pos)

# Add existing single-model cache proteins (they already have sequences resolved)
single_cache_path = os.path.join(BASE_DIR, "backend/app/trained_models/esm1v_cache.pkl")
if os.path.exists(single_cache_path):
    with open(single_cache_path, 'rb') as f:
        single_cache = pickle.load(f)
    print(f"  Single-model cache has {len(single_cache)} proteins — will reuse protein list")

print(f"  Proteins with sequences: {len(needed)}")
total_pos = sum(len(v['positions']) for v in needed.values())
print(f"  Total (protein, position) pairs: {total_pos}")

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

# ── 2. Run each model and accumulate log-probs ────────────────────────────────
# accum[pid][pos] = list of np.array(33,) — one per model
accum = {pid: {pos: [] for pos in info['positions']} for pid, info in needed.items()}

for model_idx, model_name in enumerate(ESM1V_MODELS):
    print(f"\n{'='*60}")
    print(f"Model {model_idx+1}/5: {model_name}")
    print(f"{'='*60}")
    t_model = time.time()

    loader_fn = getattr(esm.pretrained, model_name)
    model, alphabet = loader_fn()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"  Loaded on {device}")

    for i, (pid, info) in enumerate(sorted(needed.items())):
        seq = info['seq']
        positions = info['positions']
        seq_clean = ''.join(c for c in seq.upper() if c in VALID_AA)
        if len(seq_clean) < 10:
            continue

        try:
            data = [(pid, seq_clean)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(tokens, repr_layers=[], return_contacts=False)
                logits = results["logits"][0]
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

            for pos in positions:
                token_idx = pos
                if token_idx < log_probs.shape[0]:
                    accum[pid][pos].append(log_probs[token_idx])

        except Exception as e:
            print(f"  FAILED {pid} (model {model_idx+1}): {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_model
            rate = (i + 1) / elapsed
            eta = (len(needed) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(needed)}] {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    # Free model memory before loading next
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Model {model_idx+1} done in {time.time()-t_model:.0f}s")

    # Save intermediate checkpoint after each model
    ckpt_path = OUT_PATH.replace('.pkl', f'_ckpt{model_idx+1}.pkl')
    with open(ckpt_path, 'wb') as f:
        pickle.dump(accum, f)
    print(f"  Checkpoint saved: {ckpt_path}")

# ── 3. Average across models and save final cache ─────────────────────────────
print("\nAveraging log-probs across 5 models...")
ensemble_cache = {}
for pid, pos_dict in accum.items():
    ensemble_cache[pid] = {}
    for pos, lp_list in pos_dict.items():
        if lp_list:
            ensemble_cache[pid][pos] = np.mean(lp_list, axis=0).astype(np.float32)

with open(OUT_PATH, 'wb') as f:
    pickle.dump(ensemble_cache, f)

n_pos = sum(len(v) for v in ensemble_cache.values())
print(f"\nDone. Saved ESM-1v ensemble cache:")
print(f"  Proteins: {len(ensemble_cache)}")
print(f"  Positions: {n_pos}")
print(f"  Output: {OUT_PATH}")
