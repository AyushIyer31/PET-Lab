"""Compute ESM-1v zero-shot mutation scores for all proteins in training set.

ESM-1v (650M params) is trained specifically for variant effect prediction.
For each mutation (WT→MUT at position P), it computes an evolutionary fitness
score: log P(MUT | context) - log P(WT | context). Higher = more tolerated.

Strategy: unmasked forward pass (one pass per protein) for speed.
Runtime: ~3-5 hours on CPU for ~529 proteins.

Output: backend/app/trained_models/esm1v_cache.pkl
Format: {protein_id: {position: np.array(33 log-probs, ESM alphabet)}}
        Same format as esm_loglik_cache.pkl (ESM-2 version).
"""

import os, json, time, pickle, requests
import numpy as np
import pandas as pd
import torch
import esm

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_PATH   = os.path.join(BASE_DIR, "backend/app/trained_models/esm1v_cache.pkl")
FIREPROT   = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
THERMO     = os.path.join(BASE_DIR, "thermomutdb.json")

# ── 1. Load training data to find all (protein_id, position, wt, mut) tuples ──
print("Loading training data...")
fp = pd.read_csv(FIREPROT)
with open(THERMO) as f:
    thermo = json.load(f)

# Build set of needed (protein_id, position) pairs and collect sequences
needed = {}   # {protein_id: {'seq': str, 'positions': set()}}

# FireProtDB — has sequence column, pdb_id is protein_id
for _, row in fp.iterrows():
    if pd.isna(row.get('ddG')) or pd.isna(row.get('pdb_id')):
        continue
    raw_pid = str(row['pdb_id'])
    pid = raw_pid.split('|')[0].strip()
    seq = str(row.get('sequence', '') or '').strip()
    pos = int(row['position'])
    if not seq or seq == 'nan':
        continue
    if pid not in needed:
        needed[pid] = {'seq': seq, 'positions': set(), 'source': 'fireprot'}
    needed[pid]['positions'].add(pos)

print(f"  FireProtDB: {len(needed)} proteins")

# ThermoMutDB — no sequence, need UniProt fetch
thermo_pids = {}
for r in thermo:
    if r.get('ddg') is None:
        continue
    pid = str(r.get('PDB_wild', '') or '').strip().upper()
    uniprot = str(r.get('uniprot') or r.get('swissprot') or '').strip()
    if not pid or pid == 'NAN' or len(pid) < 3:
        continue
    try:
        pos = int(r['pos'])
    except (KeyError, ValueError, TypeError):
        continue
    if pid not in thermo_pids:
        thermo_pids[pid] = {'uniprot': uniprot, 'positions': set()}
    thermo_pids[pid]['positions'].add(pos)

# Fetch missing sequences from UniProt
print(f"  ThermoMutDB: {len(thermo_pids)} proteins, fetching sequences from UniProt...")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "protein-stability-model/1.0"})
fetched = 0
failed_fetch = 0

for pid, info in thermo_pids.items():
    if pid in needed:
        # already have sequence from FireProtDB
        needed[pid]['positions'].update(info['positions'])
        continue
    uniprot = info['uniprot']
    if not uniprot or uniprot == 'nan':
        failed_fetch += 1
        continue
    # Try UniProt FASTA API
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        resp = SESSION.get(url, timeout=10)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            seq = ''.join(lines[1:])  # skip header
            if seq:
                needed[pid] = {'seq': seq, 'positions': info['positions'], 'source': 'thermo_uniprot'}
                fetched += 1
        else:
            failed_fetch += 1
        time.sleep(0.05)
    except Exception as e:
        failed_fetch += 1

print(f"  UniProt fetched: {fetched}, failed: {failed_fetch}")
print(f"  Total proteins with sequences: {len(needed)}")
total_positions = sum(len(v['positions']) for v in needed.values())
print(f"  Total unique (protein, position) pairs: {total_positions}")

# ── 2. Load ESM-1v model ──
print("\nLoading ESM-1v model (650M params)...")
t0 = time.time()
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
model.eval()
batch_converter = alphabet.get_batch_converter()
print(f"  Model loaded in {time.time()-t0:.1f}s")
print(f"  Alphabet size: {len(alphabet)}")
print(f"  Running on: CPU (no GPU detected)" if not torch.cuda.is_available() else "  Running on: GPU")

# ── 3. Compute ESM-1v scores for each protein ──
print(f"\nScoring {len(needed)} proteins...")
esm1v_cache = {}
failed_score = []

# Try to load partial cache if interrupted
if os.path.exists(OUT_PATH):
    with open(OUT_PATH, 'rb') as f:
        esm1v_cache = pickle.load(f)
    print(f"  Resuming from partial cache: {len(esm1v_cache)} proteins already done")

for i, (pid, info) in enumerate(sorted(needed.items())):
    if pid in esm1v_cache:
        continue  # already computed

    seq = info['seq']
    positions = info['positions']

    # Validate sequence
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    seq_clean = ''.join(c for c in seq.upper() if c in valid_aa)
    if len(seq_clean) < 10:
        failed_score.append(pid)
        continue

    try:
        t1 = time.time()
        data = [(pid, seq_clean)]
        _, _, tokens = batch_converter(data)

        with torch.no_grad():
            results = model(tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"][0]  # shape: (seq_len+2, vocab_size)

        # log_softmax over vocab at each position
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        # Store for each needed position (1-indexed → token index = position)
        pos_scores = {}
        for pos in positions:
            token_idx = pos  # tokens[0]=<cls>, tokens[1]=residue1, ..., tokens[pos]=residue_pos
            if token_idx >= log_probs.shape[0]:
                continue
            pos_scores[pos] = log_probs[token_idx]  # array of size vocab

        esm1v_cache[pid] = pos_scores
        elapsed = time.time() - t1

        if i % 10 == 0 or i < 5:
            print(f"  [{i+1}/{len(needed)}] {pid}: {len(seq_clean)} aa, "
                  f"{len(pos_scores)} positions scored, {elapsed:.1f}s")

        # Save checkpoint every 25 proteins
        if (i + 1) % 25 == 0:
            with open(OUT_PATH, 'wb') as f:
                pickle.dump(esm1v_cache, f)
            print(f"  Checkpoint saved ({len(esm1v_cache)} proteins)")

    except Exception as e:
        failed_score.append((pid, str(e)[:80]))
        print(f"  FAILED {pid}: {e}")

# ── 4. Save final cache ──
with open(OUT_PATH, 'wb') as f:
    pickle.dump(esm1v_cache, f)

print(f"\nDone.")
print(f"  Cached: {len(esm1v_cache)} proteins")
print(f"  Failed scoring: {len(failed_score)}")
if failed_score[:5]:
    print(f"  Sample failures: {failed_score[:5]}")
print(f"  Saved to: {OUT_PATH}")

# Quick sanity check
sample_pid = next(iter(esm1v_cache))
sample_pos = next(iter(esm1v_cache[sample_pid]))
sample_arr = esm1v_cache[sample_pid][sample_pos]
print(f"\nSanity check — {sample_pid} pos {sample_pos}: array shape={sample_arr.shape}, "
      f"min={sample_arr.min():.2f}, max={sample_arr.max():.2f}")

# Show alphabet mapping for reference
aa_list = 'ACDEFGHIKLMNPQRSTVWY'
print(f"Alphabet indices for standard AA:")
for aa in aa_list[:5]:
    idx = alphabet.get_idx(aa)
    print(f"  {aa} → idx {idx}")
