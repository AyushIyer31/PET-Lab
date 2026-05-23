"""Patch ESM-1v cache: re-score 71 proteins that appear in both FireProtDB and
ThermoMutDB. The fix_esm1v_cache.py script overwrote their positions with only
ThermoMutDB positions. This script re-scores them with merged (FP+TM) positions.
"""

import os, json, re, time, pickle, requests
import numpy as np
import pandas as pd
import torch
import esm

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "backend/app/trained_models/esm1v_cache.pkl")
THERMO     = os.path.join(BASE_DIR, "thermomutdb.json")
FIREPROT   = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")

def parse_mutation_code(code):
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', str(code).strip().upper())
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None

# ── 1. Load cache and identify proteins needing re-score ────────────────────
print("Loading cache...")
cache = pickle.load(open(CACHE_PATH, 'rb'))
cache_upper = {k.upper(): v for k, v in cache.items()}

fp = pd.read_csv(FIREPROT)
with open(THERMO) as f:
    thermo_data = json.load(f)

# Find FireProtDB proteins with missing positions
missing_fp_positions = {}  # {pid_upper: set(missing_positions)}
for _, row in fp.iterrows():
    if pd.isna(row.get('ddG')):
        continue
    pid = str(row['pdb_id']).split('|')[0].strip().upper()
    try:
        pos = int(row['position'])
    except:
        continue
    if pos not in cache_upper.get(pid, {}):
        if pid not in missing_fp_positions:
            missing_fp_positions[pid] = set()
        missing_fp_positions[pid].add(pos)

print(f"  Proteins with missing FireProtDB positions: {len(missing_fp_positions)}")
print(f"  Total missing positions: {sum(len(v) for v in missing_fp_positions.values())}")

# ── 2. Build merged position sets for affected proteins ─────────────────────
# Add existing ThermoMutDB positions too (keep them)
proteins_to_rescore = {}  # {pid: set(all_positions)}
for pid, fp_pos in missing_fp_positions.items():
    existing_pos = set(cache_upper.get(pid, {}).keys())
    proteins_to_rescore[pid] = existing_pos | fp_pos

print(f"\nProteins to re-score (merged positions): {len(proteins_to_rescore)}")

# ── 3. Get UniProt IDs for each protein ──────────────────────────────────────
# Check pdb_to_uniprot mapping first
pdb_uni_path = os.path.join(BASE_DIR, "backend/app/trained_models/pdb_to_uniprot.pkl")
pdb_to_uniprot = {}
if os.path.exists(pdb_uni_path):
    pdb_to_uniprot = {k.upper(): v for k, v in pickle.load(open(pdb_uni_path, 'rb')).items()}

# Also collect from ThermoMutDB
tm_uniprot = {}
for entry in thermo_data:
    pdb = str(entry.get('PDB_wild', '') or '').strip().upper()
    uniprot = str(entry.get('uniprot') or entry.get('swissprot') or '').strip()
    if pdb and uniprot and uniprot != 'nan':
        tm_uniprot[pdb] = uniprot

# ── 4. Fetch sequences from UniProt ─────────────────────────────────────────
print("\nFetching sequences from UniProt for affected proteins...")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "protein-stability-patch/1.0"})

proteins_with_seq = {}
for pid in proteins_to_rescore:
    uniprot = pdb_to_uniprot.get(pid) or tm_uniprot.get(pid)
    if not uniprot:
        print(f"  WARN: no UniProt ID for {pid}")
        continue
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        resp = SESSION.get(url, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            seq = ''.join(lines[1:])
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            seq_clean = ''.join(c for c in seq.upper() if c in valid_aa)
            if len(seq_clean) >= 10:
                proteins_with_seq[pid] = seq_clean
        time.sleep(0.05)
    except Exception as e:
        print(f"  WARN: fetch failed for {pid}: {e}")

print(f"  Fetched sequences: {len(proteins_with_seq)}/{len(proteins_to_rescore)}")

# ── 5. Load ESM-1v model ────────────────────────────────────────────────────
print("\nLoading ESM-1v model...")
t0 = time.time()
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
model.eval()
batch_converter = alphabet.get_batch_converter()
print(f"  Loaded in {time.time()-t0:.1f}s")
MAX_LEN = 1022

# ── 6. Re-score affected proteins ───────────────────────────────────────────
print(f"\nRe-scoring {len(proteins_with_seq)} proteins with merged positions...")
for i, (pid, seq) in enumerate(sorted(proteins_with_seq.items())):
    positions = proteins_to_rescore[pid]
    if len(seq) > MAX_LEN:
        print(f"  [{i+1}] {pid}: SKIP (len={len(seq)})")
        continue
    try:
        t1 = time.time()
        _, _, tokens = batch_converter([(pid, seq)])
        with torch.no_grad():
            results = model(tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"][0]
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        pos_scores = {}
        for pos in positions:
            if 1 <= pos < log_probs.shape[0] - 1:
                pos_scores[pos] = log_probs[pos]

        # Preserve existing TM positions that aren't in our position set
        # (shouldn't happen, but be safe)
        cache[pid] = pos_scores
        cache_upper[pid] = pos_scores
        print(f"  [{i+1}/{len(proteins_with_seq)}] {pid}: {len(seq)}aa, "
              f"{len(pos_scores)}/{len(positions)} positions, {time.time()-t1:.1f}s")
    except Exception as e:
        print(f"  FAILED {pid}: {e}")

# ── 7. Save ──────────────────────────────────────────────────────────────────
with open(CACHE_PATH, 'wb') as f:
    pickle.dump(cache, f)
print(f"\nSaved {len(cache)} proteins to cache.")

# ── 8. Verify ────────────────────────────────────────────────────────────────
print("\nFinal coverage check:")
cache_upper2 = {k.upper(): v for k, v in cache.items()}

fp_cov = 0
fp_tot = 0
for _, row in fp.iterrows():
    if pd.isna(row.get('ddG')):
        continue
    pid = str(row['pdb_id']).split('|')[0].strip().upper()
    try:
        pos = int(row['position'])
    except:
        continue
    fp_tot += 1
    if cache_upper2.get(pid, {}).get(pos) is not None:
        fp_cov += 1
print(f"  FireProtDB: {fp_cov}/{fp_tot} ({100*fp_cov/max(fp_tot,1):.1f}%)")

tm_cov = 0
tm_tot = 0
for entry in thermo_data:
    mc = entry.get('mutation_code', '')
    wt, pos, mut = parse_mutation_code(str(mc))
    if wt is None or entry.get('ddg') is None:
        continue
    pdb = str(entry.get('PDB_wild', '') or '').strip().upper()
    if not pdb or pdb == 'NAN':
        continue
    tm_tot += 1
    if cache_upper2.get(pdb, {}).get(pos) is not None:
        tm_cov += 1
print(f"  ThermoMutDB: {tm_cov}/{tm_tot} ({100*tm_cov/max(tm_tot,1):.1f}%)")
