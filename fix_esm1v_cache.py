"""Fix ESM-1v cache: recompute ThermoMutDB proteins with CORRECT positions.

Bug: compute_esm1v.py used r['pos'] which is always 0 in thermomutdb.json.
     Training uses parse_mutation_code(r['mutation_code']) which gives real position.
Fix: re-run ESM-1v for all ThermoMutDB proteins using mutation_code positions.
     Keep FireProtDB entries intact (they used row['position'] correctly).

Expected result: ESM-1v coverage 42.4% → ~98% (all 17791 training mutations)
Runtime: ~15-25 min on CPU (347 ThermoMutDB proteins × ~3s each)
"""

import os, json, re, time, pickle, requests
import numpy as np
import pandas as pd
import torch
import esm

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "backend/app/trained_models/esm1v_cache.pkl")
THERMO     = os.path.join(BASE_DIR, "thermomutdb.json")
FIREPROT   = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")

def parse_mutation_code(code):
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', str(code).strip().upper())
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None

# ── 1. Identify FireProtDB proteins (keep these in cache as-is) ─────────────
print("Loading FireProtDB to identify proteins to keep...")
fp = pd.read_csv(FIREPROT)
fp_pids = set()
for _, row in fp.iterrows():
    if pd.isna(row.get('ddG')):
        continue
    raw = str(row['pdb_id'])
    pid = raw.split('|')[0].strip().upper()
    fp_pids.add(pid)
print(f"  FireProtDB proteins (keep): {len(fp_pids)}")

# ── 2. Load existing cache, keep only FireProtDB entries ────────────────────
print("Loading existing ESM-1v cache...")
existing_cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        existing_cache = pickle.load(f)
print(f"  Existing cache: {len(existing_cache)} proteins")

# Keep only FireProtDB entries
new_cache = {pid: existing_cache[pid] for pid in existing_cache if pid.upper() in fp_pids}
print(f"  Keeping {len(new_cache)} FireProtDB proteins from existing cache")

# ── 3. Build ThermoMutDB protein → {positions, uniprot} mapping ─────────────
print("\nParsing ThermoMutDB for correct positions (from mutation_code)...")
with open(THERMO) as f:
    thermo = json.load(f)

thermo_proteins = {}  # {pdb_upper: {uniprot, positions}}
for entry in thermo:
    mc = entry.get('mutation_code', '')
    wt, pos, mut = parse_mutation_code(str(mc))
    if wt is None:
        continue
    if entry.get('ddg') is None:
        continue
    pdb = str(entry.get('PDB_wild', '') or '').strip().upper()
    if not pdb or pdb == 'NAN' or len(pdb) < 3:
        continue
    uniprot = str(entry.get('uniprot') or entry.get('swissprot') or '').strip()
    if pdb not in thermo_proteins:
        thermo_proteins[pdb] = {'uniprot': uniprot, 'positions': set()}
    thermo_proteins[pdb]['positions'].add(pos)

print(f"  ThermoMutDB unique proteins: {len(thermo_proteins)}")
print(f"  Total unique (protein, position) pairs: {sum(len(v['positions']) for v in thermo_proteins.values())}")

# ── 4. Fetch sequences from UniProt ─────────────────────────────────────────
print("\nFetching sequences from UniProt...")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "protein-stability-model-fix/1.0"})

proteins_with_seq = {}  # {pdb: {seq, positions}}
fetched = 0
failed = 0

for pdb, info in thermo_proteins.items():
    uniprot = info['uniprot']
    if not uniprot or uniprot == 'nan':
        failed += 1
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
                proteins_with_seq[pdb] = {'seq': seq_clean, 'positions': info['positions']}
                fetched += 1
            else:
                failed += 1
        else:
            failed += 1
        time.sleep(0.05)
    except Exception as e:
        failed += 1
        print(f"  WARN: failed to fetch {pdb} (UniProt {uniprot}): {e}")

print(f"  Fetched: {fetched}, Failed: {failed}")
print(f"  Proteins ready to score: {len(proteins_with_seq)}")

# ── 5. Load ESM-1v model ────────────────────────────────────────────────────
print("\nLoading ESM-1v model...")
t0 = time.time()
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
model.eval()
batch_converter = alphabet.get_batch_converter()
print(f"  Loaded in {time.time()-t0:.1f}s")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model = model.to(device)
print(f"  Device: {device}")

# ── 6. Score each protein ───────────────────────────────────────────────────
print(f"\nScoring {len(proteins_with_seq)} ThermoMutDB proteins...")
failed_score = []
MAX_LEN = 1022  # ESM-1v max (excl. BOS/EOS tokens)

for i, (pdb, info) in enumerate(sorted(proteins_with_seq.items())):
    seq = info['seq']
    positions = info['positions']

    # ESM-1v 1024-token limit (seq + BOS + EOS = seq_len + 2)
    if len(seq) > MAX_LEN:
        failed_score.append((pdb, f'seq too long: {len(seq)}'))
        if i % 50 == 0:
            print(f"  [{i+1}/{len(proteins_with_seq)}] {pdb}: SKIP (len={len(seq)})")
        continue

    try:
        t1 = time.time()
        data = [(pdb, seq)]
        _, _, tokens = batch_converter(data)
        if device == 'cuda':
            tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"][0]  # (seq_len+2, vocab)

        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        # Store at each needed position (1-indexed → token index = position)
        pos_scores = {}
        for pos in positions:
            token_idx = pos  # token[0]=BOS, token[1]=res1, ..., token[pos]=res_pos
            if 1 <= token_idx < log_probs.shape[0] - 1:  # skip BOS/EOS
                pos_scores[pos] = log_probs[token_idx]

        new_cache[pdb] = pos_scores
        elapsed = time.time() - t1

        if i % 25 == 0 or i < 5:
            print(f"  [{i+1}/{len(proteins_with_seq)}] {pdb}: {len(seq)}aa, "
                  f"{len(pos_scores)}/{len(positions)} positions, {elapsed:.1f}s")

        # Checkpoint every 50 proteins
        if (i + 1) % 50 == 0:
            with open(CACHE_PATH, 'wb') as f:
                pickle.dump(new_cache, f)
            total_pos = sum(len(v) for v in new_cache.values())
            print(f"  Checkpoint: {len(new_cache)} proteins, {total_pos} positions")

    except Exception as e:
        failed_score.append((pdb, str(e)[:100]))
        print(f"  FAILED {pdb}: {e}")

# ── 7. Save final cache ─────────────────────────────────────────────────────
with open(CACHE_PATH, 'wb') as f:
    pickle.dump(new_cache, f)

total_pos = sum(len(v) for v in new_cache.values())
print(f"\nDone.")
print(f"  Total proteins in cache: {len(new_cache)}")
print(f"  Total (protein, position) pairs: {total_pos}")
print(f"  Failed scoring: {len(failed_score)}")
if failed_score[:5]:
    print(f"  Failures: {failed_score[:5]}")
print(f"  Saved to: {CACHE_PATH}")

# ── 8. Quick coverage check ──────────────────────────────────────────────────
print("\nVerifying coverage against training data...")
cache_upper = {k.upper(): v for k, v in new_cache.items()}

# ThermoMutDB check
tm_covered = 0
tm_total = 0
for entry in thermo:
    mc = entry.get('mutation_code', '')
    wt, pos, mut = parse_mutation_code(str(mc))
    if wt is None or entry.get('ddg') is None:
        continue
    pdb = str(entry.get('PDB_wild', '') or '').strip().upper()
    if not pdb or pdb == 'NAN':
        continue
    tm_total += 1
    if cache_upper.get(pdb, {}).get(pos) is not None:
        tm_covered += 1
print(f"  ThermoMutDB coverage: {tm_covered}/{tm_total} ({100*tm_covered/max(tm_total,1):.1f}%)")

# FireProtDB check
fp_covered = 0
fp_total = 0
for _, row in fp.iterrows():
    if pd.isna(row.get('ddG')):
        continue
    pid = str(row['pdb_id']).split('|')[0].strip().upper()
    try:
        pos = int(row['position'])
    except:
        continue
    fp_total += 1
    if cache_upper.get(pid, {}).get(pos) is not None:
        fp_covered += 1
print(f"  FireProtDB coverage: {fp_covered}/{fp_total} ({100*fp_covered/max(fp_total,1):.1f}%)")
