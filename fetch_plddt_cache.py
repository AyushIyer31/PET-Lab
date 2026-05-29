"""Fetch per-residue pLDDT scores from AlphaFold DB for all proteins in training set."""
import json, pickle, time, requests, pandas as pd
from collections import defaultdict

CACHE_PATH = "backend/app/trained_models/plddt_cache.pkl"

# ── Collect all UniProt IDs ──
fp = pd.read_csv("fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
with open("thermomutdb.json") as f:
    thermo = json.load(f)

uniprot_ids = set()
for col in ['uniprot_id', 'uniprot']:
    if col in fp.columns:
        uniprot_ids.update(str(v) for v in fp[col].dropna().unique())

for r in thermo:
    uid = r.get('uniprot') or r.get('swissprot') or r.get('uniprot_id')
    if uid and str(uid) != 'nan':
        for u in str(uid).split('|'):
            u = u.strip()
            if 4 <= len(u) <= 10:
                uniprot_ids.add(u)

uniprot_ids = {u for u in uniprot_ids if u and u != 'nan'}
print(f"Fetching pLDDT for {len(uniprot_ids)} UniProt IDs...")

# ── Fetch pLDDT from AlphaFold API ──
plddt_cache = {}   # {uniprot_id: [plddt_float, ...]}  (1-indexed by position)
failed = []

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "protein-stability-model/1.0"})

for i, uid in enumerate(sorted(uniprot_ids)):
    try:
        meta_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
        meta_resp = SESSION.get(meta_url, timeout=15)
        if meta_resp.status_code != 200:
            failed.append((uid, meta_resp.status_code))
            continue

        meta = meta_resp.json()
        if not meta:
            failed.append((uid, "empty"))
            continue

        plddt_url = meta[0].get("plddtDocUrl") or meta[0].get("paeDocUrl")
        if not plddt_url:
            # Try direct confidence file URL pattern
            plddt_url = meta[0].get("cifUrl", "").replace("-model_v", "-confidence_v").replace(".cif", ".json")

        conf_resp = SESSION.get(plddt_url, timeout=15)
        if conf_resp.status_code != 200:
            failed.append((uid, f"plddt {conf_resp.status_code}"))
            continue

        conf_data = conf_resp.json()
        scores = conf_data.get("confidenceScore") or conf_data.get("plddt")
        if scores:
            plddt_cache[uid] = scores   # list of floats, 0-indexed (pos 0 = residue 1)
            if i % 25 == 0:
                print(f"  [{i+1}/{len(uniprot_ids)}] {uid}: {len(scores)} residues, "
                      f"mean pLDDT={sum(scores)/len(scores):.1f}")
        else:
            failed.append((uid, "no scores"))

        time.sleep(0.1)   # polite rate limiting

    except Exception as e:
        failed.append((uid, str(e)[:60]))
        time.sleep(0.5)

print(f"\nDone. Cached {len(plddt_cache)}/{len(uniprot_ids)} proteins.")
print(f"Failed: {len(failed)}")
if failed[:10]:
    print("Sample failures:", failed[:10])

with open(CACHE_PATH, "wb") as f:
    pickle.dump(plddt_cache, f)
print(f"Saved to {CACHE_PATH}")
