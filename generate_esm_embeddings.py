"""Generate ESM-2 per-residue embeddings for all training proteins.

Fetches sequences for ThermoMutDB proteins from RCSB PDB (they have no sequences
in the JSON), combines with FireProtDB sequences (already in CSV), then runs ESM-2
(esm2_t30_150M_UR50D, 640-dim) on all proteins.

Output: esm_embeddings_cache.pkl
  {protein_id: numpy array of shape [seq_len, 640]}

This gives ~100% coverage across FireProtDB + ThermoMutDB, eliminating the
zero-padding problem that hurt v14.
"""

import os
import json
import time
import pickle
import urllib.request
import urllib.error
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREPROT_PATH = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
THERMOMUTDB_PATH = os.path.join(BASE_DIR, "thermomutdb.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "backend/app/trained_models/esm_embeddings_cache.pkl")

MAX_SEQ_LEN = 1024  # ESM-2 context limit; truncate longer sequences


def fetch_sequence_rcsb(pdb_id, max_retries=3):
    """Fetch the first chain's amino acid sequence from RCSB PDB."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/download"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                text = r.read().decode()
            # Parse FASTA — take the first sequence only
            lines = text.strip().split('\n')
            seq = ''
            for line in lines:
                if line.startswith('>'):
                    if seq:
                        break  # stop after first sequence
                else:
                    seq += line.strip()
            # Keep only standard amino acids
            valid = set('ACDEFGHIKLMNPQRSTVWY')
            seq = ''.join(aa for aa in seq.upper() if aa in valid)
            return seq if len(seq) >= 10 else None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def load_all_sequences():
    """Collect sequences for all training proteins."""
    sequences = {}  # protein_id -> sequence

    # --- FireProtDB (all 125 proteins have sequences in CSV) ---
    print("Loading FireProtDB sequences...")
    df = pd.read_csv(FIREPROT_PATH)
    fp_proteins = df[df['sequence'].notna() & (df['sequence'] != '')][['pdb_id', 'sequence']].copy()
    fp_proteins['pdb_id'] = fp_proteins['pdb_id'].astype(str).str.split('|').str[0].str.strip().str.upper()
    fp_proteins = fp_proteins.drop_duplicates('pdb_id')
    for _, row in fp_proteins.iterrows():
        pid = row['pdb_id']
        seq = str(row['sequence']).strip()
        valid = set('ACDEFGHIKLMNPQRSTVWY')
        seq = ''.join(aa for aa in seq.upper() if aa in valid)
        if len(seq) >= 10:
            sequences[pid] = seq
    print(f"  FireProtDB: {len(sequences)} proteins with sequences")

    # --- ThermoMutDB (no sequences in JSON — fetch from RCSB) ---
    print("Loading ThermoMutDB PDB IDs...")
    with open(THERMOMUTDB_PATH) as f:
        thermo_data = json.load(f)
    thermo_pdb_ids = set()
    for entry in thermo_data:
        pid = str(entry.get('PDB_wild', '')).strip().upper()[:4]
        if len(pid) == 4:
            thermo_pdb_ids.add(pid)
    # Remove ones already from FireProtDB
    to_fetch = [pid for pid in thermo_pdb_ids if pid not in sequences]
    print(f"  Need to fetch {len(to_fetch)} sequences from RCSB PDB...")

    fetched = 0
    failed = 0
    for i, pid in enumerate(to_fetch):
        seq = fetch_sequence_rcsb(pid)
        if seq:
            sequences[pid] = seq
            fetched += 1
        else:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(to_fetch)}] fetched={fetched} failed={failed}")
        time.sleep(0.05)  # polite rate limit

    print(f"  ThermoMutDB: fetched {fetched} sequences, failed {failed}")
    print(f"Total proteins with sequences: {len(sequences)}")
    return sequences


def generate_embeddings(sequences):
    """Run ESM-2 (150M) on all sequences, return dict of {protein_id: embeddings}."""
    import torch
    import esm

    print("\nLoading ESM-2 model (esm2_t30_150M_UR50D, 640-dim)...")
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Sort by sequence length for efficient batching
    sorted_proteins = sorted(sequences.items(), key=lambda x: len(x[1]))

    embeddings = {}
    total = len(sorted_proteins)
    print(f"Generating embeddings for {total} proteins...")

    with torch.no_grad():
        for i, (pid, seq) in enumerate(sorted_proteins):
            # Truncate to ESM-2 context limit
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]

            data = [(pid, seq)]
            _, _, tokens = batch_converter(data)

            # Get per-residue representations from last layer (layer 30)
            results = model(tokens, repr_layers=[30], return_contacts=False)
            reps = results["representations"][30]  # [1, seq_len+2, 640]
            # Remove BOS/EOS tokens
            reps = reps[0, 1:len(seq)+1, :].cpu().numpy()  # [seq_len, 640]

            embeddings[pid] = reps

            if (i + 1) % 25 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] {pid} seq_len={len(seq)} emb_shape={reps.shape}")

    return embeddings


def main():
    print("=" * 60)
    print("ESM-2 EMBEDDING GENERATION")
    print("Model: esm2_t30_150M_UR50D (640-dim per residue)")
    print("=" * 60)

    # Check if output already exists
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'rb') as f:
            existing = pickle.load(f)
        print(f"Existing cache found: {len(existing)} proteins. Continuing...")
        return existing

    # Step 1: Gather sequences
    sequences = load_all_sequences()

    # Step 2: Generate ESM-2 embeddings
    t0 = time.time()
    embeddings = generate_embeddings(sequences)
    elapsed = time.time() - t0
    print(f"\nEmbedding generation took {elapsed/60:.1f} minutes")

    # Step 3: Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    total_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"Saved {len(embeddings)} protein embeddings → {OUTPUT_PATH} ({total_mb:.1f} MB)")

    return embeddings


if __name__ == '__main__':
    main()
