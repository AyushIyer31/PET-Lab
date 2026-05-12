"""Compute ESM-2 per-position log-likelihood for all amino acids.

For each mutation site in the training data, computes the marginal
log-likelihood ratio:

    ΔLL = log P(mut_aa | context) − log P(wt_aa | context)

where P(aa | context) is the masked language model probability of amino acid aa
at position i given the rest of the protein sequence, computed with ESM-2
(esm2_t30_150M_UR50D, 150M parameters).

This is the same zero-shot mutation effect score used by ESM-1v (Meier et al. 2021),
which achieves 0.44 Spearman correlation with experimental ΔΔG on ProtaBank
without any fine-tuning. Adding it as a feature in a supervised model should
significantly increase accuracy.

Method
------
For each unique (protein, mutation_position) pair:
  1. Replace position i with the mask token <mask>
  2. Run ESM-2 forward pass
  3. Extract log-softmax over 20 standard amino acids at position i
  4. Store as np.array(20,) of log-probs (AA order = ACDEFGHIKLMNPQRSTVWY)

Output
------
backend/app/trained_models/esm_loglik_cache.pkl
  { protein_id (str) : { resnum (int) : np.array(20,) log-probs } }

References
----------
Meier J et al. (2021) Language models enable zero-shot prediction of the effects
  of mutations on protein function. NeurIPS 2021.
Lin Z et al. (2023) Evolutionary-scale prediction of atomic level protein
  structure with a language model. Science 379, 1123-1130.
"""

import os
import time
import pickle
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREPROT_PATH = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
THERMOMUTDB_PATH = os.path.join(BASE_DIR, "thermomutdb.json")
ESM_EMBED_CACHE = os.path.join(BASE_DIR, "backend/app/trained_models/esm_embeddings_cache.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "backend/app/trained_models/esm_loglik_cache.pkl")

# Standard amino acid order (alphabetical, matches ESM-1v convention)
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
MAX_SEQ_LEN = 1024


def load_mutation_positions():
    """Collect all (protein_id, resnum) pairs from the training data."""
    positions = {}   # protein_id -> set of resnums

    # FireProtDB
    df = pd.read_csv(FIREPROT_PATH)
    df['pdb_id'] = df['pdb_id'].astype(str).str.split('|').str[0].str.strip().str.upper()
    for _, row in df.iterrows():
        pid = row['pdb_id']
        try:
            pos = int(row['position'])
        except (ValueError, TypeError):
            continue
        positions.setdefault(pid, set()).add(pos)

    # ThermoMutDB — parse position from 'mutation_code' like "E49M"
    # NOTE: the 'pos' field is NOT the residue position — it's an internal
    # zero-based index with different semantics. Always use mutation_code.
    with open(THERMOMUTDB_PATH) as f:
        thermo = json.load(f)
    for entry in thermo:
        pid = str(entry.get('PDB_wild', '')).strip().upper()[:4]
        if len(pid) != 4:
            continue
        # Parse position from mutation_code like "E49M" or "A123G"
        mut = str(entry.get('mutation_code', '')).strip()
        # Skip multi-site mutations (contain comma)
        if ',' in mut:
            continue
        try:
            pos = int(mut[1:-1])
            if pos >= 1:
                positions.setdefault(pid, set()).add(pos)
        except (ValueError, IndexError):
            continue

    return positions


def load_sequences():
    """Load ESM embeddings cache to get list of proteins, then fetch sequences."""
    with open(ESM_EMBED_CACHE, 'rb') as f:
        emb_cache = pickle.load(f)
    protein_ids = list(emb_cache.keys())
    print(f"Found {len(protein_ids)} proteins in ESM embeddings cache")

    # Try to load sequences from FireProtDB first
    seqs = {}
    df = pd.read_csv(FIREPROT_PATH)
    df['pdb_id'] = df['pdb_id'].astype(str).str.split('|').str[0].str.strip().str.upper()
    for _, row in df.iterrows():
        pid = row['pdb_id']
        seq = str(row.get('sequence', '')).strip()
        valid = set('ACDEFGHIKLMNPQRSTVWY')
        seq = ''.join(aa for aa in seq.upper() if aa in valid)
        if len(seq) >= 10:
            seqs[pid] = seq

    # For proteins not in FireProtDB, fetch from RCSB
    import urllib.request
    missing = [pid for pid in protein_ids if pid not in seqs]
    print(f"Fetching sequences for {len(missing)} proteins from RCSB...")
    for i, pid in enumerate(missing):
        url = f"https://www.rcsb.org/fasta/entry/{pid}/download"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                text = r.read().decode()
            lines = text.strip().split('\n')
            seq = ''
            for line in lines:
                if line.startswith('>'):
                    if seq:
                        break
                else:
                    seq += line.strip()
            valid = set('ACDEFGHIKLMNPQRSTVWY')
            seq = ''.join(aa for aa in seq.upper() if aa in valid)
            if len(seq) >= 10:
                seqs[pid] = seq
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(missing)}] fetched={len(seqs)}")
        time.sleep(0.05)

    print(f"Total sequences: {len(seqs)}")
    return seqs


def compute_loglik_cache(sequences, positions):
    """Run ESM-2 masked prediction for each (protein, position) pair.

    Returns
    -------
    dict { protein_id (str) : { resnum (int) : np.array(20,) log-probs } }
    """
    import torch
    import esm

    print("\nLoading ESM-2 model (esm2_t30_150M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Build token indices for 20 standard AAs
    aa_token_indices = [alphabet.get_idx(aa) for aa in AA_ORDER]
    # Mask token index
    mask_idx = alphabet.mask_idx

    cache = {}
    total_proteins = sum(1 for pid in positions if pid in sequences)
    total_positions = sum(len(v) for pid, v in positions.items() if pid in sequences)

    print(f"Processing {total_proteins} proteins, {total_positions} unique positions...")
    t0 = time.time()
    n_done = 0
    n_pos_done = 0

    with torch.no_grad():
        for n_done, pid in enumerate(positions):
            if pid not in sequences:
                continue
            seq = sequences[pid]
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]
            L = len(seq)

            # Get the mutation positions for this protein
            mut_resnums = sorted(positions[pid])
            # Filter to valid 1-indexed positions
            mut_resnums = [r for r in mut_resnums if 1 <= r <= L]
            if not mut_resnums:
                continue

            protein_loglik = {}

            for resnum in mut_resnums:
                idx_0 = resnum - 1  # 0-indexed in seq
                # Create masked sequence
                masked_seq = seq[:idx_0] + '<mask>' + seq[idx_0+1:]
                data = [(pid, masked_seq)]
                _, _, tokens = batch_converter(data)

                # Forward pass
                results = model(tokens, repr_layers=[], return_contacts=False)
                # Logits at the masked position (token idx offset by 1 for BOS)
                logits = results['logits'][0, idx_0 + 1, :]   # [vocab_size]

                # Extract log-probs for 20 standard AAs
                aa_logits = logits[aa_token_indices]
                log_probs = aa_logits - torch.logsumexp(aa_logits, dim=0)
                protein_loglik[resnum] = log_probs.cpu().numpy().astype(np.float32)
                n_pos_done += 1

            cache[pid] = protein_loglik

            if (n_done + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = n_pos_done / elapsed
                eta_min = (total_positions - n_pos_done) / max(rate, 0.01) / 60
                print(f"  [{n_done+1}/{total_proteins}] proteins done | "
                      f"{n_pos_done}/{total_positions} positions | "
                      f"ETA: {eta_min:.1f} min")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    return cache


def main():
    print("=" * 70)
    print("ESM-2 MASKED LOG-LIKELIHOOD GENERATION")
    print("Model: esm2_t30_150M_UR50D | ΔLL = log P(mut) − log P(wt)")
    print("Reference: Meier et al. (2021) NeurIPS; Lin et al. (2023) Science")
    print("=" * 70)

    # Check if output already exists
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'rb') as f:
            existing = pickle.load(f)
        n_pos = sum(len(v) for v in existing.values())
        print(f"\nExisting cache: {len(existing)} proteins, {n_pos} positions.")
        print("Delete the file to regenerate.")
        return existing

    # Load mutation positions from training data
    print("\nLoading mutation positions from training data...")
    positions = load_mutation_positions()
    n_total_pos = sum(len(v) for v in positions.values())
    print(f"  {len(positions)} proteins, {n_total_pos} unique mutation positions")

    # Load sequences
    sequences = load_sequences()

    # Compute log-likelihood cache
    cache = compute_loglik_cache(sequences, positions)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(cache, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6

    # Summary statistics
    n_proteins = len(cache)
    n_positions = sum(len(v) for v in cache.values())
    print(f"\nSaved → {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"  {n_proteins} proteins, {n_positions} positions")

    # Verify a sample
    for pid, logliks in cache.items():
        if len(logliks) > 0:
            sample_pos = next(iter(logliks))
            sample = logliks[sample_pos]
            print(f"\nSample: {pid} pos {sample_pos}")
            print(f"  log-probs shape: {sample.shape}, range: [{sample.min():.2f}, {sample.max():.2f}]")
            # Show top-3 predicted AAs
            top3 = sorted(zip(AA_ORDER, sample), key=lambda x: -x[1])[:3]
            print(f"  Top-3 AAs: {top3}")
            break


if __name__ == '__main__':
    main()
