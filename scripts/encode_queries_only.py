#!/usr/bin/env python3
"""
Pre-compute query embeddings and save them.
This avoids repeated model loading during evaluation.
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Force single-threaded before torch import
import torch
torch.set_num_threads(1)

# Disable MPS
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GROUND_TRUTH_FILE = PROJECT_ROOT / "ground_truth" / "labels" / "ground_truth_v1.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "enriched" / "query_embeddings.pkl"


def main():
    # Load ground truth queries
    print("Loading ground truth queries...")
    with open(GROUND_TRUTH_FILE) as f:
        data = json.load(f)

    queries = list(set(label["query"] for label in data.get("labels", [])))
    print(f"Found {len(queries)} unique queries")

    # Load encoder
    print("\nLoading SPECTER2 encoder...")
    from src.embedding import Specter2Encoder
    encoder = Specter2Encoder()
    encoder.load()

    # Encode all queries
    print("\nEncoding queries...")
    query_embeddings = {}
    for i, query in enumerate(queries):
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(queries)}")
        query_embeddings[query] = encoder.encode(query)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(query_embeddings, f)
    print(f"\nSaved {len(query_embeddings)} query embeddings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
