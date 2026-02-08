#!/usr/bin/env python3
"""
Simple Evaluation Script with Threading Fixes
==============================================
"""
import os
# Set threading env vars BEFORE any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import math
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Set torch threads before importing torch
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
GROUND_TRUTH_FILE = PROJECT_ROOT / "ground_truth" / "labels" / "ground_truth_v1.json"


def load_ground_truth():
    """Load ground truth labels."""
    with open(GROUND_TRUTH_FILE) as f:
        data = json.load(f)

    gt = defaultdict(dict)
    for label in data.get("labels", []):
        query = label["query"]
        researcher = label["researcher_name"]
        score = label["relevance_score"]
        gt[query][researcher] = score

    return dict(gt)


def dcg(relevances, k=None):
    if k is not None:
        relevances = relevances[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(relevances, ideal_relevances, k=None):
    actual = dcg(relevances, k)
    ideal = dcg(sorted(ideal_relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0


def precision_at_k(relevances, k, threshold=2):
    top_k = relevances[:k]
    if not top_k:
        return 0
    return sum(1 for r in top_k if r >= threshold) / len(top_k)


def reciprocal_rank(relevances, threshold=2):
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1 / (i + 1)
    return 0


def main():
    print("Loading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} queries")

    # Variants to evaluate
    variants = ["raw_text_optimized", "raw_text_enriched"]

    # Load encoder once
    print("\nLoading SPECTER2 encoder...")
    from src.embedding import Specter2Encoder
    from src.indexer import FaissIndexer

    encoder = Specter2Encoder()
    encoder.load()
    print("Encoder loaded successfully!")

    results = {}

    for variant in variants:
        pkl_file = PROCESSED_DIR / f"researchers_emb_{variant}.pkl"
        index_file = INDEX_DIR / f"faiss_{variant}.index"

        if not pkl_file.exists() or not index_file.exists():
            print(f"\nVariant {variant} not found, skipping...")
            continue

        print(f"\nEvaluating {variant}...")

        # Load data
        with open(pkl_file, 'rb') as f:
            researchers = pickle.load(f)

        indexer = FaissIndexer()
        indexer.load(str(index_file))

        # Evaluate each query
        metrics_list = []
        for i, (query, judgments) in enumerate(ground_truth.items()):
            if (i + 1) % 5 == 0:
                print(f"  Query {i+1}/{len(ground_truth)}...")

            # Encode query
            query_emb = encoder.encode(query)

            # Search
            distances, indices = indexer.search(query_emb, 5)

            # Get result names
            result_names = []
            for idx in indices:
                if 0 <= idx < len(researchers):
                    result_names.append(researchers[idx].get('name', ''))

            # Map to relevance scores
            relevances = [judgments.get(name, 0) for name in result_names]
            ideal_relevances = list(judgments.values())

            # Compute metrics
            metrics_list.append({
                "ndcg_5": ndcg(relevances, ideal_relevances, k=5),
                "p_at_1": precision_at_k(relevances, k=1),
                "p_at_3": precision_at_k(relevances, k=3),
                "mrr": reciprocal_rank(relevances),
            })

        # Aggregate
        n = len(metrics_list)
        agg = {}
        for metric in ["ndcg_5", "p_at_1", "p_at_3", "mrr"]:
            values = [m[metric] for m in metrics_list]
            agg[metric] = sum(values) / n if n > 0 else 0

        results[variant] = agg
        print(f"  nDCG@5: {agg['ndcg_5']:.3f}, P@1: {agg['p_at_1']:.1%}, MRR: {agg['mrr']:.3f}")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Variant':<30} {'nDCG@5':>10} {'P@1':>10} {'P@3':>10} {'MRR':>10}")
    print("-" * 70)

    for variant, agg in results.items():
        print(f"{variant:<30} {agg['ndcg_5']:>10.3f} {agg['p_at_1']:>9.1%} {agg['p_at_3']:>9.1%} {agg['mrr']:>10.3f}")

    print("-" * 70)

    # Save results
    output_file = DATA_DIR / "enriched" / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
