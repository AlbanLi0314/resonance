#!/usr/bin/env python3
"""
Evaluate Enriched Indices with Ground Truth
============================================
Standalone evaluation script for comparing different text variants.
Avoids threading issues by using simpler initialization.
"""

import json
import math
import pickle
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
GROUND_TRUTH_FILE = PROJECT_ROOT / "ground_truth" / "labels" / "ground_truth_v1.json"


def load_ground_truth() -> Dict[str, Dict[str, int]]:
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


def dcg(relevances: List[int], k: int = None) -> float:
    """Compute Discounted Cumulative Gain"""
    if k is not None:
        relevances = relevances[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(relevances: List[int], ideal_relevances: List[int], k: int = None) -> float:
    """Compute Normalized DCG"""
    actual = dcg(relevances, k)
    ideal = dcg(sorted(ideal_relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0


def precision_at_k(relevances: List[int], k: int, threshold: int = 2) -> float:
    """Compute Precision@K with relevance threshold"""
    top_k = relevances[:k]
    if not top_k:
        return 0
    return sum(1 for r in top_k if r >= threshold) / len(top_k)


def reciprocal_rank(relevances: List[int], threshold: int = 2) -> float:
    """Compute Reciprocal Rank"""
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1 / (i + 1)
    return 0


def evaluate_variant(variant: str, ground_truth: Dict[str, Dict[str, int]], top_k: int = 5) -> Dict:
    """
    Evaluate a specific text variant against ground truth.

    Returns metrics dict or None if variant doesn't exist.
    """
    # Load variant-specific files
    pkl_file = PROCESSED_DIR / f"researchers_emb_{variant}.pkl"
    index_file = INDEX_DIR / f"faiss_{variant}.index"

    if not pkl_file.exists() or not index_file.exists():
        print(f"  Variant {variant} not found, skipping...")
        return None

    # Load data
    print(f"  Loading {variant}...")
    with open(pkl_file, 'rb') as f:
        researchers = pickle.load(f)

    # Load encoder (this is the expensive part)
    from src.embedding import Specter2Encoder
    from src.indexer import FaissIndexer

    encoder = Specter2Encoder()
    encoder.load()

    indexer = FaissIndexer()
    indexer.load(str(index_file))

    # Evaluate each query
    metrics_by_query = {}

    for query, judgments in ground_truth.items():
        # Encode query
        query_emb = encoder.encode(query)

        # Search
        distances, indices = indexer.search(query_emb, top_k)

        # Get result names
        result_names = []
        for idx in indices:
            if 0 <= idx < len(researchers):
                result_names.append(researchers[idx].get('name', ''))

        # Map to relevance scores
        relevances = [judgments.get(name, 0) for name in result_names]
        ideal_relevances = list(judgments.values())

        # Compute metrics
        metrics = {
            "ndcg_5": ndcg(relevances, ideal_relevances, k=5),
            "p_at_1": precision_at_k(relevances, k=1),
            "p_at_3": precision_at_k(relevances, k=3),
            "mrr": reciprocal_rank(relevances),
        }
        metrics_by_query[query] = metrics

    # Aggregate
    n = len(metrics_by_query)
    aggregate = {}
    for metric in ["ndcg_5", "p_at_1", "p_at_3", "mrr"]:
        values = [m[metric] for m in metrics_by_query.values()]
        aggregate[metric] = sum(values) / n if n > 0 else 0
    aggregate["num_queries"] = n

    return {
        "variant": variant,
        "aggregate": aggregate,
        "by_query": metrics_by_query
    }


def compare_variants(variants: List[str], ground_truth: Dict[str, Dict[str, int]]):
    """Compare multiple variants and print comparison table."""

    print("\n" + "=" * 70)
    print("COMPARING TEXT VARIANTS")
    print("=" * 70)

    results = {}

    for variant in variants:
        print(f"\nEvaluating {variant}...")
        result = evaluate_variant(variant, ground_truth)
        if result:
            results[variant] = result
            agg = result["aggregate"]
            print(f"  nDCG@5: {agg['ndcg_5']:.3f}, P@1: {agg['p_at_1']:.1%}, MRR: {agg['mrr']:.3f}")

    # Comparison table
    print("\n" + "-" * 70)
    print(f"{'Variant':<30} {'nDCG@5':>10} {'P@1':>10} {'P@3':>10} {'MRR':>10}")
    print("-" * 70)

    for variant, result in results.items():
        agg = result["aggregate"]
        print(f"{variant:<30} {agg['ndcg_5']:>10.3f} {agg['p_at_1']:>9.1%} {agg['p_at_3']:>9.1%} {agg['mrr']:>10.3f}")

    print("-" * 70)

    # Find best variant
    if results:
        best_variant = max(results.items(), key=lambda x: x[1]["aggregate"]["ndcg_5"])
        print(f"\nBest variant by nDCG@5: {best_variant[0]} ({best_variant[1]['aggregate']['ndcg_5']:.3f})")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate enriched indices")
    parser.add_argument("--variant", type=str, default="all",
                       help="Specific variant to evaluate (default: all)")
    parser.add_argument("--output", type=str,
                       help="Save results to JSON file")
    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} queries")

    # Determine variants to evaluate
    if args.variant == "all":
        variants = [
            "raw_text_optimized",
            "raw_text_enriched",
        ]
        # Also check for baseline
        baseline_pkl = PROCESSED_DIR / "researchers_with_emb.pkl"
        if baseline_pkl.exists():
            # Check if it's different from optimized
            variants_to_check = variants.copy()
    else:
        variants = [args.variant]

    # Run comparison
    results = compare_variants(variants, ground_truth)

    # Save results
    if args.output and results:
        output_data = {
            variant: {
                "aggregate": res["aggregate"],
                "by_query": res["by_query"]
            }
            for variant, res in results.items()
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
