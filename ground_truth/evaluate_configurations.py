#!/usr/bin/env python3
"""
Configuration Comparison Evaluation
====================================
Compares different search and reranking configurations against ground truth.

Configurations tested:
1. Dense only (baseline)
2. Dense + Cross-encoder reranking
3. Hybrid (Dense + BM25 with RRF)
4. Hybrid + Cross-encoder reranking (recommended)

Metrics:
- nDCG@k: Normalized Discounted Cumulative Gain
- P@k: Precision at k
- MRR: Mean Reciprocal Rank
"""
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_ground_truth(path: Path = None) -> Dict:
    """Load ground truth labels"""
    if path is None:
        path = PROJECT_ROOT / "ground_truth" / "labels" / "ground_truth_llm.json"
    with open(path) as f:
        return json.load(f)


def get_queries_with_labels(gt_data: Dict) -> Dict[str, Dict[str, int]]:
    """
    Organize ground truth by query.

    Returns:
        Dict mapping query -> {researcher_name: relevance_score}
    """
    queries = {}
    for label in gt_data.get("labels", []):
        query = label["query"]
        name = label["researcher_name"]
        score = label["relevance_score"]

        if query not in queries:
            queries[query] = {}
        queries[query][name] = score

    return queries


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG@k"""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = np.array(relevances)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Compute NDCG@k"""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(relevances: List[int], k: int, threshold: int = 2) -> float:
    """Compute Precision@k (count scores >= threshold as relevant)"""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    relevant = sum(1 for r in relevances if r >= threshold)
    return relevant / k


def reciprocal_rank(relevances: List[int], threshold: int = 2) -> float:
    """Compute Reciprocal Rank (1/rank of first relevant result)"""
    for i, r in enumerate(relevances):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_results(
    results: List[Dict],
    query_labels: Dict[str, int]
) -> Dict[str, float]:
    """
    Evaluate a single query's results against ground truth.

    Args:
        results: List of result dicts with 'name' field
        query_labels: Dict mapping researcher name to relevance score

    Returns:
        Dict with evaluation metrics
    """
    # Get relevance scores for returned results
    relevances = []
    for r in results:
        name = r.get("name", "")
        score = query_labels.get(name, 0)  # Default to 0 if not in ground truth
        relevances.append(score)

    return {
        "ndcg_5": ndcg_at_k(relevances, 5),
        "ndcg_3": ndcg_at_k(relevances, 3),
        "p_at_1": precision_at_k(relevances, 1),
        "p_at_3": precision_at_k(relevances, 3),
        "p_at_5": precision_at_k(relevances, 5),
        "mrr": reciprocal_rank(relevances),
    }


def run_evaluation(
    matcher,
    queries_with_labels: Dict[str, Dict[str, int]],
    search_mode: str = "dense",
    skip_rerank: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Run evaluation for a specific configuration.

    Args:
        matcher: Initialized AcademicMatcher
        queries_with_labels: Dict mapping query -> {name: score}
        search_mode: "dense" or "hybrid"
        skip_rerank: Whether to skip reranking
        verbose: Print per-query results

    Returns:
        Aggregated metrics
    """
    all_metrics = []

    for query, labels in queries_with_labels.items():
        try:
            # Run search
            results = matcher.search(
                query,
                recall_k=20,
                final_k=5,
                skip_rerank=skip_rerank,
                search_mode=search_mode
            )

            # Evaluate
            metrics = evaluate_results(results, labels)
            all_metrics.append(metrics)

            if verbose:
                print(f"  Query: {query[:50]}...")
                print(f"    nDCG@5: {metrics['ndcg_5']:.3f}, P@1: {metrics['p_at_1']:.1%}")

        except Exception as e:
            print(f"  Error on query '{query[:30]}...': {e}")

    # Aggregate
    if not all_metrics:
        return {}

    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[key] = float(np.mean(values))

    return aggregated


def main():
    import argparse
    from src.matcher import AcademicMatcher

    parser = argparse.ArgumentParser(description="Evaluate search configurations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("CONFIGURATION COMPARISON EVALUATION")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt_data = load_ground_truth()
    queries_with_labels = get_queries_with_labels(gt_data)
    print(f"Found {len(queries_with_labels)} queries with labels")

    # Define configurations to test
    configs = [
        {
            "name": "Dense Only (Baseline)",
            "search_mode": "dense",
            "rerank_mode": "none",
        },
        {
            "name": "Dense + Cross-Encoder",
            "search_mode": "dense",
            "rerank_mode": "cross_encoder",
        },
        {
            "name": "Hybrid (Dense + BM25)",
            "search_mode": "hybrid",
            "rerank_mode": "none",
        },
        {
            "name": "Hybrid + Cross-Encoder (Recommended)",
            "search_mode": "hybrid",
            "rerank_mode": "cross_encoder",
        },
    ]

    results = {}

    for config in configs:
        print("\n" + "-" * 60)
        print(f"Testing: {config['name']}")
        print("-" * 60)

        # Initialize matcher with this config
        matcher = AcademicMatcher()
        matcher.initialize(
            search_mode=config["search_mode"],
            rerank_mode=config["rerank_mode"]
        )

        # Determine skip_rerank based on rerank_mode
        skip_rerank = config["rerank_mode"] == "none"

        # Run evaluation
        start_time = time.time()
        metrics = run_evaluation(
            matcher,
            queries_with_labels,
            search_mode=config["search_mode"],
            skip_rerank=skip_rerank,
            verbose=args.verbose
        )
        elapsed = time.time() - start_time

        metrics["time_seconds"] = elapsed
        results[config["name"]] = metrics

        # Print results
        print(f"\nResults for {config['name']}:")
        print(f"  nDCG@5:  {metrics.get('ndcg_5', 0):.3f}")
        print(f"  nDCG@3:  {metrics.get('ndcg_3', 0):.3f}")
        print(f"  P@1:     {metrics.get('p_at_1', 0):.1%}")
        print(f"  P@3:     {metrics.get('p_at_3', 0):.1%}")
        print(f"  P@5:     {metrics.get('p_at_5', 0):.1%}")
        print(f"  MRR:     {metrics.get('mrr', 0):.3f}")
        print(f"  Time:    {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Configuration':<35} {'nDCG@5':>8} {'P@1':>8} {'MRR':>8}")
    print("-" * 60)

    baseline_ndcg = results.get("Dense Only (Baseline)", {}).get("ndcg_5", 0)

    for name, metrics in results.items():
        ndcg = metrics.get("ndcg_5", 0)
        p1 = metrics.get("p_at_1", 0)
        mrr = metrics.get("mrr", 0)

        improvement = ""
        if baseline_ndcg > 0 and name != "Dense Only (Baseline)":
            pct_change = (ndcg - baseline_ndcg) / baseline_ndcg * 100
            improvement = f" ({pct_change:+.1f}%)"

        print(f"{name:<35} {ndcg:>7.3f}{improvement:>10} {p1:>7.1%} {mrr:>7.3f}")

    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "ground_truth" / "configuration_comparison.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
