#!/usr/bin/env python3
"""
Evaluate System with Ground Truth Labels
=========================================
Uses human-labeled relevance judgments for proper IR evaluation.

Metrics computed:
- nDCG (Normalized Discounted Cumulative Gain)
- Precision@K with relevance threshold
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).parent
LABELS_FILE = OUTPUT_DIR / "labels" / "ground_truth_v1.json"


def load_ground_truth() -> Dict[str, Dict[str, int]]:
    """
    Load ground truth labels.

    Returns:
        Dict mapping query -> {researcher_name: relevance_score}
    """
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"No ground truth file at {LABELS_FILE}")

    with open(LABELS_FILE) as f:
        data = json.load(f)

    # Organize by query
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


def recall_at_k(relevances: List[int], all_relevant: int, k: int, threshold: int = 2) -> float:
    """Compute Recall@K"""
    if all_relevant == 0:
        return 0
    top_k = relevances[:k]
    found = sum(1 for r in top_k if r >= threshold)
    return found / all_relevant


def reciprocal_rank(relevances: List[int], threshold: int = 2) -> float:
    """Compute Reciprocal Rank"""
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1 / (i + 1)
    return 0


def average_precision(relevances: List[int], threshold: int = 2) -> float:
    """Compute Average Precision"""
    relevant_count = 0
    precision_sum = 0

    for i, rel in enumerate(relevances):
        if rel >= threshold:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)

    total_relevant = sum(1 for r in relevances if r >= threshold)
    return precision_sum / total_relevant if total_relevant > 0 else 0


def evaluate_system(
    ground_truth: Dict[str, Dict[str, int]],
    use_rerank: bool = False,
    top_k: int = 5
) -> Dict:
    """
    Evaluate the search system using ground truth.

    Args:
        ground_truth: Query -> {researcher: score} mapping
        use_rerank: Whether to use LLM reranking
        top_k: Number of results to retrieve

    Returns:
        Dict with evaluation metrics
    """
    from src.matcher import AcademicMatcher

    print("Initializing matcher...")
    matcher = AcademicMatcher()
    matcher.initialize()

    results_by_query = {}
    metrics_by_query = {}

    print(f"\nEvaluating {len(ground_truth)} queries...")

    for i, (query, judgments) in enumerate(ground_truth.items()):
        print(f"  [{i+1}/{len(ground_truth)}] {query[:50]}...")

        # Get system results
        try:
            results = matcher.search(query, skip_rerank=not use_rerank, final_k=top_k)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        # Map results to relevance scores
        result_names = [r["name"] for r in results]
        relevances = [judgments.get(name, 0) for name in result_names]

        # All known relevant items
        all_relevant = [score for score in judgments.values() if score >= 2]
        ideal_relevances = list(judgments.values())

        # Compute metrics for this query
        metrics = {
            "ndcg_5": ndcg(relevances, ideal_relevances, k=5),
            "ndcg_3": ndcg(relevances, ideal_relevances, k=3),
            "p_at_1": precision_at_k(relevances, k=1),
            "p_at_3": precision_at_k(relevances, k=3),
            "p_at_5": precision_at_k(relevances, k=5),
            "recall_at_5": recall_at_k(relevances, len(all_relevant), k=5),
            "rr": reciprocal_rank(relevances),
            "ap": average_precision(relevances),
        }

        metrics_by_query[query] = metrics
        results_by_query[query] = {
            "results": result_names,
            "relevances": relevances,
            "judgments": judgments
        }

    # Aggregate metrics
    n = len(metrics_by_query)
    if n == 0:
        return {"error": "No queries evaluated"}

    aggregate = {}
    for metric_name in ["ndcg_5", "ndcg_3", "p_at_1", "p_at_3", "p_at_5", "recall_at_5", "rr", "ap"]:
        values = [m[metric_name] for m in metrics_by_query.values()]
        aggregate[f"mean_{metric_name}"] = sum(values) / n

    aggregate["mrr"] = aggregate["mean_rr"]
    aggregate["map"] = aggregate["mean_ap"]
    aggregate["num_queries"] = n

    return {
        "aggregate": aggregate,
        "by_query": metrics_by_query,
        "details": results_by_query
    }


def print_evaluation(eval_results: Dict, title: str = "EVALUATION RESULTS"):
    """Print evaluation results"""

    if "error" in eval_results:
        print(f"Error: {eval_results['error']}")
        return

    agg = eval_results["aggregate"]

    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    print(f"\n📊 AGGREGATE METRICS ({agg['num_queries']} queries)")
    print(f"   {'Metric':<20} {'Value':>10}")
    print(f"   {'-'*20} {'-'*10}")
    print(f"   {'nDCG@5':<20} {agg['mean_ndcg_5']:>10.3f}")
    print(f"   {'nDCG@3':<20} {agg['mean_ndcg_3']:>10.3f}")
    print(f"   {'Precision@1':<20} {agg['mean_p_at_1']:>10.1%}")
    print(f"   {'Precision@3':<20} {agg['mean_p_at_3']:>10.1%}")
    print(f"   {'Precision@5':<20} {agg['mean_p_at_5']:>10.1%}")
    print(f"   {'Recall@5':<20} {agg['mean_recall_at_5']:>10.1%}")
    print(f"   {'MRR':<20} {agg['mrr']:>10.3f}")
    print(f"   {'MAP':<20} {agg['map']:>10.3f}")

    # Best/worst queries
    by_query = eval_results["by_query"]
    sorted_queries = sorted(by_query.items(), key=lambda x: x[1]["ndcg_5"], reverse=True)

    print(f"\n🏆 BEST QUERIES (by nDCG@5)")
    for query, metrics in sorted_queries[:3]:
        print(f"   • {query[:50]}...")
        print(f"     nDCG={metrics['ndcg_5']:.3f}, P@1={metrics['p_at_1']:.0%}")

    print(f"\n📉 WORST QUERIES (by nDCG@5)")
    for query, metrics in sorted_queries[-3:]:
        print(f"   • {query[:50]}...")
        print(f"     nDCG={metrics['ndcg_5']:.3f}, P@1={metrics['p_at_1']:.0%}")

    print("=" * 60)


def compare_configurations(ground_truth: Dict[str, Dict[str, int]]):
    """Compare different system configurations"""

    print("\n" + "=" * 60)
    print("  COMPARING CONFIGURATIONS")
    print("=" * 60)

    configs = [
        ("Without LLM Rerank", {"use_rerank": False}),
        ("With LLM Rerank", {"use_rerank": True}),
    ]

    results = {}
    for name, kwargs in configs:
        print(f"\n--- {name} ---")
        eval_result = evaluate_system(ground_truth, **kwargs)
        results[name] = eval_result
        if "aggregate" in eval_result:
            agg = eval_result["aggregate"]
            print(f"nDCG@5: {agg['mean_ndcg_5']:.3f}, P@1: {agg['mean_p_at_1']:.1%}, MRR: {agg['mrr']:.3f}")

    # Comparison table
    print("\n" + "-" * 60)
    print(f"{'Configuration':<25} {'nDCG@5':>10} {'P@1':>10} {'MRR':>10}")
    print("-" * 60)
    for name, result in results.items():
        if "aggregate" in result:
            agg = result["aggregate"]
            print(f"{name:<25} {agg['mean_ndcg_5']:>10.3f} {agg['mean_p_at_1']:>9.1%} {agg['mrr']:>10.3f}")
    print("-" * 60)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate with ground truth")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with and without reranking")
    parser.add_argument("--rerank", action="store_true",
                        help="Use LLM reranking")
    parser.add_argument("--output", type=str,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Load ground truth
    try:
        ground_truth = load_ground_truth()
        print(f"Loaded ground truth for {len(ground_truth)} queries")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nYou need to create ground truth labels first:")
        print("  1. python ground_truth/generate_candidates.py")
        print("  2. python ground_truth/interactive_labeler.py")
        return

    if args.compare:
        results = compare_configurations(ground_truth)
    else:
        results = evaluate_system(ground_truth, use_rerank=args.rerank)
        print_evaluation(results, "WITH RERANK" if args.rerank else "WITHOUT RERANK")

    if args.output:
        # Remove non-serializable parts
        output = {k: v for k, v in results.items() if k != "details"}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
