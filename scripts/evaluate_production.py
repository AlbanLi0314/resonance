#!/usr/bin/env python3
"""
Evaluate Academic Matcher on Production Test Queries
=====================================================
Runs test queries against the search system and measures:
- Precision@1, Precision@5
- Mean Reciprocal Rank (MRR)
- Recall (how many expected matches found)
- Hit Rate (queries with at least one relevant result)
"""
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matcher import AcademicMatcher


def load_test_queries(path: str) -> List[Dict]:
    """Load test queries from JSON file"""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def evaluate_single_query(
    matcher: AcademicMatcher,
    query: str,
    expected_matches: List[str],
    use_rerank: bool = True,
    top_k: int = 5
) -> Dict:
    """
    Evaluate a single query.

    Returns:
        Dict with metrics for this query
    """
    # Run search
    results = matcher.search(query, skip_rerank=not use_rerank, final_k=top_k)

    # Get result names (lowercase for matching)
    result_names = [r["name"].lower() for r in results]
    expected_lower = [name.lower() for name in expected_matches]

    # Calculate metrics

    # Precision@1: Is the first result in expected matches?
    p_at_1 = 1 if result_names and result_names[0] in expected_lower else 0

    # Precision@5: What fraction of top 5 are relevant?
    hits_in_top5 = sum(1 for name in result_names[:5] if name in expected_lower)
    p_at_5 = hits_in_top5 / min(5, len(result_names)) if result_names else 0

    # Recall: What fraction of expected matches were found?
    found_expected = sum(1 for name in expected_lower if name in result_names)
    recall = found_expected / len(expected_lower) if expected_lower else 0

    # Reciprocal Rank: 1/rank of first relevant result
    rr = 0
    for i, name in enumerate(result_names):
        if name in expected_lower:
            rr = 1 / (i + 1)
            break

    # Hit: Did we find at least one relevant result?
    hit = 1 if found_expected > 0 else 0

    return {
        "p_at_1": p_at_1,
        "p_at_5": p_at_5,
        "recall": recall,
        "reciprocal_rank": rr,
        "hit": hit,
        "hits_in_top5": hits_in_top5,
        "total_expected": len(expected_lower),
        "results": [r["name"] for r in results],
    }


def run_evaluation(
    matcher: AcademicMatcher,
    test_queries: List[Dict],
    use_rerank: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run full evaluation on all test queries.

    Returns:
        Dict with aggregate metrics and per-query results
    """
    results = []

    for i, tq in enumerate(test_queries):
        query = tq["query"]
        expected = tq["expected_matches"]
        category = tq.get("category", "Unknown")

        if verbose:
            print(f"\n[{i+1}/{len(test_queries)}] {category}: {query[:50]}...")

        metrics = evaluate_single_query(
            matcher, query, expected, use_rerank=use_rerank
        )
        metrics["query_id"] = tq["id"]
        metrics["query"] = query
        metrics["category"] = category
        metrics["expected"] = expected

        if verbose:
            status = "HIT" if metrics["hit"] else "MISS"
            print(f"    {status} | P@1={metrics['p_at_1']:.0f} P@5={metrics['p_at_5']:.2f} "
                  f"Recall={metrics['recall']:.2f} RR={metrics['reciprocal_rank']:.2f}")
            print(f"    Top 3: {', '.join(metrics['results'][:3])}")

        results.append(metrics)

    # Aggregate metrics
    n = len(results)
    aggregate = {
        "num_queries": n,
        "mean_p_at_1": sum(r["p_at_1"] for r in results) / n,
        "mean_p_at_5": sum(r["p_at_5"] for r in results) / n,
        "mean_recall": sum(r["recall"] for r in results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in results) / n,
        "hit_rate": sum(r["hit"] for r in results) / n,
    }

    # By category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_metrics = {}
    for cat, cat_results in categories.items():
        cn = len(cat_results)
        category_metrics[cat] = {
            "num_queries": cn,
            "mean_p_at_1": sum(r["p_at_1"] for r in cat_results) / cn,
            "mrr": sum(r["reciprocal_rank"] for r in cat_results) / cn,
            "hit_rate": sum(r["hit"] for r in cat_results) / cn,
        }

    return {
        "aggregate": aggregate,
        "by_category": category_metrics,
        "per_query": results,
    }


def print_evaluation_report(eval_results: Dict):
    """Print formatted evaluation report"""
    agg = eval_results["aggregate"]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Metrics ({agg['num_queries']} queries):")
    print(f"  Precision@1:  {agg['mean_p_at_1']:.1%}")
    print(f"  Precision@5:  {agg['mean_p_at_5']:.1%}")
    print(f"  Mean Recall:  {agg['mean_recall']:.1%}")
    print(f"  MRR:          {agg['mrr']:.3f}")
    print(f"  Hit Rate:     {agg['hit_rate']:.1%}")

    print(f"\nBy Category:")
    print(f"  {'Category':<20} {'P@1':>8} {'MRR':>8} {'Hit%':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    for cat, metrics in sorted(eval_results["by_category"].items()):
        print(f"  {cat:<20} {metrics['mean_p_at_1']:>7.0%} {metrics['mrr']:>8.2f} {metrics['hit_rate']:>7.0%}")

    # Show worst performing queries
    print(f"\nLowest Performing Queries:")
    sorted_queries = sorted(eval_results["per_query"], key=lambda x: x["reciprocal_rank"])
    for q in sorted_queries[:5]:
        if q["reciprocal_rank"] < 0.5:
            print(f"  - {q['query'][:50]}...")
            print(f"    Expected: {', '.join(q['expected'][:3])}")
            print(f"    Got: {', '.join(q['results'][:3])}")

    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Academic Matcher")
    parser.add_argument("--test-file", type=str,
                        default="data/test_queries_production.json",
                        help="Path to test queries JSON")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable LLM reranking")
    parser.add_argument("--output", type=str,
                        help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="Only show summary")
    args = parser.parse_args()

    # Load test queries
    print("Loading test queries...")
    test_queries = load_test_queries(args.test_file)
    print(f"Loaded {len(test_queries)} test queries")

    # Initialize matcher
    print("\nInitializing Academic Matcher...")
    matcher = AcademicMatcher()
    matcher.initialize()

    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)

    use_rerank = not args.no_rerank
    print(f"LLM Reranking: {'Enabled' if use_rerank else 'Disabled'}")

    eval_results = run_evaluation(
        matcher, test_queries,
        use_rerank=use_rerank,
        verbose=not args.quiet
    )

    # Print report
    print_evaluation_report(eval_results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
