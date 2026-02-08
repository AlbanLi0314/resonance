#!/usr/bin/env python3
"""
Step 2: Test Search
==================
This script tests the search functionality after building the index.

Usage:
    python scripts/step2_test_search.py

    # Test without LLM (embedding only)
    python scripts/step2_test_search.py --no-rerank

    # Custom query
    python scripts/step2_test_search.py --query "machine learning for materials"

    # Run validation with golden test set
    python scripts/step2_test_search.py --validate
"""
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matcher import AcademicMatcher, print_results
from src.config import print_config, DATA_DIR


# Default test queries (fallback if test_queries.json not found)
DEFAULT_TEST_QUERIES = [
    "DNA origami folding and assembly optimization techniques",
    "Machine learning for battery materials discovery",
    "Polymer synthesis and hydrogel fabrication",
    "Heterogeneous catalysis for CO2 conversion",
    "2D materials synthesis and characterization",
]

TEST_QUERIES_FILE = DATA_DIR / "test_queries.json"


def load_test_queries() -> list:
    """Load test queries from JSON file or use defaults"""
    if TEST_QUERIES_FILE.exists():
        with open(TEST_QUERIES_FILE, 'r') as f:
            data = json.load(f)
            return data.get("queries", [])
    return [{"query": q} for q in DEFAULT_TEST_QUERIES]


def run_test(query: str, matcher: AcademicMatcher, skip_rerank: bool = False):
    """Run a single test query"""
    print("\n" + "-" * 60)
    print(f"QUERY: {query}")
    print("-" * 60)

    results = matcher.search(query, skip_rerank=skip_rerank)
    print_results(results)

    return results


def check_result_relevance(results: list, expected_keywords: list) -> dict:
    """
    Check if results contain expected keywords.

    Returns:
        Dict with relevance metrics
    """
    if not results or not expected_keywords:
        return {"score": 0, "matched": [], "missing": expected_keywords}

    # Combine all result text for checking
    all_text = ""
    for r in results:
        all_text += f" {r.get('name', '')} {r.get('department', '')} {r.get('research_interests', '')} {r.get('raw_text', '')}"
    all_text = all_text.lower()

    matched = []
    missing = []

    for kw in expected_keywords:
        if kw.lower() in all_text:
            matched.append(kw)
        else:
            missing.append(kw)

    score = len(matched) / len(expected_keywords) if expected_keywords else 0

    return {
        "score": round(score, 2),
        "matched": matched,
        "missing": missing
    }


def run_validation(matcher: AcademicMatcher, skip_rerank: bool = False) -> dict:
    """
    Run validation against golden test set.

    Returns:
        Validation report dict
    """
    print("\n" + "=" * 60)
    print("VALIDATION MODE: Testing against golden test set")
    print("=" * 60)

    test_queries = load_test_queries()
    report = {
        "total_queries": len(test_queries),
        "results": [],
        "avg_relevance_score": 0,
    }

    total_score = 0

    for tq in test_queries:
        query = tq.get("query", "")
        expected_keywords = tq.get("expected_relevant_keywords", [])
        query_id = tq.get("id", "unknown")

        print(f"\n--- Test {query_id}: {query[:50]}...")

        results = matcher.search(query, skip_rerank=skip_rerank)

        # Check relevance
        relevance = check_result_relevance(results, expected_keywords)
        total_score += relevance["score"]

        # Print summary
        print(f"    Results: {len(results)}")
        print(f"    Relevance score: {relevance['score']:.0%}")
        if relevance["matched"]:
            print(f"    Matched keywords: {relevance['matched']}")
        if relevance["missing"]:
            print(f"    Missing keywords: {relevance['missing']}")

        report["results"].append({
            "query_id": query_id,
            "query": query,
            "num_results": len(results),
            "top_result": results[0].get("name") if results else None,
            "relevance": relevance
        })

    report["avg_relevance_score"] = round(total_score / len(test_queries), 2) if test_queries else 0

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total queries tested: {report['total_queries']}")
    print(f"Average relevance score: {report['avg_relevance_score']:.0%}")

    passed = sum(1 for r in report["results"] if r["relevance"]["score"] >= 0.5)
    print(f"Queries with >=50% relevance: {passed}/{report['total_queries']}")

    if report["avg_relevance_score"] < 0.5:
        print("\n⚠️ WARNING: Low average relevance score!")
        print("   Consider reviewing embeddings or test queries.")
    else:
        print("\n✅ Validation passed!")

    return report


def main():
    parser = argparse.ArgumentParser(description="Test academic matcher search")
    parser.add_argument("--query", "-q", type=str, help="Custom query to test")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip LLM reranking (embedding only)")
    parser.add_argument("--all", action="store_true",
                        help="Run all test queries")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation against golden test set")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Test Search")
    print("=" * 60)

    # Print config
    print_config()

    # Initialize matcher
    print("\nInitializing matcher...")
    matcher = AcademicMatcher()

    try:
        matcher.initialize(skip_rerank=args.no_rerank)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure you've run step1_build_index.py first!")
        sys.exit(1)

    # Run tests
    if args.validate:
        # Validation mode
        report = run_validation(matcher, skip_rerank=args.no_rerank)
        # Save report
        report_path = DATA_DIR / "validation_results.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nValidation report saved to: {report_path}")
    elif args.query:
        # Single custom query
        run_test(args.query, matcher, skip_rerank=args.no_rerank)
    elif args.all:
        # All test queries
        test_queries = load_test_queries()
        for tq in test_queries:
            query = tq.get("query") if isinstance(tq, dict) else tq
            run_test(query, matcher, skip_rerank=args.no_rerank)
    else:
        # Default: first test query
        test_queries = load_test_queries()
        first_query = test_queries[0].get("query") if isinstance(test_queries[0], dict) else test_queries[0]
        run_test(first_query, matcher, skip_rerank=args.no_rerank)

    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
