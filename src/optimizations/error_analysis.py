"""
Error Analysis
==============
Analyzes failure cases to understand what the system gets wrong.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import Counter


def run_error_analysis(output_file: Path = None) -> Dict:
    """
    Analyze search failures to understand failure modes.

    Returns:
        Error analysis results
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.embedding import Specter2Encoder, format_researcher_text
    from src.config import DATA_DIR

    print("Running Error Analysis...")

    # Load data
    researchers_file = DATA_DIR / "overnight_results" / "researchers_large.json"
    if not researchers_file.exists():
        researchers_file = DATA_DIR / "raw" / "researchers_expanded.json"
    if not researchers_file.exists():
        researchers_file = DATA_DIR / "raw" / "researchers.json"

    with open(researchers_file) as f:
        data = json.load(f)
    researchers = data["researchers"]
    print(f"Loaded {len(researchers)} researchers")

    # Load test queries
    queries_file = DATA_DIR / "overnight_results" / "test_queries_large.json"
    if not queries_file.exists():
        queries_file = DATA_DIR / "test_queries_expanded.json"
    if not queries_file.exists():
        queries_file = DATA_DIR / "test_queries.json"

    with open(queries_file) as f:
        queries_data = json.load(f)
    test_queries = queries_data.get("queries", [])
    print(f"Loaded {len(test_queries)} test queries")

    # Prepare data
    documents = [format_researcher_text(r, "optimized") for r in researchers]
    id_to_idx = {r["id"]: i for i, r in enumerate(researchers)}
    idx_to_id = {i: r["id"] for i, r in enumerate(researchers)}

    # Load encoder and generate embeddings (use CPU to avoid MPS mutex issues)
    print("Loading encoder (CPU mode for stability)...")
    encoder = Specter2Encoder(force_cpu=True)
    encoder.load()

    print("Generating embeddings...")
    embeddings = encoder.encode_batch(documents, show_progress=True)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def search(query_emb, top_k=10):
        query_norm = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(embeddings_norm, query_norm)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(i), float(similarities[i])) for i in top_indices]

    # Analyze each query
    results = {
        "total_queries": len(test_queries),
        "successes": [],
        "failures": [],
        "near_misses": [],  # Expected in top 5 but not top 1
        "analysis": {}
    }

    failure_domains = Counter()
    success_domains = Counter()

    for q in test_queries:
        query_text = q.get("query", "")
        expected_id = q.get("expected_id", "")
        domain = q.get("domain", "Unknown")

        if not expected_id or expected_id not in id_to_idx:
            continue

        expected_idx = id_to_idx[expected_id]
        expected_researcher = researchers[expected_idx]

        # Get query embedding and search
        query_emb = encoder.encode(query_text)
        search_results = search(query_emb, 10)
        result_indices = [idx for idx, _ in search_results]
        result_scores = {idx: score for idx, score in search_results}

        # Analyze result
        if expected_idx in result_indices:
            rank = result_indices.index(expected_idx) + 1

            if rank == 1:
                # Success
                success_domains[domain] += 1
                results["successes"].append({
                    "query_id": q.get("id", ""),
                    "query": query_text,
                    "expected_name": expected_researcher.get("name", ""),
                    "rank": rank,
                    "score": result_scores[expected_idx],
                    "domain": domain
                })
            else:
                # Near miss
                actual_top = researchers[result_indices[0]]
                results["near_misses"].append({
                    "query_id": q.get("id", ""),
                    "query": query_text,
                    "expected_name": expected_researcher.get("name", ""),
                    "expected_rank": rank,
                    "expected_score": result_scores[expected_idx],
                    "actual_top_name": actual_top.get("name", ""),
                    "actual_top_score": result_scores[result_indices[0]],
                    "domain": domain
                })
        else:
            # Failure
            failure_domains[domain] += 1
            actual_top = researchers[result_indices[0]]
            results["failures"].append({
                "query_id": q.get("id", ""),
                "query": query_text,
                "expected_name": expected_researcher.get("name", ""),
                "expected_domain": expected_researcher.get("_domain", ""),
                "actual_top_name": actual_top.get("name", ""),
                "actual_top_domain": actual_top.get("_domain", ""),
                "domain": domain
            })

    # Analysis
    results["analysis"] = {
        "success_rate": len(results["successes"]) / len(test_queries) if test_queries else 0,
        "near_miss_rate": len(results["near_misses"]) / len(test_queries) if test_queries else 0,
        "failure_rate": len(results["failures"]) / len(test_queries) if test_queries else 0,
        "top_5_rate": (len(results["successes"]) + len(results["near_misses"])) / len(test_queries) if test_queries else 0,
        "failures_by_domain": dict(failure_domains),
        "successes_by_domain": dict(success_domains),
    }

    # Common failure patterns
    if results["failures"]:
        domain_mismatches = sum(1 for f in results["failures"]
                               if f.get("expected_domain") != f.get("actual_top_domain"))
        results["analysis"]["cross_domain_failures"] = domain_mismatches
        results["analysis"]["cross_domain_failure_rate"] = domain_mismatches / len(results["failures"])

    print(f"\nSuccess rate (P@1): {results['analysis']['success_rate']:.2%}")
    print(f"Top-5 rate: {results['analysis']['top_5_rate']:.2%}")
    print(f"Failure rate: {results['analysis']['failure_rate']:.2%}")

    if results["failures"]:
        print(f"\nFailure domains: {dict(failure_domains)}")

    # Recommendations
    recommendations = []

    if results["analysis"]["failure_rate"] > 0.3:
        recommendations.append("High failure rate - consider using hybrid search or query expansion")

    if results["near_misses"]:
        recommendations.append("Near misses detected - reranking with LLM should help")

    if failure_domains:
        worst_domain = failure_domains.most_common(1)[0][0] if failure_domains else None
        if worst_domain:
            recommendations.append(f"Domain '{worst_domain}' has most failures - may need domain-specific tuning")

    results["recommendations"] = recommendations

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    from pathlib import Path
    output = Path(__file__).parent.parent.parent / "data" / "overnight_results" / "error_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_error_analysis(output)
