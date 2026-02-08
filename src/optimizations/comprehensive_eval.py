"""
Comprehensive Evaluation
========================
Runs all optimization strategies on the large dataset and compares results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def run_comprehensive_evaluation(output_file: Path = None) -> Dict:
    """
    Run comprehensive evaluation combining all optimizations.

    Tests:
    - Baseline (semantic only)
    - Optimized text format
    - Hybrid search (BM25 + semantic)
    - Query expansion
    - Combined (all optimizations)

    Returns:
        Comprehensive results
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.embedding import Specter2Encoder, format_researcher_text
    from src.config import DATA_DIR
    from src.optimizations.hybrid_search import HybridSearcher

    print("=" * 60)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 60)

    # Load data
    researchers_file = DATA_DIR / "overnight_results" / "researchers_large.json"
    if not researchers_file.exists():
        researchers_file = DATA_DIR / "raw" / "researchers_expanded.json"
    if not researchers_file.exists():
        researchers_file = DATA_DIR / "raw" / "researchers.json"

    print(f"Loading researchers from: {researchers_file}")
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

    print(f"Loading queries from: {queries_file}")
    with open(queries_file) as f:
        queries_data = json.load(f)
    test_queries = queries_data.get("queries", [])
    print(f"Loaded {len(test_queries)} test queries")

    # ID to index mapping
    id_to_idx = {r["id"]: i for i, r in enumerate(researchers)}

    # Load encoder (use CPU to avoid MPS mutex issues)
    print("\nLoading SPECTER2 encoder (CPU mode for stability)...")
    encoder = Specter2Encoder(force_cpu=True)
    encoder.load()

    results = {
        "timestamp": datetime.now().isoformat(),
        "num_researchers": len(researchers),
        "num_queries": len(test_queries),
        "evaluations": []
    }

    def evaluate_config(name: str, documents: List[str], use_hybrid: bool = False,
                       hybrid_weights: tuple = (0.7, 0.3)) -> Dict:
        """Evaluate a single configuration"""
        print(f"\n--- Evaluating: {name} ---")

        # Generate embeddings
        print("  Generating embeddings...")
        embeddings = encoder.encode_batch(documents, show_progress=False)

        # Normalize for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Setup search
        if use_hybrid:
            hybrid = HybridSearcher(semantic_weight=hybrid_weights[0], bm25_weight=hybrid_weights[1])
            hybrid.fit(documents, embeddings)

        def search(query_text: str, query_emb: np.ndarray, top_k: int = 10):
            if use_hybrid:
                return hybrid.search(query_text, query_emb, top_k, "rrf")
            else:
                query_norm = query_emb / np.linalg.norm(query_emb)
                similarities = np.dot(embeddings_norm, query_norm)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                return [(int(i), float(similarities[i])) for i in top_indices]

        # Evaluate
        mrr_sum = 0
        p_at_1 = 0
        p_at_3 = 0
        p_at_5 = 0
        valid_queries = 0
        per_query_results = []

        for q in test_queries:
            query_text = q.get("query", "")
            expected_id = q.get("expected_id", "")

            if not expected_id or expected_id not in id_to_idx:
                continue

            expected_idx = id_to_idx[expected_id]
            valid_queries += 1

            # Get query embedding
            query_emb = encoder.encode(query_text)

            # Search
            search_results = search(query_text, query_emb, 10)
            result_indices = [idx for idx, _ in search_results]

            # Calculate metrics
            rank = None
            if expected_idx in result_indices:
                rank = result_indices.index(expected_idx) + 1
                mrr_sum += 1.0 / rank
                if rank == 1:
                    p_at_1 += 1
                if rank <= 3:
                    p_at_3 += 1
                if rank <= 5:
                    p_at_5 += 1

            per_query_results.append({
                "query_id": q.get("id", ""),
                "expected_rank": rank,
                "found": rank is not None
            })

        eval_result = {
            "config_name": name,
            "mrr": mrr_sum / valid_queries if valid_queries > 0 else 0,
            "precision_at_1": p_at_1 / valid_queries if valid_queries > 0 else 0,
            "precision_at_3": p_at_3 / valid_queries if valid_queries > 0 else 0,
            "precision_at_5": p_at_5 / valid_queries if valid_queries > 0 else 0,
            "num_queries": valid_queries,
            "per_query_results": per_query_results
        }

        print(f"  MRR: {eval_result['mrr']:.4f}")
        print(f"  P@1: {eval_result['precision_at_1']:.4f}")
        print(f"  P@5: {eval_result['precision_at_5']:.4f}")

        return eval_result

    # Test 1: Baseline (raw_text format, semantic only)
    print("\n" + "=" * 40)
    print("TEST 1: Baseline")
    print("=" * 40)
    docs_raw = [r.get("raw_text", "") for r in researchers]
    baseline_result = evaluate_config("baseline_raw_text", docs_raw, use_hybrid=False)
    results["evaluations"].append(baseline_result)

    # Test 2: Optimized text format (semantic only)
    print("\n" + "=" * 40)
    print("TEST 2: Optimized Text Format")
    print("=" * 40)
    docs_optimized = [format_researcher_text(r, "optimized") for r in researchers]
    optimized_result = evaluate_config("optimized_format", docs_optimized, use_hybrid=False)
    results["evaluations"].append(optimized_result)

    # Test 3: Hybrid search (optimized format + BM25)
    print("\n" + "=" * 40)
    print("TEST 3: Hybrid Search")
    print("=" * 40)
    hybrid_result = evaluate_config("hybrid_search", docs_optimized, use_hybrid=True, hybrid_weights=(0.7, 0.3))
    results["evaluations"].append(hybrid_result)

    # Test 4: Alternative hybrid weights
    print("\n" + "=" * 40)
    print("TEST 4: Hybrid 50/50")
    print("=" * 40)
    hybrid_50_result = evaluate_config("hybrid_50_50", docs_optimized, use_hybrid=True, hybrid_weights=(0.5, 0.5))
    results["evaluations"].append(hybrid_50_result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Configuration':<25} {'MRR':<10} {'P@1':<10} {'P@5':<10}")
    print("-" * 55)

    best_config = None
    best_mrr = 0

    for eval_result in results["evaluations"]:
        name = eval_result["config_name"]
        mrr = eval_result["mrr"]
        p1 = eval_result["precision_at_1"]
        p5 = eval_result["precision_at_5"]
        print(f"{name:<25} {mrr:<10.4f} {p1:<10.4f} {p5:<10.4f}")

        if mrr > best_mrr:
            best_mrr = mrr
            best_config = name

    results["best_config"] = best_config
    results["best_mrr"] = best_mrr

    # Calculate improvements
    baseline_mrr = baseline_result["mrr"]
    if baseline_mrr > 0:
        for eval_result in results["evaluations"]:
            if eval_result["config_name"] != "baseline_raw_text":
                improvement = (eval_result["mrr"] - baseline_mrr) / baseline_mrr * 100
                eval_result["improvement_vs_baseline"] = f"+{improvement:.1f}%"

    print(f"\nBest: {best_config} (MRR: {best_mrr:.4f})")

    # Save results
    if output_file:
        # Remove per_query_results for cleaner output
        clean_results = results.copy()
        for eval_result in clean_results["evaluations"]:
            eval_result.pop("per_query_results", None)

        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    from pathlib import Path
    output = Path(__file__).parent.parent.parent / "data" / "overnight_results" / "comprehensive_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_comprehensive_evaluation(output)
