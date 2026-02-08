#!/usr/bin/env python3
"""
Simplified Dual Adapter Evaluation
===================================
Only loads the dual adapter encoder to avoid memory issues.
Compares against saved baseline metrics.
"""
import json
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_ground_truth():
    """Load ground truth labels"""
    gt_path = PROJECT_ROOT / "ground_truth" / "labels" / "ground_truth_llm.json"
    with open(gt_path) as f:
        return json.load(f)


def get_queries_with_labels(gt_data: Dict) -> Dict[str, Dict[str, int]]:
    """Organize ground truth by query"""
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
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = np.array(relevances)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevances: List[int], k: int, threshold: int = 2) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(1 for r in relevances if r >= threshold) / k


def mrr(relevances: List[int], threshold: int = 2) -> float:
    for i, r in enumerate(relevances):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def main():
    import faiss
    from src.hybrid_search import BM25Index

    print("=" * 60)
    print("DUAL ADAPTER EVALUATION (Simplified)")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt_data = load_ground_truth()
    queries_with_labels = get_queries_with_labels(gt_data)
    print(f"Found {len(queries_with_labels)} queries")

    # Load dual adapter encoder
    print("\nLoading dual adapter encoder...")
    from src.dual_encoder import DualAdapterEncoder
    encoder = DualAdapterEncoder()
    encoder.load()

    # Load dual adapter data
    print("\nLoading researcher data (dual adapter)...")
    with open(PROJECT_ROOT / "data/processed/researchers_dual_adapter.pkl", "rb") as f:
        researchers = pickle.load(f)
    print(f"Loaded {len(researchers)} researchers")

    # Load index
    index = faiss.read_index(
        str(PROJECT_ROOT / "data/index/faiss_dual_adapter.index")
    )
    print(f"Loaded FAISS index with {index.ntotal} vectors")

    # Build BM25 for hybrid search
    print("\nBuilding BM25 index...")
    bm25 = BM25Index()
    docs = [r.get("raw_text", "") for r in researchers]
    bm25.build(docs)

    # Evaluate
    print("\nRunning evaluation...")
    all_metrics = []
    start_time = time.time()

    for query, labels in queries_with_labels.items():
        # Encode query with adhoc_query adapter
        query_emb = encoder.encode_query(query).astype('float32').reshape(1, -1)

        # Dense search
        distances, indices = index.search(query_emb, 20)
        dense_results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]

        # Sparse search
        sparse_results = bm25.search(query, top_k=20)

        # RRF fusion
        rrf_k = 60
        alpha = 0.7
        rrf_scores = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (rrf_k + rank + 1)

        final_indices = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:5]

        # Get relevance scores
        relevances = []
        for idx in final_indices:
            if 0 <= idx < len(researchers):
                name = researchers[idx].get("name", "")
                score = labels.get(name, 0)
                relevances.append(score)

        metrics = {
            "ndcg_5": ndcg_at_k(relevances, 5),
            "ndcg_3": ndcg_at_k(relevances, 3),
            "p_at_1": precision_at_k(relevances, 1),
            "p_at_3": precision_at_k(relevances, 3),
            "mrr": mrr(relevances)
        }
        all_metrics.append(metrics)

    elapsed = time.time() - start_time

    # Aggregate results
    results = {"time_seconds": elapsed}
    for key in ["ndcg_5", "ndcg_3", "p_at_1", "p_at_3", "mrr"]:
        values = [m[key] for m in all_metrics]
        results[key] = float(np.mean(values))

    # Load baseline for comparison
    baseline_path = PROJECT_ROOT / "ground_truth" / "configuration_comparison.json"
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    baseline = baseline_data.get("Hybrid (Dense + BM25)", {})

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<15} {'Baseline':>12} {'Dual Adapter':>12} {'Change':>12}")
    print("-" * 55)

    baseline_ndcg = baseline.get("ndcg_5", 0)
    dual_ndcg = results["ndcg_5"]

    metrics_to_show = [
        ("nDCG@5", "ndcg_5"),
        ("nDCG@3", "ndcg_3"),
        ("P@1", "p_at_1"),
        ("P@3", "p_at_3"),
        ("MRR", "mrr")
    ]

    for display_name, key in metrics_to_show:
        base_val = baseline.get(key, 0)
        dual_val = results[key]

        if key.startswith("p_"):
            change = f"{(dual_val - base_val)*100:+.1f}pp"
            print(f"{display_name:<15} {base_val:>11.1%} {dual_val:>11.1%} {change:>12}")
        else:
            change = f"{(dual_val - base_val)/base_val*100:+.1f}%" if base_val > 0 else "N/A"
            print(f"{display_name:<15} {base_val:>12.3f} {dual_val:>12.3f} {change:>12}")

    print("-" * 55)
    print(f"{'Time':<15} {baseline.get('time_seconds', 0):>11.1f}s {elapsed:>11.1f}s")

    # Summary
    improvement = (dual_ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0

    print("\n" + "=" * 60)
    if improvement > 0:
        print(f"✅ IMPROVEMENT: +{improvement:.1f}% nDCG@5")
        print(f"   Baseline: {baseline_ndcg:.3f} → Dual Adapter: {dual_ndcg:.3f}")
    elif improvement < -1:
        print(f"⚠️ REGRESSION: {improvement:.1f}% nDCG@5")
        print(f"   Baseline: {baseline_ndcg:.3f} → Dual Adapter: {dual_ndcg:.3f}")
    else:
        print(f"➖ NO SIGNIFICANT CHANGE: {improvement:.1f}% nDCG@5")
    print("=" * 60)

    # Save results
    output = {
        "dual_adapter": results,
        "baseline_hybrid": baseline,
        "improvement_pct": improvement
    }
    output_path = PROJECT_ROOT / "ground_truth" / "dual_adapter_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
