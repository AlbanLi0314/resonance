#!/usr/bin/env python3
"""
Evaluate Dual Adapter Configuration
====================================
Compares the performance of:
1. Original: Single generic adapter (specter2) for both queries and documents
2. Dual Adapter: adhoc_query for queries + proximity for documents

Expected improvement: 5-15% nDCG
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
    """Compute Precision@k"""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    relevant = sum(1 for r in relevances if r >= threshold)
    return relevant / k


def mrr(relevances: List[int], threshold: int = 2) -> float:
    """Compute Mean Reciprocal Rank"""
    for i, r in enumerate(relevances):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_config(name: str, encoder, researchers: List[Dict],
                   index, queries_with_labels: Dict,
                   use_hybrid: bool = True) -> Dict:
    """Evaluate a configuration"""
    print(f"\nEvaluating: {name}")
    print("-" * 40)

    from src.hybrid_search import BM25Index

    # Build BM25 index for hybrid search
    bm25 = None
    if use_hybrid:
        bm25 = BM25Index()
        docs = [r.get("raw_text", "") for r in researchers]
        bm25.build(docs)

    all_metrics = []
    start_time = time.time()

    for query, labels in queries_with_labels.items():
        # Encode query
        query_emb = encoder.encode_query(query).astype('float32').reshape(1, -1)

        # Dense search
        distances, indices = index.search(query_emb, 20)
        dense_results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]

        if use_hybrid and bm25:
            # Sparse search
            sparse_results = bm25.search(query, top_k=20)

            # RRF fusion
            rrf_k = 60
            rrf_scores = {}
            alpha = 0.7  # 70% dense, 30% sparse

            for rank, (idx, _) in enumerate(dense_results):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (rrf_k + rank + 1)

            for rank, (idx, _) in enumerate(sparse_results):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (rrf_k + rank + 1)

            final_indices = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:5]
        else:
            final_indices = [idx for idx, _ in dense_results[:5]]

        # Get relevance scores
        relevances = []
        for idx in final_indices:
            if 0 <= idx < len(researchers):
                name = researchers[idx].get("name", "")
                score = labels.get(name, 0)
                relevances.append(score)

        # Compute metrics
        metrics = {
            "ndcg_5": ndcg_at_k(relevances, 5),
            "ndcg_3": ndcg_at_k(relevances, 3),
            "p_at_1": precision_at_k(relevances, 1),
            "p_at_3": precision_at_k(relevances, 3),
            "mrr": mrr(relevances)
        }
        all_metrics.append(metrics)

    elapsed = time.time() - start_time

    # Aggregate
    aggregated = {"time_seconds": elapsed}
    for key in ["ndcg_5", "ndcg_3", "p_at_1", "p_at_3", "mrr"]:
        values = [m[key] for m in all_metrics]
        aggregated[key] = float(np.mean(values))

    print(f"  nDCG@5: {aggregated['ndcg_5']:.3f}")
    print(f"  P@1:    {aggregated['p_at_1']:.1%}")
    print(f"  MRR:    {aggregated['mrr']:.3f}")
    print(f"  Time:   {elapsed:.1f}s")

    return aggregated


def main():
    import faiss

    print("=" * 60)
    print("DUAL ADAPTER EVALUATION")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt_data = load_ground_truth()
    queries_with_labels = get_queries_with_labels(gt_data)
    print(f"Found {len(queries_with_labels)} queries")

    results = {}

    # === Configuration 1: Original (single generic adapter) ===
    print("\n" + "=" * 60)
    print("CONFIG 1: Original (single specter2 adapter)")
    print("=" * 60)

    from src.embedding import Specter2Encoder

    original_encoder = Specter2Encoder()
    original_encoder.load()

    # Wrap to have encode_query method
    class OriginalEncoderWrapper:
        def __init__(self, encoder):
            self.encoder = encoder
        def encode_query(self, text):
            return self.encoder.encode(text)

    original_wrapped = OriginalEncoderWrapper(original_encoder)

    # Load original data
    with open(PROJECT_ROOT / "data/processed/researchers_optimized.pkl", "rb") as f:
        original_researchers = pickle.load(f)

    original_index = faiss.read_index(
        str(PROJECT_ROOT / "data/index/faiss_optimized.index")
    )

    results["Original (single adapter)"] = evaluate_config(
        "Original (single specter2 adapter)",
        original_wrapped,
        original_researchers,
        original_index,
        queries_with_labels,
        use_hybrid=True
    )

    # === Configuration 2: Dual Adapter ===
    print("\n" + "=" * 60)
    print("CONFIG 2: Dual Adapter (adhoc_query + proximity)")
    print("=" * 60)

    from src.dual_encoder import DualAdapterEncoder

    dual_encoder = DualAdapterEncoder()
    dual_encoder.load()

    # Load dual adapter data
    with open(PROJECT_ROOT / "data/processed/researchers_dual_adapter.pkl", "rb") as f:
        dual_researchers = pickle.load(f)

    dual_index = faiss.read_index(
        str(PROJECT_ROOT / "data/index/faiss_dual_adapter.index")
    )

    results["Dual Adapter (adhoc_query + proximity)"] = evaluate_config(
        "Dual Adapter",
        dual_encoder,
        dual_researchers,
        dual_index,
        queries_with_labels,
        use_hybrid=True
    )

    # === Summary ===
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    baseline_ndcg = results["Original (single adapter)"]["ndcg_5"]

    print(f"\n{'Configuration':<45} {'nDCG@5':>10} {'Change':>10} {'P@1':>10} {'MRR':>10}")
    print("-" * 85)

    for name, metrics in results.items():
        ndcg = metrics["ndcg_5"]
        p1 = metrics["p_at_1"]
        mrr_val = metrics["mrr"]

        change = ""
        if name != "Original (single adapter)":
            pct = (ndcg - baseline_ndcg) / baseline_ndcg * 100
            change = f"{pct:+.1f}%"

        print(f"{name:<45} {ndcg:>10.3f} {change:>10} {p1:>9.1%} {mrr_val:>10.3f}")

    print("=" * 85)

    # Save results
    output_path = PROJECT_ROOT / "ground_truth" / "dual_adapter_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Determine winner
    dual_ndcg = results["Dual Adapter (adhoc_query + proximity)"]["ndcg_5"]
    improvement = (dual_ndcg - baseline_ndcg) / baseline_ndcg * 100

    print("\n" + "=" * 60)
    if improvement > 0:
        print(f"✅ DUAL ADAPTER IMPROVED PERFORMANCE BY {improvement:.1f}%")
    else:
        print(f"⚠️ DUAL ADAPTER DID NOT IMPROVE (change: {improvement:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
