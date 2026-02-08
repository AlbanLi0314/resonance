#!/usr/bin/env python3
"""
Evaluate using adhoc_query adapter for queries only.
Documents already encoded with proximity adapter in the index.
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
    gt_path = PROJECT_ROOT / "ground_truth" / "labels" / "ground_truth_llm.json"
    with open(gt_path) as f:
        return json.load(f)


def get_queries_with_labels(gt_data):
    queries = {}
    for label in gt_data.get("labels", []):
        query = label["query"]
        name = label["researcher_name"]
        score = label["relevance_score"]
        if query not in queries:
            queries[query] = {}
        queries[query][name] = score
    return queries


def dcg_at_k(relevances, k):
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = np.array(relevances)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevances, k, threshold=2):
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(1 for r in relevances if r >= threshold) / k


def mrr(relevances, threshold=2):
    for i, r in enumerate(relevances):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


class AdhocQueryEncoder:
    """Simple encoder that only uses adhoc_query adapter"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        print("Loading SPECTER2 with adhoc_query adapter...")

        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

        # Load only adhoc_query adapter
        self.model.load_adapter("allenai/specter2_adhoc_query", source="hf",
                               load_as="adhoc_query", set_active=True)
        self.model.to(self.device)
        self.model.eval()
        print("✓ Loaded")
        return self

    def encode(self, text):
        import torch
        inputs = self.tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()


def main():
    import faiss
    from src.hybrid_search import BM25Index

    print("=" * 60)
    print("ASYMMETRIC ADAPTER EVALUATION")
    print("Query: adhoc_query | Documents: proximity")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt_data = load_ground_truth()
    queries_with_labels = get_queries_with_labels(gt_data)
    print(f"Found {len(queries_with_labels)} queries")

    # Load encoder
    encoder = AdhocQueryEncoder()
    encoder.load()

    # Load data (proximity-encoded)
    print("\nLoading researcher data...")
    with open(PROJECT_ROOT / "data/processed/researchers_dual_adapter.pkl", "rb") as f:
        researchers = pickle.load(f)
    print(f"Loaded {len(researchers)} researchers")

    # Load index
    index = faiss.read_index(str(PROJECT_ROOT / "data/index/faiss_dual_adapter.index"))
    print(f"Loaded FAISS index ({index.ntotal} vectors)")

    # BM25 for hybrid
    print("\nBuilding BM25 index...")
    bm25 = BM25Index()
    docs = [r.get("raw_text", "") for r in researchers]
    bm25.build(docs)

    # Evaluate
    print("\nRunning evaluation...")
    all_metrics = []
    start_time = time.time()

    for i, (query, labels) in enumerate(queries_with_labels.items()):
        # Encode query
        query_emb = encoder.encode(query).astype('float32').reshape(1, -1)

        # Dense search
        distances, indices = index.search(query_emb, 20)
        dense_results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]

        # Sparse search
        sparse_results = bm25.search(query, top_k=20)

        # RRF fusion
        rrf_k, alpha = 60, 0.7
        rrf_scores = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + alpha / (rrf_k + rank + 1)
        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - alpha) / (rrf_k + rank + 1)

        final_indices = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:5]

        # Relevance
        relevances = []
        for idx in final_indices:
            if 0 <= idx < len(researchers):
                name = researchers[idx].get("name", "")
                relevances.append(labels.get(name, 0))

        all_metrics.append({
            "ndcg_5": ndcg_at_k(relevances, 5),
            "ndcg_3": ndcg_at_k(relevances, 3),
            "p_at_1": precision_at_k(relevances, 1),
            "p_at_3": precision_at_k(relevances, 3),
            "mrr": mrr(relevances)
        })

        if (i + 1) % 15 == 0:
            print(f"  Processed {i + 1}/{len(queries_with_labels)} queries...")

    elapsed = time.time() - start_time

    # Aggregate
    results = {"time_seconds": elapsed}
    for key in ["ndcg_5", "ndcg_3", "p_at_1", "p_at_3", "mrr"]:
        results[key] = float(np.mean([m[key] for m in all_metrics]))

    # Baseline
    with open(PROJECT_ROOT / "ground_truth/configuration_comparison.json") as f:
        baseline = json.load(f).get("Hybrid (Dense + BM25)", {})

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    baseline_ndcg = baseline.get("ndcg_5", 0.713)
    dual_ndcg = results["ndcg_5"]
    improvement = (dual_ndcg - baseline_ndcg) / baseline_ndcg * 100

    print(f"\n{'Metric':<12} {'Baseline':>10} {'Asymmetric':>12} {'Change':>10}")
    print("-" * 50)
    print(f"{'nDCG@5':<12} {baseline_ndcg:>10.3f} {dual_ndcg:>12.3f} {improvement:>+9.1f}%")
    print(f"{'P@1':<12} {baseline.get('p_at_1', 0):>9.1%} {results['p_at_1']:>11.1%}")
    print(f"{'MRR':<12} {baseline.get('mrr', 0):>10.3f} {results['mrr']:>12.3f}")
    print(f"{'Time':<12} {baseline.get('time_seconds', 0):>9.1f}s {elapsed:>11.1f}s")
    print("-" * 50)

    print("\n" + "=" * 60)
    if improvement > 1:
        print(f"✅ IMPROVEMENT: +{improvement:.1f}% nDCG@5")
    elif improvement < -1:
        print(f"⚠️ REGRESSION: {improvement:.1f}% nDCG@5")
    else:
        print(f"➖ NO SIGNIFICANT CHANGE: {improvement:.1f}%")
    print("=" * 60)

    # Save
    output = {"asymmetric": results, "baseline": baseline, "improvement_pct": improvement}
    with open(PROJECT_ROOT / "ground_truth/asymmetric_results.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
