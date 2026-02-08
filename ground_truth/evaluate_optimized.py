#!/usr/bin/env python3
"""
Evaluate Optimized Index
========================
Compare original vs optimized index performance.
"""
import json
import pickle
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding import Specter2Encoder
from src.indexer import FaissIndexer
from src.hybrid_search import HybridSearcher
from src.text_optimizer import expand_query


def load_ground_truth():
    with open(PROJECT_ROOT / "ground_truth" / "labels" / "ground_truth_llm.json") as f:
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
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_index(encoder, indexer, researchers, queries_with_labels, use_hybrid=True, use_query_expansion=False):
    """Evaluate an index configuration."""
    if use_hybrid:
        searcher = HybridSearcher(researchers, indexer, encoder)
        searcher.build_bm25_index()

    metrics_list = []

    for query, labels in queries_with_labels.items():
        # Optionally expand query
        search_query = expand_query(query) if use_query_expansion else query

        # Search
        if use_hybrid:
            results = searcher.search(search_query, top_k=5)
        else:
            query_emb = encoder.encode(search_query)
            distances, indices = indexer.search(query_emb, k=5)
            results = [researchers[idx] for idx in indices if idx >= 0]

        # Get relevances
        relevances = [labels.get(r.get('name', ''), 0) for r in results]

        metrics_list.append({
            'ndcg_5': ndcg_at_k(relevances, 5),
            'p_at_1': 1.0 if relevances and relevances[0] >= 2 else 0.0,
            'mrr': next((1.0/(i+1) for i, r in enumerate(relevances) if r >= 2), 0.0)
        })

    # Aggregate
    return {
        'ndcg_5': np.mean([m['ndcg_5'] for m in metrics_list]),
        'p_at_1': np.mean([m['p_at_1'] for m in metrics_list]),
        'mrr': np.mean([m['mrr'] for m in metrics_list])
    }


def main():
    print("=" * 60)
    print("OPTIMIZED INDEX EVALUATION")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    gt = load_ground_truth()
    queries = get_queries_with_labels(gt)
    print(f"Found {len(queries)} queries")

    # Load encoder
    print("\nLoading encoder...")
    encoder = Specter2Encoder()
    encoder.load()

    # Load original index
    print("\nLoading original index...")
    with open(PROJECT_ROOT / "data" / "processed" / "researchers_with_emb.pkl", 'rb') as f:
        original_researchers = pickle.load(f)
    original_indexer = FaissIndexer()
    original_indexer.load(str(PROJECT_ROOT / "data" / "index" / "faiss.index"))

    # Load optimized index
    print("\nLoading optimized index...")
    optimized_path = PROJECT_ROOT / "data" / "processed" / "researchers_optimized.pkl"
    optimized_index_path = PROJECT_ROOT / "data" / "index" / "faiss_optimized.index"

    if not optimized_path.exists():
        print("ERROR: Optimized index not found!")
        print("Run: python scripts/rebuild_optimized_index.py")
        return

    with open(optimized_path, 'rb') as f:
        optimized_researchers = pickle.load(f)
    optimized_indexer = FaissIndexer()
    optimized_indexer.load(str(optimized_index_path))

    # Evaluate configurations
    print("\n" + "-" * 60)
    configs = [
        ("Original + Hybrid", original_researchers, original_indexer, True, False),
        ("Optimized + Hybrid", optimized_researchers, optimized_indexer, True, False),
        ("Optimized + Hybrid + Query Expansion", optimized_researchers, optimized_indexer, True, True),
    ]

    results = {}
    for name, researchers, indexer, use_hybrid, use_qe in configs:
        print(f"\nEvaluating: {name}...")
        metrics = evaluate_index(encoder, indexer, researchers, queries, use_hybrid, use_qe)
        results[name] = metrics
        print(f"  nDCG@5: {metrics['ndcg_5']:.3f}")
        print(f"  P@1:    {metrics['p_at_1']:.1%}")
        print(f"  MRR:    {metrics['mrr']:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<40} {'nDCG@5':>8} {'P@1':>8} {'MRR':>8}")
    print("-" * 60)

    baseline = results.get("Original + Hybrid", {})
    for name, metrics in results.items():
        ndcg = metrics['ndcg_5']
        p1 = metrics['p_at_1']
        mrr = metrics['mrr']

        improvement = ""
        if baseline and name != "Original + Hybrid":
            pct = (ndcg - baseline['ndcg_5']) / baseline['ndcg_5'] * 100
            improvement = f" ({pct:+.1f}%)"

        print(f"{name:<40} {ndcg:>7.3f}{improvement:>10} {p1:>7.1%} {mrr:>7.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
