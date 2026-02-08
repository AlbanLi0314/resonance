#!/usr/bin/env python3
"""
Evaluate Enhanced Search with Query Expansion + Gemini Reranking
================================================================
Tests 4 configurations:
1. Baseline (current hybrid search)
2. Query Expansion only
3. Gemini Reranking only
4. Query Expansion + Gemini Reranking
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

from src.query_expansion import expand_query
from src.gemini_reranker import rerank_with_gemini
from src.hybrid_search import BM25Index


def load_data():
    """Load all necessary data"""
    # Ground truth
    with open(PROJECT_ROOT / "ground_truth/labels/ground_truth_llm.json") as f:
        gt = json.load(f)

    queries_labels = {}
    for label in gt["labels"]:
        q = label["query"]
        if q not in queries_labels:
            queries_labels[q] = {}
        queries_labels[q][label["researcher_name"]] = label["relevance_score"]

    # Query embeddings
    with open(PROJECT_ROOT / "data/processed/query_embeddings_adhoc.pkl", "rb") as f:
        query_embeddings = pickle.load(f)

    # Researchers
    with open(PROJECT_ROOT / "data/processed/researchers_dual_adapter.pkl", "rb") as f:
        researchers = pickle.load(f)

    # FAISS index
    import faiss
    index = faiss.read_index(str(PROJECT_ROOT / "data/index/faiss_dual_adapter.index"))

    return queries_labels, query_embeddings, researchers, index


def dcg(rels, k):
    rels = rels[:k]
    if not rels:
        return 0.0
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg(rels, k):
    d = dcg(rels, k)
    ideal = dcg(sorted(rels, reverse=True), k)
    return d / ideal if ideal > 0 else 0.0


def p_at_1(rels):
    return 1.0 if rels and rels[0] >= 2 else 0.0


def mrr(rels):
    for i, r in enumerate(rels):
        if r >= 2:
            return 1 / (i + 1)
    return 0.0


def hybrid_search(query_emb, index, bm25, researchers, query_text, top_k=10):
    """Perform hybrid search and return candidates"""
    query_emb = query_emb.astype("float32").reshape(1, -1)

    # Dense
    _, idxs = index.search(query_emb, 20)
    dense = [(int(i), rank) for rank, i in enumerate(idxs[0]) if i >= 0]

    # Sparse
    sparse = [(i, rank) for rank, (i, _) in enumerate(bm25.search(query_text, 20))]

    # RRF
    scores = {}
    for i, rank in dense:
        scores[i] = scores.get(i, 0) + 0.7 / (60 + rank + 1)
    for i, rank in sparse:
        scores[i] = scores.get(i, 0) + 0.3 / (60 + rank + 1)

    top_indices = sorted(scores.keys(), key=lambda x: -scores[x])[:top_k]
    return [researchers[i] for i in top_indices if 0 <= i < len(researchers)]


def evaluate_config(name, queries_labels, query_embeddings, researchers, index, bm25,
                   use_expansion=False, use_rerank=False, encoder=None):
    """Evaluate a configuration"""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}")
    print(f"  Query Expansion: {'ON' if use_expansion else 'OFF'}")
    print(f"  Gemini Rerank: {'ON' if use_rerank else 'OFF'}")
    print("=" * 60)

    all_ndcg5, all_p1, all_mrr = [], [], []
    start_time = time.time()

    expanded_queries = {}

    for i, (query, labels) in enumerate(queries_labels.items()):
        # Query expansion
        search_query = query
        if use_expansion:
            if query not in expanded_queries:
                expanded_queries[query] = expand_query(query)
            search_query = expanded_queries[query]

        # Get query embedding
        if query in query_embeddings:
            query_emb = query_embeddings[query]
        else:
            # Need to encode expanded query - use original for now
            query_emb = query_embeddings.get(query, None)
            if query_emb is None:
                continue

        # Hybrid search (get more candidates for reranking)
        candidates = hybrid_search(
            query_emb, index, bm25, researchers,
            search_query if use_expansion else query,
            top_k=10 if use_rerank else 5
        )

        # Gemini reranking
        if use_rerank and candidates:
            candidates = rerank_with_gemini(query, candidates, top_k=5)
        else:
            candidates = candidates[:5]

        # Get relevance scores
        rels = [labels.get(c.get("name", ""), 0) for c in candidates]

        all_ndcg5.append(ndcg(rels, 5))
        all_p1.append(p_at_1(rels))
        all_mrr.append(mrr(rels))

        if (i + 1) % 15 == 0:
            print(f"  Processed {i + 1}/{len(queries_labels)} queries...")

    elapsed = time.time() - start_time

    results = {
        "ndcg_5": float(np.mean(all_ndcg5)),
        "p_at_1": float(np.mean(all_p1)),
        "mrr": float(np.mean(all_mrr)),
        "time_seconds": elapsed
    }

    print(f"\nResults:")
    print(f"  nDCG@5: {results['ndcg_5']:.3f}")
    print(f"  P@1:    {results['p_at_1']:.1%}")
    print(f"  MRR:    {results['mrr']:.3f}")
    print(f"  Time:   {elapsed:.1f}s")

    return results


def main():
    print("=" * 60)
    print("ENHANCED SEARCH EVALUATION")
    print("Query Expansion + Gemini Reranking")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    queries_labels, query_embeddings, researchers, index = load_data()
    print(f"  {len(queries_labels)} queries")
    print(f"  {len(researchers)} researchers")

    # Build BM25
    print("\nBuilding BM25 index...")
    bm25 = BM25Index()
    bm25.build([r.get("raw_text", "") for r in researchers])

    # Test configurations
    configs = [
        ("Baseline (Hybrid only)", False, False),
        ("+ Query Expansion", True, False),
        ("+ Gemini Reranking", False, True),
        ("+ Both (Expansion + Rerank)", True, True),
    ]

    results = {}
    for name, use_exp, use_rerank in configs:
        results[name] = evaluate_config(
            name, queries_labels, query_embeddings, researchers, index, bm25,
            use_expansion=use_exp, use_rerank=use_rerank
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_p1 = results["Baseline (Hybrid only)"]["p_at_1"]

    print(f"\n{'Configuration':<35} {'nDCG@5':>10} {'P@1':>10} {'Change':>10} {'Time':>8}")
    print("-" * 75)

    for name, metrics in results.items():
        ndcg5 = metrics["ndcg_5"]
        p1 = metrics["p_at_1"]
        time_s = metrics["time_seconds"]

        change = ""
        if name != "Baseline (Hybrid only)":
            change_pct = (p1 - baseline_p1) / baseline_p1 * 100 if baseline_p1 > 0 else 0
            change = f"{change_pct:+.1f}%"

        print(f"{name:<35} {ndcg5:>10.3f} {p1:>9.1%} {change:>10} {time_s:>7.1f}s")

    print("-" * 75)

    # Find best
    best_name = max(results.keys(), key=lambda x: results[x]["p_at_1"])
    best_p1 = results[best_name]["p_at_1"]
    improvement = (best_p1 - baseline_p1) / baseline_p1 * 100 if baseline_p1 > 0 else 0

    print(f"\n✅ BEST: {best_name}")
    print(f"   P@1: {baseline_p1:.1%} → {best_p1:.1%} ({improvement:+.1f}%)")
    print("=" * 70)

    # Save
    output = {"results": results, "best_config": best_name}
    with open(PROJECT_ROOT / "ground_truth/enhanced_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: ground_truth/enhanced_evaluation.json")


if __name__ == "__main__":
    main()
