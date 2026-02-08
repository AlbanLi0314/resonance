#!/usr/bin/env python3
"""
Evaluate Web-Enriched Data with Hybrid Search
==============================================
Uses:
- SPECTER2 embeddings (existing, stable)
- BM25 on web-enriched text (new, more keywords)
"""
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent


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


class BM25Index:
    """Simple BM25 implementation"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0

    def build(self, documents: List[str]):
        import re
        self.docs = []
        for doc in documents:
            tokens = re.findall(r'\w+', doc.lower())
            self.docs.append(tokens)
            self.doc_len.append(len(tokens))

        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1

        # Document frequencies
        for doc in self.docs:
            seen = set()
            for token in doc:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

        # IDF
        N = len(self.docs)
        for token, df in self.doc_freqs.items():
            self.idf[token] = np.log((N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        import re
        query_tokens = re.findall(r'\w+', query.lower())

        scores = []
        for idx, doc in enumerate(self.docs):
            score = 0
            doc_len = self.doc_len[idx]

            for token in query_tokens:
                if token in self.idf:
                    tf = doc.count(token)
                    numerator = self.idf[token] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += numerator / denominator

            scores.append((idx, score))

        return sorted(scores, key=lambda x: -x[1])[:top_k]


def hybrid_search(query_emb, index, bm25, researchers, query_text, top_k=5,
                  dense_weight=0.7, sparse_weight=0.3):
    """RRF fusion of dense and sparse search"""
    query_emb = query_emb.astype("float32").reshape(1, -1)

    # Dense search
    _, idxs = index.search(query_emb, 20)
    dense = [(int(i), rank) for rank, i in enumerate(idxs[0]) if i >= 0]

    # Sparse search (BM25)
    sparse = [(i, rank) for rank, (i, _) in enumerate(bm25.search(query_text, 20))]

    # RRF fusion
    k = 60
    scores = {}
    for i, rank in dense:
        scores[i] = scores.get(i, 0) + dense_weight / (k + rank + 1)
    for i, rank in sparse:
        scores[i] = scores.get(i, 0) + sparse_weight / (k + rank + 1)

    top_indices = sorted(scores.keys(), key=lambda x: -scores[x])[:top_k]
    return [researchers[i] for i in top_indices if 0 <= i < len(researchers)]


def main():
    print("=" * 60)
    print("EVALUATE WEB-ENRICHED HYBRID SEARCH")
    print("=" * 60)

    # Load ground truth
    gt_path = PROJECT_ROOT / "ground_truth/labels/ground_truth_llm.json"
    print(f"\nLoading ground truth: {gt_path}")
    with open(gt_path) as f:
        gt = json.load(f)

    queries_labels = {}
    for label in gt["labels"]:
        q = label["query"]
        if q not in queries_labels:
            queries_labels[q] = {}
        queries_labels[q][label["researcher_name"]] = label["relevance_score"]

    print(f"Queries: {len(queries_labels)}")

    # Load query embeddings (existing, stable)
    query_emb_path = PROJECT_ROOT / "data/processed/query_embeddings_adhoc.pkl"
    print(f"Loading query embeddings: {query_emb_path}")
    with open(query_emb_path, "rb") as f:
        query_embeddings = pickle.load(f)

    # Load FAISS index (existing, stable)
    index_path = PROJECT_ROOT / "data/index/faiss_dual_adapter.index"
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(str(index_path))

    # Load original researchers (for embeddings alignment)
    orig_pkl = PROJECT_ROOT / "data/processed/researchers_dual_adapter.pkl"
    print(f"Loading original researchers: {orig_pkl}")
    with open(orig_pkl, "rb") as f:
        orig_researchers = pickle.load(f)

    # Load web-enriched data
    enriched_path = PROJECT_ROOT / "data/enriched/researchers_web_enriched.json"
    print(f"Loading web-enriched data: {enriched_path}")
    with open(enriched_path) as f:
        enriched_data = json.load(f)["researchers"]

    # Create name -> enriched mapping
    enriched_map = {r["name"]: r for r in enriched_data}

    # Build enriched texts for BM25
    print("\nBuilding enriched texts for BM25...")
    enriched_texts = []
    for r in orig_researchers:
        name = r.get("name", "")
        enriched = enriched_map.get(name, {})

        # Build enriched text
        parts = [name]
        if r.get("department"):
            parts.append(r["department"])
        if r.get("research_interests"):
            parts.append(r["research_interests"][:500])

        # Add web enrichment
        if enriched.get("web_research_summary"):
            parts.append(enriched["web_research_summary"])
        if enriched.get("web_keywords"):
            parts.append(" ".join(enriched["web_keywords"]))
        if enriched.get("web_research_areas"):
            parts.append(" ".join(enriched["web_research_areas"]))

        enriched_texts.append(" ".join(parts))

    # Build BM25 index
    print("Building BM25 index on enriched text...")
    bm25 = BM25Index()
    bm25.build(enriched_texts)

    # Also build BM25 on original text for comparison
    print("Building BM25 index on original text...")
    bm25_orig = BM25Index()
    bm25_orig.build([r.get("raw_text", "") for r in orig_researchers])

    # Evaluate configurations
    configs = [
        ("Original (hybrid on original text)", bm25_orig),
        ("Enriched (hybrid on enriched text)", bm25),
    ]

    results = {}

    for config_name, bm25_index in configs:
        print(f"\n{'=' * 50}")
        print(f"Evaluating: {config_name}")
        print("=" * 50)

        all_ndcg5, all_p1, all_mrr = [], [], []

        for query, labels in queries_labels.items():
            if query not in query_embeddings:
                continue

            query_emb = query_embeddings[query]

            # Hybrid search
            candidates = hybrid_search(
                query_emb, index, bm25_index, orig_researchers, query, top_k=5
            )

            # Get relevance scores
            rels = [labels.get(c.get("name", ""), 0) for c in candidates]

            all_ndcg5.append(ndcg(rels, 5))
            all_p1.append(p_at_1(rels))
            all_mrr.append(mrr(rels))

        results[config_name] = {
            "ndcg_5": float(np.mean(all_ndcg5)),
            "p_at_1": float(np.mean(all_p1)),
            "mrr": float(np.mean(all_mrr))
        }

        print(f"  nDCG@5: {results[config_name]['ndcg_5']:.3f}")
        print(f"  P@1:    {results[config_name]['p_at_1']:.1%}")
        print(f"  MRR:    {results[config_name]['mrr']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Configuration':<45} {'nDCG@5':>10} {'P@1':>10} {'MRR':>10}")
    print("-" * 75)

    for name, metrics in results.items():
        print(f"{name:<45} {metrics['ndcg_5']:>10.3f} {metrics['p_at_1']:>9.1%} {metrics['mrr']:>10.3f}")

    print("-" * 75)

    # Calculate improvement
    orig_p1 = results["Original (hybrid on original text)"]["p_at_1"]
    enrich_p1 = results["Enriched (hybrid on enriched text)"]["p_at_1"]
    improvement = (enrich_p1 - orig_p1) / orig_p1 * 100 if orig_p1 > 0 else 0

    print(f"\nP@1 Change: {orig_p1:.1%} → {enrich_p1:.1%} ({improvement:+.1f}%)")

    # Save results
    output_path = PROJECT_ROOT / "ground_truth/web_enriched_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
