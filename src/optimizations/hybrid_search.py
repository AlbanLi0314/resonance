"""
Hybrid Search: BM25 + Semantic Fusion
=====================================
Combines keyword-based BM25 search with semantic embedding search.
Often achieves 10-20% improvement over pure semantic search.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import re


class BM25:
    """
    BM25 (Best Matching 25) implementation for keyword search.

    BM25 is a probabilistic retrieval function that ranks documents
    based on query term frequencies and document lengths.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Document length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}  # term -> number of docs containing term
        self.idf = {}  # term -> IDF score
        self.doc_term_freqs = []  # list of {term: freq} for each doc

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove very short tokens and common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def fit(self, corpus: List[str]):
        """
        Fit BM25 on a corpus of documents.

        Args:
            corpus: List of document strings
        """
        self.corpus = corpus
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()

        # Process each document
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies in this document
            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)

            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] += 1

        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # Calculate IDF for each term
        N = len(corpus)
        for term, df in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5))
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25 score for a query against a document.

        Args:
            query: Query string
            doc_idx: Index of document in corpus

        Returns:
            BM25 score
        """
        query_tokens = self._tokenize(query)
        doc_term_freq = self.doc_term_freqs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue

            tf = doc_term_freq.get(term, 0)
            idf = self.idf[term]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_idx, score) tuples sorted by score descending
        """
        scores = []
        for i in range(len(self.corpus)):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridSearcher:
    """
    Hybrid search combining BM25 and semantic search.

    Fusion methods:
    - RRF (Reciprocal Rank Fusion): Simple and effective
    - Linear combination: Weighted sum of normalized scores
    """

    def __init__(self, semantic_weight: float = 0.7, bm25_weight: float = 0.3):
        """
        Initialize hybrid searcher.

        Args:
            semantic_weight: Weight for semantic scores (default: 0.7)
            bm25_weight: Weight for BM25 scores (default: 0.3)
        """
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.bm25 = BM25()
        self.corpus = []
        self.embeddings = None

    def fit(self, documents: List[str], embeddings: np.ndarray):
        """
        Fit the hybrid searcher.

        Args:
            documents: List of document strings (for BM25)
            embeddings: Numpy array of embeddings (for semantic search)
        """
        self.corpus = documents
        self.embeddings = embeddings
        self.bm25.fit(documents)

    def _semantic_search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Semantic search using cosine similarity"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Calculate cosine similarities
        similarities = np.dot(doc_norms, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(int(i), float(similarities[i])) for i in top_indices]

    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return {}

        max_score = max(s for _, s in scores)
        min_score = min(s for _, s in scores)

        if max_score == min_score:
            return {i: 1.0 for i, _ in scores}

        return {i: (s - min_score) / (max_score - min_score) for i, s in scores}

    def _rrf_fusion(self, semantic_results: List[Tuple[int, float]],
                    bm25_results: List[Tuple[int, float]], k: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list
        """
        rrf_scores = {}

        # Add semantic scores
        for rank, (doc_idx, _) in enumerate(semantic_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)

        # Add BM25 scores
        for rank, (doc_idx, _) in enumerate(bm25_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)

        # Sort by RRF score
        results = [(i, s) for i, s in rrf_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _linear_fusion(self, semantic_results: List[Tuple[int, float]],
                       bm25_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Linear combination of normalized scores.

        Combined = semantic_weight * semantic_score + bm25_weight * bm25_score
        """
        semantic_normalized = self._normalize_scores(semantic_results)
        bm25_normalized = self._normalize_scores(bm25_results)

        # Combine scores
        all_docs = set(semantic_normalized.keys()) | set(bm25_normalized.keys())
        combined = {}

        for doc_idx in all_docs:
            sem_score = semantic_normalized.get(doc_idx, 0)
            bm25_score = bm25_normalized.get(doc_idx, 0)
            combined[doc_idx] = self.semantic_weight * sem_score + self.bm25_weight * bm25_score

        # Sort by combined score
        results = [(i, s) for i, s in combined.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def search(self, query: str, query_embedding: np.ndarray,
               top_k: int = 10, fusion_method: str = "rrf") -> List[Tuple[int, float]]:
        """
        Hybrid search.

        Args:
            query: Query string (for BM25)
            query_embedding: Query embedding (for semantic)
            top_k: Number of results to return
            fusion_method: "rrf" or "linear"

        Returns:
            List of (doc_idx, score) tuples
        """
        # Get results from both methods
        semantic_k = min(top_k * 3, len(self.corpus))  # Get more candidates for fusion
        semantic_results = self._semantic_search(query_embedding, semantic_k)
        bm25_results = self.bm25.search(query, semantic_k)

        # Fuse results
        if fusion_method == "rrf":
            fused = self._rrf_fusion(semantic_results, bm25_results)
        else:
            fused = self._linear_fusion(semantic_results, bm25_results)

        return fused[:top_k]


def run_hybrid_experiments(output_file: Path = None) -> Dict:
    """
    Run hybrid search experiments comparing different configurations.

    Returns:
        Experiment results
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.embedding import Specter2Encoder, format_researcher_text
    from src.config import DATA_DIR

    print("Running Hybrid Search Experiments...")

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

    # Prepare texts and embeddings
    print("Preparing documents...")
    documents = [format_researcher_text(r, "optimized") for r in researchers]

    # Create ID to index mapping
    id_to_idx = {r["id"]: i for i, r in enumerate(researchers)}

    # Load encoder and generate embeddings (use CPU to avoid MPS mutex issues)
    print("Loading SPECTER2 encoder (CPU mode for stability)...")
    encoder = Specter2Encoder(force_cpu=True)
    encoder.load()

    print("Generating embeddings...")
    embeddings = encoder.encode_batch(documents, show_progress=True)

    # Initialize hybrid searcher
    print("Initializing hybrid searcher...")
    hybrid = HybridSearcher()
    hybrid.fit(documents, embeddings)

    # Run experiments with different configurations
    configs = [
        ("semantic_only", 1.0, 0.0),
        ("bm25_only", 0.0, 1.0),
        ("hybrid_70_30", 0.7, 0.3),
        ("hybrid_50_50", 0.5, 0.5),
        ("hybrid_30_70", 0.3, 0.7),
    ]

    results = {
        "num_researchers": len(researchers),
        "num_queries": len(test_queries),
        "experiments": []
    }

    for config_name, sem_weight, bm25_weight in configs:
        print(f"\nTesting {config_name} (semantic={sem_weight}, bm25={bm25_weight})...")

        hybrid.semantic_weight = sem_weight
        hybrid.bm25_weight = bm25_weight

        mrr_sum = 0
        p_at_1 = 0
        p_at_3 = 0
        p_at_5 = 0

        for q in test_queries:
            query_text = q.get("query", "")
            expected_id = q.get("expected_id", "")

            if not expected_id or expected_id not in id_to_idx:
                continue

            expected_idx = id_to_idx[expected_id]

            # Get query embedding
            query_emb = encoder.encode(query_text)

            # Search
            if sem_weight == 1.0 and bm25_weight == 0.0:
                # Semantic only
                search_results = hybrid._semantic_search(query_emb, 10)
            elif sem_weight == 0.0 and bm25_weight == 1.0:
                # BM25 only
                search_results = hybrid.bm25.search(query_text, 10)
            else:
                # Hybrid
                search_results = hybrid.search(query_text, query_emb, 10, "rrf")

            result_indices = [idx for idx, _ in search_results]

            # Calculate metrics
            if expected_idx in result_indices:
                rank = result_indices.index(expected_idx) + 1
                mrr_sum += 1.0 / rank
                if rank == 1:
                    p_at_1 += 1
                if rank <= 3:
                    p_at_3 += 1
                if rank <= 5:
                    p_at_5 += 1

        num_queries = len(test_queries)
        exp_result = {
            "config": config_name,
            "semantic_weight": sem_weight,
            "bm25_weight": bm25_weight,
            "mrr": mrr_sum / num_queries if num_queries > 0 else 0,
            "precision_at_1": p_at_1 / num_queries if num_queries > 0 else 0,
            "precision_at_3": p_at_3 / num_queries if num_queries > 0 else 0,
            "precision_at_5": p_at_5 / num_queries if num_queries > 0 else 0,
        }

        print(f"  MRR: {exp_result['mrr']:.4f}, P@1: {exp_result['precision_at_1']:.4f}")
        results["experiments"].append(exp_result)

    # Find best configuration
    best = max(results["experiments"], key=lambda x: x["mrr"])
    results["best_config"] = best["config"]
    results["best_mrr"] = best["mrr"]

    # Calculate improvement over semantic-only baseline
    semantic_baseline = next((e for e in results["experiments"] if e["config"] == "semantic_only"), None)
    if semantic_baseline and best["config"] != "semantic_only":
        improvement = (best["mrr"] - semantic_baseline["mrr"]) / semantic_baseline["mrr"] * 100
        results["improvement_over_baseline"] = f"+{improvement:.1f}%"

    print(f"\nBest configuration: {results['best_config']} (MRR: {results['best_mrr']:.4f})")

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    from pathlib import Path
    output = Path(__file__).parent.parent.parent / "data" / "overnight_results" / "hybrid_search_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_hybrid_experiments(output)
    print(f"\nFinal result: {json.dumps(result, indent=2)}")
