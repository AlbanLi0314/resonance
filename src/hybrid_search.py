"""
Hybrid Search Module
====================
Combines BM25 (sparse/keyword) with Dense (semantic) retrieval
using Reciprocal Rank Fusion (RRF) for optimal results.

Research shows hybrid search improves recall 15-30% over single methods.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class SearchResult:
    """Single search result with scores"""
    researcher: Dict
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0


class BM25Index:
    """
    BM25 sparse retrieval index for keyword matching.

    BM25 excels at exact keyword matching while dense retrieval
    excels at semantic similarity. Combining them gives best results.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = {}  # term -> number of docs containing term
        self.idf = {}  # term -> IDF score
        self.doc_term_freqs = []  # list of {term: freq} for each doc
        self.N = 0  # total number of documents

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    def build(self, documents: List[str]):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
        """
        self.corpus = documents
        self.N = len(documents)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = {}

        # First pass: compute document frequencies
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies in this document
            term_freqs = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            self.doc_term_freqs.append(term_freqs)

            # Update document frequencies (count unique terms per doc)
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Compute average document length
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Compute IDF for all terms
        for term, df in self.doc_freqs.items():
            # IDF with smoothing to avoid division by zero
            self.idf[term] = np.log((self.N - df + 0.5) / (df + 0.5) + 1)

        print(f"BM25 index built: {self.N} documents, {len(self.doc_freqs)} unique terms")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.N)

        for token in query_tokens:
            if token not in self.idf:
                continue

            idf = self.idf[token]

            for doc_idx in range(self.N):
                tf = self.doc_term_freqs[doc_idx].get(token, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[doc_idx]

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_idx] += idf * numerator / denominator

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return (index, score) pairs
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return results


class HybridSearcher:
    """
    Hybrid search combining BM25 (sparse) and FAISS (dense) retrieval
    with Reciprocal Rank Fusion (RRF).

    RRF is score-agnostic - it only uses ranking positions, making it
    robust to different score scales between methods.

    Usage:
        searcher = HybridSearcher(researchers, faiss_indexer, encoder)
        results = searcher.search(query, top_k=10)
    """

    def __init__(self,
                 researchers: List[Dict],
                 faiss_indexer,
                 encoder,
                 rrf_k: int = 60):
        """
        Initialize hybrid searcher.

        Args:
            researchers: List of researcher dicts with raw_text
            faiss_indexer: FAISS indexer for dense retrieval
            encoder: Text encoder (SPECTER2)
            rrf_k: RRF constant (default 60, as per original paper)
        """
        self.researchers = researchers
        self.faiss_indexer = faiss_indexer
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.bm25_index = None

    def build_bm25_index(self, text_field: str = "raw_text"):
        """
        Build BM25 index from researcher profiles.

        Args:
            text_field: Field to use for text (default "raw_text")
        """
        documents = [r.get(text_field, "") for r in self.researchers]
        self.bm25_index = BM25Index()
        self.bm25_index.build(documents)

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense (semantic) search using FAISS"""
        query_embedding = self.encoder.encode(query)
        distances, indices = self.faiss_indexer.search(query_embedding, top_k)

        # Return (index, distance) pairs
        results = [(int(idx), float(dist)) for idx, dist in zip(indices, distances) if idx >= 0]
        return results

    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Sparse (BM25) search"""
        if self.bm25_index is None:
            self.build_bm25_index()
        return self.bm25_index.search(query, top_k)

    def _reciprocal_rank_fusion(self,
                                 dense_results: List[Tuple[int, float]],
                                 sparse_results: List[Tuple[int, float]],
                                 alpha: float = 0.5) -> List[Tuple[int, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum over retrievers of: 1 / (k + rank)

        This is score-agnostic and robust to different score scales.

        Args:
            dense_results: Results from dense retrieval [(idx, score), ...]
            sparse_results: Results from sparse retrieval [(idx, score), ...]
            alpha: Weight for dense vs sparse (0.5 = equal, higher = more dense)

        Returns:
            Fused results sorted by RRF score
        """
        rrf_scores = {}

        # Add dense retrieval contribution
        for rank, (doc_idx, _) in enumerate(dense_results):
            score = alpha / (self.rrf_k + rank + 1)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + score

        # Add sparse retrieval contribution
        for rank, (doc_idx, _) in enumerate(sparse_results):
            score = (1 - alpha) / (self.rrf_k + rank + 1)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + score

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])

        return sorted_results

    def search(self,
               query: str,
               top_k: int = 20,
               dense_k: int = 50,
               sparse_k: int = 50,
               alpha: float = 0.7,
               return_details: bool = False) -> List[Dict]:
        """
        Perform hybrid search combining dense and sparse retrieval.

        Args:
            query: Search query
            top_k: Number of final results to return
            dense_k: Number of candidates from dense retrieval
            sparse_k: Number of candidates from sparse retrieval
            alpha: Weight for dense vs sparse (0.7 = 70% dense, 30% sparse)
            return_details: Include ranking details in results

        Returns:
            List of researcher dicts with scores
        """
        # Get results from both retrievers
        dense_results = self._dense_search(query, dense_k)
        sparse_results = self._sparse_search(query, sparse_k)

        # Fuse with RRF
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results, alpha)

        # Build result list
        results = []

        # Create lookup for ranks
        dense_ranks = {idx: rank for rank, (idx, _) in enumerate(dense_results)}
        sparse_ranks = {idx: rank for rank, (idx, _) in enumerate(sparse_results)}

        for doc_idx, rrf_score in fused_results[:top_k]:
            if doc_idx < 0 or doc_idx >= len(self.researchers):
                continue

            researcher = self.researchers[doc_idx].copy()
            researcher['rrf_score'] = rrf_score

            # Convert RRF score to a 0-1 similarity score for display
            # Max possible RRF score is alpha/(k+1) + (1-alpha)/(k+1) = 1/(k+1)
            max_rrf = 1 / (self.rrf_k + 1)
            researcher['embedding_score'] = min(rrf_score / max_rrf, 1.0)

            if return_details:
                researcher['dense_rank'] = dense_ranks.get(doc_idx, None)
                researcher['sparse_rank'] = sparse_ranks.get(doc_idx, None)

            results.append(researcher)

        return results


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Testing BM25 Index...")

    # Test documents
    docs = [
        "DNA origami self-assembly nanomaterials nanotechnology",
        "Machine learning deep learning neural networks AI",
        "Battery materials lithium ion energy storage",
        "Polymer synthesis RAFT polymerization chemistry",
        "DNA nanotechnology nucleic acid structures"
    ]

    bm25 = BM25Index()
    bm25.build(docs)

    # Test queries
    queries = [
        "DNA nanotechnology",
        "machine learning",
        "polymer chemistry"
    ]

    for query in queries:
        results = bm25.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for idx, score in results:
            print(f"  {idx}: {docs[idx][:50]}... (score: {score:.3f})")

    print("\nBM25 test completed!")


# ============================================================
# Simplified HybridSearch wrapper for API use
# ============================================================
class HybridSearch:
    """
    Simplified hybrid search interface for API usage.

    Automatically initializes BM25 index and uses provided encoder.

    Usage:
        hybrid = HybridSearch(researchers, encoder)
        results = hybrid.search("quantum computing", top_k=5)
    """

    def __init__(self, researchers: List[Dict], encoder):
        """
        Initialize hybrid search.

        Args:
            researchers: List of researcher dicts (must have 'embedding' field)
            encoder: SPECTER2 encoder instance
        """
        self.researchers = researchers
        self.encoder = encoder

        # Build FAISS index from embeddings
        from src.indexer import FaissIndexer
        embeddings = np.array([r['embedding'] for r in researchers])
        self.faiss_indexer = FaissIndexer(dimension=embeddings.shape[1])
        self.faiss_indexer.build(embeddings)

        # Initialize hybrid searcher
        self.searcher = HybridSearcher(
            researchers=researchers,
            faiss_indexer=self.faiss_indexer,
            encoder=encoder,
            rrf_k=60
        )

        # Build BM25 index
        self.searcher.build_bm25_index(text_field="raw_text")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for researchers matching the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of researcher dicts with scores
        """
        results = self.searcher.search(
            query=query,
            top_k=top_k,
            dense_k=20,
            sparse_k=20,
            alpha=0.7,  # 70% dense, 30% sparse
            return_details=False
        )

        # Add score field for API response
        for r in results:
            r['score'] = r.get('embedding_score', r.get('rrf_score', 0))

        return results
