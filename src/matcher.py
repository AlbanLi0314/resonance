"""
Academic Matcher - Main Class
Orchestrates the two-stage matching pipeline:
  Stage 1: FAISS embedding recall (fast) OR Hybrid search (dense + BM25)
  Stage 2: Cross-encoder reranking (fast, accurate) OR LLM reranking (slow, with reasons)

Optimized Configuration (based on evaluation):
  - Hybrid search with RRF fusion: +15-30% recall over single methods
  - Cross-encoder reranking: +5-15% P@1 over LLM reranking
"""
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Literal

from .config import (
    RESEARCHERS_JSON,
    RESEARCHERS_WITH_EMB_PKL,
    FAISS_INDEX_FILE,
    RECALL_TOP_K,
    FINAL_TOP_K,
    GEMINI_API_KEY
)
from .embedding import Specter2Encoder
from .indexer import FaissIndexer
from .reranker import LLMReranker

# New optimized modules
from .hybrid_search import HybridSearcher
from .cross_encoder_reranker import CrossEncoderReranker


class AcademicMatcher:
    """
    Main class for academic researcher matching

    Supports multiple search and reranking strategies:
    - Search: "dense" (FAISS only) or "hybrid" (FAISS + BM25 with RRF)
    - Rerank: "none", "cross_encoder" (fast, accurate), or "llm" (Gemini)

    Usage:
        # Recommended configuration (based on evaluation)
        matcher = AcademicMatcher()
        matcher.initialize(search_mode="hybrid", rerank_mode="cross_encoder")
        results = matcher.search("DNA origami folding optimization")

        # Legacy configuration
        matcher.initialize(skip_rerank=True)  # Dense only, no reranking
    """

    def __init__(self):
        self.encoder: Optional[Specter2Encoder] = None
        self.indexer: Optional[FaissIndexer] = None
        self.reranker = None  # Can be LLMReranker or CrossEncoderReranker
        self.hybrid_searcher: Optional[HybridSearcher] = None
        self.researchers: List[Dict] = []
        self.initialized = False
        self.search_mode: str = "dense"
        self.rerank_mode: str = "none"

    def initialize(self,
                   researchers_pkl: str = None,
                   faiss_index: str = None,
                   gemini_api_key: str = None,
                   skip_rerank: bool = False,
                   search_mode: Literal["dense", "hybrid"] = "dense",
                   rerank_mode: Literal["none", "cross_encoder", "llm"] = None):
        """
        Load all components for search.

        Args:
            researchers_pkl: Path to processed researchers pickle file
            faiss_index: Path to FAISS index file
            gemini_api_key: Gemini API key (for LLM reranking)
            skip_rerank: Legacy param - if True, sets rerank_mode="none"
            search_mode: "dense" (FAISS only) or "hybrid" (FAISS + BM25)
            rerank_mode: "none", "cross_encoder", or "llm"
        """
        researchers_pkl = researchers_pkl or str(RESEARCHERS_WITH_EMB_PKL)
        faiss_index = faiss_index or str(FAISS_INDEX_FILE)
        gemini_api_key = gemini_api_key or GEMINI_API_KEY

        # Handle legacy skip_rerank parameter
        if rerank_mode is None:
            rerank_mode = "none" if skip_rerank else "none"  # Default to no reranking

        self.search_mode = search_mode
        self.rerank_mode = rerank_mode

        # Count steps
        steps = 3
        if search_mode == "hybrid":
            steps += 1
        if rerank_mode != "none":
            steps += 1

        current_step = 0
        print("Initializing Academic Matcher...")
        print(f"  Search mode: {search_mode}")
        print(f"  Rerank mode: {rerank_mode}")

        # Load encoder
        current_step += 1
        print(f"{current_step}/{steps} Loading SPECTER2 encoder...")
        self.encoder = Specter2Encoder()
        self.encoder.load()

        # Load researchers data
        current_step += 1
        print(f"{current_step}/{steps} Loading researcher data...")
        with open(researchers_pkl, 'rb') as f:
            self.researchers = pickle.load(f)
        print(f"     Loaded {len(self.researchers)} researchers")

        # Load FAISS index
        current_step += 1
        print(f"{current_step}/{steps} Loading FAISS index...")
        self.indexer = FaissIndexer()
        self.indexer.load(faiss_index)

        # Initialize hybrid search if requested
        if search_mode == "hybrid":
            current_step += 1
            print(f"{current_step}/{steps} Building BM25 index for hybrid search...")
            self.hybrid_searcher = HybridSearcher(
                self.researchers,
                self.indexer,
                self.encoder
            )
            self.hybrid_searcher.build_bm25_index()

        # Initialize reranker
        if rerank_mode == "cross_encoder":
            current_step += 1
            print(f"{current_step}/{steps} Loading cross-encoder reranker...")
            self.reranker = CrossEncoderReranker(preset="balanced")
            self.reranker.load()
        elif rerank_mode == "llm":
            current_step += 1
            print(f"{current_step}/{steps} Initializing LLM reranker...")
            self.reranker = LLMReranker(api_key=gemini_api_key)
            self.reranker.load()
        else:
            print("     (Skipping reranking)")

        self.initialized = True
        print("Academic Matcher ready!")
        
    def search(self, query: str,
               recall_k: int = None,
               final_k: int = None,
               skip_rerank: bool = False,
               search_mode: str = None,
               alpha: float = 0.7) -> List[Dict]:
        """
        Execute two-stage search with configurable strategies.

        Args:
            query: User's search query / need description
            recall_k: Number of candidates to recall in Stage 1
            final_k: Number of final results to return
            skip_rerank: If True, skip reranking (legacy param)
            search_mode: Override instance search_mode ("dense" or "hybrid")
            alpha: For hybrid search, weight for dense vs sparse (0.7 = 70% dense)

        Returns:
            List of matched researchers with ranks and reasons
        """
        if not self.initialized:
            raise RuntimeError("Matcher not initialized. Call initialize() first.")

        recall_k = recall_k or RECALL_TOP_K
        final_k = final_k or FINAL_TOP_K
        search_mode = search_mode or self.search_mode

        # Determine if we should rerank
        should_rerank = (self.rerank_mode != "none") and (not skip_rerank) and (self.reranker is not None)

        # Stage 1: Retrieval
        if search_mode == "hybrid" and self.hybrid_searcher is not None:
            print(f"Stage 1: Hybrid search (dense + BM25 with RRF)...")
            candidates = self.hybrid_searcher.search(
                query,
                top_k=recall_k,
                alpha=alpha,
                return_details=True
            )
            print(f"         Found {len(candidates)} candidates")
        else:
            print(f"Stage 1: Dense retrieval (FAISS)...")
            candidates = self._embedding_recall(query, recall_k)
            print(f"         Found {len(candidates)} candidates")

        # Stage 2: Reranking (if enabled)
        if should_rerank:
            reranker_name = self.rerank_mode
            print(f"Stage 2: Reranking with {reranker_name}...")
            results = self.reranker.rerank(query, candidates, final_k)
            print(f"         Returned {len(results)} final results")
            return results
        else:
            # Return retrieval results directly
            return self._format_embedding_results(candidates, final_k)
    
    def _embedding_recall(self, query: str, top_k: int) -> List[Dict]:
        """Stage 1: FAISS similarity search"""
        
        # Encode query
        query_embedding = self.encoder.encode(query)
        
        # Search
        distances, indices = self.indexer.search(query_embedding, top_k)
        
        # Build candidate list
        candidates = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.researchers):
                continue
            
            researcher = self.researchers[idx].copy()
            # Convert distance to similarity score (0-1, higher is better)
            # L2 distance: smaller is better, so we invert
            researcher['embedding_score'] = float(1 / (1 + dist))
            researcher['embedding_distance'] = float(dist)
            candidates.append(researcher)
        
        return candidates
    
    def _format_embedding_results(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Format embedding results (when skipping rerank)"""
        results = []
        for i, c in enumerate(candidates[:top_k]):
            results.append({
                'rank': i + 1,
                'id': c.get('id', 'unknown'),
                'name': c.get('name', 'Unknown'),
                'position': c.get('position', 'Unknown'),
                'department': c.get('department', 'Unknown'),
                'lab': c.get('lab', ''),
                'email': c.get('email', ''),
                'personal_website': c.get('personal_website', ''),
                'research_interests': c.get('research_interests', ''),
                'raw_text': c.get('raw_text', ''),
                'embedding_score': c.get('embedding_score', 0),
                'reason': 'Matched by research area similarity (embedding only)'
            })
        return results


def print_results(results: List[Dict]):
    """Pretty print search results"""
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"\n#{r['rank']}: {r['name']}")
        print(f"    Position: {r['position']}")
        print(f"    Department: {r['department']}")
        if r.get('lab'):
            print(f"    Lab: {r['lab']}")
        print(f"    Email: {r['email']}")
        if r.get('personal_website'):
            print(f"    Website: {r['personal_website']}")
        if r.get('embedding_score'):
            print(f"    Score: {r['embedding_score']:.3f}")
        print(f"    Why: {r['reason']}")
    
    print("\n" + "=" * 60)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    # This requires pre-built index
    # Run step1_build_index.py first
    
    print("Testing AcademicMatcher...")
    print("Make sure you've run step1_build_index.py first!")
    
    matcher = AcademicMatcher()
    matcher.initialize()
    
    query = "I want to discuss DNA origami folding and assembly techniques"
    results = matcher.search(query)
    
    print_results(results)
