"""
Cross-Encoder Reranker
======================
Uses a dedicated cross-encoder model for reranking search results.

Cross-encoders jointly process query and document together,
enabling deeper semantic understanding than bi-encoders.
They are 10-100x faster than LLM APIs and specifically trained for retrieval.

Pre-trained on MS-MARCO, a large-scale information retrieval dataset.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RerankedResult:
    """Single reranked result"""
    rank: int
    researcher: Dict
    score: float
    reason: str


class CrossEncoderReranker:
    """
    Reranker using cross-encoder from sentence-transformers.

    Uses 'cross-encoder/ms-marco-MiniLM-L6-v2' by default, which is:
    - Trained on MS-MARCO (500k real search queries)
    - Fast (~100 pairs/sec on CPU)
    - Optimized for passage retrieval

    Usage:
        reranker = CrossEncoderReranker()
        reranker.load()
        results = reranker.rerank(query, candidates, top_k=5)
    """

    # Available cross-encoder models (speed vs accuracy tradeoff)
    MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-2-v2",      # Fastest, lower accuracy
        "balanced": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Good balance (default)
        "accurate": "cross-encoder/ms-marco-MiniLM-L-12-v2", # Higher accuracy, slower
    }

    def __init__(self,
                 model_name: str = None,
                 preset: str = "balanced",
                 max_length: int = 512):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Specific model name (overrides preset)
            preset: One of "fast", "balanced", "accurate"
            max_length: Maximum sequence length
        """
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.MODELS.get(preset, self.MODELS["balanced"])

        self.max_length = max_length
        self.model = None

    def load(self):
        """Load the cross-encoder model"""
        from sentence_transformers import CrossEncoder

        print(f"Loading cross-encoder: {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            max_length=self.max_length
        )
        print("Cross-encoder loaded successfully!")
        return self

    def _format_document(self, researcher: Dict) -> str:
        """
        Format researcher profile as document for reranking.

        Prioritizes research-relevant information.
        """
        parts = []

        # Name and position
        name = researcher.get('name', 'Unknown')
        position = researcher.get('position', '')
        department = researcher.get('department', '')
        parts.append(f"{name}, {position} at {department}")

        # Research interests (highest priority)
        if researcher.get('research_interests'):
            parts.append(researcher['research_interests'][:500])

        # Raw text as fallback
        elif researcher.get('raw_text'):
            # Take first 500 chars, try to end at sentence
            text = researcher['raw_text'][:500]
            # Try to end at a sentence boundary
            last_period = text.rfind('.')
            if last_period > 200:
                text = text[:last_period + 1]
            parts.append(text)

        # Biography
        if researcher.get('biography'):
            bio = researcher['biography'][:300]
            parts.append(bio)

        return ' '.join(parts)

    def _generate_reason(self, query: str, researcher: Dict, score: float) -> str:
        """
        Generate a brief explanation for why this researcher matches.

        Uses simple heuristics - could be enhanced with LLM for better explanations.
        """
        name = researcher.get('name', 'Unknown')
        department = researcher.get('department', 'their department')

        if score > 0.8:
            return f"Highly relevant expertise in {department.lower()}"
        elif score > 0.5:
            return f"Strong match - works on related topics in {department.lower()}"
        elif score > 0.2:
            return f"Relevant background in {department.lower()}"
        else:
            return f"Potential match based on research area"

    def rerank(self,
               query: str,
               candidates: List[Dict],
               top_k: int = 5,
               return_scores: bool = True) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: User's search query
            candidates: List of researcher dicts from retrieval
            top_k: Number of results to return
            return_scores: Include cross-encoder scores in results

        Returns:
            List of reranked researcher dicts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not candidates:
            return []

        # Format query-document pairs
        pairs = []
        for c in candidates:
            doc = self._format_document(c)
            pairs.append([query, doc])

        # Score all pairs
        scores = self.model.predict(pairs)

        # Normalize scores to 0-1 range (cross-encoder scores can be negative)
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores) * 0.5

        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]

        # Build result list
        results = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            candidate = candidates[idx].copy()
            score = float(scores[idx])
            norm_score = float(normalized_scores[idx])

            result = {
                'rank': rank + 1,
                'id': candidate.get('id', 'unknown'),
                'name': candidate.get('name', 'Unknown'),
                'position': candidate.get('position', 'Unknown'),
                'department': candidate.get('department', 'Unknown'),
                'lab': candidate.get('lab', ''),
                'email': candidate.get('email', ''),
                'personal_website': candidate.get('personal_website', ''),
                'research_interests': candidate.get('research_interests', ''),
                'raw_text': candidate.get('raw_text', ''),
                'reason': self._generate_reason(query, candidate, norm_score),
            }

            if return_scores:
                result['cross_encoder_score'] = score
                result['embedding_score'] = norm_score

            results.append(result)

        return results

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Args:
            query: Search query
            document: Document text

        Returns:
            Relevance score
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        score = self.model.predict([[query, document]])
        return float(score[0])


class HybridReranker:
    """
    Combines cross-encoder scores with original retrieval scores.

    Useful when you want to preserve some signal from the initial retrieval
    while benefiting from cross-encoder reranking.
    """

    def __init__(self,
                 cross_encoder: CrossEncoderReranker,
                 alpha: float = 0.7):
        """
        Initialize hybrid reranker.

        Args:
            cross_encoder: CrossEncoderReranker instance
            alpha: Weight for cross-encoder score (1-alpha for original score)
        """
        self.cross_encoder = cross_encoder
        self.alpha = alpha

    def rerank(self,
               query: str,
               candidates: List[Dict],
               top_k: int = 5) -> List[Dict]:
        """
        Rerank using weighted combination of scores.

        Args:
            query: Search query
            candidates: Candidates with 'embedding_score' or 'rrf_score'
            top_k: Number of results

        Returns:
            Reranked results
        """
        if not candidates:
            return []

        # Get cross-encoder scores
        pairs = [[query, self.cross_encoder._format_document(c)] for c in candidates]
        ce_scores = self.cross_encoder.model.predict(pairs)

        # Normalize cross-encoder scores to 0-1
        min_ce = float(np.min(ce_scores))
        max_ce = float(np.max(ce_scores))
        if max_ce > min_ce:
            norm_ce = (ce_scores - min_ce) / (max_ce - min_ce)
        else:
            norm_ce = np.ones_like(ce_scores) * 0.5

        # Get original scores and normalize
        orig_scores = np.array([
            c.get('rrf_score', c.get('embedding_score', 0))
            for c in candidates
        ])
        min_orig = float(np.min(orig_scores))
        max_orig = float(np.max(orig_scores))
        if max_orig > min_orig:
            norm_orig = (orig_scores - min_orig) / (max_orig - min_orig)
        else:
            norm_orig = np.ones_like(orig_scores) * 0.5

        # Combine scores
        combined = self.alpha * norm_ce + (1 - self.alpha) * norm_orig

        # Sort and build results
        ranked_indices = np.argsort(combined)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k]):
            candidate = candidates[idx].copy()
            result = {
                'rank': rank + 1,
                'id': candidate.get('id', 'unknown'),
                'name': candidate.get('name', 'Unknown'),
                'position': candidate.get('position', 'Unknown'),
                'department': candidate.get('department', 'Unknown'),
                'lab': candidate.get('lab', ''),
                'email': candidate.get('email', ''),
                'personal_website': candidate.get('personal_website', ''),
                'research_interests': candidate.get('research_interests', ''),
                'raw_text': candidate.get('raw_text', ''),
                'reason': self.cross_encoder._generate_reason(
                    query, candidate, float(norm_ce[idx])
                ),
                'embedding_score': float(combined[idx]),
                'cross_encoder_score': float(ce_scores[idx]),
            }
            results.append(result)

        return results


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Testing CrossEncoderReranker...")

    reranker = CrossEncoderReranker(preset="balanced")
    reranker.load()

    # Test candidates
    candidates = [
        {
            "id": "test_001",
            "name": "John Smith",
            "position": "PhD Student",
            "department": "Materials Science",
            "raw_text": "John studies DNA origami and self-assembly of nanomaterials. "
                       "His research focuses on programmable DNA nanostructures for drug delivery.",
            "email": "js@cornell.edu"
        },
        {
            "id": "test_002",
            "name": "Emily Chen",
            "position": "Professor",
            "department": "Chemical Engineering",
            "raw_text": "Emily researches machine learning for battery materials discovery. "
                       "She uses DFT calculations and neural networks to predict new materials.",
            "email": "ec@cornell.edu"
        },
        {
            "id": "test_003",
            "name": "Michael Wang",
            "position": "Associate Professor",
            "department": "Biomedical Engineering",
            "raw_text": "Michael develops nucleic acid nanostructures for targeted drug delivery. "
                       "His lab works on RNA and DNA origami for therapeutic applications.",
            "email": "mw@cornell.edu"
        }
    ]

    query = "DNA nanostructure folding techniques"

    print(f"\nQuery: '{query}'")
    print("-" * 50)

    results = reranker.rerank(query, candidates, top_k=3)

    for r in results:
        print(f"#{r['rank']}: {r['name']} (score: {r['cross_encoder_score']:.3f})")
        print(f"    {r['reason']}")
        print()

    print("Test completed!")
