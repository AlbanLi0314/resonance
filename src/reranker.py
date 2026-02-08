"""
LLM Reranker using Gemini
Takes top-k candidates and returns final ranked results with reasons
"""
import json
import re
from typing import List, Dict, Any


class LLMReranker:
    """
    Uses Gemini to rerank candidates and generate recommendation reasons
    
    Usage:
        reranker = LLMReranker(api_key="xxx")
        results = reranker.rerank(query, candidates, top_k=5)
    """
    
    def __init__(self, api_key: str = None, model: str = "gemini-3-flash-preview"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
    def load(self):
        """Initialize Gemini client"""
        import google.generativeai as genai
        
        if not self.api_key:
            import os
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set!")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        print(f"Gemini client initialized with model: {self.model}")
        return self
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank candidates using LLM
        
        Args:
            query: User's search query / need description
            candidates: List of researcher dicts from FAISS recall
            top_k: Number of final results to return
            
        Returns:
            List of dicts with rank, id, name, reason, etc.
        """
        if self.client is None:
            raise RuntimeError("Client not initialized. Call load() first.")
        
        # Build prompt
        prompt = self._build_prompt(query, candidates, top_k)
        
        try:
            # Call Gemini
            response = self.client.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON response
            results = self._parse_response(response_text, candidates)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"LLM reranking failed: {e}")
            print("Falling back to embedding-based ranking...")
            return self._fallback_ranking(candidates, top_k)
    
    def _build_prompt(self, query: str, candidates: List[Dict], top_k: int) -> str:
        """Build the prompt for Gemini"""
        
        # Format candidates (truncate text to avoid token limit)
        candidates_text = ""
        for i, c in enumerate(candidates):
            raw_text = c.get('raw_text', '')[:500]  # Truncate
            candidates_text += f"""
--- Candidate {i+1} ---
ID: {c.get('id', 'unknown')}
Name: {c.get('name', 'Unknown')}
Position: {c.get('position', 'Unknown')}
Department: {c.get('department', 'Unknown')}
Research: {raw_text}
"""
        
        prompt = f"""You are an academic matching assistant. A user wants to find researchers to discuss a topic.

USER'S NEED:
{query}

CANDIDATE RESEARCHERS:
{candidates_text}

TASK:
1. Analyze how well each candidate matches the user's need
2. Select the top {top_k} best matches
3. For each selected researcher, provide a brief reason why they are a good match

OUTPUT FORMAT (JSON only, no markdown):
[
  {{
    "rank": 1,
    "id": "researcher_id",
    "name": "Researcher Name",
    "reason": "Brief explanation of why this person is a good match"
  }},
  ...
]

Important:
- Return ONLY valid JSON, no other text
- Rank by relevance (1 = best match)
- Keep reasons concise (1-2 sentences)
- Consider expertise overlap, methodology match, and potential for productive discussion
"""
        return prompt
    
    def _parse_response(self, response_text: str, candidates: List[Dict]) -> List[Dict]:
        """Parse LLM response into structured results"""
        
        # Clean up response (remove markdown code blocks if present)
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```json?\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        
        # Parse JSON
        try:
            results = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response text: {text[:500]}")
            return self._fallback_ranking(candidates, 5)
        
        # Enrich results with full candidate info
        id_to_candidate = {c['id']: c for c in candidates}
        enriched_results = []
        
        for r in results:
            candidate_id = r.get('id')
            if candidate_id in id_to_candidate:
                candidate = id_to_candidate[candidate_id]
                enriched_results.append({
                    'rank': r.get('rank', len(enriched_results) + 1),
                    'id': candidate_id,
                    'name': candidate.get('name', 'Unknown'),
                    'position': candidate.get('position', 'Unknown'),
                    'department': candidate.get('department', 'Unknown'),
                    'lab': candidate.get('lab', ''),
                    'email': candidate.get('email', ''),
                    'personal_website': candidate.get('personal_website', ''),
                    'reason': r.get('reason', 'Relevant expertise'),
                    'embedding_score': candidate.get('embedding_score', 0)
                })
        
        return enriched_results
    
    def _fallback_ranking(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Fallback: return candidates as-is (already sorted by embedding distance)"""
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
                'reason': 'Matched by research area similarity',
                'embedding_score': c.get('embedding_score', 0)
            })
        return results


# ============================================================
# Quick test (requires API key)
# ============================================================
if __name__ == "__main__":
    import os
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set. Skipping live test.")
        print("Set it with: export GEMINI_API_KEY='your-key'")
    else:
        print("Testing LLMReranker...")
        
        reranker = LLMReranker(api_key=api_key)
        reranker.load()
        
        # Fake candidates
        candidates = [
            {
                "id": "test_001",
                "name": "John Smith",
                "position": "PhD Student",
                "department": "Materials Science",
                "raw_text": "John studies DNA origami and self-assembly of nanomaterials.",
                "email": "js@cornell.edu"
            },
            {
                "id": "test_002",
                "name": "Emily Chen",
                "position": "Professor",
                "department": "Chemical Engineering",
                "raw_text": "Emily researches machine learning for battery materials discovery.",
                "email": "ec@cornell.edu"
            }
        ]
        
        query = "I want to discuss DNA nanostructure folding techniques"
        results = reranker.rerank(query, candidates, top_k=2)
        
        print("\nResults:")
        for r in results:
            print(f"  #{r['rank']}: {r['name']} - {r['reason']}")
        
        print("\nTest completed!")
