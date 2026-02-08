"""
Gemini-based Reranking
======================
Uses Gemini to rerank search results based on semantic relevance.
Unlike generic cross-encoders, Gemini understands academic context.
"""
import os
import json
from typing import List, Dict
from pathlib import Path

# Load .env
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

import google.generativeai as genai

# Configure
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def rerank_with_gemini(
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
    model_name: str = "gemini-3-flash-preview"
) -> List[Dict]:
    """
    Rerank candidates using Gemini's understanding of academic relevance.

    Args:
        query: User's search query
        candidates: List of researcher dicts with name, department, research_interests, raw_text
        top_k: Number of results to return
        model_name: Gemini model to use

    Returns:
        Reranked list of researchers (top_k)
    """
    if not candidates:
        return []

    model = genai.GenerativeModel(model_name)

    # Build candidate descriptions
    candidate_texts = []
    for i, c in enumerate(candidates):
        name = c.get("name", "Unknown")
        dept = c.get("department", "")
        interests = c.get("research_interests", "")[:500]
        raw = c.get("raw_text", "")[:300]

        desc = f"[{i}] {name}"
        if dept:
            desc += f" ({dept})"
        if interests:
            desc += f"\n    Research: {interests[:200]}"
        elif raw:
            desc += f"\n    Info: {raw[:200]}"

        candidate_texts.append(desc)

    candidates_str = "\n".join(candidate_texts)

    prompt = f"""You are an expert at matching research queries to academic researchers.

Query: "{query}"

Candidate Researchers:
{candidates_str}

Task: Rank these researchers by how well their research matches the query.

Instructions:
- Consider research area overlap, not just keyword matching
- A researcher working on related fundamental science is relevant
- Return ONLY the indices in order of relevance, separated by commas
- Return exactly {min(top_k, len(candidates))} indices
- Most relevant first

Example output format: 3, 1, 4, 0, 2

Your ranking (indices only):"""

    try:
        response = model.generate_content(prompt)
        ranking_text = response.text.strip()

        # Parse indices
        indices = []
        for part in ranking_text.replace("\n", ",").split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(candidates) and idx not in indices:
                    indices.append(idx)

        # Fill in missing indices if needed
        for i in range(len(candidates)):
            if i not in indices:
                indices.append(i)

        # Reorder candidates
        reranked = [candidates[i] for i in indices[:top_k]]

        # Add rerank position
        for i, r in enumerate(reranked):
            r["rerank_position"] = i + 1

        return reranked

    except Exception as e:
        print(f"Gemini reranking failed: {e}")
        return candidates[:top_k]  # Return original order on failure


def rerank_batch(
    queries_and_candidates: List[tuple],
    top_k: int = 5,
    model_name: str = "gemini-3-flash-preview"
) -> List[List[Dict]]:
    """
    Rerank multiple query-candidates pairs.

    Args:
        queries_and_candidates: List of (query, candidates_list) tuples
        top_k: Number of results per query
        model_name: Gemini model

    Returns:
        List of reranked candidate lists
    """
    results = []
    for i, (query, candidates) in enumerate(queries_and_candidates):
        reranked = rerank_with_gemini(query, candidates, top_k, model_name)
        results.append(reranked)
        if (i + 1) % 10 == 0:
            print(f"  Reranked {i + 1}/{len(queries_and_candidates)} queries")
    return results


if __name__ == "__main__":
    # Test
    test_query = "quantum computing materials"
    test_candidates = [
        {
            "name": "Dr. Alice Smith",
            "department": "Chemistry",
            "research_interests": "Organic synthesis, polymer chemistry, drug delivery"
        },
        {
            "name": "Dr. Bob Johnson",
            "department": "Physics",
            "research_interests": "Quantum mechanics, superconductivity, topological materials"
        },
        {
            "name": "Dr. Carol Williams",
            "department": "Materials Science",
            "research_interests": "Battery materials, energy storage, electrochemistry"
        },
        {
            "name": "Dr. David Brown",
            "department": "Computer Science",
            "research_interests": "Machine learning, data mining, natural language processing"
        },
        {
            "name": "Dr. Eve Davis",
            "department": "Physics",
            "research_interests": "Quantum information, qubit design, cryogenic systems"
        }
    ]

    print("Testing Gemini Reranking")
    print("=" * 60)
    print(f"Query: {test_query}")
    print("\nOriginal order:")
    for i, c in enumerate(test_candidates):
        print(f"  {i}. {c['name']} - {c['research_interests'][:50]}...")

    reranked = rerank_with_gemini(test_query, test_candidates, top_k=5)

    print("\nReranked order:")
    for i, c in enumerate(reranked):
        print(f"  {i + 1}. {c['name']} - {c['research_interests'][:50]}...")
