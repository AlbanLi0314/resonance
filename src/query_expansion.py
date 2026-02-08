"""
Query Expansion with Gemini 3
==============================
Expands user queries with synonyms, related terms, and academic vocabulary
to improve recall and precision.
"""
import os
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

from google import genai

# Configure Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def expand_query(query: str, model_name: str = "gemini-3-flash-preview") -> str:
    """
    Expand a search query with related academic terms.

    Args:
        query: Original user query
        model_name: Gemini model to use

    Returns:
        Expanded query string
    """
    prompt = f"""You are an academic research expert. Expand this search query with related scientific terms, synonyms, and specific techniques.

Original query: "{query}"

Instructions:
- Add 5-8 highly relevant academic terms
- Include specific techniques, methods, or subfields
- Include common synonyms used in academic papers
- Keep it concise - just the terms, separated by commas
- Do NOT add unrelated terms
- Output ONLY the expanded terms, nothing else

Example:
Query: "battery materials"
Output: lithium-ion batteries, electrode materials, energy storage, solid electrolytes, cathode materials, anode design, electrochemistry

Now expand this query:
Query: "{query}"
Output:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        expanded_terms = response.text.strip()

        # Combine original query with expanded terms
        full_query = f"{query}, {expanded_terms}"
        return full_query
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return query  # Return original on failure


def expand_queries_batch(queries: list, model_name: str = "gemini-3-flash-preview") -> dict:
    """
    Expand multiple queries.

    Args:
        queries: List of query strings
        model_name: Gemini model to use

    Returns:
        Dict mapping original query to expanded query
    """
    results = {}
    for i, q in enumerate(queries):
        expanded = expand_query(q, model_name)
        results[q] = expanded
        if (i + 1) % 10 == 0:
            print(f"  Expanded {i + 1}/{len(queries)} queries")
    return results


if __name__ == "__main__":
    # Test
    test_queries = [
        "quantum materials",
        "3D printing",
        "sustainable energy",
        "biomedical imaging"
    ]

    print("Testing Query Expansion")
    print("=" * 60)

    for q in test_queries:
        expanded = expand_query(q)
        print(f"\nOriginal: {q}")
        print(f"Expanded: {expanded}")
