#!/usr/bin/env python3
"""
Query Intent Classification with Gemini
========================================
Classifies user queries to optimize search strategy.
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional

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

# Configure Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def classify_query_intent(
    query: str,
    model_name: str = "gemini-3-flash-preview"
) -> Dict:
    """
    Classify the intent behind a research query.

    Returns:
        Dict with:
        - intent: "find_advisor" | "find_collaborator" | "find_expert" | "explore_field" | "specific_technique"
        - confidence: 0-100
        - search_hints: suggestions for optimizing search
        - refined_query: cleaned/refined version of the query
    """
    prompt = f"""Classify this academic research query to optimize search.

Query: "{query}"

Classify into ONE of these intents:
- "find_advisor": Looking for a PhD/postdoc advisor
- "find_collaborator": Looking for research collaboration
- "find_expert": Looking for someone with specific expertise
- "explore_field": Exploring who works in a broad field
- "specific_technique": Looking for specific equipment/technique expertise

Return JSON:
{{
    "intent": "<intent type>",
    "confidence": <0-100>,
    "search_hints": {{
        "prioritize_faculty": true/false,
        "prioritize_publications": true/false,
        "prioritize_equipment": true/false,
        "field_keywords": ["keyword1", "keyword2"]
    }},
    "refined_query": "<cleaner version of the query for search>"
}}

JSON response:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        result_text = response.text.strip()

        # Parse JSON
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "{" in result_text:
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            result_text = result_text[json_start:json_end]

        data = json.loads(result_text)

        return {
            "intent": data.get("intent", "find_expert"),
            "confidence": min(100, max(0, int(data.get("confidence", 70)))),
            "search_hints": data.get("search_hints", {}),
            "refined_query": data.get("refined_query", query)
        }

    except Exception as e:
        return {
            "intent": "find_expert",
            "confidence": 50,
            "search_hints": {},
            "refined_query": query,
            "error": str(e)
        }


def get_search_strategy(intent_result: Dict) -> Dict:
    """
    Convert intent classification to search strategy parameters.

    Returns:
        Dict with search parameters to apply
    """
    intent = intent_result.get("intent", "find_expert")
    hints = intent_result.get("search_hints", {})

    # Default strategy
    strategy = {
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "filter_faculty_only": False,
        "boost_publications": False,
        "extra_keywords": []
    }

    # Adjust based on intent
    if intent == "find_advisor":
        strategy["filter_faculty_only"] = True
        strategy["dense_weight"] = 0.6
        strategy["sparse_weight"] = 0.4

    elif intent == "find_collaborator":
        strategy["dense_weight"] = 0.8
        strategy["sparse_weight"] = 0.2
        strategy["boost_publications"] = True

    elif intent == "specific_technique":
        strategy["dense_weight"] = 0.5
        strategy["sparse_weight"] = 0.5

    elif intent == "explore_field":
        strategy["dense_weight"] = 0.8
        strategy["sparse_weight"] = 0.2

    # Add field keywords from hints
    if hints.get("field_keywords"):
        strategy["extra_keywords"] = hints["field_keywords"]

    return strategy


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("QUERY INTENT CLASSIFICATION TEST")
    print("=" * 60)

    test_queries = [
        "I'm looking for a PhD advisor in battery materials",
        "Who can I collaborate with on machine learning for materials?",
        "Expert in transmission electron microscopy",
        "What research is being done on sustainable polymers?",
        "Labs with atomic layer deposition equipment"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = classify_query_intent(query)
        print(f"  Intent: {result['intent']} ({result['confidence']}%)")
        print(f"  Refined: {result['refined_query']}")

        strategy = get_search_strategy(result)
        print(f"  Strategy: dense={strategy['dense_weight']}, sparse={strategy['sparse_weight']}")
        if strategy['filter_faculty_only']:
            print(f"  → Filter: Faculty only")
        if strategy['extra_keywords']:
            print(f"  → Keywords: {strategy['extra_keywords']}")
