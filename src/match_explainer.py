#!/usr/bin/env python3
"""
Match Explanation with Gemini
==============================
Generates human-readable explanations for why each researcher matches a query.
This adds interpretability to the search results - a key feature for Hackathon demos.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional

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
from google.genai import types

# Configure Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def explain_match(
    query: str,
    researcher: Dict,
    model_name: str = "gemini-3-flash-preview"
) -> Dict:
    """
    Generate an explanation for why a researcher matches a query.

    Args:
        query: User's search query
        researcher: Researcher dict with name, department, research_interests, etc.
        model_name: Gemini model to use

    Returns:
        Dict with explanation, confidence, and key_overlap fields
    """
    name = researcher.get("name", "Unknown")
    dept = researcher.get("department", "")
    interests = researcher.get("research_interests", "")[:600]
    bio = researcher.get("biography", "")[:400]

    # Build researcher profile
    profile_parts = [f"Name: {name}"]
    if dept:
        profile_parts.append(f"Department: {dept}")
    if interests:
        clean_interests = interests.replace("Research Interests\n", "").strip()
        profile_parts.append(f"Research: {clean_interests[:500]}")
    if bio:
        clean_bio = bio.replace("Biography\n", "").strip()
        profile_parts.append(f"Background: {clean_bio[:300]}")

    profile = "\n".join(profile_parts)

    prompt = f"""You are an expert academic advisor helping match researchers to queries.

Query: "{query}"

Researcher Profile:
{profile}

Task: Explain why this researcher is relevant to the query.

Provide a JSON response with:
{{
    "explanation": "2-3 sentence explanation of why this researcher matches the query. Be specific about research overlap.",
    "confidence": <0-100 integer, how confident this is a good match>,
    "key_overlap": ["keyword1", "keyword2", "keyword3"],
    "match_type": "exact" | "related" | "tangential"
}}

Rules:
- Be honest about match quality - don't oversell weak matches
- "exact": researcher's primary focus matches query
- "related": significant overlap but not primary focus
- "tangential": some connection but not core research
- Key overlap should be 2-4 specific terms that connect query to researcher

Your JSON response:"""

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
            "explanation": data.get("explanation", ""),
            "confidence": min(100, max(0, int(data.get("confidence", 50)))),
            "key_overlap": data.get("key_overlap", []),
            "match_type": data.get("match_type", "related")
        }

    except Exception as e:
        return {
            "explanation": f"Match explanation unavailable: {str(e)[:50]}",
            "confidence": 50,
            "key_overlap": [],
            "match_type": "unknown"
        }


def explain_matches_batch(
    query: str,
    researchers: List[Dict],
    top_k: int = 5,
    model_name: str = "gemini-3-flash-preview"
) -> List[Dict]:
    """
    Generate explanations for multiple researchers in a single API call.
    More efficient than calling explain_match individually.

    Args:
        query: User's search query
        researchers: List of researcher dicts
        top_k: Number of researchers to explain
        model_name: Gemini model

    Returns:
        List of researcher dicts with added explanation fields
    """
    researchers = researchers[:top_k]

    # Build all profiles
    profiles = []
    for i, r in enumerate(researchers):
        name = r.get("name", "Unknown")
        dept = r.get("department", "")
        interests = r.get("research_interests", "")[:300]

        profile = f"[{i}] {name}"
        if dept:
            profile += f" ({dept})"
        if interests:
            clean = interests.replace("Research Interests\n", "").strip()
            profile += f": {clean[:200]}"

        profiles.append(profile)

    profiles_str = "\n".join(profiles)

    prompt = f"""You are an expert academic advisor explaining search results.

Query: "{query}"

Matched Researchers:
{profiles_str}

Task: For each researcher, explain why they match the query.

Return a JSON array with one object per researcher:
[
    {{
        "index": 0,
        "explanation": "2-3 sentence explanation",
        "confidence": <0-100>,
        "key_overlap": ["term1", "term2"],
        "match_type": "exact" | "related" | "tangential"
    }},
    ...
]

Be honest - if a match is weak, say so. Output ONLY the JSON array:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        result_text = response.text.strip()

        # Parse JSON array
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "[" in result_text:
            json_start = result_text.find("[")
            json_end = result_text.rfind("]") + 1
            result_text = result_text[json_start:json_end]

        explanations = json.loads(result_text)

        # Merge explanations into researchers
        results = []
        for r in researchers:
            r_copy = dict(r)
            r_copy["explanation"] = ""
            r_copy["confidence"] = 50
            r_copy["key_overlap"] = []
            r_copy["match_type"] = "unknown"
            results.append(r_copy)

        for exp in explanations:
            idx = exp.get("index", -1)
            if 0 <= idx < len(results):
                results[idx]["explanation"] = exp.get("explanation", "")
                results[idx]["confidence"] = min(100, max(0, int(exp.get("confidence", 50))))
                results[idx]["key_overlap"] = exp.get("key_overlap", [])
                results[idx]["match_type"] = exp.get("match_type", "related")

        return results

    except Exception as e:
        # Return researchers with default explanations
        results = []
        for r in researchers:
            r_copy = dict(r)
            r_copy["explanation"] = "Explanation unavailable"
            r_copy["confidence"] = 50
            r_copy["key_overlap"] = []
            r_copy["match_type"] = "unknown"
            results.append(r_copy)
        return results


def format_explanation_for_display(researcher: Dict) -> str:
    """
    Format a researcher's explanation for display.

    Args:
        researcher: Researcher dict with explanation fields

    Returns:
        Formatted string for display
    """
    name = researcher.get("name", "Unknown")
    dept = researcher.get("department", "")
    explanation = researcher.get("explanation", "No explanation available")
    confidence = researcher.get("confidence", 0)
    key_overlap = researcher.get("key_overlap", [])
    match_type = researcher.get("match_type", "unknown")

    # Confidence emoji
    if confidence >= 80:
        conf_indicator = "🟢"
    elif confidence >= 50:
        conf_indicator = "🟡"
    else:
        conf_indicator = "🔴"

    # Match type label
    type_labels = {
        "exact": "Direct Match",
        "related": "Related Research",
        "tangential": "Partial Match",
        "unknown": ""
    }
    type_label = type_labels.get(match_type, "")

    output = f"**{name}**"
    if dept:
        output += f" ({dept})"
    output += f"\n{conf_indicator} {confidence}% confidence"
    if type_label:
        output += f" · {type_label}"
    output += f"\n\n{explanation}"
    if key_overlap:
        output += f"\n\n*Key terms: {', '.join(key_overlap)}*"

    return output


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    import pickle

    print("=" * 60)
    print("MATCH EXPLANATION TEST")
    print("=" * 60)

    # Load researchers
    with open("data/processed/researchers_dual_adapter.pkl", "rb") as f:
        researchers = pickle.load(f)

    # Test query
    query = "battery materials for electric vehicles"

    # Get some sample researchers
    sample = [r for r in researchers if "batter" in r.get("research_interests", "").lower()][:2]
    sample += [r for r in researchers if "energy" in r.get("research_interests", "").lower()][:2]
    sample = sample[:4] if sample else researchers[:4]

    print(f"\nQuery: {query}")
    print(f"Testing with {len(sample)} researchers...")

    # Test batch explanation
    results = explain_matches_batch(query, sample, top_k=4)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(format_explanation_for_display(r))
