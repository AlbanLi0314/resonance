#!/usr/bin/env python3
"""
Web Search Enrichment for Researchers
======================================
Uses Google's Gemini API with grounding to search for Cornell researcher information.
This is more accurate than Semantic Scholar because it gets Cornell-specific data.
"""
import json
import time
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

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

PROJECT_ROOT = Path(__file__).parent.parent

# Configure Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def search_and_extract_researcher_info(
    name: str,
    department: str = "",
    existing_info: str = "",
    model_name: str = "gemini-3-flash-preview"
) -> Optional[Dict]:
    """
    Use Gemini with grounding to search for Cornell researcher information.

    Returns structured researcher data.
    """
    prompt = f"""Search for information about this Cornell University researcher and extract their research details.

Researcher: {name}
Department: {department or 'Unknown'}
{"Existing info: " + existing_info[:500] if existing_info else ""}

Please search the web and provide:
1. Their main research areas and topics (be specific)
2. Key research themes or focus areas
3. Notable publications or projects
4. Any specific techniques or methods they use

Format your response as JSON with these fields:
{{
    "research_areas": ["area1", "area2", ...],
    "research_summary": "2-3 sentence summary of their research",
    "keywords": ["keyword1", "keyword2", ...],
    "notable_work": "Any notable publications or projects mentioned"
}}

Only include information you find about THIS specific researcher at Cornell.
If you cannot find reliable information, return {{"error": "not found"}}."""

    try:
        # Use Gemini with Google Search grounding
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        result_text = response.text.strip()

        # Try to parse JSON from response
        # Find JSON block
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "{" in result_text and "}" in result_text:
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            result_text = result_text[json_start:json_end]

        data = json.loads(result_text)

        if data.get("error"):
            return None

        return data

    except Exception as e:
        print(f"    Error searching for {name}: {e}")
        return None


def enrich_researcher(researcher: Dict) -> Dict:
    """
    Enrich a single researcher with web search data.
    """
    name = researcher.get("name", "")
    department = researcher.get("department", "")
    existing_info = researcher.get("research_interests", "") or researcher.get("biography", "")

    # Search for info
    info = search_and_extract_researcher_info(name, department, existing_info)

    if not info:
        return researcher

    # Create enriched copy
    enriched = researcher.copy()

    # Add web search data
    enriched["web_research_areas"] = info.get("research_areas", [])
    enriched["web_research_summary"] = info.get("research_summary", "")
    enriched["web_keywords"] = info.get("keywords", [])
    enriched["web_notable_work"] = info.get("notable_work", "")
    enriched["web_enriched"] = True

    # Build enriched raw_text
    text_parts = []

    # Original info
    text_parts.append(f"Name: {name}")
    if department:
        text_parts.append(f"Department: {department}")
    if researcher.get("research_interests"):
        text_parts.append(f"Research Interests: {researcher['research_interests'][:500]}")

    # Add web search info
    if info.get("research_areas"):
        text_parts.append(f"Research Areas: {', '.join(info['research_areas'])}")
    if info.get("research_summary"):
        text_parts.append(f"Research Summary: {info['research_summary']}")
    if info.get("keywords"):
        text_parts.append(f"Keywords: {', '.join(info['keywords'])}")
    if info.get("notable_work"):
        text_parts.append(f"Notable Work: {info['notable_work']}")

    enriched["raw_text_web_enriched"] = "\n".join(text_parts)

    return enriched


def needs_enrichment(researcher: Dict, min_text_len: int = 500) -> bool:
    """
    Check if a researcher needs web search enrichment.
    """
    raw_text = researcher.get("raw_text", "")
    research_interests = researcher.get("research_interests", "")

    # Needs enrichment if:
    # 1. Short raw_text
    # 2. No research interests
    # 3. Already has web enrichment
    if researcher.get("web_enriched"):
        return False

    return len(raw_text) < min_text_len or not research_interests


def main():
    print("=" * 60)
    print("WEB SEARCH DATA ENRICHMENT")
    print("Using Gemini with Google Search grounding")
    print("=" * 60)

    # Load current data
    input_path = PROJECT_ROOT / "data/raw/researchers.json"
    print(f"\nLoading researchers from {input_path}...")

    with open(input_path) as f:
        data = json.load(f)

    researchers = data["researchers"]
    print(f"Loaded {len(researchers)} researchers")

    # Find researchers needing enrichment
    to_enrich = [r for r in researchers if needs_enrichment(r)]
    print(f"Need enrichment: {len(to_enrich)} researchers")

    # Check for existing checkpoint
    checkpoint_path = PROJECT_ROOT / "data/enriched/web_enrichment_checkpoint.json"
    already_enriched = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            already_enriched = json.load(f)
        print(f"Found checkpoint with {len(already_enriched)} already enriched")

    # Filter out already enriched
    to_enrich = [r for r in to_enrich if r["name"] not in already_enriched]
    print(f"Remaining to enrich: {len(to_enrich)}")

    if not to_enrich:
        print("Nothing to enrich!")
        return

    # Enrich each researcher
    print(f"\nEnriching with web search...")
    print(f"(Estimated time: ~{len(to_enrich) * 3 / 60:.0f} minutes)")

    success_count = 0

    import sys

    for i, r in enumerate(to_enrich):
        print(f"  [{i+1}/{len(to_enrich)}] {r['name']}...", end=" ")
        sys.stdout.flush()

        try:
            enriched = enrich_researcher(r)

            if enriched.get("web_enriched"):
                success_count += 1
                already_enriched[r["name"]] = {
                    "web_research_areas": enriched.get("web_research_areas", []),
                    "web_research_summary": enriched.get("web_research_summary", ""),
                    "web_keywords": enriched.get("web_keywords", []),
                    "web_notable_work": enriched.get("web_notable_work", ""),
                    "raw_text_web_enriched": enriched.get("raw_text_web_enriched", "")
                }
                print("OK")
            else:
                print("SKIP")
        except Exception as e:
            print(f"ERROR: {e}")

        sys.stdout.flush()

        # Save checkpoint every 10
        if (i + 1) % 10 == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(already_enriched, f, indent=2)
            print(f"  Checkpoint saved: {len(already_enriched)} enriched")
            sys.stdout.flush()

        # Rate limiting - 3 seconds between calls
        time.sleep(3)

    # Save final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(already_enriched, f, indent=2)

    # Apply enrichment to all researchers
    print("\nApplying enrichment to all researchers...")

    enriched_researchers = []
    for r in researchers:
        if r["name"] in already_enriched:
            enriched = r.copy()
            enriched.update(already_enriched[r["name"]])
            enriched["web_enriched"] = True
            enriched_researchers.append(enriched)
        else:
            enriched_researchers.append(r)

    # Save enriched data
    output_json = PROJECT_ROOT / "data/enriched/researchers_web_enriched.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump({"researchers": enriched_researchers}, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)

    web_enriched = sum(1 for r in enriched_researchers if r.get("web_enriched"))

    print(f"Total researchers: {len(enriched_researchers)}")
    print(f"Web enriched: {web_enriched} ({web_enriched/len(enriched_researchers)*100:.0f}%)")
    print(f"New enriched this run: {success_count}")
    print("=" * 60)

    print(f"\nOutput: {output_json}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
