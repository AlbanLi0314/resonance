#!/usr/bin/env python3
"""
Enrich Researcher Data with Semantic Scholar
=============================================
Fetches papers, abstracts, and citation counts for each researcher.
Semantic Scholar API is free with rate limits (100 requests/5 min).
"""
import json
import time
import pickle
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_URL = f"{S2_API_BASE}/author/search"
S2_AUTHOR_URL = f"{S2_API_BASE}/author"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests


def search_author(name: str) -> Optional[Dict]:
    """
    Search for an author by name on Semantic Scholar.

    Returns author info with ID if found.
    """
    try:
        params = {
            "query": name,
            "limit": 5,
            "fields": "name,affiliations,paperCount,citationCount,hIndex"
        }
        response = requests.get(S2_SEARCH_URL, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                # Try to find best match (Cornell affiliation preferred)
                for author in data["data"]:
                    affiliations = author.get("affiliations", []) or []
                    aff_str = " ".join(affiliations).lower()
                    if "cornell" in aff_str:
                        return author
                # If no Cornell match, return first result
                return data["data"][0]
        elif response.status_code == 429:
            print(f"    Rate limited, waiting 60s...")
            time.sleep(60)
            return search_author(name)  # Retry
        else:
            return None
    except Exception as e:
        print(f"    Error searching {name}: {e}")
        return None


def get_author_papers(author_id: str, limit: int = 10) -> List[Dict]:
    """
    Get papers for an author by their Semantic Scholar ID.
    """
    try:
        url = f"{S2_AUTHOR_URL}/{author_id}/papers"
        params = {
            "limit": limit,
            "fields": "title,abstract,year,citationCount,venue,fieldsOfStudy"
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        elif response.status_code == 429:
            print(f"    Rate limited, waiting 60s...")
            time.sleep(60)
            return get_author_papers(author_id, limit)
        else:
            return []
    except Exception as e:
        print(f"    Error getting papers: {e}")
        return []


def enrich_researcher(researcher: Dict) -> Dict:
    """
    Enrich a single researcher with Semantic Scholar data.
    """
    name = researcher.get("name", "")

    # Search for author
    author_info = search_author(name)
    time.sleep(REQUEST_DELAY)

    if not author_info:
        return researcher

    # Get papers
    author_id = author_info.get("authorId")
    if author_id:
        papers = get_author_papers(author_id, limit=10)
        time.sleep(REQUEST_DELAY)
    else:
        papers = []

    # Enrich researcher data
    enriched = researcher.copy()

    # Add Semantic Scholar metadata
    enriched["s2_author_id"] = author_id
    enriched["s2_paper_count"] = author_info.get("paperCount", 0)
    enriched["s2_citation_count"] = author_info.get("citationCount", 0)
    enriched["s2_h_index"] = author_info.get("hIndex", 0)

    # Add papers with abstracts
    enriched_papers = []
    abstracts = []
    fields_of_study = set()

    for paper in papers:
        paper_data = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year"),
            "citations": paper.get("citationCount", 0),
            "venue": paper.get("venue", "")
        }
        enriched_papers.append(paper_data)

        if paper.get("abstract"):
            abstracts.append(paper["abstract"])

        for field in paper.get("fieldsOfStudy", []) or []:
            fields_of_study.add(field)

    enriched["papers"] = enriched_papers
    enriched["paper_abstracts"] = abstracts
    enriched["fields_of_study"] = list(fields_of_study)

    # Build enriched raw_text
    text_parts = []

    # Original info
    text_parts.append(f"Name: {name}")
    if researcher.get("department"):
        text_parts.append(f"Department: {researcher['department']}")
    if researcher.get("research_interests"):
        text_parts.append(f"Research Interests: {researcher['research_interests']}")

    # Add fields of study
    if fields_of_study:
        text_parts.append(f"Fields: {', '.join(fields_of_study)}")

    # Add paper info
    if enriched_papers:
        text_parts.append(f"Publications ({len(enriched_papers)} recent papers):")
        for p in enriched_papers[:5]:
            if p["title"]:
                text_parts.append(f"- {p['title']}")

    # Add abstracts (most valuable for embedding)
    if abstracts:
        text_parts.append("Research Summary (from papers):")
        # Combine first 3 abstracts, truncated
        combined_abstract = " ".join(abstracts[:3])[:1500]
        text_parts.append(combined_abstract)

    enriched["raw_text_enriched"] = "\n".join(text_parts)

    return enriched


def main():
    print("=" * 60)
    print("SEMANTIC SCHOLAR DATA ENRICHMENT")
    print("=" * 60)

    # Load current data
    input_path = PROJECT_ROOT / "data/processed/researchers_dual_adapter.pkl"
    print(f"\nLoading researchers from {input_path}...")

    with open(input_path, "rb") as f:
        researchers = pickle.load(f)

    print(f"Loaded {len(researchers)} researchers")

    # Enrich each researcher
    print(f"\nEnriching with Semantic Scholar data...")
    print(f"(This will take ~{len(researchers) * 2 / 60:.0f} minutes due to API rate limits)")

    enriched_researchers = []
    success_count = 0

    for i, r in enumerate(tqdm(researchers, desc="Enriching")):
        enriched = enrich_researcher(r)
        enriched_researchers.append(enriched)

        if enriched.get("s2_author_id"):
            success_count += 1

        # Progress update every 20
        if (i + 1) % 20 == 0:
            print(f"\n  Progress: {i+1}/{len(researchers)}, Found: {success_count}")

    # Save enriched data
    output_path = PROJECT_ROOT / "data/processed/researchers_s2_enriched.pkl"
    print(f"\nSaving to {output_path}...")

    with open(output_path, "wb") as f:
        pickle.dump(enriched_researchers, f)

    # Also save as JSON for inspection
    json_path = PROJECT_ROOT / "data/processed/researchers_s2_enriched.json"
    json_data = []
    for r in enriched_researchers:
        r_copy = {k: v for k, v in r.items() if k != "embedding"}
        json_data.append(r_copy)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)

    with_papers = sum(1 for r in enriched_researchers if r.get("papers"))
    with_abstracts = sum(1 for r in enriched_researchers if r.get("paper_abstracts"))
    avg_papers = sum(len(r.get("papers", [])) for r in enriched_researchers) / len(enriched_researchers)

    print(f"Total researchers: {len(enriched_researchers)}")
    print(f"Found on Semantic Scholar: {success_count} ({success_count/len(enriched_researchers)*100:.0f}%)")
    print(f"With papers: {with_papers} ({with_papers/len(enriched_researchers)*100:.0f}%)")
    print(f"With abstracts: {with_abstracts} ({with_abstracts/len(enriched_researchers)*100:.0f}%)")
    print(f"Average papers per researcher: {avg_papers:.1f}")
    print("=" * 60)

    print(f"\nOutput files:")
    print(f"  PKL: {output_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
