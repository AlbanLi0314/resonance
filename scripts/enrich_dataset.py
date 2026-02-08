#!/usr/bin/env python3
"""
Comprehensive Dataset Enrichment Pipeline
==========================================
Implements the 5-phase embedding quality improvement plan.

Phase 1: Data Cleaning & Preparation
Phase 2: Enrich Existing Researchers
Phase 3: Add New Researchers
Phase 4: Optimize Text for Embedding
Phase 5: Output & Validation
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "enriched"

# Input files
CURRENT_RESEARCHERS = DATA_DIR / "raw" / "researchers.json"
PAPERS_METADATA = PROJECT_ROOT / "papers_metadata.json"
PHD_XMOL = PROJECT_ROOT / "phd_xmol_publications.json"
PHD_WITH_PUBS = PROJECT_ROOT / "phd_with_publications.json"
ALL_FACULTY = PROJECT_ROOT / "all_faculty.json"

# Patterns to remove from raw_text
ARTIFACT_PATTERNS = [
    r'^View all\s*\t*\s*Faculty\s*\t*\s*\n?',
    r'\t+',
    r'↗',
    r'\s*Graduate Field Affiliations\s*\n(\s*[A-Za-z\s]+\s*\n)*',
    r'Location[A-Za-z\s]+,?\s*Room\s*\d+\s*\n?',
    r'Phone\s*\d{3}[-.]?\d{3}[-.]?\d{4}\s*\n?',
    r'WebsiteVisit Website\s*\n?',
    r'Website[A-Za-z\s]+↗?\s*\n?',
    r'Google Scholar\s*↗?\s*\n?',
    r'Select Publications\s*\n',
    r'Select Awards and Honors\s*\n',
    r'Teaching Interests\s*\n',
    r'Service Interests\s*\n',
    r'In the News\s*\n',
    r'Education\s*\n',
    r'\n\s*\n\s*\n+',  # Multiple blank lines
]

# Bad abstract patterns
BAD_ABSTRACT_PATTERNS = [
    'Copyright ©',
    'Erratum',
    'Author information:',
    'PMCID:',
    '北京衮雪',
    'Conflict of interest statement',
    '[This corrects the article',
]

# HTML patterns to clean from titles
HTML_PATTERNS = [
    (r'</?i>', ''),
    (r'</?b>', ''),
    (r'</?sub>', ''),
    (r'</?sup>', ''),
    (r'&amp;', '&'),
    (r'&lt;', '<'),
    (r'&gt;', '>'),
    (r'\s+', ' '),
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_id(name: str, prefix: str = "researcher") -> str:
    """Generate unique ID from name."""
    clean_name = re.sub(r'[^a-z0-9]', '_', name.lower())
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"{prefix}_{clean_name}_{hash_suffix}"


def clean_text(text: str) -> str:
    """Remove navigation artifacts and clean text."""
    if not text:
        return ""

    result = text
    for pattern in ARTIFACT_PATTERNS:
        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE | re.MULTILINE)

    # Clean up whitespace
    result = re.sub(r'\s+', ' ', result)
    result = result.strip()

    return result


def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""

    result = text
    for pattern, replacement in HTML_PATTERNS:
        result = re.sub(pattern, replacement, result)

    return result.strip()


def is_bad_abstract(abstract: str) -> bool:
    """Check if abstract contains problematic content."""
    if not abstract:
        return True

    for pattern in BAD_ABSTRACT_PATTERNS:
        if pattern in abstract:
            return True

    return False


def fuzzy_match(name: str, candidates: List[str], threshold: float = 0.85) -> Optional[str]:
    """Find best fuzzy match for a name."""
    name_lower = name.lower().strip()

    # Exact match first
    for candidate in candidates:
        if candidate.lower().strip() == name_lower:
            return candidate

    # Fuzzy match
    best_match = None
    best_score = 0

    for candidate in candidates:
        score = SequenceMatcher(None, name_lower, candidate.lower().strip()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate

    return best_match


# ============================================================================
# PHASE 1: DATA CLEANING & PREPARATION
# ============================================================================

def phase1_clean_and_prepare() -> Tuple[List[Dict], Dict[str, Dict], Dict[str, List[Dict]]]:
    """
    Phase 1: Clean existing data and prepare publication database.

    Returns:
        - cleaned_researchers: List of cleaned researcher records
        - papers_by_doi: Dict mapping DOI to paper metadata
        - papers_by_name: Dict mapping author name to list of papers
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Data Cleaning & Preparation")
    print("=" * 60)

    # Load current researchers
    current_data = load_json(CURRENT_RESEARCHERS)
    researchers = current_data.get('researchers', current_data)
    print(f"Loaded {len(researchers)} current researchers")

    # Load papers metadata
    papers = load_json(PAPERS_METADATA)
    print(f"Loaded {len(papers)} papers from papers_metadata.json")

    # Load PhD with publications (for proper titles)
    phd_pubs = load_json(PHD_WITH_PUBS)
    print(f"Loaded {len(phd_pubs)} PhD records from phd_with_publications.json")

    # -------------------------------------------------------------------------
    # Task 1.1: Clean researcher profiles
    # -------------------------------------------------------------------------
    print("\n[Task 1.1] Cleaning researcher profiles...")

    cleaned_count = 0
    for r in researchers:
        original_text = r.get('raw_text', '')
        cleaned_text = clean_text(original_text)

        if cleaned_text != original_text:
            cleaned_count += 1

        r['raw_text_original'] = original_text
        r['raw_text'] = cleaned_text

        # Also clean biography and research_interests
        if r.get('biography'):
            r['biography'] = clean_text(r['biography'])
        if r.get('research_interests'):
            r['research_interests'] = clean_text(r['research_interests'])

    print(f"   Cleaned {cleaned_count} profiles")

    # -------------------------------------------------------------------------
    # Task 1.2: Clean papers metadata
    # -------------------------------------------------------------------------
    print("\n[Task 1.2] Cleaning papers metadata...")

    clean_papers = []
    bad_papers = 0

    for p in papers:
        abstract = p.get('abstract', '')

        if is_bad_abstract(abstract):
            bad_papers += 1
            continue

        # Clean the paper
        clean_paper = {
            'name': p.get('name', ''),
            'doi': p.get('doi', '').lower() if p.get('doi') else None,
            'title': clean_html(p.get('title', '')),
            'abstract': abstract,
            'topics': p.get('topics', []),
            'concepts': p.get('concepts', []),
            'author_position': p.get('author_position', 0),
        }
        clean_papers.append(clean_paper)

    print(f"   Kept {len(clean_papers)} clean papers, removed {bad_papers} problematic")

    # -------------------------------------------------------------------------
    # Task 1.3: Create unified publication database
    # -------------------------------------------------------------------------
    print("\n[Task 1.3] Creating unified publication database...")

    # Build DOI → paper mapping
    papers_by_doi = {}
    for p in clean_papers:
        if p['doi']:
            papers_by_doi[p['doi']] = p

    # Build name → papers mapping
    papers_by_name = defaultdict(list)
    for p in clean_papers:
        name = p['name'].lower().strip()
        papers_by_name[name].append(p)

    # Merge with phd_with_publications for proper titles
    titles_added = 0
    for phd in phd_pubs:
        phd_name = phd['name'].lower().strip()
        for pub in phd.get('publications', []):
            doi = pub.get('doi', '').lower() if pub.get('doi') else None
            if doi and doi in papers_by_doi:
                # Update with proper title if available
                proper_title = pub.get('title', '')
                if proper_title and len(proper_title) < 200:  # Proper titles are short
                    papers_by_doi[doi]['title'] = clean_html(proper_title)
                    titles_added += 1

    print(f"   Papers indexed by DOI: {len(papers_by_doi)}")
    print(f"   Authors with papers: {len(papers_by_name)}")
    print(f"   Titles updated from phd_with_publications: {titles_added}")

    return researchers, papers_by_doi, dict(papers_by_name)


# ============================================================================
# PHASE 2: ENRICH EXISTING RESEARCHERS
# ============================================================================

def phase2_enrich_existing(
    researchers: List[Dict],
    papers_by_name: Dict[str, List[Dict]],
    phd_pubs: List[Dict],
    faculty_lookup: Dict[str, Dict]
) -> List[Dict]:
    """
    Phase 2: Enrich existing researchers with publications and advisor info.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Enrich Existing Researchers")
    print("=" * 60)

    # Build lookup for phd_with_publications
    phd_lookup = {p['name'].lower().strip(): p for p in phd_pubs}

    # -------------------------------------------------------------------------
    # Task 2.1: Enrich with publications
    # -------------------------------------------------------------------------
    print("\n[Task 2.1] Enriching with publications...")

    enriched_with_papers = 0
    total_papers_added = 0

    for r in researchers:
        name_lower = r['name'].lower().strip()

        # Check if we have papers for this person
        if name_lower in papers_by_name:
            papers = papers_by_name[name_lower]

            # Add publications to researcher
            r['publications'] = papers

            # Extract all topics and concepts
            all_topics = []
            all_concepts = []
            for p in papers:
                all_topics.extend(p.get('topics', []))
                all_concepts.extend(p.get('concepts', []))

            # Deduplicate
            r['research_keywords'] = list(set(all_topics + all_concepts))

            enriched_with_papers += 1
            total_papers_added += len(papers)

    print(f"   Researchers enriched with papers: {enriched_with_papers}")
    print(f"   Total papers added: {total_papers_added}")

    # -------------------------------------------------------------------------
    # Task 2.2: Add advisor info from phd_with_publications
    # -------------------------------------------------------------------------
    print("\n[Task 2.2] Adding advisor and lab info...")

    advisor_info_added = 0

    for r in researchers:
        name_lower = r['name'].lower().strip()

        if name_lower in phd_lookup:
            phd_info = phd_lookup[name_lower]

            # Add advisor info if not present
            if not r.get('advisor') and phd_info.get('advisor'):
                r['advisor'] = phd_info.get('advisor')
                r['advisor_full_name'] = phd_info.get('advisor_full_name', phd_info.get('advisor'))

            # Add lab info
            if not r.get('lab') and phd_info.get('lab_name'):
                r['lab'] = phd_info.get('lab_name')
                r['lab_url'] = phd_info.get('lab_url', '')

            advisor_info_added += 1

    print(f"   Advisor/lab info added: {advisor_info_added}")

    # -------------------------------------------------------------------------
    # Task 2.3: Infer research from advisor (for PhDs without papers)
    # -------------------------------------------------------------------------
    print("\n[Task 2.3] Inferring research from advisors...")

    research_inferred = 0

    for r in researchers:
        # Only for PhD students without publications
        position = r.get('position', '').lower()
        is_phd = 'phd' in position or 'student' in position
        has_papers = bool(r.get('publications'))
        has_research = bool(r.get('research_interests')) and len(r.get('research_interests', '')) > 100

        if is_phd and not has_papers and not has_research:
            advisor = r.get('advisor', '')
            if advisor:
                # Find advisor in faculty
                advisor_match = fuzzy_match(advisor, list(faculty_lookup.keys()))
                if advisor_match:
                    faculty = faculty_lookup[advisor_match]
                    advisor_research = faculty.get('research_interests', '')

                    if advisor_research and len(advisor_research) > 100:
                        r['research_interests_inferred'] = advisor_research
                        r['research_inferred_from_advisor'] = True
                        research_inferred += 1

    print(f"   Research inferred from advisors: {research_inferred}")

    return researchers


# ============================================================================
# PHASE 3: ADD NEW RESEARCHERS
# ============================================================================

def phase3_add_new_researchers(
    researchers: List[Dict],
    papers_by_name: Dict[str, List[Dict]],
    phd_xmol: List[Dict],
    faculty_lookup: Dict[str, Dict]
) -> List[Dict]:
    """
    Phase 3: Add new PhD students from phd_xmol_publications.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Add New Researchers")
    print("=" * 60)

    # Get existing names
    existing_names = {r['name'].lower().strip() for r in researchers}

    # -------------------------------------------------------------------------
    # Task 3.1: Add new PhD students
    # -------------------------------------------------------------------------
    print("\n[Task 3.1] Adding new PhD students...")

    new_phds_added = 0
    new_with_papers = 0
    new_with_advisor_research = 0

    for phd in phd_xmol:
        name = phd['name']
        name_lower = name.lower().strip()

        # Skip if already exists
        if name_lower in existing_names:
            continue

        # Create new researcher record
        new_researcher = {
            'id': generate_id(name, 'phd'),
            'name': name,
            'position': 'PhD Student',
            'department': phd.get('department', 'Materials Science and Engineering'),
            'email': '',
            'advisor': phd.get('advisor', ''),
            'advisor_full_name': phd.get('advisor_full_name', phd.get('advisor', '')),
            'lab': phd.get('lab_name', ''),
            'lab_url': phd.get('lab_url', ''),
            'personal_website': '',
            'sources': ['phd_xmol_publications.json'],
        }

        # Add publications if available
        if name_lower in papers_by_name:
            papers = papers_by_name[name_lower]
            new_researcher['publications'] = papers

            # Extract keywords
            all_topics = []
            all_concepts = []
            for p in papers:
                all_topics.extend(p.get('topics', []))
                all_concepts.extend(p.get('concepts', []))

            new_researcher['research_keywords'] = list(set(all_topics + all_concepts))
            new_with_papers += 1
        else:
            # Try to infer from advisor
            advisor = phd.get('advisor', '')
            if advisor:
                advisor_match = fuzzy_match(advisor, list(faculty_lookup.keys()))
                if advisor_match:
                    faculty = faculty_lookup[advisor_match]
                    advisor_research = faculty.get('research_interests', '')

                    if advisor_research and len(advisor_research) > 100:
                        new_researcher['research_interests_inferred'] = advisor_research
                        new_researcher['research_inferred_from_advisor'] = True
                        new_with_advisor_research += 1

        researchers.append(new_researcher)
        existing_names.add(name_lower)
        new_phds_added += 1

    print(f"   New PhD students added: {new_phds_added}")
    print(f"   New PhDs with papers: {new_with_papers}")
    print(f"   New PhDs with advisor research: {new_with_advisor_research}")

    return researchers


# ============================================================================
# PHASE 4: OPTIMIZE TEXT FOR EMBEDDING
# ============================================================================

def phase4_optimize_text(researchers: List[Dict]) -> List[Dict]:
    """
    Phase 4: Create optimized text representations for embedding.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Optimize Text for Embedding")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Task 4.1 & 4.2: Create multiple text variants
    # -------------------------------------------------------------------------
    print("\n[Task 4.1] Creating optimized text formats...")

    for r in researchers:
        # Preserve original
        if 'raw_text_original' not in r:
            r['raw_text_original'] = r.get('raw_text', '')

        # Build components
        name = r.get('name', '')
        position = r.get('position', '')
        department = r.get('department', '')
        advisor = r.get('advisor', '')
        lab = r.get('lab', '')

        # Research interests (original or inferred)
        research = r.get('research_interests', '') or r.get('research_interests_inferred', '')

        # Biography
        bio = r.get('biography', '')

        # Publications
        pubs = r.get('publications', [])
        pub_titles = [p.get('title', '') for p in pubs[:5]]  # Top 5
        pub_abstracts = [p.get('abstract', '')[:500] for p in pubs[:3]]  # Top 3, truncated

        # Keywords
        keywords = r.get('research_keywords', [])

        # ---------------------------------------------------------------------
        # Variant 1: Cleaned (just artifacts removed)
        # ---------------------------------------------------------------------
        r['raw_text_cleaned'] = clean_text(r.get('raw_text', ''))

        # ---------------------------------------------------------------------
        # Variant 2: Enriched (cleaned + publications + keywords)
        # ---------------------------------------------------------------------
        enriched_parts = [r['raw_text_cleaned']]

        if pubs:
            pub_text = "Publications: " + "; ".join(pub_titles[:5])
            enriched_parts.append(pub_text)

        if keywords:
            kw_text = "Research areas: " + ", ".join(keywords[:20])
            enriched_parts.append(kw_text)

        r['raw_text_enriched'] = " ".join(enriched_parts)

        # ---------------------------------------------------------------------
        # Variant 3: Optimized (fully structured format)
        # ---------------------------------------------------------------------
        opt_parts = []

        # Header
        if position and department:
            opt_parts.append(f"{name}, {position} in {department} at Cornell University.")
        else:
            opt_parts.append(f"{name} at Cornell University.")

        # Advisor/lab info
        if advisor and lab:
            opt_parts.append(f"Works with {advisor} in the {lab}.")
        elif advisor:
            opt_parts.append(f"Works with {advisor}.")

        # Research focus
        if research:
            # Clean and truncate research interests
            research_clean = clean_text(research)[:1000]
            opt_parts.append(f"Research focus: {research_clean}")

        # Publications
        if pub_titles:
            opt_parts.append("Recent publications: " + "; ".join(pub_titles[:5]))

        # Publication abstracts (condensed)
        if pub_abstracts:
            abstracts_combined = " ".join(pub_abstracts)[:1500]
            opt_parts.append(f"Research details: {abstracts_combined}")

        # Keywords
        if keywords:
            opt_parts.append("Research areas: " + ", ".join(keywords[:30]))

        r['raw_text_optimized'] = " ".join(opt_parts)

        # ---------------------------------------------------------------------
        # Variant 4: Keywords-dense (for hybrid search)
        # ---------------------------------------------------------------------
        kw_parts = [name]

        if position:
            kw_parts.append(position)
        if department:
            kw_parts.append(department)
        if advisor:
            kw_parts.append(f"Advisor: {advisor}")

        # Add all keywords
        if keywords:
            kw_parts.extend(keywords)

        # Add paper titles (good keywords)
        if pub_titles:
            kw_parts.extend(pub_titles)

        r['raw_text_keywords'] = " | ".join(kw_parts)

    # Report statistics
    avg_original = sum(len(r.get('raw_text_original', '')) for r in researchers) / len(researchers)
    avg_cleaned = sum(len(r.get('raw_text_cleaned', '')) for r in researchers) / len(researchers)
    avg_enriched = sum(len(r.get('raw_text_enriched', '')) for r in researchers) / len(researchers)
    avg_optimized = sum(len(r.get('raw_text_optimized', '')) for r in researchers) / len(researchers)
    avg_keywords = sum(len(r.get('raw_text_keywords', '')) for r in researchers) / len(researchers)

    print(f"\n   Text length statistics (avg chars):")
    print(f"   - raw_text_original:  {avg_original:.0f}")
    print(f"   - raw_text_cleaned:   {avg_cleaned:.0f}")
    print(f"   - raw_text_enriched:  {avg_enriched:.0f}")
    print(f"   - raw_text_optimized: {avg_optimized:.0f}")
    print(f"   - raw_text_keywords:  {avg_keywords:.0f}")

    return researchers


# ============================================================================
# PHASE 5: OUTPUT & VALIDATION
# ============================================================================

def phase5_output_and_validate(
    researchers: List[Dict],
    papers_by_name: Dict[str, List[Dict]]
) -> Dict:
    """
    Phase 5: Output enriched data and generate validation report.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: Output & Validation")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Save enriched researchers
    # -------------------------------------------------------------------------
    print("\n[Task 5.1] Saving enriched dataset...")

    # Main enriched file
    output_data = {
        "metadata": {
            "total_researchers": len(researchers),
            "enrichment_date": "2026-02-05",
            "text_variants": ["raw_text_original", "raw_text_cleaned",
                           "raw_text_enriched", "raw_text_optimized", "raw_text_keywords"]
        },
        "researchers": researchers
    }

    save_json(output_data, OUTPUT_DIR / "researchers_enriched.json")
    print(f"   Saved: {OUTPUT_DIR / 'researchers_enriched.json'}")

    # Also save in format compatible with existing pipeline
    # (raw_text field should use optimized version)
    pipeline_researchers = []
    for r in researchers:
        pr = r.copy()
        pr['raw_text'] = pr.get('raw_text_optimized', pr.get('raw_text', ''))
        pipeline_researchers.append(pr)

    pipeline_data = {"researchers": pipeline_researchers}
    save_json(pipeline_data, DATA_DIR / "raw" / "researchers_enriched.json")
    print(f"   Saved: {DATA_DIR / 'raw' / 'researchers_enriched.json'}")

    # -------------------------------------------------------------------------
    # Save publications database
    # -------------------------------------------------------------------------
    print("\n[Task 5.2] Saving publications database...")

    all_papers = []
    for name, papers in papers_by_name.items():
        all_papers.extend(papers)

    save_json(all_papers, OUTPUT_DIR / "publications_unified.json")
    print(f"   Saved: {OUTPUT_DIR / 'publications_unified.json'}")

    # -------------------------------------------------------------------------
    # Generate validation report
    # -------------------------------------------------------------------------
    print("\n[Task 5.3] Generating validation report...")

    # Count statistics
    total = len(researchers)
    faculty = sum(1 for r in researchers if 'professor' in r.get('position', '').lower())
    phd = sum(1 for r in researchers if 'phd' in r.get('position', '').lower() or 'student' in r.get('position', '').lower())

    with_pubs = sum(1 for r in researchers if r.get('publications'))
    with_keywords = sum(1 for r in researchers if r.get('research_keywords'))
    with_advisor = sum(1 for r in researchers if r.get('advisor'))
    with_inferred = sum(1 for r in researchers if r.get('research_inferred_from_advisor'))

    avg_text_len = sum(len(r.get('raw_text_optimized', '')) for r in researchers) / total

    # Short profiles check
    short_profiles = sum(1 for r in researchers if len(r.get('raw_text_optimized', '')) < 500)

    report = {
        "summary": {
            "total_researchers": total,
            "faculty": faculty,
            "phd_students": phd,
            "other": total - faculty - phd,
        },
        "enrichment_stats": {
            "with_publications": with_pubs,
            "with_research_keywords": with_keywords,
            "with_advisor_info": with_advisor,
            "with_inferred_research": with_inferred,
        },
        "text_quality": {
            "avg_optimized_text_length": round(avg_text_len),
            "short_profiles_remaining": short_profiles,
            "short_profile_pct": round(100 * short_profiles / total, 1),
        },
        "improvement_vs_baseline": {
            "researchers_added": total - 206,  # Original was 206
            "pct_increase": round(100 * (total - 206) / 206, 1),
        }
    }

    save_json(report, OUTPUT_DIR / "enrichment_report.json")
    print(f"   Saved: {OUTPUT_DIR / 'enrichment_report.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("ENRICHMENT COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"""
    Total researchers: {total} (was 206, +{total - 206})
    - Faculty: {faculty}
    - PhD students: {phd}

    Enrichment coverage:
    - With publications: {with_pubs} ({100*with_pubs/total:.1f}%)
    - With keywords: {with_keywords} ({100*with_keywords/total:.1f}%)
    - With advisor info: {with_advisor} ({100*with_advisor/total:.1f}%)
    - With inferred research: {with_inferred} ({100*with_inferred/total:.1f}%)

    Text quality:
    - Avg optimized text: {avg_text_len:.0f} chars
    - Short profiles (<500): {short_profiles} ({100*short_profiles/total:.1f}%)
    """)

    return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete enrichment pipeline."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DATASET ENRICHMENT PIPELINE")
    print("=" * 60)

    # Load additional data needed
    phd_xmol = load_json(PHD_XMOL)
    phd_pubs = load_json(PHD_WITH_PUBS)
    all_faculty = load_json(ALL_FACULTY)

    # Build faculty lookup
    faculty_lookup = {f['name'].lower().strip(): f for f in all_faculty}

    # Phase 1: Clean and prepare
    researchers, papers_by_doi, papers_by_name = phase1_clean_and_prepare()

    # Phase 2: Enrich existing
    researchers = phase2_enrich_existing(
        researchers, papers_by_name, phd_pubs, faculty_lookup
    )

    # Phase 3: Add new researchers
    researchers = phase3_add_new_researchers(
        researchers, papers_by_name, phd_xmol, faculty_lookup
    )

    # Phase 4: Optimize text
    researchers = phase4_optimize_text(researchers)

    # Phase 5: Output and validate
    report = phase5_output_and_validate(researchers, papers_by_name)

    print("\n✅ Pipeline completed successfully!")
    print(f"\nNext steps:")
    print(f"1. Build new index: python scripts/step1_build_index.py --data enriched")
    print(f"2. Evaluate: python ground_truth/evaluate_with_ground_truth.py")

    return report


if __name__ == "__main__":
    main()
