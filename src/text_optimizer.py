"""
Text Optimizer for Researcher Profiles
======================================
Cleans and optimizes researcher profile text for better embeddings.

Key optimizations:
1. Remove navigation/scraping artifacts
2. Extract and prioritize research-relevant content
3. Create structured embedding text
4. Query expansion for better matching
"""
import re
from typing import List, Dict, Optional


# Common noise patterns from web scraping
NOISE_PATTERNS = [
    r'View all\s*Faculty',
    r'Breadcrumb',
    r'Home\s*Research\s*Our Experts',
    r'Skip to main content',
    r'Search\s*form',
    r'Main navigation',
    r'Footer',
    r'Copyright.*\d{4}',
    r'All rights reserved',
    r'Privacy Policy',
    r'Terms of Use',
    r'\[.*?\]',  # Square bracket artifacts
    r'↗',  # Arrow symbols
    r'→',
    r'\s+',  # Multiple whitespace
]

# Patterns indicating research content
RESEARCH_INDICATORS = [
    r'research\s+(?:focus|interests?|areas?)',
    r'(?:his|her|their)\s+research',
    r'works?\s+on',
    r'studies?\s+',
    r'investigat(?:es?|ing)',
    r'develops?\s+',
    r'expertise\s+in',
]


def clean_text(text: str) -> str:
    """
    Remove navigation artifacts and noise from scraped text.

    Args:
        text: Raw scraped text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove common noise patterns
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove email-like patterns (but keep for reference)
    # text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_research_section(text: str) -> str:
    """
    Extract the most research-relevant portion of text.

    Args:
        text: Cleaned profile text

    Returns:
        Research-focused text segment
    """
    if not text:
        return ""

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Score sentences by research relevance
    scored_sentences = []
    for sent in sentences:
        score = 0
        sent_lower = sent.lower()

        # Check for research indicators
        for pattern in RESEARCH_INDICATORS:
            if re.search(pattern, sent_lower):
                score += 2

        # Check for technical terms (capitalized words, acronyms)
        acronyms = len(re.findall(r'\b[A-Z]{2,}\b', sent))
        score += acronyms * 0.5

        # Penalize very short sentences
        if len(sent.split()) < 5:
            score -= 1

        scored_sentences.append((sent, score))

    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: -x[1])

    # Take sentences until we have ~500 chars
    result = []
    char_count = 0
    for sent, score in scored_sentences:
        if char_count > 500 and score < 1:
            break
        result.append(sent)
        char_count += len(sent)
        if char_count > 1000:
            break

    return ' '.join(result)


def create_embedding_text(researcher: Dict, format_type: str = "optimized") -> str:
    """
    Create optimized text for embedding generation.

    Args:
        researcher: Researcher dict with profile fields
        format_type: One of "optimized", "structured", "clean", "raw"

    Returns:
        Optimized text for embedding
    """
    name = researcher.get('name', 'Unknown')
    position = researcher.get('position', '')
    department = researcher.get('department', '')

    if format_type == "raw":
        return researcher.get('raw_text', '')

    # Start with identity
    parts = [f"{name}, {position}, {department}."]

    if format_type == "clean":
        # Just clean the raw text
        raw = researcher.get('raw_text', '')
        cleaned = clean_text(raw)
        parts.append(cleaned)
        return ' '.join(parts)

    # Prioritized fields for optimized/structured formats

    # 1. Research interests (highest priority)
    if researcher.get('research_interests'):
        parts.append(f"Research focus: {researcher['research_interests']}")

    # 2. Biography (if available)
    if researcher.get('biography'):
        bio = clean_text(researcher['biography'])
        # Extract research-relevant portion
        research_bio = extract_research_section(bio)
        if research_bio:
            parts.append(research_bio)

    # 3. Papers (if available)
    if researcher.get('papers'):
        paper_titles = [p.get('title', '') for p in researcher['papers'][:5]]
        paper_titles = [t for t in paper_titles if t]
        if paper_titles:
            parts.append(f"Publications: {'; '.join(paper_titles)}")

    # 4. Enriched summary (if available from LLM enrichment)
    if researcher.get('enriched_summary'):
        parts.append(researcher['enriched_summary'])

    # 5. Fall back to cleaned raw_text if we have little content
    if len(' '.join(parts)) < 200:
        raw = researcher.get('raw_text', '')
        cleaned = clean_text(raw)
        research_portion = extract_research_section(cleaned)
        if research_portion:
            parts.append(research_portion)

    return ' '.join(parts)


def expand_query(query: str, use_llm: bool = False) -> str:
    """
    Expand query with related terms for better recall.

    Args:
        query: Original search query
        use_llm: Whether to use LLM for expansion (slower but better)

    Returns:
        Expanded query
    """
    # Simple rule-based expansion
    expansions = {
        'ML': 'machine learning',
        'AI': 'artificial intelligence',
        'DL': 'deep learning',
        'NLP': 'natural language processing',
        'CV': 'computer vision',
        'RL': 'reinforcement learning',
        'DNA': 'deoxyribonucleic acid nucleic acid',
        'RNA': 'ribonucleic acid',
        'CRISPR': 'CRISPR gene editing genome',
        'MBE': 'molecular beam epitaxy',
        'CVD': 'chemical vapor deposition',
        'ALD': 'atomic layer deposition',
        'GaN': 'gallium nitride',
        'semiconductor': 'semiconductor electronics transistor',
        'battery': 'battery energy storage lithium',
        'solar': 'solar photovoltaic renewable energy',
        'polymer': 'polymer macromolecule synthesis',
        'catalyst': 'catalyst catalysis reaction',
        'quantum': 'quantum mechanics physics',
        'nano': 'nanomaterial nanotechnology nanoscale',
        'bio': 'biological biomedical life science',
    }

    expanded = query
    query_lower = query.lower()

    for abbrev, expansion in expansions.items():
        if abbrev.lower() in query_lower:
            expanded += f" {expansion}"

    return expanded


def create_optimized_corpus(researchers: List[Dict]) -> List[Dict]:
    """
    Create optimized corpus with cleaned text for all researchers.

    Args:
        researchers: List of researcher dicts

    Returns:
        List of researchers with optimized_text field added
    """
    optimized = []

    for r in researchers:
        r_copy = r.copy()
        r_copy['optimized_text'] = create_embedding_text(r, format_type="optimized")
        r_copy['text_length'] = len(r_copy['optimized_text'])
        optimized.append(r_copy)

    return optimized


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    # Test cleaning
    noisy_text = """
    View all Faculty
    Breadcrumb Home Research Our Experts
    John Smith
    Professor

    His research focuses on DNA origami and self-assembly of nanomaterials.
    He investigates novel approaches to create programmable nanostructures.

    Contact: john@cornell.edu
    Phone: 607-555-1234
    Website ↗
    """

    cleaned = clean_text(noisy_text)
    print("Original length:", len(noisy_text))
    print("Cleaned length:", len(cleaned))
    print("Cleaned text:", cleaned)
    print()

    # Test query expansion
    query = "GaN power electronics"
    expanded = expand_query(query)
    print(f"Original query: {query}")
    print(f"Expanded query: {expanded}")
