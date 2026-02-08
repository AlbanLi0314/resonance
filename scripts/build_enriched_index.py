#!/usr/bin/env python3
"""
Build FAISS Index with Enriched Dataset
========================================
Builds indices using different text variants from the enriched dataset
to compare embedding quality.

Usage:
    python scripts/build_enriched_index.py                    # Build all variants
    python scripts/build_enriched_index.py --variant optimized  # Build specific variant
    python scripts/build_enriched_index.py --evaluate         # Build and evaluate
"""

import sys
import json
import pickle
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding import Specter2Encoder
from src.indexer import FaissIndexer

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENRICHED_DIR = DATA_DIR / "enriched"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# Input file
ENRICHED_RESEARCHERS = DATA_DIR / "raw" / "researchers_enriched.json"

# Text variants to test
TEXT_VARIANTS = [
    "raw_text_original",   # Baseline - original unmodified
    "raw_text_cleaned",    # Artifacts removed
    "raw_text_enriched",   # Cleaned + publications + keywords
    "raw_text_optimized",  # Fully structured format
    "raw_text_keywords",   # Keywords-dense format
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build enriched FAISS index")
    parser.add_argument("--variant", type=str, default="all",
                       choices=TEXT_VARIANTS + ["all"],
                       help="Which text variant to use (default: all)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation after building")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for embedding (default: 8)")
    return parser.parse_args()


def load_enriched_researchers():
    """Load enriched researchers data."""
    print(f"Loading enriched data from {ENRICHED_RESEARCHERS}...")

    with open(ENRICHED_RESEARCHERS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    researchers = data.get('researchers', data)
    print(f"Loaded {len(researchers)} researchers")

    return researchers


def build_index_for_variant(
    researchers: list,
    variant: str,
    encoder: Specter2Encoder,
    batch_size: int = 8
) -> tuple:
    """
    Build index for a specific text variant.

    Returns:
        (researchers_with_emb, faiss_index, stats)
    """
    print(f"\n{'='*60}")
    print(f"Building index for variant: {variant}")
    print(f"{'='*60}")

    # Prepare texts
    texts = []
    valid_researchers = []

    for r in researchers:
        text = r.get(variant, '')

        # Fallback to raw_text if variant not available
        if not text or len(text) < 50:
            text = r.get('raw_text', '')

        if text and len(text) >= 50:
            texts.append(text)
            valid_researchers.append(r)

    print(f"Valid researchers: {len(valid_researchers)}/{len(researchers)}")

    if not valid_researchers:
        print("ERROR: No valid researchers found!")
        return None, None, None

    # Calculate text statistics
    text_lengths = [len(t) for t in texts]
    avg_len = sum(text_lengths) / len(text_lengths)
    min_len = min(text_lengths)
    max_len = max(text_lengths)

    print(f"Text length stats: avg={avg_len:.0f}, min={min_len}, max={max_len}")

    # Generate embeddings
    print(f"\nGenerating embeddings (batch_size={batch_size})...")
    start_time = time.time()

    # Use encode_batch for efficient batch processing
    embeddings = encoder.encode_batch(texts, batch_size=batch_size, show_progress=True)

    elapsed = time.time() - start_time
    print(f"Embedding generation complete: {elapsed:.1f}s ({len(texts)/elapsed:.1f} researchers/s)")

    # Add embeddings to researchers
    for i, r in enumerate(valid_researchers):
        r['embedding'] = embeddings[i]

    # Build FAISS index
    print("\nBuilding FAISS index...")
    indexer = FaissIndexer()
    indexer.build(embeddings)

    stats = {
        "variant": variant,
        "total_researchers": len(researchers),
        "valid_researchers": len(valid_researchers),
        "avg_text_length": avg_len,
        "min_text_length": min_len,
        "max_text_length": max_len,
        "embedding_time_seconds": elapsed,
    }

    return valid_researchers, indexer, stats


def save_index(
    researchers_with_emb: list,
    indexer: FaissIndexer,
    variant: str,
    stats: dict
):
    """Save the index and processed data."""
    # Create variant-specific filenames
    pkl_file = PROCESSED_DIR / f"researchers_emb_{variant}.pkl"
    index_file = INDEX_DIR / f"faiss_{variant}.index"
    stats_file = ENRICHED_DIR / f"build_stats_{variant}.json"

    # Create directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    # Save researchers with embeddings
    print(f"Saving {pkl_file}...")
    with open(pkl_file, 'wb') as f:
        pickle.dump(researchers_with_emb, f)

    # Save FAISS index
    print(f"Saving {index_file}...")
    indexer.save(str(index_file))

    # Save stats
    stats["pkl_file"] = str(pkl_file)
    stats["index_file"] = str(index_file)
    stats["build_time"] = datetime.now().isoformat()

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats to {stats_file}")


def run_quick_sanity_check(researchers_with_emb: list, indexer: FaissIndexer):
    """Run a quick sanity check on the built index."""
    import numpy as np

    print("\n--- Quick Sanity Check ---")

    # Test query: find similar to first researcher
    test_researcher = researchers_with_emb[0]
    test_emb = np.array(test_researcher['embedding']).astype('float32')

    distances, indices = indexer.search(test_emb, top_k=5)

    print(f"Query: {test_researcher['name']}")
    print(f"Top 5 similar researchers:")
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        r = researchers_with_emb[idx]
        print(f"  {i+1}. {r['name']} (distance: {dist:.4f})")


def main():
    args = parse_args()

    # Load data
    researchers = load_enriched_researchers()

    # Initialize encoder
    print("\nInitializing SPECTER2 encoder...")
    encoder = Specter2Encoder()
    encoder.load()
    print("Encoder ready")

    # Determine which variants to build
    if args.variant == "all":
        variants = TEXT_VARIANTS
    else:
        variants = [args.variant]

    # Build indices
    all_stats = []

    for variant in variants:
        researchers_with_emb, indexer, stats = build_index_for_variant(
            researchers.copy(),  # Copy to avoid mutation
            variant,
            encoder,
            batch_size=args.batch_size
        )

        if researchers_with_emb and indexer:
            save_index(researchers_with_emb, indexer, variant, stats)
            run_quick_sanity_check(researchers_with_emb, indexer)
            all_stats.append(stats)

    # Summary
    print("\n" + "="*60)
    print("BUILD SUMMARY")
    print("="*60)

    for stats in all_stats:
        print(f"\n{stats['variant']}:")
        print(f"  Researchers: {stats['valid_researchers']}")
        print(f"  Avg text length: {stats['avg_text_length']:.0f} chars")
        print(f"  Build time: {stats['embedding_time_seconds']:.1f}s")

    # Create symlinks for the "best" variant (optimized) as default
    if "raw_text_optimized" in variants:
        default_pkl = PROCESSED_DIR / "researchers_with_emb.pkl"
        default_index = INDEX_DIR / "faiss.index"
        optimized_pkl = PROCESSED_DIR / "researchers_emb_raw_text_optimized.pkl"
        optimized_index = INDEX_DIR / "faiss_raw_text_optimized.index"

        # Copy files (symlinks don't work well with pickle)
        import shutil
        if optimized_pkl.exists():
            shutil.copy(optimized_pkl, default_pkl)
            print(f"\nCopied {optimized_pkl.name} → {default_pkl.name}")
        if optimized_index.exists():
            shutil.copy(optimized_index, default_index)
            print(f"Copied {optimized_index.name} → {default_index.name}")

    print("\n✅ Index building complete!")

    if args.evaluate:
        print("\n" + "="*60)
        print("Running evaluation...")
        print("="*60)
        # TODO: Call evaluation script
        print("Evaluation not yet implemented in this script.")
        print("Run: python ground_truth/evaluate_with_ground_truth.py")


if __name__ == "__main__":
    main()
