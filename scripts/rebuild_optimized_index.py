#!/usr/bin/env python3
"""
Rebuild Index with Optimized Text
==================================
Cleans researcher profiles and rebuilds the FAISS index.

This should improve matching by:
1. Removing navigation/scraping noise
2. Prioritizing research-relevant content
3. Creating structured embedding text
"""
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_optimizer import create_embedding_text, clean_text, create_optimized_corpus
from src.embedding import Specter2Encoder
from src.indexer import FaissIndexer

# Paths
RESEARCHERS_PKL = PROJECT_ROOT / "data" / "processed" / "researchers_with_emb.pkl"
OUTPUT_PKL = PROJECT_ROOT / "data" / "processed" / "researchers_optimized.pkl"
OUTPUT_INDEX = PROJECT_ROOT / "data" / "index" / "faiss_optimized.index"


def main():
    print("=" * 60)
    print("REBUILDING INDEX WITH OPTIMIZED TEXT")
    print("=" * 60)

    # Load researchers
    print("\n1. Loading researchers...")
    with open(RESEARCHERS_PKL, 'rb') as f:
        researchers = pickle.load(f)
    print(f"   Loaded {len(researchers)} researchers")

    # Create optimized text
    print("\n2. Optimizing text...")
    for i, r in enumerate(researchers):
        original_text = r.get('raw_text', '')
        optimized_text = create_embedding_text(r, format_type="optimized")

        r['original_raw_text'] = original_text
        r['raw_text'] = optimized_text  # Replace for embedding

        if i < 3:  # Show samples
            print(f"\n   Sample {i+1}: {r['name']}")
            print(f"   Original length: {len(original_text)}")
            print(f"   Optimized length: {len(optimized_text)}")
            print(f"   Optimized text preview: {optimized_text[:200]}...")

    # Analyze improvement
    print("\n3. Analyzing text quality...")
    orig_lens = [len(r.get('original_raw_text', '')) for r in researchers]
    opt_lens = [len(r.get('raw_text', '')) for r in researchers]

    print(f"   Original avg length: {sum(orig_lens)/len(orig_lens):.0f} chars")
    print(f"   Optimized avg length: {sum(opt_lens)/len(opt_lens):.0f} chars")

    # Check for short optimized texts
    short = sum(1 for l in opt_lens if l < 100)
    print(f"   Very short (<100 chars): {short} researchers")

    # Load encoder
    print("\n4. Loading SPECTER2 encoder...")
    encoder = Specter2Encoder()
    encoder.load()

    # Generate embeddings
    print("\n5. Generating embeddings...")
    texts = [r['raw_text'] for r in researchers]
    embeddings = encoder.encode_batch(texts, batch_size=8, show_progress=True)
    print(f"   Generated {len(embeddings)} embeddings of shape {embeddings.shape}")

    # Update researchers with embeddings
    for i, r in enumerate(researchers):
        r['embedding'] = embeddings[i]

    # Save optimized researchers
    print("\n6. Saving optimized data...")
    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(researchers, f)
    print(f"   Saved to: {OUTPUT_PKL}")

    # Build and save index
    print("\n7. Building FAISS index...")
    indexer = FaissIndexer()
    indexer.build(embeddings)
    indexer.save(str(OUTPUT_INDEX))
    print(f"   Saved to: {OUTPUT_INDEX}")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Optimized researchers: {OUTPUT_PKL}")
    print(f"Optimized index: {OUTPUT_INDEX}")
    print("\nTo evaluate, run:")
    print("  python ground_truth/evaluate_optimized.py")


if __name__ == "__main__":
    main()
