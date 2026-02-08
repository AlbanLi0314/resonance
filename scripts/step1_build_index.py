#!/usr/bin/env python3
"""
Step 1: Build FAISS Index (with Checkpoint/Resume for 10K+ scale)
==================================================================
This script:
1. Loads researchers.json
2. Validates and filters data
3. Generates embeddings with SPECTER2 (with checkpointing)
4. Builds FAISS index
5. Saves processed data and index

Designed for 10,000+ researchers with:
- Checkpoint every N researchers (default: 500)
- Auto-resume from last checkpoint if interrupted
- Per-item error handling (bad entries don't crash the job)
- Progress tracking with ETA

Run this ONCE after getting data from scraper team.
After this, you don't need GPU anymore.

Usage:
    python scripts/step1_build_index.py
    python scripts/step1_build_index.py --checkpoint-size 1000
    python scripts/step1_build_index.py --clean  # Remove checkpoints and start fresh

Or in Colab:
    !python step1_build_index.py
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

from src.config import (
    RESEARCHERS_JSON,
    RESEARCHERS_WITH_EMB_PKL,
    FAISS_INDEX_FILE,
    MIN_RAW_TEXT_LENGTH,
    PROCESSED_DATA_DIR,
    print_config
)
from src.embedding import Specter2Encoder, format_researcher_text
from src.indexer import FaissIndexer
from src.data_validator import validate_dataset, print_validation_report, save_validation_report
from src.sanity_check import run_sanity_check

# Checkpoint configuration
CHECKPOINT_DIR = PROCESSED_DATA_DIR / "checkpoints"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"
DEFAULT_CHECKPOINT_SIZE = 500


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build FAISS index with checkpointing")
    parser.add_argument("--checkpoint-size", type=int, default=DEFAULT_CHECKPOINT_SIZE,
                        help=f"Save checkpoint every N researchers (default: {DEFAULT_CHECKPOINT_SIZE})")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all checkpoints and start fresh")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for embedding (default: 8)")
    parser.add_argument("--use-expanded", action="store_true",
                        help="Use expanded dataset (researchers_expanded.json)")
    parser.add_argument("--text-format", type=str, default="optimized",
                        choices=["optimized", "raw_text", "structured", "papers_only"],
                        help="Text format for embedding (default: optimized)")
    return parser.parse_args()


def clean_checkpoints():
    """Remove all checkpoint files"""
    if CHECKPOINT_DIR.exists():
        import shutil
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"Removed checkpoint directory: {CHECKPOINT_DIR}")


def load_progress() -> dict:
    """Load progress from checkpoint"""
    if not PROGRESS_FILE.exists():
        return {"processed_indices": [], "failed_indices": [], "last_updated": None}

    with open(PROGRESS_FILE, 'r') as f:
        return json.load(f)


def save_progress(progress: dict):
    """Save progress to checkpoint file"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def save_checkpoint(researchers_with_emb: list, checkpoint_num: int):
    """Save a batch of researchers with embeddings"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{checkpoint_num:04d}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(researchers_with_emb, f)
    print(f"  💾 Checkpoint saved: {checkpoint_file.name} ({len(researchers_with_emb)} researchers)")


def load_all_checkpoints() -> list:
    """Load and merge all checkpoint files"""
    if not CHECKPOINT_DIR.exists():
        return []

    all_researchers = []
    checkpoint_files = sorted(CHECKPOINT_DIR.glob("checkpoint_*.pkl"))

    for cf in checkpoint_files:
        with open(cf, 'rb') as f:
            batch = pickle.load(f)
            all_researchers.extend(batch)

    return all_researchers


def load_researchers(json_path: str) -> tuple:
    """
    Load researchers from JSON file.

    Returns:
        (researchers_list, metadata_dict)
    """
    print(f"Loading data from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    researchers = data.get('researchers', [])
    metadata = data.get('metadata', {})

    print(f"Loaded {len(researchers)} researchers")
    print(f"Source: {metadata.get('source', 'Unknown')}")
    print(f"Departments: {metadata.get('departments', [])}")

    return researchers, metadata


def validate_and_filter_researchers(researchers: list, min_text_length: int = 50) -> tuple:
    """
    Validate data schema and filter out researchers with insufficient information.

    Returns:
        (valid_researchers, validation_report)
    """
    # Step 1: Schema validation (checks required fields, normalizes aliases, removes duplicates)
    print(f"\n📋 Validating data schema...")
    valid_researchers, validation_report = validate_dataset(researchers)
    print_validation_report(validation_report)

    # Step 2: Additional filtering (raw_text length)
    print(f"\n📏 Filtering by text length (min: {min_text_length} chars)...")
    filtered = []
    short_text_count = 0

    for r in valid_researchers:
        raw_text = r.get('raw_text', '')
        if len(raw_text) < min_text_length:
            short_text_count += 1
            continue
        filtered.append(r)

    if short_text_count > 0:
        print(f"   Filtered out {short_text_count} researchers with short raw_text")
        validation_report["short_text_filtered"] = short_text_count

    print(f"   Final count: {len(filtered)} researchers ready for embedding")

    return filtered, validation_report


def format_time(seconds: float) -> str:
    """Format seconds into human readable string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def generate_embeddings_with_checkpoints(
    researchers: list,
    encoder: Specter2Encoder,
    checkpoint_size: int = 500,
    batch_size: int = 8,
    text_format: str = "optimized"
) -> tuple:
    """
    Generate embeddings with checkpointing and resume capability.

    Returns:
        (researchers_with_embeddings, failed_indices)
    """
    # Load existing progress
    progress = load_progress()
    processed_indices = set(progress.get("processed_indices", []))
    failed_indices = set(progress.get("failed_indices", []))

    # Determine what still needs processing
    all_indices = set(range(len(researchers)))
    remaining_indices = sorted(all_indices - processed_indices - failed_indices)

    if processed_indices:
        print(f"\n📂 Resuming from checkpoint:")
        print(f"   Already processed: {len(processed_indices)}")
        print(f"   Failed: {len(failed_indices)}")
        print(f"   Remaining: {len(remaining_indices)}")

    if not remaining_indices:
        print("All researchers already processed!")
        return load_all_checkpoints(), list(failed_indices)

    print(f"\n🚀 Generating embeddings for {len(remaining_indices)} researchers...")
    print(f"   Checkpoint every: {checkpoint_size} researchers")
    print(f"   Batch size: {batch_size}")

    # Process in checkpoint-sized chunks
    current_batch = []
    checkpoint_num = len(list(CHECKPOINT_DIR.glob("checkpoint_*.pkl"))) if CHECKPOINT_DIR.exists() else 0

    start_time = time.time()

    for i, idx in enumerate(remaining_indices):
        researcher = researchers[idx].copy()  # Don't modify original

        try:
            # Generate embedding for single researcher using optimized text format
            text = format_researcher_text(researcher, text_format)
            embedding = encoder.encode(text)
            researcher['embedding'] = embedding
            researcher['_formatted_text'] = text  # Store for debugging
            researcher['_original_index'] = idx  # Track original position
            current_batch.append(researcher)
            processed_indices.add(idx)

        except Exception as e:
            print(f"\n  ⚠️ Failed to encode researcher {idx} ({researcher.get('name', 'Unknown')}): {e}")
            failed_indices.add(idx)

        # Progress update every 50 items
        if (i + 1) % 50 == 0 or (i + 1) == len(remaining_indices):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining_indices) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{len(remaining_indices)} ({100*(i+1)/len(remaining_indices):.1f}%) "
                  f"| Rate: {rate:.1f}/s | ETA: {format_time(eta)}")

        # Save checkpoint
        if len(current_batch) >= checkpoint_size:
            save_checkpoint(current_batch, checkpoint_num)
            checkpoint_num += 1
            current_batch = []

            # Update progress file
            progress["processed_indices"] = list(processed_indices)
            progress["failed_indices"] = list(failed_indices)
            save_progress(progress)

    # Save final partial batch
    if current_batch:
        save_checkpoint(current_batch, checkpoint_num)
        progress["processed_indices"] = list(processed_indices)
        progress["failed_indices"] = list(failed_indices)
        save_progress(progress)

    # Load all checkpoints and merge
    print(f"\n📦 Merging all checkpoints...")
    all_researchers = load_all_checkpoints()

    # Sort by original index to maintain order
    all_researchers.sort(key=lambda x: x.get('_original_index', 0))

    # Remove tracking field
    for r in all_researchers:
        r.pop('_original_index', None)

    elapsed_total = time.time() - start_time
    print(f"✅ Embedding complete! Total time: {format_time(elapsed_total)}")
    print(f"   Successful: {len(all_researchers)}, Failed: {len(failed_indices)}")

    return all_researchers, list(failed_indices)


def build_and_save_index(researchers: list, index_path: str):
    """Build FAISS index and save"""
    print(f"\n🔨 Building FAISS index...")

    import numpy as np

    # Extract embeddings
    embeddings = np.array([r['embedding'] for r in researchers])

    # Build index
    indexer = FaissIndexer(dimension=embeddings.shape[1])
    indexer.build(embeddings)

    # Save index
    indexer.save(index_path)


def save_researchers(researchers: list, pkl_path: str):
    """Save processed researchers to pickle"""
    print(f"\n💾 Saving processed data to: {pkl_path}")

    # Create directory if needed
    Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, 'wb') as f:
        pickle.dump(researchers, f)

    print(f"Saved {len(researchers)} researchers with embeddings")


def cleanup_checkpoints():
    """Remove checkpoint files after successful completion"""
    if CHECKPOINT_DIR.exists():
        import shutil
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"🧹 Cleaned up checkpoint directory")


def main():
    args = parse_args()

    print("=" * 60)
    print("Step 1: Build FAISS Index")
    print("         (with Checkpoint/Resume support)")
    print("=" * 60)

    # Handle --clean flag
    if args.clean:
        clean_checkpoints()
        print("Starting fresh...\n")

    # Print config
    print_config()

    # Determine which dataset to use
    if args.use_expanded:
        from src.config import DATA_DIR
        input_file = DATA_DIR / "raw" / "researchers_expanded.json"
        print(f"\n📂 Using EXPANDED dataset: {input_file.name}")
    else:
        input_file = RESEARCHERS_JSON

    # Check input file exists
    if not input_file.exists():
        print(f"\n❌ ERROR: Input file not found: {input_file}")
        print("Please make sure the file exists in data/raw/")
        sys.exit(1)

    # Load data
    researchers, metadata = load_researchers(str(input_file))

    # Show text format being used
    print(f"\n📝 Text format: {args.text_format}")
    if args.text_format == "optimized":
        print("   Format: '{name}, {department}: {research_interests}'")
        print("   (This achieved +40% P@1 improvement in experiments)")

    # Validate and filter
    researchers, validation_report = validate_and_filter_researchers(
        researchers, MIN_RAW_TEXT_LENGTH
    )

    # Save validation report
    validation_report_path = PROCESSED_DATA_DIR / "validation_report.json"
    save_validation_report(validation_report, str(validation_report_path))

    if len(researchers) == 0:
        print("\n❌ ERROR: No valid researchers found!")
        print("Check the validation report for details.")
        sys.exit(1)

    # Load encoder
    print("\n🤖 Loading SPECTER2 model...")
    encoder = Specter2Encoder()
    encoder.load()

    # Generate embeddings with checkpointing
    researchers_with_emb, failed = generate_embeddings_with_checkpoints(
        researchers,
        encoder,
        checkpoint_size=args.checkpoint_size,
        batch_size=args.batch_size,
        text_format=args.text_format
    )

    if len(researchers_with_emb) == 0:
        print("\n❌ ERROR: No embeddings generated!")
        sys.exit(1)

    # Report failures
    if failed:
        print(f"\n⚠️ Warning: {len(failed)} researchers failed to encode")
        print(f"   Failed indices: {failed[:10]}{'...' if len(failed) > 10 else ''}")

        # Save failure report
        failure_report = {
            "total_failed": len(failed),
            "failed_indices": failed,
        }
        failure_report_path = PROCESSED_DATA_DIR / "embedding_failures.json"
        with open(failure_report_path, 'w') as f:
            json.dump(failure_report, f, indent=2)
        print(f"   Failure report saved to: {failure_report_path}")

    # Run sanity check on embeddings
    print("\n🔍 Running embedding sanity check...")
    sanity_results = run_sanity_check(researchers_with_emb, verbose=True)

    if not sanity_results["passed"]:
        print("\n⚠️ WARNING: Sanity check found issues with embeddings!")
        print("   Review the warnings above before using this index.")

    # Save sanity check results
    sanity_report_path = PROCESSED_DATA_DIR / "sanity_check_report.json"
    # Convert numpy types for JSON serialization
    sanity_for_json = {k: v for k, v in sanity_results.items()
                       if k not in ["similar_pairs", "dissimilar_pairs"]}
    sanity_for_json["similar_pairs_summary"] = len(sanity_results.get("similar_pairs", []))
    sanity_for_json["dissimilar_pairs_summary"] = len(sanity_results.get("dissimilar_pairs", []))
    with open(sanity_report_path, 'w') as f:
        json.dump(sanity_for_json, f, indent=2)

    # Build index
    build_and_save_index(researchers_with_emb, str(FAISS_INDEX_FILE))

    # Save processed data
    save_researchers(researchers_with_emb, str(RESEARCHERS_WITH_EMB_PKL))

    # Clean up checkpoints
    cleanup_checkpoints()

    # Summary
    print("\n" + "=" * 60)
    print("✅ BUILD COMPLETE!")
    print("=" * 60)
    print(f"Processed researchers: {len(researchers_with_emb)}")
    print(f"Failed researchers: {len(failed)}")
    print(f"Index file: {FAISS_INDEX_FILE}")
    print(f"Data file: {RESEARCHERS_WITH_EMB_PKL}")
    print(f"\nReports saved to: {PROCESSED_DATA_DIR}")
    print("  - validation_report.json")
    print("  - sanity_check_report.json")
    if failed:
        print("  - embedding_failures.json")
    print("\nYou can now run step2_test_search.py to test the search!")


if __name__ == "__main__":
    main()
