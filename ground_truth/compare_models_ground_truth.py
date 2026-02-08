#!/usr/bin/env python3
"""
Compare Embedding Models with Ground Truth
==========================================
Tests multiple embedding models against LLM-generated ground truth labels.
"""
import json
import math
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent
LABELS_FILE = OUTPUT_DIR / "labels" / "ground_truth_v1.json"
PROJECT_ROOT = OUTPUT_DIR.parent


def load_ground_truth():
    """Load ground truth labels organized by query"""
    with open(LABELS_FILE) as f:
        data = json.load(f)

    gt = defaultdict(dict)
    for label in data.get("labels", []):
        query = label["query"]
        researcher = label["researcher_name"]
        score = label.get("relevance_score")
        if score is not None:
            gt[query][researcher] = score
    return dict(gt)


def load_researchers():
    """Load researcher data"""
    with open(PROJECT_ROOT / "data" / "raw" / "researchers.json") as f:
        data = json.load(f)
    return data["researchers"]


def dcg(relevances, k=None):
    if k:
        relevances = relevances[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(relevances, ideal, k=None):
    actual = dcg(relevances, k)
    ideal_dcg = dcg(sorted(ideal, reverse=True), k)
    return actual / ideal_dcg if ideal_dcg > 0 else 0


def evaluate_model(model_name, model_id, researchers, ground_truth, text_format="optimized"):
    """Evaluate a single model against ground truth"""

    print(f"\n  Loading {model_name}...")
    model = SentenceTransformer(model_id)

    # Prepare researcher texts
    researcher_texts = []
    researcher_names = []
    for r in researchers:
        if text_format == "optimized":
            text = f"{r['name']}, {r.get('department', '')}: {r.get('research_interests', r.get('raw_text', '')[:200])}"
        else:
            text = r.get('raw_text', '')[:500]
        researcher_texts.append(text)
        researcher_names.append(r['name'])

    # Encode researchers
    print(f"  Encoding {len(researchers)} researchers...")
    researcher_embeddings = model.encode(researcher_texts, show_progress_bar=False)

    # Evaluate each query
    metrics = {
        'ndcg_5': [], 'ndcg_3': [],
        'p_at_1': [], 'p_at_3': [], 'p_at_5': [],
        'mrr': []
    }

    for query, judgments in ground_truth.items():
        # Encode query
        query_emb = model.encode([query])[0]

        # Compute cosine similarities
        sims = np.dot(researcher_embeddings, query_emb) / (
            np.linalg.norm(researcher_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )

        # Get top 5 results
        top_indices = np.argsort(sims)[-5:][::-1]
        top_names = [researcher_names[i] for i in top_indices]

        # Map to relevance scores
        relevances = [judgments.get(name, 0) for name in top_names]
        ideal = list(judgments.values())

        # Compute metrics
        metrics['ndcg_5'].append(ndcg(relevances, ideal, 5))
        metrics['ndcg_3'].append(ndcg(relevances, ideal, 3))
        metrics['p_at_1'].append(1 if relevances[0] >= 2 else 0)
        metrics['p_at_3'].append(sum(1 for r in relevances[:3] if r >= 2) / 3)
        metrics['p_at_5'].append(sum(1 for r in relevances[:5] if r >= 2) / 5)

        # MRR
        rr = 0
        for i, rel in enumerate(relevances):
            if rel >= 2:
                rr = 1 / (i + 1)
                break
        metrics['mrr'].append(rr)

    # Average metrics
    return {k: sum(v) / len(v) for k, v in metrics.items()}


def main():
    print("=" * 60)
    print("EMBEDDING MODEL COMPARISON WITH GROUND TRUTH")
    print("=" * 60)

    # Load data
    print("\nLoading ground truth...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} queries")

    print("\nLoading researchers...")
    researchers = load_researchers()
    print(f"Loaded {len(researchers)} researchers")

    # Models to test
    models = [
        ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
        ("MPNet", "sentence-transformers/all-mpnet-base-v2"),
        ("BGE-base", "BAAI/bge-base-en-v1.5"),
        ("GTE-base", "thenlper/gte-base"),
    ]

    results = {}

    print("\n" + "-" * 60)
    print("EVALUATING MODELS")
    print("-" * 60)

    for model_name, model_id in models:
        try:
            metrics = evaluate_model(model_name, model_id, researchers, ground_truth)
            results[model_name] = metrics
            print(f"  {model_name}: nDCG@5={metrics['ndcg_5']:.3f}, P@1={metrics['p_at_1']:.1%}, MRR={metrics['mrr']:.3f}")
        except Exception as e:
            print(f"  {model_name}: ERROR - {e}")

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'nDCG@5':>10} {'nDCG@3':>10} {'P@1':>10} {'P@3':>10} {'MRR':>10}")
    print("-" * 65)

    # Sort by nDCG@5
    for model_name, m in sorted(results.items(), key=lambda x: -x[1]['ndcg_5']):
        print(f"{model_name:<15} {m['ndcg_5']:>10.3f} {m['ndcg_3']:>10.3f} {m['p_at_1']:>9.1%} {m['p_at_3']:>9.1%} {m['mrr']:>10.3f}")

    # Compare to current SPECTER2 baseline
    print("-" * 65)
    print(f"{'SPECTER2 (curr)':<15} {'0.823':>10} {'-':>10} {'55.6%':>10} {'-':>10} {'0.633':>10}")
    print("=" * 60)

    # Save results
    output_file = OUTPUT_DIR / "model_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
