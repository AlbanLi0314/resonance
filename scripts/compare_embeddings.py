#!/usr/bin/env python3
"""
Embedding Model Comparison
==========================
Autonomous script to compare different embedding models for academic matching.

Tests 8 models and generates a comprehensive comparison report.
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "embedding_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models to test
MODELS = [
    {
        "name": "specter2",
        "model_id": "allenai/specter2",
        "type": "academic",
        "description": "Current baseline - trained on scientific papers"
    },
    {
        "name": "specter_v1",
        "model_id": "sentence-transformers/allenai-specter",
        "type": "academic",
        "description": "Original SPECTER model"
    },
    {
        "name": "scibert",
        "model_id": "allenai/scibert_scivocab_uncased",
        "type": "scientific",
        "description": "SciBERT - trained on scientific corpus"
    },
    {
        "name": "minilm",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "general",
        "description": "Fast general-purpose model"
    },
    {
        "name": "mpnet",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "type": "general",
        "description": "High-quality general-purpose model"
    },
    {
        "name": "e5_base",
        "model_id": "intfloat/e5-base-v2",
        "type": "general",
        "description": "State-of-the-art retrieval model"
    },
    {
        "name": "bge_base",
        "model_id": "BAAI/bge-base-en-v1.5",
        "type": "general",
        "description": "Top performer on MTEB benchmarks"
    },
    {
        "name": "gte_base",
        "model_id": "thenlper/gte-base",
        "type": "general",
        "description": "General Text Embeddings - strong retrieval"
    },
]


def log(message: str):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_data() -> Tuple[List[Dict], List[Dict]]:
    """Load researchers and test queries"""
    # Load researchers
    researchers_file = DATA_DIR / "overnight_results" / "researchers_large.json"
    if not researchers_file.exists():
        researchers_file = DATA_DIR / "raw" / "researchers.json"

    with open(researchers_file) as f:
        data = json.load(f)
    researchers = data["researchers"]

    # Load queries
    queries_file = DATA_DIR / "overnight_results" / "test_queries_large.json"
    if not queries_file.exists():
        queries_file = DATA_DIR / "test_queries.json"

    with open(queries_file) as f:
        queries_data = json.load(f)
    queries = queries_data.get("queries", [])

    return researchers, queries


def format_researcher_text(researcher: Dict) -> str:
    """Format researcher for embedding (optimized format)"""
    name = researcher.get("name", "Unknown")
    dept = researcher.get("department", "Unknown")
    interests = researcher.get("research_interests", "")
    return f"{name}, {dept}: {interests}"


def calculate_metrics(rankings: List[int], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
    """Calculate retrieval metrics from rankings"""
    metrics = {}

    # MRR
    mrr_sum = 0
    for rank in rankings:
        if rank is not None:
            mrr_sum += 1.0 / rank
    metrics["mrr"] = mrr_sum / len(rankings) if rankings else 0

    # Precision@K
    for k in k_values:
        hits = sum(1 for r in rankings if r is not None and r <= k)
        metrics[f"p@{k}"] = hits / len(rankings) if rankings else 0

    # Recall (found anywhere in top 10)
    found = sum(1 for r in rankings if r is not None)
    metrics["recall@10"] = found / len(rankings) if rankings else 0

    return metrics


def evaluate_model(model_name: str, model_id: str, researchers: List[Dict],
                   queries: List[Dict], distance: str = "cosine") -> Dict:
    """Evaluate a single embedding model"""
    from sentence_transformers import SentenceTransformer

    log(f"  Loading model: {model_id}")
    start_load = time.time()

    try:
        # Handle E5 models which need "query: " prefix
        is_e5 = "e5" in model_id.lower()
        is_bge = "bge" in model_id.lower()

        model = SentenceTransformer(model_id, device="cpu")
        load_time = time.time() - start_load
        log(f"  Model loaded in {load_time:.1f}s")

        # Prepare documents
        documents = [format_researcher_text(r) for r in researchers]
        id_to_idx = {r["id"]: i for i, r in enumerate(researchers)}

        # Encode documents
        log(f"  Encoding {len(documents)} documents...")
        start_encode = time.time()

        if is_e5:
            # E5 models need "passage: " prefix for documents
            doc_texts = ["passage: " + d for d in documents]
        elif is_bge:
            doc_texts = documents  # BGE doesn't need prefix for docs
        else:
            doc_texts = documents

        doc_embeddings = model.encode(doc_texts, show_progress_bar=False, convert_to_numpy=True)
        encode_time = time.time() - start_encode
        docs_per_sec = len(documents) / encode_time
        log(f"  Encoded documents in {encode_time:.1f}s ({docs_per_sec:.1f} docs/sec)")

        # Normalize for cosine similarity
        if distance == "cosine":
            doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Evaluate queries
        rankings = []
        query_times = []

        for q in queries:
            query_text = q.get("query", "")
            expected_id = q.get("expected_id", "")

            if not expected_id or expected_id not in id_to_idx:
                continue

            expected_idx = id_to_idx[expected_id]

            # Encode query
            start_query = time.time()
            if is_e5:
                query_text_encoded = "query: " + query_text
            elif is_bge:
                query_text_encoded = query_text  # BGE works without prefix
            else:
                query_text_encoded = query_text

            query_emb = model.encode([query_text_encoded], show_progress_bar=False, convert_to_numpy=True)[0]

            if distance == "cosine":
                query_emb = query_emb / np.linalg.norm(query_emb)
                similarities = np.dot(doc_embeddings, query_emb)
                top_indices = np.argsort(similarities)[::-1][:10]
            else:  # L2
                distances = np.linalg.norm(doc_embeddings - query_emb, axis=1)
                top_indices = np.argsort(distances)[:10]

            query_times.append(time.time() - start_query)

            # Find rank
            if expected_idx in top_indices:
                rank = np.where(top_indices == expected_idx)[0][0] + 1
                rankings.append(rank)
            else:
                rankings.append(None)

        # Calculate metrics
        metrics = calculate_metrics(rankings)
        metrics["load_time"] = load_time
        metrics["encode_time"] = encode_time
        metrics["docs_per_sec"] = docs_per_sec
        metrics["avg_query_time"] = np.mean(query_times) if query_times else 0
        metrics["embedding_dim"] = doc_embeddings.shape[1]
        metrics["num_queries"] = len(rankings)
        metrics["distance"] = distance
        metrics["status"] = "success"

        return metrics

    except Exception as e:
        log(f"  ERROR: {str(e)[:100]}")
        return {
            "status": "failed",
            "error": str(e)[:200],
            "distance": distance
        }


def run_comparison():
    """Run full model comparison"""
    log("=" * 60)
    log("EMBEDDING MODEL COMPARISON")
    log("=" * 60)

    # Load data
    log("Loading data...")
    researchers, queries = load_data()
    log(f"Loaded {len(researchers)} researchers, {len(queries)} queries")

    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_researchers": len(researchers),
        "num_queries": len(queries),
        "models": []
    }

    # Test each model
    for i, model_info in enumerate(MODELS):
        log(f"\n[{i+1}/{len(MODELS)}] Testing: {model_info['name']}")
        log(f"  Type: {model_info['type']}")

        model_result = {
            "name": model_info["name"],
            "model_id": model_info["model_id"],
            "type": model_info["type"],
            "description": model_info["description"],
            "evaluations": []
        }

        # Test with cosine distance
        log("  Testing with cosine distance...")
        cosine_metrics = evaluate_model(
            model_info["name"],
            model_info["model_id"],
            researchers,
            queries,
            distance="cosine"
        )
        model_result["evaluations"].append(cosine_metrics)

        if cosine_metrics.get("status") == "success":
            log(f"  Cosine - MRR: {cosine_metrics['mrr']:.4f}, P@1: {cosine_metrics['p@1']:.4f}, P@5: {cosine_metrics['p@5']:.4f}")

        # Test with L2 distance
        log("  Testing with L2 distance...")
        l2_metrics = evaluate_model(
            model_info["name"],
            model_info["model_id"],
            researchers,
            queries,
            distance="l2"
        )
        model_result["evaluations"].append(l2_metrics)

        if l2_metrics.get("status") == "success":
            log(f"  L2 - MRR: {l2_metrics['mrr']:.4f}, P@1: {l2_metrics['p@1']:.4f}, P@5: {l2_metrics['p@5']:.4f}")

        results["models"].append(model_result)

        # Save intermediate results
        with open(OUTPUT_DIR / "results_by_model.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Find best model
    best_mrr = 0
    best_config = None

    for model in results["models"]:
        for eval_result in model["evaluations"]:
            if eval_result.get("status") == "success":
                mrr = eval_result.get("mrr", 0)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_config = {
                        "model_name": model["name"],
                        "model_id": model["model_id"],
                        "distance": eval_result["distance"],
                        "mrr": mrr,
                        "p@1": eval_result.get("p@1", 0),
                        "p@5": eval_result.get("p@5", 0)
                    }

    results["best_config"] = best_config

    # Save final results
    with open(OUTPUT_DIR / "results_by_model.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "best_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)

    # Generate report
    generate_report(results)

    log("\n" + "=" * 60)
    log("COMPARISON COMPLETE")
    log("=" * 60)
    log(f"Best model: {best_config['model_name']} ({best_config['distance']})")
    log(f"Best MRR: {best_mrr:.4f}")
    log(f"Results saved to: {OUTPUT_DIR}")

    return results


def generate_report(results: Dict):
    """Generate markdown comparison report"""
    report = f"""# Embedding Model Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Researchers tested**: {results['num_researchers']}
- **Queries tested**: {results['num_queries']}
- **Models compared**: {len(results['models'])}

## Best Configuration

"""

    if results.get("best_config"):
        bc = results["best_config"]
        report += f"""| Setting | Value |
|---------|-------|
| Model | **{bc['model_name']}** |
| Model ID | `{bc['model_id']}` |
| Distance | {bc['distance']} |
| MRR | **{bc['mrr']:.4f}** |
| P@1 | {bc['p@1']:.2%} |
| P@5 | {bc['p@5']:.2%} |

"""

    # Results table
    report += """## Full Results

### Cosine Distance

| Model | Type | MRR | P@1 | P@5 | P@10 | Dim | Speed |
|-------|------|-----|-----|-----|------|-----|-------|
"""

    for model in results["models"]:
        for eval_r in model["evaluations"]:
            if eval_r.get("distance") == "cosine" and eval_r.get("status") == "success":
                report += f"| {model['name']} | {model['type']} | "
                report += f"{eval_r['mrr']:.4f} | {eval_r['p@1']:.2%} | {eval_r['p@5']:.2%} | "
                report += f"{eval_r.get('recall@10', 0):.2%} | {eval_r.get('embedding_dim', 'N/A')} | "
                report += f"{eval_r.get('docs_per_sec', 0):.1f}/s |\n"
            elif eval_r.get("distance") == "cosine" and eval_r.get("status") == "failed":
                report += f"| {model['name']} | {model['type']} | FAILED | - | - | - | - | - |\n"

    report += """
### L2 Distance

| Model | Type | MRR | P@1 | P@5 | P@10 |
|-------|------|-----|-----|-----|------|
"""

    for model in results["models"]:
        for eval_r in model["evaluations"]:
            if eval_r.get("distance") == "l2" and eval_r.get("status") == "success":
                report += f"| {model['name']} | {model['type']} | "
                report += f"{eval_r['mrr']:.4f} | {eval_r['p@1']:.2%} | {eval_r['p@5']:.2%} | "
                report += f"{eval_r.get('recall@10', 0):.2%} |\n"
            elif eval_r.get("distance") == "l2" and eval_r.get("status") == "failed":
                report += f"| {model['name']} | {model['type']} | FAILED | - | - | - |\n"

    report += """
## Model Descriptions

| Model | Description |
|-------|-------------|
"""
    for model in results["models"]:
        report += f"| {model['name']} | {model['description']} |\n"

    report += """
## Recommendations

1. **Use the best performing model** identified above
2. **Consider speed vs accuracy tradeoff** - MiniLM is fastest, but may sacrifice quality
3. **Test with real data** - synthetic data may not reflect real-world performance
4. **Combine with hybrid search** - BM25 + embeddings often outperforms either alone

## Next Steps

1. Integrate best model into main matcher
2. Re-run hybrid search experiments with new embeddings
3. Test with real Cornell researcher data when available
"""

    with open(OUTPUT_DIR / "EMBEDDING_COMPARISON.md", 'w') as f:
        f.write(report)

    log(f"Report saved to: {OUTPUT_DIR / 'EMBEDDING_COMPARISON.md'}")


if __name__ == "__main__":
    run_comparison()
