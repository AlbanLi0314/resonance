#!/usr/bin/env python3
"""
Embedding Optimization Experiments
===================================
Systematically tests different embedding configurations to find the optimal
setup for academic researcher matching.

Run with: python scripts/run_experiments.py --all
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, PROCESSED_DATA_DIR

# ============================================================
# Configuration
# ============================================================

EXPERIMENT_DIR = DATA_DIR / "experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Load researcher data - use expanded dataset if available
RESEARCHERS_JSON_EXPANDED = DATA_DIR / "raw" / "researchers_expanded.json"
RESEARCHERS_JSON_BASIC = DATA_DIR / "raw" / "researchers.json"
RESEARCHERS_JSON = RESEARCHERS_JSON_EXPANDED if RESEARCHERS_JSON_EXPANDED.exists() else RESEARCHERS_JSON_BASIC

# Test queries file
TEST_QUERIES_EXPANDED = DATA_DIR / "test_queries_expanded.json"
TEST_QUERIES_BASIC = DATA_DIR / "test_queries.json"

# Ground truth will be loaded dynamically from test queries file
GROUND_TRUTH = {}

# Text format functions
def format_raw_text(r: dict) -> str:
    """T1: Full raw_text (baseline)"""
    return r.get("raw_text", "")

def format_research_interests(r: dict) -> str:
    """T2: Only research interests"""
    return r.get("research_interests", "")

def format_structured(r: dict) -> str:
    """T3: Structured format"""
    interests = r.get("research_interests", "")
    papers = r.get("papers", [])
    paper_titles = ", ".join([p.get("title", "") for p in papers[:3]])
    return f"Research focus: {interests}. Recent work: {paper_titles}"

def format_papers_only(r: dict) -> str:
    """T4: Papers only"""
    papers = r.get("papers", [])
    texts = []
    for p in papers:
        texts.append(f"{p.get('title', '')}. {p.get('abstract', '')}")
    return " ".join(texts)

def format_minimal(r: dict) -> str:
    """T5: Minimal structured"""
    name = r.get("name", "")
    dept = r.get("department", "")
    interests = r.get("research_interests", "")
    return f"{name}, {dept}: {interests}"

TEXT_FORMATS = {
    "T1_raw_text": format_raw_text,
    "T2_research_interests": format_research_interests,
    "T3_structured": format_structured,
    "T4_papers_only": format_papers_only,
    "T5_minimal": format_minimal,
}

# Model configurations
MODELS = {
    "M1_specter2": {
        "name": "allenai/specter2_base",
        "type": "specter2",
        "adapter": "allenai/specter2",
    },
    "M2_minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence-transformer",
    },
    "M3_bge": {
        "name": "BAAI/bge-base-en-v1.5",
        "type": "sentence-transformer",
    },
    "M4_nomic": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "type": "sentence-transformer",
        "trust_remote_code": True,
    },
    "M5_mpnet": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "type": "sentence-transformer",
    },
}

DISTANCE_METRICS = ["L2", "cosine", "IP"]


# ============================================================
# Embedding Encoder Classes
# ============================================================

class BaseEncoder:
    """Base class for encoders"""
    def __init__(self, model_config: dict):
        self.config = model_config
        self.model = None

    def load(self):
        raise NotImplementedError

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class Specter2Encoder(BaseEncoder):
    """SPECTER2 with adapter"""
    def load(self):
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel
        import torch

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["name"])
        self.model = AutoAdapterModel.from_pretrained(self.config["name"])
        self.model.load_adapter(self.config["adapter"], source="hf", set_active=True)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        import torch
        embeddings = []
        batch_size = 8

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_emb)

        return np.vstack(embeddings)


class SentenceTransformerEncoder(BaseEncoder):
    """Sentence Transformers models"""
    def load(self):
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self.config.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True

        self.model = SentenceTransformer(self.config["name"], **kwargs)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


def get_encoder(model_id: str) -> BaseEncoder:
    """Factory function for encoders"""
    config = MODELS[model_id]
    if config["type"] == "specter2":
        return Specter2Encoder(config)
    else:
        return SentenceTransformerEncoder(config)


# ============================================================
# Metrics Calculation
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate L2 distance"""
    return float(np.linalg.norm(a - b))


def calculate_similarity_matrix(embeddings: np.ndarray, metric: str) -> np.ndarray:
    """Calculate pairwise similarity/distance matrix"""
    n = len(embeddings)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if metric == "cosine":
                matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
            elif metric == "L2":
                # Convert distance to similarity (smaller distance = higher similarity)
                dist = l2_distance(embeddings[i], embeddings[j])
                matrix[i, j] = 1 / (1 + dist)
            elif metric == "IP":
                matrix[i, j] = float(np.dot(embeddings[i], embeddings[j]))

    return matrix


def calculate_intra_inter_ratio(
    embeddings: np.ndarray,
    researchers: List[dict],
    metric: str
) -> Tuple[float, float, float]:
    """
    Calculate intra-department vs inter-department similarity ratio
    Returns: (intra_sim, inter_sim, ratio)
    """
    sim_matrix = calculate_similarity_matrix(embeddings, metric)

    # Group by department
    dept_indices = {}
    for i, r in enumerate(researchers):
        dept = r.get("department", "Unknown")
        if dept not in dept_indices:
            dept_indices[dept] = []
        dept_indices[dept].append(i)

    intra_sims = []
    inter_sims = []

    for i in range(len(researchers)):
        for j in range(i + 1, len(researchers)):
            dept_i = researchers[i].get("department", "")
            dept_j = researchers[j].get("department", "")

            sim = sim_matrix[i, j]

            if dept_i == dept_j:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    intra_avg = np.mean(intra_sims) if intra_sims else 0
    inter_avg = np.mean(inter_sims) if inter_sims else 0
    ratio = intra_avg / inter_avg if inter_avg > 0 else 0

    return float(intra_avg), float(inter_avg), float(ratio)


def search_with_embeddings(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    metric: str,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Search for most similar documents
    Returns: List of (index, score) tuples
    """
    scores = []

    for i, doc_emb in enumerate(doc_embeddings):
        if metric == "cosine":
            score = cosine_similarity(query_embedding, doc_emb)
        elif metric == "L2":
            dist = l2_distance(query_embedding, doc_emb)
            score = 1 / (1 + dist)
        elif metric == "IP":
            score = float(np.dot(query_embedding, doc_emb))
        scores.append((i, score))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def calculate_retrieval_metrics(
    encoder: BaseEncoder,
    researchers: List[dict],
    doc_embeddings: np.ndarray,
    text_format_fn,
    metric: str
) -> Dict:
    """
    Calculate MRR and Precision@K for ground truth queries
    """
    reciprocal_ranks = []
    precision_at_1 = []
    precision_at_3 = []
    precision_at_5 = []

    # Create ID to index mapping
    id_to_idx = {r["id"]: i for i, r in enumerate(researchers)}

    for query, expected_id in GROUND_TRUTH.items():
        if expected_id not in id_to_idx:
            continue

        expected_idx = id_to_idx[expected_id]

        # Encode query
        query_emb = encoder.encode([query])[0]

        # Search
        results = search_with_embeddings(query_emb, doc_embeddings, metric, top_k=10)
        result_indices = [idx for idx, _ in results]

        # Calculate rank
        if expected_idx in result_indices:
            rank = result_indices.index(expected_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
            precision_at_1.append(1 if rank == 1 else 0)
            precision_at_3.append(1 if rank <= 3 else 0)
            precision_at_5.append(1 if rank <= 5 else 0)
        else:
            reciprocal_ranks.append(0)
            precision_at_1.append(0)
            precision_at_3.append(0)
            precision_at_5.append(0)

    return {
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0,
        "precision_at_1": float(np.mean(precision_at_1)) if precision_at_1 else 0,
        "precision_at_3": float(np.mean(precision_at_3)) if precision_at_3 else 0,
        "precision_at_5": float(np.mean(precision_at_5)) if precision_at_5 else 0,
        "num_queries": len(reciprocal_ranks),
    }


def calculate_embedding_stats(embeddings: np.ndarray) -> Dict:
    """Calculate embedding statistics"""
    return {
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        "variance": float(np.mean(np.var(embeddings, axis=0))),
        "dimension": embeddings.shape[1],
    }


# ============================================================
# Experiment Runner
# ============================================================

@dataclass
class ExperimentResult:
    experiment_id: str
    text_format: str
    model: str
    distance_metric: str
    intra_sim: float
    inter_sim: float
    intra_inter_ratio: float
    mrr: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    embedding_variance: float
    embedding_dim: int
    encode_time_sec: float
    timestamp: str


def run_single_experiment(
    text_format_id: str,
    model_id: str,
    metric: str,
    researchers: List[dict],
    encoder: Optional[BaseEncoder] = None,
    cached_embeddings: Optional[np.ndarray] = None,
) -> ExperimentResult:
    """Run a single experiment configuration"""

    experiment_id = f"{text_format_id}_{model_id}_{metric}"
    print(f"\n{'='*60}")
    print(f"Running: {experiment_id}")
    print(f"{'='*60}")

    # Get text format function
    text_fn = TEXT_FORMATS[text_format_id]

    # Prepare texts
    texts = [text_fn(r) for r in researchers]
    print(f"  Text format: {text_format_id}")
    print(f"  Sample text: {texts[0][:100]}...")

    # Load encoder if not provided
    if encoder is None:
        print(f"  Loading model: {model_id}...")
        encoder = get_encoder(model_id)
        encoder.load()

    # Generate embeddings if not cached
    if cached_embeddings is None:
        print(f"  Generating embeddings...")
        start_time = time.time()
        embeddings = encoder.encode(texts)
        encode_time = time.time() - start_time
        print(f"  Encode time: {encode_time:.2f}s")
    else:
        embeddings = cached_embeddings
        encode_time = 0

    # Calculate metrics
    print(f"  Calculating metrics (metric={metric})...")

    intra_sim, inter_sim, ratio = calculate_intra_inter_ratio(embeddings, researchers, metric)
    print(f"  Intra-dept similarity: {intra_sim:.4f}")
    print(f"  Inter-dept similarity: {inter_sim:.4f}")
    print(f"  Ratio: {ratio:.4f}")

    retrieval_metrics = calculate_retrieval_metrics(
        encoder, researchers, embeddings, text_fn, metric
    )
    print(f"  MRR: {retrieval_metrics['mrr']:.4f}")
    print(f"  P@1: {retrieval_metrics['precision_at_1']:.4f}")

    emb_stats = calculate_embedding_stats(embeddings)

    result = ExperimentResult(
        experiment_id=experiment_id,
        text_format=text_format_id,
        model=model_id,
        distance_metric=metric,
        intra_sim=intra_sim,
        inter_sim=inter_sim,
        intra_inter_ratio=ratio,
        mrr=retrieval_metrics["mrr"],
        precision_at_1=retrieval_metrics["precision_at_1"],
        precision_at_3=retrieval_metrics["precision_at_3"],
        precision_at_5=retrieval_metrics["precision_at_5"],
        embedding_variance=emb_stats["variance"],
        embedding_dim=emb_stats["dimension"],
        encode_time_sec=encode_time,
        timestamp=datetime.now().isoformat(),
    )

    return result, encoder, embeddings


def load_researchers() -> List[dict]:
    """Load researcher data"""
    with open(RESEARCHERS_JSON) as f:
        data = json.load(f)
    return data["researchers"]


def load_ground_truth() -> dict:
    """Load ground truth mappings from test queries file"""
    global GROUND_TRUTH

    queries_file = TEST_QUERIES_EXPANDED if TEST_QUERIES_EXPANDED.exists() else TEST_QUERIES_BASIC
    with open(queries_file) as f:
        data = json.load(f)

    for q in data.get("queries", []):
        query = q.get("query", "")
        expected_id = q.get("expected_id", "")
        if query and expected_id:
            GROUND_TRUTH[query] = expected_id

    print(f"Loaded {len(GROUND_TRUTH)} ground truth queries from {queries_file.name}")
    return GROUND_TRUTH


def run_phase_1(researchers: List[dict], results: List[ExperimentResult]) -> str:
    """Phase 1: Text Format Optimization with baseline model"""
    print("\n" + "="*70)
    print("PHASE 1: Text Format Optimization")
    print("="*70)

    model_id = "M1_specter2"
    metric = "cosine"  # Use cosine for comparison

    encoder = None
    best_format = None
    best_ratio = 0

    for text_format_id in TEXT_FORMATS.keys():
        result, encoder, _ = run_single_experiment(
            text_format_id, model_id, metric, researchers, encoder
        )
        results.append(result)

        if result.intra_inter_ratio > best_ratio:
            best_ratio = result.intra_inter_ratio
            best_format = text_format_id

    print(f"\n>>> Best text format: {best_format} (ratio={best_ratio:.4f})")
    return best_format


def run_phase_2(researchers: List[dict], results: List[ExperimentResult], best_text_format: str) -> str:
    """Phase 2: Model Comparison with best text format"""
    print("\n" + "="*70)
    print("PHASE 2: Model Comparison")
    print("="*70)

    metric = "cosine"
    best_model = None
    best_mrr = 0

    for model_id in MODELS.keys():
        try:
            result, _, _ = run_single_experiment(
                best_text_format, model_id, metric, researchers
            )
            results.append(result)

            # Use MRR as primary metric for model selection
            if result.mrr > best_mrr:
                best_mrr = result.mrr
                best_model = model_id
        except Exception as e:
            print(f"  ERROR with {model_id}: {e}")
            continue

    print(f"\n>>> Best model: {best_model} (MRR={best_mrr:.4f})")
    return best_model


def run_phase_3(
    researchers: List[dict],
    results: List[ExperimentResult],
    best_text_format: str,
    best_model: str
) -> str:
    """Phase 3: Distance Metric Optimization"""
    print("\n" + "="*70)
    print("PHASE 3: Distance Metric Optimization")
    print("="*70)

    encoder = get_encoder(best_model)
    encoder.load()

    text_fn = TEXT_FORMATS[best_text_format]
    texts = [text_fn(r) for r in researchers]
    embeddings = encoder.encode(texts)

    best_metric = None
    best_mrr = 0

    for metric in DISTANCE_METRICS:
        result, _, _ = run_single_experiment(
            best_text_format, best_model, metric, researchers, encoder, embeddings
        )
        results.append(result)

        if result.mrr > best_mrr:
            best_mrr = result.mrr
            best_metric = metric

    print(f"\n>>> Best distance metric: {best_metric} (MRR={best_mrr:.4f})")
    return best_metric


def run_phase_4(
    researchers: List[dict],
    results: List[ExperimentResult],
    best_text_format: str,
    best_model: str,
    best_metric: str
):
    """Phase 4: Final Validation and Comparison"""
    print("\n" + "="*70)
    print("PHASE 4: Final Validation")
    print("="*70)

    # Find baseline result
    baseline = None
    optimized = None

    for r in results:
        if r.text_format == "T1_raw_text" and r.model == "M1_specter2" and r.distance_metric == "cosine":
            baseline = r
        if r.text_format == best_text_format and r.model == best_model and r.distance_metric == best_metric:
            optimized = r

    if baseline and optimized:
        print("\n" + "-"*50)
        print("COMPARISON: Baseline vs Optimized")
        print("-"*50)
        print(f"{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
        print("-"*50)

        metrics = [
            ("Intra/Inter Ratio", baseline.intra_inter_ratio, optimized.intra_inter_ratio),
            ("MRR", baseline.mrr, optimized.mrr),
            ("Precision@1", baseline.precision_at_1, optimized.precision_at_1),
            ("Precision@3", baseline.precision_at_3, optimized.precision_at_3),
            ("Precision@5", baseline.precision_at_5, optimized.precision_at_5),
        ]

        for name, base_val, opt_val in metrics:
            change = ((opt_val - base_val) / base_val * 100) if base_val > 0 else 0
            change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
            print(f"{name:<25} {base_val:<15.4f} {opt_val:<15.4f} {change_str:<15}")


def save_results(results: List[ExperimentResult], best_config: dict):
    """Save all results to files"""

    # Save raw results
    results_file = EXPERIMENT_DIR / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save best config
    config_file = EXPERIMENT_DIR / "best_config.json"
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved to: {config_file}")

    # Generate summary markdown
    summary = generate_summary_markdown(results, best_config)
    summary_file = EXPERIMENT_DIR / "experiment_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_file}")


def generate_summary_markdown(results: List[ExperimentResult], best_config: dict) -> str:
    """Generate human-readable summary"""

    # Sort results by MRR
    sorted_results = sorted(results, key=lambda x: x.mrr, reverse=True)

    md = f"""# Embedding Optimization Experiment Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Text Format | `{best_config['text_format']}` |
| Model | `{best_config['model']}` |
| Distance Metric | `{best_config['distance_metric']}` |

## Performance Summary

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| MRR | {best_config.get('baseline_mrr', 'N/A'):.4f} | {best_config.get('optimized_mrr', 'N/A'):.4f} | {best_config.get('mrr_improvement', 'N/A')} |
| P@1 | {best_config.get('baseline_p1', 'N/A'):.4f} | {best_config.get('optimized_p1', 'N/A'):.4f} | {best_config.get('p1_improvement', 'N/A')} |
| Intra/Inter Ratio | {best_config.get('baseline_ratio', 'N/A'):.4f} | {best_config.get('optimized_ratio', 'N/A'):.4f} | {best_config.get('ratio_improvement', 'N/A')} |

## All Experiment Results (sorted by MRR)

| Rank | Experiment | MRR | P@1 | Ratio |
|------|------------|-----|-----|-------|
"""

    for i, r in enumerate(sorted_results, 1):
        md += f"| {i} | {r.experiment_id} | {r.mrr:.4f} | {r.precision_at_1:.4f} | {r.intra_inter_ratio:.4f} |\n"

    md += """
## Recommendations

Based on the experiment results:

1. **Use the optimal configuration** shown above for production
2. **Text format matters** - structured formats often outperform raw text
3. **Model selection** - domain-specific models may not always be best
4. **Distance metric** - cosine similarity typically works well for normalized embeddings

## Next Steps

1. Apply the optimal configuration to the full 10K dataset
2. Re-run validation with real Cornell data
3. Consider fine-tuning if results are still unsatisfactory
"""

    return md


def main():
    parser = argparse.ArgumentParser(description="Run embedding optimization experiments")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Run specific phase")
    args = parser.parse_args()

    print("="*70)
    print("EMBEDDING OPTIMIZATION EXPERIMENTS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading researcher data...")
    researchers = load_researchers()
    print(f"Loaded {len(researchers)} researchers")
    print(f"Using dataset: {RESEARCHERS_JSON.name}")

    # Load ground truth
    load_ground_truth()

    results: List[ExperimentResult] = []

    if args.all or args.phase == 1:
        best_text_format = run_phase_1(researchers, results)
    else:
        best_text_format = "T1_raw_text"  # Default

    if args.all or args.phase == 2:
        best_model = run_phase_2(researchers, results, best_text_format)
    else:
        best_model = "M1_specter2"  # Default

    if args.all or args.phase == 3:
        best_metric = run_phase_3(researchers, results, best_text_format, best_model)
    else:
        best_metric = "cosine"  # Default

    if args.all or args.phase == 4:
        run_phase_4(researchers, results, best_text_format, best_model, best_metric)

    # Compile best config
    baseline = next((r for r in results if r.experiment_id == "T1_raw_text_M1_specter2_cosine"), None)
    optimized = next((r for r in results if r.text_format == best_text_format and r.model == best_model and r.distance_metric == best_metric), None)

    best_config = {
        "text_format": best_text_format,
        "model": best_model,
        "model_name": MODELS[best_model]["name"],
        "distance_metric": best_metric,
        "baseline_mrr": baseline.mrr if baseline else 0,
        "optimized_mrr": optimized.mrr if optimized else 0,
        "baseline_p1": baseline.precision_at_1 if baseline else 0,
        "optimized_p1": optimized.precision_at_1 if optimized else 0,
        "baseline_ratio": baseline.intra_inter_ratio if baseline else 0,
        "optimized_ratio": optimized.intra_inter_ratio if optimized else 0,
        "mrr_improvement": f"+{((optimized.mrr - baseline.mrr) / baseline.mrr * 100):.1f}%" if baseline and optimized and baseline.mrr > 0 else "N/A",
        "p1_improvement": f"+{((optimized.precision_at_1 - baseline.precision_at_1) / baseline.precision_at_1 * 100):.1f}%" if baseline and optimized and baseline.precision_at_1 > 0 else "N/A",
        "ratio_improvement": f"+{((optimized.intra_inter_ratio - baseline.intra_inter_ratio) / baseline.intra_inter_ratio * 100):.1f}%" if baseline and optimized and baseline.intra_inter_ratio > 0 else "N/A",
    }

    # Save results
    save_results(results, best_config)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest configuration:")
    print(f"  Text Format: {best_text_format}")
    print(f"  Model: {best_model}")
    print(f"  Distance Metric: {best_metric}")
    print(f"\nResults saved to: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
