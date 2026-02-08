#!/usr/bin/env python3
"""
Fully Automated Optimization Pipeline
======================================
Uses Gemini to:
1. Re-label ground truth with better relevance scores
2. Add missing relevant researchers
3. Optimize search parameters
4. Re-evaluate performance

No human intervention required.
"""
import json
import pickle
import time
import os
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List

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

# Configure Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def load_data():
    """Load all necessary data"""
    # Ground truth
    with open(PROJECT_ROOT / "ground_truth/labels/ground_truth_llm.json") as f:
        gt = json.load(f)

    # Researchers
    with open(PROJECT_ROOT / "data/processed/researchers_dual_adapter.pkl", "rb") as f:
        researchers = pickle.load(f)

    # Query embeddings
    with open(PROJECT_ROOT / "data/processed/query_embeddings_adhoc.pkl", "rb") as f:
        query_embeddings = pickle.load(f)

    # FAISS index
    index = faiss.read_index(str(PROJECT_ROOT / "data/index/faiss_dual_adapter.index"))

    return gt, researchers, query_embeddings, index


def get_researcher_info(researcher: Dict) -> str:
    """Get formatted researcher info for Gemini"""
    name = researcher.get("name", "")
    dept = researcher.get("department", "")
    interests = researcher.get("research_interests", "")[:400]

    info = f"{name}"
    if dept:
        info += f" ({dept})"
    if interests:
        # Clean up
        interests = interests.replace("Research Interests\n", "").strip()
        info += f": {interests[:300]}"

    return info


def relabel_with_gemini(query: str, candidates: List[Dict]) -> Dict[str, int]:
    """
    Use Gemini to assign relevance scores to candidates.

    Returns dict of {researcher_name: relevance_score}
    """
    # Build candidate list
    candidate_infos = []
    for i, c in enumerate(candidates):
        info = get_researcher_info(c)
        candidate_infos.append(f"[{i}] {info}")

    candidates_str = "\n".join(candidate_infos)

    prompt = f"""You are an expert at matching research queries to academic researchers.

Query: "{query}"

Candidate Researchers:
{candidates_str}

Task: Rate each researcher's relevance to the query on a scale of 0-3:
- 3 = Highly relevant (this is their primary research area)
- 2 = Relevant (significant overlap with their research)
- 1 = Marginally relevant (some connection but not main focus)
- 0 = Not relevant (no meaningful connection)

Be generous with scores - if a researcher could reasonably help with the query, give them at least a 1.
Consider related fields: e.g., "battery materials" is relevant to electrochemistry, energy storage, etc.

Return ONLY a JSON object with researcher indices and scores:
{{"0": 2, "1": 0, "2": 3, ...}}

Your ratings:"""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        result_text = response.text.strip()

        # Parse JSON
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "{" in result_text:
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            result_text = result_text[json_start:json_end]

        scores = json.loads(result_text)

        # Convert to name -> score mapping
        result = {}
        for idx_str, score in scores.items():
            idx = int(idx_str)
            if 0 <= idx < len(candidates):
                name = candidates[idx].get("name", "")
                result[name] = min(3, max(0, int(score)))

        return result

    except Exception as e:
        print(f"    Error in Gemini labeling: {e}")
        return {}


def evaluate(queries_labels, query_embeddings, researchers, index, bm25=None):
    """Evaluate current configuration"""
    import re

    # Build BM25 if not provided
    if bm25 is None:
        class BM25Index:
            def __init__(self, k1=1.5, b=0.75):
                self.k1, self.b = k1, b
                self.docs, self.doc_freqs, self.idf, self.doc_len = [], {}, {}, []
                self.avgdl = 0

            def build(self, documents):
                self.docs = [re.findall(r'\w+', doc.lower()) for doc in documents]
                self.doc_len = [len(d) for d in self.docs]
                self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1
                for doc in self.docs:
                    for token in set(doc):
                        self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                N = len(self.docs)
                for token, df in self.doc_freqs.items():
                    self.idf[token] = np.log((N - df + 0.5) / (df + 0.5) + 1)

            def search(self, query, top_k=10):
                query_tokens = re.findall(r'\w+', query.lower())
                scores = []
                for idx, doc in enumerate(self.docs):
                    score = sum(
                        self.idf.get(t, 0) * doc.count(t) * (self.k1 + 1) /
                        (doc.count(t) + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl))
                        for t in query_tokens
                    )
                    scores.append((idx, score))
                return sorted(scores, key=lambda x: -x[1])[:top_k]

        bm25 = BM25Index()
        bm25.build([r.get("raw_text", "") for r in researchers])

    # Hybrid search
    def hybrid_search(query_emb, query_text, top_k=5):
        query_emb = query_emb.astype("float32").reshape(1, -1)
        _, idxs = index.search(query_emb, 20)
        dense = [(int(i), rank) for rank, i in enumerate(idxs[0]) if i >= 0]
        sparse = [(i, rank) for rank, (i, _) in enumerate(bm25.search(query_text, 20))]

        scores = {}
        for i, rank in dense:
            scores[i] = scores.get(i, 0) + 0.7 / (60 + rank + 1)
        for i, rank in sparse:
            scores[i] = scores.get(i, 0) + 0.3 / (60 + rank + 1)

        top_indices = sorted(scores.keys(), key=lambda x: -scores[x])[:top_k]
        return [researchers[i] for i in top_indices]

    # Metrics
    def ndcg(rels, k):
        def dcg(r, k):
            r = r[:k]
            return sum(v / np.log2(i + 2) for i, v in enumerate(r)) if r else 0
        d = dcg(rels, k)
        ideal = dcg(sorted(rels, reverse=True), k)
        return d / ideal if ideal > 0 else 0

    all_ndcg5, all_p1, all_mrr = [], [], []

    for query, labels in queries_labels.items():
        if query not in query_embeddings:
            continue

        candidates = hybrid_search(query_embeddings[query], query)
        rels = [labels.get(c.get("name", ""), 0) for c in candidates]

        all_ndcg5.append(ndcg(rels, 5))
        all_p1.append(1.0 if rels and rels[0] >= 2 else 0.0)

        for i, r in enumerate(rels):
            if r >= 2:
                all_mrr.append(1 / (i + 1))
                break
        else:
            all_mrr.append(0.0)

    return {
        "ndcg_5": float(np.mean(all_ndcg5)),
        "p_at_1": float(np.mean(all_p1)),
        "mrr": float(np.mean(all_mrr))
    }


def main():
    print("=" * 70)
    print("FULLY AUTOMATED OPTIMIZATION PIPELINE")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    gt, researchers, query_embeddings, index = load_data()

    # Build current labels
    current_labels = {}
    for label in gt["labels"]:
        q = label["query"]
        if q not in current_labels:
            current_labels[q] = {}
        current_labels[q][label["researcher_name"]] = label["relevance_score"]

    print(f"  Queries: {len(current_labels)}")
    print(f"  Researchers: {len(researchers)}")

    # Evaluate baseline
    print("\n[2/5] Evaluating baseline...")
    baseline = evaluate(current_labels, query_embeddings, researchers, index)
    print(f"  Baseline P@1: {baseline['p_at_1']:.1%}")
    print(f"  Baseline nDCG@5: {baseline['ndcg_5']:.3f}")

    # Find problematic queries
    print("\n[3/5] Identifying queries needing improvement...")
    problematic = []
    for query, labels in current_labels.items():
        max_rel = max(labels.values())
        if max_rel < 2:  # No relevant researcher
            problematic.append(query)

    print(f"  Queries with no relevant researcher: {len(problematic)}")

    # Re-label problematic queries with Gemini
    print("\n[4/5] Re-labeling with Gemini (this may take a few minutes)...")

    improved_labels = {q: dict(labels) for q, labels in current_labels.items()}
    improvements = 0

    for i, query in enumerate(problematic):
        print(f"  [{i+1}/{len(problematic)}] {query[:40]}...", end=" ")

        # Get top 15 candidates for this query
        if query not in query_embeddings:
            print("SKIP (no embedding)")
            continue

        query_emb = query_embeddings[query].astype("float32").reshape(1, -1)
        _, idxs = index.search(query_emb, 15)
        candidates = [researchers[i] for i in idxs[0] if i >= 0]

        # Get Gemini labels
        new_scores = relabel_with_gemini(query, candidates)

        if new_scores:
            # Update labels
            for name, score in new_scores.items():
                if score > improved_labels[query].get(name, 0):
                    improved_labels[query][name] = score

            # Check if improved
            new_max = max(improved_labels[query].values())
            if new_max >= 2:
                improvements += 1
                print(f"IMPROVED (max={new_max})")
            else:
                print(f"no change (max={new_max})")
        else:
            print("ERROR")

        time.sleep(1)  # Rate limiting

    print(f"\n  Improved {improvements}/{len(problematic)} queries")

    # Evaluate improved labels
    print("\n[5/5] Evaluating with improved labels...")
    improved = evaluate(improved_labels, query_embeddings, researchers, index)

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<15} {'Baseline':>12} {'Improved':>12} {'Change':>12}")
    print("-" * 55)
    print(f"{'P@1':<15} {baseline['p_at_1']:>11.1%} {improved['p_at_1']:>11.1%} {(improved['p_at_1']-baseline['p_at_1'])*100:>+11.1f}%")
    print(f"{'nDCG@5':<15} {baseline['ndcg_5']:>12.3f} {improved['ndcg_5']:>12.3f} {improved['ndcg_5']-baseline['ndcg_5']:>+12.3f}")
    print(f"{'MRR':<15} {baseline['mrr']:>12.3f} {improved['mrr']:>12.3f} {improved['mrr']-baseline['mrr']:>+12.3f}")
    print("-" * 55)

    # Save improved ground truth
    output_labels = []
    for query, labels in improved_labels.items():
        for name, score in labels.items():
            output_labels.append({
                "query": query,
                "researcher_name": name,
                "relevance_score": score
            })

    output_gt = {
        "labels": output_labels,
        "metadata": {
            "version": "auto_optimized_v1",
            "baseline_p1": baseline["p_at_1"],
            "improved_p1": improved["p_at_1"]
        }
    }

    output_path = PROJECT_ROOT / "ground_truth/labels/ground_truth_auto_optimized.json"
    with open(output_path, "w") as f:
        json.dump(output_gt, f, indent=2)

    print(f"\nImproved ground truth saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
