"""
Query Expansion with LLM
========================
Uses Gemini to expand user queries with synonyms and related terms
before embedding, improving recall for vague or short queries.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import os


class QueryExpander:
    """
    Expands queries using Gemini LLM.

    Example:
        Input: "DNA stuff"
        Output: "DNA origami, nucleic acid nanotechnology, DNA self-assembly, DNA nanostructures"
    """

    def __init__(self, api_key: str = None):
        """
        Initialize query expander.

        Args:
            api_key: Gemini API key (defaults to env var)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = None
        self.cache = {}  # Cache expanded queries

    def load(self):
        """Load Gemini model"""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set!")

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("Query expander initialized with Gemini")

    def expand(self, query: str, num_expansions: int = 5) -> str:
        """
        Expand a query with related terms.

        Args:
            query: Original user query
            num_expansions: Number of related terms to add

        Returns:
            Expanded query string
        """
        # Check cache
        cache_key = f"{query}_{num_expansions}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.model is None:
            self.load()

        prompt = f"""You are helping expand a search query for finding academic researchers.
Given the user's query, provide {num_expansions} related technical terms or synonyms that would help find relevant researchers.

User query: "{query}"

Respond with ONLY a comma-separated list of technical terms, no explanations.
Example response: "term1, term2, term3, term4, term5"
"""

        try:
            response = self.model.generate_content(prompt)
            expanded_terms = response.text.strip()

            # Combine original query with expanded terms
            expanded_query = f"{query}, {expanded_terms}"

            # Cache result
            self.cache[cache_key] = expanded_query

            return expanded_query

        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query  # Return original on failure

    def expand_batch(self, queries: List[str], delay: float = 0.5) -> List[str]:
        """
        Expand multiple queries.

        Args:
            queries: List of query strings
            delay: Delay between API calls to avoid rate limiting

        Returns:
            List of expanded query strings
        """
        expanded = []
        for i, q in enumerate(queries):
            print(f"Expanding query {i+1}/{len(queries)}: {q[:50]}...")
            expanded.append(self.expand(q))
            if i < len(queries) - 1:
                time.sleep(delay)
        return expanded


def run_query_expansion_experiments(output_file: Path = None) -> Dict:
    """
    Run experiments comparing search with and without query expansion.

    Returns:
        Experiment results
    """
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.embedding import Specter2Encoder, format_researcher_text
    from src.config import DATA_DIR

    print("Running Query Expansion Experiments...")

    # Load data
    researchers_file = DATA_DIR / "overnight_results" / "researchers_large.json"
    if not researchers_file.exists():
        researchers_file = DATA_DIR / "raw" / "researchers_expanded.json"
    if not researchers_file.exists():
        researchers_file = DATA_DIR / "raw" / "researchers.json"

    print(f"Loading researchers from: {researchers_file}")
    with open(researchers_file) as f:
        data = json.load(f)
    researchers = data["researchers"]
    print(f"Loaded {len(researchers)} researchers")

    # Load test queries
    queries_file = DATA_DIR / "overnight_results" / "test_queries_large.json"
    if not queries_file.exists():
        queries_file = DATA_DIR / "test_queries_expanded.json"
    if not queries_file.exists():
        queries_file = DATA_DIR / "test_queries.json"

    print(f"Loading queries from: {queries_file}")
    with open(queries_file) as f:
        queries_data = json.load(f)
    test_queries = queries_data.get("queries", [])
    print(f"Loaded {len(test_queries)} test queries")

    # Prepare documents and embeddings
    print("Preparing documents...")
    documents = [format_researcher_text(r, "optimized") for r in researchers]

    # Create ID to index mapping
    id_to_idx = {r["id"]: i for i, r in enumerate(researchers)}

    # Load encoder (use CPU to avoid MPS mutex issues)
    print("Loading SPECTER2 encoder (CPU mode for stability)...")
    encoder = Specter2Encoder(force_cpu=True)
    encoder.load()

    print("Generating document embeddings...")
    doc_embeddings = encoder.encode_batch(documents, show_progress=True)

    # Normalize for cosine similarity
    doc_embeddings_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    # Initialize query expander
    print("Initializing query expander...")
    try:
        expander = QueryExpander()
        expander.load()
        expansion_available = True
    except Exception as e:
        print(f"Query expansion not available: {e}")
        expansion_available = False

    def search(query_emb, top_k=10):
        """Simple semantic search"""
        query_norm = query_emb / np.linalg.norm(query_emb)
        similarities = np.dot(doc_embeddings_norm, query_norm)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(i), float(similarities[i])) for i in top_indices]

    def evaluate(queries_to_test, use_expansion=False, expansion_terms=5):
        """Evaluate search performance"""
        mrr_sum = 0
        p_at_1 = 0
        p_at_3 = 0
        p_at_5 = 0
        valid_queries = 0

        for q in queries_to_test:
            query_text = q.get("query", "")
            expected_id = q.get("expected_id", "")

            if not expected_id or expected_id not in id_to_idx:
                continue

            expected_idx = id_to_idx[expected_id]
            valid_queries += 1

            # Optionally expand query
            if use_expansion and expansion_available:
                query_text = expander.expand(query_text, expansion_terms)

            # Get query embedding
            query_emb = encoder.encode(query_text)

            # Search
            results = search(query_emb, 10)
            result_indices = [idx for idx, _ in results]

            # Calculate metrics
            if expected_idx in result_indices:
                rank = result_indices.index(expected_idx) + 1
                mrr_sum += 1.0 / rank
                if rank == 1:
                    p_at_1 += 1
                if rank <= 3:
                    p_at_3 += 1
                if rank <= 5:
                    p_at_5 += 1

        return {
            "mrr": mrr_sum / valid_queries if valid_queries > 0 else 0,
            "precision_at_1": p_at_1 / valid_queries if valid_queries > 0 else 0,
            "precision_at_3": p_at_3 / valid_queries if valid_queries > 0 else 0,
            "precision_at_5": p_at_5 / valid_queries if valid_queries > 0 else 0,
            "num_queries": valid_queries,
        }

    results = {
        "num_researchers": len(researchers),
        "num_queries": len(test_queries),
        "expansion_available": expansion_available,
        "experiments": []
    }

    # Baseline: No expansion
    print("\nEvaluating baseline (no expansion)...")
    baseline = evaluate(test_queries, use_expansion=False)
    baseline["config"] = "no_expansion"
    results["experiments"].append(baseline)
    print(f"  MRR: {baseline['mrr']:.4f}, P@1: {baseline['precision_at_1']:.4f}")

    # With expansion (if available)
    if expansion_available:
        for num_terms in [3, 5, 7]:
            print(f"\nEvaluating with expansion ({num_terms} terms)...")
            with_expansion = evaluate(test_queries, use_expansion=True, expansion_terms=num_terms)
            with_expansion["config"] = f"expansion_{num_terms}_terms"
            results["experiments"].append(with_expansion)
            print(f"  MRR: {with_expansion['mrr']:.4f}, P@1: {with_expansion['precision_at_1']:.4f}")

            # Rate limiting
            time.sleep(1)

    # Find best configuration
    best = max(results["experiments"], key=lambda x: x["mrr"])
    results["best_config"] = best["config"]
    results["best_mrr"] = best["mrr"]

    # Calculate improvement
    if best["config"] != "no_expansion":
        improvement = (best["mrr"] - baseline["mrr"]) / baseline["mrr"] * 100 if baseline["mrr"] > 0 else 0
        results["improvement_over_baseline"] = f"+{improvement:.1f}%"

    print(f"\nBest configuration: {results['best_config']} (MRR: {results['best_mrr']:.4f})")

    # Save example expansions
    if expansion_available:
        results["example_expansions"] = []
        for q in test_queries[:5]:
            original = q.get("query", "")
            expanded = expander.expand(original, 5)
            results["example_expansions"].append({
                "original": original,
                "expanded": expanded
            })

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    from pathlib import Path
    output = Path(__file__).parent.parent.parent / "data" / "overnight_results" / "query_expansion_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_query_expansion_experiments(output)
    print(f"\nFinal result: {json.dumps(result, indent=2)}")
