"""
Embedding Sanity Check for Academic Matcher
Verifies that embeddings are meaningful before building the full index
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import cosine


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return 1 - cosine(v1, v2)


def find_similar_pairs(researchers: List[dict], field: str = "department") -> List[Tuple[int, int]]:
    """
    Find pairs of researchers that SHOULD be similar based on metadata.

    Args:
        researchers: List of researcher dicts with embeddings
        field: Field to group by (e.g., "department", "lab", "advisor")

    Returns:
        List of (idx1, idx2) pairs that share the same field value
    """
    # Group by field value
    groups = {}
    for i, r in enumerate(researchers):
        value = r.get(field, "").strip().lower()
        if value:
            if value not in groups:
                groups[value] = []
            groups[value].append(i)

    # Find pairs within each group
    pairs = []
    for indices in groups.values():
        if len(indices) >= 2:
            # Take first two from each group
            pairs.append((indices[0], indices[1]))

    return pairs


def find_dissimilar_pairs(researchers: List[dict], field: str = "department") -> List[Tuple[int, int]]:
    """
    Find pairs of researchers that should be DISSIMILAR based on metadata.

    Args:
        researchers: List of researcher dicts with embeddings
        field: Field to use for comparison

    Returns:
        List of (idx1, idx2) pairs from different field values
    """
    # Group by field value
    groups = {}
    for i, r in enumerate(researchers):
        value = r.get(field, "").strip().lower()
        if value:
            if value not in groups:
                groups[value] = []
            groups[value].append(i)

    # Find pairs across different groups
    pairs = []
    group_keys = list(groups.keys())

    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            if groups[group_keys[i]] and groups[group_keys[j]]:
                pairs.append((groups[group_keys[i]][0], groups[group_keys[j]][0]))
                if len(pairs) >= 5:  # Limit to 5 pairs
                    return pairs

    return pairs


def run_sanity_check(researchers: List[dict], verbose: bool = True) -> dict:
    """
    Run sanity check on embeddings.

    Verifies:
    1. All embeddings have correct shape (768,)
    2. No zero/nan embeddings
    3. Similar researchers have higher similarity than dissimilar ones

    Args:
        researchers: List of researcher dicts with 'embedding' field
        verbose: Whether to print results

    Returns:
        Dict with sanity check results
    """
    results = {
        "passed": True,
        "total_researchers": len(researchers),
        "shape_errors": 0,
        "zero_embeddings": 0,
        "nan_embeddings": 0,
        "similar_pairs": [],
        "dissimilar_pairs": [],
        "avg_similar_score": None,
        "avg_dissimilar_score": None,
        "warnings": [],
    }

    if verbose:
        print("\n" + "=" * 50)
        print("EMBEDDING SANITY CHECK")
        print("=" * 50)

    # Check 1: Embedding shapes and values
    if verbose:
        print("\n1. Checking embedding integrity...")

    for i, r in enumerate(researchers):
        emb = r.get("embedding")

        if emb is None:
            results["shape_errors"] += 1
            continue

        if not isinstance(emb, np.ndarray):
            emb = np.array(emb)

        if emb.shape != (768,):
            results["shape_errors"] += 1

        if np.all(emb == 0):
            results["zero_embeddings"] += 1

        if np.any(np.isnan(emb)):
            results["nan_embeddings"] += 1

    if results["shape_errors"] > 0:
        results["passed"] = False
        results["warnings"].append(f"{results['shape_errors']} embeddings have wrong shape")

    if results["zero_embeddings"] > 0:
        results["passed"] = False
        results["warnings"].append(f"{results['zero_embeddings']} embeddings are all zeros")

    if results["nan_embeddings"] > 0:
        results["passed"] = False
        results["warnings"].append(f"{results['nan_embeddings']} embeddings contain NaN")

    if verbose:
        print(f"   Shape errors: {results['shape_errors']}")
        print(f"   Zero embeddings: {results['zero_embeddings']}")
        print(f"   NaN embeddings: {results['nan_embeddings']}")

    # Check 2: Semantic similarity sanity check
    if verbose:
        print("\n2. Checking semantic similarity...")

    # Find similar pairs (same department)
    similar_pairs = find_similar_pairs(researchers, "department")
    dissimilar_pairs = find_dissimilar_pairs(researchers, "department")

    # Calculate similarities
    similar_scores = []
    for idx1, idx2 in similar_pairs:
        emb1 = np.array(researchers[idx1]["embedding"])
        emb2 = np.array(researchers[idx2]["embedding"])
        sim = cosine_similarity(emb1, emb2)
        similar_scores.append(sim)
        results["similar_pairs"].append({
            "name1": researchers[idx1].get("name"),
            "name2": researchers[idx2].get("name"),
            "field": researchers[idx1].get("department"),
            "similarity": round(sim, 4)
        })

    dissimilar_scores = []
    for idx1, idx2 in dissimilar_pairs:
        emb1 = np.array(researchers[idx1]["embedding"])
        emb2 = np.array(researchers[idx2]["embedding"])
        sim = cosine_similarity(emb1, emb2)
        dissimilar_scores.append(sim)
        results["dissimilar_pairs"].append({
            "name1": researchers[idx1].get("name"),
            "dept1": researchers[idx1].get("department"),
            "name2": researchers[idx2].get("name"),
            "dept2": researchers[idx2].get("department"),
            "similarity": round(sim, 4)
        })

    if similar_scores:
        results["avg_similar_score"] = round(np.mean(similar_scores), 4)
    if dissimilar_scores:
        results["avg_dissimilar_score"] = round(np.mean(dissimilar_scores), 4)

    if verbose:
        print(f"\n   Same-department pairs (should be similar):")
        for pair in results["similar_pairs"][:5]:
            print(f"     {pair['name1']} <-> {pair['name2']}")
            print(f"       [{pair['field']}] similarity: {pair['similarity']:.4f}")

        print(f"\n   Cross-department pairs (baseline):")
        for pair in results["dissimilar_pairs"][:5]:
            print(f"     {pair['name1']} ({pair['dept1'][:20]})")
            print(f"       <-> {pair['name2']} ({pair['dept2'][:20]})")
            print(f"       similarity: {pair['similarity']:.4f}")

        if results["avg_similar_score"] and results["avg_dissimilar_score"]:
            print(f"\n   Average same-dept similarity: {results['avg_similar_score']:.4f}")
            print(f"   Average cross-dept similarity: {results['avg_dissimilar_score']:.4f}")

            # Sanity check: same-dept should generally be higher
            if results["avg_similar_score"] <= results["avg_dissimilar_score"]:
                results["warnings"].append(
                    "Same-department similarity is not higher than cross-department. "
                    "Embeddings may not capture research domain well."
                )
                if verbose:
                    print(f"\n   ⚠️ WARNING: Same-dept similarity should be higher than cross-dept!")

    # Summary
    if verbose:
        print("\n" + "=" * 50)
        if results["passed"] and not results["warnings"]:
            print("✅ SANITY CHECK PASSED")
        elif results["warnings"]:
            print("⚠️ SANITY CHECK PASSED WITH WARNINGS")
            for w in results["warnings"]:
                print(f"   - {w}")
        else:
            print("❌ SANITY CHECK FAILED")
            for w in results["warnings"]:
                print(f"   - {w}")
        print("=" * 50)

    return results


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    # Create fake embeddings for testing
    print("Testing sanity check with fake data...")

    fake_researchers = [
        {"name": "Alice", "department": "Chemistry", "embedding": np.random.randn(768)},
        {"name": "Bob", "department": "Chemistry", "embedding": np.random.randn(768)},
        {"name": "Charlie", "department": "Physics", "embedding": np.random.randn(768)},
        {"name": "Diana", "department": "Physics", "embedding": np.random.randn(768)},
        {"name": "Eve", "department": "Biology", "embedding": np.random.randn(768)},
    ]

    results = run_sanity_check(fake_researchers)
    print(f"\nResults: {results['passed']}")
