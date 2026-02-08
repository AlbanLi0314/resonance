#!/usr/bin/env python3
"""
Generate Query-Researcher Pairs for Ground Truth Labeling
==========================================================
Creates a set of pairs for human relevance judgment.

For each query, selects:
- Top K results from current system (evaluate current performance)
- Random researchers (baseline / hard negatives)

Output: JSON and CSV files for labeling
"""
import json
import csv
import random
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_researchers() -> List[Dict]:
    """Load researcher data"""
    data_path = PROJECT_ROOT / "data" / "raw" / "researchers.json"
    with open(data_path) as f:
        data = json.load(f)
    return data["researchers"]


def get_system_results(query: str, top_k: int = 5) -> List[Dict]:
    """Get top results from current system"""
    try:
        from src.matcher import AcademicMatcher
        matcher = AcademicMatcher()
        matcher.initialize()
        results = matcher.search(query, skip_rerank=True, final_k=top_k)
        return results
    except Exception as e:
        print(f"Warning: Could not get system results: {e}")
        return []


def generate_queries() -> List[Dict]:
    """Generate diverse set of queries for labeling"""

    # Mix of query types
    queries = [
        # Specific technical queries
        {"query": "semiconductor thin film growth by molecular beam epitaxy", "type": "specific"},
        {"query": "gallium nitride power electronics", "type": "specific"},
        {"query": "CRISPR gene editing in mammalian cells", "type": "specific"},
        {"query": "polymer electrolyte membranes for fuel cells", "type": "specific"},
        {"query": "atomic layer deposition of oxide thin films", "type": "specific"},

        # Broader research area queries
        {"query": "nanomaterials for energy storage", "type": "broad"},
        {"query": "computational modeling of materials", "type": "broad"},
        {"query": "biomedical imaging techniques", "type": "broad"},
        {"query": "sustainable chemical processes", "type": "broad"},
        {"query": "quantum information science", "type": "broad"},

        # Application-focused queries
        {"query": "developing better batteries for electric vehicles", "type": "application"},
        {"query": "creating biocompatible implants", "type": "application"},
        {"query": "improving solar cell efficiency", "type": "application"},
        {"query": "designing drug delivery nanoparticles", "type": "application"},
        {"query": "building soft robots", "type": "application"},

        # Interdisciplinary queries
        {"query": "machine learning for materials discovery", "type": "interdisciplinary"},
        {"query": "microfluidics for biological assays", "type": "interdisciplinary"},
        {"query": "biomechanics of soft tissues", "type": "interdisciplinary"},
        {"query": "electrochemistry for CO2 reduction", "type": "interdisciplinary"},
        {"query": "3D printing of functional materials", "type": "interdisciplinary"},

        # Technique-focused queries
        {"query": "X-ray diffraction characterization", "type": "technique"},
        {"query": "electron microscopy at atomic resolution", "type": "technique"},
        {"query": "nanoindentation mechanical testing", "type": "technique"},
        {"query": "single molecule fluorescence microscopy", "type": "technique"},
        {"query": "density functional theory calculations", "type": "technique"},

        # Natural language / user-style queries
        {"query": "who works on making stronger lightweight metals", "type": "natural"},
        {"query": "looking for expertise in protein folding", "type": "natural"},
        {"query": "need help with rheology measurements", "type": "natural"},
        {"query": "research on preventing corrosion", "type": "natural"},
        {"query": "studying how cells respond to mechanical forces", "type": "natural"},

        # From production test queries (verified relevant)
        {"query": "semiconductor thin films and oxide electronics", "type": "verified"},
        {"query": "biomaterials for bone tissue engineering", "type": "verified"},
        {"query": "nanoparticle synthesis and colloidal nanomaterials", "type": "verified"},
        {"query": "polymer materials design and block copolymer self-assembly", "type": "verified"},
        {"query": "battery technology and electrochemical energy storage", "type": "verified"},
        {"query": "topological materials and quantum computing applications", "type": "verified"},
        {"query": "additive manufacturing and 3D printing of materials", "type": "verified"},
        {"query": "drug delivery systems and nanomedicine", "type": "verified"},
        {"query": "thermal transport and heat transfer in nanomaterials", "type": "verified"},
        {"query": "protein engineering and synthetic biology", "type": "verified"},

        # Challenge queries (likely to be harder)
        {"query": "phase field modeling of solidification", "type": "challenge"},
        {"query": "metamaterials with negative refractive index", "type": "challenge"},
        {"query": "topological insulators for spintronics", "type": "challenge"},
        {"query": "perovskite photovoltaics stability", "type": "challenge"},
        {"query": "liquid crystal elastomers for actuators", "type": "challenge"},
    ]

    return queries


def generate_labeling_tasks(
    num_queries: int = 50,
    system_results_per_query: int = 3,
    random_per_query: int = 1,
    seed: int = 42
) -> List[Dict]:
    """
    Generate labeling tasks (query-researcher pairs).

    Args:
        num_queries: Number of queries to include
        system_results_per_query: Top K system results to judge
        random_per_query: Number of random researchers to add
        seed: Random seed for reproducibility

    Returns:
        List of labeling tasks
    """
    random.seed(seed)

    print("Loading researchers...")
    researchers = load_researchers()
    researcher_by_name = {r["name"]: r for r in researchers}

    print("Generating queries...")
    all_queries = generate_queries()

    # Select queries (up to num_queries)
    selected_queries = all_queries[:num_queries]

    print(f"Selected {len(selected_queries)} queries")

    tasks = []
    task_id = 1

    # Initialize matcher once
    print("\nInitializing matcher for generating candidates...")
    try:
        from src.matcher import AcademicMatcher
        matcher = AcademicMatcher()
        matcher.initialize()
        matcher_available = True
    except Exception as e:
        print(f"Warning: Matcher not available: {e}")
        matcher_available = False

    print("\nGenerating labeling tasks...")

    for i, q in enumerate(selected_queries):
        query = q["query"]
        query_type = q["type"]

        print(f"  [{i+1}/{len(selected_queries)}] {query[:50]}...")

        # Get researchers to judge for this query
        researchers_to_judge = []
        seen_names = set()

        # 1. Get system results
        if matcher_available:
            try:
                results = matcher.search(query, skip_rerank=True, final_k=system_results_per_query + 2)
                for r in results[:system_results_per_query]:
                    if r["name"] not in seen_names:
                        researchers_to_judge.append({
                            "name": r["name"],
                            "source": "system_result",
                            "system_rank": len(researchers_to_judge) + 1
                        })
                        seen_names.add(r["name"])
            except Exception as e:
                print(f"    Warning: Could not get system results: {e}")

        # 2. Add random researchers
        available = [r for r in researchers if r["name"] not in seen_names]
        random_sample = random.sample(available, min(random_per_query, len(available)))
        for r in random_sample:
            researchers_to_judge.append({
                "name": r["name"],
                "source": "random",
                "system_rank": None
            })
            seen_names.add(r["name"])

        # Create tasks for this query
        for rj in researchers_to_judge:
            researcher = researcher_by_name.get(rj["name"])
            if not researcher:
                continue

            # Extract key info for labeler
            research_summary = researcher.get("research_interests", "")
            if not research_summary:
                raw = researcher.get("raw_text", "")
                # Try to extract research-relevant portion
                if "Research Interests" in raw:
                    start = raw.find("Research Interests")
                    research_summary = raw[start:start+500]
                else:
                    research_summary = raw[:500]

            task = {
                "task_id": task_id,
                "query": query,
                "query_type": query_type,
                "researcher_name": rj["name"],
                "researcher_department": researcher.get("department", ""),
                "researcher_position": researcher.get("position", ""),
                "researcher_summary": research_summary[:500],
                "source": rj["source"],
                "system_rank": rj["system_rank"],
                "relevance_score": None,  # To be filled by labeler
                "notes": ""  # Optional labeler notes
            }
            tasks.append(task)
            task_id += 1

    return tasks


def save_tasks(tasks: List[Dict], output_dir: Path):
    """Save tasks to JSON and CSV formats"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON (full data)
    json_path = output_dir / "labeling_tasks.json"
    with open(json_path, "w") as f:
        json.dump({
            "version": "1.0",
            "total_tasks": len(tasks),
            "instructions": "Score each query-researcher pair from 0-3. See README.md for guidelines.",
            "tasks": tasks
        }, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save CSV (for spreadsheet editing)
    csv_path = output_dir / "labeling_tasks.csv"
    fieldnames = [
        "task_id", "query", "query_type",
        "researcher_name", "researcher_department", "researcher_position",
        "researcher_summary", "source", "system_rank",
        "relevance_score", "notes"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            writer.writerow(task)
    print(f"Saved CSV: {csv_path}")

    # Save summary
    summary_path = output_dir / "task_summary.txt"
    query_types = {}
    sources = {}
    for t in tasks:
        query_types[t["query_type"]] = query_types.get(t["query_type"], 0) + 1
        sources[t["source"]] = sources.get(t["source"], 0) + 1

    with open(summary_path, "w") as f:
        f.write("LABELING TASK SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total tasks: {len(tasks)}\n")
        f.write(f"Unique queries: {len(set(t['query'] for t in tasks))}\n\n")
        f.write("By query type:\n")
        for qt, count in sorted(query_types.items()):
            f.write(f"  {qt}: {count}\n")
        f.write("\nBy source:\n")
        for src, count in sorted(sources.items()):
            f.write(f"  {src}: {count}\n")
    print(f"Saved summary: {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate labeling tasks")
    parser.add_argument("--num-queries", type=int, default=50,
                        help="Number of queries (default: 50)")
    parser.add_argument("--system-results", type=int, default=3,
                        help="System results per query (default: 3)")
    parser.add_argument("--random", type=int, default=1,
                        help="Random researchers per query (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING LABELING TASKS")
    print("=" * 60)
    print(f"Queries: {args.num_queries}")
    print(f"System results per query: {args.system_results}")
    print(f"Random per query: {args.random}")
    print(f"Expected total tasks: ~{args.num_queries * (args.system_results + args.random)}")
    print()

    tasks = generate_labeling_tasks(
        num_queries=args.num_queries,
        system_results_per_query=args.system_results,
        random_per_query=args.random,
        seed=args.seed
    )

    output_dir = Path(__file__).parent
    save_tasks(tasks, output_dir)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Open labeling_tasks.csv in a spreadsheet")
    print("2. Fill in 'relevance_score' column (0-3)")
    print("3. Save and run: python ground_truth/analyze_labels.py")
    print()


if __name__ == "__main__":
    main()
