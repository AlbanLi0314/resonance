#!/usr/bin/env python3
"""
Analyze Ground Truth Labels
===========================
Generate statistics and insights from labeled data.
"""
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

OUTPUT_DIR = Path(__file__).parent
LABELS_FILE = OUTPUT_DIR / "labels" / "ground_truth_v1.json"
CSV_FILE = OUTPUT_DIR / "labeling_tasks.csv"


def load_labels() -> List[Dict]:
    """Load labels from JSON file"""
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            data = json.load(f)
        return data.get("labels", [])

    # Try loading from CSV
    if CSV_FILE.exists():
        labels = []
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("relevance_score") and row["relevance_score"].strip():
                    try:
                        labels.append({
                            "task_id": int(row["task_id"]),
                            "query": row["query"],
                            "researcher_name": row["researcher_name"],
                            "relevance_score": int(row["relevance_score"]),
                            "source": row["source"],
                            "system_rank": int(row["system_rank"]) if row.get("system_rank") else None
                        })
                    except ValueError:
                        pass
        return labels

    return []


def analyze_labels(labels: List[Dict]) -> Dict:
    """Compute analysis metrics"""

    if not labels:
        return {"error": "No labels found"}

    # Basic stats
    scores = [l["relevance_score"] for l in labels]
    total = len(scores)

    # Score distribution
    distribution = {
        0: scores.count(0),
        1: scores.count(1),
        2: scores.count(2),
        3: scores.count(3)
    }

    # By query
    by_query = defaultdict(list)
    for l in labels:
        by_query[l["query"]].append(l)

    # By source (system vs random)
    by_source = defaultdict(list)
    for l in labels:
        by_source[l["source"]].append(l["relevance_score"])

    # System results analysis
    system_results = [l for l in labels if l["source"] == "system_result"]
    random_results = [l for l in labels if l["source"] == "random"]

    # Relevance rate (score >= 2)
    def relevance_rate(score_list):
        if not score_list:
            return 0
        return sum(1 for s in score_list if s >= 2) / len(score_list)

    # P@K analysis (for system results)
    p_at_1 = 0
    p_at_3 = 0
    queries_with_system = 0

    for query, query_labels in by_query.items():
        system_for_query = sorted(
            [l for l in query_labels if l["source"] == "system_result"],
            key=lambda x: x.get("system_rank") or 999
        )
        if system_for_query:
            queries_with_system += 1
            # P@1
            if system_for_query[0]["relevance_score"] >= 2:
                p_at_1 += 1
            # P@3
            relevant_in_3 = sum(1 for l in system_for_query[:3] if l["relevance_score"] >= 2)
            p_at_3 += relevant_in_3 / min(3, len(system_for_query))

    if queries_with_system > 0:
        p_at_1 /= queries_with_system
        p_at_3 /= queries_with_system

    # Queries by difficulty (based on how many relevant researchers found)
    query_difficulty = {}
    for query, query_labels in by_query.items():
        relevant_count = sum(1 for l in query_labels if l["relevance_score"] >= 2)
        total_labeled = len(query_labels)
        query_difficulty[query] = {
            "relevant": relevant_count,
            "total": total_labeled,
            "rate": relevant_count / total_labeled if total_labeled > 0 else 0
        }

    # Hardest queries (fewest relevant results)
    sorted_by_difficulty = sorted(query_difficulty.items(), key=lambda x: x[1]["rate"])

    return {
        "total_labels": total,
        "unique_queries": len(by_query),
        "unique_researchers": len(set(l["researcher_name"] for l in labels)),
        "score_distribution": distribution,
        "average_score": sum(scores) / total,
        "relevance_rate_all": relevance_rate(scores),
        "system_results": {
            "count": len(system_results),
            "average_score": sum(l["relevance_score"] for l in system_results) / len(system_results) if system_results else 0,
            "relevance_rate": relevance_rate([l["relevance_score"] for l in system_results])
        },
        "random_results": {
            "count": len(random_results),
            "average_score": sum(l["relevance_score"] for l in random_results) / len(random_results) if random_results else 0,
            "relevance_rate": relevance_rate([l["relevance_score"] for l in random_results])
        },
        "precision_at_1": p_at_1,
        "precision_at_3": p_at_3,
        "hardest_queries": [q for q, d in sorted_by_difficulty[:5]],
        "easiest_queries": [q for q, d in sorted_by_difficulty[-5:]]
    }


def print_analysis(analysis: Dict):
    """Print analysis in readable format"""

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print("=" * 60)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 60)

    print(f"\n📊 OVERVIEW")
    print(f"   Total labels:        {analysis['total_labels']}")
    print(f"   Unique queries:      {analysis['unique_queries']}")
    print(f"   Unique researchers:  {analysis['unique_researchers']}")
    print(f"   Average score:       {analysis['average_score']:.2f}")

    print(f"\n📈 SCORE DISTRIBUTION")
    dist = analysis['score_distribution']
    total = sum(dist.values())
    for score in [3, 2, 1, 0]:
        count = dist[score]
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        labels = ["Not Relevant", "Marginal", "Relevant", "Highly Relevant"]
        print(f"   {score} ({labels[score]:<15}): {count:>3} ({pct:>5.1f}%) {bar}")

    print(f"\n🎯 SYSTEM PERFORMANCE (based on labels)")
    print(f"   Precision@1:         {analysis['precision_at_1']:.1%}")
    print(f"   Precision@3:         {analysis['precision_at_3']:.1%}")

    print(f"\n🔍 SYSTEM vs RANDOM RESULTS")
    sys = analysis['system_results']
    rnd = analysis['random_results']
    print(f"   System results:")
    print(f"     Count:             {sys['count']}")
    print(f"     Avg score:         {sys['average_score']:.2f}")
    print(f"     Relevance rate:    {sys['relevance_rate']:.1%}")
    print(f"   Random results:")
    print(f"     Count:             {rnd['count']}")
    print(f"     Avg score:         {rnd['average_score']:.2f}")
    print(f"     Relevance rate:    {rnd['relevance_rate']:.1%}")

    # System should be much better than random
    if sys['relevance_rate'] > rnd['relevance_rate'] * 2:
        print(f"   ✅ System is {sys['relevance_rate']/rnd['relevance_rate']:.1f}x better than random")
    elif sys['relevance_rate'] > rnd['relevance_rate']:
        print(f"   ⚠️  System is only {sys['relevance_rate']/rnd['relevance_rate']:.1f}x better than random")
    else:
        print(f"   ❌ System is NOT better than random!")

    print(f"\n📉 HARDEST QUERIES (fewest relevant results)")
    for q in analysis['hardest_queries'][:3]:
        print(f"   • {q[:60]}...")

    print(f"\n📈 EASIEST QUERIES (most relevant results)")
    for q in analysis['easiest_queries'][:3]:
        print(f"   • {q[:60]}...")

    print("=" * 60)


def main():
    print("Loading labels...")
    labels = load_labels()

    if not labels:
        print(f"\nNo labels found!")
        print(f"Expected locations:")
        print(f"  - {LABELS_FILE}")
        print(f"  - {CSV_FILE} (with relevance_score column filled)")
        print(f"\nRun the labeler first:")
        print(f"  python ground_truth/interactive_labeler.py")
        return

    print(f"Found {len(labels)} labels")

    analysis = analyze_labels(labels)
    print_analysis(analysis)

    # Save analysis
    analysis_file = OUTPUT_DIR / "labels" / "analysis.json"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
