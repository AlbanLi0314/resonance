#!/usr/bin/env python3
"""
Interactive Ground Truth Labeler
================================
Terminal-based tool for labeling query-researcher relevance.

Features:
- Shows query and researcher details
- Accepts 0-3 relevance scores
- Supports skip, back, and save commands
- Tracks progress and saves incrementally
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

OUTPUT_DIR = Path(__file__).parent
TASKS_FILE = OUTPUT_DIR / "labeling_tasks.json"
LABELS_DIR = OUTPUT_DIR / "labels"
LABELS_FILE = LABELS_DIR / "ground_truth_v1.json"


def load_tasks() -> List[Dict]:
    """Load labeling tasks"""
    if not TASKS_FILE.exists():
        print(f"Error: {TASKS_FILE} not found.")
        print("Run: python ground_truth/generate_candidates.py first")
        sys.exit(1)

    with open(TASKS_FILE) as f:
        data = json.load(f)
    return data["tasks"]


def load_existing_labels() -> Dict[int, Dict]:
    """Load any existing labels"""
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            data = json.load(f)
        return {label["task_id"]: label for label in data.get("labels", [])}
    return {}


def save_labels(labels: Dict[int, Dict]):
    """Save labels to file"""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate stats
    scores = [l["relevance_score"] for l in labels.values() if l["relevance_score"] is not None]
    stats = {
        "total_labeled": len(scores),
        "score_distribution": {
            "0_not_relevant": scores.count(0),
            "1_marginal": scores.count(1),
            "2_relevant": scores.count(2),
            "3_highly_relevant": scores.count(3),
        },
        "average_score": sum(scores) / len(scores) if scores else 0
    }

    output = {
        "version": "1.0",
        "stats": stats,
        "labels": list(labels.values())
    }

    with open(LABELS_FILE, "w") as f:
        json.dump(output, f, indent=2)


def clear_screen():
    """Clear terminal screen"""
    print("\033[2J\033[H", end="")


def print_task(task: Dict, index: int, total: int, existing_label: Optional[Dict]):
    """Display a task for labeling"""

    clear_screen()

    print("=" * 70)
    print(f"  GROUND TRUTH LABELER  |  Task {index + 1} of {total}")
    print("=" * 70)

    # Query
    print(f"\n  QUERY ({task['query_type']}):")
    print(f"  \033[1;36m{task['query']}\033[0m")

    # Researcher
    print(f"\n  RESEARCHER:")
    print(f"  \033[1;33m{task['researcher_name']}\033[0m")
    print(f"  {task['researcher_position']}")
    print(f"  {task['researcher_department']}")

    # Research summary
    print(f"\n  RESEARCH SUMMARY:")
    summary = task["researcher_summary"].replace("\n", " ")[:400]
    # Word wrap
    words = summary.split()
    line = "  "
    for word in words:
        if len(line) + len(word) > 68:
            print(line)
            line = "  "
        line += word + " "
    if line.strip():
        print(line)

    # Source info
    print(f"\n  [Source: {task['source']}", end="")
    if task["system_rank"]:
        print(f", System Rank: #{task['system_rank']}", end="")
    print("]")

    # Existing label
    if existing_label and existing_label.get("relevance_score") is not None:
        score = existing_label["relevance_score"]
        print(f"\n  \033[1;32mPreviously labeled: {score}\033[0m")

    print("\n" + "-" * 70)
    print("  RELEVANCE SCALE:")
    print("    3 = Highly Relevant (excellent match, primary focus)")
    print("    2 = Relevant (related area, reasonable match)")
    print("    1 = Marginally Relevant (weak connection)")
    print("    0 = Not Relevant (wrong field)")
    print("-" * 70)


def get_input(prompt: str) -> str:
    """Get user input with prompt"""
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "quit"


def run_labeler():
    """Main labeling loop"""

    tasks = load_tasks()
    labels = load_existing_labels()

    # Find first unlabeled task
    start_index = 0
    for i, task in enumerate(tasks):
        if task["task_id"] not in labels:
            start_index = i
            break

    index = start_index
    total = len(tasks)

    print(f"\nLoaded {total} tasks, {len(labels)} already labeled.")
    print(f"Starting at task {start_index + 1}.")
    print("\nCommands: 0-3 (score), s (skip), b (back), q (quit), h (help)")
    input("Press Enter to start...")

    while True:
        if index < 0:
            index = 0
        if index >= total:
            print("\n\033[1;32mAll tasks completed!\033[0m")
            break

        task = tasks[index]
        task_id = task["task_id"]
        existing = labels.get(task_id)

        print_task(task, index, total, existing)

        # Get score
        prompt = "\n  Enter score (0-3), or command: "
        user_input = get_input(prompt)

        if user_input in ["0", "1", "2", "3"]:
            score = int(user_input)
            labels[task_id] = {
                "task_id": task_id,
                "query": task["query"],
                "researcher_name": task["researcher_name"],
                "relevance_score": score,
                "source": task["source"],
                "system_rank": task["system_rank"]
            }
            save_labels(labels)
            print(f"  \033[1;32m✓ Saved score {score}\033[0m")
            index += 1

        elif user_input == "s" or user_input == "skip":
            print("  Skipped")
            index += 1

        elif user_input == "b" or user_input == "back":
            index -= 1

        elif user_input == "q" or user_input == "quit":
            save_labels(labels)
            print(f"\n\033[1;33mProgress saved. {len(labels)} labels total.\033[0m")
            break

        elif user_input == "h" or user_input == "help":
            print("\n  Commands:")
            print("    0, 1, 2, 3  - Set relevance score")
            print("    s, skip     - Skip this task")
            print("    b, back     - Go back one task")
            print("    q, quit     - Save and quit")
            print("    h, help     - Show this help")
            input("\n  Press Enter to continue...")

        elif user_input == "":
            # Just re-display
            pass

        else:
            print(f"  Unknown command: {user_input}")
            print("  Enter 0-3 for score, or h for help")

    # Final stats
    scores = [l["relevance_score"] for l in labels.values() if l["relevance_score"] is not None]
    if scores:
        print(f"\n" + "=" * 50)
        print("LABELING SUMMARY")
        print("=" * 50)
        print(f"Total labeled: {len(scores)}")
        print(f"Score distribution:")
        print(f"  3 (Highly Relevant): {scores.count(3)}")
        print(f"  2 (Relevant):        {scores.count(2)}")
        print(f"  1 (Marginal):        {scores.count(1)}")
        print(f"  0 (Not Relevant):    {scores.count(0)}")
        print(f"Average score: {sum(scores)/len(scores):.2f}")
        print(f"\nLabels saved to: {LABELS_FILE}")


def main():
    print("=" * 50)
    print("  INTERACTIVE GROUND TRUTH LABELER")
    print("=" * 50)
    print("\nThis tool helps you label query-researcher pairs")
    print("for ground truth evaluation.\n")

    if not TASKS_FILE.exists():
        print(f"Error: No labeling tasks found at {TASKS_FILE}")
        print("First run: python ground_truth/generate_candidates.py")
        return

    run_labeler()


if __name__ == "__main__":
    main()
