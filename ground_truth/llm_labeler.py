#!/usr/bin/env python3
"""
LLM-Based Ground Truth Labeler
==============================
Uses Gemini API to automatically label query-researcher relevance pairs.

This creates "silver standard" labels that can be:
1. Used directly for evaluation
2. Reviewed/corrected by humans for gold standard
3. Used to bootstrap fine-tuning data
"""
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional
import os

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).parent
TASKS_FILE = OUTPUT_DIR / "labeling_tasks.json"
LABELS_DIR = OUTPUT_DIR / "labels"
LABELS_FILE = LABELS_DIR / "ground_truth_llm.json"

# Load API key from .env or environment
from src.config import GEMINI_API_KEY


LABELING_PROMPT = """You are an expert academic advisor at a top research university. Your task is to evaluate how well a researcher matches a research query.

## QUERY
"{query}"

## RESEARCHER PROFILE
Name: {name}
Position: {position}
Department: {department}

Research Description:
{summary}

---

## SCORING RUBRIC

Score 3 - HIGHLY RELEVANT (Exact Match)
Definition: The query topic is this researcher's PRIMARY research focus. You would immediately recommend them.
Examples:
- Query "battery electrochemistry" → Researcher whose lab focuses on lithium-ion battery materials ✓
- Query "protein folding" → Structural biologist studying protein misfolding diseases ✓
- Query "CRISPR gene editing" → Researcher who develops CRISPR tools and methods ✓

Score 2 - RELEVANT (Good Match)
Definition: The researcher has substantial expertise in this area, though it may not be their primary focus. A reasonable recommendation.
Examples:
- Query "battery electrochemistry" → Materials scientist who has published several papers on electrode materials ✓
- Query "machine learning for materials" → Computational materials scientist who uses ML in some projects ✓
- Query "nanomaterials" → Chemist whose work includes nanoparticle synthesis among other topics ✓

Score 1 - MARGINALLY RELEVANT (Weak Match)
Definition: Tangential connection only. The researcher MIGHT have some peripheral knowledge but would NOT be a good recommendation.
Examples:
- Query "battery electrochemistry" → General electrochemist who doesn't work on batteries ✓
- Query "tissue engineering" → Biomedical imaging researcher (same broad field, different focus) ✓
- Query "polymer synthesis" → Someone who USES polymers but doesn't develop them ✓

Score 0 - NOT RELEVANT (No Match)
Definition: No meaningful connection. Different field entirely.
Examples:
- Query "battery electrochemistry" → Researcher studying fluid mechanics ✓
- Query "CRISPR gene editing" → Materials scientist working on metals ✓
- Query "machine learning" → Experimental chemist with no computational work ✓

---

## EVALUATION CRITERIA (Check each one)

1. TOPIC MATCH: Does the researcher's described work directly address the query topic?
   - Look for specific keywords, techniques, or applications mentioned
   - Consider both explicit mentions and closely related work

2. EXPERTISE DEPTH: Is this a primary focus or peripheral interest?
   - Primary focus (publications, lab theme) → Score 3
   - Significant but secondary → Score 2
   - Mentioned but not emphasized → Score 1

3. PRACTICAL RECOMMENDATION: Would you confidently recommend this person?
   - "Yes, they're perfect for this" → Score 3
   - "Yes, they could help" → Score 2
   - "Maybe, but there are probably better options" → Score 1
   - "No, wrong person" → Score 0

---

## IMPORTANT GUIDELINES

- Be STRICT with Score 3: Reserve it for researchers where the query matches their PRIMARY research theme
- Department alone is NOT sufficient: "Materials Science" department doesn't mean they work on all materials topics
- Look for SPECIFIC evidence in the research description
- When uncertain between two scores, choose the LOWER score
- Administrative roles (Director, Dean) without research description → Score based only on stated research

---

## YOUR RESPONSE

Evaluate the researcher against the query using the rubric above.

Respond with ONLY a JSON object:
{{"score": <0-3>, "reasoning": "<1-2 sentences citing specific evidence from their profile>"}}

Example good reasoning:
- "Score 3: Their lab explicitly focuses on lithium-ion battery cathode materials, directly matching the query."
- "Score 2: While their main focus is polymer chemistry, they mention nanoparticle synthesis in several projects."
- "Score 1: They work in biomedical engineering but focus on imaging, not the tissue engineering asked about."
- "Score 0: Their research on fluid dynamics has no connection to the semiconductor query."
"""


class LLMLabeler:
    """Uses Gemini to label query-researcher pairs"""

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.client = None

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Check .env file or environment.")

    def initialize(self):
        """Initialize Gemini client"""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        print(f"LLM Labeler initialized with model: {self.model}")

    def label_pair(self, query: str, researcher: Dict, max_retries: int = 3) -> Dict:
        """
        Label a single query-researcher pair.

        Returns:
            Dict with score, reasoning, and metadata
        """
        prompt = LABELING_PROMPT.format(
            query=query,
            name=researcher.get("researcher_name", "Unknown"),
            position=researcher.get("researcher_position", "Unknown"),
            department=researcher.get("researcher_department", "Unknown"),
            summary=researcher.get("researcher_summary", "No summary available")[:800]
        )

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(prompt)
                text = response.text.strip()

                # Parse JSON response
                # Handle markdown code blocks
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                result = json.loads(text)

                # Validate score
                score = int(result.get("score", -1))
                if score not in [0, 1, 2, 3]:
                    raise ValueError(f"Invalid score: {score}")

                return {
                    "score": score,
                    "reasoning": result.get("reasoning", ""),
                    "raw_response": response.text[:200]
                }

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return {
                    "score": None,
                    "reasoning": f"JSON parse error: {e}",
                    "raw_response": text[:200] if 'text' in dir() else "No response"
                }

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    # Rate limit - wait and retry
                    wait_time = 30 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue

                return {
                    "score": None,
                    "reasoning": f"Error: {str(e)[:100]}",
                    "raw_response": ""
                }

        return {"score": None, "reasoning": "Max retries exceeded", "raw_response": ""}


def load_tasks() -> List[Dict]:
    """Load labeling tasks"""
    with open(TASKS_FILE) as f:
        data = json.load(f)
    return data["tasks"]


def load_existing_labels() -> Dict[int, Dict]:
    """Load any existing LLM labels"""
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            data = json.load(f)
        return {label["task_id"]: label for label in data.get("labels", [])}
    return {}


def save_labels(labels: Dict[int, Dict], stats: Dict = None):
    """Save labels to file"""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate stats if not provided
    if stats is None:
        scores = [l["relevance_score"] for l in labels.values() if l.get("relevance_score") is not None]
        stats = {
            "total_labeled": len(scores),
            "failed": sum(1 for l in labels.values() if l.get("relevance_score") is None),
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
        "labeler": "gemini-llm",
        "stats": stats,
        "labels": list(labels.values())
    }

    with open(LABELS_FILE, "w") as f:
        json.dump(output, f, indent=2)


def run_llm_labeling(
    tasks: List[Dict],
    existing_labels: Dict[int, Dict],
    labeler: LLMLabeler,
    batch_size: int = 10,
    delay_between_requests: float = 0.5
) -> Dict[int, Dict]:
    """
    Run LLM labeling on all tasks.

    Args:
        tasks: List of labeling tasks
        existing_labels: Already completed labels (to skip)
        labeler: LLMLabeler instance
        batch_size: Save checkpoint every N labels
        delay_between_requests: Seconds to wait between API calls

    Returns:
        Dict of task_id -> label
    """
    labels = existing_labels.copy()
    total = len(tasks)
    completed = len(existing_labels)
    failed = 0

    print(f"\nLabeling {total} tasks ({completed} already done)...")
    print(f"Delay between requests: {delay_between_requests}s")
    print("-" * 50)

    for i, task in enumerate(tasks):
        task_id = task["task_id"]

        # Skip if already labeled
        if task_id in labels and labels[task_id].get("relevance_score") is not None:
            continue

        # Progress
        pct = (i + 1) / total * 100
        print(f"[{i+1}/{total}] ({pct:.0f}%) {task['query'][:40]}... ", end="", flush=True)

        # Label this pair
        result = labeler.label_pair(task["query"], task)

        if result["score"] is not None:
            labels[task_id] = {
                "task_id": task_id,
                "query": task["query"],
                "query_type": task.get("query_type", ""),
                "researcher_name": task["researcher_name"],
                "relevance_score": result["score"],
                "reasoning": result["reasoning"],
                "source": task.get("source", ""),
                "system_rank": task.get("system_rank"),
                "labeler": "gemini-llm"
            }
            print(f"Score: {result['score']} - {result['reasoning'][:50]}")
            completed += 1
        else:
            print(f"FAILED - {result['reasoning'][:50]}")
            failed += 1
            labels[task_id] = {
                "task_id": task_id,
                "query": task["query"],
                "researcher_name": task["researcher_name"],
                "relevance_score": None,
                "error": result["reasoning"],
                "labeler": "gemini-llm"
            }

        # Save checkpoint
        if (i + 1) % batch_size == 0:
            save_labels(labels)
            print(f"    [Checkpoint saved: {completed} completed, {failed} failed]")

        # Rate limiting
        time.sleep(delay_between_requests)

    return labels


def print_summary(labels: Dict[int, Dict]):
    """Print labeling summary"""
    scores = [l["relevance_score"] for l in labels.values() if l.get("relevance_score") is not None]
    failed = sum(1 for l in labels.values() if l.get("relevance_score") is None)

    print("\n" + "=" * 50)
    print("LLM LABELING COMPLETE")
    print("=" * 50)
    print(f"Total labeled:    {len(scores)}")
    print(f"Failed:           {failed}")
    print(f"Average score:    {sum(scores)/len(scores):.2f}" if scores else "N/A")
    print(f"\nScore distribution:")
    print(f"  3 (Highly Relevant): {scores.count(3)}")
    print(f"  2 (Relevant):        {scores.count(2)}")
    print(f"  1 (Marginal):        {scores.count(1)}")
    print(f"  0 (Not Relevant):    {scores.count(0)}")

    # System vs random comparison
    system_scores = [l["relevance_score"] for l in labels.values()
                     if l.get("relevance_score") is not None and l.get("source") == "system_result"]
    random_scores = [l["relevance_score"] for l in labels.values()
                     if l.get("relevance_score") is not None and l.get("source") == "random"]

    if system_scores and random_scores:
        print(f"\nSystem results avg: {sum(system_scores)/len(system_scores):.2f}")
        print(f"Random results avg: {sum(random_scores)/len(random_scores):.2f}")

    print(f"\nLabels saved to: {LABELS_FILE}")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based ground truth labeling")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Save checkpoint every N labels (default: 10)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Maximum tasks to label (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing labels")
    args = parser.parse_args()

    print("=" * 50)
    print("  LLM-BASED GROUND TRUTH LABELER")
    print("=" * 50)

    # Check API key
    if not GEMINI_API_KEY:
        print("\nError: GEMINI_API_KEY not set!")
        print("Add it to .env file or set environment variable.")
        return

    # Load tasks
    print("\nLoading labeling tasks...")
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks")

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
        print(f"Limiting to {args.max_tasks} tasks")

    # Load existing labels if resuming
    existing_labels = {}
    if args.resume:
        existing_labels = load_existing_labels()
        print(f"Resuming with {len(existing_labels)} existing labels")

    # Initialize labeler
    print("\nInitializing LLM labeler...")
    labeler = LLMLabeler()
    labeler.initialize()

    # Run labeling
    try:
        labels = run_llm_labeling(
            tasks,
            existing_labels,
            labeler,
            batch_size=args.batch_size,
            delay_between_requests=args.delay
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        labels = existing_labels

    # Save final results
    save_labels(labels)
    print_summary(labels)

    # Also copy to the standard ground truth location for evaluation
    standard_labels_file = LABELS_DIR / "ground_truth_v1.json"
    if not standard_labels_file.exists():
        import shutil
        shutil.copy(LABELS_FILE, standard_labels_file)
        print(f"\nCopied to: {standard_labels_file}")


if __name__ == "__main__":
    main()
