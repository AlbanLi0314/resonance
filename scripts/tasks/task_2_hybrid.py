#!/usr/bin/env python3
"""Auto-generated task script for 2_hybrid"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.optimizations.hybrid_search import run_hybrid_experiments
from pathlib import Path

result = run_hybrid_experiments(
    output_file=Path("data/overnight_results/hybrid_search_results.json")
)
print(f"Best config: {result.get('best_config', 'unknown')}")
print(f"Best MRR: {result.get('best_mrr', 0):.4f}")

