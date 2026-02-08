#!/usr/bin/env python3
"""Auto-generated task script for 3_expansion"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.optimizations.query_expansion import run_query_expansion_experiments
from pathlib import Path

result = run_query_expansion_experiments(
    output_file=Path("data/overnight_results/query_expansion_results.json")
)
print(f"Best config: {result.get('best_config', 'unknown')}")

