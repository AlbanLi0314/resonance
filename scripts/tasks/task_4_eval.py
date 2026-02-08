#!/usr/bin/env python3
"""Auto-generated task script for 4_eval"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.optimizations.comprehensive_eval import run_comprehensive_evaluation
from pathlib import Path

result = run_comprehensive_evaluation(
    output_file=Path("data/overnight_results/comprehensive_results.json")
)
print(f"Best config: {result.get('best_config', 'unknown')}")

