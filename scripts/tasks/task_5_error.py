#!/usr/bin/env python3
"""Auto-generated task script for 5_error"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.optimizations.error_analysis import run_error_analysis
from pathlib import Path

result = run_error_analysis(
    output_file=Path("data/overnight_results/error_analysis.json")
)
print(f"Success rate: {result['analysis'].get('success_rate', 0):.2%}")

