#!/usr/bin/env python3
"""Auto-generated task script for 1_dataset"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from scripts.generate_large_dataset import generate_large_dataset
from pathlib import Path

result = generate_large_dataset(
    num_researchers=300,
    output_file=Path("data/overnight_results/researchers_large.json")
)
print(f"Generated {result['num_researchers']} researchers")

