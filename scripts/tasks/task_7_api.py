#!/usr/bin/env python3
"""Auto-generated task script for 7_api"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.optimizations.create_api import create_api_module
from pathlib import Path

result = create_api_module(
    output_file=Path("src/api.py")
)
print(f"API created: {result['output_file']}")

