#!/usr/bin/env python3
"""Auto-generated task script for 6_demo"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.optimizations.create_demo import create_streamlit_demo
from pathlib import Path

result = create_streamlit_demo(
    output_file=Path("demo_app.py")
)
print(f"Demo created: {result['output_file']}")

