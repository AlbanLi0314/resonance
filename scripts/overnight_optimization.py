#!/usr/bin/env python3
"""
Overnight Optimization Runner (Subprocess Version)
===================================================
Autonomous script that runs for 4-5 hours optimizing the academic matcher.

Each task runs in its own subprocess to avoid PyTorch/MPS threading issues.

Features:
- Try/except around every task (one failure won't crash everything)
- Checkpoints after each task
- Comprehensive logging
- Final report generation

Usage:
    python scripts/overnight_optimization.py >> logs/overnight.log 2>&1 &

Or to run in foreground with live output:
    python scripts/overnight_optimization.py

Check progress:
    tail -f logs/overnight.log
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "data" / "overnight_results"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Logging Utilities
# ============================================================

class Logger:
    """Dual logging to console and file"""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_str = f"{elapsed/60:.1f}min"

        formatted = f"[{timestamp}] [{elapsed_str}] [{level}] {message}"
        print(formatted, flush=True)

        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")

    def info(self, message: str):
        self.log(message, "INFO")

    def error(self, message: str):
        self.log(message, "ERROR")

    def success(self, message: str):
        self.log(message, "SUCCESS")

    def section(self, title: str):
        border = "=" * 60
        self.log(border)
        self.log(title)
        self.log(border)


# Global logger
logger = Logger(LOGS_DIR / "overnight_run.log")


# ============================================================
# Checkpoint System
# ============================================================

def load_checkpoint() -> Dict:
    """Load checkpoint from file"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_tasks": [], "failed_tasks": [], "results": {}}


def save_checkpoint(checkpoint: Dict):
    """Save checkpoint to file"""
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def task_completed(task_name: str) -> bool:
    """Check if task was already completed"""
    checkpoint = load_checkpoint()
    return task_name in checkpoint.get("completed_tasks", [])


def mark_task_complete(task_name: str, result: Any = None):
    """Mark task as completed and save result"""
    checkpoint = load_checkpoint()
    if task_name not in checkpoint["completed_tasks"]:
        checkpoint["completed_tasks"].append(task_name)
    if result is not None:
        checkpoint["results"][task_name] = result
    save_checkpoint(checkpoint)


def mark_task_failed(task_name: str, error: str):
    """Mark task as failed"""
    checkpoint = load_checkpoint()
    if task_name not in checkpoint["failed_tasks"]:
        checkpoint["failed_tasks"].append(task_name)
    checkpoint["results"][task_name] = {"error": error}
    save_checkpoint(checkpoint)


# ============================================================
# Task Runner (Subprocess for isolation)
# ============================================================

def run_task_in_subprocess(task_name: str, script_path: str, timeout: int = 1800) -> bool:
    """
    Run a task in a separate subprocess for isolation.

    Returns True if task succeeded, False otherwise.
    """
    logger.section(f"TASK: {task_name}")

    # Check if already completed
    if task_completed(task_name):
        logger.info(f"Task already completed, skipping...")
        return True

    start_time = time.time()
    logger.info(f"Starting task in subprocess...")
    logger.info(f"Script: {script_path}")

    try:
        # Run in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.success(f"Task completed in {elapsed:.1f}s")

            # Try to parse output for result
            output = result.stdout
            if output:
                logger.info(f"Output (last 500 chars): {output[-500:]}")

            mark_task_complete(task_name, {"status": "success", "time": elapsed})
            return True
        else:
            error_msg = result.stderr[-500:] if result.stderr else "No error output"
            logger.error(f"Task failed after {elapsed:.1f}s")
            logger.error(f"Error: {error_msg}")
            mark_task_failed(task_name, error_msg)
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Task timed out after {timeout}s")
        mark_task_failed(task_name, f"Timeout after {timeout}s")
        return False

    except Exception as e:
        logger.error(f"Task failed with exception: {e}")
        mark_task_failed(task_name, str(e))
        return False


def run_inline_task(task_name: str, task_fn, skip_if_done: bool = True) -> bool:
    """
    Run a simple task inline (for non-PyTorch tasks).
    """
    logger.section(f"TASK: {task_name}")

    if skip_if_done and task_completed(task_name):
        logger.info(f"Task already completed, skipping...")
        return True

    start_time = time.time()

    try:
        logger.info(f"Starting task...")
        result = task_fn()
        elapsed = time.time() - start_time

        logger.success(f"Task completed in {elapsed:.1f}s")
        mark_task_complete(task_name, result)
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Task failed after {elapsed:.1f}s: {e}")
        mark_task_failed(task_name, str(e))
        return False


# ============================================================
# Task Implementations (as separate scripts)
# ============================================================

def create_task_script(task_name: str, code: str) -> Path:
    """Create a temporary task script"""
    script_dir = PROJECT_ROOT / "scripts" / "tasks"
    script_dir.mkdir(parents=True, exist_ok=True)

    script_path = script_dir / f"task_{task_name}.py"

    full_code = f'''#!/usr/bin/env python3
"""Auto-generated task script for {task_name}"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

{code}
'''

    with open(script_path, 'w') as f:
        f.write(full_code)

    return script_path


# ============================================================
# Task Definitions
# ============================================================

TASK_1_CODE = '''
from scripts.generate_large_dataset import generate_large_dataset
from pathlib import Path

result = generate_large_dataset(
    num_researchers=300,
    output_file=Path("data/overnight_results/researchers_large.json")
)
print(f"Generated {result['num_researchers']} researchers")
'''

TASK_2_CODE = '''
from src.optimizations.hybrid_search import run_hybrid_experiments
from pathlib import Path

result = run_hybrid_experiments(
    output_file=Path("data/overnight_results/hybrid_search_results.json")
)
print(f"Best config: {result.get('best_config', 'unknown')}")
print(f"Best MRR: {result.get('best_mrr', 0):.4f}")
'''

TASK_3_CODE = '''
from src.optimizations.query_expansion import run_query_expansion_experiments
from pathlib import Path

result = run_query_expansion_experiments(
    output_file=Path("data/overnight_results/query_expansion_results.json")
)
print(f"Best config: {result.get('best_config', 'unknown')}")
'''

TASK_4_CODE = '''
from src.optimizations.comprehensive_eval import run_comprehensive_evaluation
from pathlib import Path

result = run_comprehensive_evaluation(
    output_file=Path("data/overnight_results/comprehensive_results.json")
)
print(f"Best config: {result.get('best_config', 'unknown')}")
'''

TASK_5_CODE = '''
from src.optimizations.error_analysis import run_error_analysis
from pathlib import Path

result = run_error_analysis(
    output_file=Path("data/overnight_results/error_analysis.json")
)
print(f"Success rate: {result['analysis'].get('success_rate', 0):.2%}")
'''

TASK_6_CODE = '''
from src.optimizations.create_demo import create_streamlit_demo
from pathlib import Path

result = create_streamlit_demo(
    output_file=Path("demo_app.py")
)
print(f"Demo created: {result['output_file']}")
'''

TASK_7_CODE = '''
from src.optimizations.create_api import create_api_module
from pathlib import Path

result = create_api_module(
    output_file=Path("src/api.py")
)
print(f"API created: {result['output_file']}")
'''


def task_generate_final_report():
    """Generate comprehensive final report"""
    checkpoint = load_checkpoint()

    report = f"""# Overnight Optimization Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Status | Count |
|--------|-------|
| Completed Tasks | {len(checkpoint.get('completed_tasks', []))} |
| Failed Tasks | {len(checkpoint.get('failed_tasks', []))} |

## Completed Tasks

"""
    for task in checkpoint.get('completed_tasks', []):
        report += f"- ✅ {task}\n"

    report += "\n## Failed Tasks\n\n"
    for task in checkpoint.get('failed_tasks', []):
        result = checkpoint.get('results', {}).get(task, {})
        error = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown'
        report += f"- ❌ {task}: {error[:100]}...\n"

    report += "\n## Results Summary\n\n"

    # Add hybrid search results if available
    hybrid_file = RESULTS_DIR / "hybrid_search_results.json"
    if hybrid_file.exists():
        with open(hybrid_file) as f:
            hybrid = json.load(f)
        report += f"### Hybrid Search\n"
        report += f"- Best config: {hybrid.get('best_config', 'N/A')}\n"
        report += f"- Best MRR: {hybrid.get('best_mrr', 0):.4f}\n\n"

    # Add comprehensive results if available
    comp_file = RESULTS_DIR / "comprehensive_results.json"
    if comp_file.exists():
        with open(comp_file) as f:
            comp = json.load(f)
        report += f"### Comprehensive Evaluation\n"
        report += f"- Best config: {comp.get('best_config', 'N/A')}\n"
        report += f"- Best MRR: {comp.get('best_mrr', 0):.4f}\n\n"

    report += """
## Files Generated

```
data/overnight_results/
├── researchers_large.json
├── test_queries_large.json
├── hybrid_search_results.json
├── query_expansion_results.json
├── comprehensive_results.json
├── error_analysis.json
└── FINAL_REPORT.md

demo_app.py                # Streamlit web demo
src/api.py                 # Production API
```

## Next Steps

1. Review the results above
2. Test the web demo: `streamlit run demo_app.py`
3. When real data arrives, rebuild index with optimized settings
"""

    report_path = RESULTS_DIR / "FINAL_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Final report saved to: {report_path}")
    return {"report_path": str(report_path)}


# ============================================================
# Main Runner
# ============================================================

def main():
    """Main overnight optimization runner"""

    logger.section("OVERNIGHT OPTIMIZATION STARTED")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Expected duration: 4-5 hours")

    total_start = time.time()

    results = {}

    # Task 1: Generate dataset (runs inline, no PyTorch)
    script1 = create_task_script("1_dataset", TASK_1_CODE)
    success = run_task_in_subprocess("1_generate_large_dataset", str(script1), timeout=300)
    results["1_generate_large_dataset"] = "success" if success else "failed"
    time.sleep(5)  # Pause between tasks

    # Task 2: Hybrid search (subprocess for PyTorch isolation)
    script2 = create_task_script("2_hybrid", TASK_2_CODE)
    success = run_task_in_subprocess("2_hybrid_search", str(script2), timeout=1800)
    results["2_hybrid_search"] = "success" if success else "failed"
    time.sleep(5)

    # Task 3: Query expansion (subprocess)
    script3 = create_task_script("3_expansion", TASK_3_CODE)
    success = run_task_in_subprocess("3_query_expansion", str(script3), timeout=1800)
    results["3_query_expansion"] = "success" if success else "failed"
    time.sleep(5)

    # Task 4: Comprehensive evaluation (subprocess)
    script4 = create_task_script("4_eval", TASK_4_CODE)
    success = run_task_in_subprocess("4_comprehensive_experiments", str(script4), timeout=3600)
    results["4_comprehensive_experiments"] = "success" if success else "failed"
    time.sleep(5)

    # Task 5: Error analysis (subprocess)
    script5 = create_task_script("5_error", TASK_5_CODE)
    success = run_task_in_subprocess("5_error_analysis", str(script5), timeout=1800)
    results["5_error_analysis"] = "success" if success else "failed"
    time.sleep(5)

    # Task 6: Web demo (inline, no PyTorch)
    script6 = create_task_script("6_demo", TASK_6_CODE)
    success = run_task_in_subprocess("6_web_demo", str(script6), timeout=60)
    results["6_web_demo"] = "success" if success else "failed"
    time.sleep(2)

    # Task 7: API (inline, no PyTorch)
    script7 = create_task_script("7_api", TASK_7_CODE)
    success = run_task_in_subprocess("7_api", str(script7), timeout=60)
    results["7_api"] = "success" if success else "failed"
    time.sleep(2)

    # Task 8: Final report (always runs inline)
    run_inline_task("8_final_report", task_generate_final_report)
    results["8_final_report"] = "success"

    total_elapsed = time.time() - total_start

    logger.section("OVERNIGHT OPTIMIZATION COMPLETE")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Results: {results}")

    # Print summary
    checkpoint = load_checkpoint()
    completed = len(checkpoint.get('completed_tasks', []))
    failed = len(checkpoint.get('failed_tasks', []))
    total = completed + failed

    logger.info(f"Completed: {completed}/{total}")
    logger.info(f"Failed: {failed}/{total}")
    logger.info(f"Final report: {RESULTS_DIR / 'FINAL_REPORT.md'}")

    return results


if __name__ == "__main__":
    main()
