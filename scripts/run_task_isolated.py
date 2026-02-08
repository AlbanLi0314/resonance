#!/usr/bin/env python3
"""
Isolated Task Runner
====================
Runs optimization tasks in a truly isolated subprocess using spawn.
This avoids MPS mutex issues on Apple Silicon.
"""
import sys
import os
import multiprocessing as mp

# Force spawn method for true process isolation
mp.set_start_method('spawn', force=True)


def run_hybrid_search():
    """Run hybrid search in isolated process"""
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Disable MPS before importing torch
    import torch
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.optimizations.hybrid_search import run_hybrid_experiments
    output = Path(__file__).parent.parent / "data" / "overnight_results" / "hybrid_search_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_hybrid_experiments(output)
    print(f"Best config: {result.get('best_config')}")
    print(f"Best MRR: {result.get('best_mrr', 0):.4f}")
    return result


def run_query_expansion():
    """Run query expansion in isolated process"""
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    import torch
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.optimizations.query_expansion import run_query_expansion_experiments
    output = Path(__file__).parent.parent / "data" / "overnight_results" / "query_expansion_results.json"
    result = run_query_expansion_experiments(output)
    print(f"Best config: {result.get('best_config')}")
    return result


def run_comprehensive():
    """Run comprehensive evaluation in isolated process"""
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    import torch
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.optimizations.comprehensive_eval import run_comprehensive_evaluation
    output = Path(__file__).parent.parent / "data" / "overnight_results" / "comprehensive_results.json"
    result = run_comprehensive_evaluation(output)
    print(f"Best config: {result.get('best_config')}")
    return result


def run_error_analysis():
    """Run error analysis in isolated process"""
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    import torch
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.optimizations.error_analysis import run_error_analysis as _run_error_analysis
    output = Path(__file__).parent.parent / "data" / "overnight_results" / "error_analysis.json"
    result = _run_error_analysis(output)
    print(f"Success rate: {result['analysis'].get('success_rate', 0):.2%}")
    return result


def worker(task_name, func):
    """Worker function that runs in isolated subprocess"""
    try:
        print(f"[{task_name}] Starting...")
        result = func()
        print(f"[{task_name}] Completed successfully!")
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"[{task_name}] Failed: {e}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "all"

    tasks = {
        "hybrid": run_hybrid_search,
        "expansion": run_query_expansion,
        "comprehensive": run_comprehensive,
        "error": run_error_analysis,
    }

    if task == "all":
        for name, func in tasks.items():
            print(f"\n{'='*60}")
            print(f"RUNNING: {name}")
            print(f"{'='*60}")
            # Run each task in a fresh subprocess
            p = mp.Process(target=func)
            p.start()
            p.join(timeout=3600)  # 1 hour timeout
            if p.exitcode != 0:
                print(f"Task {name} failed with exit code {p.exitcode}")
    elif task in tasks:
        p = mp.Process(target=tasks[task])
        p.start()
        p.join(timeout=3600)
        sys.exit(p.exitcode or 0)
    else:
        print(f"Unknown task: {task}")
        print(f"Available tasks: {list(tasks.keys())} or 'all'")
        sys.exit(1)
