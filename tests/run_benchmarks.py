#!/usr/bin/env python3
"""
Benchmark runner for OG-Core Dask performance tests.

This script runs comprehensive benchmarks and generates reports comparing
current performance across different configurations.

Usage:
    python run_benchmarks.py [--quick] [--report-only] [--save-baseline]
    
Options:
    --quick         Run only fast benchmark tests
    --report-only   Generate report from existing results without running new tests  
    --save-baseline Save current results as baseline for future comparisons
    --compare-baseline Compare current results against saved baseline
"""

import os
import sys
import json
import argparse
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add the parent directory to path so we can import test modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_dask_benchmarks import (
    BenchmarkResult, 
    load_benchmark_results, 
    generate_benchmark_report,
    BENCHMARK_RESULTS_DIR
)


def run_benchmark_tests(quick: bool = False) -> bool:
    """
    Run benchmark tests using pytest.
    
    Args:
        quick: If True, skip slow tests
        
    Returns:
        True if tests ran successfully, False otherwise
    """
    cmd = [
        sys.executable, "-m", "pytest", 
        "-v", 
        "-m", "benchmark",
        "--tb=short"
    ]
    
    if quick:
        cmd.extend(["-m", "not slow"])
    
    # Add the benchmark test files
    cmd.extend(["test_dask_benchmarks.py", "test_real_txfunc_benchmarks.py"])
    
    print("Running benchmark tests...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")  
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running benchmark tests: {e}")
        return False


def save_baseline_results(baseline_name: Optional[str] = None):
    """
    Save current benchmark results as a baseline for future comparisons.
    
    Args:
        baseline_name: Optional name for the baseline. If None, uses timestamp.
    """
    results = load_benchmark_results()
    if not results:
        print("No benchmark results found to save as baseline.")
        return
    
    if baseline_name is None:
        baseline_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    baseline_file = os.path.join(BENCHMARK_RESULTS_DIR, f"baseline_{baseline_name}.json")
    
    # Group results by configuration for baseline
    baseline_data = {
        "created": datetime.now().isoformat(),
        "platform": platform.system(),
        "results": [r.to_dict() for r in results if r.success]
    }
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"Saved {len(baseline_data['results'])} benchmark results as baseline: {baseline_file}")


def load_baseline_results(baseline_name: Optional[str] = None) -> Optional[Dict]:
    """
    Load baseline results for comparison.
    
    Args:
        baseline_name: Name of baseline to load. If None, loads most recent.
        
    Returns:
        Baseline data dictionary or None if not found
    """
    if not os.path.exists(BENCHMARK_RESULTS_DIR):
        return None
    
    baseline_files = [f for f in os.listdir(BENCHMARK_RESULTS_DIR) if f.startswith("baseline_")]
    
    if not baseline_files:
        return None
    
    if baseline_name:
        baseline_file = f"baseline_{baseline_name}.json"
        if baseline_file not in baseline_files:
            print(f"Baseline '{baseline_name}' not found. Available: {baseline_files}")
            return None
    else:
        # Use most recent baseline
        baseline_files.sort(reverse=True)
        baseline_file = baseline_files[0]
    
    filepath = os.path.join(BENCHMARK_RESULTS_DIR, baseline_file)
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading baseline {baseline_file}: {e}")
        return None


def compare_with_baseline(baseline_name: Optional[str] = None):
    """
    Compare current benchmark results with baseline.
    
    Args:
        baseline_name: Name of baseline to compare against
    """
    baseline_data = load_baseline_results(baseline_name)
    if not baseline_data:
        print("No baseline data found for comparison.")
        return
    
    current_results = load_benchmark_results()
    successful_current = [r for r in current_results if r.success]
    
    if not successful_current:
        print("No current successful benchmark results found for comparison.")
        return
    
    baseline_results = [BenchmarkResult(**r) for r in baseline_data["results"]]
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON REPORT")
    print("="*80)
    print(f"Baseline created: {baseline_data['created']}")
    print(f"Baseline platform: {baseline_data['platform']}")
    print(f"Current platform: {platform.system()}")
    print(f"Baseline results: {len(baseline_results)}")
    print(f"Current results: {len(successful_current)}")
    
    # Create lookup by test configuration
    baseline_lookup = {}
    for r in baseline_results:
        key = (r.test_name, r.scheduler, r.num_workers)
        baseline_lookup[key] = r
    
    current_lookup = {}
    for r in successful_current:
        key = (r.test_name, r.scheduler, r.num_workers)
        current_lookup[key] = r
    
    # Compare matching configurations
    common_keys = set(baseline_lookup.keys()) & set(current_lookup.keys())
    
    if not common_keys:
        print("\nNo matching test configurations found between baseline and current results.")
        return
    
    print(f"\nComparing {len(common_keys)} matching test configurations:")
    print("-" * 80)
    print(f"{'Test':<25} {'Scheduler':<15} {'Time Change':<12} {'Memory Change':<15}")
    print("-" * 80)
    
    time_improvements = []
    memory_improvements = []
    
    for key in sorted(common_keys):
        baseline = baseline_lookup[key]
        current = current_lookup[key]
        
        time_change = ((current.compute_time - baseline.compute_time) / baseline.compute_time) * 100
        memory_change = ((current.peak_memory_mb - baseline.peak_memory_mb) / baseline.peak_memory_mb) * 100
        
        time_improvements.append(time_change)
        memory_improvements.append(memory_change)
        
        time_str = f"{time_change:+.1f}%"
        memory_str = f"{memory_change:+.1f}%"
        
        # Color coding (simplified for text output)
        time_indicator = "🟢" if time_change < -5 else "🔴" if time_change > 5 else "🟡"
        memory_indicator = "🟢" if memory_change < -5 else "🔴" if memory_change > 5 else "🟡"
        
        print(f"{key[0]:<25} {key[1]:<15} {time_str:<12} {memory_str:<15} {time_indicator}{memory_indicator}")
    
    # Summary statistics
    avg_time_change = sum(time_improvements) / len(time_improvements)
    avg_memory_change = sum(memory_improvements) / len(memory_improvements)
    
    print("-" * 80)
    print(f"Average time change: {avg_time_change:+.1f}%")
    print(f"Average memory change: {avg_memory_change:+.1f}%")
    
    # Highlight significant changes
    significant_time_improvements = [t for t in time_improvements if t < -10]
    significant_time_regressions = [t for t in time_improvements if t > 10]
    significant_memory_improvements = [m for m in memory_improvements if m < -10]
    significant_memory_regressions = [m for m in memory_improvements if m > 10]
    
    if significant_time_improvements:
        print(f"🟢 {len(significant_time_improvements)} tests showed >10% time improvement")
    if significant_time_regressions:
        print(f"🔴 {len(significant_time_regressions)} tests showed >10% time regression")
    if significant_memory_improvements:
        print(f"🟢 {len(significant_memory_improvements)} tests showed >10% memory improvement")
    if significant_memory_regressions:
        print(f"🔴 {len(significant_memory_regressions)} tests showed >10% memory regression")


def cleanup_old_results(days_to_keep: int = 30):
    """
    Clean up old benchmark result files.
    
    Args:
        days_to_keep: Number of days of results to keep
    """
    if not os.path.exists(BENCHMARK_RESULTS_DIR):
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
    removed_count = 0
    
    for filename in os.listdir(BENCHMARK_RESULTS_DIR):
        if filename.startswith("benchmark_") and filename.endswith(".json"):
            filepath = os.path.join(BENCHMARK_RESULTS_DIR, filename)
            if os.path.getmtime(filepath) < cutoff_time:
                os.remove(filepath)
                removed_count += 1
    
    if removed_count > 0:
        print(f"Removed {removed_count} old benchmark result files.")


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run OG-Core Dask performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py                    # Run all benchmarks and generate report
  python run_benchmarks.py --quick           # Run only fast benchmarks
  python run_benchmarks.py --report-only     # Generate report from existing results
  python run_benchmarks.py --save-baseline current  # Save current results as baseline
  python run_benchmarks.py --compare-baseline       # Compare with most recent baseline
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only quick benchmark tests (skip slow tests)"
    )
    
    parser.add_argument(
        "--report-only", 
        action="store_true", 
        help="Generate report from existing results without running new tests"
    )
    
    parser.add_argument(
        "--save-baseline", 
        type=str, 
        metavar="NAME",
        help="Save current results as baseline with given name"
    )
    
    parser.add_argument(
        "--compare-baseline", 
        nargs="?",
        const="", 
        metavar="NAME",
        help="Compare current results with baseline (latest if no name given)"
    )
    
    parser.add_argument(
        "--cleanup",
        type=int,
        default=0,
        metavar="DAYS",
        help="Clean up benchmark results older than DAYS (0 = no cleanup)"
    )
    
    args = parser.parse_args()
    
    # Cleanup old results if requested
    if args.cleanup > 0:
        cleanup_old_results(args.cleanup)
    
    # Run benchmarks unless report-only mode
    if not args.report_only:
        print(f"Platform: {platform.system()}")
        print(f"Python: {sys.version}")
        
        success = run_benchmark_tests(quick=args.quick)
        if not success:
            print("Benchmark tests failed.")
            return 1
    
    # Save baseline if requested
    if args.save_baseline:
        save_baseline_results(args.save_baseline)
    
    # Compare with baseline if requested  
    if args.compare_baseline is not None:
        baseline_name = args.compare_baseline if args.compare_baseline else None
        compare_with_baseline(baseline_name)
    
    # Generate standard report
    generate_benchmark_report()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())