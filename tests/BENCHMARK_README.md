# OG-Core Dask Performance Benchmarks

This directory contains comprehensive benchmark tests for measuring and optimizing Dask performance in OG-Core, with particular focus on addressing Windows performance issues.

## Overview

The benchmark suite includes:

1. **Mock Benchmarks** (`test_dask_benchmarks.py`) - Synthetic workloads for consistent testing
2. **Real Benchmarks** (`test_real_txfunc_benchmarks.py`) - Actual tax function estimation benchmarks  
3. **Benchmark Runner** (`run_benchmarks.py`) - Automated benchmark execution and reporting
4. **Platform Optimization** - Tests to identify optimal configurations per platform

## Quick Start

### Running Basic Benchmarks

```bash
# Run all benchmark tests
cd tests
python run_benchmarks.py

# Run only quick benchmarks (skip slow tests)
python run_benchmarks.py --quick

# Generate report from existing results
python run_benchmarks.py --report-only
```

### Running Specific Test Categories

```bash
# Run only mock benchmarks
pytest -m benchmark test_dask_benchmarks.py -v

# Run only real tax function benchmarks  
pytest -m "benchmark and real" test_real_txfunc_benchmarks.py -v

# Run platform-specific optimization tests
pytest -m "benchmark and platform" -v

# Run memory-focused tests
pytest -m "benchmark and memory" -v
```

## Benchmark Categories

### Mock Benchmarks (`test_dask_benchmarks.py`)

These tests use synthetic computations that mimic the computational patterns of tax function estimation but with controlled, reproducible workloads.

**Key Tests:**
- `test_small_dataset_multiprocessing` - Basic multiprocessing performance
- `test_small_dataset_threaded` - Threaded scheduler performance  
- `test_medium_dataset_comparison` - Compare schedulers on realistic data size
- `test_distributed_clients_comparison` - Compare Dask client configurations
- `test_memory_scaling` - How memory usage scales with data size
- `test_worker_scaling` - Performance scaling with worker count
- `test_large_dataset_stress` - Stress test with large datasets

### Real Benchmarks (`test_real_txfunc_benchmarks.py`)

These tests use the actual `txfunc.tax_func_estimate` function with realistic tax data to measure real-world performance.

**Key Tests:**
- `test_real_small_no_client` - Direct scheduler performance
- `test_real_small_threaded_client` - Distributed threaded client
- `test_real_small_process_client` - Distributed process client (Unix/macOS only)
- `test_real_medium_comparison` - Compare configurations on medium dataset
- `test_real_memory_efficiency` - Memory efficiency with real workloads
- `test_platform_specific_optimal_config` - Find optimal config for current platform

### Platform-Specific Tests

Special tests that automatically detect the current platform and run appropriate benchmark configurations:

- **Windows**: Focuses on threaded schedulers, skips problematic multiprocessing
- **macOS/Linux**: Tests both threading and multiprocessing configurations
- **Automatic Optimization**: Identifies the fastest configuration for your platform

## Understanding Benchmark Results

### Benchmark Output

Each benchmark produces a `BenchmarkResult` with the following metrics:

```python
@dataclass
class BenchmarkResult:
    test_name: str           # Name of the test
    platform: str            # Operating system
    scheduler: str           # Dask scheduler used
    num_workers: int         # Number of workers
    compute_time: float      # Execution time in seconds
    peak_memory_mb: float    # Peak memory usage in MB
    avg_memory_mb: float     # Average memory usage in MB
    data_size_mb: float      # Input data size in MB
    num_tasks: int           # Number of parallel tasks
    success: bool            # Whether test succeeded
    error_message: str       # Error details if failed
```

### Key Metrics to Monitor

1. **Compute Time** - Lower is better
2. **Peak Memory** - Memory efficiency indicator
3. **Memory/Data Ratio** - Should be reasonable (< 5x typically)
4. **Success Rate** - Reliability indicator

### Sample Output

```
Small dataset multiprocessing: 2.341s, 145.2MB peak
Small dataset threaded: 1.892s, 112.4MB peak
Medium multiprocessing: 5.123s, 287.6MB peak
Medium threaded: 4.201s, 223.1MB peak

Optimal configuration for Windows: distributed_threaded
Time: 4.201s, Memory: 223.1MB
```

## Establishing Baselines

### Save Current Performance as Baseline

```bash
# Save current results as baseline
python run_benchmarks.py --save-baseline current

# Save with custom name
python run_benchmarks.py --save-baseline before_optimization
```

### Compare Against Baseline

```bash
# Compare with most recent baseline
python run_benchmarks.py --compare-baseline

# Compare with specific baseline
python run_benchmarks.py --compare-baseline before_optimization
```

### Baseline Comparison Output

```
BENCHMARK COMPARISON REPORT
================================================================================
Baseline created: 2024-01-15T10:30:00
Current platform: Windows
Comparing 12 matching test configurations:
--------------------------------------------------------------------------------
Test                     Scheduler       Time Change  Memory Change  
--------------------------------------------------------------------------------
small_dataset_threaded   threads         -15.2%       -8.3%          🟢🟢
medium_dataset_threaded  threads         -12.7%       -5.1%          🟢🟢
real_small_threaded      distributed     -23.4%       -12.8%         🟢🟢
--------------------------------------------------------------------------------
Average time change: -17.1%
Average memory change: -8.7%
🟢 8 tests showed >10% time improvement
🟢 3 tests showed >10% memory improvement
```

## Test Data Generation

The benchmark tests use two types of data generation:

### Mock Data (`generate_mock_tax_data`)
- Synthetic tax return data with realistic distributions
- Configurable number of records and years
- Reproducible (fixed random seed)
- Fast generation for quick testing

### Realistic Data (`create_realistic_micro_data`)
- More accurate simulation of real tax data
- Proper income distributions and tax calculations
- Includes all fields required by `txfunc.tax_func_estimate`
- Slower generation but more representative

## Configuration Options

### Pytest Markers

The tests use pytest markers for easy selection:

- `@pytest.mark.benchmark` - All benchmark tests
- `@pytest.mark.distributed` - Tests using Dask distributed clients
- `@pytest.mark.memory` - Memory-focused tests
- `@pytest.mark.performance` - Compute time-focused tests
- `@pytest.mark.real` - Tests using real tax function code
- `@pytest.mark.platform` - Platform-specific optimization tests
- `@pytest.mark.slow` - Long-running tests

### Environment Variables

You can control benchmark behavior with environment variables:

```bash
# Skip multiprocessing tests (useful on Windows)
export SKIP_MULTIPROCESSING=1

# Limit number of workers for testing
export MAX_WORKERS=2

# Set custom timeout for long tests
export BENCHMARK_TIMEOUT=300
```

## Interpreting Results for Optimization

### Windows Performance Issues

Common patterns on Windows:
- Multiprocessing scheduler: Very slow due to serialization overhead
- Threaded scheduler: Much faster, good memory efficiency
- Distributed processes: Often fails or very slow
- Distributed threads: Usually optimal for Windows

### Optimal Configurations by Platform

Based on benchmark results:

**Windows:**
```python
# Recommended configuration
cluster = LocalCluster(
    n_workers=num_workers,
    threads_per_worker=2,
    processes=False,  # Use threads, not processes
    memory_limit='4GB',
)
client = Client(cluster)
```

**macOS/Linux:**
```python
# Can use either, but processes often better for CPU-bound work
client = Client(
    n_workers=num_workers,
    threads_per_worker=1,
    processes=True,
)
```

### Memory Optimization Indicators

Watch for these patterns:
- Memory/Data ratio > 10x: Likely memory leak or inefficient processing
- Peak memory growing non-linearly: Poor memory scaling
- High average vs peak difference: Memory is efficiently released

### Performance Regression Detection

Set up automated regression detection:

```bash
# Run benchmarks and compare to baseline
python run_benchmarks.py --compare-baseline

# Alert on >20% performance regression
if [ $? -ne 0 ]; then
    echo "Performance regression detected!"
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **Test Failures on Windows**
   ```
   Error: Unable to start multiprocessing workers
   Solution: Use --quick flag or set SKIP_MULTIPROCESSING=1
   ```

2. **Memory Issues**
   ```
   Error: Memory usage exceeded limit
   Solution: Reduce dataset size or increase memory limits
   ```

3. **Client Connection Failures**
   ```
   Error: Could not connect to Dask cluster
   Solution: Check port availability, try different client configuration
   ```

### Debug Mode

Run with verbose output for debugging:

```bash
# Verbose pytest output
pytest -v -s test_dask_benchmarks.py

# Python debug output
export DASK_LOGGING__DISTRIBUTED=DEBUG
python run_benchmarks.py
```

### Cleaning Up

```bash
# Clean old benchmark results (keep last 30 days)
python run_benchmarks.py --cleanup 30

# Remove all benchmark results
rm -rf tests/benchmark_results/
```

## Contributing

When adding new benchmark tests:

1. Use appropriate pytest markers
2. Include both success and failure cases
3. Measure both time and memory
4. Support platform-specific variations
5. Add documentation for new metrics
6. Test on multiple platforms when possible

### Adding New Benchmarks

```python
@pytest.mark.benchmark
@pytest.mark.custom_category  # Add custom marker
def test_new_benchmark(self, test_data):
    \"\"\"Description of what this benchmark measures.\"\"\"
    result = self.run_benchmark(
        "new_benchmark_name",
        test_data,
        # ... configuration
    )
    save_benchmark_result(result)
    assert result.success, f"Benchmark failed: {result.error_message}"
```

## Future Improvements

Planned enhancements:
- Automated performance regression detection in CI
- Integration with OG-Core's existing test suite  
- Benchmark result visualization dashboard
- Memory profiling with line-by-line analysis
- Cross-platform performance comparison reports
- Integration with cloud-based benchmarking services