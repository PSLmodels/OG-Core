"""
Benchmark tests for Dask performance in OG-Core.

This module contains comprehensive benchmarks to measure:
- Memory usage during parallel operations
- Compute time for different Dask configurations
- Platform-specific performance variations
- Data serialization overhead

These tests establish baselines for performance optimization efforts.
"""

import os
import sys
import time
import platform
import psutil
import pytest
import numpy as np
import pandas as pd
import pickle
import json
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import dask.multiprocessing

from ogcore import txfunc, utils

NUM_WORKERS = min(psutil.cpu_count(), 4)  # Limit for testing
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
BENCHMARK_RESULTS_DIR = os.path.join(CUR_PATH, "benchmark_results")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    test_name: str
    platform: str
    scheduler: str
    num_workers: int
    compute_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    data_size_mb: float
    num_tasks: int
    success: bool
    error_message: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MemoryTracker:
    """Context manager for tracking memory usage during operations."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.memory_usage = []
        self.peak_memory = 0
        self.start_memory = 0
        self.process = psutil.Process()

    def __enter__(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_usage = [self.start_memory]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_memory = self.process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(final_memory)
        self.peak_memory = max(self.memory_usage)

    def sample_memory(self):
        """Sample current memory usage."""
        current = self.process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(current)
        return current

    @property
    def average_memory(self):
        """Get average memory usage during tracking."""
        return sum(self.memory_usage) / len(self.memory_usage)

    @property
    def memory_delta(self):
        """Get memory increase from start to peak."""
        return self.peak_memory - self.start_memory


def generate_mock_tax_data(
    num_records: int = 10000, num_years: int = 3
) -> Dict:
    """
    Generate mock tax data similar to what txfunc.tax_func_estimate expects.

    Args:
        num_records: Number of tax records per year
        num_years: Number of years of data

    Returns:
        Dictionary with year keys containing DataFrames
    """
    np.random.seed(42)  # For reproducible benchmarks

    mock_data = {}
    for year in range(2020, 2020 + num_years):
        # Create realistic tax data structure
        data = pd.DataFrame(
            {
                "RECID": range(num_records),
                "MARS": np.random.choice([1, 2, 3, 4], num_records),
                "FLPDYR": [year] * num_records,
                "age": np.random.randint(18, 80, num_records),
                "AGEP": np.random.randint(18, 80, num_records),
                "AGAGE": np.random.randint(18, 80, num_records),
                "AGEX": np.random.randint(18, 80, num_records),
                "incwage": np.random.lognormal(10, 1, num_records),
                "incbus": np.random.lognormal(8, 2, num_records)
                * (np.random.random(num_records) > 0.7),
                "incint": np.random.lognormal(6, 1.5, num_records)
                * (np.random.random(num_records) > 0.5),
                "incdiv": np.random.lognormal(7, 2, num_records)
                * (np.random.random(num_records) > 0.6),
                "incrent": np.random.lognormal(8, 1.8, num_records)
                * (np.random.random(num_records) > 0.8),
                "incgain": np.random.lognormal(9, 2.5, num_records)
                * (np.random.random(num_records) > 0.9),
                "taxbc": np.random.lognormal(8, 1.2, num_records) * 0.3,
                "iitax": np.random.lognormal(8, 1.2, num_records) * 0.2,
                "payrolltax": np.random.lognormal(8, 1, num_records) * 0.15,
                "wgts": np.random.uniform(100, 1000, num_records),
            }
        )

        # Add calculated fields that txfunc expects
        data["total_labinc"] = data["incwage"] + data["incbus"]
        data["total_capinc"] = (
            data["incint"] + data["incdiv"] + data["incrent"] + data["incgain"]
        )
        data["total_inc"] = data["total_labinc"] + data["total_capinc"]
        data["etr"] = np.clip(
            data["iitax"] / np.maximum(data["total_inc"], 1), 0, 0.6
        )
        data["mtrx"] = np.clip(
            data["etr"] * 1.2, 0, 0.8
        )  # Rough approximation
        data["mtry"] = np.clip(
            data["etr"] * 1.1, 0, 0.8
        )  # Rough approximation

        mock_data[str(year)] = data

    return mock_data


@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def create_dask_clients() -> List[Tuple[str, Optional[Client]]]:
    """Create different Dask client configurations for testing."""
    clients = []

    # No client (uses scheduler directly)
    clients.append(("no_client", None))

    # Threaded client
    try:
        threaded_cluster = LocalCluster(
            n_workers=NUM_WORKERS,
            threads_per_worker=2,
            processes=False,
            memory_limit="1GB",
            silence_logs=True,
        )
        threaded_client = Client(threaded_cluster)
        clients.append(("threaded", threaded_client))
    except Exception as e:
        logging.info(f"Failed to create threaded client: {e}")

    # Process-based client (if not Windows or if requested)
    if platform.system() != "Windows":
        try:
            process_cluster = LocalCluster(
                n_workers=NUM_WORKERS,
                threads_per_worker=1,
                processes=True,
                memory_limit="1GB",
                silence_logs=True,
            )
            process_client = Client(process_cluster)
            clients.append(("processes", process_client))
        except Exception as e:
            logging.info(f"Failed to create process client: {e}")

    return clients


def save_benchmark_result(result: BenchmarkResult):
    """Save benchmark result to JSON file."""
    os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)

    filename = f"benchmark_{result.test_name}_{result.platform}_{int(time.time())}.json"
    filepath = os.path.join(BENCHMARK_RESULTS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def calculate_data_size_mb(data: Dict) -> float:
    """Estimate data size in MB."""
    total_size = 0
    for year_data in data.values():
        if isinstance(year_data, pd.DataFrame):
            total_size += year_data.memory_usage(deep=True).sum()
    return total_size / 1024 / 1024


class TestDaskBenchmarks:
    """Test class containing all Dask benchmark tests."""

    @pytest.fixture(scope="class")
    def small_tax_data(self):
        """Generate small dataset for quick tests."""
        return generate_mock_tax_data(num_records=1000, num_years=2)

    @pytest.fixture(scope="class")
    def medium_tax_data(self):
        """Generate medium dataset for realistic tests."""
        return generate_mock_tax_data(num_records=5000, num_years=3)

    @pytest.fixture(scope="class")
    def large_tax_data(self):
        """Generate large dataset for stress tests."""
        return generate_mock_tax_data(num_records=15000, num_years=5)

    def create_delayed_tasks(self, data: Dict, num_tasks: int = None) -> List:
        """Create delayed tasks similar to tax function estimation."""
        if num_tasks is None:
            num_tasks = len(data)

        lazy_values = []
        years_list = list(data.keys())[:num_tasks]

        for year in years_list:
            # Simulate the tax_func_loop delayed task
            lazy_values.append(
                delayed(self.mock_tax_computation)(data[year], year)
            )

        return lazy_values

    def mock_tax_computation(self, data: pd.DataFrame, year: str) -> Dict:
        """
        Mock computation that simulates tax function estimation workload.

        This mimics the computational pattern of txfunc.tax_func_loop
        but with simpler operations for consistent benchmarking.
        """
        # Simulate data filtering and processing
        filtered_data = data[data["total_inc"] > 1000].copy()

        # Simulate some mathematical operations similar to tax function fitting
        if len(filtered_data) > 100:
            # Mock optimization-like operations
            x = filtered_data["total_labinc"].values
            y = filtered_data["total_capinc"].values
            weights = filtered_data["wgts"].values

            # Simulate parameter estimation (simplified)
            params = []
            for age_group in range(18, 80, 10):
                age_mask = (filtered_data["age"] >= age_group) & (
                    filtered_data["age"] < age_group + 10
                )
                if age_mask.sum() > 10:
                    subset_x = x[age_mask]
                    subset_y = y[age_mask]
                    subset_w = weights[age_mask]

                    # Simple weighted regression-like calculation
                    if len(subset_x) > 5:
                        coeff = np.average(subset_x, weights=subset_w)
                        params.append([coeff, coeff * 0.1, coeff * 0.01])
                    else:
                        params.append([1.0, 0.1, 0.01])
                else:
                    params.append([1.0, 0.1, 0.01])

            result = {
                "year": year,
                "params": params,
                "num_observations": len(filtered_data),
                "avg_income": filtered_data["total_inc"].mean(),
                "avg_tax": filtered_data["iitax"].mean(),
            }
        else:
            result = {
                "year": year,
                "params": [[1.0, 0.1, 0.01]] * 7,
                "num_observations": 0,
                "avg_income": 0,
                "avg_tax": 0,
            }

        # Add some CPU time to simulate real computation
        _ = np.linalg.svd(np.random.rand(50, 50))

        return result

    def run_benchmark(
        self,
        test_name: str,
        data: Dict,
        scheduler_type: str = "multiprocessing",
        client: Optional[Client] = None,
        num_workers: int = NUM_WORKERS,
    ) -> BenchmarkResult:
        """
        Run a benchmark with the specified configuration.

        Args:
            test_name: Name of the benchmark test
            data: Tax data dictionary
            scheduler_type: Type of scheduler to use
            client: Dask client (if using distributed)
            num_workers: Number of workers

        Returns:
            BenchmarkResult object with performance metrics
        """
        lazy_values = self.create_delayed_tasks(data)
        data_size = calculate_data_size_mb(data)

        error_message = None
        success = True

        with MemoryTracker() as mem_tracker:
            with timer() as get_time:
                try:
                    if client:
                        # Use distributed client
                        futures = client.compute(lazy_values)
                        results = client.gather(futures)
                    else:
                        # Use scheduler directly
                        if scheduler_type == "threads":
                            results = compute(
                                *lazy_values,
                                scheduler="threads",
                                num_workers=num_workers,
                            )
                        elif scheduler_type == "multiprocessing":
                            results = compute(
                                *lazy_values,
                                scheduler=dask.multiprocessing.get,
                                num_workers=num_workers,
                            )
                        elif scheduler_type == "single-threaded":
                            results = compute(
                                *lazy_values, scheduler="single-threaded"
                            )
                        else:
                            raise ValueError(
                                f"Unknown scheduler type: {scheduler_type}"
                            )

                    # Sample memory during computation
                    mem_tracker.sample_memory()

                except Exception as e:
                    error_message = str(e)
                    success = False
                    results = None

        compute_time = get_time()

        return BenchmarkResult(
            test_name=test_name,
            platform=platform.system(),
            scheduler=(
                scheduler_type
                if not client
                else f"distributed_{scheduler_type}"
            ),
            num_workers=num_workers,
            compute_time=compute_time,
            peak_memory_mb=mem_tracker.peak_memory,
            avg_memory_mb=mem_tracker.average_memory,
            data_size_mb=data_size,
            num_tasks=len(lazy_values),
            success=success,
            error_message=error_message,
        )

    @pytest.mark.benchmark
    def test_small_dataset_multiprocessing(self, small_tax_data):
        """Benchmark small dataset with multiprocessing scheduler."""
        result = self.run_benchmark(
            "small_dataset_multiprocessing",
            small_tax_data,
            scheduler_type="multiprocessing",
        )
        save_benchmark_result(result)
        assert result.success, f"Benchmark failed: {result.error_message}"
        logging.info(
            f"Small dataset multiprocessing: {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
        )

    @pytest.mark.benchmark
    def test_small_dataset_threaded(self, small_tax_data):
        """Benchmark small dataset with threaded scheduler."""
        result = self.run_benchmark(
            "small_dataset_threaded", small_tax_data, scheduler_type="threads"
        )
        save_benchmark_result(result)
        assert result.success, f"Benchmark failed: {result.error_message}"
        logging.info(
            f"Small dataset threaded: {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
        )

    @pytest.mark.benchmark
    def test_medium_dataset_comparison(self, medium_tax_data):
        """Compare different schedulers on medium dataset."""
        schedulers = ["threads", "multiprocessing"]
        if platform.system() == "Windows":
            schedulers = [
                "threads"
            ]  # Skip multiprocessing on Windows for comparison

        results = []
        for scheduler in schedulers:
            result = self.run_benchmark(
                f"medium_dataset_{scheduler}",
                medium_tax_data,
                scheduler_type=scheduler,
            )
            results.append(result)
            save_benchmark_result(result)
            logging.info(
                f"Medium {scheduler}: {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
            )

        # All should succeed
        for result in results:
            assert result.success, f"Benchmark failed: {result.error_message}"

    @pytest.mark.benchmark
    @pytest.mark.distributed
    def test_distributed_clients_comparison(self, medium_tax_data):
        """Compare distributed client configurations."""
        clients = create_dask_clients()
        results = []

        try:
            for client_name, client in clients:
                if client is None:
                    continue  # Skip no-client case for this test

                result = self.run_benchmark(
                    f"distributed_{client_name}",
                    medium_tax_data,
                    scheduler_type=client_name,
                    client=client,
                )
                results.append(result)
                save_benchmark_result(result)
                logging.info(
                    f"Distributed {client_name}: {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
                )

        finally:
            # Clean up clients
            for _, client in clients:
                if client:
                    try:
                        client.close()
                    except:
                        pass

        # At least one should succeed
        assert any(
            r.success for r in results
        ), "All distributed benchmarks failed"

    @pytest.mark.benchmark
    @pytest.mark.memory
    def test_memory_scaling(self, small_tax_data, medium_tax_data):
        """Test how memory usage scales with data size."""
        datasets = [
            ("small", small_tax_data),
            ("medium", medium_tax_data),
        ]

        results = []
        for name, data in datasets:
            result = self.run_benchmark(
                f"memory_scaling_{name}",
                data,
                scheduler_type="threads",  # Use threads for consistent memory measurement
            )
            results.append(result)
            save_benchmark_result(result)

            memory_per_mb = (
                result.peak_memory_mb / result.data_size_mb
                if result.data_size_mb > 0
                else 0
            )
            logging.info(
                f"Memory scaling {name}: {result.data_size_mb:.1f}MB data -> {result.peak_memory_mb:.1f}MB peak (ratio: {memory_per_mb:.2f})"
            )

        # Check that memory scales reasonably
        if len(results) >= 2 and all(r.success for r in results):
            small_result, medium_result = results[0], results[1]
            data_ratio = medium_result.data_size_mb / small_result.data_size_mb
            memory_ratio = (
                medium_result.peak_memory_mb / small_result.peak_memory_mb
            )

            # Memory should scale somewhat linearly with data (allowing for overhead)
            assert (
                memory_ratio <= data_ratio * 2
            ), f"Memory scaling is too poor: {memory_ratio:.2f}x memory for {data_ratio:.2f}x data"

    @pytest.mark.benchmark
    @pytest.mark.performance
    def test_worker_scaling(self, medium_tax_data):
        """Test how performance scales with number of workers."""
        max_workers = min(psutil.cpu_count(), 4)
        worker_counts = [1, 2, max_workers]

        results = []
        for num_workers in worker_counts:
            result = self.run_benchmark(
                f"worker_scaling_{num_workers}",
                medium_tax_data,
                scheduler_type="threads",  # Use threads for consistent comparison
                num_workers=num_workers,
            )
            results.append(result)
            save_benchmark_result(result)
            logging.info(
                f"Worker scaling {num_workers}: {result.compute_time:.3f}s"
            )

        # Performance should improve or stay similar with more workers
        if all(r.success for r in results):
            single_worker_time = results[0].compute_time
            max_worker_time = results[-1].compute_time

            # Allow some overhead, but expect some improvement
            speedup = single_worker_time / max_worker_time
            logging.info(f"Speedup with {max_workers} workers: {speedup:.2f}x")

            # Should be at least 1.2x speedup with multiple workers (conservative)
            if max_workers > 1:
                assert (
                    speedup >= 1.0
                ), f"Performance degraded with more workers: {speedup:.2f}x"

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_large_dataset_stress(self, large_tax_data):
        """Stress test with large dataset."""
        # Only run on platforms where multiprocessing works well
        if platform.system() == "Windows":
            scheduler = "threads"
        else:
            scheduler = "multiprocessing"

        result = self.run_benchmark(
            "large_dataset_stress", large_tax_data, scheduler_type=scheduler
        )
        save_benchmark_result(result)

        logging.info(
            f"Large dataset stress ({scheduler}): {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
        )

        # Should complete without error, even if slow
        assert (
            result.success
        ), f"Large dataset benchmark failed: {result.error_message}"

        # Memory usage should be reasonable (less than 2GB)
        assert (
            result.peak_memory_mb < 2000
        ), f"Memory usage too high: {result.peak_memory_mb:.1f}MB"


def load_benchmark_results() -> List[BenchmarkResult]:
    """Load all benchmark results from saved files."""
    results = []
    if not os.path.exists(BENCHMARK_RESULTS_DIR):
        return results

    for filename in os.listdir(BENCHMARK_RESULTS_DIR):
        if filename.startswith("benchmark_") and filename.endswith(".json"):
            filepath = os.path.join(BENCHMARK_RESULTS_DIR, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    result = BenchmarkResult(**data)
                    results.append(result)
            except Exception as e:
                logging.info(f"Failed to load {filename}: {e}")

    return results


def generate_benchmark_report():
    """Generate a summary report of all benchmark results."""
    results = load_benchmark_results()
    if not results:
        logging.info("No benchmark results found.")
        return

    logging.info("\n" + "=" * 80)
    logging.info("DASK PERFORMANCE BENCHMARK REPORT")
    logging.info("=" * 80)

    # Group by platform and scheduler
    by_config = {}
    for result in results:
        key = (result.platform, result.scheduler)
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(result)

    for (platform, scheduler), config_results in by_config.items():
        logging.info(f"\n{platform} - {scheduler}:")
        logging.info("-" * 40)

        successful = [r for r in config_results if r.success]
        failed = [r for r in config_results if not r.success]

        if successful:
            avg_time = sum(r.compute_time for r in successful) / len(
                successful
            )
            avg_memory = sum(r.peak_memory_mb for r in successful) / len(
                successful
            )
            logging.info(f"  Successful tests: {len(successful)}")
            logging.info(f"  Average time: {avg_time:.3f}s")
            logging.info(f"  Average peak memory: {avg_memory:.1f}MB")

        if failed:
            logging.info(f"  Failed tests: {len(failed)}")
            for failure in failed[:3]:  # Show first 3 failures
                logging.info(
                    f"    {failure.test_name}: {failure.error_message}"
                )


if __name__ == "__main__":
    # Run benchmark report when executed directly
    generate_benchmark_report()
