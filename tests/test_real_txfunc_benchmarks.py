"""
Real tax function benchmark tests using actual OG-Core txfunc module.

This test module uses the real txfunc.tax_func_estimate function to benchmark
actual performance with different Dask configurations, providing more realistic
performance measurements than the mock tests.
"""

import os
import sys
import time
import platform
import tempfile
import shutil
from pathlib import Path
import json
import logging
import pytest
import numpy as np
import pandas as pd
from distributed import Client, LocalCluster
import dask.multiprocessing

from ogcore import txfunc, utils
from test_dask_benchmarks import (
    BenchmarkResult,
    MemoryTracker,
    timer,
    save_benchmark_result,
    generate_mock_tax_data,
)

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def create_realistic_micro_data(
    num_records_per_year: int = 5000, num_years: int = 3
):
    """
    Create micro data that closely matches what txfunc expects.

    This generates data with the exact structure and value ranges that
    the real tax function estimation uses.
    """
    micro_data = {}
    np.random.seed(12345)  # Fixed seed for reproducible benchmarks

    for year_idx, year in enumerate(range(2021, 2021 + num_years)):
        # Create base demographics
        n_records = num_records_per_year
        ages = np.random.randint(18, 80, n_records)

        # Create income data with realistic distributions
        # Labor income (wages + business)
        wage_income = np.random.lognormal(10.5, 1.2, n_records) * 1000
        business_income = np.where(
            np.random.random(n_records) < 0.15,  # 15% have business income
            np.random.lognormal(9.0, 1.8, n_records) * 1000,
            0,
        )

        # Capital income
        interest_income = np.where(
            np.random.random(n_records) < 0.6,  # 60% have interest
            np.random.lognormal(6.5, 1.5, n_records) * 100,
            0,
        )
        dividend_income = np.where(
            np.random.random(n_records) < 0.3,  # 30% have dividends
            np.random.lognormal(7.5, 2.0, n_records) * 100,
            0,
        )
        capital_gains = np.where(
            np.random.random(n_records) < 0.1,  # 10% have capital gains
            np.random.lognormal(8.5, 2.5, n_records) * 1000,
            0,
        )

        total_labor_income = wage_income + business_income
        total_capital_income = (
            interest_income + dividend_income + capital_gains
        )
        total_income = total_labor_income + total_capital_income

        # Generate tax data based on income
        # Simplified progressive tax calculation for realistic tax amounts
        marginal_rates = np.where(
            total_income < 20000,
            0.10,
            np.where(
                total_income < 50000,
                0.15,
                np.where(
                    total_income < 100000,
                    0.22,
                    np.where(total_income < 200000, 0.28, 0.32),
                ),
            ),
        )

        # Effective tax rates (lower than marginal due to deductions, etc.)
        effective_rates = (
            marginal_rates * 0.7 * (total_income / (total_income + 10000))
        )
        tax_liability = total_income * effective_rates

        # Add some noise to make it more realistic
        tax_liability *= np.random.normal(1.0, 0.1, n_records)
        tax_liability = np.maximum(tax_liability, 0)  # No negative taxes

        # Calculate marginal tax rates (approximate)
        mtr_labor = marginal_rates * np.random.normal(1.0, 0.05, n_records)
        mtr_capital = (
            marginal_rates * 0.8 * np.random.normal(1.0, 0.05, n_records)
        )  # Usually lower

        # Create weights (survey weights)
        weights = np.random.uniform(500, 2000, n_records)

        # Create the DataFrame with all required columns
        data = pd.DataFrame(
            {
                # Identifiers
                "RECID": np.arange(n_records),
                "MARS": np.random.choice(
                    [1, 2, 3, 4], n_records, p=[0.4, 0.45, 0.1, 0.05]
                ),
                "FLPDYR": year,
                # Demographics
                "age": ages,
                "AGEP": ages,  # Primary taxpayer age
                "AGAGE": ages,  # Duplicate for compatibility
                "AGEX": ages,  # Another age field
                "year": year,  # Year field expected by txfunc
                # Income components
                "e00200": wage_income,  # Wages and salaries
                "e00900": business_income,  # Business income
                "e00300": interest_income,  # Interest income
                "e00600": dividend_income,  # Dividend income
                "p22250": capital_gains,  # Capital gains
                "e02000": 0,  # Other income (set to 0 for simplicity)
                # Tax amounts
                "c05800": tax_liability,  # Income tax before credits
                "iitax": tax_liability,  # Final income tax liability
                "payrolltax": total_labor_income * 0.153,  # Payroll tax (FICA)
                # Calculated fields that txfunc expects - using exact column names from test data
                "total_labinc": total_labor_income,
                "total_capinc": total_capital_income,
                "market_income": total_income,  # This is the missing column!
                "etr": np.clip(
                    tax_liability / np.maximum(total_income, 1), 0, 0.6
                ),
                "mtr_labinc": np.clip(
                    mtr_labor, 0, 0.8
                ),  # MTR on labor (correct name)
                "mtr_capinc": np.clip(
                    mtr_capital, 0, 0.8
                ),  # MTR on capital (correct name)
                "total_tax_liab": tax_liability,  # Total tax liability
                "payroll_tax_liab": total_labor_income
                * 0.153,  # Payroll tax liability
                # Weights - use the standard name from test data
                "weight": weights,  # Sample weight (standard name)
                "s006": weights,  # Alternative weight name for compatibility
                "wgts": weights,  # Another alternative weight name
            }
        )

        # Filter out extreme or invalid cases
        valid_mask = (
            (data["total_labinc"] >= 0)
            & (data["total_capinc"] >= 0)
            & (data["market_income"] >= 0)
            & (data["etr"] >= 0)
            & (data["etr"] <= 1)
            & (data["mtr_labinc"] >= 0)
            & (data["mtr_labinc"] <= 1)
            & (data["mtr_capinc"] >= 0)
            & (data["mtr_capinc"] <= 1)
            & (data["weight"] > 0)
        )

        data = data[valid_mask].reset_index(drop=True)
        micro_data[str(year)] = data

        logging.info(
            f"Generated {len(data)} valid tax records for year {year}"
        )

    return micro_data


class TestRealTaxFuncBenchmarks:
    """Benchmark tests using real tax function estimation."""

    @pytest.fixture(scope="class")
    def small_real_data(self):
        """Generate small realistic dataset."""
        return create_realistic_micro_data(
            num_records_per_year=2000, num_years=2
        )

    @pytest.fixture(scope="class")
    def medium_real_data(self):
        """Generate medium realistic dataset."""
        return create_realistic_micro_data(
            num_records_per_year=1500, num_years=3
        )

    def run_real_benchmark(
        self,
        test_name: str,
        micro_data: dict,
        client: Client = None,
        num_workers: int = 2,
        tax_func_type: str = "DEP",
    ) -> BenchmarkResult:
        """
        Run benchmark using real tax function estimation.

        Args:
            test_name: Name of the benchmark test
            micro_data: Dictionary of tax data by year
            client: Dask client to use (None for direct scheduler)
            num_workers: Number of workers
            tax_func_type: Type of tax function to estimate

        Returns:
            BenchmarkResult with performance metrics
        """
        # Calculate data size
        data_size_mb = 0
        for year_data in micro_data.values():
            data_size_mb += (
                year_data.memory_usage(deep=True).sum() / 1024 / 1024
            )

        # Set up parameters for tax function estimation
        BW = len(micro_data)  # Bandwidth (number of years)
        S = 80  # Number of age groups
        starting_age = 20
        ending_age = 99
        start_year = 2021

        error_message = None
        success = True

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            with MemoryTracker() as mem_tracker:
                with timer() as get_time:
                    try:
                        # Call the real tax function estimation
                        result = txfunc.tax_func_estimate(
                            micro_data=micro_data,
                            BW=BW,
                            S=S,
                            starting_age=starting_age,
                            ending_age=ending_age,
                            start_year=start_year,
                            analytical_mtrs=False,
                            tax_func_type=tax_func_type,
                            age_specific=False,  # Disable for faster benchmarking
                            desc_data=False,
                            graph_data=False,
                            graph_est=False,
                            client=client,
                            num_workers=num_workers,
                            tax_func_path=None,
                        )

                        # Sample memory during computation
                        mem_tracker.sample_memory()

                        # Validate that we got reasonable results
                        if result is None:
                            raise ValueError("tax_func_estimate returned None")

                        # Check that result has expected structure
                        expected_keys = [
                            "tfunc_etr_params_S",
                            "tfunc_mtrx_params_S",
                            "tfunc_mtry_params_S",
                        ]
                        for key in expected_keys:
                            if key not in result:
                                raise ValueError(
                                    f"Missing expected result key: {key}"
                                )

                    except Exception as e:
                        error_message = str(e)
                        success = False
                        result = None

        compute_time = get_time()
        scheduler_name = "distributed" if client else "direct"

        return BenchmarkResult(
            test_name=test_name,
            platform=platform.system(),
            scheduler=scheduler_name,
            num_workers=num_workers,
            compute_time=compute_time,
            peak_memory_mb=mem_tracker.peak_memory,
            avg_memory_mb=mem_tracker.average_memory,
            data_size_mb=data_size_mb,
            num_tasks=len(micro_data),
            success=success,
            error_message=error_message,
        )

    @pytest.mark.benchmark
    @pytest.mark.real
    def test_real_small_no_client(self, small_real_data):
        """Benchmark small dataset without Dask client."""
        result = self.run_real_benchmark(
            "real_small_no_client", small_real_data, client=None, num_workers=2
        )
        save_benchmark_result(result)

        logging.info(
            f"Real small (no client): {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
        )
        assert result.success, f"Real benchmark failed: {result.error_message}"

    @pytest.mark.benchmark
    @pytest.mark.real
    @pytest.mark.distributed
    def test_real_small_threaded_client(self, small_real_data):
        """Benchmark small dataset with threaded Dask client."""
        cluster = LocalCluster(
            n_workers=2,
            threads_per_worker=2,
            processes=False,
            memory_limit="1GB",
            silence_logs=True,
        )

        try:
            client = Client(cluster)
            result = self.run_real_benchmark(
                "real_small_threaded",
                small_real_data,
                client=client,
                num_workers=2,
            )
            save_benchmark_result(result)

            logging.info(
                f"Real small (threaded): {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
            )
            assert (
                result.success
            ), f"Real threaded benchmark failed: {result.error_message}"

        finally:
            client.close()
            cluster.close()

    @pytest.mark.benchmark
    @pytest.mark.real
    @pytest.mark.distributed
    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Multiprocessing issues on Windows",
    )
    def test_real_small_process_client(self, small_real_data):
        """Benchmark small dataset with process-based Dask client."""
        cluster = LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit="1GB",
            silence_logs=True,
        )

        try:
            client = Client(cluster)
            result = self.run_real_benchmark(
                "real_small_processes",
                small_real_data,
                client=client,
                num_workers=2,
            )
            save_benchmark_result(result)

            logging.info(
                f"Real small (processes): {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB peak"
            )
            assert (
                result.success
            ), f"Real process benchmark failed: {result.error_message}"

        finally:
            client.close()
            cluster.close()

    @pytest.mark.benchmark
    @pytest.mark.real
    @pytest.mark.performance
    def test_real_medium_comparison(self, medium_real_data):
        """Compare different configurations on medium dataset."""
        configs = []

        # Always test no-client configuration
        configs.append(("no_client", None))

        # Threaded client (works on all platforms)
        try:
            threaded_cluster = LocalCluster(
                n_workers=2,
                threads_per_worker=2,
                processes=False,
                memory_limit="1GB",
                silence_logs=True,
            )
            threaded_client = Client(threaded_cluster)
            configs.append(("threaded", threaded_client))
        except Exception as e:
            logging.info(f"Failed to create threaded client: {e}")

        # Process client (skip on Windows due to known issues)
        if platform.system() != "Windows":
            try:
                process_cluster = LocalCluster(
                    n_workers=2,
                    threads_per_worker=1,
                    processes=True,
                    memory_limit="1GB",
                    silence_logs=True,
                )
                process_client = Client(process_cluster)
                configs.append(("processes", process_client))
            except Exception as e:
                logging.info(f"Failed to create process client: {e}")

        results = []

        try:
            for config_name, client in configs:
                result = self.run_real_benchmark(
                    f"real_medium_{config_name}",
                    medium_real_data,
                    client=client,
                    num_workers=2,
                )
                results.append(result)
                save_benchmark_result(result)

                logging.info(
                    f"Real medium ({config_name}): {result.compute_time:.3f}s, {result.peak_memory_mb:.1f}MB"
                )

        finally:
            # Clean up clients
            for _, client in configs:
                if client:
                    try:
                        client.close()
                        if hasattr(client, "cluster"):
                            client.cluster.close()
                    except:
                        pass

        # At least one configuration should succeed
        successful_results = [r for r in results if r.success]
        assert (
            len(successful_results) > 0
        ), "All real medium benchmark configurations failed"

        # Report performance comparison
        if len(successful_results) > 1:
            fastest = min(successful_results, key=lambda x: x.compute_time)
            logging.info(
                f"Fastest configuration: {fastest.scheduler} ({fastest.compute_time:.3f}s)"
            )

    @pytest.mark.benchmark
    @pytest.mark.real
    @pytest.mark.memory
    def test_real_memory_efficiency(self, small_real_data, medium_real_data):
        """Test memory efficiency scaling with real tax function estimation."""
        datasets = [
            ("small", small_real_data),
            ("medium", medium_real_data),
        ]

        results = []
        for name, data in datasets:
            result = self.run_real_benchmark(
                f"real_memory_{name}",
                data,
                client=None,  # Use direct scheduler for consistent memory measurement
                num_workers=2,
            )
            results.append(result)
            save_benchmark_result(result)

            memory_efficiency = (
                result.peak_memory_mb / result.data_size_mb
                if result.data_size_mb > 0
                else float("inf")
            )
            logging.info(
                f"Real memory {name}: {result.data_size_mb:.1f}MB data -> {result.peak_memory_mb:.1f}MB peak (efficiency: {memory_efficiency:.2f})"
            )

        # Check reasonable memory scaling
        if len(results) >= 2 and all(r.success for r in results):
            small_result, medium_result = results[0], results[1]

            data_ratio = medium_result.data_size_mb / small_result.data_size_mb
            memory_ratio = (
                medium_result.peak_memory_mb / small_result.peak_memory_mb
            )

            logging.info(
                f"Memory scaling: {memory_ratio:.2f}x memory for {data_ratio:.2f}x data"
            )

            # Memory should not scale too poorly (allow some overhead for processing)
            assert (
                memory_ratio <= data_ratio * 3
            ), f"Poor memory scaling: {memory_ratio:.2f}x memory for {data_ratio:.2f}x data"


@pytest.mark.benchmark
@pytest.mark.real
@pytest.mark.platform
def test_platform_specific_optimal_config():
    """
    Test to identify the optimal Dask configuration for the current platform.

    This test runs multiple configurations and identifies which performs best
    on the current platform, helping inform platform-specific optimizations.
    """
    # Generate small test dataset
    test_data = create_realistic_micro_data(
        num_records_per_year=800, num_years=2
    )

    configurations = []

    # Configuration 1: No client (direct scheduler with multiprocessing)
    configurations.append(("direct_multiprocessing", None, "multiprocessing"))

    # Configuration 2: No client (direct scheduler with threads)
    configurations.append(("direct_threaded", None, "threads"))

    # Configuration 3: Distributed threaded
    try:
        cluster = LocalCluster(
            n_workers=2,
            threads_per_worker=2,
            processes=False,
            silence_logs=True,
        )
        client = Client(cluster)
        configurations.append(("distributed_threaded", client, "threaded"))
    except Exception as e:
        logging.info(f"Could not create threaded client: {e}")

    # Configuration 4: Distributed processes (skip on Windows)
    if platform.system() != "Windows":
        try:
            cluster = LocalCluster(
                n_workers=2,
                threads_per_worker=1,
                processes=True,
                silence_logs=True,
            )
            client = Client(cluster)
            configurations.append(
                ("distributed_processes", client, "processes")
            )
        except Exception as e:
            logging.info(f"Could not create process client: {e}")

    results = []

    try:
        for config_name, client, scheduler_type in configurations:
            with MemoryTracker() as mem_tracker:
                with timer() as get_time:
                    try:
                        if client:
                            # Use distributed client
                            result = txfunc.tax_func_estimate(
                                micro_data=test_data,
                                BW=len(test_data),
                                S=80,
                                starting_age=20,
                                ending_age=99,
                                start_year=2021,
                                analytical_mtrs=False,
                                tax_func_type="DEP",
                                age_specific=False,  # Faster for benchmarking
                                desc_data=False,
                                graph_data=False,
                                graph_est=False,
                                client=client,
                                num_workers=2,
                                tax_func_path=None,
                            )
                        else:
                            # Use direct scheduler - simulate the actual call pattern
                            if scheduler_type == "multiprocessing":
                                result = txfunc.tax_func_estimate(
                                    micro_data=test_data,
                                    BW=len(test_data),
                                    S=80,
                                    starting_age=20,
                                    ending_age=99,
                                    start_year=2021,
                                    analytical_mtrs=False,
                                    tax_func_type="DEP",
                                    age_specific=False,
                                    desc_data=False,
                                    graph_data=False,
                                    graph_est=False,
                                    client=None,  # This will use multiprocessing scheduler
                                    num_workers=2,
                                    tax_func_path=None,
                                )
                            else:  # threads
                                # For threaded, we'd need to modify the txfunc code
                                # For now, skip this case in real testing
                                continue

                        success = result is not None
                        error_message = None

                    except Exception as e:
                        success = False
                        error_message = str(e)
                        result = None

            compute_time = get_time()

            benchmark_result = BenchmarkResult(
                test_name=f"platform_optimal_{config_name}",
                platform=platform.system(),
                scheduler=config_name,
                num_workers=2,
                compute_time=compute_time,
                peak_memory_mb=mem_tracker.peak_memory,
                avg_memory_mb=mem_tracker.average_memory,
                data_size_mb=sum(
                    df.memory_usage(deep=True).sum()
                    for df in test_data.values()
                )
                / 1024
                / 1024,
                num_tasks=len(test_data),
                success=success,
                error_message=error_message,
            )

            results.append(benchmark_result)
            save_benchmark_result(benchmark_result)

            if success:
                logging.info(
                    f"{config_name}: {compute_time:.3f}s, {mem_tracker.peak_memory:.1f}MB"
                )
            else:
                logging.info(f"{config_name}: FAILED - {error_message}")

    finally:
        # Clean up any clients
        for _, client, _ in configurations:
            if client:
                try:
                    client.close()
                    if hasattr(client, "cluster"):
                        client.cluster.close()
                except Exception:
                    pass

    # Report optimal configuration
    successful_results = [r for r in results if r.success]
    if successful_results:
        optimal = min(successful_results, key=lambda x: x.compute_time)
        logging.info(
            f"\nOptimal configuration for {platform.system()}: {optimal.scheduler}"
        )
        logging.info(
            f"Time: {optimal.compute_time:.3f}s, Memory: {optimal.peak_memory_mb:.1f}MB"
        )

        # Save platform-specific recommendation
        recommendation = {
            "platform": platform.system(),
            "optimal_config": optimal.scheduler,
            "performance": {
                "time": optimal.compute_time,
                "memory": optimal.peak_memory_mb,
            },
            "all_results": [r.to_dict() for r in successful_results],
        }

        rec_file = os.path.join(
            os.path.dirname(__file__),
            "benchmark_results",
            f"platform_recommendation_{platform.system().lower()}.json",
        )
        os.makedirs(os.path.dirname(rec_file), exist_ok=True)

        with open(rec_file, "w") as f:
            json.dump(recommendation, f, indent=2)

        logging.info(f"Saved platform recommendation to {rec_file}")

    assert len(successful_results) > 0, "No benchmark configurations succeeded"
