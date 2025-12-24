"""
Example demonstrating the tax_filer parameter in OG-Core.

This script shows how to model income tax non-filers using the tax_filer
parameter. It compares a baseline where the lowest income group (j=0) are
non-filers to a reform where all income groups file taxes.

Non-filers:
- Pay zero income tax (only payroll taxes)
- Face zero marginal tax rates on labor and capital income
- Experience no tax distortions on labor supply and savings decisions

This feature is useful for:
- Modeling filing thresholds (e.g., standard deduction effects)
- Analyzing tax compliance policies
- Studying the economic effects of tax filing requirements
"""

import multiprocessing
from distributed import Client
import time
import numpy as np
import os
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.constants import REFORM_DIR, BASELINE_DIR
from ogcore.utils import safe_read_pickle
from ogcore import output_tables as ot
import pandas as pd


def main():
    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 7)
    print("=" * 70)
    print("OG-CORE EXAMPLE: MODELING INCOME TAX NON-FILERS")
    print("=" * 70)
    print(f"Number of workers = {num_workers}")

    client = Client(n_workers=num_workers, threads_per_worker=1)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(CUR_DIR, "NonFiler_Example")
    base_dir = os.path.join(save_dir, BASELINE_DIR)
    reform_dir = os.path.join(save_dir, REFORM_DIR)

    # Start timer
    run_start_time = time.time()

    # Common parameters for both baseline and reform
    # These create a simpler model for faster demonstration
    common_spec = {
        "frisch": 0.41,
        "start_year": 2024,
        "cit_rate": [[0.21]],
        "debt_ratio_ss": 0.4,
        "S": 80,  # 80 age periods
        "J": 7,  # 7 lifetime income groups
    }

    print("\n" + "-" * 70)
    print("BASELINE: Income group j=0 are NON-FILERS")
    print("-" * 70)
    print("\nIn the baseline, the lowest lifetime income group (j=0) does not")
    print("file income taxes. They pay only payroll taxes and face zero")
    print("marginal tax rates on labor and capital income.")

    # Baseline specification: j=0 are non-filers
    baseline_spec = common_spec.copy()
    baseline_spec.update(
        {
            # tax_filer is a J-length vector:
            #   0.0 = non-filer (no income tax, zero MTRs)
            #   1.0 = filer (normal income tax treatment)
            #   Values between 0-1 represent partial filing (e.g., 0.5 = 50% file)
            "tax_filer": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    p_baseline = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    p_baseline.update_specifications(baseline_spec)

    print(f"\nBaseline tax_filer parameter: {p_baseline.tax_filer}")
    print(f"  • Group j=0 (lowest income):  NON-FILER (tax_filer[0] = 0.0)")
    print(f"  • Groups j=1 to j=6:          FILERS (tax_filer = 1.0)")

    start_time = time.time()
    print("\nRunning baseline steady state...")
    runner(p_baseline, time_path=False, client=client)
    print(f"Baseline run time: {time.time() - start_time:.1f} seconds")

    # Load baseline results
    baseline_ss = safe_read_pickle(os.path.join(base_dir, "SS", "SS_vars.pkl"))
    baseline_params = safe_read_pickle(
        os.path.join(base_dir, "model_params.pkl")
    )

    print("\n" + "-" * 70)
    print("REFORM: ALL income groups are FILERS")
    print("-" * 70)
    print("\nIn the reform, all income groups file taxes, including j=0.")
    print("This creates tax distortions for the lowest income group.")

    # Reform specification: all groups are filers
    reform_spec = common_spec.copy()
    reform_spec.update(
        {
            "tax_filer": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    p_reform = Specifications(
        baseline=False,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=reform_dir,
    )
    p_reform.update_specifications(reform_spec)

    print(f"\nReform tax_filer parameter: {p_reform.tax_filer}")
    print(f"  • All groups j=0 to j=6:      FILERS (tax_filer = 1.0)")

    start_time = time.time()
    print("\nRunning reform steady state...")
    runner(p_reform, time_path=False, client=client)
    print(f"Reform run time: {time.time() - start_time:.1f} seconds")

    # Load reform results
    reform_ss = safe_read_pickle(os.path.join(reform_dir, "SS", "SS_vars.pkl"))
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )

    print("\n" + "=" * 70)
    print("RESULTS: ECONOMIC EFFECTS OF REQUIRING j=0 TO FILE")
    print("=" * 70)

    # Create macro results table using OG-Core's built-in function
    macro_results = ot.macro_table(
        baseline_ss,
        baseline_params,
        reform_tpi=reform_ss,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=1,
        include_SS=True,
        include_overall=False,
        start_year=baseline_spec["start_year"],
    )

    print("\nMacroeconomic Variables (% change from baseline):")
    print(macro_results.to_string())

    # Calculate tax revenue change
    base_revenue = baseline_ss["total_tax_revenue"]
    reform_revenue = reform_ss["total_tax_revenue"]
    if isinstance(base_revenue, np.ndarray):
        base_revenue = (
            base_revenue.item() if base_revenue.size == 1 else base_revenue[-1]
        )
    if isinstance(reform_revenue, np.ndarray):
        reform_revenue = (
            reform_revenue.item()
            if reform_revenue.size == 1
            else reform_revenue[-1]
        )

    revenue_pct_change = ((reform_revenue - base_revenue) / base_revenue) * 100
    print(f"\nTotal Tax Revenue: {revenue_pct_change:+.2f}%")

    # Analyze household-level effects for j=0
    print("\n" + "-" * 70)
    print("HOUSEHOLD-LEVEL EFFECTS: Income Group j=0")
    print("-" * 70)

    if "nssmat" in baseline_ss and "nssmat" in reform_ss:
        # Average labor supply for j=0
        base_labor = np.mean(baseline_ss["nssmat"][:, 0])
        reform_labor = np.mean(reform_ss["nssmat"][:, 0])
        labor_pct_change = ((reform_labor - base_labor) / base_labor) * 100

        print(f"\nAverage labor supply (j=0):")
        print(f"  Baseline (non-filer): {base_labor:.4f}")
        print(f"  Reform (filer):       {reform_labor:.4f}")
        print(f"  Change:               {labor_pct_change:+.2f}%")

    if "cssmat" in baseline_ss and "cssmat" in reform_ss:
        # Average consumption for j=0
        base_cons = np.mean(baseline_ss["cssmat"][:, 0])
        reform_cons = np.mean(reform_ss["cssmat"][:, 0])
        cons_pct_change = ((reform_cons - base_cons) / base_cons) * 100

        print(f"\nAverage consumption (j=0):")
        print(f"  Baseline (non-filer): {base_cons:.4f}")
        print(f"  Reform (filer):       {reform_cons:.4f}")
        print(f"  Change:               {cons_pct_change:+.2f}%")

    if "bssmat" in baseline_ss and "bssmat" in reform_ss:
        # Average savings for j=0
        base_savings = np.mean(baseline_ss["bssmat"][:, 0])
        reform_savings = np.mean(reform_ss["bssmat"][:, 0])
        savings_pct_change = (
            (reform_savings - base_savings) / base_savings
        ) * 100

        print(f"\nAverage savings (j=0):")
        print(f"  Baseline (non-filer): {base_savings:.4f}")
        print(f"  Reform (filer):       {reform_savings:.4f}")
        print(f"  Change:               {savings_pct_change:+.2f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        """
When the lowest income group transitions from non-filer to filer status:

1. TAX REVENUE INCREASES: The government collects income taxes from j=0,
   who previously paid only payroll taxes.

2. LABOR SUPPLY DECREASES: Group j=0 now faces positive marginal tax rates,
   creating a substitution effect that reduces labor supply.

3. SAVINGS DECREASE: Lower after-tax returns reduce savings incentives for
   j=0, affecting the capital stock.

4. GDP FALLS: The combination of lower labor supply and capital stock
   reduces aggregate output through general equilibrium effects.

5. INTEREST RATE RISES: Lower capital stock increases the marginal product
   of capital, raising the equilibrium interest rate.

This demonstrates that filing thresholds (which create non-filer groups)
can have significant efficiency effects by reducing tax distortions for
low-income households.
"""
    )

    print("=" * 70)
    print(f"Total run time: {time.time() - run_start_time:.1f} seconds")
    print(f"\nResults saved to: {save_dir}")
    print("=" * 70)

    # Save macro results to CSV
    macro_results.to_csv(os.path.join(save_dir, "nonfiler_macro_results.csv"))
    print(
        f"\nMacro results: {os.path.join(save_dir, 'nonfiler_macro_results.csv')}"
    )

    client.close()


if __name__ == "__main__":
    main()
