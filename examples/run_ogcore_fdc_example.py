"""
Example script for a pension reform that introduces funded defined
contribution (FDC) accounts in OG-Core.

The baseline has a pay-as-you-go notional defined contribution (NDC)
system with a notional contribution rate of 16 percent. The reform is a
"carve-out" for younger workers: households aged 40 or younger in the
start year have 4 of their 16 percentage points paid into funded
individual accounts instead of their notional accounts (as in Sweden's
premium pension). Older households keep the baseline system. The
change is entered by birth cohort through ``pension_regimes``.

The funded accounts are invested at the household portfolio return,
add to the capital stock, and pay an actuarially fair life annuity at
the retirement age. Contributions and benefits are not part of the
government budget.
"""

# import modules
import multiprocessing
from distributed import Client
import time
import os
from ogcore import output_tables as ot
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.constants import REFORM_DIR, BASELINE_DIR
from ogcore.utils import safe_read_pickle
from ogcore.config import set_logging_level

set_logging_level(verbose=True)


def main():
    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 7)
    print("Number of workers = ", num_workers)
    client = Client(n_workers=num_workers, threads_per_worker=1)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(CUR_DIR, "OG-Core-FDC-Example")
    base_dir = os.path.join(save_dir, BASELINE_DIR)
    reform_dir = os.path.join(save_dir, REFORM_DIR)

    run_start_time = time.time()

    START_YEAR = 2025
    # Baseline: a notional defined contribution system
    og_spec = {
        "start_year": START_YEAR,
        "pension_system": "Notional Defined Contribution",
        "tau_p": 0.16,
        "initial_guess_r_SS": 0.04,
    }
    # Optional: cap the labor income credited to the notional accounts
    # and, in the reform, paid into the funded accounts at 1.5 times
    # average labor income in the first period
    # (avg_labor_inc_t1_model_units). The two pillars share this cap, as
    # in Sweden. Entering the factor twice holds the cap fixed in model
    # units, so it grows with wages; a single value is fixed in real
    # terms.
    # og_spec.update(
    #     {"pension_earn_cap": True, "dc_earn_cap_factor": [1.5, 1.5]}
    # )

    """
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    """
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    p.update_specifications(og_spec)
    start_time = time.time()
    runner(p, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    """
    ------------------------------------------------------------------------
    Run reform policy: carve funded accounts out of the notional rate for
    cohorts aged 40 or younger in the start year
    ------------------------------------------------------------------------
    """
    reform_spec = dict(og_spec)
    reform_spec["pension_regimes"] = [
        {
            "year": START_YEAR,
            "max_age_at_start": 40,
            "tau_p": 0.12,
            "fdc_tau": 0.04,
        }
    ]
    p2 = Specifications(
        baseline=False,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=reform_dir,
    )
    p2.update_specifications(reform_spec)
    start_time = time.time()
    runner(p2, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    # Percentage changes in macro aggregates and prices
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w", "B"],
        output_type="pct_diff",
        num_years=10,
        start_year=START_YEAR,
    )
    # The FDC fund's assets in the reform, in model units
    fund = ot.macro_table(
        reform_tpi,
        reform_params,
        var_list=["F"],
        output_type="levels",
        num_years=10,
        start_year=START_YEAR,
    )

    print("total time was ", (time.time() - run_start_time))
    print("Percentage changes in aggregates:", ans)
    print("FDC fund assets in the reform:", fund)
    ans.to_csv(os.path.join(save_dir, "fdc_example_output.csv"))
    client.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
