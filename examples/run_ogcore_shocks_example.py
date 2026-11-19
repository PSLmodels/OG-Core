"""Run a five-state earnings-risk baseline and corporate tax reform.

From the repository root, run python examples/run_ogcore_shocks_example.py.
The example uses the compact 40-age, two-type calibration. Both policies
have the same earnings risk, so the comparison isolates the tax change.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from distributed import Client

from ogcore.constants import BASELINE_DIR, REFORM_DIR
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle


def main():
    examples = Path(__file__).resolve().parent
    output = examples / "OG-Core-Shocks-Example"
    with (examples.parent / "tests" / "testing_params.json").open() as f:
        settings = json.load(f)
    # Illustrative log earnings process per model period, discretized with
    # Rouwenhorst. HSV income taxes use the same schedule in both runs.
    settings.update(
        household_solver="egm",
        nz=5,
        z_persistence=0.9,
        z_innovation_sd=0.15,
        tax_func_type="HSV",
        etr_params=[[[2.5, 0.1]]],
        mtrx_params=[[[2.5, 0.1]]],
        mtry_params=[[[2.5, 0.1]]],
    )

    workers = min(7, max(1, (os.cpu_count() or 1) - 1))
    paths = []
    with Client(n_workers=workers, threads_per_worker=1) as client:
        for directory, rate in ((BASELINE_DIR, 0.21), (REFORM_DIR, 0.35)):
            p = Specifications(
                baseline=directory == BASELINE_DIR,
                baseline_dir=str(output / BASELINE_DIR),
                output_base=str(output / directory),
                num_workers=workers,
            )
            p.update_specifications(dict(settings, cit_rate=[[rate]]))
            runner(p, time_path=True, client=client)
            tpi = safe_read_pickle(output / directory / "TPI" / "TPI_vars.pkl")
            error = np.abs(tpi["resource_constraint_error"]).max()
            print(
                f"{directory} maximum resource-constraint error = {error:.3g}"
            )
            paths.append(tpi)

    baseline, reform = paths
    comparison = pd.DataFrame(
        {
            f"{var}_pct_change": 100 * (reform[var] / baseline[var] - 1)
            for var in ("Y", "C", "K", "L")
        }
    )
    # These Ginis include the dispersion within each age and income type.
    for label, path in (("baseline", baseline), ("reform", reform)):
        comparison[f"{label}_wealth_gini"] = path["inequality"]["b_s"]["gini"]
    comparison.to_csv(output / "example_output.csv", index_label="period")
    print(comparison.head(10).to_string())


if __name__ == "__main__":
    main()
