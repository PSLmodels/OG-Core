"""
Side-by-side comparison: a baseline run with ``use_sparse_FOC_jac`` off
(default) and on. Reports the wall-time speedup, diffs the converged
steady-state and TPI paths, prints the resource-constraint residual, and
issues a NO DRIFT / DRIFT DETECTED verdict against a 0.1% threshold.

With no arguments, runs OG-Core's standard example baseline (the same
configuration as ``run_ogcore_example.py``). With a country package name
(e.g. ``ogphl``, ``ogzaf``, ``ogidn``, ``ogeth``) as a single argument,
runs that country's packaged baseline twice. The country package must be
importable in the active environment; outputs land in the current working
directory.

The reform leg is skipped; this is about solver speed and correctness on
a single run.

Run from the repo root:

    python examples/run_sparse_FOC_jac_compare.py          # OG-Core
    python examples/run_sparse_FOC_jac_compare.py ogphl    # PHL
"""

# import modules
import importlib
import json
import multiprocessing
import os
import sys
import time
from importlib.resources import files

import numpy as np
from distributed import Client

from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle


# Default config for OG-Core mode (no country arg). Matches
# run_ogcore_example.py.
_alpha_T = np.zeros(50)
_alpha_T[0:2] = 0.09
_alpha_T[2:10] = 0.09 + 0.01
_alpha_T[10:40] = 0.09 - 0.01
_alpha_T[40:] = 0.09
_alpha_G = np.zeros(7)
_alpha_G[0:3] = 0.05 - 0.01
_alpha_G[3:6] = 0.05 - 0.005
_alpha_G[6:] = 0.05
OGCORE_SPEC = {
    "frisch": 0.41,
    "start_year": 2021,
    "cit_rate": [[0.21]],
    "debt_ratio_ss": 1.0,
    "alpha_T": _alpha_T.tolist(),
    "alpha_G": _alpha_G.tolist(),
    "initial_guess_r_SS": 0.04,
}

KEY_AGGREGATES = (
    "Y",
    "C",
    "K",
    "L",
    "B",
    "I_total",
    "r",
    "w",
    "r_p",
    "r_gov",
    "TR",
    "total_tax_revenue",
    "D",
    "BQ",
)

# "No drift" threshold: aggregate differences within 0.1% are economically
# indistinguishable from the model's own convergence noise.
NO_DRIFT_THRESHOLD = 1e-3


def _max_rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


def _diff_dict(d_dense, d_sparse, var_list=None):
    """Return [(var, rel_diff), ...] sorted by rel_diff descending."""
    keys = (
        var_list
        if var_list is not None
        else sorted(set(d_dense) & set(d_sparse))
    )
    out = []
    for var in keys:
        if var in d_dense and var in d_sparse:
            try:
                rel = _max_rel_diff(d_dense[var], d_sparse[var])
            except (TypeError, ValueError):
                continue
            if rel == rel:
                out.append((var, rel))
    out.sort(key=lambda x: -x[1])
    return out


def _load_country_defaults(pkg):
    """Load <pkg>_default_parameters.json, with a 2-D shim for older country
    calibrations whose replacement_rate_adjust is still 1-D."""
    with files(pkg).joinpath(f"{pkg}_default_parameters.json").open("r") as f:
        defaults = json.load(f)
    rra = defaults.get("replacement_rate_adjust")
    if isinstance(rra, list) and rra and not isinstance(rra[0], list):
        defaults["replacement_rate_adjust"] = [rra]
    return defaults


def _apply_country_calibration(pkg, p):
    """Try the country's offline Calibration; quietly skip on error."""
    try:
        Cal = importlib.import_module(pkg + ".calibrate").Calibration
        try:
            c = Cal(p, update_from_api=False)
        except TypeError:
            c = Cal(p)
        p.update_specifications(c.get_dict())
    except Exception as e:
        print(f"  (calibration skipped: {type(e).__name__}: {str(e)[:80]})")


def _run_one(label, country_pkg, out_dir, num_workers, client, sparse_jac):
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=out_dir,
        output_base=out_dir,
    )
    if country_pkg is None:
        p.update_specifications(OGCORE_SPEC)
    else:
        p.update_specifications(_load_country_defaults(country_pkg))
        _apply_country_calibration(country_pkg, p)
    p.update_specifications({"use_sparse_FOC_jac": bool(sparse_jac)})
    print(f"\n[{label}] use_sparse_FOC_jac = {p.use_sparse_FOC_jac}")
    start = time.time()
    runner(p, time_path=True, client=client)
    wall = time.time() - start
    print(f"[{label}] wall time = {wall:.2f} s")
    ss = safe_read_pickle(os.path.join(out_dir, "SS", "SS_vars.pkl"))
    tpi = safe_read_pickle(os.path.join(out_dir, "TPI", "TPI_vars.pkl"))
    return wall, ss, tpi


def main(country_pkg=None):
    num_workers = min(multiprocessing.cpu_count(), 7)
    label = country_pkg if country_pkg else "ogcore (standard example)"
    print(f"Workers: {num_workers}  |  model: {label}")
    client = Client(n_workers=num_workers, threads_per_worker=1)

    # Outputs land in the current working directory so they're easy to find
    # regardless of where this script file lives.
    root = os.path.join(
        os.getcwd(),
        "sparse-FOC-jac-compare",
        country_pkg if country_pkg else "ogcore",
    )
    dense_dir = os.path.join(root, "dense")
    sparse_dir = os.path.join(root, "sparse")

    t_dense, ss_dense, tpi_dense = _run_one(
        f"{label} DENSE (default)",
        country_pkg,
        dense_dir,
        num_workers,
        client,
        False,
    )
    t_sparse, ss_sparse, tpi_sparse = _run_one(
        f"{label} SPARSE (use_sparse_FOC_jac=True)",
        country_pkg,
        sparse_dir,
        num_workers,
        client,
        True,
    )

    tpi_diffs = _diff_dict(tpi_dense, tpi_sparse, KEY_AGGREGATES)
    ss_diffs = _diff_dict(ss_dense, ss_sparse)

    tpi_worst_var, tpi_worst = tpi_diffs[0] if tpi_diffs else ("n/a", 0.0)
    ss_worst_var, ss_worst = ss_diffs[0] if ss_diffs else ("n/a", 0.0)
    worst = max(tpi_worst, ss_worst)

    rc_d = float(
        np.max(np.abs(tpi_dense.get("resource_constraint_error", np.zeros(1))))
    )
    rc_s = float(
        np.max(
            np.abs(tpi_sparse.get("resource_constraint_error", np.zeros(1)))
        )
    )

    speedup = t_dense / t_sparse if t_sparse > 0 else float("inf")
    bar = "=" * 64
    print()
    print(bar)
    print(f"  MODEL:  {label}")
    print("  SPEED")
    print(f"    dense  :  {t_dense:7.2f} s")
    print(f"    sparse :  {t_sparse:7.2f} s   ->   {speedup:.2f}x faster")
    print()
    print("  DRIFT  (max relative difference, sparse vs dense)")
    print(
        f"    TPI worst:  {tpi_worst_var:22s} "
        f"{tpi_worst * 100:9.4f}%   ({tpi_worst:.2e})"
    )
    print(
        f"    SS  worst:  {ss_worst_var:22s} "
        f"{ss_worst * 100:9.4f}%   ({ss_worst:.2e})"
    )
    if tpi_diffs:
        print()
        print("    All TPI aggregates, sorted by drift:")
        for var, rel in tpi_diffs:
            print(f"      {var:22s} {rel * 100:9.4f}%   ({rel:.2e})")
    print()
    print("  ACCURACY FLOOR  (resource-constraint residual)")
    print(f"    dense  :  {rc_d:.2e}")
    print(f"    sparse :  {rc_s:.2e}")
    print()
    threshold_pct = NO_DRIFT_THRESHOLD * 100
    if worst <= NO_DRIFT_THRESHOLD:
        print(
            f"  RESULT:  NO DRIFT  "
            f"(worst {worst * 100:.4f}%  <=  {threshold_pct:g}% threshold)"
        )
    else:
        print(
            f"  RESULT:  DRIFT DETECTED  "
            f"(worst {worst * 100:.4f}%  >   {threshold_pct:g}% threshold) "
            f"-- investigate"
        )
    print(bar)

    client.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
