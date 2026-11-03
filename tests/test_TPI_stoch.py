"""
Tests of the transition path with the stochastic-income household
solver (household_solver = 'egm'). The household block along the time
path is checked against the steady state (at steady-state prices, from
the steady-state distribution, it stays there) and against OG-Core's
pension and initial-condition formulas; with one shock state, the
transition path matches OG-Core's default solver.

The tests use OG-Core's test calibration (S = 40, J = 2, T = 120) with
HSV income taxes; converged transition paths are marked local.
"""

import os
import pickle

import numpy as np
import pytest
from distributed import Client, LocalCluster
from ogcore import SS, TPI, household_stoch, pensions, utils
from ogcore import aggregates as aggr
from stoch_test_utils import egm_specs, hsv_calibration

# prices and aggregates at which households are solved outside general
# equilibrium
R, W, FACTOR = 0.05, 1.0, 100000.0
BQ = np.array([0.02, 0.03])
TR = 0.03
# variables whose reform effects (percent changes) are compared, as in
# examples/run_ogcore_example.py, and the tolerance in percentage points
MACRO_VARS = ["Y", "C", "K", "L", "r", "w"]
PP_TOL = 0.01


def constant_paths(p, sol, r=None):
    """
    Per-household paths at the prices and aggregates of the steady-state
    household solution sol (and the interest rate path r if given).
    """
    TS = p.T + p.S
    r = np.full(TS, R) if r is None else r
    n_eff = household_stoch.mean_effective_labor(
        sol.dist, sol.n_policy, sol.z_grid
    )
    return household_stoch.household_inputs_tpi(
        p,
        r,
        np.full(TS, W),
        np.ones((TS, p.M)),
        np.tile(BQ, (TS, 1)),
        np.full(TS, float(aggr.get_RM(1.0, p, "SS"))),
        np.full(TS, TR),
        FACTOR,
        p.ubi_nom_array / FACTOR,
        sol.theta,
        np.tile(n_eff, (TS, 1, 1)),
    )


def conditional(dist):
    """Distribution within each (age, type) cell."""
    return dist / dist.sum(axis=(2, 3), keepdims=True)


@pytest.fixture(scope="module")
def stationary():
    """
    The test calibration with constant demographics, three shock states
    and a wealth tax, and its steady-state household solution at fixed
    prices.
    """
    p = hsv_calibration()
    p.update_specifications(
        {"constant_demographics": True, "p_wealth": [0.02]}
    )
    p = egm_specs(p, nz=3, sigma=0.2)
    sol = household_stoch.solve_all_households_ss(
        R, W, np.ones(p.M), BQ, TR, FACTOR, p, Y=1.0
    )
    return p, sol


def test_household_block_stays_at_steady_state(stationary):
    """
    At steady-state prices and policy, from the steady-state
    distribution, the household block along the time path gives the
    steady state's means, tax liabilities and rates, Euler errors and
    inequality in every period.
    """
    p, sol = stationary
    tp = household_stoch.solve_households_tpi(
        p,
        constant_paths(p, sol),
        conditional(sol.dist),
        sol.b_grid,
        final=True,
    )
    shape = (p.T, p.S, p.J)
    expected = {
        "b_s": sol.cell_means(sol.b_s()),
        "b_sp1": sol.cell_means(sol.b_policy),
        "n": sol.cell_means(sol.n_policy),
        "n_eff": household_stoch.mean_effective_labor(
            sol.dist, sol.n_policy, sol.z_grid
        ),
        "c": sol.cell_means(sol.c_policy),
        **household_stoch.tax_cell_means(sol, R, W, FACTOR, p),
    }
    assert (expected["w_tax_liab"] > 0).any()
    for k, v in expected.items():
        np.testing.assert_allclose(
            getattr(tp, k),
            np.broadcast_to(v, shape),
            rtol=1e-9,
            atol=1e-12,
            err_msg=k,
        )
    rates = household_stoch.tax_rate_cell_means(sol, R, W, FACTOR, p)
    for k, v in rates.items():
        np.testing.assert_allclose(
            tp.rates[k], np.broadcast_to(v, shape), atol=1e-12, err_msg=k
        )
    euler = household_stoch.euler_errors(sol)["max_with_mass_by_cell"]
    np.testing.assert_allclose(
        tp.euler_savings[:, :-1], np.broadcast_to(euler, (p.T,) + euler.shape)
    )
    assert (tp.euler_savings[:, -1] == 0).all()
    assert not tp.failed.any() and not tp.above_grid.any()
    ineq = household_stoch.inequality_tpi(tp)
    ineq_ss = household_stoch.inequality_ss(sol, R, W, p)
    for k in household_stoch.TPI_LOG_MOMENTS:
        np.testing.assert_allclose(
            ineq[k]["var_of_logs"], ineq_ss[k]["var_of_logs"], rtol=1e-9
        )
    for k, v in ineq["b_s"].items():
        np.testing.assert_allclose(v, v[0], rtol=1e-9, err_msg=k)
    assert 0 < ineq["b_s"]["gini"][0] < 1


def test_cohort_inputs_follow_diagonal():
    """
    A cohort's inputs are the paths and policy along its diagonal: age s
    in period t0 + s, for the ages it lives on the time path.
    """
    p = egm_specs(hsv_calibration())
    TS, S, J = p.T + p.S, p.S, p.J
    t_idx = np.arange(TS)[:, None, None]
    s_idx = np.arange(S)[None, :, None]
    j_idx = np.arange(J)[None, None, :]
    # every (period, age, type) has its own value
    by_cell = (t_idx + 1000.0 * s_idx + 1e6 * j_idx) * np.ones((TS, S, J))
    paths = {k: by_cell + i for i, k in enumerate(["bq", "rm", "tr", "ubi"])}
    paths["pension"] = by_cell + 7
    paths.update(
        {
            "r": 0.04 + 1e-4 * np.arange(TS),
            "w": 1.0 + 1e-3 * np.arange(TS),
            "p_tilde": 1.0 + 1e-5 * np.arange(TS),
            "c_min_exp": 1e-6 * np.arange(TS),
            "factor": FACTOR,
        }
    )
    rng = np.random.default_rng(0)
    for name in (
        "tau_payroll",
        "income_tax_filer",
        "wealth_tax_filer",
        "labor_income_tax_noncompliance_rate",
        "capital_income_tax_noncompliance_rate",
    ):
        setattr(p, name, rng.uniform(0, 0.2, (TS, J)))
    p.etr_params = rng.uniform(0, 1, (TS, S, 2))
    p.chi_n = rng.uniform(1, 2, (TS, S))
    p.e = rng.uniform(0.5, 2, (p.T, S, J))
    p._e_long_cache = None
    p.rho = rng.uniform(0, 0.1, (TS, S, J))
    p.h_wealth = rng.uniform(0, 1, TS)
    p.tau_bq = rng.uniform(0, 0.1, TS)
    e_long = np.concatenate([p.e, np.tile(p.e[-1:], (S, 1, 1))])
    for t0 in (-(S - 1), -5, 0, 7, p.T - 1):
        inp, ages, times = household_stoch.lifecycle_inputs_tpi(p, paths, t0)
        np.testing.assert_array_equal(ages, np.arange(max(0, -t0), S))
        np.testing.assert_array_equal(times, t0 + ages)
        for k in ("r", "w", "p_tilde", "c_min_exp"):
            np.testing.assert_array_equal(getattr(inp, k), paths[k][times])
        for k in ("bq", "rm", "tr", "ubi", "pension"):
            np.testing.assert_array_equal(
                getattr(inp, k), paths[k][times, ages]
            )
        np.testing.assert_array_equal(inp.e, e_long[times, ages])
        np.testing.assert_array_equal(inp.rho, p.rho[times, ages])
        np.testing.assert_array_equal(inp.chi_n, p.chi_n[times, ages])
        np.testing.assert_array_equal(
            inp.etr_params, p.etr_params[times, ages]
        )
        np.testing.assert_array_equal(inp.h_wealth, p.h_wealth[times])
        np.testing.assert_array_equal(inp.tau_payroll, p.tau_payroll[times])
        np.testing.assert_array_equal(
            inp.labor_nc, p.labor_income_tax_noncompliance_rate[times]
        )
        np.testing.assert_array_equal(
            inp.bq_tax, p.tau_bq[times][:, None] * paths["bq"][times, ages]
        )


@pytest.mark.parametrize(
    "pension_system", ["US-Style Social Security", "Defined Benefits"]
)
def test_pension_tpi_matches_ogcore(pension_system):
    """
    The pension benefits households take as given are OG-Core's along
    the time path (``pensions.pension_amount`` over all periods), with
    replacement rates and their adjustment changing over time and, for
    defined benefits, labor before the path from n_preTP. (Effective
    labor units e are constant over time here: OG-Core's household
    problem takes them along each cohort's diagonal, as pension_tpi does,
    but its aggregate defined benefits use the steady-state profile.)
    """
    p = hsv_calibration()
    p.pension_system = pension_system
    TS = p.T + p.S
    rng = np.random.default_rng(1)
    r = rng.uniform(0.04, 0.06, TS)
    w = rng.uniform(0.9, 1.1, TS)
    n = rng.uniform(0.2, 0.5, (TS, p.S, p.J))
    theta = np.array([0.3, 0.4])
    p.replacement_rate_adjust = np.ones((TS, p.J))
    p.replacement_rate_adjust[10:30] = 1.2
    p.alpha_db = np.full((TS, p.J), 0.01)
    p.alpha_db[20:40] = 0.015
    p.n_preTP = 0.9 * n[0]
    pension = household_stoch.pension_tpi(p, r, w, FACTOR, theta, n)
    expected = pensions.pension_amount(
        r[: p.T],
        w[: p.T],
        n[: p.T],
        1,
        theta,
        0,
        None,
        False,
        "TPI",
        p.e,
        FACTOR,
        p,
    )
    assert expected.max() > 0
    np.testing.assert_allclose(pension[: p.T], expected, rtol=1e-12)


def test_initial_distribution(stationary):
    """
    Households alive in period 0 have the steady state's distribution
    within their age and type, rescaled to the initial savings and moved
    onto the time path's asset grids with their means preserved.
    """
    p, sol = stationary
    ss_vars = {"dist": sol.dist, "b_grid": sol.b_grid, "z_grid": sol.z_grid}
    cond = conditional(sol.dist)
    b_sp1 = sol.cell_means(sol.b_policy)
    # the steady state's own savings and grids: unchanged
    init = household_stoch.initial_distribution_tpi(
        p, ss_vars, b_sp1, sol.b_grid
    )
    np.testing.assert_allclose(init[1:], cond[1:], atol=1e-12)
    newborn = np.zeros_like(init[0])
    newborn[:, 0, :] = p.z_dist0
    np.testing.assert_array_equal(init[0], newborn)
    # more wealth, on wider grids
    grid = sol.b_grid * 1.5
    init = household_stoch.initial_distribution_tpi(
        p, ss_vars, 1.1 * b_sp1, grid
    )
    np.testing.assert_allclose(init.sum(axis=(2, 3)), 1.0)
    means = (init * grid[None, :, :, None]).sum(axis=(2, 3))
    np.testing.assert_allclose(means[1:], 1.1 * b_sp1[:-1], rtol=1e-10)
    np.testing.assert_allclose(init.sum(axis=2), cond.sum(axis=2), atol=1e-12)
    with pytest.raises(ValueError, match="egm"):
        household_stoch.initial_distribution_tpi(
            p, {"b_grid": sol.b_grid}, b_sp1, grid
        )


def test_dask_matches_serial(stationary, monkeypatch):
    """
    Cohorts solved on Dask workers give the same household block as in
    one process.
    """
    p, sol = stationary
    TS = p.T + p.S
    # a transition in the interest rate
    paths = constant_paths(p, sol, R + 0.01 * np.exp(-np.arange(TS) / 10))
    init = conditional(sol.dist)
    serial = TPI.solve_households_egm(p, paths, init, sol.b_grid, final=True)

    def no_fallback(*args, **kwargs):
        raise AssertionError("fell back to solving in this process")

    # with a client, the cohorts must be solved on the workers
    monkeypatch.setattr(household_stoch, "solve_households_tpi", no_fallback)
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    try:
        par = TPI.solve_households_egm(
            p, paths, init, sol.b_grid, client, final=True
        )
    finally:
        client.close()
        cluster.close()
    for k in household_stoch.TPI_CELL_MEANS:
        np.testing.assert_allclose(
            getattr(par, k), getattr(serial, k), atol=1e-14, err_msg=k
        )
    for k in serial.rates:
        np.testing.assert_allclose(par.rates[k], serial.rates[k], atol=1e-14)
    np.testing.assert_allclose(par.euler_savings, serial.euler_savings)
    assert par.diagnostics.n_points == serial.diagnostics.n_points
    # the transition moves away from the steady state
    assert np.abs(serial.b_sp1[0] / serial.b_sp1[-1] - 1).max() > 1e-3


def run_model(p):
    """Solve and save the steady state, then solve the time path."""
    ss = SS.run_SS(p)
    ss_dir = os.path.join(p.output_base, "SS")
    utils.mkdirs(ss_dir)
    with open(os.path.join(ss_dir, "SS_vars.pkl"), "wb") as f:
        pickle.dump(ss, f)
    return ss, TPI.run_TPI(p)


def model_specs(tmp_path, baseline=True, solver="egm", nz=1, **updates):
    """The HSV test calibration, saving output under tmp_path."""
    p = hsv_calibration()
    p.baseline = baseline
    p.baseline_dir = os.path.join(tmp_path, "baseline")
    p.output_base = os.path.join(
        tmp_path, "baseline" if baseline else "reform"
    )
    p.update_specifications(updates)
    if solver == "egm":
        p = egm_specs(p, nz=nz)
    return p


def test_run_TPI_egm_outputs(tmp_path, monkeypatch):
    """
    A few iterations of run_TPI with shocks give the outputs by period,
    age and type as means over the distribution, the asset grids, and
    inequality by period.
    """
    monkeypatch.setattr(TPI, "ENFORCE_SOLUTION_CHECKS", False)
    p = model_specs(tmp_path, nz=2, maxiter=2)
    ss, tpi = run_model(p)
    shape = (p.T, p.S, p.J)
    for k in ("b_sp1", "b_s", "n", "c", "etr", "mtrx", "euler_savings"):
        assert tpi[k].shape == shape, k
    np.testing.assert_array_equal(tpi["b_grid"], ss["b_grid"])
    np.testing.assert_allclose(tpi["z_grid"], p.z_grid)
    for k in ("b_s", "c", "labor_income", "before_tax_income"):
        for v in tpi["inequality"][k].values():
            assert v.shape == (p.T,) and np.isfinite(v).all()
    # effective labor L is not computed from mean hours
    assert np.abs(tpi["L"] / aggr.get_L(tpi["n"], p, "TPI") - 1).max() > 1e-3
    assert np.abs(tpi["euler_savings"]).max() < 1e-3


@pytest.fixture(scope="module")
def one_state_runs(tmp_path_factory):
    """
    Baseline and reform (CIT rate 21% to 35%) transition paths of the
    test calibration with OG-Core's solver and with the stochastic
    solver with one shock state.
    """
    out = {}
    for solver in ("root", "egm"):
        tmp = tmp_path_factory.mktemp(solver)
        for baseline, cit in ((True, 0.21), (False, 0.35)):
            p = model_specs(tmp, baseline, solver, cit_rate=[[cit]])
            out[solver, baseline] = (p,) + run_model(p)
    return out


def pct_changes(runs, solver):
    base, reform = runs[solver, True][2], runs[solver, False][2]
    return {k: 100 * (reform[k] / base[k] - 1) for k in MACRO_VARS}


@pytest.mark.local
def test_one_state_tpi_matches_default(one_state_runs):
    """
    With one shock state, the baseline transition path matches OG-Core's
    within the steady-state gap of Young's method, and the reform's
    effects (percent changes from the baseline) match within 0.01
    percentage points in every period.
    """
    base_root = one_state_runs["root", True][2]
    base_egm = one_state_runs["egm", True][2]
    for k in MACRO_VARS + ["B", "BQ", "TR"]:
        gap = np.abs(base_egm[k] / base_root[k] - 1).max()
        assert gap < 1e-3, (k, gap)
    root, egm = (
        pct_changes(one_state_runs, "root"),
        pct_changes(one_state_runs, "egm"),
    )
    for k in MACRO_VARS:
        diff = np.abs(egm[k] - root[k]).max()
        print(k, f"max |difference| = {diff:.1e} pp")
        assert diff < PP_TOL, (k, diff)


@pytest.mark.local
def test_one_state_tpi_reaches_steady_state(one_state_runs):
    """
    Baseline and reform paths satisfy the resource constraint and end at
    their steady states.
    """
    for solver in ("root", "egm"):
        for baseline in (True, False):
            p, ss, tpi = one_state_runs[solver, baseline]
            assert np.abs(tpi["resource_constraint_error"]).max() < p.RC_TPI
            for k in ("Y", "K", "r", "w"):
                assert abs(tpi[k][-1] / ss[k] - 1) < 1e-4, (solver, k)


@pytest.mark.local
def test_shocks_tpi(tmp_path, one_state_runs):
    """
    With five shock states, the baseline and reform paths converge,
    satisfy the resource constraint and end at their steady states, and
    the reform moves the economy in the same direction as with one
    state.
    """
    runs = {}
    for baseline, cit in ((True, 0.21), (False, 0.35)):
        p = model_specs(tmp_path, baseline, nz=5, cit_rate=[[cit]])
        p_, ss, tpi = (p,) + run_model(p)
        runs["egm", baseline] = (p_, ss, tpi)
        assert np.abs(tpi["resource_constraint_error"]).max() < p.RC_TPI
        for k in ("Y", "K", "r", "w"):
            assert abs(tpi[k][-1] / ss[k] - 1) < 1e-4, k
        assert tpi["inequality"]["b_s"]["gini"].min() > 0
    shocks = pct_changes(runs, "egm")
    one = pct_changes(one_state_runs, "egm")
    for k in MACRO_VARS:
        assert (np.sign(shocks[k][:10]) == np.sign(one[k][:10])).all(), k


@pytest.mark.local
def test_defined_benefits_tpi(tmp_path):
    """
    With defined benefit pensions, which depend on each cohort's
    earnings history, the one-state transition path matches OG-Core's.
    """
    updates = {"pension_system": "Defined Benefits", "alpha_db": [[0.01]]}
    _, root = run_model(
        model_specs(tmp_path / "root", solver="root", **updates)
    )
    _, egm = run_model(model_specs(tmp_path / "egm", **updates))
    assert egm["agg_pension_outlays"].min() > 0
    for k in MACRO_VARS + ["agg_pension_outlays"]:
        gap = np.abs(egm[k] / root[k] - 1).max()
        assert gap < 1e-3, (k, gap)
