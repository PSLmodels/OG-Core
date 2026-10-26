"""
Tests of the steady state with the stochastic-income household solver
(household_solver = 'egm'): with one shock state it reproduces the
steady state of OG-Core's default solver, and with shocks it converges
and shows precautionary saving.

The tests use OG-Core's test calibration (S = 40, J = 2) with nonlinear
(HSV) income taxes, so that taxes averaged over the distribution differ
from taxes at mean income; tests marked local use the default
calibration. The household block itself is tested in
test_household_stoch*.py.
"""

import copy

import numpy as np
import pytest
from ogcore import SS, TPI, household_stoch, tax
from ogcore import aggregates as aggr
from ogcore.parameters import Specifications
from stoch_test_utils import ogcore_test_calibration, set_ar1_shocks

# General equilibrium variables the one-state steady state must
# reproduce, and their relative tolerance. Young's method spreads each
# household's savings over two grid nodes, which puts aggregate savings
# a few hundredths of a percent above OG-Core's.
GE_VARS = ["r", "w", "Y", "K", "L", "BQ", "TR", "factor"]
OTHER_AGGREGATES = [
    "B",
    "C",
    "I",
    "D",
    "r_p",
    "total_tax_revenue",
    "agg_pension_outlays",
    "theta",
]
REL_TOL = 1e-3
# HSV tax functions, ETR = 1 - 2.5 income^-0.1 (about 15% to 20% in
# the test calibration)
HSV_PARAMS = [[[2.5, 0.1]]]


def hsv_calibration():
    """OG-Core's test calibration with HSV income taxes."""
    p = ogcore_test_calibration()
    p.update_specifications(
        {
            "tax_func_type": "HSV",
            "etr_params": HSV_PARAMS,
            "mtrx_params": HSV_PARAMS,
            "mtry_params": HSV_PARAMS,
        }
    )
    return p


def egm_specs(p, nz=1, rho=0.9, sigma=0.15):
    """p with the stochastic household solver and nz shock states."""
    p = copy.deepcopy(p)
    if nz == 1:
        p.update_specifications({"household_solver": "egm"})
        return p
    return set_ar1_shocks(p, nz, rho, sigma)


@pytest.fixture(scope="module")
def ss_root():
    """Steady state of the test calibration with OG-Core's solver."""
    return SS.run_SS(hsv_calibration())


@pytest.fixture(scope="module")
def ss_one_state():
    """Steady state with the stochastic solver and one shock state."""
    p = egm_specs(hsv_calibration())
    return p, SS.run_SS(p)


@pytest.fixture(scope="module")
def ss_shocks():
    """Steady state with five shock states."""
    p = egm_specs(hsv_calibration(), nz=5)
    return p, SS.run_SS(p)


def household_solution(p, ss, b_grid=None):
    """
    inner_loop at the prices of the steady state ss, on its asset grids
    or b_grid.
    """
    outer_loop_vars = (
        None,
        None,
        ss["r_p"],
        ss["r"],
        ss["w"],
        ss["p_m"],
        ss["Y"],
        ss["BQ"],
        ss["TR"],
        None,
        ss["factor"],
    )
    if b_grid is None:
        b_grid = ss["b_grid"]
    return SS.inner_loop(outer_loop_vars, p, None, b_grid=b_grid)


def assert_close(test, expected, names, rtol=REL_TOL):
    gaps = {
        k: float(
            np.max(np.abs(np.asarray(test[k]) / np.asarray(expected[k]) - 1))
        )
        for k in names
    }
    print("relative differences:", {k: f"{v:.1e}" for k, v in gaps.items()})
    bad = {k: v for k, v in gaps.items() if not v < rtol}
    assert not bad, f"relative differences above {rtol}: {bad}"


def test_one_state_reproduces_default_ss(ss_root, ss_one_state):
    """
    With one shock state, the steady state from the stochastic solver
    matches OG-Core's (pensions, transfers and bequests on).
    """
    p, ss = ss_one_state
    assert p.nz == 1 and p.household_solver == "egm"
    assert ss_root["agg_pension_outlays"] > 0
    assert_close(ss, ss_root, GE_VARS)


def test_one_state_reproduces_default_aggregates(ss_root, ss_one_state):
    """
    Other aggregates and the replacement rates match as well, and
    government spending, a small residual of the budget, matches as a
    share of GDP.
    """
    ss = ss_one_state[1]
    assert_close(ss, ss_root, OTHER_AGGREGATES)
    assert abs(ss["G"] - ss_root["G"]) / ss_root["Y"] < REL_TOL


def test_one_state_reproduces_default_profiles(ss_root, ss_one_state):
    """Savings, labor and consumption by age and type match OG-Core's."""
    ss = ss_one_state[1]
    for k in ("b_sp1", "n", "c"):
        np.testing.assert_allclose(ss[k], ss_root[k], rtol=0, atol=2e-3)
    np.testing.assert_allclose(ss["etr"], ss_root["etr"], rtol=0, atol=1e-3)


@pytest.mark.parametrize("fixture", ["ss_one_state", "ss_shocks"])
def test_resource_constraint(fixture, request):
    """The steady state satisfies the resource constraint."""
    p, ss = request.getfixturevalue(fixture)
    assert np.abs(ss["resource_constraint_error"]).max() < p.RC_SS


def test_shocks_distribution(ss_shocks):
    """
    With shocks, the distribution sums to the population by age and
    type, nobody reaches the top of the asset grids, and the outputs by
    age and type are means over the distribution.
    """
    p, ss = ss_shocks
    nb = ss["b_grid"].shape[1]
    shape = (p.S, p.J, nb, p.nz)
    for k in ("dist", "b_policy", "c_policy", "n_policy"):
        assert ss[k].shape == shape
    assert ss["b_grid"].shape == (p.J, nb)
    np.testing.assert_allclose(ss["z_grid"], p.z_grid)
    dist = ss["dist"]
    np.testing.assert_allclose(dist.sum(axis=(2, 3)), p.omega_SS, atol=1e-12)
    assert dist[:, :, -1, :].sum() < 1e-10

    def cell_means(x):
        return (dist * x).sum(axis=(2, 3)) / dist.sum(axis=(2, 3))

    b_s = np.broadcast_to(ss["b_grid"][None, :, :, None], shape)
    n_eff = cell_means(ss["n_policy"] * p.z_grid)
    np.testing.assert_allclose(ss["b_sp1"], cell_means(ss["b_policy"]))
    np.testing.assert_allclose(ss["b_s"], cell_means(b_s))
    np.testing.assert_allclose(ss["n"], cell_means(ss["n_policy"]))
    np.testing.assert_allclose(ss["L"], aggr.get_L(n_eff, p, "SS"))
    np.testing.assert_allclose(
        ss["B"], aggr.get_B(ss["b_sp1"], p, "SS", False)
    )
    # consumption from the budget constraint at the reported prices
    np.testing.assert_allclose(ss["c"], cell_means(ss["c_policy"]), rtol=1e-5)
    # Euler errors where households are
    assert np.abs(ss["euler_savings"]).max() < 1e-4


def test_precautionary_saving(ss_one_state, ss_shocks):
    """
    Income risk raises saving: more capital and a lower interest rate
    than with one state, and more wealth and consumption inequality.
    """
    one, shocks = ss_one_state[1], ss_shocks[1]
    assert shocks["K"] > one["K"]
    assert shocks["r"] < one["r"]
    for v in ("b_sp1", "c", "labor_income"):
        assert shocks["inequality"][v]["gini"] > one["inequality"][v]["gini"]


def test_inner_loop_returns_household_solution(ss_one_state):
    """
    With the stochastic solver, inner_loop returns the household
    solution in place of the Euler errors and solves on the grids it is
    given.
    """
    p, ss = ss_one_state
    out = household_solution(p, ss)
    hh = out[0]
    assert isinstance(hh, household_stoch.HouseholdSolution)
    np.testing.assert_array_equal(hh.b_grid, ss["b_grid"])
    np.testing.assert_allclose(out[1], ss["b_sp1"], rtol=1e-5)
    np.testing.assert_allclose(out[2], ss["n"], rtol=1e-5)
    # at the steady state the implied prices are the ones given
    np.testing.assert_allclose(out[3], ss["r"], rtol=1e-5)
    np.testing.assert_allclose(out[6], ss["w"], rtol=1e-5)


def test_tax_cell_means_match_node_by_node(ss_shocks):
    """
    Tax liabilities and rates averaged over the distribution agree with
    OG-Core's tax functions evaluated one (asset, shock) node at a time.
    """
    p, ss = ss_shocks
    hh = household_solution(p, ss)[0]
    # a wealth tax, so that its liability is not zero
    p = copy.deepcopy(p)
    p.p_wealth = np.full_like(p.p_wealth, 0.02)
    r, w, factor = ss["r_p"], ss["w"], ss["factor"]
    liab = household_stoch.tax_cell_means(hh, r, w, factor, p)
    rates = household_stoch.tax_rate_cell_means(hh, r, w, factor, p)
    S, J, nb, nz = hh.dist.shape

    def by_type(params):
        return np.tile(np.asarray(params[-1])[:, None, :], (1, J, 1))

    etr_params = by_type(p.etr_params)
    e = p.e[-1]
    lnc = np.tile(p.labor_income_tax_noncompliance_rate[-1], (S, 1))
    knc = np.tile(p.capital_income_tax_noncompliance_rate[-1], (S, 1))
    filer = np.tile(p.income_tax_filer[-1], (S, 1))
    totals = {}
    for ib in range(nb):
        for iz in range(nz):
            b = np.tile(hh.b_grid[:, ib], (S, 1))
            n = hh.n_policy[:, :, ib, iz]
            ez = e * hh.z_grid[iz]
            mass = hh.dist[:, :, ib, iz]
            values = {
                "inc_pay_tax_liab": tax.income_tax_liab(
                    r, w, b, n, factor, 0, None, "SS", ez, etr_params, p
                ),
                "w_tax_liab": tax.wealth_tax_liab(r, b, 0, None, "SS", p),
                "etr": tax.ETR_income(
                    r, w, b, n, factor, ez, etr_params, lnc, knc, filer, p
                ),
            }
            for name, capital, nc in (
                ("mtrx", False, lnc),
                ("mtry", True, knc),
            ):
                values[name] = tax.MTR_income(
                    r,
                    w,
                    b,
                    n,
                    factor,
                    capital,
                    ez,
                    etr_params,
                    by_type(getattr(p, name + "_params")),
                    nc,
                    filer,
                    p,
                )
            for k, v in values.items():
                totals[k] = totals.get(k, 0.0) + mass * v
    assert (liab["w_tax_liab"] > 0).any()
    for k, v in list(liab.items()) + list(rates.items()):
        np.testing.assert_allclose(v, totals[k] / p.omega_SS, err_msg=k)
    # taxes are nonlinear in income: their means over the distribution
    # are not the taxes at mean income
    n_eff = hh.cell_means(hh.n_policy * hh.z_grid)
    at_means = tax.income_tax_liab(
        r, w, ss["b_s"], n_eff, factor, 0, None, "SS", e, etr_params, p
    )
    assert np.abs(liab["inc_pay_tax_liab"] / at_means - 1).max() > 1e-3


def test_liability_arguments():
    """
    tax.net_taxes and aggregates.revenue use the liabilities they are
    given: the ones they would compute themselves change nothing, and
    larger ones raise net taxes and revenue one for one.
    """
    p = hsv_calibration()
    S, J = p.S, p.J
    rng = np.random.default_rng(0)
    r, w, factor = 0.06, 1.1, 100000.0
    b = rng.uniform(0.0, 2.0, (S, J))
    n = rng.uniform(0.1, 0.5, (S, J))
    bq = rng.uniform(0.0, 0.05, (S, J))
    tr = np.full((S, J), 0.01)
    ubi = np.zeros((S, J))
    theta = np.full(J, 0.2)
    e = p.e[-1]
    etr_params = np.tile(np.asarray(p.etr_params[-1])[:, None, :], (1, J, 1))
    liab = {
        "inc_pay_tax_liab": tax.income_tax_liab(
            r, w, b, n, factor, 0, None, "SS", e, etr_params, p
        ),
        "w_tax_liab": tax.wealth_tax_liab(r, b, 0, None, "SS", p),
    }
    args = (r, w, b, n, bq, factor, tr, ubi, theta, None, None, False)
    args = args + ("SS", e, etr_params, p)
    more = {k: v + 0.01 for k, v in liab.items()}
    net_tax = tax.net_taxes(*args)
    np.testing.assert_array_equal(tax.net_taxes(*args, **liab), net_tax)
    np.testing.assert_allclose(tax.net_taxes(*args, **more), net_tax + 0.02)
    c = rng.uniform(0.1, 0.5, (p.I, S, J))
    args = (r, w, b, n, bq, c, np.array([1.0]), np.array([0.5]))
    args = args + (np.array([2.0]), np.array([1.0]), factor, ubi, theta)
    args = args + (etr_params, e, p, None, "SS")
    revenue = aggr.revenue(*args)
    for x, y in zip(aggr.revenue(*args, **liab), revenue):
        np.testing.assert_array_equal(x, y)
    # total, income and payroll, and wealth tax revenue (omega_SS sums
    # to one)
    revenue_more = aggr.revenue(*args, **more)
    for i, change in ((0, 0.02), (1, 0.01), (5, 0.01)):
        np.testing.assert_allclose(revenue_more[i], revenue[i] + change)


def test_refit_asset_grids(ss_shocks):
    """
    Grids that fit the distribution are kept; a grid whose top is far
    above the households' savings is refitted.
    """
    p, ss = ss_shocks
    hh = household_solution(p, ss)[0]
    b_grid, refit = household_stoch.refit_asset_grids(hh, p)
    assert not refit.any()
    np.testing.assert_array_equal(b_grid, ss["b_grid"])
    too_wide = ss["b_grid"].copy()
    too_wide[0] *= 3.0
    hh = household_solution(p, ss, too_wide)[0]
    b_grid, refit = household_stoch.refit_asset_grids(hh, p)
    assert refit.tolist() == [True] + [False] * (p.J - 1)
    np.testing.assert_array_equal(b_grid[1:], too_wide[1:])
    assert b_grid[0, -1] < 0.5 * too_wide[0, -1]
    # the refitted grid fits
    hh = household_solution(p, ss, b_grid)[0]
    assert not household_stoch.refit_asset_grids(hh, p)[1].any()


def test_SS_solver_keeps_grids(ss_one_state, monkeypatch):
    """
    SS_solver's functional iteration with the stochastic solver fits
    the asset grids at the first iteration and keeps them.
    """
    p, ss = ss_one_state
    # a few iterations show the grids are kept; they are not enough for
    # the resource constraint check
    p = copy.deepcopy(p)
    p.maxiter = 3
    monkeypatch.setattr(SS, "ENFORCE_SOLUTION_CHECKS", False)
    calls = []
    solve = household_stoch.solve_all_households_ss

    def solve_and_record(*args, b_grid=None, **kwargs):
        sol = solve(*args, b_grid=b_grid, **kwargs)
        calls.append((b_grid, sol.b_grid))
        return sol

    monkeypatch.setattr(
        household_stoch, "solve_all_households_ss", solve_and_record
    )
    out = SS.SS_solver(
        ss["b_sp1"],
        ss["n"],
        ss["r_p"],
        ss["r"],
        ss["w"],
        ss["p_m"],
        ss["Y"],
        ss["BQ"],
        ss["TR"],
        None,
        ss["factor"],
        p,
        None,
    )
    assert calls[0][0] is None and len(calls) > 1
    for b_grid, _ in calls[1:]:
        np.testing.assert_array_equal(b_grid, calls[0][1])
    assert_close(out, ss, GE_VARS)


def test_run_TPI_egm_not_implemented():
    p = egm_specs(ogcore_test_calibration())
    with pytest.raises(NotImplementedError):
        TPI.run_TPI(p)


@pytest.mark.local
def test_one_state_reproduces_default_ss_full_calibration():
    """
    OG-Core's default calibration (S = 80, J = 7): with one shock state
    the stochastic solver's steady state matches OG-Core's.
    """
    ss_root = SS.run_SS(Specifications(baseline=True))
    p = egm_specs(Specifications(baseline=True))
    ss = SS.run_SS(p)
    assert_close(ss, ss_root, GE_VARS + OTHER_AGGREGATES)
    assert np.abs(ss["resource_constraint_error"]).max() < p.RC_SS


@pytest.mark.local
def test_shocks_full_calibration():
    """OG-Core's default calibration with five shock states."""
    p = egm_specs(Specifications(baseline=True), nz=5)
    ss = SS.run_SS(p)
    assert np.abs(ss["resource_constraint_error"]).max() < p.RC_SS
    assert ss["dist"][:, :, -1, :].sum() < 1e-10
