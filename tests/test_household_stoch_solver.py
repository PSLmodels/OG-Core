"""
Tests of the household_stoch solver with idiosyncratic shocks: shock
process, inputs (bequests, transfers, remittances, pensions), the
stationary distribution, accuracy (Euler errors), the borrowing
constraint, failure handling, backends, and economic properties
(continuity in the shock variance, precautionary saving), including
the accuracy checks with five shock states.
"""

import logging
import time

import numpy as np
import pytest
import scipy.interpolate as itp
import scipy.stats as sps
from ogcore import aggregates as aggr
from ogcore import household as hh_core
from ogcore import pensions

from ogcore import household_stoch as household
from ogcore import household_stoch_kernels as kernels
from stoch_test_utils import (
    default_specs,
    ogcore_test_calibration,
    set_ar1_shocks,
    set_shocks,
)

R, W, FACTOR = 0.05, 1.2, 100_000
BQ = np.ones(7) * 0.02
TR = 0.01
THETA = np.full(7, 0.3)
EULER_TOL = 1e-4  # off-grid Euler errors (relative consumption)


# %%
# Shock process


@pytest.mark.parametrize(
    "nz,rho,sigma", [(2, 0.5, 0.1), (5, 0.9, 0.15), (7, 0.97, 0.05)]
)
def test_rouwenhorst_moments(nz, rho, sigma):
    """
    The Rouwenhorst chain matches the AR(1)'s unconditional variance and
    conditional mean exactly, its stationary distribution is
    Binomial(nz - 1, 1/2), and E[z] = 1.
    """
    z, P, pi = household.rouwenhorst(nz, rho, sigma)
    assert z.shape == (nz,) and P.shape == (nz, nz)
    assert np.allclose(P.sum(axis=1), 1.0) and (P >= 0).all()
    assert np.allclose(pi, sps.binom.pmf(np.arange(nz), nz - 1, 0.5))
    assert np.allclose(pi @ P, pi)
    assert np.allclose(household.stationary_distribution(P), pi)
    assert np.isclose(pi @ z, 1.0)
    x = np.log(z)
    mu = pi @ x
    assert np.isclose(pi @ (x - mu) ** 2, sigma**2 / (1 - rho**2))
    # E[x' | x] = mu + rho (x - mu)
    assert np.allclose(P @ x, mu + rho * (x - mu))
    assert np.all(np.diff(z) > 0)


def test_rouwenhorst_one_state_and_bad_input():
    z, P, pi = household.rouwenhorst(1, 0.9, 0.1)
    assert np.allclose(z, [1.0]) and np.allclose(P, [[1.0]])
    with pytest.raises(ValueError):
        household.rouwenhorst(3, 1.0, 0.1)
    with pytest.raises(ValueError):
        household.rouwenhorst(3, 0.9, -0.1)
    with pytest.raises(ValueError):
        household.rouwenhorst(0, 0.9, 0.1)


def test_ar1_shock_process():
    p = set_ar1_shocks(default_specs(), 5, 0.9, 0.15)
    z, P, pi = household.rouwenhorst(5, 0.9, 0.15)
    assert p.nz == 5
    assert np.allclose(p.z_grid, z) and np.allclose(p.z_trans, P)
    assert np.allclose(p.z_dist0, pi)


# %%
# Inputs


def test_household_inputs_match_ogcore():
    """
    Per-household receipts in the steady state are computed as in
    OG-Core's SS.inner_loop.
    """
    p = default_specs()
    p.alpha_RM_T = 0.02
    p.c_min = np.array([0.01])
    p.tau_c = np.full((p.T + p.S, p.I), 0.07)
    p.ubi_nom_array = np.full((p.T + p.S, p.S, p.J), 1500.0)
    Y = 1.3
    out = household.household_inputs_ss(p, R, W, [1.0], BQ, TR, FACTOR, Y)
    p_i = np.dot(p.io_matrix, np.array([1.0]))
    assert np.isclose(
        out["p_tilde"], aggr.get_ptilde(p_i, p.tau_c[-1, :], p.alpha_c)
    )
    assert np.isclose(out["c_min_exp"], 1.07 * 0.01)
    assert np.allclose(out["bq"], hh_core.get_bq(BQ, None, p, "SS"))
    assert np.allclose(out["tr"], hh_core.get_tr(TR, None, p, "SS"))
    RM = aggr.get_RM(Y, p, "SS")
    assert np.allclose(out["rm"], hh_core.get_rm(RM, None, p, "SS"))
    assert np.allclose(out["ubi"], 1500.0 / FACTOR)
    # remittances need GDP
    with pytest.raises(ValueError):
        household.household_inputs_ss(p, R, W, [1.0], BQ, TR, FACTOR)


def test_lifecycle_inputs_ss():
    p = default_specs()
    p.income_tax_filer[-1, 3] = 0.0
    js = [1, 3]
    S = p.S
    bq = np.arange(S * p.J, dtype=float).reshape(S, p.J)
    inp = household.lifecycle_inputs_ss(
        p, R, W, 1.1, FACTOR, bq, 0.0, 0.02, 0.0, 0.1, 0.03, js=js
    )
    assert inp.S == S and inp.M == 2
    assert np.allclose(inp.bq, bq[:, js])
    assert np.allclose(inp.e, p.e[-1][:, js])
    assert np.allclose(inp.rho, p.rho[-1][:, js])
    assert np.allclose(inp.income_filer[:, 1], 0.0)
    assert np.allclose(inp.beta, p.beta[js])
    assert np.allclose(inp.etr_params, np.array(p.etr_params)[-1])
    other = inp.bq + 0.02 + 0.1 - p.tau_bq[-1] * inp.bq - 0.03
    assert np.allclose(inp.other_income(), other)
    sub = inp.select([1])
    assert np.allclose(sub.bq, bq[:, [3]]) and sub.js[0] == 3


def test_pension_ss_us_style():
    """
    US-style benefits are adjust * theta_j * w from the retirement age
    on, and theta from mean effective labor is OG-Core's
    replacement_rate_vals.
    """
    p = default_specs()
    p.replacement_rate_adjust = np.full((p.T + p.S, p.J), 1.1)
    pension = household.pension_ss(p, W, FACTOR, theta=THETA)
    retire = p.retire[-1]
    assert np.allclose(pension[:retire], 0.0)
    assert np.allclose(pension[retire:], 1.1 * THETA * W)
    n_eff = np.full((p.S, p.J), 0.35)
    theta = pensions.replacement_rate_vals(n_eff, W, FACTOR, None, p)
    assert np.allclose(
        household.pension_ss(p, W, FACTOR, n_eff=n_eff),
        household.pension_ss(p, W, FACTOR, theta=theta),
    )


def test_make_asset_grid():
    g = household.make_asset_grid(10.0, 50, 3.0)
    assert g[0] == 0.0 and np.isclose(g[-1], 10.0)
    assert np.all(np.diff(g) > 0) and np.all(np.diff(g, 2) > 0)
    with pytest.raises(ValueError):
        household.make_asset_grid(0.0, 10)


# %%
# Five shock states, default calibration, fixed prices


def _five_state_specs(sigma=0.15):
    return set_ar1_shocks(default_specs(), 5, 0.9, sigma)


@pytest.fixture(scope="module")
def five_state():
    p = _five_state_specs()
    sol = household.solve_all_households_ss(
        R, W, [1.0], BQ, TR, FACTOR, p, theta=THETA
    )
    errors = household.euler_errors(sol)
    return p, sol, errors


def test_five_state_no_failures(five_state):
    p, sol, _ = five_state
    d = sol.diagnostics
    print(d.summary())
    assert d.static_failures == 0 and d.egm_failures == 0
    assert d.euler_failures == 0 and d.extrapolated == 0
    assert np.isfinite(sol.b_policy).all()
    assert d.n_points == p.S * p.J * sol.b_grid.shape[1] * p.nz


def test_five_state_distribution(five_state):
    """
    The distribution is population weighted: each (age, type) cell has
    mass omega_SS[s, j]; newborns hold zero assets with the newborn
    shock distribution; no mass reaches the top of the grid.
    """
    p, sol, _ = five_state
    dist = sol.dist
    assert (dist >= 0).all()
    assert np.allclose(dist.sum(axis=(2, 3)), p.omega_SS, rtol=1e-12)
    assert np.isclose(dist.sum(), p.omega_SS.sum())
    assert np.allclose(
        dist[0, :, 0, :], p.omega_SS[0][:, None] * p.z_dist0[None, :]
    )
    assert np.allclose(dist[0, :, 1:, :], 0.0)
    dd = sol.dist_diagnostics
    assert (dd["mass_above_grid"] == 0).all()
    assert (dd["mass_below_grid"] == 0).all()
    assert (dd["top_node_mass"] < 1e-10).all()


def test_five_state_euler_errors(five_state):
    """
    Euler equation errors between grid nodes are below 1e-4 wherever
    households are.
    """
    _, _, errors = five_state
    by_age = np.nanmax(np.abs(errors["errors"]), axis=(1, 2, 3))
    print("max Euler error by age:", np.array2string(by_age, precision=1))
    print("max with mass:", errors["max_with_mass"])
    print("mean with mass:", errors["mean_with_mass"])
    assert errors["nan"] == 0
    assert errors["max_with_mass"] < EULER_TOL
    assert errors["share_above"][1e-4] == 0.0


def test_five_state_monotone_policies(five_state):
    """
    Savings and consumption rise with assets wherever households are,
    and savings rise with assets at every node outside ages 34-63.

    Near zero assets at ages 34-63, OG-Core's default DEP tax functions
    raise the tax rate on all income steeply with the first dollars of
    capital income, so after-tax resources, and with them savings, can
    fall with assets there; no households are there. (Policies need not
    rise with productivity: rich type-6 households near retirement work a
    little more and save slightly less when z is higher.)
    """
    p, sol, _ = five_state
    mass = sol.dist > 0
    pair_b = mass[:, :, 1:, :] | mass[:, :, :-1, :]
    assert (np.diff(sol.b_policy, axis=2)[pair_b] > 0).all()
    assert (np.diff(sol.c_policy, axis=2)[pair_b] > 0).all()
    dep_free = [s for s in range(p.S) if s < 34 or s > 63]
    assert (np.diff(sol.b_policy[dep_free], axis=2) >= 0).all()


def test_five_state_budget_and_labor_foc(five_state):
    """
    At grid nodes the budget constraint and the labor FOC hold to
    rounding, checked with the pointwise functions (OG-Core's tax
    functions) for a sample of ages and types.
    """
    p, sol, _ = five_state
    inp = sol.inputs
    for s in (0, 20, 45, 79):
        for j in (0, 6):
            for iz, z in enumerate(sol.z_grid):
                b = sol.b_grid[j]
                n = sol.n_policy[s, j, :, iz]
                c = sol.c_policy[s, j, :, iz]
                net_tax = household.get_net_taxes(
                    R,
                    W,
                    b,
                    n,
                    inp.bq[s, j],
                    FACTOR,
                    inp.tr[s, j],
                    inp.ubi[s, j],
                    inp.pension[s, j],
                    None,
                    j,
                    "SS",
                    p.e[-1, s, j] * z,
                    np.array(p.etr_params)[-1, s],
                    p,
                )
                bc = household.BC_residual(
                    c,
                    n,
                    b,
                    sol.b_policy[s, j, :, iz],
                    R,
                    W,
                    1.0,
                    np.ones(1),
                    p.e[-1, s, j],
                    z,
                    inp.bq[s, j],
                    inp.rm[s, j],
                    net_tax,
                    p.tau_c[-1, :],
                    p,
                )
                foc = household.FOC_labor(
                    R,
                    W,
                    1.0,
                    b,
                    c,
                    n,
                    FACTOR,
                    p.e[-1, s, j],
                    z,
                    p.chi_n[-1, s],
                    np.array(p.etr_params)[-1, s],
                    np.array(p.mtrx_params)[-1, s],
                    None,
                    j,
                    p,
                    "SS",
                )
                scale = 1 + np.abs((1 + R) * b)
                assert np.abs(bc / scale).max() < 1e-10
                assert (
                    np.abs(
                        foc / household.marg_ut_labor(n, p.chi_n[-1, s], p)
                    ).max()
                    < 1e-8
                )


# %%
# Borrowing constraint


def _borrowing_problem():
    """
    No bequest motive and steeply rising earnings: the young would like
    to borrow, so the borrowing constraint binds at low assets.
    """
    p = set_shocks(default_specs(), [0.7, 1.3], [[0.8, 0.2], [0.2, 0.8]])
    p.S, p.J, p.T = 10, 1, 10
    p.beta = np.array([0.95])
    p.sigma = 2.0
    p.g_y = 0.0
    p.chi_b = np.array([0.0])
    p.ltilde, p.b_ellipse, p.upsilon = 1.0, 0.6, 2.0
    p.rho = np.r_[np.full(p.S - 1, 0.01), 1.0]
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.capital_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.h_wealth = np.zeros(p.T)
    p.m_wealth = np.ones(p.T)
    p.p_wealth = np.zeros(p.T)
    p.tau_payroll = np.zeros((p.T, p.J))
    etr = np.zeros((p.S, 12))
    mtrx = np.zeros((p.S, 12))
    mtry = np.zeros((p.S, 12))
    etr[:, 10] = mtrx[:, 10] = mtry[:, 10] = 0.15  # flat 15% (DEP shift)
    S = p.S
    args = {
        "r": np.full(S, 0.02),
        "w": np.ones(S),
        "p_tilde": np.ones(S),
        "factor": 1.0,
        "tr": np.zeros(S),
        "bq": np.zeros(S),
        "ubi": 0.0,
        "b_grid": household.make_asset_grid(6.0, 60, 2.0),
        "sigma": p.sigma,
        "pension": np.zeros(S),
        "chi_n": np.full(S, 1.0),
        "rho": p.rho,
        "e": np.linspace(0.3, 3.0, S),
        "etr_params": etr,
        "mtrx_params": mtrx,
        "mtry_params": mtry,
        "j": 0,
        "t": None,
        "p": p,
        "method": "SS",
    }
    return p, args


def _euler_consumption(p, args, n_pol, c_pol, s, iz, b_next):
    """
    Consumption the savings Euler equation implies at age s, shock iz
    and savings b_next, from the pointwise c_from_b_splus1 with next
    period's policies interpolated (monotone cubic) at b_next.
    """
    grid = args["b_grid"]
    b_next = np.atleast_1d(np.asarray(b_next, dtype=float))

    def at_next(policy):
        return np.stack(
            [
                itp.PchipInterpolator(grid, policy[s + 1, :, izp])(b_next)
                for izp in range(p.nz)
            ],
            axis=-1,
        )

    return np.atleast_1d(
        household.c_from_b_splus1(
            args["r"][s + 1],
            args["w"][s + 1],
            1.0,
            1.0,
            b_next,
            at_next(n_pol),
            at_next(c_pol),
            args["factor"],
            p.rho[s],
            args["etr_params"][s + 1],
            args["mtry_params"][s + 1],
            0,
            None,
            args["e"][s + 1],
            iz,
            p,
            "SS",
        )
    )


def test_borrowing_constraint():
    """
    Where the constraint binds, savings are exactly the borrowing limit,
    the budget and labor FOC hold, and the Euler equation holds as an
    inequality: marginal utility now is at least the value of saving.
    Just above the constrained region the Euler equation holds.
    """
    p, args = _borrowing_problem()
    b_pol, c_pol, n_pol, diag = household.solve_HH(
        **args, return_diagnostics=True
    )
    grid = args["b_grid"]
    print(diag.summary())
    assert diag.constrained > 0
    assert diag.static_failures == 0 and diag.egm_failures == 0
    assert diag.euler_failures == 0
    assert diag.max_euler_residual <= household.SolverOptions().euler_tol
    constrained = b_pol == 0.0
    assert constrained[:-1].any()
    for s in range(p.S - 1):
        for iz in range(p.nz):
            con = constrained[s, :, iz]
            k = np.argmin(con)  # first unconstrained node
            # the Euler equation at the first unconstrained nodes
            free = np.arange(k, k + 3)
            c_euler = _euler_consumption(
                p, args, n_pol, c_pol, s, iz, b_pol[s, free, iz]
            )
            assert np.abs(c_pol[s, free, iz] / c_euler - 1).max() < 1e-4
            if not con.any():
                continue
            # the constrained region is the bottom of the grid
            assert con[:k].all() and not con[k:].any()
            # Euler inequality at the limit (b' = 0 is a node, so next
            # period's policies need no interpolation)
            c_limit = _euler_consumption(p, args, n_pol, c_pol, s, iz, 0.0)
            assert (c_pol[s, con, iz] <= c_limit * (1 + 1e-10)).all()
    # budget and labor FOC at constrained nodes
    for s, iz in np.argwhere(constrained[:-1].any(axis=1)):
        con = constrained[s, :, iz]
        z = p.z_grid[iz]
        n = n_pol[s, con, iz]
        c = c_pol[s, con, iz]
        net_tax = household.get_net_taxes(
            args["r"][s],
            args["w"][s],
            grid[con],
            n,
            0.0,
            args["factor"],
            0.0,
            0.0,
            0.0,
            None,
            0,
            "SS",
            args["e"][s] * z,
            args["etr_params"][s],
            p,
        )
        bc = household.BC_residual(
            c,
            n,
            grid[con],
            np.zeros(con.sum()),
            args["r"][s],
            args["w"][s],
            1.0,
            np.ones(1),
            args["e"][s],
            z,
            0.0,
            0.0,
            net_tax,
            p.tau_c[-1, :],
            p,
        )
        foc = household.FOC_labor(
            args["r"][s],
            args["w"][s],
            1.0,
            grid[con],
            c,
            n,
            args["factor"],
            args["e"][s],
            z,
            args["chi_n"][s],
            args["etr_params"][s],
            args["mtrx_params"][s],
            None,
            0,
            p,
            "SS",
        )
        assert np.abs(bc).max() < 1e-12
        assert np.abs(foc).max() < 1e-8


@pytest.fixture(scope="module")
def constrained_five_state():
    """
    Default calibration (DEP taxes) with five shock states, no bequest
    motive and no transfers: most of the young hold no assets.
    """
    p = _five_state_specs()
    p.chi_b = np.zeros(p.J)
    sol = household.solve_all_households_ss(
        R, W, [1.0], BQ, 0.0, FACTOR, p, theta=THETA
    )
    return p, sol


def test_constraint_binds_with_shocks(constrained_five_state):
    """
    With shocks and OG-Core's tax functions the constraint binds where
    households are; there the Euler inequality holds, checked with the
    pointwise c_from_b_splus1 at b' = 0 (a grid node, so no
    interpolation), and the Euler errors between nodes are small on
    average. (They are larger next to kinks, where the policy turns
    from b' = 0 to positive savings at this age or the next, which
    interpolation between nodes does not resolve.)
    """
    p, sol = constrained_five_state
    d = sol.diagnostics
    print(d.summary())
    assert d.static_failures == 0 and d.egm_failures == 0
    assert d.euler_failures == 0
    at_limit = sol.b_policy == 0.0
    assert sol.dist[at_limit].sum() > 0.1 * sol.dist.sum()
    inp = sol.inputs
    etr = np.array(p.etr_params)[-1]
    mtry = np.array(p.mtry_params)[-1]
    for s in range(0, p.S - 1, 7):
        for j in range(p.J):
            for iz in range(p.nz):
                con = at_limit[s, j, :, iz]
                if not con.any():
                    continue
                c_limit = household.c_from_b_splus1(
                    R,
                    W,
                    1.0,
                    1.0,
                    np.zeros(1),
                    sol.n_policy[s + 1, j, :1, :],
                    sol.c_policy[s + 1, j, :1, :],
                    FACTOR,
                    p.rho[-1, s, j],
                    etr[s + 1],
                    mtry[s + 1],
                    j,
                    None,
                    p.e[-1, s + 1, j],
                    iz,
                    p,
                    "SS",
                )
                c = sol.c_policy[s, j, con, iz]
                assert (c <= c_limit * (1 + 1e-10)).all()
                if s < 34 or s > 63:
                    # the constrained region is the bottom of the grid
                    # (at ages 34-63 households with no assets pay less
                    # tax than those with a little, and may save more)
                    assert con[: con.sum()].all()
    errors = household.euler_errors(sol)
    err = np.nan_to_num(np.abs(errors["errors"]))
    weight = 0.5 * (sol.dist[:-1, :, 1:, :] + sol.dist[:-1, :, :-1, :])
    mean_err = (weight * err).sum() / weight.sum()
    share_large = weight[err > 1e-3].sum() / weight.sum()
    print("mean Euler error:", mean_err, "share above 1e-3:", share_large)
    assert mean_err < 1e-5
    assert share_large < 2e-3


def test_grid_must_start_at_the_borrowing_limit():
    p, inp, grid = _one_type_inputs()
    with pytest.raises(ValueError, match="start at 0"):
        household.solve_lifecycle(inp, grid + 0.01, p.z_grid, p.z_trans)
    with pytest.raises(ValueError, match="increasing"):
        household.solve_lifecycle(inp, grid[::-1], p.z_grid, p.z_trans)


# %%
# The Euler equation's right-hand side


def test_value_of_savings_matches_pointwise():
    """
    The solver's batched expectation over next period's shocks
    (_value_of_savings) matches the pointwise c_from_b_splus1, with an
    asymmetric transition matrix, fractional filers, noncompliance, a
    wealth tax and growth, with and without analytical MTRs.
    """
    rng = np.random.default_rng(3)
    p = default_specs()
    T, S, J = p.T, p.S, p.J
    z_grid = np.array([0.6, 1.0, 1.5])
    z_trans = np.array([[0.7, 0.25, 0.05], [0.1, 0.6, 0.3], [0.02, 0.18, 0.8]])
    set_shocks(p, z_grid, z_trans)
    p.income_tax_filer = np.full((T + S, J), 0.7)
    p.wealth_tax_filer = np.full((T + S, J), 0.6)
    p.h_wealth = np.full(T + S, 0.5)
    p.m_wealth = np.full(T + S, 1.0)
    p.p_wealth = np.full(T + S, 0.01)
    p.labor_income_tax_noncompliance_rate = np.full((T + S, J), 0.1)
    p.capital_income_tax_noncompliance_rate = np.full((T + S, J), 0.2)
    p.g_y = 0.02
    nb = 12
    grid = household.make_asset_grid(10.0, nb)
    b3 = np.broadcast_to(grid[None, :, None], (1, nb, 3))
    for analytical in (False, True):
        p.analytical_mtrs = analytical
        for j in (0, 4):
            inp = household.lifecycle_inputs_ss(
                p, R, W, 1.0, FACTOR, 0.02, 0.0, 0.01, 0.0, 0.0, js=[j]
            )
            backend = household._make_backend(inp, "numba")
            for s in (3, 30, 70):
                n_next = rng.uniform(0.1, 0.8, (1, nb, 3))
                c_next = rng.uniform(0.2, 2.0, (1, nb, 3))
                v = household._value_of_savings(
                    inp, backend, s, b3, n_next, c_next, z_grid, z_trans
                )
                c_solver = kernels.inv_mu_c(v, inp.sigma)[0]
                for iz in range(3):
                    c_ref = household.c_from_b_splus1(
                        R,
                        W,
                        1.0,
                        1.0,
                        grid,
                        n_next[0],
                        c_next[0],
                        FACTOR,
                        p.rho[-1, s, j],
                        np.array(p.etr_params)[-1, s + 1],
                        np.array(p.mtry_params)[-1, s + 1],
                        j,
                        None,
                        p.e[-1, s + 1, j],
                        iz,
                        p,
                        "SS",
                    )
                    assert np.allclose(c_solver[:, iz], c_ref, rtol=1e-12)
    # the expectation runs over the row of the current shock
    v_t = household._value_of_savings(
        inp, backend, 30, b3, n_next, c_next, z_grid, z_trans.T.copy()
    )
    v = household._value_of_savings(
        inp, backend, 30, b3, n_next, c_next, z_grid, z_trans
    )
    assert np.abs(v_t / v - 1).max() > 0.01


# %%
# Failures are counted, logged and raised


def _one_type_inputs():
    p = _five_state_specs()
    hh = household.household_inputs_ss(p, R, W, [1.0], BQ, TR, FACTOR)
    pension = household.pension_ss(p, W, FACTOR, theta=THETA)
    inp = household.lifecycle_inputs_ss(
        p,
        R,
        W,
        hh["p_tilde"],
        FACTOR,
        hh["bq"],
        hh["rm"],
        hh["tr"],
        hh["ubi"],
        pension,
        hh["c_min_exp"],
        js=[0],
    )
    return p, inp, household.make_asset_grid(15.0, 30, 4.0)


def _failing_static(b_from):
    """
    Wrap the numba backend's static solve to report failure at grid
    nodes with assets b >= b_from, whatever the savings (so the direct
    Euler solve cannot repair them).
    """
    original = household._NumbaBackend.static

    def static(self, s, n0, b, *args):
        n, c, status, res = original(self, s, n0, b, *args)
        status = np.where(b >= b_from, kernels.NOT_CONVERGED, status)
        return n, c, status, res

    return static


def test_failures_counted_and_logged(monkeypatch, caplog):
    """
    A few failures are counted, their ages recorded and a warning
    logged, without stopping the solve.
    """
    p, inp, grid = _one_type_inputs()
    monkeypatch.setattr(
        household._NumbaBackend, "static", _failing_static(grid[-1])
    )
    options = household.SolverOptions(backend="numba", max_fail_frac=0.05)
    with caplog.at_level(logging.WARNING, logger="household_stoch"):
        *_, diag = household.solve_lifecycle(
            inp, grid, p.z_grid, p.z_trans, options
        )
    # the top node fails in every shock state at every age
    assert diag.static_failures == p.S * p.nz
    assert diag.failure_ages == list(range(p.S))
    assert "Household solver failures" in caplog.text


def test_failures_raise(monkeypatch):
    """Failures above max_fail_frac stop the solve."""
    p, inp, grid = _one_type_inputs()
    with monkeypatch.context() as mp:
        mp.setattr(
            household._NumbaBackend, "static", _failing_static(grid[-3])
        )
        options = household.SolverOptions(backend="numba", max_fail_frac=1e-4)
        with pytest.raises(household.HouseholdSolverError):
            household.solve_lifecycle(inp, grid, p.z_grid, p.z_trans, options)
    # a starved Newton solver fails for real
    with pytest.raises(household.HouseholdSolverError):
        household.solve_lifecycle(
            inp, grid, p.z_grid, p.z_trans, household.SolverOptions(maxiter=1)
        )


def test_failures_where_nobody_is_do_not_stop_the_steady_state(monkeypatch):
    """
    solve_all_households_ss weighs failures by the population at the
    failed grid points: failures far above where households are are
    logged, not raised, while the same failures stop solve_lifecycle,
    which counts grid points, and failures where households are stop
    both.
    """
    p = _five_state_specs()
    grid = household.make_asset_grid(150.0, 60, 4.0)
    monkeypatch.setattr(
        household._NumbaBackend, "static", _failing_static(100.0)
    )
    sol = household.solve_all_households_ss(
        R, W, [1.0], BQ, TR, FACTOR, p, theta=THETA, b_grid=grid
    )
    d = sol.diagnostics
    assert d.static_failures > 1e-4 * d.n_points
    assert d.failed_mass == 0.0
    assert d.failed_nodes.sum() == d.static_failures + d.euler_failures
    with pytest.raises(household.HouseholdSolverError):
        household.solve_lifecycle(sol.inputs, grid, p.z_grid, p.z_trans)
    # newborns hold zero assets
    monkeypatch.setattr(
        household._NumbaBackend, "static", _failing_static(0.0)
    )
    with pytest.raises(household.HouseholdSolverError, match="population"):
        household.solve_all_households_ss(
            R, W, [1.0], BQ, TR, FACTOR, p, theta=THETA, b_grid=grid
        )


def test_inputs_are_validated():
    p, inp, grid = _one_type_inputs()
    z, P = p.z_grid, p.z_trans
    with pytest.raises(ValueError, match="asset grid"):
        household.solve_lifecycle(inp, np.r_[grid[:-1], np.nan], z, P)
    with pytest.raises(ValueError, match="z_trans"):
        household.solve_lifecycle(inp, grid, z, P * 1.01)
    with pytest.raises(ValueError, match="z_grid"):
        household.solve_lifecycle(inp, grid, -z, P)
    bad = household.LifecycleInputs(**vars(inp))
    bad.tr = inp.tr.copy()
    bad.tr[3, 0] = np.nan
    with pytest.raises(ValueError, match="other than labor"):
        household.solve_lifecycle(bad, grid, z, P)
    with pytest.raises(ValueError, match="euler_tol"):
        household.solve_lifecycle(
            inp, grid, z, P, household.SolverOptions(euler_tol=1e-12)
        )


def test_euler_errors_report_failed_solves(five_state, monkeypatch):
    """
    Midpoints where a within-period solve fails count as NaN errors,
    reported separately and as infinite in the maximum, not dropped.
    """
    p, sol, _ = five_state
    original = household._static_at

    def failing(inp, backend, s, b, b_next, zs, n0, options):
        n, c = original(inp, backend, s, b, b_next, zs, n0, options)
        if s == 10:
            c = c.copy()
            c[0, :3, :] = np.nan
        return n, c

    monkeypatch.setattr(household, "_static_at", failing)
    errors = household.euler_errors(sol, ages=[9, 10])
    assert errors["nan"] > 0
    assert errors["max"] == np.inf
    assert np.isnan(errors["errors"][:9]).all()  # ages not checked
    assert np.isfinite(errors["errors"][10, 1:]).all()


# %%
# Pieces of the solver


def _knot_case():
    """
    Synthetic knots for two types and two shocks on one grid (b' at
    knot k is the grid node). Type 0, shock 0: b' = b + 0.25, so the
    knots for low b' lie below the borrowing limit; shock 1: b' =
    (b - 0.1) / 2, a constrained region below b = 0.1, and one knot
    out of order. Type 1, shock 0: a single usable knot.
    """
    grid = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8])
    nb = grid.size
    b_endo = np.full((2, nb, 2), np.nan)
    b_endo[0, :, 0] = grid - 0.25
    b_endo[0, :, 1] = 2 * grid + 0.1
    b_endo[0, 4, 1] = 10.0  # out of order: dropped
    b_endo[1, :, 1] = grid - 0.25
    b_endo[1, 6, 0] = 1.0  # the only knot of type 1, shock 0
    ok = np.isfinite(b_endo)
    n_endo = 0.3 + 0.1 * np.broadcast_to(grid[None, :, None], b_endo.shape)
    return np.tile(grid, (2, 1)), b_endo, n_endo, ok


def test_to_exogenous_on_synthetic_knots():
    grid, b_endo, n_endo, ok = _knot_case()
    b_next, n_guess, gap, counts = household._to_exogenous(
        b_endo, n_endo, ok, grid
    )
    g = grid[0]
    # type 0, shock 0: linear knots, reproduced exactly; node 0 is a
    # gap node with the linear guess across the limit; the top node is
    # extrapolated
    assert np.allclose(b_next[0, :, 0], g + 0.25)
    assert np.allclose(n_guess[0, :, 0], 0.3 + 0.1 * (g + 0.25))
    assert gap[0, :, 0].tolist() == [True] + [False] * 7
    # type 0, shock 1: constrained below the knot for b' = 0
    assert b_next[0, 0, 1] == 0.0 and not gap[0, :, 1].any()
    assert np.allclose(b_next[0, 1:, 1], (g[1:] - 0.1) / 2)
    # type 1, shock 0: one usable knot, every node solved directly
    assert gap[1, :, 0].all() and np.allclose(b_next[1, :, 0], g)
    assert np.isnan(n_guess[1, :, 0]).all()
    assert counts == {
        "constrained": 1,
        "gap": 1 + 8 + 1,
        "extrapolated": 2,
        "dropped": 1,
    }


def test_direct_euler_solve_backends_agree(five_state):
    """
    The compiled and numpy direct Euler solves give the same savings,
    labor and outcome at every node of an age, from guesses far from
    the solution, and agree with the solution's policies where the
    Euler equation held at the node.
    """
    p, sol, _ = five_state
    inp = sol.inputs
    shape = sol.b_policy.shape[1:]
    s = 40
    grid = sol.b_grid
    b3 = np.broadcast_to(grid[:, :, None], shape)
    numba_backend = household._make_backend(inp, "numba")
    v = household._value_of_savings(
        inp,
        numba_backend,
        s,
        b3,
        sol.n_policy[s + 1],
        sol.c_policy[s + 1],
        sol.z_grid,
        sol.z_trans,
    )
    euler_c = household._EulerConsumption(grid, kernels.inv_mu_c(v, inp.sigma))
    pa = household._point_arrays(inp, s, shape, sol.z_grid)
    idx = np.arange(np.prod(shape))
    guess = 1.3 * sol.b_policy[s].ravel() + 0.1
    n0 = np.full(idx.size, 0.5)
    options = household.SolverOptions()
    out = {}
    for name in ("numba", "numpy"):
        backend = household._make_backend(inp, name)
        out[name] = household._solve_euler_direct(
            backend, s, euler_c, idx, guess, n0, pa, options
        )
    assert (out["numba"][2] == out["numpy"][2]).all()
    assert (out["numba"][2] == kernels.EULER_ROOT).all()
    assert np.allclose(out["numba"][0], out["numpy"][0], rtol=1e-10)
    assert np.allclose(out["numba"][1], out["numpy"][1], atol=1e-10)
    assert np.allclose(out["numba"][0], sol.b_policy[s].ravel(), rtol=1e-4)


@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_direct_euler_solve_beyond_the_grid(five_state, backend):
    """
    The direct solve finds savings above the top of the grid, up to
    twice the top, and fails beyond: with a constant Euler consumption
    c(b, b*), the root is b*.
    """
    p, sol, _ = five_state
    inp = sol.inputs
    shape = sol.b_policy.shape[1:]
    M, nb, nz = shape
    s, m, i, iz = 30, 0, nb - 1, 2
    idx = np.array([(m * nb + i) * nz + iz])
    pa = household._point_arrays(inp, s, shape, sol.z_grid)
    sub = {k: v[idx] for k, v in pa.items()}
    b = sol.b_grid[m, i]
    top = sol.b_grid[m, -1]
    be = household._make_backend(inp, backend)
    for factor, solved in (
        (0.5, True),
        (1.1, True),
        (1.9, True),
        (2.2, False),
    ):
        target = factor * top
        _, c_t, _, _ = be.static(
            s,
            np.array([0.3]),
            np.array([b]),
            np.array([target]),
            np.zeros(1),
            sub,
            1e-12,
            60,
        )
        euler_c = household._EulerConsumption(
            sol.b_grid, np.full(shape, c_t[0])
        )
        bp, n, outcome = household._solve_euler_direct(
            be,
            s,
            euler_c,
            idx,
            np.array([b]),
            np.array([0.3]),
            pa,
            household.SolverOptions(),
        )
        if solved:
            assert outcome[0] == kernels.EULER_ROOT
            assert np.isclose(bp[0], target, rtol=1e-8)
        else:
            assert outcome[0] == kernels.EULER_FAILED


# %%
# Backends


def _small_calibration(nz=3):
    # J = 2, S = 40, linear taxes
    return set_ar1_shocks(ogcore_test_calibration(), nz, 0.8, 0.2)


def test_numpy_backend_matches_numba():
    """
    The numpy fallback (OG-Core's tax functions, finite-difference
    Jacobians) and the numba kernels give the same policies.
    """
    p = _small_calibration()
    BQ2 = np.array([0.02, 0.03])
    hh = household.household_inputs_ss(p, R, W, [1.0], BQ2, TR, FACTOR)
    pension = household.pension_ss(p, W, FACTOR, theta=np.full(p.J, 0.2))
    inp = household.lifecycle_inputs_ss(
        p,
        R,
        W,
        hh["p_tilde"],
        FACTOR,
        hh["bq"],
        hh["rm"],
        hh["tr"],
        hh["ubi"],
        pension,
        hh["c_min_exp"],
    )
    grid = household.make_asset_grid(8.0, 25, 3.0)
    out = {}
    for backend in ("numba", "numpy"):
        options = household.SolverOptions(backend=backend)
        out[backend] = household.solve_lifecycle(
            inp, grid, p.z_grid, p.z_trans, options
        )
        assert out[backend][3].backend == backend
    for k in range(3):
        assert np.allclose(out["numba"][k], out["numpy"][k], atol=1e-8)


def test_numpy_backend_matches_numba_dep():
    """
    The same with OG-Core's default DEP tax functions, whose
    derivatives the numpy backend takes by finite differences.
    """
    p = set_ar1_shocks(default_specs(), 3, 0.9, 0.15)
    hh = household.household_inputs_ss(p, R, W, [1.0], BQ, TR, FACTOR)
    pension = household.pension_ss(p, W, FACTOR, theta=THETA)
    inp = household.lifecycle_inputs_ss(
        p,
        R,
        W,
        hh["p_tilde"],
        FACTOR,
        hh["bq"],
        hh["rm"],
        hh["tr"],
        hh["ubi"],
        pension,
        hh["c_min_exp"],
        js=[0, 6],
    )
    grid = household.make_asset_grid(15.0, 30, 4.0)
    out = {}
    for backend in ("numba", "numpy"):
        options = household.SolverOptions(backend=backend)
        out[backend] = household.solve_lifecycle(
            inp, grid, p.z_grid, p.z_trans, options
        )
        d = out[backend][3]
        assert d.static_failures == d.egm_failures == d.euler_failures == 0
    for k in range(3):
        assert np.allclose(out["numba"][k], out["numpy"][k], atol=1e-8)


def test_nonparametric_taxes_use_numpy_backend():
    """
    'mono' tax functions are Python callables: the solver falls back to
    numpy and reproduces the solution with equivalent linear taxes.
    """
    p, args = _borrowing_problem()
    p.chi_b = np.array([5.0])  # bequest motive back on
    S = p.S
    rates = {"etr_params": 0.15, "mtrx_params": 0.2, "mtry_params": 0.1}
    lin_args = dict(args)
    p.tax_func_type = "linear"
    for k, v in rates.items():
        lin_args[k] = np.full((S, 1), v)
    b1, c1, n1 = household.solve_HH(**lin_args)

    def flat(rate):
        return lambda income: rate + 0.0 * np.asarray(income, dtype=float)

    mono_args = dict(args)
    for k, v in rates.items():
        mono_args[k] = [[flat(v)] for _ in range(S)]
    p.tax_func_type = "mono"
    b2, c2, n2, diag = household.solve_HH(**mono_args, return_diagnostics=True)
    assert diag.backend == "numpy"
    for a, b in ((b1, b2), (c1, c2), (n1, n2)):
        assert np.allclose(a, b, atol=1e-8)
    with pytest.raises(ValueError):
        household.solve_HH(
            **mono_args, options=household.SolverOptions(backend="numba")
        )


# %%
# Analytical MTRs


@pytest.fixture(scope="module")
def analytical_five_state():
    p = _five_state_specs()
    p.analytical_mtrs = True
    sol = household.solve_all_households_ss(
        R, W, [1.0], BQ, TR, FACTOR, p, theta=THETA
    )
    return p, sol


def test_analytical_mtrs_five_states(analytical_five_state):
    """
    OG-Core's analytical MTRs exceed 1 - tau_payroll at low labor income
    for households with high capital income, so some households do not
    work (labor supply at its lower bound, where the labor FOC holds as
    an inequality). The solver handles these corners without failures
    (a dropped endogenous-grid knot or two aside). Labor can jump between
    the corner and an interior choice from one node to the next, so
    Euler errors are larger between such nodes; few households are
    there.
    """
    p, sol = analytical_five_state
    d = sol.diagnostics
    print(d.summary())
    assert d.static_failures == d.euler_failures == 0
    assert d.egm_failures <= 1e-5 * d.n_knots
    assert d.labor_corners > 0
    assert np.isfinite(sol.c_policy).all() and (sol.c_policy > 0).all()
    errors = household.euler_errors(sol)
    print({k: v for k, v in errors.items() if k not in ("errors", "b")})
    assert errors["nan_with_mass"] == 0
    assert errors["mean_with_mass"] < 1e-5
    assert errors["share_above"][1e-3] < 1e-3


def test_analytical_mtrs_global_labor_choice(analytical_five_state):
    """
    With analytical MTRs the within-period solution is the best of the
    local optima (no work, or an interior root of the labor FOC): at
    every node with mass, scanning labor supply on a fine grid finds no
    higher within-period utility, whatever labor the solve starts from.
    """
    p, sol = analytical_five_state
    inp = sol.inputs
    backend = household._make_backend(inp, "numba")
    s_list = (5, 30, 50, 70)
    lt = inp.ltilde
    floor = kernels.labor_floor(inp.b_ellipse, inp.upsilon, lt)
    ns = floor + (lt - floor) * np.linspace(0, 1, 2001) ** 2
    worst = 0.0
    for s in s_list:
        shape = sol.b_policy.shape[1:]
        pa = household._point_arrays(inp, s, shape, sol.z_grid)
        b = household._flat(sol.b_grid[:, :, None], shape)
        bp = sol.b_policy[s].ravel()
        mass = sol.dist[s].ravel() > 0
        idx = np.flatnonzero(mass)
        sub = {k: v[idx] for k, v in pa.items()}
        zeros = np.zeros(idx.size)
        for n0 in (1e-4, 0.3 * lt, 0.9 * lt):
            n, c, st, _ = backend.static(
                s,
                np.full(idx.size, n0),
                b[idx],
                bp[idx],
                zeros,
                sub,
                1e-12,
                60,
            )
            assert np.allclose(n, sol.n_policy[s].ravel()[idx], atol=1e-8)
        u_sol = kernels.period_utility_np(
            n, c, 1.0, inp.sigma, inp.chi_n[s], inp.b_ellipse, inp.upsilon, lt
        )
        # utility on the labor grid: consumption from the budget at each n
        for k in range(0, ns.size, 200):
            nn = ns[k : k + 200]
            Q = idx.size * nn.size
            rep = np.repeat(np.arange(idx.size), nn.size)
            F, _, cc, _ = household._NumpyBackend(inp)._static_F(
                s,
                np.arange(Q),
                np.tile(nn, idx.size),
                b[idx][rep],
                bp[idx][rep],
                np.zeros(Q),
                {key: v[rep] for key, v in sub.items()},
            )
            u_grid = kernels.period_utility_np(
                np.tile(nn, idx.size),
                cc,
                1.0,
                inp.sigma,
                inp.chi_n[s],
                inp.b_ellipse,
                inp.upsilon,
                lt,
            ).reshape(idx.size, nn.size)
            gain = (u_grid.max(axis=1) - u_sol) / np.abs(u_sol)
            worst = max(worst, gain.max())
    print("largest utility gain from a labor grid:", worst)
    assert worst < 1e-6


# %%
# Pensions consistent with own earnings


def test_pension_fixed_point():
    """
    With theta left out, replacement rates are consistent with each
    type's mean effective earnings in the solution.
    """
    p = _five_state_specs()
    sol = household.solve_all_households_ss(R, W, [1.0], BQ, TR, FACTOR, p)
    n_eff = household.mean_effective_labor(sol.dist, sol.n_policy, sol.z_grid)
    theta = pensions.replacement_rate_vals(n_eff, W, FACTOR, None, p)
    assert sol.iterations["pension"] >= 2
    assert np.allclose(sol.theta, theta, rtol=1e-12)
    assert np.allclose(
        household.pension_ss(p, W, FACTOR, theta=theta),
        sol.pension,
        atol=1e-10,
    )


# %%
# Economics at fixed prices


def _aggregate_savings(sol):
    return (sol.dist * sol.b_policy).sum()


@pytest.fixture(scope="module")
def by_variance():
    """
    Solutions with one shock state and with growing shock variance, all
    on the asset grids sized for the largest variance.
    """
    widest = household.solve_all_households_ss(
        R, W, [1.0], BQ, TR, FACTOR, _five_state_specs(0.2), theta=THETA
    )
    sols = {0.2: widest}
    for sigma in (0.0, 1e-3, 0.1):
        if sigma == 0.0:
            p = default_specs()
        else:
            p = _five_state_specs(sigma)
        sols[sigma] = household.solve_all_households_ss(
            R, W, [1.0], BQ, TR, FACTOR, p, theta=THETA, b_grid=widest.b_grid
        )
    for sol in sols.values():
        assert (sol.dist_diagnostics["mass_above_grid"] == 0).all()
    return sols


def test_continuity_in_shock_variance(by_variance):
    """
    As the shock variance goes to zero, the five-state solution
    approaches the one-state solution (same asset grids).
    """
    base = by_variance[0.0]
    small = by_variance[1e-3]
    for x in ("b_policy", "n_policy", "c_policy"):
        gap = np.abs(
            small.cell_means(getattr(small, x))
            - base.cell_means(getattr(base, x))
        ).max()
        big = np.abs(
            by_variance[0.1].cell_means(getattr(by_variance[0.1], x))
            - base.cell_means(getattr(base, x))
        ).max()
        print(x, "gap at sigma 1e-3:", gap, "at 0.1:", big)
        assert gap < 1e-3 * big
    assert np.isclose(
        _aggregate_savings(small), _aggregate_savings(base), rtol=1e-5
    )


def test_precautionary_saving(by_variance):
    """At fixed prices, savings rise with the variance of the shocks."""
    K = [_aggregate_savings(by_variance[s]) for s in (0.0, 0.1, 0.2)]
    print("aggregate savings for sigma = 0, 0.1, 0.2:", K)
    assert K[0] < K[1] < K[2]


# %%
# Speed (SS household solve under 5 s with J = 7, five states)


@pytest.mark.local
def test_ss_household_solve_time(five_state):
    p, sol, _ = five_state
    for _ in range(2):  # the first call may compile
        t0 = time.perf_counter()
        household.solve_lifecycle(sol.inputs, sol.b_grid, p.z_grid, p.z_trans)
        elapsed = time.perf_counter() - t0
    print(f"SS household solve (J = 7, nz = 5, nb = 100): {elapsed:.2f}s")
    assert elapsed < 5.0


def test_kernels_can_be_disabled(monkeypatch):
    """The utility kernels give the same answers without numba."""
    c = np.linspace(0.001, 5, 50)
    with_numba = kernels.mu_c(c, 1.5)
    monkeypatch.setattr(kernels, "USE_NUMBA", False)
    assert np.allclose(kernels.mu_c(c, 1.5), with_numba)
