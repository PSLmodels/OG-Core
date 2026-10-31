# %%
"""
Tests of household_stoch: the pointwise household functions, the
single-type solver, and replication of OG-Core's steady-state household
solution with one shock state.

Solver internals (kernels, distribution, accuracy with shocks) are
tested in test_household_stoch_kernels.py and
test_household_stoch_solver.py.
"""

import copy

import numpy as np
import pytest
import scipy.interpolate as itp
import ogcore.household as hh_core
from ogcore import SS, pensions, tax

from ogcore import household_stoch as household
from stoch_test_utils import (
    assert_ogcore_converged,
    default_specs,
    set_shocks,
    simulate_profiles,
    solve_ogcore_benchmark,
    ogcore_test_calibration,
)

# %%
# Functions that household_stoch takes unchanged from ogcore.household are
# tested in OG-Core's own suite. Here we only check they are OG-Core's.


@pytest.mark.parametrize("name", household.OGCORE_HOUSEHOLD_FUNCTIONS)
def test_reexports_ogcore_household(name):
    assert getattr(household, name) is getattr(hh_core, name)


# %%
# Shock-process parameters


def test_default_one_state():
    p = default_specs()
    assert p.household_solver == "root"
    assert p.nz == 1
    assert np.array_equal(p.z_grid, [1.0])
    assert np.array_equal(p.z_trans, [[1.0]])
    assert np.array_equal(p.z_dist0, [1.0])


def test_shock_process_override():
    p = default_specs()
    Z_tfp = p.Z.copy()
    set_shocks(p, [0.8, 1.2], [[0.9, 0.1], [0.2, 0.8]])
    assert p.nz == 2
    assert np.allclose(p.z_grid, [0.8, 1.2])
    assert np.allclose(p.z_trans, [[0.9, 0.1], [0.2, 0.8]])
    # newborns draw from the stationary distribution
    assert np.allclose(p.z_dist0, [2 / 3, 1 / 3])
    # OG-Core's TFP parameter p.Z must be left alone
    assert np.array_equal(p.Z, Z_tfp)
    # the override survives later updates of other parameters
    p.update_specifications({"z_innovation_sd": 0.3})
    assert np.allclose(p.z_grid, [0.8, 1.2])


def test_shock_process_ar1_rebuilt_on_update():
    """
    p.z_grid and p.z_trans hold the Rouwenhorst process, not the empty
    z_grid / z_trans inputs, so updating the AR(1) parameters must
    rebuild them rather than treat the old process as an override.
    """
    p = default_specs()
    p.update_specifications(
        {"household_solver": "egm", "nz": 3, "z_innovation_sd": 0.1}
    )
    z_grid, z_trans, pi = household.rouwenhorst(3, p.z_persistence, 0.1)
    assert np.allclose(p.z_grid, z_grid)
    assert np.allclose(p.z_trans, z_trans)
    assert np.allclose(p.z_dist0, pi)
    p.update_specifications({"nz": 5, "z_innovation_sd": 0.2})
    z_grid, z_trans, pi = household.rouwenhorst(5, p.z_persistence, 0.2)
    assert np.allclose(p.z_grid, z_grid)
    assert np.allclose(p.z_trans, z_trans)


def test_shocks_require_egm_solver():
    p = default_specs()
    with pytest.raises(ValueError, match="household_solver"):
        p.update_specifications({"nz": 3})


@pytest.mark.parametrize(
    "nz,z_grid,z_trans",
    [
        (2, [0.8, 1.2], None),
        (3, [0.8, 1.2], [[0.9, 0.1], [0.1, 0.9]]),
        (2, [0.8, 1.2], [[1.0]]),
        (2, [0.8, 1.2], [[0.9, 0.2], [0.1, 0.9]]),
        (2, [0.8, 1.2], [[1.1, -0.1], [0.1, 0.9]]),
        (2, [0.0, 1.2], [[0.9, 0.1], [0.1, 0.9]]),
    ],
    ids=[
        "z_trans missing",
        "nz does not match z_grid",
        "wrong shape",
        "rows do not sum to 1",
        "negative entry",
        "zero productivity",
    ],
)
def test_shock_process_rejects_bad_input(nz, z_grid, z_trans):
    with pytest.raises(ValueError):
        household.shock_process(nz, 0.9, 0.15, z_grid, z_trans)


# %%

# Test cases for inv_mu_c are the reverse of the test cases for
# marg_ut_cons from test_household.py
inv_mu_c_test_data = [
    (10, 1, 0.1),
    (55.90169944, 2.5, 0.2),
    (
        np.array([9.18958684, 0.002913041, 0.273217159]),
        3.2,
        np.array([0.5, 6.2, 1.5]),
    ),
]


@pytest.mark.parametrize(
    "value,sigma,expected",
    inv_mu_c_test_data,
    ids=["Scalar 1", "Scalar 2", "Vector"],
)
def test_inv_mu_c(value, sigma, expected):
    """
    Test the inverse marginal utility of consumption function `inv_mu_c`.
    """
    test_value = household.inv_mu_c(value, sigma)
    assert np.allclose(test_value, expected)


@pytest.mark.parametrize("sigma", [1.0, 1.5, 3.2])
def test_inv_mu_c_inverts_marg_ut_cons(sigma):
    """
    inv_mu_c is the exact inverse of OG-Core's marg_ut_cons, including
    its linear stitching below c = 0.003 and for large c.
    """
    c = np.concatenate(
        [np.linspace(1e-5, 0.003, 20), np.linspace(0.003, 500, 50)]
    )
    assert np.allclose(
        household.inv_mu_c(hh_core.marg_ut_cons(c, sigma), sigma),
        c,
        rtol=1e-12,
    )


# %%
# marg_ut_beq: chi_b * b**(-sigma), stitched linearly below b = 0.001

MU_BEQ_EPS = 0.001


def _mu_beq_stitched(b, sigma, chi_b):
    slope = -sigma * MU_BEQ_EPS ** (-sigma - 1)
    intercept = MU_BEQ_EPS ** (-sigma) - slope * MU_BEQ_EPS
    return chi_b * (slope * b + intercept)


@pytest.mark.parametrize(
    "b,sigma,j,expected",
    [
        (2.0, 2.0, 1, 2.5 * 2.0**-2.0),
        (0.0005, 2.0, 0, _mu_beq_stitched(0.0005, 2.0, 1.5)),
        (
            np.array([2.0, 0.0005]),
            2.0,
            1,
            np.array([2.5 * 2.0**-2.0, _mu_beq_stitched(0.0005, 2.0, 2.5)]),
        ),
        (np.array([2.0, 3.0]), 1.5, 2, 5.0 * np.array([2.0, 3.0]) ** -1.5),
        (0.0, 1.5, 2, _mu_beq_stitched(0.0, 1.5, 5.0)),
    ],
    ids=[
        "Scalar",
        "Scalar, stitched",
        "Vector, mixed",
        "Vector",
        "At zero",
    ],
)
def test_marg_ut_beq(b, sigma, j, expected):
    p = default_specs()
    p.chi_b = np.array([1.5, 2.5, 5.0])
    assert np.allclose(household.marg_ut_beq(b, sigma, j, p), expected)


def test_marg_ut_beq_continuous_at_stitch():
    p = default_specs()
    p.chi_b = np.array([3.0])
    below = household.marg_ut_beq(MU_BEQ_EPS * (1 - 1e-12), 1.5, 0, p)
    above = household.marg_ut_beq(MU_BEQ_EPS * (1 + 1e-12), 1.5, 0, p)
    assert np.isclose(below, above, rtol=1e-9)


# %%


def setup_c_from_n_params():
    p = default_specs()
    p.S, p.J, p.T = 5, 2, 10
    p.g_y, p.sigma, p.ltilde = 0.02, 2.0, 1.0
    p.b_ellipse, p.upsilon = 5.0, 2.0
    p.chi_n = np.ones((p.S, p.J))
    # payroll tax by period and type (T x J since OG-Core 0.20.4)
    p.tau_payroll = np.linspace(0.05, 0.10, p.T)[:, None] + np.array(
        [0.0, 0.03]
    )
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.labor_income_tax_noncompliance_rate[:, 1] = 0.10
    e_path = np.tile(
        np.linspace(0.5, 1.5, p.S).reshape(1, p.S, 1), (p.T, 1, p.J)
    )
    e_path[:, :, 1] *= 1.2
    p.e = e_path
    etr_params = np.zeros((p.T, p.S, 12))
    mtrx_params = np.zeros((p.T, p.S, 12))
    mtrx_params[:, :, 10] = 0.25
    return {
        "p": p,
        "etr_params": etr_params,
        "mtrx_params": mtrx_params,
        "n": 0.4,
        "b": 10.0,
        "p_tilde": 1.0,
        "r": 0.05,
        "w": 1.2,
        "factor": 1.0,
        "z": 1.0,
    }


@pytest.mark.parametrize("j", [0, 1])
def test_c_from_n_vec(j):
    """
    c_from_n inverts the labor FOC: c**(-sigma) w e z (1 - tau_p - MTR)
    / p_tilde = MDU(n), with type j's payroll tax rate.
    """
    params = setup_c_from_n_params()
    p = params["p"]
    S = p.S
    r, w, b, n, z = (np.ones(S) * params[k] for k in ("r", "w", "b", "n", "z"))
    e = np.squeeze(p.e[-1, :, j])
    chi_n = p.chi_n[:, j]
    mtrx_params_ss = params["mtrx_params"][-1, :, :]
    mtr_labor = tax.MTR_income(
        r,
        w,
        b,
        n,
        params["factor"],
        False,
        e * z,
        None,
        mtrx_params_ss,
        p.labor_income_tax_noncompliance_rate[-1, j],
        p.income_tax_filer[-1, j],
        p,
    )
    deriv = 1 - p.tau_payroll[-1, j] - mtr_labor
    mdu_labor = household.marg_ut_labor(n, chi_n, p)
    expected = (params["p_tilde"] * mdu_labor / (w * e * z * deriv)) ** (
        -1 / p.sigma
    )
    test_c = household.c_from_n(
        n,
        b,
        params["p_tilde"],
        r,
        w,
        params["factor"],
        e=None,
        z=z,
        chi_n=chi_n,
        etr_params=params["etr_params"],
        mtrx_params=mtrx_params_ss,
        t=None,
        j=j,
        p=p,
        method="SS",
    )
    assert np.allclose(test_c, expected)
    # and the labor FOC holds at (c, n)
    foc = household.FOC_labor(
        r,
        w,
        params["p_tilde"],
        b,
        test_c,
        n,
        params["factor"],
        e,
        z,
        chi_n,
        params["etr_params"],
        mtrx_params_ss,
        None,
        j,
        p,
        "SS",
    )
    assert np.allclose(foc, 0, atol=1e-12)


# %%
# b_from_c_EOL


def _p_b_from_c():
    p = default_specs()
    p.J = 2
    p.chi_b = np.array([0.5, 0.8])
    p.sigma = 2.0
    return p


@pytest.mark.parametrize(
    "c, p_tilde, j, sigma, expected_b",
    [
        (1.0, 1.1, 0, 2.0, 0.71970167),
        (np.array([1.0, 2.0]), 1.1, 1, 2.0, np.array([0.9103586, 1.8207172])),
        (
            np.array([1.0, 2.0]),
            np.array([1.1, 1.2]),
            0,
            2.0,
            np.array([0.71970167, 1.50340773]),
        ),
        (2.5, 1.0, 1, 3.0, (2.5 * (0.8 * 1.0) ** (1 / 3.0)) / np.exp(0.03)),
    ],
    ids=["Scalar inputs", "Array c", "Array c and p_tilde", "Different sigma"],
)
def test_b_from_c_EOL(c, p_tilde, j, sigma, expected_b):
    p = _p_b_from_c()
    test_b = household.b_from_c_EOL(c, p_tilde, j, sigma, p)
    assert np.allclose(test_b, expected_b)


def test_b_from_c_EOL_mortality():
    """With last-age mortality rho < 1 the bequest scales by rho**(1/sigma)."""
    p = _p_b_from_c()
    b1 = household.b_from_c_EOL(1.0, 1.1, 0, 2.0, p)
    b_half = household.b_from_c_EOL(1.0, 1.1, 0, 2.0, p, rho=0.5)
    assert np.isclose(b_half, b1 * 0.5**0.5)


# %%
# Budget constraint


@pytest.mark.parametrize(
    "r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, expected_cons",
    [
        (0.04, 1.5, 1.0, 10.0, 11.0, 0.8, 0.5, 2.0, 1.0, 1.2, -0.88221474),
        (
            np.array([0.04, 0.05]),
            np.array([1.5, 1.6]),
            np.array([1.0, 1.1]),
            np.array([10.0, 12.0]),
            np.array([11.0, 13.0]),
            np.array([0.8, 0.7]),
            np.array([0.5, 0.6]),
            np.array([2.0, 2.5]),
            np.array([1.0, 1.1]),
            np.array([1.2, 0.9]),
            np.array([-0.88221474, -1.3216522]),
        ),
    ],
    ids=["Scalar inputs", "Vector inputs"],
)
def test_get_cons(
    r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, expected_cons
):
    """
    Consumption from the budget constraint (no remittances, minimum
    consumption or consumption taxes).
    """
    p = default_specs()
    p.g_y = 0.02
    test_cons = household.get_cons(
        r,
        w,
        p_tilde,
        np.array([1.0]),
        b,
        b_splus1,
        n,
        bq,
        0.0,
        net_tax,
        e,
        z,
        np.array([0.0]),
        p,
    )
    assert np.allclose(test_cons, expected_cons)


def test_get_cons_full_budget():
    """
    With remittances, minimum consumption and consumption taxes the
    budget constraint is OG-Core's with labor productivity e z.
    """
    p = default_specs()
    p.c_min = np.array([0.01])
    r, w, p_tilde = 0.04, 1.5, 1.05
    p_i, tau_c = np.array([1.1]), np.array([0.05])
    b = np.array([0.0, 1.0, 10.0])
    b_splus1 = np.array([0.5, 1.2, 9.0])
    n = np.array([0.3, 0.5, 0.2])
    bq, rm = 0.1, 0.02
    net_tax = np.array([0.05, 0.1, 0.3])
    e = np.array([1.0, 1.5, 0.7])
    z = np.array([0.8, 1.0, 1.3])
    test = household.get_cons(
        r, w, p_tilde, p_i, b, b_splus1, n, bq, rm, net_tax, e, z, tau_c, p
    )
    expected = (
        (1 + r) * b
        + w * e * z * n
        + bq
        + rm
        - net_tax
        - np.exp(p.g_y) * b_splus1
        - (1 + 0.05) * 1.1 * 0.01
    ) / p_tilde
    assert np.allclose(test, expected)
    assert np.allclose(
        test,
        hh_core.get_cons(
            r,
            w,
            p_tilde,
            p_i,
            b,
            b_splus1,
            n,
            bq,
            rm,
            net_tax,
            e * z,
            tau_c,
            p,
            "SS",
        ),
    )


def _net_tax_setup():
    p = default_specs()
    p.h_wealth = np.full(p.T + p.S, 0.5)
    p.m_wealth = np.full(p.T + p.S, 1.0)
    p.p_wealth = np.full(p.T + p.S, 0.01)
    p.tau_bq = np.full(p.T + p.S, 0.2)
    # payroll tax rates that differ by type
    p.tau_payroll = np.tile(np.linspace(0.02, 0.05, p.J), (p.T + p.S, 1))
    p.labor_income_tax_noncompliance_rate = np.full((p.T + p.S, p.J), 0.05)
    p.capital_income_tax_noncompliance_rate = np.full((p.T + p.S, p.J), 0.1)
    p.income_tax_filer = np.full((p.T + p.S, p.J), 0.9)
    p.wealth_tax_filer = np.full((p.T + p.S, p.J), 0.8)
    S = p.S
    args = dict(
        r=0.05,
        w=1.2,
        b=np.linspace(0.0, 5.0, S),
        n=np.linspace(0.2, 0.5, S),
        bq=np.full(S, 0.1),
        factor=100_000,
        tr=np.full(S, 0.02),
        ubi=np.full(S, 0.01),
    )
    return p, args


def test_get_net_taxes_matches_ogcore():
    """
    Over a whole life (S-vectors), net taxes with pensions computed by
    age equal OG-Core's tax.net_taxes, which places US-style benefits by
    array position.
    """
    p, a = _net_tax_setup()
    j = 2
    e = p.e[-1, :, j]
    etr = np.array(p.etr_params[-1])
    theta = np.full(p.J, 0.3)
    expected = tax.net_taxes(
        a["r"],
        a["w"],
        a["b"],
        a["n"],
        a["bq"],
        a["factor"],
        a["tr"],
        a["ubi"],
        theta[j],
        None,
        j,
        False,
        "SS",
        e,
        etr,
        p,
    )
    pension = household.pension_ss(p, a["w"], a["factor"], theta=theta)[:, j]
    test = household.get_net_taxes(
        a["r"],
        a["w"],
        a["b"],
        a["n"],
        a["bq"],
        a["factor"],
        a["tr"],
        a["ubi"],
        pension,
        None,
        j,
        "SS",
        e,
        etr,
        p,
    )
    assert np.allclose(test, expected, rtol=1e-12, atol=1e-14)
    assert (pension[: p.retire[-1]] == 0).all()
    assert (pension[p.retire[-1] :] > 0).all()


def test_pensions_by_age_at_one_point():
    """
    At a single retirement age, OG-Core's net_taxes gives no pension
    (benefits are placed by array position), which is why the household
    grid solve takes pensions by age; get_net_taxes at that age equals
    the whole-life calculation.
    """
    p, a = _net_tax_setup()
    j, s = 3, p.retire[-1] + 2
    e = p.e[-1, :, j]
    etr = np.array(p.etr_params[-1])
    theta = np.full(p.J, 0.25)
    pension = household.pension_ss(p, a["w"], a["factor"], theta=theta)[:, j]
    point = [a[k][s] for k in ("b", "n", "bq")]
    whole_life = household.get_net_taxes(
        a["r"],
        a["w"],
        a["b"],
        a["n"],
        a["bq"],
        a["factor"],
        a["tr"],
        a["ubi"],
        pension,
        None,
        j,
        "SS",
        e,
        etr,
        p,
    )[s]
    at_age = household.get_net_taxes(
        a["r"],
        a["w"],
        point[0],
        point[1],
        point[2],
        a["factor"],
        a["tr"][s],
        a["ubi"][s],
        pension[s],
        None,
        j,
        "SS",
        e[s],
        etr[s],
        p,
    )
    ogcore_at_age = tax.net_taxes(
        a["r"],
        a["w"],
        point[0],
        np.atleast_1d(point[1]),
        point[2],
        a["factor"],
        a["tr"][s],
        a["ubi"][s],
        theta[j],
        None,
        j,
        False,
        "SS",
        e[s],
        etr[s],
        p,
    )
    assert np.isclose(at_age, whole_life, rtol=1e-12)
    # OG-Core's version misses the pension at a single age
    assert np.isclose(ogcore_at_age - at_age, pension[s], rtol=1e-12)


# %%


def setup_c_from_b_splus1_params(taxes=False):
    """
    Set up parameters for testing the c_from_b_splus1 function.
    """
    p = set_shocks(default_specs(), [0.8, 1.2], [[0.9, 0.1], [0.1, 0.9]])
    p.S, p.J, p.T = 5, 1, 5
    p.beta = np.array([0.96])
    p.sigma = 2.0
    p.g_y = 0.02
    p.chi_b = np.array([2.0])
    p.e = np.tile(np.linspace(1.0, 1.2, p.S).reshape(1, p.S, 1), (p.T, 1, 1))
    p.retire = p.S + 1  # Everyone works

    # Tax parameters
    p.capital_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    etr_params = np.zeros((p.T, p.S, 12))
    mtry_params = np.zeros((p.T, p.S, 12))
    if taxes:
        # Add a simple MTR on capital income
        mtry_params[:, :, 10] = 0.15
        p.h_wealth = np.ones(p.T) * 0.4
        p.m_wealth = np.ones(p.T) * 1.0
        p.p_wealth = np.ones(p.T) * 0.01
    else:
        p.h_wealth = np.zeros(p.T)
        p.m_wealth = np.ones(p.T)
        p.p_wealth = np.zeros(p.T)

    # Function inputs
    s = 1  # current age is s=1 (second period of life)
    j = 0
    t = 0
    z_index = 0  # current productivity state is low
    r_splus1 = 0.04
    w_splus1 = 1.5
    p_tilde_splus1 = 1.0
    p_tilde_s = 1.0
    b_splus1 = np.array([5.0, 10.0])  # nb=2
    # Next period's policy functions (nb x nz)
    n_splus1_policy = np.array([[0.3, 0.4], [0.3, 0.4]])
    c_splus1_policy = np.array([[2.0, 2.5], [8.0, 9.0]])
    factor = 1.0
    rho_s = 0.1  # mortality rate at age s

    # Manually calculate expected consumption
    beta = p.beta[j]
    bequest_utility = rho_s * household.marg_ut_beq(b_splus1, p.sigma, j, p)

    # Calculate expectation of marginal utility
    consumption_utility_matrix = np.zeros(
        (b_splus1.shape[0], p.z_grid.shape[0])
    )
    for zp_index, zp in enumerate(p.z_grid):
        mtr_capital = tax.MTR_income(
            r_splus1,
            w_splus1,
            b_splus1,
            n_splus1_policy[:, zp_index],
            factor,
            True,
            p.e[t, s + 1, j] * zp,
            etr_params[t, s + 1, :],
            mtry_params[t, s + 1, :],
            p.capital_income_tax_noncompliance_rate[t, j],
            p.income_tax_filer[t + 1, j],
            p,
        )
        mtr_wealth = tax.MTR_wealth(
            b_splus1,
            p.h_wealth[t + 1],
            p.m_wealth[t + 1],
            p.p_wealth[t + 1],
            p.wealth_tax_filer[t + 1, j],
        )
        deriv = (1 + r_splus1) - (r_splus1 * mtr_capital) - mtr_wealth

        mu_c_splus1 = household.marg_ut_cons(
            c_splus1_policy[:, zp_index], p.sigma
        )
        consumption_utility_matrix[:, zp_index] = (
            deriv * mu_c_splus1 / p_tilde_splus1
        )

    prob_z_splus1 = p.z_trans[z_index, :]
    E_MU_c = consumption_utility_matrix @ prob_z_splus1

    # Savings Euler equation (OG-Core's FOC_savings):
    # MU(c) / p_tilde = e^{-sigma g_y} [rho chi_b MU_b + beta (1 - rho) E]
    mu_c_rhs = bequest_utility + beta * (1 - rho_s) * E_MU_c
    expected_c = (p_tilde_s * np.exp(-p.sigma * p.g_y) * mu_c_rhs) ** (
        -1 / p.sigma
    )

    # Gather args for function call
    args = (
        r_splus1,
        w_splus1,
        p_tilde_splus1,
        p_tilde_s,
        b_splus1,
        n_splus1_policy,
        c_splus1_policy,
        factor,
        rho_s,
        etr_params[t + 1, s + 1, :],
        mtry_params[t + 1, s + 1, :],
        j,
        t + 1,
        p.e[t, s + 1, j],
        z_index,
        p,
        "TPI",
    )

    return args, expected_c


@pytest.mark.parametrize(
    "setup_params",
    [{"taxes": False}, {"taxes": True}],
    ids=["No Taxes", "With Capital/Wealth Taxes"],
)
def test_c_from_b_splus1(setup_params):
    """
    Test of the household_stoch.c_from_b_splus1 function.
    """
    args, expected_c = setup_c_from_b_splus1_params(**setup_params)

    test_c = household.c_from_b_splus1(*args)

    assert np.allclose(test_c, expected_c)


# %% Tests for FOC_labor and get_y


def test_get_y_stoch():
    """
    Test of the household_stoch.get_y() function.
    """
    # Setup parameters
    p = default_specs()
    p.S = 3
    p.J = 1
    p.T = 1
    # Set a 3D e-path as the function expects
    p.e = np.array([[[1.0], [1.5], [0.2]]])

    # Test case variables
    r_p = np.array([0.05, 0.04, 0.09])
    w = np.array([1.2, 0.8, 2.5])
    b_s = np.array([0.5, 0.99, 9])
    n = np.array([0.8, 3.2, 0.2])
    # Case 1: Scalar productivity shock
    z1 = 1.1
    # Case 2: Vector of productivity shocks
    z2 = np.array([0.9, 1.0, 1.2])

    # Manually calculate expected income
    e_ss = np.squeeze(p.e[-1, :, :])
    expected_y1 = r_p * b_s + w * e_ss * z1 * n
    expected_y2 = r_p * b_s + w * e_ss * z2 * n

    # Run tests
    test_y1 = household.get_y(r_p, w, b_s, n, z1, p, "SS")
    test_y2 = household.get_y(r_p, w, b_s, n, z2, p, "SS")

    assert np.allclose(test_y1, expected_y1)
    assert np.allclose(test_y2, expected_y2)


def _foc_setup():
    p = default_specs()
    p.sigma = 1.5
    p.g_y = 0.04
    p.b_ellipse = 0.527
    p.upsilon = 1.45
    p.ltilde = 1.2
    p.J = 1
    p.S = 3
    p.T = 3
    p.chi_n = np.array([0.75, 0.8, 0.9]).reshape(3, 1)
    p.e = np.tile(np.array([1.0, 0.9, 1.4]).reshape(1, p.S, p.J), (p.T, 1, 1))
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T + p.S, p.J))
    p.tau_payroll = np.full((p.T + p.S, p.J), 0.15)
    mtrx_params = np.zeros((p.S, 12))
    mtrx_params[:, 10] = 0.22  # Simple 22% MTR on labor income
    return p, np.zeros((p.S, 12)), mtrx_params


@pytest.mark.parametrize(
    "z",
    [1.1, np.array([0.9, 1.0, 1.2])],
    ids=["SS with scalar z", "SS with vector z"],
)
def test_FOC_labor_stoch(z):
    """
    Test of the household_stoch.FOC_labor() function against the FOC
    written out by hand.
    """
    p, etr_params, mtrx_params = _foc_setup()
    r, w, p_tilde, factor, j = 0.05, 1.2, 1.0, 1.0, 0
    b = np.array([0.0, 0.8, 0.5])
    n = np.array([0.9, 0.8, 0.5])
    c = np.array([1.1, 0.9, 1.3])
    e = np.squeeze(p.e[-1, :, j])
    mtrx = tax.MTR_income(
        r,
        w,
        b,
        n,
        factor,
        False,
        e * z,
        etr_params,
        mtrx_params,
        0.0,
        p.income_tax_filer[-1, j],
        p,
    )
    expected = household.marg_ut_cons(c, p.sigma) / p_tilde * w * (
        1 - p.tau_payroll[-1, j] - mtrx
    ) * e * z - household.marg_ut_labor(n, np.squeeze(p.chi_n), p)
    test_error = household.FOC_labor(
        r,
        w,
        p_tilde,
        b,
        c,
        n,
        factor,
        e,
        z,
        np.squeeze(p.chi_n),
        etr_params,
        mtrx_params,
        None,
        j,
        p,
        "SS",
    )
    assert np.allclose(test_error, expected)


# %% Test for BC_residual


@pytest.mark.parametrize("vector", [False, True], ids=["Scalar", "Vector"])
def test_BC_residual(vector):
    """
    BC_residual is p_tilde times (consumption the budget allows - c): zero
    at the budget, positive when consuming less.
    """
    p = default_specs()
    p.g_y = 0.02
    p.c_min = np.array([0.02])
    if vector:
        r, w, p_tilde = np.array([0.04, 0.05]), np.array([1.5, 1.6]), 1.1
        b, n = np.array([10.0, 12.0]), np.array([0.8, 0.7])
        b_splus1 = np.array([11.0, 13.0])
        bq, rm = np.array([0.5, 0.6]), np.array([0.01, 0.02])
        net_tax = np.array([2.0, 2.5])
        e, z = np.array([1.0, 1.1]), np.array([1.2, 0.9])
    else:
        r, w, p_tilde = 0.04, 1.5, 1.0
        b, n, b_splus1 = 10.0, 0.8, 11.0
        bq, rm, net_tax, e, z = 0.5, 0.01, 2.0, 1.0, 1.2
    p_i, tau_c = np.array([1.0]), np.array([0.1])
    resources = (
        (1 + r) * b
        + w * e * z * n
        + bq
        + rm
        - net_tax
        - np.exp(p.g_y) * b_splus1
        - 1.1 * 0.02
    )
    c_budget = resources / p_tilde
    args = (b, b_splus1, r, w, p_tilde, p_i, e, z, bq, rm, net_tax, tau_c)

    def resid(c):
        return household.BC_residual(c, n, *args, p)

    assert np.allclose(resid(c_budget), 0.0)
    assert np.allclose(resid(c_budget - 1.0), p_tilde * 1.0)
    assert np.allclose(resid(c_budget + 0.5), -p_tilde * 0.5)


# %% EOL_system and HH_system


def _small_household():
    p = default_specs()
    p.sigma, p.g_y, p.ltilde = 1.5, 0.02, 1.2
    p.b_ellipse, p.upsilon = 0.527, 1.45
    p.J, p.S, p.T = 1, 3, 3
    p.chi_n = np.array([0.75, 0.8, 0.9]).reshape(3, 1)
    p.e = np.tile(np.array([1.0, 0.9, 1.4]).reshape(1, p.S, p.J), (p.T, 1, 1))
    p.chi_b = np.array([2.5])
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.tau_payroll = np.full((p.T, p.J), 0.1)
    p.c_min = np.array([0.01])
    etr_params = np.zeros(12)
    mtrx_params = np.zeros(12)
    mtrx_params[10] = 0.20  # flat 20% MTR on labor
    return p, etr_params, mtrx_params


@pytest.mark.parametrize("vector", [False, True], ids=["Scalar", "Vector"])
def test_EOL_system(vector):
    """
    EOL_system: c from the labor FOC, bequest from the end-of-life Euler
    equation, residual of the full budget constraint.
    """
    p, etr_params, mtrx_params = _small_household()
    r, w, p_tilde, factor, j = 0.04, 1.5, 1.0, 1.0, 0
    tr, ubi, bq, pension, rm = 0.1, 0.02, 0.2, 0.3, 0.05
    e, chi_n = p.e[-1, -1, j], p.chi_n[-1, j]
    if vector:
        n, b, z = (
            np.array([0.4, 0.6]),
            np.array([7.0, 9.0]),
            np.array([0.9, 1.2]),
        )
    else:
        n, b, z = 0.5, 8.0, 1.1
    n = np.atleast_1d(n)
    c = household.c_from_n(
        n,
        b,
        p_tilde,
        r,
        w,
        factor,
        e,
        z,
        chi_n,
        etr_params,
        mtrx_params,
        None,
        j,
        p,
        "SS",
    )
    b_splus1 = c * (p.chi_b[j] * p_tilde) ** (1 / p.sigma) / np.exp(p.g_y)
    net_tax = household.get_net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        pension,
        None,
        j,
        "SS",
        e * z,
        etr_params,
        p,
    )
    expected = (
        (1 + r) * b
        + w * e * z * n
        + bq
        + rm
        - net_tax
        - np.exp(p.g_y) * b_splus1
        - (1 + p.tau_c[-1, 0]) * 0.01
        - p_tilde * c
    )
    test = household.EOL_system(
        n,
        b,
        p_tilde,
        r,
        w,
        tr,
        ubi,
        bq,
        pension,
        factor,
        e,
        z,
        chi_n,
        etr_params,
        mtrx_params,
        None,
        j,
        p,
        "SS",
        rm=rm,
    )
    assert np.allclose(test, expected)
    if not vector:
        assert isinstance(test, float)


def test_HH_system():
    """HH_system returns the budget and labor-FOC residuals at (b, n)."""
    p, etr_params, mtrx_params = _small_household()
    r, w, p_tilde, factor, j = 0.05, 1.2, 1.0, 1.0, 0
    tr, ubi, bq, pension, rm = 0.0, 0.0, 0.1, 0.1, 0.03
    e, chi_n, z = p.e[-1, 1, j], p.chi_n[1, j], 1.1
    b, n, c, b_splus1 = 8.0, 0.7, 7.5, 8.2
    net_tax = household.get_net_taxes(
        r,
        w,
        b,
        np.atleast_1d(n),
        bq,
        factor,
        tr,
        ubi,
        pension,
        None,
        j,
        "SS",
        e * z,
        etr_params,
        p,
    )
    bc = household.BC_residual(
        c,
        n,
        b,
        b_splus1,
        r,
        w,
        p_tilde,
        np.ones(1),
        e,
        z,
        bq,
        rm,
        net_tax,
        p.tau_c[-1, :],
        p,
    )
    foc = household.FOC_labor(
        r,
        w,
        p_tilde,
        b,
        c,
        n,
        factor,
        e,
        z,
        chi_n,
        etr_params,
        mtrx_params,
        None,
        j,
        p,
        "SS",
    )
    test = household.HH_system(
        np.array([b, n]),
        c,
        b_splus1,
        r,
        w,
        p_tilde,
        factor,
        tr,
        ubi,
        bq,
        pension,
        e,
        z,
        chi_n,
        etr_params,
        mtrx_params,
        j,
        None,
        p,
        "SS",
        rm=rm,
    )
    assert np.allclose(test, [np.squeeze(bc), np.squeeze(foc)])


# %% Test for solve_HH


def _solve_hh_problem():
    """Small life-cycle problem: S = 4, one type, two shock states."""
    p = set_shocks(default_specs(), [0.8, 1.2], [[0.9, 0.1], [0.1, 0.9]])
    p.S, p.J, p.T = 4, 1, 4
    p.beta = np.array([0.96])
    p.sigma = 2.0
    p.g_y = 0.01
    p.chi_b = np.array([2.0])
    p.ltilde = 1.0
    p.b_ellipse = 5.0
    p.upsilon = 2.0
    p.retire = np.array([p.S + 1])
    p.e = np.tile(np.linspace(1.0, 1.2, p.S).reshape(1, p.S, 1), (p.T, 1, 1))
    p.chi_n = np.tile(
        np.array([0.5, 0.6, 0.7, 0.8]).reshape(1, p.S, 1), (p.T, 1, 1)
    )
    p.rho = np.array([0.1, 0.1, 0.1, 1.0])
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.capital_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.h_wealth = np.zeros(p.T)
    p.m_wealth = np.ones(p.T)
    p.p_wealth = np.zeros(p.T)
    p.tau_payroll = np.full((p.T, p.J), 0.05)
    etr = np.zeros((p.S, 12))
    mtrx = np.zeros((p.S, 12))
    mtry = np.zeros((p.S, 12))
    mtrx[:, 10] = 0.20  # 20% MTR on labor
    mtry[:, 10] = 0.10  # 10% MTR on capital
    S = p.S
    args = {
        "r": np.ones(S) * 0.04,
        "w": np.ones(S) * 1.2,
        "p_tilde": np.ones(S),
        "factor": 1.0,
        "tr": np.zeros(S),
        "bq": np.zeros(S),
        "ubi": 0.0,
        "b_grid": np.linspace(0.0, 20, 100),
        "sigma": p.sigma,
        "pension": np.zeros(S),
        "chi_n": p.chi_n[-1, :, 0],
        "rho": p.rho,
        "e": p.e[-1, :, 0],
        "etr_params": etr,
        "mtrx_params": mtrx,
        "mtry_params": mtry,
        "j": 0,
        "t": np.zeros(S, dtype=int),
        "p": p,
        "method": "SS",
    }
    return p, args


@pytest.fixture(scope="module")
def solve_hh_result():
    p, args = _solve_hh_problem()
    b_policy, c_policy, n_policy, diag = household.solve_HH(
        **args, return_diagnostics=True
    )
    return p, args, b_policy, c_policy, n_policy, diag


def test_solve_HH(solve_hh_result):
    """
    The single-type solver returns policies with the right shapes that
    satisfy the constraints, are monotone in assets, and at every grid
    node satisfy the budget constraint and labor FOC exactly.
    """
    p, args, b_policy, c_policy, n_policy, diag = solve_hh_result
    b_grid = args["b_grid"]
    nb = len(b_grid)

    expected_shape = (p.S, nb, p.nz)
    for pol in (b_policy, c_policy, n_policy):
        assert isinstance(pol, np.ndarray)
        assert pol.shape == expected_shape
        assert np.isfinite(pol).all()
    assert diag.static_failures == 0 and diag.egm_failures == 0

    assert np.all(c_policy > 0)
    assert np.all(b_policy >= b_grid[0])  # borrowing limit
    assert np.all(n_policy > 0)
    assert np.all(n_policy < p.ltilde)

    # savings and consumption increase with assets
    assert np.all(np.diff(b_policy, axis=1) > 0)
    assert np.all(np.diff(c_policy, axis=1) > 0)

    # budget constraint and labor FOC at every node
    for s in range(p.S):
        for iz, z in enumerate(p.z_grid):
            n = n_policy[s, :, iz]
            c = c_policy[s, :, iz]
            net_tax = household.get_net_taxes(
                args["r"][s],
                args["w"][s],
                b_grid,
                n,
                args["bq"][s],
                args["factor"],
                args["tr"][s],
                args["ubi"],
                args["pension"][s],
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
                b_grid,
                b_policy[s, :, iz],
                args["r"][s],
                args["w"][s],
                args["p_tilde"][s],
                np.ones(1),
                args["e"][s],
                z,
                args["bq"][s],
                0.0,
                net_tax,
                p.tau_c[-1, :],
                p,
            )
            foc = household.FOC_labor(
                args["r"][s],
                args["w"][s],
                args["p_tilde"][s],
                b_grid,
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
            assert np.abs(bc).max() < 1e-10
            assert np.abs(foc).max() < 1e-8


def test_solve_HH_end_of_life(solve_hh_result):
    """At the last age, b' = (chi_b p_tilde)^(1/sigma) c e^{-g_y}."""
    p, args, b_policy, c_policy, n_policy, _ = solve_hh_result
    expected = household.b_from_c_EOL(c_policy[-1], 1.0, 0, p.sigma, p)
    assert np.allclose(b_policy[-1], expected, rtol=1e-12)


# %%
# One type of OG-Core's J = 2 test calibration


def test_ogcore_HH_soln():
    """
    Compare the stochastic solver (one shock state) with OG-Core's
    SS.inner_loop for type j = 0 of OG-Core's J = 2 test calibration.
    """
    p = ogcore_test_calibration()
    p.e = np.ones((p.T + p.S, p.S, p.J))
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.capital_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.eta = np.zeros((p.T + p.S, p.S, p.J))
    p.eta_RM = np.zeros((p.T + p.S, p.S, p.J))
    p.ubi_nom_array = np.zeros((p.T + p.S, p.S, p.J))
    p.PIA_rate_bkt_1 = 0.0
    p.PIA_rate_bkt_2 = 0.0
    p.PIA_rate_bkt_3 = 0.0
    r_p, w, factor = 0.05, 1.2, 100_000
    BQ = np.ones(p.J) * 0.02
    TR = np.array([0.01])
    tr = hh_core.get_tr(TR, None, p, "SS")
    bq = hh_core.get_bq(BQ, None, p, "SS")

    b_grid = household.make_asset_grid(7.0, 50, 2.0)
    b_policy, c_policy, n_policy = household.solve_HH(
        np.ones(p.S) * r_p,
        np.ones(p.S) * w,
        np.ones(p.S),  # p_tilde
        factor,
        tr[:, 0],
        bq[:, 0],
        0.0,  # ubi
        b_grid,
        p.sigma,
        0.0,  # pension
        p.chi_n[-1, :],
        p.rho[-1, :, 0],
        np.ones(p.S),  # e
        np.array(p.etr_params)[-1, :, :],
        np.array(p.mtrx_params)[-1, :, :],
        np.array(p.mtry_params)[-1, :, :],
        0,
        "unused",
        p,
        "SS",
    )

    # OG-Core benchmark, warm-started from the stochastic solver's
    # profile (from b = 0.05, n = 0.5 inner_loop does not converge here)
    b_guess, n_guess = simulate_profiles(
        b_policy[:, :, 0], n_policy[:, :, 0], b_grid
    )
    bssmat = np.tile(b_guess.reshape(p.S, 1), (1, p.J))
    nssmat = np.tile(n_guess.reshape(p.S, 1), (1, p.J))
    outer_loop_vars = (
        bssmat,
        nssmat,
        r_p,
        r_p,
        w,
        np.array([1.0]),
        1.4,  # Y (only enters remittances, which are zero)
        BQ,
        float(TR[0]),
        0.0,
        factor,
    )
    euler, b_core, n_core = SS.inner_loop(outer_loop_vars, p, None)[:3]
    assert_ogcore_converged(euler, js=[0])
    b_s = np.zeros(p.S)
    b_s[1:] = b_core[:-1, 0]
    taxes = household.get_net_taxes(
        r_p,
        w,
        b_s,
        n_core[:, 0],
        bq[:, 0],
        factor,
        tr[:, 0],
        0.0,
        0.0,
        None,
        0,
        "SS",
        np.ones(p.S),
        np.array(p.etr_params)[-1],
        p,
    )
    c_core = hh_core.get_cons(
        r_p,
        w,
        1.0,
        np.ones(1),
        b_s,
        b_core[:, 0],
        n_core[:, 0],
        bq[:, 0],
        0.0,
        taxes,
        np.ones(p.S),
        p.tau_c[-1, :],
        p,
        "SS",
    )

    # stochastic policies at OG-Core's asset holdings
    def at_core(policy):
        return np.array(
            [
                itp.PchipInterpolator(b_grid, policy[s, :, 0])(b_s[s])
                for s in range(p.S)
            ]
        )

    assert np.allclose(at_core(b_policy), b_core[:, 0], atol=1e-3)
    assert np.allclose(at_core(n_policy), n_core[:, 0], atol=1e-3)
    assert np.allclose(at_core(c_policy), c_core, atol=1e-3)


# %%
# Steady-state replication of OG-Core for all J = 7 types (one shock state)

REPLICATION_ATOL = 1e-4
R_SS, W_SS, FACTOR_SS = 0.05, 1.2, 100_000
BQ_SS = np.ones(7) * 0.02
TR_SS = np.array([0.01])
Y_SS = 1.4


def _replication_specs(pensions_on, full_budget=False):
    """
    Default calibration (one shock state). Without pensions and with
    no transfers, remittances or UBI; with
    pensions, transfers at their defaults; ``full_budget`` adds minimum
    consumption, a consumption tax, remittances, UBI, wealth, bequest
    and payroll taxes (payroll taxes higher for one type), a non-filer
    type and noncompliance.
    """
    p = default_specs()
    if not pensions_on:
        p.eta = np.zeros((p.T + p.S, p.S, p.J))
        p.eta_RM = np.zeros((p.T + p.S, p.S, p.J))
        p.ubi_nom_array = np.zeros((p.T + p.S, p.S, p.J))
        p.retirement_age = np.array([p.S + p.E + 1])
        p.retire = np.array([p.S + p.E + 1])
        p.PIA_rate_bkt_1 = 0.0
        p.PIA_rate_bkt_2 = 0.0
        p.PIA_rate_bkt_3 = 0.0
    if full_budget:
        T, S, J = p.T, p.S, p.J
        p.c_min = np.array([0.01])
        p.tau_c = np.full((T + S, p.I), 0.05)
        p.alpha_RM_T = 0.01
        p.ubi_nom_array = np.full((T + S, S, J), 2000.0)
        p.h_wealth = np.full(T + S, 0.5)
        p.m_wealth = np.full(T + S, 1.0)
        p.p_wealth = np.full(T + S, 0.01)
        p.tau_bq = np.full(T + S, 0.2)
        p.tau_payroll = np.full((T + S, J), 0.03)
        p.tau_payroll[:, 4] = 0.06  # by type since OG-Core 0.20.4
        p.income_tax_filer = np.ones((T + S, J))
        p.income_tax_filer[:, 0] = 0.0
        p.wealth_tax_filer = np.ones((T + S, J))
        p.wealth_tax_filer[:, 1] = 0.0
        p.labor_income_tax_noncompliance_rate = np.zeros((T + S, J))
        p.labor_income_tax_noncompliance_rate[:, 2] = 0.1
        p.capital_income_tax_noncompliance_rate = np.zeros((T + S, J))
        p.capital_income_tax_noncompliance_rate[:, 3] = 0.2
    return p


def _solve_and_compare(pensions_on, full_budget=False, nb=None):
    """
    Solve the stochastic model and OG-Core's SS.inner_loop at the same
    prices; return everything needed to compare them.
    """
    p = _replication_specs(pensions_on, full_budget)
    if nb is not None:
        p.nb = nb
    options = household.SolverOptions()
    theta = None if pensions_on else np.zeros(p.J)
    sol = household.solve_all_households_ss(
        R_SS,
        W_SS,
        [1.0],
        BQ_SS,
        TR_SS,
        FACTOR_SS,
        p,
        Y=Y_SS,
        theta=theta,
        options=options,
    )
    # OG-Core benchmark, warm-started from the stochastic solver's profiles
    # (cold-started, inner_loop fails for j = 1 and j = 6)
    bssmat = np.zeros((p.S, p.J))
    nssmat = np.zeros((p.S, p.J))
    for j in range(p.J):
        bssmat[:, j], nssmat[:, j] = simulate_profiles(
            sol.b_policy[:, j, :, 0], sol.n_policy[:, j, :, 0], sol.b_grid[j]
        )
    euler, b_core, n_core = solve_ogcore_benchmark(
        p, bssmat, nssmat, R_SS, W_SS, BQ_SS, TR_SS, FACTOR_SS, Y_SS
    )
    theta_core = None
    if pensions_on:
        # OG-Core's inner_loop makes theta consistent with its own labor
        # supply; compare policies at OG-Core's theta
        theta_core = pensions.replacement_rate_vals(
            n_core, W_SS, FACTOR_SS, None, p
        )
        sol_endogenous = sol
        sol = household.solve_all_households_ss(
            R_SS,
            W_SS,
            [1.0],
            BQ_SS,
            TR_SS,
            FACTOR_SS,
            p,
            Y=Y_SS,
            theta=theta_core,
            b_grid=sol.b_grid,
            options=options,
        )
    else:
        sol_endogenous = None
    b_s = np.vstack([np.zeros((1, p.J)), b_core[:-1, :]])

    # stochastic policies at OG-Core's asset holdings
    def at_core(policy):
        out = np.zeros((p.S, p.J))
        for j in range(p.J):
            for s in range(p.S):
                out[s, j] = itp.PchipInterpolator(
                    sol.b_grid[j], policy[s, j, :, 0]
                )(b_s[s, j])
        return out

    return {
        "p": p,
        "sol": sol,
        "sol_endogenous": sol_endogenous,
        "euler_core": euler,
        "b_core": b_core,
        "n_core": n_core,
        "b_s_core": b_s,
        "theta_core": theta_core,
        "b_interpolated": at_core(sol.b_policy),
        "n_interpolated": at_core(sol.n_policy),
    }


@pytest.fixture(scope="module")
def ss_comparison_setup():
    """Pensions, transfers, remittances and UBI off."""
    return _solve_and_compare(pensions_on=False)


@pytest.fixture(scope="module")
def ss_comparison_pensions():
    """Default calibration: pensions and transfers on."""
    return _solve_and_compare(pensions_on=True)


@pytest.fixture(scope="module")
def ss_comparison_full_budget():
    """Every term of OG-Core's budget constraint active."""
    return _solve_and_compare(pensions_on=True, full_budget=True)


def assert_replicates(setup, js=None, atol=REPLICATION_ATOL):
    """
    Assert the stochastic solver's policies, evaluated at OG-Core's asset
    holdings, match OG-Core's b and n profiles for types ``js``.
    """
    p = setup["p"]
    js = list(range(p.J)) if js is None else js
    assert_ogcore_converged(setup["euler_core"], js=js)
    print("\n--- Model Comparison Diagnostics ---")
    for var in ("b", "n"):
        diff = np.abs(setup[f"{var}_interpolated"] - setup[f"{var}_core"])
        diff = diff[:, js]
        s_max, j_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(
            f"Max difference in {var}: {diff.max():.2e} "
            f"at (s, j) = ({s_max}, {js[j_idx]})"
        )
        assert np.allclose(
            setup[f"{var}_interpolated"][:, js],
            setup[f"{var}_core"][:, js],
            atol=atol,
        ), f"{var} profiles differ from OG-Core by more than {atol}"


def test_solve_all_households_ss_shapes(ss_comparison_setup):
    """
    The policy functions from solve_all_households_ss have the right
    dimensions, and each type got its own asset grid.
    """
    sol = ss_comparison_setup["sol"]
    p = ss_comparison_setup["p"]
    nb = sol.b_grid.shape[1]
    expected_shape = (p.S, p.J, nb, p.nz)
    assert sol.b_policy.shape == expected_shape
    assert sol.c_policy.shape == expected_shape
    assert sol.n_policy.shape == expected_shape
    assert sol.dist.shape == expected_shape
    assert sol.b_grid.shape == (p.J, nb)
    assert (sol.b_grid[:, 0] == 0).all()
    # j = 6 saves most: its grid reaches furthest
    assert np.argmax(sol.b_grid[:, -1]) == p.J - 1


def test_ogcore_benchmark_converged(ss_comparison_setup):
    """
    The OG-Core benchmark is only meaningful if SS.inner_loop converged:
    check its Euler errors for every type.
    """
    assert_ogcore_converged(ss_comparison_setup["euler_core"])


def test_solve_all_households_replication(ss_comparison_setup):
    """
    With one shock state the stochastic solver replicates OG-Core's
    SS.inner_loop for all seven types, including j = 6 (which saves far
    more than the others, so a grid common to all types would stop below
    its savings).
    """
    assert_replicates(ss_comparison_setup)


def test_replication_with_pensions(ss_comparison_pensions):
    """
    Default calibration with US-style Social Security and transfers:
    policies match OG-Core given OG-Core's replacement rates, and the
    replacement rates the solver makes consistent with its own mean
    earnings match OG-Core's.
    """
    setup = ss_comparison_pensions
    assert_replicates(setup)
    theta_own = setup["sol_endogenous"].theta
    assert setup["sol_endogenous"].iterations["pension"] > 0
    assert np.allclose(theta_own, setup["theta_core"], rtol=1e-4)
    # pensions are paid from the retirement age on
    pension = setup["sol"].pension
    retire = setup["p"].retire[-1]
    assert (pension[:retire] == 0).all() and (pension[retire:] > 0).all()


def test_replication_full_budget(ss_comparison_full_budget):
    """
    Every term of OG-Core's budget constraint at once (minimum
    consumption, consumption tax, remittances, UBI, wealth, bequest and
    payroll taxes, filer status, noncompliance, pensions).
    """
    assert_replicates(ss_comparison_full_budget)


def test_replication_improves_with_grid(ss_comparison_setup):
    """
    The gap to OG-Core shrinks as the asset grid is refined.
    """
    base = ss_comparison_setup
    gaps = []
    for nb in (50, 100, 200):
        p = copy.deepcopy(base["p"])
        p.nb = nb
        sol = household.solve_all_households_ss(
            R_SS,
            W_SS,
            [1.0],
            BQ_SS,
            TR_SS,
            FACTOR_SS,
            p,
            Y=Y_SS,
            theta=np.zeros(p.J),
        )
        gap = 0.0
        for j in range(p.J):
            for s in range(p.S):
                b_int = itp.PchipInterpolator(
                    sol.b_grid[j], sol.b_policy[s, j, :, 0]
                )(base["b_s_core"][s, j])
                gap = max(gap, abs(b_int - base["b_core"][s, j]))
        gaps.append(gap)
    print("max |b gap| for nb = 50, 100, 200:", gaps)
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[1] < REPLICATION_ATOL
