"""
End-to-end checks against OG-Core's own first-order conditions for
calibrations the replication tests in hh_test.py do not cover:
analytical MTRs, the GS, HSV and DEP_totalinc tax functions, time
endowments other than 1, fractional filers with noncompliance and a
wealth tax, and the numpy backend on DEP taxes.

With one shock state, each type's life-cycle profile from b = 0 under
the solver's policies is plugged into OG-Core's
``SS.euler_equation_solver``, whose savings and labor FOC residuals,
normalized by marginal utilities, must be small. Also here: pensions
other than US-style Social Security, per type against OG-Core, and the
pension fixed point for each system.
"""

import numpy as np
import pytest
import scipy.interpolate as itp
from ogcore import SS, pensions
from ogcore import household as hh_core

from ogcore import household_stoch as household
from stoch_test_utils import default_specs, set_ar1_shocks

R, W, FACTOR = 0.05, 1.2, 100_000
BQ = np.ones(7) * 0.02
TR = 0.01
FOC_TOL = 1e-4  # normalized OG-Core FOC residuals along the profile


def _one_state_specs():
    """Default calibration, one shock state, pensions off."""
    p = default_specs()
    # euler_equation_solver recomputes US-style benefits from the
    # profile it is given; with zero PIA rates they are zero, as here
    p.retirement_age = np.array([p.S + p.E + 1])
    p.retire = np.array([p.S + p.E + 1])
    p.PIA_rate_bkt_1 = p.PIA_rate_bkt_2 = p.PIA_rate_bkt_3 = 0.0
    return p


def _profile(sol, j):
    """Savings, labor and consumption along type j's life from b = 0."""
    grid = sol.b_grid[j]
    S = sol.b_policy.shape[0]
    b = 0.0
    b_next, n, c = np.zeros(S), np.zeros(S), np.zeros(S)
    for s in range(S):
        n[s] = itp.PchipInterpolator(grid, sol.n_policy[s, j, :, 0])(b)
        c[s] = itp.PchipInterpolator(grid, sol.c_policy[s, j, :, 0])(b)
        b_next[s] = itp.PchipInterpolator(grid, sol.b_policy[s, j, :, 0])(b)
        b = b_next[s]
    return b_next, n, c


def _ogcore_foc_residuals(p, sol, j):
    """
    OG-Core's savings and labor FOC residuals along type j's profile,
    relative to the marginal utility of consumption (divided by
    p_tilde) and the marginal disutility of labor.
    """
    hh = household.household_inputs_ss(p, R, W, [1.0], BQ, TR, FACTOR)
    b_next, n, c = _profile(sol, j)
    errors = SS.euler_equation_solver(
        np.append(b_next, n),
        R,
        W,
        hh["p_tilde"],
        hh["p_i"],
        hh["bq"][:, j],
        hh["rm"][:, j],
        hh["tr"][:, j],
        hh["ubi"][:, j],
        FACTOR,
        j,
        p,
    )
    S = p.S
    mu = hh_core.marg_ut_cons(c, p.sigma) / hh["p_tilde"]
    mdu = hh_core.marg_ut_labor(n, p.chi_n[-1], p)
    return np.abs(errors[:S] / mu).max(), np.abs(errors[S:] / mdu).max()


def _set_tax_functions(p, tax_func_type, params, analytical):
    params = np.asarray(params, dtype=float)
    etr = np.tile(params, (p.T + p.S, p.S, 1))
    p.tax_func_type = tax_func_type
    p.etr_params = etr
    p.mtrx_params = etr * 1.0
    p.mtry_params = etr * 1.0
    if tax_func_type == "GS":
        p.mtrx_params[..., 0] *= 1.05  # MTRs differ from the ETR's form
        p.mtry_params[..., 0] *= 0.95
    p.analytical_mtrs = analytical


def _variant(name):
    p = _one_state_specs()
    options = None
    T, S, J = p.T, p.S, p.J
    if name == "analytical MTRs":
        p.analytical_mtrs = True
    elif name.startswith("ltilde"):
        p.ltilde = float(name.split()[-1])
    elif name == "filers, noncompliance, wealth tax":
        p.income_tax_filer = np.full((T + S, J), 0.6)
        p.wealth_tax_filer = np.full((T + S, J), 0.5)
        p.h_wealth = np.full(T + S, 0.5)
        p.m_wealth = np.full(T + S, 1.0)
        p.p_wealth = np.full(T + S, 0.01)
        p.labor_income_tax_noncompliance_rate = np.full((T + S, J), 0.1)
        p.capital_income_tax_noncompliance_rate = np.full((T + S, J), 0.2)
    elif name == "numpy backend":
        options = household.SolverOptions(backend="numpy")
    elif name != "DEP":
        form, analytical = name.split()
        params = {
            **TAX_PARAMS,
            "DEP_totalinc": [2e-11, 3e-6, 0.30, 0.05, 0.01, -0.02],
        }[form]
        _set_tax_functions(p, form, params, analytical == "analytical")
    return p, options


VARIANTS = [
    "DEP",
    "analytical MTRs",
    "ltilde 0.8",
    "ltilde 1.3",
    "filers, noncompliance, wealth tax",
    "GS estimated",
    "GS analytical",
    "HSV estimated",
    "HSV analytical",
    "DEP_totalinc estimated",
    "DEP_totalinc analytical",
    "numpy backend",
]


@pytest.mark.parametrize("name", VARIANTS)
def test_ogcore_focs_hold(name):
    """
    For every variant the solver converges without failures, and
    OG-Core's own FOCs hold along the solver's life-cycle profiles.
    """
    p, options = _variant(name)
    sol = household.solve_all_households_ss(
        R, W, [1.0], BQ, TR, FACTOR, p, theta=np.zeros(p.J), options=options
    )
    d = sol.diagnostics
    assert d.static_failures == 0 and d.egm_failures == 0
    assert d.euler_failures == 0
    if name == "numpy backend":
        assert d.backend == "numpy"
    worst = {j: _ogcore_foc_residuals(p, sol, j) for j in (0, 3, 6)}
    print(
        name,
        {j: f"savings {a:.1e}, labor {b:.1e}" for j, (a, b) in worst.items()},
    )
    for j, (savings, labor) in worst.items():
        assert savings < FOC_TOL and labor < FOC_TOL, (j, savings, labor)


TAX_PARAMS = {
    "GS": [0.35, 0.9, 2e-5],
    "HSV": [5.0, 0.15],
}


@pytest.mark.parametrize(
    "form,analytical",
    [("GS", False), ("HSV", True)],
    ids=["GS", "HSV analytical"],
)
def test_five_states_other_tax_functions(form, analytical):
    """
    Five shock states with other tax functions: no failures, Euler
    errors between nodes below 1e-4 wherever households are.
    """
    p = set_ar1_shocks(default_specs(), 5, 0.9, 0.15)
    _set_tax_functions(p, form, TAX_PARAMS[form], analytical)
    sol = household.solve_all_households_ss(
        R, W, [1.0], BQ, TR, FACTOR, p, theta=np.full(p.J, 0.3)
    )
    d = sol.diagnostics
    print(d.summary())
    assert d.static_failures == d.egm_failures == d.euler_failures == 0
    errors = household.euler_errors(sol)
    print({k: v for k, v in errors.items() if k not in ("errors", "b")})
    assert errors["nan_with_mass"] == 0
    assert errors["max_with_mass"] < 1e-4


@pytest.mark.parametrize("form", ["GS", "HSV"])
def test_numpy_backend_zero_income(form):
    """
    GS and HSV are singular at zero income, which households with no
    assets reach when the labor floor is 0 (upsilon < 2). The numpy
    backend evaluates them at a minimum income, as the kernels do, and
    the two backends agree.
    """
    p = set_ar1_shocks(default_specs(), 3, 0.9, 0.15)
    p.upsilon = 1.5
    _set_tax_functions(p, form, TAX_PARAMS[form], False)
    hh = household.household_inputs_ss(p, R, W, [1.0], BQ, TR, FACTOR)
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
        0.0,
        hh["c_min_exp"],
        js=[0, 3],
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


# %%
# Pension systems other than US-style Social Security


PENSION_SYSTEMS = [
    "Defined Benefits",
    "Notional Defined Contribution",
    "Points System",
]


def _pension_specs(system, nz=3):
    p = set_ar1_shocks(default_specs(), nz, 0.9, 0.15)
    p.pension_system = system
    # benefit parameters are zero by default
    # alpha_db is T+S x J since OG-Core 0.21.0; let it differ by type
    p.alpha_db = np.tile(np.linspace(0.012, 0.018, p.J), (p.T + p.S, 1))
    p.tau_p = 0.2
    p.vpoint = 0.4
    return p


@pytest.mark.parametrize("system", PENSION_SYSTEMS)
def test_pension_ss_matches_ogcore_by_type(system):
    """
    Benefits for all types at once equal OG-Core's pension_amount for
    each type separately, on a labor profile that differs by type.
    """
    p = _pension_specs(system)
    n_eff = np.tile(np.linspace(0.3, 0.45, p.S).reshape(p.S, 1), (1, p.J))
    n_eff *= np.linspace(0.9, 1.2, p.J)[None, :]
    pension = household.pension_ss(p, W, FACTOR, n_eff=n_eff, r=R)
    for j in range(p.J):
        expected = pensions.pension_amount(
            R,
            W,
            n_eff[:, j],
            1,
            None,
            None,
            j,
            False,
            "SS",
            p.e[-1, :, j],
            FACTOR,
            p,
        )
        assert np.allclose(pension[:, j], expected, rtol=1e-12, atol=1e-14)
    assert np.allclose(pension[: p.retire[-1]], 0.0)
    assert (pension[p.retire[-1] :] > 0).all()
    with pytest.raises(ValueError):
        household.pension_ss(p, W, FACTOR)


@pytest.mark.parametrize("system", PENSION_SYSTEMS)
def test_pension_fixed_point_other_systems(system):
    """
    With benefits left out, they are made consistent with each type's
    mean effective labor in the solution.
    """
    p = _pension_specs(system)
    sol = household.solve_all_households_ss(R, W, [1.0], BQ, TR, FACTOR, p)
    n_eff = household.mean_effective_labor(sol.dist, sol.n_policy, sol.z_grid)
    implied = household.pension_ss(p, W, FACTOR, n_eff=n_eff, r=R)
    assert sol.iterations["pension"] >= 2
    assert np.abs(implied - sol.pension).max() < 1e-9
    assert (sol.pension[p.retire[-1] :] > 0).all()
