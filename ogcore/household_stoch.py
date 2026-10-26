"""
------------------------------------------------------------------------
Household functions with idiosyncratic income shocks.

Households of lifetime-income type j and age s face an idiosyncratic
productivity shock z that follows a discrete Markov chain, on top of
OG-Core's deterministic ability profile e_{j,s}: labor income is
w e_{j,s} z n. With one shock state (z = 1) the model is OG-Core's.

Contents
    * shock process (Rouwenhorst discretization of an AR(1))
    * pointwise household functions (first-order conditions, budget
      constraint) that mirror ogcore.household; the solver's kernels are
      tested against them
    * inputs by age and type: prices, bequests, transfers, remittances,
      pensions, taxes (``LifecycleInputs``)
    * a batched endogenous-grid solver with an explicit borrowing-
      constrained region (numba kernels in household_stoch_kernels)
    * the population-weighted stationary distribution (Young's method),
      theta from mean earnings, and accuracy diagnostics

The shock process is set by the parameters ``nz``, ``z_persistence``
and ``z_innovation_sd`` (or ``z_grid`` and ``z_trans`` directly); see
``shock_process``, which ``Specifications`` calls to build ``p.z_grid``,
``p.z_trans`` and ``p.z_dist0``.

Functions that are unchanged from OG-Core are imported from
``ogcore.household`` rather than copied, so they track OG-Core master.
They are re-exported here so ``household_stoch.<name>`` keeps working.
------------------------------------------------------------------------
"""

# Packages
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as sps
from ogcore import aggregates as aggr
from ogcore import household as hh_core
from ogcore import pensions, tax, utils
from ogcore.household import (  # noqa: F401 (re-exported)
    constraint_checker_SS,
    constraint_checker_TPI,
    get_bq,
    get_ci,
    get_rm,
    get_tr,
    marg_ut_cons,
    marg_ut_labor,
)

from ogcore import household_stoch_kernels as kernels

logger = logging.getLogger(__name__)

# Names this module takes unchanged from ogcore.household
OGCORE_HOUSEHOLD_FUNCTIONS = (
    "constraint_checker_SS",
    "constraint_checker_TPI",
    "get_bq",
    "get_ci",
    "get_rm",
    "get_tr",
    "marg_ut_cons",
    "marg_ut_labor",
)

"""
------------------------------------------------------------------------
    Shock process
------------------------------------------------------------------------
"""


def stationary_distribution(z_trans):
    """
    Stationary distribution of a Markov chain.

    Args:
        z_trans (array_like): transition matrix, (nz, nz), rows sum to 1

    Returns:
        pi (Numpy array): stationary distribution, (nz,)
    """
    z_trans = np.atleast_2d(np.asarray(z_trans, dtype=float))
    nz = z_trans.shape[0]
    if nz == 1:
        return np.ones(1)
    # pi (P - I) = 0 and sum(pi) = 1
    A = np.vstack([z_trans.T - np.eye(nz), np.ones((1, nz))])
    rhs = np.zeros(nz + 1)
    rhs[-1] = 1.0
    pi = np.linalg.lstsq(A, rhs, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def rouwenhorst(nz, rho, sigma, mean_one=True):
    r"""
    Rouwenhorst discretization of an AR(1) process for log productivity,

    .. math::
        \log z' = \rho \log z + \varepsilon,\quad
        \varepsilon \sim N(0, \sigma^2),

    following Kopecky and Suen (2010). The log grid is evenly spaced on
    :math:`\pm\sqrt{n_z - 1}\,\sigma_z` with
    :math:`\sigma_z = \sigma / \sqrt{1 - \rho^2}`; the chain matches the
    AR(1)'s autocorrelation and unconditional variance exactly, and its
    stationary distribution is Binomial(nz - 1, 1/2).

    Args:
        nz (int): number of states
        rho (float): persistence per model period, in (-1, 1)
        sigma (float): standard deviation of the innovation per model
            period
        mean_one (bool): scale levels so that E[z] = 1 under the
            stationary distribution, which keeps the meaning of e

    Returns:
        z_grid (Numpy array): productivity levels, (nz,)
        z_trans (Numpy array): transition matrix, (nz, nz)
        pi (Numpy array): stationary distribution, (nz,)
    """
    nz = int(nz)
    if nz < 1:
        raise ValueError("nz must be at least 1")
    if not -1 < rho < 1:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if sigma < 0:
        raise ValueError(f"sigma must be nonnegative, got {sigma}")
    if nz == 1:
        return np.ones(1), np.ones((1, 1)), np.ones(1)
    q = (1 + rho) / 2
    P = np.array([[q, 1 - q], [1 - q, q]])
    for n in range(3, nz + 1):
        P_new = np.zeros((n, n))
        P_new[:-1, :-1] += q * P
        P_new[:-1, 1:] += (1 - q) * P
        P_new[1:, :-1] += (1 - q) * P
        P_new[1:, 1:] += q * P
        P_new[1:-1, :] /= 2
        P = P_new
    sigma_z = sigma / np.sqrt(1 - rho**2)
    psi = np.sqrt(nz - 1) * sigma_z
    log_z = np.linspace(-psi, psi, nz)
    pi = sps.binom.pmf(np.arange(nz), nz - 1, 0.5)
    z_grid = np.exp(log_z)
    if mean_one:
        z_grid = z_grid / (pi @ z_grid)
    return z_grid, P, pi


def shock_process(nz, rho, sigma, z_grid=None, z_trans=None):
    r"""
    Discrete Markov process for idiosyncratic productivity shocks
    :math:`z`: ``z_grid`` and ``z_trans`` as given (a direct override),
    or else the Rouwenhorst discretization of an AR(1) for log z with
    E[z] = 1 (see ``rouwenhorst``). Newborns draw z from the stationary
    distribution.

    ``Specifications.compute_default_params`` calls this to set
    ``p.z_grid``, ``p.z_trans`` and ``p.z_dist0`` from the parameters
    ``nz``, ``z_persistence``, ``z_innovation_sd``, ``z_grid`` and
    ``z_trans``. The transition matrix is ``z_trans`` because ``p.Z``
    is total factor productivity in OG-Core.

    Args:
        nz (int): number of shock states
        rho (float): persistence of log z per model period
        sigma (float): innovation standard deviation per model period
        z_grid (array_like): productivity levels, (nz,); None or empty
            for the AR(1)
        z_trans (array_like): Markov transition matrix, (nz, nz), with
            ``z_trans[i, k]`` = Pr(z' = z_grid[k] | z = z_grid[i]);
            given together with ``z_grid``

    Returns:
        z_grid (Numpy array): productivity levels, (nz,)
        z_trans (Numpy array): transition matrix, (nz, nz)
        z_dist0 (Numpy array): shock distribution of newborns, (nz,)

    Raises:
        ValueError: if only one of ``z_grid`` and ``z_trans`` is given,
            if shapes do not match ``nz`` or if rows of ``z_trans`` are
            not probabilities
    """
    nz = int(nz)
    given = [x is not None and np.size(x) > 0 for x in (z_grid, z_trans)]
    if not any(given):
        return rouwenhorst(nz, rho, sigma)
    if not all(given):
        raise ValueError("z_grid and z_trans must be given together")
    z_grid = np.atleast_1d(np.asarray(z_grid, dtype=float))
    z_trans = np.atleast_2d(np.asarray(z_trans, dtype=float))
    if z_grid.shape != (nz,):
        raise ValueError(
            f"z_grid must have shape ({nz},) for nz = {nz}, got {z_grid.shape}"
        )
    if (z_grid <= 0).any():
        raise ValueError("Productivity levels must be positive")
    if z_trans.shape != (nz, nz):
        raise ValueError(
            f"z_trans must have shape ({nz}, {nz}), got {z_trans.shape}"
        )
    if (z_trans < 0).any() or not np.allclose(z_trans.sum(axis=1), 1.0):
        raise ValueError("Rows of z_trans must be probability vectors")
    return z_grid, z_trans, stationary_distribution(z_trans)


def _shock_process(p):
    """(z_grid, z_trans, z_dist0) of p as arrays."""
    return (
        np.asarray(p.z_grid, dtype=float),
        np.asarray(p.z_trans, dtype=float),
        np.asarray(p.z_dist0, dtype=float),
    )


"""
------------------------------------------------------------------------
    Pointwise household functions

    These mirror ogcore.household with labor productivity e z instead of
    e. The batched solver does not call them; tests use them to check
    its kernels.
------------------------------------------------------------------------
"""


def marg_ut_beq(b, sigma, j, p):
    r"""
    Compute the marginal utility of savings.

    .. math::
        MU_{b} = \chi^b_{j}b_{j,s,t}^{-\sigma}

    Stitched linearly below b = 0.001 so that it is finite at the
    borrowing limit b = 0.

    Args:
        b (array_like): household savings
        sigma (float): coefficient of relative risk aversion
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        output (array_like): marginal utility of savings

    """
    if np.ndim(b) == 0:
        b = np.array([b])
    epsilon = 0.001
    bvec_cnstr = b < epsilon
    MU_b = np.zeros(b.shape)
    MU_b[~bvec_cnstr] = p.chi_b[j] * b[~bvec_cnstr] ** (-sigma)

    # This is the linear approximation of b**(-sigma)
    b2 = -sigma * (epsilon ** (-sigma - 1))
    b1 = (epsilon ** (-sigma)) - b2 * epsilon
    MU_b[bvec_cnstr] = p.chi_b[j] * (b2 * b[bvec_cnstr] + b1)

    output = MU_b
    output = np.squeeze(output)
    return output


def inv_mu_c(value, sigma):
    r"""
    Compute the inverse of the marginal utility of consumption.

    .. math::
       c = value^{-1/\sigma}

    This is the exact inverse of ``ogcore.household.marg_ut_cons``,
    including its linear stitching for c < 0.003 (marginal utilities
    above 0.003^{-sigma}).

    Args:
        value (array_like): marginal utility of consumption
        sigma (scalar): coefficient of relative risk aversion

    Returns:
        output (array_like): household consumption

    """
    output = kernels.inv_mu_c_np(np.atleast_1d(value), sigma)
    return np.squeeze(output)


def _tax_index(method, t):
    """Time index of policy parameters: -1 in the SS, t otherwise."""
    if method == "SS":
        return -1
    if method == "TPI_scalar":
        return 0
    return t


def c_from_n(
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
    t,
    j,
    p,
    method,
):
    r"""
    Calculate household consumption from labor supply Euler equation for
    group j.

    .. math::
        c_{j,s,t} = \left[ \frac{\tilde{p}_t \chi_s^n h'(n_{j,s,t})}{
        w_t e_{j, s}z_{j, s}(1 - \tau^{p}_{j,t} - \tau^{mtrx}_{s,t})}
        \right]^{-1/\sigma}

    Args:
        n (array_like): household labor supply
        b (array_like): household savings
        p_tilde (array_like): composite good price
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        factor (scalar): scaling factor converting model units to dollars
        e (array_like): effective labor units (deterministic)
        z (array_like): productivity (stochastic)
        chi_n (array_like): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate
            functions
        mtrx_params (list): parameters of the marginal tax rate
            on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        c (array_like): consumption implied by labor choice
    """
    idx = _tax_index(method, t)
    tau_payroll = p.tau_payroll[idx, j]
    tax_noncompliance = p.labor_income_tax_noncompliance_rate[idx, j]
    income_tax_filer = p.income_tax_filer[idx, j]
    if e is None:
        e = np.squeeze(p.e[idx, :, j])

    deriv = (
        1
        - tau_payroll
        - tax.MTR_income(
            r,
            w,
            b,
            n,
            factor,
            False,
            e * z,
            etr_params,
            mtrx_params,
            tax_noncompliance,
            income_tax_filer,
            p,
        )
    )
    numerator = p_tilde * marg_ut_labor(n, chi_n, p)
    denominator = w * e * z * deriv
    c = inv_mu_c(numerator / denominator, p.sigma)

    return c


def b_from_c_EOL(c, p_tilde, j, sigma, p, rho=1.0):
    r"""
    Calculate household bequests at the end of life from the savings Euler
    equation.

    .. math::
        b_{j, E+S+1, t+1} =
        [\rho_{E+S}\chi_j^b \tilde p_t]^{\frac{1}{\sigma}} c_{j, E+S, t}
        e^{-g_y}

    Args:
        c (array_like): household consumption
        p_tilde (array_like): composite good price
        j (int): index of ability type
        sigma (scalar): coefficient of relative risk aversion
        p (OG-Core Specifications object): model parameters
        rho (scalar): mortality rate at the last age (1 in OG-Core)

    Returns:
        b (array_like): household savings at the end of life
    """
    b = (c * (rho * p.chi_b[j] * p_tilde) ** (1 / sigma)) / np.exp(p.g_y)
    return b


def get_cons(
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
    e,
    z,
    tau_c,
    p,
    method="SS",
):
    r"""
    Calculate household composite consumption: OG-Core's budget
    constraint (``ogcore.household.get_cons``) with labor productivity
    :math:`e z`.

    .. math::
        c_{j,s,t} = \biggl[(1 + r_{t})b_{j,s,t} + w_t e_{j,s} z n_{j,s,t}
        + bq_{j,s,t} + rm_{j,s,t} - T_{j,s,t}
        - \sum_i (1 + \tau^c_{i,t}) p_{i,t} c_{min,i}
        - e^{g_y}b_{j,s+1,t+1}\biggr] / \tilde{p}_t

    where net taxes :math:`T` are net of transfers, UBI and pensions
    (see ``get_net_taxes``).

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        p_i (Numpy array): prices of consumption goods
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        rm (Numpy array): household remittances received
        net_tax (Numpy array): household net taxes paid
        e (Numpy array): effective labor units
        z (array_like): labor productivity
        tau_c (Numpy array): consumption tax rates
        p (OG-Core Specifications object): model parameters
        method (str): 'SS' or 'TPI', passed to OG-Core's get_cons

    Returns:
        cons (Numpy array): household consumption

    """
    return hh_core.get_cons(
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
        np.multiply(e, z),
        tau_c,
        p,
        method,
    )


def get_net_taxes(
    r, w, b, n, bq, factor, tr, ubi, pension, t, j, method, e, etr_params, p
):
    """
    Net taxes paid by a household of type j: income and payroll taxes,
    bequest and wealth taxes, less transfers, UBI and pension benefits.

    Same as ``ogcore.tax.net_taxes`` for one type, except that pension
    benefits are passed in (by age, computed outside the household's
    grid; see ``pension_ss``) instead of being placed by array position
    from theta.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        bq (Numpy array): bequests received
        factor (scalar): scaling factor converting model units to dollars
        tr (Numpy array): government transfers
        ubi (Numpy array): UBI payments
        pension (Numpy array): pension benefits
        t (int): time period (ignored in the SS)
        j (int): index of lifetime income group
        method (str): 'SS', 'TPI' (scalar t) or 'TPI_scalar'
        e (Numpy array): effective labor units times productivity
        etr_params (list): effective tax rate function parameters
        p (OG-Core Specifications object): model parameters

    Returns:
        net_tax (Numpy array): net taxes paid
    """
    idx = _tax_index(method, t)
    labor_income = w * e * n
    T_I = (
        tax.ETR_income(
            r,
            w,
            b,
            n,
            factor,
            e,
            etr_params,
            p.labor_income_tax_noncompliance_rate[idx, j],
            p.capital_income_tax_noncompliance_rate[idx, j],
            p.income_tax_filer[idx, j],
            p,
        )
        * (r * b + labor_income)
        + p.tau_payroll[idx, j] * labor_income
    )
    T_BQ = p.tau_bq[idx] * bq
    T_W = (
        tax.ETR_wealth(
            b,
            p.h_wealth[idx],
            p.m_wealth[idx],
            p.p_wealth[idx],
            p.wealth_tax_filer[idx, j],
        )
        * b
    )
    return T_I + T_BQ + T_W - pension - tr - ubi


def c_from_b_splus1(
    r_splus1,
    w_splus1,
    p_tilde_splus1,
    p_tilde,
    b_splus1,
    n_splus1_policy,
    c_splus1_policy,
    factor,
    rho,
    etr_params,
    mtry_params,
    j,
    t,
    e_splus1,
    z_index,
    p,
    method,
):
    r"""
    Calculate household consumption in period s from assets at period s+1 using
    the savings Euler equation.

    .. math::
        \frac{c_{j,s,t}^{-\sigma}}{\tilde{p}_t} = e^{-\sigma g_y}
        \biggl[\chi^b_j\rho_s(b_{j,s+1,t+1})^{-\sigma} +
        \beta_j\bigl(1 - \rho_s\bigr)\mathbb{E}\Bigl[\frac{1 + r_{t+1}
        \bigl[1 - \tau^{mtry}_{s+1,t+1}\bigr] - \tau^{mtrw}_{s+1,t+1}}
        {\tilde{p}_{t+1}}(c_{j,s+1,t+1})^{-\sigma}\Bigr]\biggr]

    Args:
        r_splus1 (scalar): the real interest rate next period
        w_splus1 (scalar): the real wage rate next period
        p_tilde_splus1 (scalar): composite good price next period
        p_tilde (scalar): composite good price
        b_splus1 (array_like): household savings one period ahead
        n_splus1_policy (array_like): household labor supply one period
            ahead across b, z
        c_splus1_policy (array_like): household consumption one period
            ahead across b, z
        factor (scalar): scaling factor converting model units to dollars
        rho (array_like): mortality rates
        etr_params (list): parameters of the effective tax rate
            functions
        mtry_params (list): parameters of the marginal tax rate
            on capital income functions
        j (int): index of ability type
        t (int): model period
        e_splus1 (array_like): effective labor units one period ahead
        z_index (array_like): index in productivity grid
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    returns:
        c (array_like): household consumption in current period
    """
    beta = p.beta[j]
    idx = _tax_index(method, t)
    tax_noncompliance = p.capital_income_tax_noncompliance_rate[idx, j]
    income_tax_filer = p.income_tax_filer[idx, j]
    wealth_tax_filer = p.wealth_tax_filer[idx, j]
    h_wealth = p.h_wealth[idx]
    m_wealth = p.m_wealth[idx]
    p_wealth = p.p_wealth[idx]

    bequest_utility = rho * marg_ut_beq(b_splus1, p.sigma, j, p)

    # Expected discounted marginal utility of consumption next period;
    # n_splus1_policy must be [nb, nz]
    if n_splus1_policy.ndim < 2:
        n_splus1_policy = np.tile(n_splus1_policy.reshape(-1, 1), (1, p.nz))
    consumption_utility_matrix = np.zeros(
        (b_splus1.shape[0], p.z_grid.shape[0])
    )
    for zp_index, zp in enumerate(p.z_grid):
        deriv = (
            (1 + r_splus1)
            - (
                r_splus1
                * tax.MTR_income(
                    r_splus1,
                    w_splus1,
                    b_splus1,
                    n_splus1_policy[:, zp_index],
                    factor,
                    True,
                    e_splus1 * zp,
                    etr_params,
                    mtry_params,
                    tax_noncompliance,
                    income_tax_filer,
                    p,
                )
            )
            - tax.MTR_wealth(
                b_splus1, h_wealth, m_wealth, p_wealth, wealth_tax_filer
            )
        )
        consumption_utility_matrix[:, zp_index] = (
            deriv
            * marg_ut_cons(c_splus1_policy[:, zp_index], p.sigma)
            / p_tilde_splus1
        )

    prob_z_splus1 = p.z_trans[z_index, :]
    E_MU_c = consumption_utility_matrix @ prob_z_splus1

    # Combine bequest utility and expectation of future consumption utility
    mu_c_rhs = bequest_utility + beta * (1 - rho) * E_MU_c
    c = inv_mu_c(p_tilde * np.exp(-p.sigma * p.g_y) * mu_c_rhs, p.sigma)

    return c


def FOC_labor(
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
    t,
    j,
    p,
    method,
):
    r"""
    Computes errors for the FOC for labor supply.

    .. math::
        w_t z e_{j,s}\bigl(1 - \tau^p_{j,t} - \tau^{mtrx}_{s,t}\bigr)
       \frac{(c_{j,s,t})^{-\sigma}}{ \tilde{p}_{t}} = \chi^n_{s}
        \biggl(\frac{b}{\tilde{l}}\biggr)\biggl(\frac{n_{j,s,t}}
        {\tilde{l}}\biggr)^{\upsilon-1}\Biggl[1 -
        \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon\Biggr]
        ^{\frac{1-\upsilon}{\upsilon}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        b (Numpy array): household savings
        c (Numpy array): household consumption
        n (Numpy array): household labor supply
        factor (scalar): scaling factor converting model units to dollars
        e (Numpy array): effective labor units
        z (array_like): productivity
        chi_n (Numpy array): utility weight on the disutility of labor
            supply
        etr_params (list): parameters of the effective tax rate
            functions
        mtrx_params (list): parameters of the marginal tax rate
            on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): 'SS', 'TPI' (scalar t) or 'TPI_scalar'

    Returns:
        FOC_error (Numpy array): error from FOC for labor supply

    """
    idx = _tax_index(method, t)
    tau_payroll = p.tau_payroll[idx, j]
    tax_noncompliance = p.labor_income_tax_noncompliance_rate[idx, j]
    income_tax_filer = p.income_tax_filer[idx, j]

    deriv = (
        1
        - tau_payroll
        - tax.MTR_income(
            r,
            w,
            b,
            n,
            factor,
            False,
            e * z,
            etr_params,
            mtrx_params,
            tax_noncompliance,
            income_tax_filer,
            p,
        )
    )
    FOC_error = marg_ut_cons(c, p.sigma) * (
        1 / p_tilde
    ) * w * deriv * e * z - marg_ut_labor(n, chi_n, p)

    return FOC_error


def get_y(r_p, w, b_s, n, z, p, method):
    r"""
    Compute household income before taxes.

    .. math::
        y_{j,s,t} = r_{p,t}b_{j,s,t} + w_{t}e_{j,s}z n_{j,s,t}

    Args:
        r_p (array_like): real interest rate on the household portfolio
        w (array_like): real wage rate
        b_s (Numpy array): household savings coming into the period
        n (Numpy array): household labor supply
        z (array_like): productivity
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
    """
    if method == "SS":
        e = np.squeeze(p.e[-1, :, :])
    elif method == "TPI":
        e = p.e
    y = r_p * b_s + w * e * z * n

    return y


def BC_residual(
    c,
    n,
    b,
    b_splus1,
    r,
    w,
    p_tilde,
    p_i,
    e,
    z,
    bq,
    rm,
    net_tax,
    tau_c,
    p,
    method="SS",
):
    r"""
    Compute the residuals of the household budget constraint,
    :math:`\tilde{p}_t` times (consumption the budget allows less c).

    .. math::
        (1 + r_{t})b_{j,s,t} + w_{t}e_{j,s}z n_{j,s,t} + bq_{j,s,t}
        + rm_{j,s,t} - T_{j,s,t} - \sum_i(1+\tau^c_i)p_i c_{min,i}
        - e^{g_y}b_{j,s+1,t+1} - \tilde{p}_t c_{j,s,t}
    """
    cons = get_cons(
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
        e,
        z,
        tau_c,
        p,
        method,
    )
    return p_tilde * (cons - c)


def EOL_system(
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
    t,
    j,
    p,
    method,
    rm=0.0,
    p_i=None,
    tau_c=None,
    rho=1.0,
):
    r"""
    Compute the residuals of the household budget constraint at the end of
    life given a guess for labor supply. Solve first for consumption given
    labor supply and then for savings given consumption. Then check the
    budget constraint.

    Args:
        n (array_like): household labor supply
        b (array_like): household savings
        p_tilde (array_like): composite good price
        r (scalar): the real interest rate
        w (scalar): the real wage rate
        tr (scalar): transfers
        ubi (scalar): UBI
        bq (scalar): bequests received
        pension (scalar): pension benefits
        factor (scalar): scaling factor converting model units to dollars
        e (scalar): effective labor units
        z (scalar): productivity
        chi_n (scalar): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate functions
        mtrx_params (list): parameters of the marginal tax rate on labor
            income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
        rm (scalar): remittances received
        p_i (array_like): prices of consumption goods (ones if None)
        tau_c (array_like): consumption tax rates (SS values if None)
        rho (scalar): mortality rate at the last age

    Returns:
        BC_error (float or array_like): residuals of the household budget
            constraint; a Python float when there is a single residual,
            as ``scipy.optimize.brentq`` requires
    """
    n = np.atleast_1d(n)
    p_i = np.ones(p.I) if p_i is None else p_i
    tau_c = p.tau_c[-1, :] if tau_c is None else tau_c

    c = c_from_n(
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
        t,
        j,
        p,
        method,
    )
    # use consumption to get savings from savings Euler equation
    b_splus1 = b_from_c_EOL(c, p_tilde, j, p.sigma, p, rho)
    net_tax = get_net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        pension,
        t,
        j,
        method,
        e * z,
        etr_params,
        p,
    )
    BC_error = BC_residual(
        c,
        n,
        b,
        b_splus1,
        r,
        w,
        p_tilde,
        p_i,
        e,
        z,
        bq,
        rm,
        net_tax,
        tau_c,
        p,
    )
    # brentq needs a scalar; numpy >= 2.3 will not convert a 1-element
    # array implicitly
    if np.size(BC_error) == 1:
        return float(np.squeeze(BC_error))
    return BC_error


def HH_system(
    x,
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
    t,
    p,
    method,
    rm=0.0,
    p_i=None,
    tau_c=None,
):
    r"""
    Compute the residuals of the household budget constraint and labor
    supply Euler equation given a guess of household assets and labor
    choice: the system the endogenous-grid step solves at each point.

    Args:
        x (array_like): vector containing household assets b and labor supply n
        c (array_like): household consumption
        b_splus1 (array_like): household savings one period ahead
        r (scalar): the real interest rate
        w (scalar): the real wage rate
        p_tilde (scalar): composite good price
        factor (scalar): scaling factor converting model units to dollars
        tr, ubi, bq, pension (scalar): transfers, UBI, bequests, pension
        e (scalar): effective labor units
        z (scalar): productivity
        chi_n (scalar): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate functions
        mtrx_params (list): parameters of the marginal tax rate on labor
            income functions
        j (int): index of ability type
        t (int): model period
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
        rm (scalar): remittances received
        p_i (array_like): prices of consumption goods (ones if None)
        tau_c (array_like): consumption tax rates (SS values if None)

    Returns:
        HH_error (array_like): residuals of the household budget
            constraint and labor supply Euler equation
    """
    b = x[0]
    n = np.atleast_1d(x[1])
    p_i = np.ones(p.I) if p_i is None else p_i
    tau_c = p.tau_c[-1, :] if tau_c is None else tau_c
    net_tax = get_net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        pension,
        t,
        j,
        method,
        e * z,
        etr_params,
        p,
    )
    BC_error = BC_residual(
        c,
        n,
        b,
        b_splus1,
        r,
        w,
        p_tilde,
        p_i,
        e,
        z,
        bq,
        rm,
        net_tax,
        tau_c,
        p,
    )
    FOC_error = FOC_labor(
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
        t,
        j,
        p,
        method,
    )
    BC_error, FOC_error = np.broadcast_arrays(BC_error, FOC_error)
    return np.squeeze(np.array([BC_error, FOC_error]))


"""
------------------------------------------------------------------------
    Inputs by age and type
------------------------------------------------------------------------
"""


@dataclass
class LifecycleInputs:
    """
    What households of a batch of M types face over their lives, by
    age s = 0, ..., S - 1. In the steady state prices and policies are
    constant; for a cohort on the transition path they are the values
    along its diagonal.

    Arrays by age are (S,), by age and type (S, M), by type (M,). Tax
    function parameters are by age: (S, K) arrays, or sequences of
    parameter objects for nonparametric ("mono") tax functions.
    """

    js: np.ndarray
    # by age
    r: np.ndarray
    w: np.ndarray
    p_tilde: np.ndarray
    chi_n: np.ndarray
    h_wealth: np.ndarray
    m_wealth: np.ndarray
    p_wealth: np.ndarray
    c_min_exp: np.ndarray
    etr_params: object
    mtrx_params: object
    mtry_params: object
    # by age and type
    e: np.ndarray
    rho: np.ndarray
    tau_payroll: np.ndarray
    bq: np.ndarray
    rm: np.ndarray
    tr: np.ndarray
    ubi: np.ndarray
    pension: np.ndarray
    bq_tax: np.ndarray
    income_filer: np.ndarray
    wealth_filer: np.ndarray
    labor_nc: np.ndarray
    capital_nc: np.ndarray
    # by type
    beta: np.ndarray
    chi_b: np.ndarray
    # scalars
    factor: float
    sigma: float
    g_y: float
    ltilde: float
    b_ellipse: float
    upsilon: float
    tax_func_type: str
    analytical_mtrs: bool

    @property
    def S(self):
        return self.e.shape[0]

    @property
    def M(self):
        return self.e.shape[1]

    def other_income(self):
        """
        Income other than labor and capital income, net of the bequest
        tax and minimum consumption expenditure, (S, M):
        bq + rm + tr + ubi + pension - tau_bq bq - c_min expenditure.
        """
        return (
            self.bq
            + self.rm
            + self.tr
            + self.ubi
            + self.pension
            - self.bq_tax
            - self.c_min_exp[:, None]
        )

    def select(self, idx):
        """Inputs for the types at positions ``idx`` of this batch."""
        idx = np.atleast_1d(idx)
        out = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if name in ("beta", "chi_b", "js"):
                val = val[idx]
            elif name in _BY_AGE_TYPE:
                val = val[:, idx]
            out[name] = val
        return LifecycleInputs(**out)


_BY_AGE_TYPE = (
    "e",
    "rho",
    "tau_payroll",
    "bq",
    "rm",
    "tr",
    "ubi",
    "pension",
    "bq_tax",
    "income_filer",
    "wealth_filer",
    "labor_nc",
    "capital_nc",
)


def _by_age_type(x, S, J, js, name):
    """Columns js of x (a scalar, an (S,) vector or (S, J)), as (S, M)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        if x.shape != (S, J):
            raise ValueError(f"{name} must be (S, J), got {x.shape}")
        return np.ascontiguousarray(x[:, js])
    if x.ndim == 1 and x.shape == (S,):
        return np.tile(x.reshape(S, 1), (1, len(js)))
    if x.ndim == 0:
        return np.full((S, len(js)), float(x))
    raise ValueError(f"{name} has unexpected shape {x.shape}")


def _tax_params_by_age(params, S):
    """Steady-state tax function parameters by age."""
    last = params[-1]
    try:
        arr = np.asarray(last, dtype=float)
    except (TypeError, ValueError):
        return last  # nonparametric (callables)
    if arr.ndim == 1:  # one parameter vector for all ages
        arr = np.tile(arr, (S, 1))
    return np.ascontiguousarray(arr)


def lifecycle_inputs_ss(
    p,
    r,
    w,
    p_tilde,
    factor,
    bq,
    rm,
    tr,
    ubi,
    pension,
    c_min_exp=0.0,
    js=None,
):
    """
    Steady-state ``LifecycleInputs`` for types ``js``.

    Args:
        p (OG-Core Specifications object): model parameters
        r (scalar): return on household savings (r_p in OG-Core)
        w (scalar): wage
        p_tilde (scalar): composite consumption good price
        factor (scalar): model-units-to-dollars scaling factor
        bq, rm, tr, ubi, pension (array_like): bequests, remittances,
            transfers, UBI and pension benefits received, each a
            scalar, an (S,) vector or an (S, J) array
        c_min_exp (scalar): minimum consumption expenditure
            sum_i (1 + tau_c_i) p_i c_min_i
        js (array_like): types to include; all if None

    Returns:
        LifecycleInputs
    """
    S, J = p.S, p.J
    js = np.arange(J) if js is None else np.atleast_1d(js).astype(int)
    full = np.ones(S)
    bq_SJ = _by_age_type(bq, S, J, js, "bq")
    rho = p.rho[-1] if np.ndim(p.rho) == 3 else p.rho
    return LifecycleInputs(
        js=js,
        r=full * r,
        w=full * w,
        p_tilde=full * p_tilde,
        chi_n=np.asarray(p.chi_n[-1], dtype=float).reshape(S),
        h_wealth=full * p.h_wealth[-1],
        m_wealth=full * p.m_wealth[-1],
        p_wealth=full * p.p_wealth[-1],
        c_min_exp=full * c_min_exp,
        etr_params=_tax_params_by_age(p.etr_params, S),
        mtrx_params=_tax_params_by_age(p.mtrx_params, S),
        mtry_params=_tax_params_by_age(p.mtry_params, S),
        e=np.ascontiguousarray(p.e[-1][:, js]),
        rho=_by_age_type(rho, S, J, js, "rho"),
        tau_payroll=np.tile(p.tau_payroll[-1, js], (S, 1)),
        bq=bq_SJ,
        rm=_by_age_type(rm, S, J, js, "rm"),
        tr=_by_age_type(tr, S, J, js, "tr"),
        ubi=_by_age_type(ubi, S, J, js, "ubi"),
        pension=_by_age_type(pension, S, J, js, "pension"),
        bq_tax=p.tau_bq[-1] * bq_SJ,
        income_filer=np.tile(p.income_tax_filer[-1, js], (S, 1)),
        wealth_filer=np.tile(p.wealth_tax_filer[-1, js], (S, 1)),
        labor_nc=np.tile(
            p.labor_income_tax_noncompliance_rate[-1, js], (S, 1)
        ),
        capital_nc=np.tile(
            p.capital_income_tax_noncompliance_rate[-1, js], (S, 1)
        ),
        beta=np.asarray(p.beta, dtype=float)[js],
        chi_b=np.asarray(p.chi_b, dtype=float)[js],
        factor=float(factor),
        sigma=float(p.sigma),
        g_y=float(p.g_y),
        ltilde=float(p.ltilde),
        b_ellipse=float(p.b_ellipse),
        upsilon=float(p.upsilon),
        tax_func_type=p.tax_func_type,
        analytical_mtrs=bool(p.analytical_mtrs),
    )


def household_inputs_ss(p, r_p, w, p_m, BQ, TR, factor, Y=None):
    """
    Per-household prices and receipts in the steady state, computed as
    in ``ogcore.SS.inner_loop``.

    Args:
        p (OG-Core Specifications object): model parameters
        r_p (scalar): return on the household portfolio
        w (scalar): wage
        p_m (array_like): output prices by industry, (M,)
        BQ (array_like): aggregate bequests, (J,) (or scalar if
            ``p.use_zeta``)
        TR (scalar): aggregate transfers
        factor (scalar): model-units-to-dollars scaling factor
        Y (scalar): GDP, for remittances (RM = alpha_RM_T Y); remittances
            are zero if None

    Returns:
        dict with p_i, p_tilde, c_min_exp and (S, J) arrays bq, rm, tr,
        ubi
    """
    p_m = np.atleast_1d(np.asarray(p_m, dtype=float))
    p_i = np.dot(p.io_matrix, p_m)
    p_tilde = aggr.get_ptilde(p_i, p.tau_c[-1, :], p.alpha_c)
    BQ = np.asarray(BQ, dtype=float)
    bq = get_bq(BQ, None, p, "SS")
    if Y is None:
        if np.any(np.asarray(p.alpha_RM_T) != 0):
            raise ValueError("Y is needed for remittances (alpha_RM_T != 0)")
        RM = 0.0
    else:
        RM = aggr.get_RM(Y, p, "SS")
    rm = get_rm(np.asarray(RM, dtype=float), None, p, "SS")
    tr = get_tr(np.asarray(TR, dtype=float), None, p, "SS")
    ubi = p.ubi_nom_array[-1, :, :] / factor
    return {
        "p_i": p_i,
        "p_tilde": float(np.squeeze(p_tilde)),
        "c_min_exp": float(np.sum((1 + p.tau_c[-1, :]) * p_i * p.c_min)),
        "bq": np.broadcast_to(bq, (p.S, p.J)),
        "rm": np.broadcast_to(rm, (p.S, p.J)),
        "tr": np.broadcast_to(tr, (p.S, p.J)),
        "ubi": np.broadcast_to(ubi, (p.S, p.J)),
    }


def pension_ss(p, w, factor, theta=None, n_eff=None, r=None):
    """
    Steady-state pension benefits by age and type, (S, J).

    Benefits are computed once per (s, j), outside the household's grid,
    and enter the budget constraint as income the household takes as
    given (OG-Core's FOCs also ignore the effect of labor supply on
    benefits). With shocks, benefits come from each type's mean
    effective labor profile E[z n | s, j] (see ``mean_effective_labor``),
    which is exact with one shock state.

    Args:
        p (OG-Core Specifications object): model parameters
        w (scalar): wage
        factor (scalar): model-units-to-dollars scaling factor
        theta (array_like): replacement rates by type (J,), US-style
            Social Security only; computed from ``n_eff`` if None
        n_eff (array_like): mean effective labor by age and type, (S, J)
        r (scalar): interest rate (Notional Defined Contribution only)

    Returns:
        pension (Numpy array): benefits by age and type, (S, J)
    """
    e = np.asarray(p.e[-1], dtype=float).reshape(p.S, p.J)
    if p.pension_system == "US-Style Social Security":
        if theta is None:
            if n_eff is None:
                raise ValueError("pension_ss needs theta or n_eff")
            theta = pensions.replacement_rate_vals(n_eff, w, factor, None, p)
        theta = np.broadcast_to(np.asarray(theta, dtype=float), (p.J,))
        return pensions.SS_amount(
            w, np.ones((p.S, p.J)), theta, None, None, False, "SS", e, p
        )
    if n_eff is None:
        raise ValueError(f"{p.pension_system} benefits need n_eff")
    # ogcore.tax.net_taxes passes Y = 1 to pension_amount
    return pensions.pension_amount(
        r,
        w,
        np.asarray(n_eff, dtype=float),
        1,
        theta,
        None,
        None,
        False,
        "SS",
        e,
        factor,
        p,
    )


"""
------------------------------------------------------------------------
    Asset grids
------------------------------------------------------------------------
"""


def make_asset_grid(b_max, nb, curvature=4.0):
    """
    Asset grid from the borrowing limit 0 (newborns hold 0) to b_max,
    denser at low assets: b_max x**curvature for x evenly spaced on
    [0, 1].

    Returns:
        b_grid (Numpy array): (nb,)
    """
    if not b_max > 0:
        raise ValueError("b_max must be positive")
    x = np.linspace(0.0, 1.0, int(nb))
    return b_max * x**curvature


def initial_b_max(inp, z_grid, horizon=20.0):
    """
    First guess at the top of each type's asset grid: ``horizon``
    periods of the largest per-period income the type can receive.
    ``solve_all_households_ss`` then resizes the grids to the support of
    the distribution.

    Returns:
        b_max (Numpy array): (M,)
    """
    labor = (inp.w[:, None] * inp.e).max(axis=0) * np.max(z_grid)
    other = (inp.bq + inp.rm + inp.tr + inp.ubi + inp.pension).max(axis=0)
    return horizon * (labor * inp.ltilde + np.maximum(other, 0.0))


"""
------------------------------------------------------------------------
    Solver
------------------------------------------------------------------------
"""


class HouseholdSolverError(RuntimeError):
    """The household solver failed at more points than allowed."""


@dataclass
class SolverOptions:
    """
    Settings for the household solver.

    Attributes:
        backend (str): "auto" (numba kernels when the tax functions are
            parametric, else numpy with OG-Core's tax functions),
            "numba" or "numpy"
        tol (float): tolerance on the labor FOC residual, relative to the
            sum of its two terms, when solving for labor supply and
            consumption at given savings (the budget constraint holds
            exactly)
        egm_tol (float): tolerance on the scaled residuals of the labor
            FOC and budget constraint when locating endogenous-grid
            knots
        euler_tol (float): largest relative Euler equation error allowed
            at a grid node (against the Euler consumption interpolated
            in savings) before the Euler equation is solved directly at
            that node
        maxiter (int): Newton iterations per point
        max_fail_frac (float): raise ``HouseholdSolverError`` if the
            solver fails for more than this share of the population
            (``solve_all_households_ss``) or, without a distribution, of
            grid points, or at more than this share of endogenous-grid
            knots; failures are always counted and logged
        grid_resize_iter (int): maximum rounds of grid resizing
            (automatic grids, whose size, curvature and margin are the
            parameters ``nb``, ``b_grid_curvature`` and
            ``b_grid_margin``)
        mass_tol (float): share of a type's population below which an
            asset node counts as empty
    """

    backend: str = "auto"
    tol: float = 1e-12
    egm_tol: float = 1e-10
    euler_tol: float = 1e-5
    maxiter: int = 60
    max_fail_frac: float = 1e-4
    grid_resize_iter: int = 4
    mass_tol: float = 1e-10


@dataclass
class SolverDiagnostics:
    """
    Counts from one call of the solver. Failures are grid points where a
    solve did not converge; the solver logs them and raises
    ``HouseholdSolverError`` above ``SolverOptions.max_fail_frac`` (by
    population mass when the distribution is known, see
    ``solve_all_households_ss``; by count otherwise).

    Attributes:
        n_points (int): grid points solved (ages x types x nb x nz)
        n_knots (int): endogenous-grid knots (one per b' node and shock)
        egm_failures (int): knots where inverting the labor FOC and
            budget constraint failed (the knot is dropped), including
            knots where the Euler equation's right-hand side is not
            finite
        infeasible_knots (int): knots where the Euler equation implies
            c <= 0, so that b' is never chosen (small b' where the
            bequest motive is strong)
        dropped_knots (int): valid knots dropped to keep the endogenous
            grid increasing
        static_failures (int): grid points where solving the labor FOC
            and budget constraint for given savings failed or gave
            c <= 0
        euler_failures (int): grid points where the Euler equation
            error stayed above ``SolverOptions.euler_tol``
        euler_kinks (int): grid points whose savings sit where the
            Euler equation's residual jumps from + to - (the labor
            choice switches, so the value function has a kink there and
            the Euler equation holds as a pair of inequalities)
        labor_corners (int): grid points with labor supply at a bound,
            where the labor FOC holds as an inequality
        constrained (int): grid points at the borrowing limit (b' = 0)
        gap (int): grid points between the borrowing limit and the first
            endogenous-grid knot above it
        extrapolated (int): grid points above the endogenous grid
        direct (int): grid points where the Euler equation was solved
            directly (gap points and points that failed the Euler check)
        max_static_residual (float): largest relative labor FOC residual
        max_euler_residual (float): largest relative Euler equation
            error at grid nodes (against the interpolated Euler
            consumption), kinks left out
        failure_ages (list): ages with any failure
        failed_nodes (Numpy array): grid points that failed (a static or
            Euler failure), bool (S, M, nb, nz)
        failed_mass (float): share of the population at failed grid
            points, once the distribution is known (else None)
        seconds (float): wall time
        backend (str): kernels used
    """

    n_points: int = 0
    n_knots: int = 0
    egm_failures: int = 0
    infeasible_knots: int = 0
    dropped_knots: int = 0
    static_failures: int = 0
    euler_failures: int = 0
    euler_kinks: int = 0
    labor_corners: int = 0
    constrained: int = 0
    gap: int = 0
    extrapolated: int = 0
    direct: int = 0
    max_static_residual: float = 0.0
    max_euler_residual: float = 0.0
    failure_ages: list = field(default_factory=list)
    failed_nodes: np.ndarray = field(default=None, repr=False)
    failed_mass: float = None
    seconds: float = 0.0
    backend: str = ""

    def summary(self):
        mass = (
            ""
            if self.failed_mass is None
            else f" ({self.failed_mass:.1e} of the population)"
        )
        return (
            f"{self.n_points} points ({self.backend}, {self.seconds:.2f}s): "
            f"{self.static_failures} static failures, "
            f"{self.egm_failures} EGM failures, "
            f"{self.euler_failures} Euler failures{mass}; "
            f"{self.constrained} constrained, {self.labor_corners} labor "
            f"corners, {self.gap} gap, {self.extrapolated} extrapolated, "
            f"{self.direct} solved directly, {self.euler_kinks} kinks, "
            f"{self.dropped_knots} dropped knots, {self.infeasible_knots} "
            f"infeasible knots; max Euler residual "
            f"{self.max_euler_residual:.1e}"
        )

    def check(self, options):
        """
        Log failures; raise if they exceed ``options.max_fail_frac``:
        of the population if ``failed_mass`` is known, else of grid
        points (and, always, of endogenous-grid knots).
        """
        point_failures = self.static_failures + self.euler_failures
        if point_failures or self.egm_failures:
            logger.warning(
                "Household solver failures: %s; ages %s",
                self.summary(),
                self.failure_ages,
            )
        if self.failed_mass is None:
            too_many = point_failures > options.max_fail_frac * max(
                self.n_points, 1
            )
        else:
            too_many = self.failed_mass > options.max_fail_frac
        too_many |= self.egm_failures > options.max_fail_frac * max(
            self.n_knots, 1
        )
        if too_many:
            raise HouseholdSolverError(
                "Household solver failed at too many points: "
                + self.summary()
                + f"; ages {self.failure_ages}"
            )


class _NumbaBackend:
    """Per-point Newton solvers compiled with numba (parametric taxes)."""

    name = "numba"

    def __init__(self, inp):
        self.inp = inp
        self.code = kernels.TAX_FUNC_CODES[inp.tax_func_type]
        self.analytical = bool(inp.analytical_mtrs)
        self.etr = np.ascontiguousarray(inp.etr_params, dtype=float)
        self.mtrx = np.ascontiguousarray(inp.mtrx_params, dtype=float)
        self.mtry = np.ascontiguousarray(inp.mtry_params, dtype=float)

    def mtry_raw(self, s, X, Y):
        rate, _, _ = kernels.tax_rates(
            self.inp.tax_func_type,
            "mtry",
            X,
            Y,
            self.etr[s],
            self.mtry[s],
            self.analytical,
        )
        return rate

    def egm(
        self, s, feasible, b0, n0, bp, b_min, A, rhs, pa, tol, maxiter, nb, nz
    ):
        inp = self.inp
        return kernels._egm_solve(
            feasible,
            b0,
            n0,
            bp,
            b_min,
            A,
            rhs,
            pa["wez"],
            pa["oth"],
            pa["fi"],
            pa["lnc"],
            pa["knc"],
            pa["fw"],
            inp.r[s],
            inp.factor,
            pa["tp"],
            inp.h_wealth[s],
            inp.m_wealth[s],
            inp.p_wealth[s],
            inp.chi_n[s],
            inp.b_ellipse,
            inp.upsilon,
            inp.ltilde,
            self.code,
            self.analytical,
            self.etr[s],
            self.mtrx[s],
            tol,
            maxiter,
            nb,
            nz,
        )

    def static(self, s, n0, b, bp_fix, kappa, pa, tol, maxiter):
        inp = self.inp
        return kernels._static_solve(
            n0,
            b,
            bp_fix,
            kappa,
            pa["wez"],
            pa["oth"],
            pa["fi"],
            pa["lnc"],
            pa["knc"],
            pa["fw"],
            inp.r[s],
            inp.p_tilde[s],
            inp.g_y,
            inp.sigma,
            inp.factor,
            pa["tp"],
            inp.h_wealth[s],
            inp.m_wealth[s],
            inp.p_wealth[s],
            inp.chi_n[s],
            inp.b_ellipse,
            inp.upsilon,
            inp.ltilde,
            self.code,
            self.analytical,
            self.etr[s],
            self.mtrx[s],
            tol,
            maxiter,
        )

    def euler_direct(self, s, euler_c, rows, b, g, n0, pa, tol, maxiter):
        """``_solve_euler_direct`` per point, compiled."""
        inp = self.inp
        return kernels._euler_direct_solve(
            b,
            g,
            n0,
            rows,
            euler_c.x,
            euler_c.y,
            euler_c.slopes,
            pa["wez"],
            pa["oth"],
            pa["fi"],
            pa["lnc"],
            pa["knc"],
            pa["fw"],
            inp.r[s],
            inp.p_tilde[s],
            inp.g_y,
            inp.sigma,
            inp.factor,
            pa["tp"],
            inp.h_wealth[s],
            inp.m_wealth[s],
            inp.p_wealth[s],
            inp.chi_n[s],
            inp.b_ellipse,
            inp.upsilon,
            inp.ltilde,
            self.code,
            self.analytical,
            self.etr[s],
            self.mtrx[s],
            tol,
            maxiter,
        )


class _NumpyBackend:
    """
    Vectorized Newton solvers on OG-Core's own tax functions
    (``ogcore.txfunc.get_tax_rates``) with finite-difference Jacobians:
    the fallback for nonparametric tax functions or without numba.
    """

    name = "numpy"

    def __init__(self, inp):
        self.inp = inp

    def _raw(self, s, kind, X, Y):
        from ogcore.txfunc import get_tax_rates

        inp = self.inp
        if inp.tax_func_type in ("GS", "HSV"):
            # singular at zero total income: evaluate at a minimum income,
            # as the kernels do
            X = np.maximum(X, kernels._MIN_INCOME - Y)
        if kind == "etr":
            out = get_tax_rates(
                inp.etr_params[s],
                X,
                Y,
                None,
                inp.tax_func_type,
                "etr",
                for_estimation=False,
            )
        else:
            if inp.analytical_mtrs:
                params = inp.etr_params[s]
            elif kind == "mtrx":
                params = inp.mtrx_params[s]
            else:
                params = inp.mtry_params[s]
            out = get_tax_rates(
                params,
                X,
                Y,
                None,
                inp.tax_func_type,
                "mtr",
                inp.analytical_mtrs,
                kind == "mtry",
                for_estimation=False,
            )
        return np.broadcast_to(np.asarray(out, dtype=float), X.shape)

    def _raw_derivs(self, s, kind, X, Y, wrt_y=True):
        # rate and forward-difference derivatives in X (and Y)
        r0 = self._raw(s, kind, X, Y)
        h_x = 1e-6 * np.maximum(np.abs(X), 1.0)
        r_x = self._raw(s, kind, X + h_x, Y)
        if not wrt_y:
            return r0, (r_x - r0) / h_x, None
        h_y = 1e-6 * np.maximum(np.abs(Y), 1.0)
        r_y = self._raw(s, kind, X, Y + h_y)
        return r0, (r_x - r0) / h_x, (r_y - r0) / h_y

    def mtry_raw(self, s, X, Y):
        return self._raw(s, "mtry", X, Y)

    def _taxes(self, s, b, n, pa, derivs):
        # same quantities as household_stoch_kernels._taxes; derivs is
        # False, "n" (derivatives in n only) or True
        inp = self.inp
        r, f, tau_p = inp.r[s], inp.factor, pa["tp"]
        wez, fi, lnc, knc, fw = (
            pa[k] for k in ("wez", "fi", "lnc", "knc", "fw")
        )
        pos = b > 0
        X = wez * n * f
        Y = r * np.where(pos, b, 0.0) * f
        if derivs:
            wrt_y = derivs is True
            etr, e_x, e_y = self._raw_derivs(s, "etr", X, Y, wrt_y)
            mtr, m_x, m_y = self._raw_derivs(s, "mtrx", X, Y, wrt_y)
        else:
            etr = self._raw(s, "etr", X, Y)
            mtr = self._raw(s, "mtrx", X, Y)
        base = X * (1 - lnc) + Y * (1 - knc)
        T = fi * etr * base / f + tau_p * wez * n
        h_w, m_w, p_w = inp.h_wealth[s], inp.m_wealth[s], inp.p_wealth[s]
        den = np.where(pos, h_w * b + m_w, 1.0)
        bp = np.where(pos, b, 0.0)
        T = T + fw * p_w * h_w * bp * bp / den
        adj = (1 - lnc) * fi
        if not derivs:
            return T, mtr * adj
        dT_dn = (fi * (e_x * base + etr * (1 - lnc)) + tau_p) * wez
        if derivs == "n":
            return T, mtr * adj, dT_dn, m_x * adj * wez * f
        # below zero assets taxes are constant in b; at b = 0 the
        # derivative is the right derivative, as in the kernels
        nonneg = b >= 0
        dT_db = np.where(
            nonneg, fi * (e_y * base + etr * (1 - knc)) * r, 0.0
        ) + fw * p_w * h_w * bp * (h_w * bp + 2 * m_w) / (den * den)
        return (
            T,
            mtr * adj,
            dT_dn,
            dT_db,
            m_x * adj * wez * f,
            np.where(nonneg, m_y * adj * r * f, 0.0),
        )

    def _egm_eval(self, s, b, n, A, rhs, scale2, pa):
        # same as household_stoch_kernels._egm_eval, vectorized
        inp = self.inp
        r, tau_p = inp.r[s], pa["tp"]
        T, mtrx, dT_dn, dT_db, dm_dn, dm_db = self._taxes(s, b, n, pa, True)
        mdu, dmdu = kernels.mdu_n_np(
            n, inp.chi_n[s], inp.b_ellipse, inp.upsilon, inp.ltilde
        )
        f1 = 1 - tau_p - mtrx - mdu / A
        f2 = ((1 + r) * b + pa["wez"] * n + pa["oth"] - T - rhs) / scale2
        J11 = -dm_dn - dmdu / A
        J12 = -dm_db
        J21 = (pa["wez"] - dT_dn) / scale2
        J22 = (1 + r - dT_db) / scale2
        return f1, f2, J11, J12, J21, J22

    def _labor(self, s, b, n0, A, rhs, scale2, pa, tol, maxiter):
        # vectorized household_stoch_kernels._labor_given_assets
        lt = self.inp.ltilde
        P = b.size
        lo = np.zeros(P)
        hi = np.full(P, lt)
        lo_seen = np.zeros(P, dtype=bool)
        hi_seen = np.zeros(P, dtype=bool)
        n = np.clip(n0, 1e-6 * lt, (1 - 1e-6) * lt)
        ok = np.zeros(P, dtype=bool)
        done = np.zeros(P, dtype=bool)
        vals = self._egm_eval(s, b, n, A, rhs, scale2, pa)
        for it in range(maxiter):
            f1, J11 = vals[0], vals[2]
            finite = np.isfinite(f1)
            conv = ~done & finite & (np.abs(f1) < tol)
            ok |= conv
            done |= conv | ~finite
            upd = ~done
            if not upd.any():
                break
            lo = np.where(upd & (f1 > 0), n, lo)
            lo_seen |= upd & (f1 > 0)
            hi = np.where(upd & (f1 <= 0), n, hi)
            hi_seen |= upd & (f1 <= 0)
            tight = upd & (hi - lo < 1e-14 * lt)
            # a corner, or a root unless f1 jumps
            ok |= tight & (~(lo_seen & hi_seen) | (np.abs(f1) < 1e-6))
            done |= tight
            upd &= ~tight
            with np.errstate(divide="ignore", invalid="ignore"):
                n_new = np.where(J11 < 0, n - f1 / J11, np.nan)
            inside = (n_new > lo) & (n_new < hi)
            n = np.where(upd, np.where(inside, n_new, 0.5 * (lo + hi)), n)
            vals = self._egm_eval(s, b, n, A, rhs, scale2, pa)
        return n, ok, vals

    def egm(
        self, s, feasible, b0, n0, bp, b_min, A, rhs, pa, tol, maxiter, nb, nz
    ):
        """
        Vectorized household_stoch_kernels._egm_point: the bracketed
        method only, without the plain Newton stage and the knot
        continuation, so every knot starts from b0, n0. Where the
        inversion has several roots it may find another one than the
        kernels; nodes are checked against the Euler equation anyway.
        """
        r = self.inp.r[s]
        P = b0.size
        scale2 = 1.0 + np.abs(rhs)
        status = np.full(P, kernels.NOT_CONVERGED, dtype=np.int64)
        b_out = np.array(b_min, dtype=float)
        n_out = np.array(n0, dtype=float)
        res = np.full(P, np.nan)
        b = np.maximum(b0, b_min)
        n, ok, vals = self._labor(s, b, n0, A, rhs, scale2, pa, tol, maxiter)
        lo = np.array(b_min, dtype=float)
        lo_ver = np.zeros(P, dtype=bool)
        hi = np.full(P, np.inf)
        step_up = 0.1 * (1.0 + np.abs(b))
        active = np.ones(P, dtype=bool)
        for it in range(4 * maxiter):
            f1, H, J11, J12, J21, J22 = vals
            finite = np.isfinite(H) & np.isfinite(f1)
            status[active & ~finite] = kernels.BAD_VALUE
            active &= finite
            below = active & (b == b_min) & (H >= 0)
            b_out[below] = b_min[below] - H[below] * scale2[below] / (1 + r)
            n_out[below] = n[below]
            res[below] = np.abs(f1[below])
            status[below & ok] = kernels.CONVERGED
            active &= ~below
            conv = active & ok & (np.abs(H) < tol)
            raise_lo = active & (H < 0) & (b >= lo)
            lo = np.where(raise_lo, b, lo)
            lo_ver |= raise_lo
            hi = np.where(active & (H >= 0), np.minimum(hi, b), hi)
            tight = active & lo_ver & (hi - lo < 1e-15 * (1 + np.abs(lo)))
            # a root, unless H jumps there (then no knot exists)
            done = conv | (tight & ok & (np.abs(H) < 1e-8))
            b_out[done], n_out[done] = b[done], n[done]
            res[done] = np.maximum(np.abs(f1), np.abs(H))[done]
            status[done] = kernels.CONVERGED
            active &= ~(done | tight)
            if not active.any():
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                dH = J22 - J21 * J12 / J11
                b_new = np.where(dH > 0, b - H / dH, np.nan)
            outside = ~((b_new > lo) & (b_new < hi))
            to_min = outside & ~lo_ver & ~(b_new > lo)
            rest = outside & ~to_min
            bis = rest & np.isfinite(hi)
            up = rest & ~np.isfinite(hi)
            b_new = np.where(to_min, b_min, b_new)
            b_new = np.where(bis, 0.5 * (lo + hi), b_new)
            b_new = np.where(up, lo + step_up, b_new)
            step_up = np.where(up, 2 * step_up, step_up)
            b = np.where(active, b_new, b)
            n, ok, vals = self._labor(
                s, b, n, A, rhs, scale2, pa, tol, maxiter
            )
        b_out[active], n_out[active] = b[active], n[active]
        status[~feasible] = kernels.INFEASIBLE
        return b_out, n_out, status, res

    def _static_F(self, s, idx, n, b, bp_fix, kappa, pa):
        # vectorized household_stoch_kernels._static_eval at points idx
        inp = self.inp
        pai = {k: v[idx] for k, v in pa.items()}
        wez = pai["wez"]
        T, mtrx, dT_dn, dm_dn = self._taxes(s, b[idx], n, pai, "n")
        growth = np.exp(inp.g_y)
        denom = inp.p_tilde[s] + growth * kappa[idx]
        c = (
            (1 + inp.r[s]) * b[idx]
            + wez * n
            + pai["oth"]
            - T
            - growth * bp_fix[idx]
        ) / denom
        dc = (wez - dT_dn) / denom
        D = 1 - pai["tp"] - mtrx
        mu = kernels.mu_c_np(c, inp.sigma)
        dmu = kernels.mu_c_deriv_np(c, inp.sigma)
        mdu, dmdu = kernels.mdu_n_np(
            n, inp.chi_n[s], inp.b_ellipse, inp.upsilon, inp.ltilde
        )
        gain = mu * wez * D / inp.p_tilde[s]
        F = gain - mdu
        dF = (dmu * dc * D - mu * dm_dn) * wez / inp.p_tilde[s] - dmdu
        return F, dF, c, np.abs(gain) + np.abs(mdu)

    def _static_search(
        self, s, idx, n0, lo, hi, lo_seen, hi_seen, n_floor, args, tol, maxiter
    ):
        """
        Vectorized household_stoch_kernels._static_search at points idx
        (args: b, bp_fix, kappa, pa of all points); lo, hi, lo_seen,
        hi_seen are per point of idx.
        """
        lt = self.inp.ltilde
        Q = idx.size
        lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        lo_seen, hi_seen = np.array(lo_seen), np.array(hi_seen)
        n = np.minimum(
            np.maximum(n0, lo + 1e-4 * (hi - lo)), hi - 1e-4 * (hi - lo)
        )
        status = np.full(Q, kernels.NOT_CONVERGED, dtype=np.int64)
        n_out = n.copy()
        c_out = np.full(Q, np.nan)
        res = np.full(Q, np.nan)
        act = np.arange(Q)
        for it in range(maxiter):
            if act.size == 0:
                break
            na = n[act]
            F, dF, c, scale = self._static_F(s, idx[act], na, *args)
            n_out[act], c_out[act] = na, c
            with np.errstate(invalid="ignore", divide="ignore"):
                res[act] = np.abs(F) / scale
            bad = ~(np.isfinite(F) & np.isfinite(c))
            conv = ~bad & (np.abs(F) <= tol * scale)
            pos = ~bad & ~conv & (F > 0)
            neg = ~bad & ~conv & ~(F > 0)
            corner = (pos & (na >= lt)) | (neg & (na <= n_floor))
            status[act[bad]] = kernels.BAD_VALUE
            res[act[bad]] = np.nan
            status[act[conv]] = kernels.CONVERGED
            status[act[corner]] = kernels.CORNER
            res[act[corner]] = 0.0
            up = pos & ~corner
            down = neg & ~corner
            lo[act[up]], lo_seen[act[up]] = na[up], True
            hi[act[down]], hi_seen[act[down]] = na[down], True
            tight = (
                (up | down)
                & lo_seen[act]
                & hi_seen[act]
                & (hi[act] - lo[act] <= 1e-14 * lt)
            )
            status[act[tight]] = kernels.CONVERGED
            keep = (up | down) & ~tight
            if it == maxiter - 1:
                break
            act, F, dF, na = act[keep], F[keep], dF[keep], na[keep]
            with np.errstate(divide="ignore", invalid="ignore"):
                n_new = np.where(
                    (dF < 0) & np.isfinite(dF), na - F / dF, np.nan
                )
            inside = (n_new > lo[act]) & (n_new < hi[act])
            to_top = ~inside & (F > 0) & ~hi_seen[act]
            to_bottom = ~inside & ~(F > 0) & ~lo_seen[act]
            n_new = np.where(inside, n_new, 0.5 * (lo[act] + hi[act]))
            n_new = np.where(to_top, hi[act], n_new)
            n_new = np.where(to_bottom, lo[act], n_new)
            n[act] = n_new
        return n_out, c_out, status, res

    def static(self, s, n0, b, bp_fix, kappa, pa, tol, maxiter):
        """Vectorized household_stoch_kernels._static_solve."""
        inp = self.inp
        lt = inp.ltilde
        P = b.size
        n_floor = kernels.labor_floor(inp.b_ellipse, inp.upsilon, lt)
        args = (b, bp_fix, kappa, pa)
        every = np.arange(P)
        no = np.zeros(P, dtype=bool)
        n, c, st, res = self._static_search(
            s,
            every,
            np.array(n0, dtype=float),
            np.full(P, n_floor),
            np.full(P, lt),
            no,
            no,
            n_floor,
            args,
            tol,
            maxiter,
        )
        if not inp.analytical_mtrs:
            return n, c, st, res
        # several local optima: as household_stoch_kernels._static_point
        weight = 1.0 + np.exp(inp.g_y) * kappa / inp.p_tilde[s]
        prefs = (inp.sigma, inp.chi_n[s], inp.b_ellipse, inp.upsilon, lt)

        def utility(sel, n_, c_):
            return kernels.period_utility_np(n_, c_, weight[sel], *prefs)

        found = (st == kernels.CONVERGED) | (st == kernels.CORNER)
        u = np.where(found, utility(every, n, c), -np.inf)
        bottom = found & (st == kernels.CORNER) & (n <= n_floor)
        idx = np.flatnonzero(found & ~bottom)
        if idx.size:  # is not working better than the root found?
            F0, _, c0, _ = self._static_F(
                s, idx, np.full(idx.size, n_floor), *args
            )
            better = np.isfinite(F0) & np.isfinite(c0) & (F0 <= 0)
            u0 = np.where(better, utility(idx, n_floor, c0), -np.inf)
            better &= u0 > u[idx]
            j = idx[better]
            n[j], c[j], st[j], res[j] = n_floor, c0[better], kernels.CORNER, 0
            u[j] = u0[better]
        idx = np.flatnonzero(bottom)
        if idx.size:  # interior optima where F turns from + to -
            Q = idx.size
            n_prev = np.full(Q, n_floor)
            F_prev = np.full(Q, -1.0)
            c_prev = c[idx].copy()
            alive = np.ones(Q, dtype=bool)
            for k in range(1, kernels.STATIC_SCAN + 1):
                nk = n_floor + (lt - n_floor) * (k / kernels.STATIC_SCAN) ** 2
                Fk, _, ck, _ = self._static_F(s, idx, np.full(Q, nk), *args)
                alive &= np.isfinite(Fk)
                cross = np.flatnonzero(alive & (F_prev > 0) & (Fk <= 0))
                if cross.size:
                    sub = idx[cross]
                    yes = np.ones(cross.size, dtype=bool)
                    n2, c2, st2, res2 = self._static_search(
                        s,
                        sub,
                        0.5 * (n_prev[cross] + nk),
                        n_prev[cross],
                        np.full(cross.size, nk),
                        yes,
                        yes,
                        n_floor,
                        args,
                        tol,
                        maxiter,
                    )
                    u2 = np.where(
                        st2 == kernels.CONVERGED,
                        utility(sub, n2, c2),
                        -np.inf,
                    )
                    better = u2 > u[sub]
                    j = sub[better]
                    n[j], c[j], st[j] = n2[better], c2[better], st2[better]
                    res[j], u[j] = res2[better], u2[better]
                n_prev = np.where(alive, nk, n_prev)
                F_prev = np.where(alive, Fk, F_prev)
                c_prev = np.where(alive, ck, c_prev)
            top = np.flatnonzero(alive & (F_prev > 0) & (n_prev >= lt))
            if top.size:  # F > 0 up to ltilde: a corner at the top
                sub = idx[top]
                u2 = utility(sub, lt, c_prev[top])
                better = u2 > u[sub]
                j = sub[better]
                n[j], c[j], st[j], res[j] = (
                    lt,
                    c_prev[top][better],
                    kernels.CORNER,
                    0,
                )
                u[j] = u2[better]
        return n, c, st, res

    def euler_direct(self, s, euler_c, rows, b, g, n0, pa, tol, maxiter):
        """Vectorized ``kernels._euler_direct_point``."""
        X = euler_c.x
        nb = X.shape[1]
        Q = b.size
        zeros = np.zeros(Q)
        top = X[rows, -1]

        def phi(sel, bp, n_start):
            n, c, st, _ = self.static(
                s,
                n_start,
                b[sel],
                bp,
                zeros[sel],
                {k: v[sel] for k, v in pa.items()},
                tol,
                maxiter,
            )
            good = (st == kernels.CONVERGED) | (st == kernels.CORNER)
            f = np.where(good, c - euler_c(rows[sel], bp), np.nan)
            return f, n, c

        def x_at(sel, k):
            # grid node k, or beyond the top of the grid for k >= nb (up
            # to twice the top)
            inside = X[rows[sel], np.minimum(k, nb - 1)]
            return np.where(
                k < nb, inside, top[sel] * (1 + 0.25 * 2.0 ** (k - nb))
            )

        every = np.arange(Q)
        g = np.clip(g, X[rows, 0], top)
        k_lo = np.clip((X[rows] <= g[:, None]).sum(axis=1) - 1, 0, nb - 2)
        k_hi = k_lo + 1
        x_lo, x_hi = x_at(every, k_lo), x_at(every, k_hi)
        f_lo, n_lo, c_lo = phi(every, x_lo, n0)
        f_hi, n_hi, c_hi = phi(every, x_hi, n_lo)
        state = np.full(Q, _SEARCH)
        step = np.ones(Q, dtype=np.int64)
        for it in range(4 * nb):
            search = state == _SEARCH
            if not search.any():
                break
            bad = search & ~(np.isfinite(f_lo) & np.isfinite(f_hi))
            state[bad] = _FAILED
            search &= ~bad
            state[search & (f_lo > 0) & (f_hi <= 0)] = _BRACKETED
            down = search & ~(f_lo > 0)
            up = search & (f_lo > 0) & (f_hi > 0)
            state[down & (k_lo == 0)] = _AT_LIMIT
            down &= k_lo > 0
            state[up & (k_hi >= nb + 2)] = _FAILED
            up &= k_hi < nb + 2
            d = np.flatnonzero(down)
            if d.size:  # the root is below: the lower end becomes the upper
                k_hi[d], x_hi[d] = k_lo[d], x_lo[d]
                f_hi[d], n_hi[d], c_hi[d] = f_lo[d], n_lo[d], c_lo[d]
                k_lo[d] = np.maximum(k_lo[d] - step[d], 0)
                x_lo[d] = x_at(d, k_lo[d])
                f_lo[d], n_lo[d], c_lo[d] = phi(d, x_lo[d], n_hi[d])
            u = np.flatnonzero(up)
            if u.size:  # the root is above: the upper end becomes the lower
                k_lo[u], x_lo[u] = k_hi[u], x_hi[u]
                f_lo[u], n_lo[u], c_lo[u] = f_hi[u], n_hi[u], c_hi[u]
                k_new = np.where(
                    k_lo[u] >= nb - 1,
                    k_lo[u] + 1,
                    np.minimum(k_lo[u] + step[u], nb - 1),
                )
                k_hi[u] = k_new
                x_hi[u] = x_at(u, k_new)
                f_hi[u], n_hi[u], c_hi[u] = phi(u, x_hi[u], n_lo[u])
            step[down | up] *= 2

        bp_out = np.full(Q, np.nan)
        n_out = np.array(n0, dtype=float)
        outcome = np.full(Q, kernels.EULER_FAILED, dtype=np.int64)
        lim = state == _AT_LIMIT
        bp_out[lim], n_out[lim] = x_lo[lim], n_lo[lim]
        outcome[lim] = kernels.EULER_AT_LIMIT
        br = np.flatnonzero(state == _BRACKETED)
        a, fa, n_a, c_a = x_lo[br], f_lo[br], n_lo[br], c_lo[br]
        xb, fb, n_b, c_b = x_hi[br], f_hi[br], n_hi[br], c_hi[br]
        live = np.ones(br.size, dtype=bool)
        found = np.full(br.size, kernels.EULER_FAILED, dtype=np.int64)
        rtol = kernels.EULER_ROOT_TOL
        for it in range(200):
            size = np.abs(c_b)
            conv = live & (np.abs(fb) <= rtol * size)
            tight = live & (np.abs(xb - a) <= 1e-14 * (1.0 + np.abs(a)))
            small = np.abs(fb) <= 1e-8 * size
            found[conv | (tight & small)] = kernels.EULER_ROOT
            if self.inp.analytical_mtrs:
                # phi jumps from + to -: a kink of the value function;
                # keep the end of the bracket with c > 0
                jump = tight & ~conv & ~small
                use_a = jump & ~(c_b > 0) & (c_a > 0)
                xb = np.where(use_a, a, xb)
                n_b = np.where(use_a, n_a, n_b)
                found[jump & ((c_b > 0) | use_a)] = kernels.EULER_KINK
            live &= ~(conv | tight)
            L = np.flatnonzero(live)
            if L.size == 0:
                break
            x = xb[L] - fb[L] * (xb[L] - a[L]) / (fb[L] - fa[L])
            lo_end = np.minimum(a[L], xb[L])
            hi_end = np.maximum(a[L], xb[L])
            x = np.where((x > lo_end) & (x < hi_end), x, 0.5 * (a[L] + xb[L]))
            f, n_x, c_x = phi(br[L], x, n_b[L])
            nan = ~np.isfinite(f)
            live[L[nan]] = False
            L, f, x, n_x, c_x = L[~nan], f[~nan], x[~nan], n_x[~nan], c_x[~nan]
            opp = f * fb[L] < 0
            a[L] = np.where(opp, xb[L], a[L])
            fa[L] = np.where(opp, fb[L], 0.5 * fa[L])
            n_a[L] = np.where(opp, n_b[L], n_a[L])
            c_a[L] = np.where(opp, c_b[L], c_a[L])
            xb[L], fb[L], n_b[L], c_b[L] = x, f, n_x, c_x
        good = found != kernels.EULER_FAILED
        sel = br[good]
        bp_out[sel], n_out[sel], outcome[sel] = (
            xb[good],
            n_b[good],
            found[good],
        )
        return bp_out, n_out, outcome


# states of the bracket search in _NumpyBackend.euler_direct
_SEARCH, _BRACKETED, _AT_LIMIT, _FAILED = 0, 1, 2, 3


def _make_backend(inp, backend):
    parametric = inp.tax_func_type in kernels.TAX_FUNC_CODES
    if backend == "auto":
        backend = "numba" if (kernels.USE_NUMBA and parametric) else "numpy"
    if backend == "numba":
        if not parametric:
            raise ValueError(
                f"tax_func_type {inp.tax_func_type!r} needs backend='numpy'"
            )
        return _NumbaBackend(inp)
    if backend == "numpy":
        return _NumpyBackend(inp)
    raise ValueError(f"Unknown backend {backend!r}")


def _flat(x, shape):
    return np.ascontiguousarray(np.broadcast_to(x, shape), dtype=float).ravel()


def _point_arrays(inp, s, shape, z_grid):
    """Per-point inputs at age s for the (type, asset, shock) block."""
    wez = inp.w[s] * inp.e[s][:, None, None] * z_grid[None, None, :]
    by_type = {
        "oth": inp.other_income()[s],
        "fi": inp.income_filer[s],
        "lnc": inp.labor_nc[s],
        "knc": inp.capital_nc[s],
        "fw": inp.wealth_filer[s],
        "tp": inp.tau_payroll[s],
    }
    pa = {k: _flat(v[:, None, None], shape) for k, v in by_type.items()}
    pa["wez"] = _flat(wez, shape)
    return pa


def _record_failures(diag, s, n_fail):
    if n_fail and s not in diag.failure_ages:
        diag.failure_ages.append(s)


def _value_of_savings(inp, backend, s, b3, n_next, c_next, z_grid, z_trans):
    r"""
    v = p_tilde_s e^{-sigma g_y} [rho_s chi_b MU_beq(b') + beta (1 - rho_s)
    E(dR/db' MU(c')/p_tilde_{s+1})], the marginal utility of consumption
    at age s that the savings Euler equation implies for savings b' on
    the grid (shape (M, nb, nz)).
    """
    shape = b3.shape
    s1 = s + 1
    pa1 = _point_arrays(inp, s1, shape, z_grid)
    X = pa1["wez"] * n_next.ravel() * inp.factor
    Y = inp.r[s1] * b3.ravel() * inp.factor
    mtry = (backend.mtry_raw(s1, X, Y) * (1 - pa1["knc"]) * pa1["fi"]).reshape(
        shape
    )
    mtrw = tax.MTR_wealth(
        b3,
        inp.h_wealth[s1],
        inp.m_wealth[s1],
        inp.p_wealth[s1],
        pa1["fw"].reshape(shape),
    )
    V = (
        (1 + inp.r[s1] - inp.r[s1] * mtry - mtrw)
        * kernels.mu_c(c_next, inp.sigma)
        / inp.p_tilde[s1]
    )
    EV = V @ z_trans.T
    mu_b = kernels.mu_beq(b3, inp.sigma, 1.0) * inp.chi_b[:, None, None]
    rho = inp.rho[s][:, None, None]
    beta = inp.beta[:, None, None]
    return (
        inp.p_tilde[s]
        * np.exp(-inp.sigma * inp.g_y)
        * (rho * mu_b + beta * (1 - rho) * EV)
    )


def _to_exogenous(b_endo, n_endo, ok, grid):
    """
    Map knots (b_endo[m, k, iz] is the asset level at which savings
    grid[m, k] is optimal) to savings at each grid node.

    Knots at or above the borrowing limit b_min = grid[m, 0] = 0 are
    interpolated (monotone cubic), with linear extrapolation above the
    last one. Knots below the limit come from extending the model to
    negative assets with taxes frozen at their b = 0 values; the policy
    has a kink at b = 0, so they are not interpolated across. Nodes
    below the first knot at or above the limit are

    * constrained (b' = b_min) if that knot is the one for b' = b_min;
    * otherwise gap nodes, whose savings ``_solve_euler_direct`` finds;
      their guess is linear interpolation between the knots on either
      side of the limit.

    A type and shock with fewer than two usable knots at or above the
    limit (the inversion failed for the others, as at savings levels
    far above where anyone is when the grid is too large) has all its
    nodes solved directly, with b' = b as the guess.

    Returns:
        b_next (Numpy array): savings at grid nodes, (M, nb, nz)
        n_guess (Numpy array): labor supply guess, (M, nb, nz)
        gap (Numpy array): gap nodes, bool (M, nb, nz)
        counts (dict): constrained, gap, extrapolated and dropped
    """
    M, nb, nz = b_endo.shape
    x = np.ascontiguousarray(b_endo.transpose(0, 2, 1).reshape(M * nz, nb))
    okf = ok.transpose(0, 2, 1).reshape(M * nz, nb)
    keep = kernels.monotone_valid(x, okf & np.isfinite(x))
    y_b = np.repeat(grid, nz, axis=0)
    y_n = n_endo.transpose(0, 2, 1).reshape(M * nz, nb)
    b_min = y_b[:, :1]
    use = keep & (x >= b_min)
    few = use.sum(axis=1) < 2  # rows solved directly at every node
    if few.any():
        # placeholder knots (b' = b) so that every row can be
        # interpolated; their results are replaced below
        x = x.copy()
        y_n = y_n.copy()
        keep[few] = False
        use[few] = False
        use[few, :2] = keep[few, :2] = True
        x[few, :2] = y_b[few, :2]
        y_n[few, :2] = np.nan
    out = kernels.pchip_many(x, np.stack([y_b, y_n]), use, y_b)
    b_next, n_guess = out[0], out[1]
    rows = np.arange(M * nz)
    first = np.argmax(use, axis=1)
    last = nb - 1 - np.argmax(use[:, ::-1], axis=1)
    x_hi = x[rows, first]
    below = y_b < x_hi[:, None]
    constrained = below & (first == 0)[:, None]
    gap = below & (first > 0)[:, None]
    b_next = np.where(constrained, b_min, b_next)
    n_guess = np.where(constrained, y_n[:, :1], n_guess)
    # guess at gap nodes: linear between the last kept knot below the
    # limit (if any) and the first knot above it
    k_lo = np.where(keep & (x < b_min), np.arange(nb)[None, :], -1).max(1)
    has_lo = k_lo >= 0
    k_lo = np.maximum(k_lo, 0)
    x_lo = x[rows, k_lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip((y_b - x_lo[:, None]) / (x_hi - x_lo)[:, None], 0, 1)
        for vals, target in ((y_b, b_next), (y_n, n_guess)):
            v_lo = vals[rows, k_lo][:, None]
            v_hi = vals[rows, first][:, None]
            guess = np.where(has_lo[:, None], v_lo + t * (v_hi - v_lo), v_hi)
            target[gap] = np.broadcast_to(guess, target.shape)[gap]
    extrapolated = y_b > x[rows, last][:, None]
    if few.any():
        constrained[few] = extrapolated[few] = False
        gap[few] = True
        b_next[few] = y_b[few]
        n_guess[few] = np.nan
    counts = {
        "constrained": int(constrained.sum()),
        "gap": int(gap.sum()),
        "extrapolated": int(extrapolated.sum()),
        "dropped": int((okf & ~keep & ~few[:, None]).sum()),
    }

    def back(a):
        return np.ascontiguousarray(a.reshape(M, nz, nb).transpose(0, 2, 1))

    return back(b_next), back(n_guess), back(gap), counts


class _EulerConsumption:
    """
    Consumption c_E(b') that the savings Euler equation implies for
    savings b', by type and shock: monotone cubic interpolation over the
    knots b' = grid nodes (linear beyond them). Knot values come from
    the stitched inverse of marginal utility, so c_E is defined (and
    negative) where the Euler equation has no positive solution.

    Attributes:
        x (Numpy array): savings at the knots by series m * nz + iz,
            (M * nz, nb): the asset grid of type m
        y (Numpy array): Euler consumption at the knots, (M * nz, nb)
        nz (int): shock states
    """

    def __init__(self, grid, c_E):
        M, nb, nz = c_E.shape
        self.nz = nz
        self.x = np.ascontiguousarray(np.repeat(grid, nz, axis=0))
        self.y = np.ascontiguousarray(c_E.transpose(0, 2, 1).reshape(-1, nb))
        self.valid = np.ones(self.x.shape, dtype=bool)
        self._slopes = None

    def __call__(self, rows, bp):
        """c_E at savings bp for series rows (m * nz + iz)."""
        return kernels.pchip_query(self.x, self.y, self.valid, rows, bp)

    @property
    def slopes(self):
        """PCHIP slopes at the knots, (M * nz, nb)."""
        if self._slopes is None:
            self._slopes = kernels._pchip_slopes_rows(self.x, self.y)
        return self._slopes


def _solve_euler_direct(
    backend, s, euler_c, idx, bp_guess, n_guess, pa, options
):
    """
    Savings at selected grid nodes from the savings Euler equation
    directly, instead of from interpolating knots.

    At a node with assets b, phi(b') = c(b, b') - c_E(b'), where
    c(b, b') solves the within-period problem (labor FOC and budget
    constraint) given savings b' and c_E is the Euler consumption
    (``_EulerConsumption``). c(b, .) falls in b' and c_E normally rises,
    so phi crosses zero once, from above; if phi(b_min) <= 0 the
    borrowing constraint binds and b' = b_min. Where the tax functions
    make c_E non-monotone, the root found is the one, or the
    constrained corner, that the search from the guess reaches.

    The root is bracketed starting from the grid interval around the
    guess and stepping outward over grid nodes with doubling steps (up
    to twice the top of the grid), then located by the Illinois variant
    of regula falsi to a relative consumption error of
    ``kernels.EULER_ROOT_TOL``.

    Args:
        backend: solver backend
        s (int): age
        euler_c (_EulerConsumption): Euler consumption at age s
        idx (Numpy array): flat indices of the nodes in the (M, nb, nz)
            block
        bp_guess, n_guess (Numpy arrays): guesses at the nodes
        pa (dict): per-point inputs of the block (``_point_arrays``)
        options (SolverOptions): solver settings

    Returns:
        bp, n (Numpy arrays): savings and labor at the nodes
        outcome (Numpy array): ``kernels.EULER_ROOT`` (the Euler
            equation holds), ``EULER_AT_LIMIT`` (the borrowing
            constraint binds), ``EULER_KINK`` (savings where phi jumps
            from + to -: the labor choice switches and the value
            function has a kink, so the Euler equation holds as a pair
            of inequalities) or ``EULER_FAILED``
    """
    nz = euler_c.nz
    nb = euler_c.x.shape[1]
    rows = (idx // (nb * nz)) * nz + idx % nz
    b = euler_c.x[rows, (idx // nz) % nb]
    n_start = np.where(np.isfinite(n_guess), n_guess, 0.3 * backend.inp.ltilde)
    pa_q = {k: np.ascontiguousarray(v[idx]) for k, v in pa.items()}
    return backend.euler_direct(
        s,
        euler_c,
        rows,
        b,
        np.asarray(bp_guess, dtype=float),
        n_start,
        pa_q,
        options.tol,
        options.maxiter,
    )


def _static_ok(st, c):
    """Within-period solutions that count as solved (c > 0)."""
    return ((st == kernels.CONVERGED) | (st == kernels.CORNER)) & (c > 0)


def _record_static(diag, s, st, res, c):
    ok = _static_ok(st, c)
    n_fail = int((~ok).sum())
    diag.static_failures += n_fail
    diag.labor_corners += int((st == kernels.CORNER).sum())
    if np.isfinite(res).any():
        diag.max_static_residual = max(
            diag.max_static_residual, float(np.nanmax(res))
        )
    _record_failures(diag, s, n_fail)
    return ok


def _euler_residual(c, c_e, bp):
    """
    Relative Euler equation error at nodes: c_E / c - 1 where b' > 0;
    at the borrowing limit only a violated inequality counts, c > c_E
    (and any c > 0 if the Euler equation has no positive solution).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        interior = c_e / c - 1
        at_limit = np.where(c_e > 0, np.maximum(c / c_e - 1, 0.0), np.inf)
    return np.where(bp > 0, interior, at_limit)


def _validate_inputs(inp, grid, z_grid, z_trans, options):
    """Fail early, with the input's name, on inputs the solver cannot use."""
    for name in (
        "r",
        "w",
        "p_tilde",
        "tau_payroll",
        "chi_n",
        "h_wealth",
        "m_wealth",
        "p_wealth",
        "e",
        "rho",
        "beta",
        "chi_b",
        "income_filer",
        "wealth_filer",
        "labor_nc",
        "capital_nc",
    ):
        if not np.isfinite(np.asarray(getattr(inp, name), dtype=float)).all():
            raise ValueError(f"inputs.{name} contains NaN or inf")
    if not np.isfinite(inp.other_income()).all():
        raise ValueError(
            "Income other than labor and capital income (bequests, "
            "remittances, transfers, UBI, pensions) contains NaN or inf"
        )
    for name in ("etr_params", "mtrx_params", "mtry_params"):
        try:
            params = np.asarray(getattr(inp, name), dtype=float)
        except (TypeError, ValueError):
            continue  # nonparametric tax functions
        if not np.isfinite(params).all():
            raise ValueError(f"inputs.{name} contains NaN or inf")
    if not np.isfinite(grid).all():
        raise ValueError("The asset grid contains NaN or inf")
    if not (np.isfinite(z_grid).all() and (z_grid > 0).all()):
        raise ValueError("Productivity levels z_grid must be positive")
    if (
        z_trans.shape != (z_grid.size, z_grid.size)
        or not np.isfinite(z_trans).all()
        or (z_trans < 0).any()
        or not np.allclose(z_trans.sum(axis=1), 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError(
            "z_trans must be a transition matrix matching z_grid (rows "
            "of nonnegative probabilities summing to 1)"
        )
    if not options.euler_tol >= 100 * kernels.EULER_ROOT_TOL:
        raise ValueError(
            "SolverOptions.euler_tol must be at least "
            f"{100 * kernels.EULER_ROOT_TOL:g} (the direct Euler solve "
            f"converges to {kernels.EULER_ROOT_TOL:g})"
        )


def solve_lifecycle(inp, b_grid, z_grid, z_trans, options=None, check=True):
    """
    Solve the life-cycle problem of a batch of types by backward
    induction with the endogenous grid method.

    At each age s < S - 1:

    1. the savings Euler equation gives consumption c_E(b') for each
       savings level b' on the grid and current shock
       (``_value_of_savings``);
    2. a batched Newton step solves the labor FOC and budget constraint
       for the assets b and labor n at which (c_E(b'), b') is optimal:
       the endogenous grid of knots;
    3. savings at the grid nodes come from monotone cubic interpolation
       of the knots at or above the borrowing limit; below the knot for
       b' = b_min the borrowing constraint binds (b' = b_min); between
       the limit and the first knot above it, when the knots for low b'
       lie below the limit, the Euler equation is solved directly
       (``_to_exogenous``, ``_solve_euler_direct``);
    4. at every node, labor supply and consumption solve the labor FOC
       and budget constraint exactly given b';
    5. the Euler equation is checked at every node against c_E
       (interpolated in b'), including the Euler inequality at the
       borrowing limit; where it misses by more than
       ``options.euler_tol`` it is solved directly at that node.

    At the last age the bequest FOC gives b' = kappa c and step 4 solves
    for n.

    Args:
        inp (LifecycleInputs): inputs for M types
        b_grid (array_like): asset grid, (nb,) or (M, nb), increasing,
            starting at the borrowing limit 0
        z_grid (array_like): productivity levels, (nz,)
        z_trans (array_like): shock transition matrix, (nz, nz)
        options (SolverOptions): solver settings
        check (bool): raise ``HouseholdSolverError`` if failures exceed
            ``options.max_fail_frac`` of grid points; with False the
            caller checks (``solve_all_households_ss`` weighs failures
            by the population at the failed nodes). Policies with NaN
            always raise.

    Returns:
        b_policy (Numpy array): savings b' chosen, (S, M, nb, nz)
        c_policy (Numpy array): consumption, (S, M, nb, nz)
        n_policy (Numpy array): labor supply, (S, M, nb, nz)
        diagnostics (SolverDiagnostics): counts of failures and of
            constrained/extrapolated points

    Raises:
        HouseholdSolverError: failures above ``options.max_fail_frac``
        ValueError: unusable inputs
    """
    options = options or SolverOptions()
    t0 = time.perf_counter()
    backend = _make_backend(inp, options.backend)
    S, M = inp.S, inp.M
    z_grid = np.atleast_1d(np.asarray(z_grid, dtype=float))
    z_trans = np.atleast_2d(np.asarray(z_trans, dtype=float))
    nz = z_grid.size
    grid = np.asarray(b_grid, dtype=float)
    if grid.ndim == 1:
        grid = np.tile(grid, (M, 1))
    if grid.shape[0] != M:
        raise ValueError(f"b_grid has {grid.shape[0]} rows for {M} types")
    _validate_inputs(inp, grid, z_grid, z_trans, options)
    if (np.diff(grid, axis=1) <= 0).any():
        raise ValueError("Asset grids must be strictly increasing")
    if not (grid[:, 0] == 0.0).all():
        raise ValueError(
            "Asset grids must start at 0, the borrowing limit (OG-Core's "
            "tax functions are not defined for negative capital income)"
        )
    nb = grid.shape[1]
    shape = (M, nb, nz)
    P = M * nb * nz
    b_pol = np.empty((S,) + shape)
    c_pol = np.empty((S,) + shape)
    n_pol = np.empty((S,) + shape)
    diag = SolverDiagnostics(backend=backend.name)
    failed = np.zeros((S,) + shape, dtype=bool)
    b3 = np.broadcast_to(grid[:, :, None], shape)
    b_flat = _flat(b3, shape)
    b_min = _flat(grid[:, :1, None], shape)  # borrowing limit by type
    rows = np.ascontiguousarray(
        np.broadcast_to(
            np.arange(M)[:, None, None] * nz + np.arange(nz)[None, None, :],
            shape,
        ).ravel()
    )
    zeros = np.zeros(P)
    tol, maxiter = options.tol, options.maxiter
    lt = inp.ltilde

    def finish_age(s, b_next, n, c):
        b_pol[s] = b_next.reshape(shape)
        n_pol[s] = n.reshape(shape)
        c_pol[s] = c.reshape(shape)
        if not (
            np.isfinite(b_pol[s]).all()
            and np.isfinite(c_pol[s]).all()
            and np.isfinite(n_pol[s]).all()
        ):
            raise HouseholdSolverError(
                f"Household policies at age {s} contain NaN or inf: "
                + diag.summary()
            )

    # last age: bequest FOC, MU(c) / p_tilde = rho e^{-sigma g_y} chi_b
    # b'^{-sigma}, i.e. b' = kappa c
    s = S - 1
    pa = _point_arrays(inp, s, shape, z_grid)
    kappa = np.exp(-inp.g_y) * (inp.p_tilde[s] * inp.rho[s] * inp.chi_b) ** (
        1 / inp.sigma
    )
    kappa_flat = _flat(kappa[:, None, None], shape)
    n, c, st, res = backend.static(
        s, np.full(P, 0.3 * lt), b_flat, zeros, kappa_flat, pa, tol, maxiter
    )
    good = _record_static(diag, s, st, res, c)
    failed[s] = ~good.reshape(shape)
    finish_age(s, kappa_flat * c, n, c)

    mu_c_zero = kernels.mu_c(np.zeros(1), inp.sigma)[0]
    growth = np.exp(inp.g_y)
    for s in range(S - 2, -1, -1):
        # 1. consumption implied by the Euler equation for each b'
        v = _value_of_savings(
            inp, backend, s, b3, n_pol[s + 1], c_pol[s + 1], z_grid, z_trans
        )
        finite_v = np.isfinite(v)
        feasible = finite_v & (v < mu_c_zero)  # else c <= 0: never chosen
        c_E = kernels.inv_mu_c(v, inp.sigma)
        euler_c = _EulerConsumption(grid, c_E)
        # 2. assets and labor at which (c, b') is optimal
        pa = _point_arrays(inp, s, shape, z_grid)
        A = v.ravel() * pa["wez"] / inp.p_tilde[s]
        rhs = (
            inp.p_tilde[s] * np.where(feasible, c_E, 1.0).ravel()
            + growth * b_flat
        )
        n0 = np.clip(n_pol[s + 1].ravel(), 0.05 * lt, 0.95 * lt)
        b0 = (rhs - pa["oth"] - 0.75 * pa["wez"] * n0) / (1 + inp.r[s])
        b_e, n_e, st, _ = backend.egm(
            s,
            feasible.ravel(),
            b0,
            n0,
            b_flat,
            b_min,
            A,
            rhs,
            pa,
            options.egm_tol,
            maxiter,
            nb,
            nz,
        )
        feas = feasible.ravel()
        ok = feas & (st == kernels.CONVERGED)
        n_fail = int((feas & ~ok).sum()) + int((~finite_v).sum())
        diag.egm_failures += n_fail
        diag.infeasible_knots += int((finite_v & ~feasible).sum())
        _record_failures(diag, s, n_fail)
        # 3. savings at the grid nodes
        b_next, n_guess, gap, counts = _to_exogenous(
            b_e.reshape(shape), n_e.reshape(shape), ok.reshape(shape), grid
        )
        diag.gap += counts["gap"]
        diag.extrapolated += counts["extrapolated"]
        diag.dropped_knots += counts["dropped"]
        bn = b_next.ravel()
        ng = np.where(np.isfinite(n_guess.ravel()), n_guess.ravel(), 0.3 * lt)
        kink = np.zeros(P, dtype=bool)
        idx = np.flatnonzero(gap.ravel())
        if idx.size:
            bp, n_d, out = _solve_euler_direct(
                backend, s, euler_c, idx, bn[idx], ng[idx], pa, options
            )
            ok_d = out != kernels.EULER_FAILED
            bn[idx[ok_d]], ng[idx[ok_d]] = bp[ok_d], n_d[ok_d]
            kink[idx[out == kernels.EULER_KINK]] = True
            diag.direct += int(ok_d.sum())
        # 4. labor supply and consumption at the grid nodes
        n, c, st, res = backend.static(
            s, ng, b_flat, bn, zeros, pa, tol, maxiter
        )
        # 5. the Euler equation at every node
        resid = _euler_residual(c, euler_c(rows, bn), bn)
        missed = ~(np.abs(resid) <= options.euler_tol) & ~kink
        idx = np.flatnonzero(~_static_ok(st, c) | missed)
        if idx.size:
            n_start = np.where(_static_ok(st, c)[idx], n[idx], ng[idx])
            bp, n_d, out = _solve_euler_direct(
                backend, s, euler_c, idx, bn[idx], n_start, pa, options
            )
            ok_d = out != kernels.EULER_FAILED
            j = idx[ok_d]
            if j.size:
                bn[j] = bp[ok_d]
                n[j], c[j], st[j], res[j] = backend.static(
                    s,
                    n_d[ok_d],
                    b_flat[j],
                    bn[j],
                    zeros[j],
                    {k: v[j] for k, v in pa.items()},
                    tol,
                    maxiter,
                )
                resid[j] = _euler_residual(
                    c[j], euler_c(rows[j], bn[j]), bn[j]
                )
                kink[j] = out[ok_d] == kernels.EULER_KINK
                diag.direct += int(j.size)
        good = _record_static(diag, s, st, res, c)
        missed = good & ~kink & ~(np.abs(resid) <= options.euler_tol)
        diag.euler_failures += int(missed.sum())
        diag.euler_kinks += int((good & kink).sum())
        _record_failures(diag, s, int(missed.sum()))
        regular = good & ~kink
        if regular.any():
            diag.max_euler_residual = max(
                diag.max_euler_residual,
                float(np.max(np.abs(resid[regular]))),
            )
        diag.constrained += int((bn == b_min).sum())
        failed[s] = (~good | missed).reshape(shape)
        finish_age(s, bn, n, c)

    diag.n_points = S * P
    diag.n_knots = (S - 1) * P
    diag.failure_ages = sorted(diag.failure_ages)
    diag.failed_nodes = failed
    diag.seconds = time.perf_counter() - t0
    if check:
        diag.check(options)
    logger.debug("solve_lifecycle: %s", diag.summary())
    return b_pol, c_pol, n_pol, diag


def solve_HH(
    r,
    w,
    p_tilde,
    factor,
    tr,
    bq,
    ubi,
    b_grid,
    sigma,
    pension,
    chi_n,
    rho,
    e,
    etr_params,
    mtrx_params,
    mtry_params,
    j,
    t,
    p,
    method,
    rm=0.0,
    c_min_exp=0.0,
    options=None,
    return_diagnostics=False,
):
    """
    Solve one type's life-cycle problem on an asset grid with the
    endogenous grid method (see ``solve_lifecycle``).

    Args:
        r, w, p_tilde (array_like): interest rate, wage and composite
            good price at each age, (S,) or scalars
        factor (scalar): model-units-to-dollars scaling factor
        tr, bq, ubi (array_like): transfers, bequests and UBI received
            at each age, (S,) or scalars
        b_grid (array_like): asset grid, (nb,); its first node is the
            borrowing limit
        sigma (scalar): coefficient of relative risk aversion (must
            equal p.sigma)
        pension (array_like): pension benefits at each age, (S,) or
            scalar (see ``pension_ss``)
        chi_n (array_like): disutility-of-labor weights by age, (S,)
        rho (array_like): mortality rates by age, (S,)
        e (array_like): effective labor units by age, (S,)
        etr_params, mtrx_params, mtry_params (array_like): tax function
            parameters by age, (S, K)
        j (int): type index (for beta, chi_b and filer status)
        t (int or array_like): model period(s) of each age for policy
            parameters when method is not 'SS'
        p (OG-Core Specifications object): model parameters
        method (str): 'SS' or 'TPI'
        rm (array_like): remittances received at each age
        c_min_exp (array_like): minimum consumption expenditure
        options (SolverOptions): solver settings
        return_diagnostics (bool): also return ``SolverDiagnostics``

    Returns:
        b_policy, c_policy, n_policy (Numpy arrays): savings,
        consumption and labor supply on the (age, asset, shock) grid,
        each (S, nb, nz)
    """
    if not np.isclose(sigma, p.sigma):
        raise ValueError("sigma must equal p.sigma")
    S = p.S
    if method == "SS":
        tidx = np.full(S, -1)
    else:
        tidx = np.broadcast_to(np.asarray(t, dtype=int), (S,))

    def by_age(x):
        return np.array(np.broadcast_to(np.asarray(x, dtype=float), (S,)))

    def by_age_col(x):
        return by_age(x).reshape(S, 1)

    def policy(name):
        return by_age(np.asarray(getattr(p, name))[tidx])

    def policy_j(name):
        return by_age_col(np.asarray(getattr(p, name))[tidx, j])

    bq_ = by_age_col(bq)
    inp = LifecycleInputs(
        js=np.array([j]),
        r=by_age(r),
        w=by_age(w),
        p_tilde=by_age(p_tilde),
        chi_n=by_age(chi_n),
        h_wealth=policy("h_wealth"),
        m_wealth=policy("m_wealth"),
        p_wealth=policy("p_wealth"),
        c_min_exp=by_age(c_min_exp),
        etr_params=etr_params,
        mtrx_params=mtrx_params,
        mtry_params=mtry_params,
        e=by_age_col(e),
        rho=by_age_col(rho),
        tau_payroll=policy_j("tau_payroll"),
        bq=bq_,
        rm=by_age_col(rm),
        tr=by_age_col(tr),
        ubi=by_age_col(ubi),
        pension=by_age_col(pension),
        bq_tax=policy("tau_bq").reshape(S, 1) * bq_,
        income_filer=policy_j("income_tax_filer"),
        wealth_filer=policy_j("wealth_tax_filer"),
        labor_nc=policy_j("labor_income_tax_noncompliance_rate"),
        capital_nc=policy_j("capital_income_tax_noncompliance_rate"),
        beta=np.array([np.asarray(p.beta)[j]], dtype=float),
        chi_b=np.array([np.asarray(p.chi_b)[j]], dtype=float),
        factor=float(factor),
        sigma=float(p.sigma),
        g_y=float(p.g_y),
        ltilde=float(p.ltilde),
        b_ellipse=float(p.b_ellipse),
        upsilon=float(p.upsilon),
        tax_func_type=p.tax_func_type,
        analytical_mtrs=bool(p.analytical_mtrs),
    )
    z_grid, z_trans, _ = _shock_process(p)
    b_pol, c_pol, n_pol, diag = solve_lifecycle(
        inp, b_grid, z_grid, z_trans, options
    )
    out = (b_pol[:, 0], c_pol[:, 0], n_pol[:, 0])
    if return_diagnostics:
        return out + (diag,)
    return out


"""
------------------------------------------------------------------------
    Distribution
------------------------------------------------------------------------
"""


def get_ss_distribution(
    p,
    b_policy,
    b_grid,
    initial_dist=None,
    z_dist0=None,
    js=None,
    return_diagnostics=False,
):
    """
    Steady-state distribution of households over age, type, assets and
    shocks, by Young's (2010) method: each age's mass at an asset node
    moves to the two grid nodes around the savings it chooses, with
    weights that preserve the mean, and across shocks by the transition
    matrix.

    Mortality and immigration do not depend on assets or shocks, so the
    distribution within an (age, type) cell evolves by the policies
    alone, and each cell is scaled to its population share: the result
    sums to ``p.omega_SS[s, j]`` over assets and shocks, and to 1 over
    everything (for all J types).

    Args:
        p (Specifications): model parameters
        b_policy (Numpy array): savings chosen, (S, J, nb, nz)
        b_grid (Numpy array): asset grid, (nb,) or (J, nb)
        initial_dist (Numpy array): distribution of newborns over
            (type, asset, shock), (J, nb, nz); only its shape within
            each type matters. If None, newborns hold zero assets and
            draw shocks from ``z_dist0``.
        z_dist0 (array_like): shock distribution of newborns; ``p.z_dist0``
            (default: the stationary distribution) if None
        js (array_like): the types in ``b_policy`` (all J if None)
        return_diagnostics (bool): also return a dict of checks

    Returns:
        dist (Numpy array): (S, J, nb, nz)
        diagnostics (dict, optional): 'mass_above_grid' and
            'mass_below_grid' (population share by type whose savings
            fell outside the grid and were clipped), 'top_node_mass'
            (share of each type's population at the top node)
    """
    b_policy = np.asarray(b_policy, dtype=float)
    S, M, nb, nz = b_policy.shape
    js = np.arange(M) if js is None else np.atleast_1d(js).astype(int)
    grid = np.asarray(b_grid, dtype=float)
    if grid.ndim == 1:
        grid = np.tile(grid, (M, 1))
    _, z_trans, zd0 = _shock_process(p)
    z_dist0 = zd0 if z_dist0 is None else np.asarray(z_dist0, dtype=float)
    if z_trans.shape != (nz, nz):
        raise ValueError("b_policy and p.z_trans disagree on nz")
    omega = np.asarray(p.omega_SS, dtype=float)[:, js]
    cond = np.zeros((S, M, nb, nz))
    below0 = np.zeros(M)
    if initial_dist is None:
        for m in range(M):
            g = grid[m]
            if 0.0 <= g[0]:
                k, wt = 0, 0.0
                below0[m] = float(g[0] > 0.0)
            elif 0.0 >= g[-1]:
                k, wt = nb - 2, 1.0
            else:
                k = np.searchsorted(g, 0.0, side="right") - 1
                wt = (0.0 - g[k]) / (g[k + 1] - g[k])
            cond[0, m, k, :] += (1 - wt) * z_dist0
            cond[0, m, k + 1, :] += wt * z_dist0
    else:
        init = np.asarray(initial_dist, dtype=float).reshape(M, nb, nz)
        cond[0] = init / init.sum(axis=(1, 2), keepdims=True)
    above = np.zeros((S, M))
    below = np.zeros((S, M))
    for s in range(S - 1):
        new, above[s], below[s] = kernels.young_step(
            cond[s], b_policy[s], grid, z_trans
        )
        cond[s + 1] = new / new.sum(axis=(1, 2), keepdims=True)
    dist = cond * omega[:, :, None, None]
    if not return_diagnostics:
        return dist
    pop = omega.sum(axis=0)
    diagnostics = {
        "mass_above_grid": (above * omega).sum(axis=0) / pop,
        "mass_below_grid": (below * omega).sum(axis=0) / pop
        + below0 * omega[0] / pop,
        "top_node_mass": dist[:, :, -1, :].sum(axis=(0, 2)) / pop,
    }
    return dist, diagnostics


def cell_means(dist, x):
    """
    Mean of x within each (age, type) cell: sum over assets and shocks
    of dist * x divided by the cell's mass, (S, J).
    """
    mass = dist.sum(axis=(2, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dist * x).sum(axis=(2, 3)) / mass


def mean_effective_labor(dist, n_policy, z_grid):
    """Mean effective labor E[z n | s, j], (S, J)."""
    return cell_means(dist, n_policy * np.asarray(z_grid)[None, None, None])


"""
------------------------------------------------------------------------
    Steady state household block
------------------------------------------------------------------------
"""


@dataclass
class HouseholdSolution:
    """
    Steady-state solution of the household block at given prices.

    Attributes:
        b_policy, c_policy, n_policy (Numpy arrays): savings b',
            consumption and labor supply at each (age, type, asset node,
            shock), (S, J, nb, nz)
        b_grid (Numpy array): asset grid of each type, (J, nb)
        z_grid, z_trans (Numpy arrays): shock process
        dist (Numpy array): stationary distribution, (S, J, nb, nz),
            summing to omega_SS over assets and shocks
        pension (Numpy array): pension benefits by age and type, (S, J)
        theta (Numpy array): replacement rates (US-style Social
            Security), (J,), or None
        inputs (LifecycleInputs): everything the households faced
        diagnostics (SolverDiagnostics): solver counts
        dist_diagnostics (dict): distribution checks
        iterations (dict): rounds of grid resizing ('grid', and
            'grid_settled' False if the grids did not settle) and of
            pension updates ('pension')
    """

    b_policy: np.ndarray
    c_policy: np.ndarray
    n_policy: np.ndarray
    b_grid: np.ndarray
    z_grid: np.ndarray
    z_trans: np.ndarray
    dist: np.ndarray
    pension: np.ndarray
    theta: np.ndarray
    inputs: LifecycleInputs
    diagnostics: SolverDiagnostics
    dist_diagnostics: dict
    iterations: dict

    def cell_means(self, x):
        """Mean of x (S, J, nb, nz) within each (age, type) cell."""
        return cell_means(self.dist, x)

    def b_s(self):
        """Assets held at each node, (S, J, nb, nz)."""
        return np.broadcast_to(
            self.b_grid[None, :, :, None], self.b_policy.shape
        )


def _support_top(dist, b_policy, grid, omega_j, mass_tol, failed=None):
    """
    Highest savings chosen at nodes with mass, by type, leaving out
    nodes where the solver failed (their savings are not reliable).
    """
    S, M, nb, nz = dist.shape
    top = np.zeros(M)
    for m in range(M):
        massive = dist[:, m] > mass_tol * omega_j[m]
        if failed is not None:
            massive &= ~failed[:, m]
        if massive.any():
            top[m] = max(b_policy[:, m][massive].max(), 0.0)
            node_top = grid[m][massive.any(axis=(0, 2))].max()
            top[m] = max(top[m], node_top)
    return top


def _grid_fit(dist, b_policy, grids, p, options, failed=None):
    """
    Grid tops that fit the distribution, ``p.b_grid_margin`` times the
    highest savings chosen where there is mass, by type, and which
    grids have tops off from these by more than a factor 1.25.
    """
    omega_j = np.asarray(p.omega_SS).sum(axis=0)
    top = _support_top(
        dist, b_policy, grids, omega_j, options.mass_tol, failed
    )
    target = p.b_grid_margin * np.maximum(top, 1e-3)
    off = np.abs(np.log(grids[:, -1] / target)) > np.log(1.25)
    return target, off


def refit_asset_grids(sol, p, options=None):
    """
    Asset grids that fit the distribution of a household solution, by
    the rule ``solve_all_households_ss`` uses to size its grids: a type
    keeps its grid if the top is within a factor 1.25 of
    ``p.b_grid_margin`` times the highest savings chosen where the
    distribution has mass, and otherwise gets a grid of the same size
    (``make_asset_grid``) with that top.

    Args:
        sol (HouseholdSolution): household solution
        p (OG-Core Specifications object): model parameters
        options (SolverOptions): solver settings (``mass_tol``)

    Returns:
        b_grid (Numpy array): asset grids, (J, nb)
        refit (Numpy array): bool (J,), True for types whose grid changed
    """
    options = options or SolverOptions()
    target, off = _grid_fit(
        sol.dist,
        sol.b_policy,
        sol.b_grid,
        p,
        options,
        sol.diagnostics.failed_nodes,
    )
    b_grid = np.array(sol.b_grid, dtype=float)
    for j in np.flatnonzero(off):
        b_grid[j] = make_asset_grid(
            target[j], b_grid.shape[1], p.b_grid_curvature
        )
    return b_grid, off


def solve_all_households_ss(
    r_p,
    w,
    p_m,
    BQ,
    TR,
    factor,
    p,
    Y=None,
    theta=None,
    pension=None,
    b_grid=None,
    options=None,
    pension_tol=1e-10,
    pension_maxiter=100,
):
    """
    Steady-state policies and distribution of all J types at given
    prices, aggregates and policy: the household block of OG-Core's
    ``SS.inner_loop``, with idiosyncratic shocks.

    Pension benefits: pass ``pension`` (S, J) or, for US-style Social
    Security, ``theta`` (J,). If neither is given, benefits are made
    consistent with each type's own mean effective earnings by
    fixed-point iteration, as OG-Core's household solve does.

    Asset grids: ``b_grid`` ((nb,) or (J, nb)) is used as given. If None,
    each type gets a grid of ``p.nb`` nodes with curvature
    ``p.b_grid_curvature`` (``make_asset_grid``) whose top is resized
    until it is within a factor 1.25 of ``p.b_grid_margin`` times the
    highest savings chosen where the distribution has mass, for at most
    ``options.grid_resize_iter`` rounds (a warning is logged if the
    grids have not settled).

    Failures: solver failures count by the population at the failed
    grid points; ``HouseholdSolverError`` is raised when that share
    exceeds ``options.max_fail_frac`` (failures where nobody is, for
    example at nodes no household reaches, are logged only).

    Args:
        r_p (scalar): return on the household portfolio
        w (scalar): wage
        p_m (array_like): output prices, (M,)
        BQ (array_like): aggregate bequests by type, (J,)
        TR (scalar): aggregate transfers
        factor (scalar): model-units-to-dollars scaling factor
        p (OG-Core Specifications object): model parameters
        Y (scalar): GDP, for remittances
        theta (array_like): replacement rates, (J,)
        pension (array_like): pension benefits by age and type, (S, J)
        b_grid (array_like): asset grid(s)
        options (SolverOptions): solver settings
        pension_tol (float): tolerance on benefits in the fixed point
        pension_maxiter (int): iterations of the fixed point

    Returns:
        HouseholdSolution
    """
    options = options or SolverOptions()
    z_grid, z_trans, z_dist0 = _shock_process(p)
    hh = household_inputs_ss(p, r_p, w, p_m, BQ, TR, factor, Y)
    us_style = p.pension_system == "US-Style Social Security"

    def inputs(pens):
        return lifecycle_inputs_ss(
            p,
            r_p,
            w,
            hh["p_tilde"],
            factor,
            hh["bq"],
            hh["rm"],
            hh["tr"],
            hh["ubi"],
            pens,
            hh["c_min_exp"],
        )

    endogenous = pension is None and theta is None
    if pension is not None:
        pens = np.broadcast_to(np.asarray(pension, dtype=float), (p.S, p.J))
    elif theta is not None:
        if not us_style:
            raise ValueError("theta applies to US-style Social Security")
        theta = np.broadcast_to(np.asarray(theta, dtype=float), (p.J,))
        pens = pension_ss(p, w, factor, theta=theta)
    else:
        # first guess: OG-Core's default initial labor supply
        n_eff = np.full((p.S, p.J), 0.4)
        if us_style:
            theta = pensions.replacement_rate_vals(n_eff, w, factor, None, p)
        pens = pension_ss(p, w, factor, theta=theta, n_eff=n_eff, r=r_p)

    def solve(inp, grids):
        b_pol, c_pol, n_pol, diag = solve_lifecycle(
            inp, grids, z_grid, z_trans, options, check=False
        )
        dist, dd = get_ss_distribution(
            p, b_pol, grids, z_dist0=z_dist0, return_diagnostics=True
        )
        diag.failed_mass = float(
            dist[diag.failed_nodes].sum() / max(dist.sum(), 1e-300)
        )
        return b_pol, c_pol, n_pol, diag, dist, dd

    inp = inputs(pens)
    iterations = {"grid": 0, "pension": 0, "grid_settled": True}
    if b_grid is not None:
        grids = np.asarray(b_grid, dtype=float)
        if grids.ndim == 1:
            grids = np.tile(grids, (p.J, 1))
        result = solve(inp, grids)
    else:
        b_max = initial_b_max(inp, z_grid)
        for it in range(options.grid_resize_iter + 1):
            grids = np.stack(
                [make_asset_grid(bm, p.nb, p.b_grid_curvature) for bm in b_max]
            )
            result = solve(inp, grids)
            iterations["grid"] = it
            target, off = _grid_fit(
                result[4],
                result[0],
                grids,
                p,
                options,
                result[3].failed_nodes,
            )
            if not off.any():
                break
            if it == options.grid_resize_iter:
                iterations["grid_settled"] = False
                logger.warning(
                    "Asset grids did not settle in %d rounds: grid tops %s, "
                    "targets %s (types %s)",
                    options.grid_resize_iter,
                    np.array2string(b_max[off], precision=3),
                    np.array2string(target[off], precision=3),
                    np.flatnonzero(off),
                )
                break
            b_max = np.where(off, target, b_max)

    if endogenous:
        for it in range(1, pension_maxiter + 1):
            b_pol, c_pol, n_pol, _, dist, _ = result
            n_eff = mean_effective_labor(dist, n_pol, z_grid)
            if us_style:
                theta = pensions.replacement_rate_vals(
                    n_eff, w, factor, None, p
                )
            pens_new = pension_ss(
                p, w, factor, theta=theta, n_eff=n_eff, r=r_p
            )
            change = np.max(np.abs(pens_new - pens))
            iterations["pension"] = it
            if change < pension_tol:
                break
            pens = pens_new
            inp = inputs(pens)
            result = solve(inp, grids)
        else:
            logger.warning(
                "Pension fixed point did not converge: change %.2e", change
            )
    b_pol, c_pol, n_pol, diag, dist, dd = result
    diag.check(options)
    return HouseholdSolution(
        b_policy=b_pol,
        c_policy=c_pol,
        n_policy=n_pol,
        b_grid=grids,
        z_grid=z_grid,
        z_trans=z_trans,
        dist=dist,
        pension=np.array(pens),
        theta=None if theta is None else np.array(theta),
        inputs=inp,
        diagnostics=diag,
        dist_diagnostics=dd,
        iterations=iterations,
    )


"""
------------------------------------------------------------------------
    Steady-state quantities over the distribution

    Within an (age, type) cell, OG-Core's budget constraint and
    aggregates are linear in savings, consumption and effective labor
    z n, so their cell means can go through OG-Core's (S, J) functions
    unchanged. Tax liabilities and tax rates are not, so they are
    evaluated at every asset and shock node and averaged. Pension
    benefits are computed from the cell means of effective labor, an
    approximation with shocks (see ``pension_ss``).
------------------------------------------------------------------------
"""


def _by_node(x):
    """
    An (S, J, nb, nz) array as (S, nb * nz, J): the asset and shock
    nodes of each (age, type) cell on the middle axis, so that OG-Core's
    (S, J) functions apply at every node by broadcasting.
    """
    x = np.asarray(x, dtype=float)
    S, J, nb, nz = x.shape
    return np.moveaxis(x, 1, -1).reshape(S, nb * nz, J)


def _tax_params_by_node(params, S, J, Q):
    """
    Steady-state tax function parameters at Q nodes of every (age,
    type) cell, (S, Q, J, K), or for nonparametric tax functions nested
    lists of the functions, which OG-Core's tax functions index as
    [s][node][j].
    """
    by_age = _tax_params_by_age(params, S)
    if isinstance(by_age, np.ndarray):
        return np.broadcast_to(
            by_age[:, None, None, :], (S, Q, J, by_age.shape[-1])
        )
    return [[[by_age[s]] * J] * Q for s in range(S)]


def _household_nodes(sol, p):
    """
    Assets b held, labor supply n, effective labor units e z and
    population mass at each node, (S, nb * nz, J) each.
    """
    nb = sol.b_policy.shape[2]
    z = np.tile(sol.z_grid, nb)  # node (i_b, i_z) is i_b * nz + i_z
    e = np.asarray(p.e[-1], dtype=float).reshape(p.S, p.J)
    return {
        "b": _by_node(sol.b_s()),
        "n": _by_node(sol.n_policy),
        "e": e[:, None, :] * z[None, :, None],
        "mass": _by_node(sol.dist),
    }


def _node_mean(mass, x):
    """
    Mean of node values x within each (age, type) cell, (S, J); nodes
    without mass are left out (values there may not be finite).
    """
    return np.where(mass > 0, mass * x, 0.0).sum(axis=1) / mass.sum(axis=1)


def tax_cell_means(sol, r, w, factor, p):
    """
    Mean tax liabilities within each (age, type) cell of the
    steady-state distribution: OG-Core's income and payroll tax
    liability (``tax.income_tax_liab``) and wealth tax liability
    (``tax.wealth_tax_liab``) at every asset and shock node, averaged
    over the cell. Weighted by ``p.omega_SS`` they sum to the totals
    over the distribution, so they can stand in for liabilities at the
    cell means, which differ because taxes are nonlinear.

    Args:
        sol (HouseholdSolution): steady-state household solution
        r (scalar): return on the household portfolio
        w (scalar): wage
        factor (scalar): model-units-to-dollars scaling factor
        p (OG-Core Specifications object): model parameters

    Returns:
        dict with 'inc_pay_tax_liab' (income and payroll taxes) and
        'w_tax_liab' (wealth taxes), (S, J) each, the keyword arguments
        of ``tax.net_taxes`` and ``aggregates.revenue`` that take them
    """
    x = _household_nodes(sol, p)
    etr_params = _tax_params_by_node(p.etr_params, p.S, p.J, x["b"].shape[1])
    inc_pay = tax.income_tax_liab(
        r, w, x["b"], x["n"], factor, 0, None, "SS", x["e"], etr_params, p
    )
    wealth = tax.wealth_tax_liab(r, x["b"], 0, None, "SS", p)
    return {
        "inc_pay_tax_liab": _node_mean(x["mass"], inc_pay),
        "w_tax_liab": _node_mean(x["mass"], wealth),
    }


def tax_rate_cell_means(sol, r, w, factor, p):
    """
    Mean income tax rates within each (age, type) cell of the
    steady-state distribution: OG-Core's effective rate
    (``tax.ETR_income``) and marginal rates on labor and capital income
    (``tax.MTR_income``) at every asset and shock node, averaged over
    the cell.

    Args:
        sol (HouseholdSolution): steady-state household solution
        r (scalar): return on the household portfolio
        w (scalar): wage
        factor (scalar): model-units-to-dollars scaling factor
        p (OG-Core Specifications object): model parameters

    Returns:
        dict with 'etr', 'mtrx' (labor income) and 'mtry' (capital
        income), (S, J) each
    """
    x = _household_nodes(sol, p)
    Q = x["b"].shape[1]
    etr_params = _tax_params_by_node(p.etr_params, p.S, p.J, Q)
    labor_nc = p.labor_income_tax_noncompliance_rate[-1, :]
    capital_nc = p.capital_income_tax_noncompliance_rate[-1, :]
    filer = p.income_tax_filer[-1, :]
    b, n, e = x["b"], x["n"], x["e"]
    etr = tax.ETR_income(
        r, w, b, n, factor, e, etr_params, labor_nc, capital_nc, filer, p
    )
    mtrx = tax.MTR_income(
        r,
        w,
        b,
        n,
        factor,
        False,
        e,
        etr_params,
        _tax_params_by_node(p.mtrx_params, p.S, p.J, Q),
        labor_nc,
        filer,
        p,
    )
    mtry = tax.MTR_income(
        r,
        w,
        b,
        n,
        factor,
        True,
        e,
        etr_params,
        _tax_params_by_node(p.mtry_params, p.S, p.J, Q),
        capital_nc,
        filer,
        p,
    )
    return {
        name: _node_mean(x["mass"], rate)
        for name, rate in (("etr", etr), ("mtrx", mtrx), ("mtry", mtry))
    }


def inequality_ss(sol, r, w, p):
    """
    Inequality over the full steady-state distribution of households,
    for savings b' ('b_sp1'), consumption ('c'), labor income w e z n
    ('labor_income') and before-tax income r b + w e z n
    ('before_tax_income'): the measures of
    ``ogcore.output_tables.ineq_table``, from ``ogcore.utils.Inequality``
    weighted by the population at each node.

    Args:
        sol (HouseholdSolution): steady-state household solution
        r (scalar): return on the household portfolio
        w (scalar): wage
        p (OG-Core Specifications object): model parameters

    Returns:
        dict of dicts: for each variable, 'gini', 'var_of_logs',
        '90/10 ratio', 'top_10_share' and 'top_1_share'
    """
    x = _household_nodes(sol, p)
    labor_income = w * x["e"] * x["n"]
    values = {
        "b_sp1": _by_node(sol.b_policy),
        "c": _by_node(sol.c_policy),
        "labor_income": labor_income,
        "before_tax_income": r * x["b"] + labor_income,
    }
    has_mass = x["mass"] > 0
    mass = x["mass"][has_mass].reshape(-1, 1)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, v in values.items():
            ineq = utils.Inequality(
                v[has_mass].reshape(-1, 1), mass, None, p.S, p.J
            )
            out[name] = {
                "gini": ineq.gini(),
                "var_of_logs": ineq.var_of_logs(),
                "90/10 ratio": ineq.ratio_pct1_pct2(0.90, 0.10),
                "top_10_share": ineq.top_share(0.1),
                "top_1_share": ineq.top_share(0.01),
            }
    return out


"""
------------------------------------------------------------------------
    Accuracy diagnostics
------------------------------------------------------------------------
"""


def _interp_policy(grid, values, queries):
    """
    Monotone cubic interpolation of each type's policy (as in the
    solver): grid and values (M, nb), queries (M, Q) in any order.
    """
    order = np.argsort(queries, axis=1)
    q_sorted = np.take_along_axis(queries, order, axis=1)
    vals = kernels.pchip_many(
        grid, values[None], np.ones(grid.shape, dtype=bool), q_sorted
    )[0]
    out = np.empty_like(vals)
    np.put_along_axis(out, order, vals, axis=1)
    return out


def _static_at(inp, backend, s, b, b_next, zs, n0, options):
    """
    Labor and consumption solving the labor FOC and budget constraint at
    assets b and savings b_next (at the last age, the bequest FOC),
    for points of shape (M, Q, nz) whose shock along the last axis is zs,
    starting from labor supply n0 (the interpolated labor policy, so
    that where the labor problem has several local optima the one the
    policy follows is found).
    """
    shape = b.shape
    P = b.size
    pa = _point_arrays(inp, s, shape, zs)
    if s == inp.S - 1:
        kappa = np.exp(-inp.g_y) * (
            inp.p_tilde[s] * inp.rho[s] * inp.chi_b
        ) ** (1 / inp.sigma)
        bp_fix, kap = np.zeros(P), _flat(kappa[:, None, None], shape)
    else:
        bp_fix, kap = _flat(b_next, shape), np.zeros(P)
    n, c, st, _ = backend.static(
        s,
        _flat(n0, shape),
        _flat(b, shape),
        bp_fix,
        kap,
        pa,
        options.tol,
        options.maxiter,
    )
    c = np.where(_static_ok(st, c), c, np.nan)
    return n.reshape(shape), c.reshape(shape)


def _interp_series(grid, values, queries):
    """
    Monotone cubic interpolation, as in the solver, of each (type,
    shock) series of values (M, nb, nz) at queries (M, Q, nz, ...):
    queries[m, :, iz, ...] are evaluated on series (m, iz).
    """
    M, nb, nz = values.shape
    x = np.repeat(grid, nz, axis=0)
    y = values.transpose(0, 2, 1).reshape(M * nz, nb)
    rows = np.broadcast_to(
        (np.arange(M)[:, None] * nz + np.arange(nz)[None, :]).reshape(
            (M, 1, nz) + (1,) * (queries.ndim - 3)
        ),
        queries.shape,
    )
    out = kernels.pchip_query(
        x, y, np.ones(x.shape, dtype=bool), rows.ravel(), queries.ravel()
    )
    return out.reshape(queries.shape)


def euler_errors(sol, ages=None, options=None):
    r"""
    Euler equation errors between grid nodes (Judd 1992). At the
    midpoint b between consecutive nodes, savings b'(b, z) come from
    interpolating the savings policy as the solver does; labor and
    consumption then solve the labor FOC and budget constraint exactly,
    at age s and, for each next shock z', at age s + 1 (with b'' from
    the age-(s + 1) savings policy), starting from the interpolated
    labor policy. The error is

    .. math::
        \frac{c^{Euler}}{c} - 1,

    where :math:`c^{Euler}` is the consumption the savings Euler
    equation implies. At the borrowing limit the Euler equation is an
    inequality, so only :math:`c > c^{Euler}` counts. Midpoints where a
    within-period solve fails have a NaN error, counted separately and
    treated as infinite in the maxima and shares.

    Args:
        sol (HouseholdSolution): solution to check
        ages (iterable): ages s < S - 1 to check (all if None)
        options (SolverOptions): settings for the within-period solves

    Returns:
        dict with
        'errors' (S - 1, J, nb - 1, nz): NaN for unchecked ages;
        'max': over all midpoints checked;
        'max_with_mass': over midpoints next to a node with any mass in
        ``sol.dist``;
        'max_with_mass_by_cell' (S - 1, J): the same by age and type;
        'mean_with_mass': population-weighted mean (each midpoint
        weighted by the mean mass at its two nodes);
        'share_above': {threshold: population share at midpoints with
        a larger error} for thresholds 1e-4, 1e-3 and 1e-2;
        'nan', 'nan_with_mass': midpoints whose error could not be
        computed;
        'b': the midpoints (J, nb - 1)
    """
    options = options or SolverOptions()
    inp = sol.inputs
    backend = _make_backend(inp, options.backend)
    S, M, nb, nz = sol.b_policy.shape
    grid = np.ascontiguousarray(sol.b_grid)
    z_grid, z_trans = sol.z_grid, sol.z_trans
    b_mid = 0.5 * (grid[:, 1:] + grid[:, :-1])
    shape = (M, nb - 1, nz)
    b_q = np.ascontiguousarray(np.broadcast_to(b_mid[:, :, None], shape))
    errors = np.full((S - 1, M, nb - 1, nz), np.nan)
    checked = np.zeros((S - 1, M, nb - 1, nz), dtype=bool)
    ages = range(S - 1) if ages is None else ages
    for s in ages:
        checked[s] = True
        # age s: savings by interpolation, then labor and consumption
        b_next = _interp_series(grid, sol.b_policy[s], b_q)
        n_guess = _interp_series(grid, sol.n_policy[s], b_q)
        n, c = _static_at(
            inp, backend, s, b_q, b_next, z_grid, n_guess, options
        )
        # age s + 1 in each next shock z': series (m, z') at b_next
        b_next_z = np.ascontiguousarray(
            np.broadcast_to(b_next[:, :, None, :], (M, nb - 1, nz, nz))
        )  # [m, i, z', z]: savings from shock z, evaluated on series z'
        n_guess1 = _interp_series(grid, sol.n_policy[s + 1], b_next_z)
        if s + 1 < S - 1:
            b_next2 = _interp_series(grid, sol.b_policy[s + 1], b_next_z)
        V = np.empty(shape + (nz,))
        for izp in range(nz):
            zs = np.full(nz, z_grid[izp])
            n1, c1 = _static_at(
                inp,
                backend,
                s + 1,
                b_next,
                None if s + 1 == S - 1 else b_next2[:, :, izp, :],
                zs,
                n_guess1[:, :, izp, :],
                options,
            )
            pa1 = _point_arrays(inp, s + 1, shape, zs)
            X = pa1["wez"] * n1.ravel() * inp.factor
            Y = inp.r[s + 1] * b_next.ravel() * inp.factor
            mtry = (
                backend.mtry_raw(s + 1, X, Y) * (1 - pa1["knc"]) * pa1["fi"]
            ).reshape(shape)
            mtrw = tax.MTR_wealth(
                b_next,
                inp.h_wealth[s + 1],
                inp.m_wealth[s + 1],
                inp.p_wealth[s + 1],
                pa1["fw"].reshape(shape),
            )
            V[..., izp] = (
                (1 + inp.r[s + 1] - inp.r[s + 1] * mtry - mtrw)
                * kernels.mu_c(c1, inp.sigma)
                / inp.p_tilde[s + 1]
            )
        EV = np.einsum("mikl,kl->mik", V, z_trans)
        mu_b = (
            kernels.mu_beq(b_next, inp.sigma, 1.0) * inp.chi_b[:, None, None]
        )
        rho = inp.rho[s][:, None, None]
        beta = inp.beta[:, None, None]
        v = (
            inp.p_tilde[s]
            * np.exp(-inp.sigma * inp.g_y)
            * (rho * mu_b + beta * (1 - rho) * EV)
        )
        c_euler = kernels.inv_mu_c(v, inp.sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            err = c_euler / c - 1
            at_limit = b_next <= grid[:, :1, None] + 1e-12
            errors[s] = np.where(
                at_limit, np.maximum(c / c_euler - 1, 0.0), err
            )
    weight = 0.5 * (sol.dist[:-1, :, 1:, :] + sol.dist[:-1, :, :-1, :])
    weight = np.where(checked, weight, 0.0)
    with_mass = checked & (weight > 0)
    nan = checked & np.isnan(errors)
    abs_err = np.where(nan, np.inf, np.abs(errors))  # NaN: failed solve
    total = max(weight.sum(), 1e-300)
    finite = checked & ~nan
    return {
        "errors": errors,
        "max": float(np.max(abs_err, where=checked, initial=0.0)),
        "max_with_mass": float(np.max(abs_err, where=with_mass, initial=0.0)),
        "max_with_mass_by_cell": np.max(
            abs_err, axis=(2, 3), where=with_mass, initial=0.0
        ),
        "mean_with_mass": float(
            (weight * np.where(finite, abs_err, 0.0)).sum()
            / max(weight[finite].sum(), 1e-300)
        ),
        "share_above": {
            t: float(weight[abs_err > t].sum() / total)
            for t in (1e-4, 1e-3, 1e-2)
        },
        "nan": int(nan.sum()),
        "nan_with_mass": int((nan & with_mass).sum()),
        "b": b_mid,
    }
