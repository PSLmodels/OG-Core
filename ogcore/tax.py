"""
------------------------------------------------------------------------
Functions for taxes in the steady state and along the transition path.
------------------------------------------------------------------------
"""

# Packages
import numpy as np
from ogcore import utils, pensions
from ogcore.txfunc import get_tax_rates

"""
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
"""


def ETR_wealth(b, h_wealth, m_wealth, p_wealth):
    r"""
    Calculates the effective tax rate on wealth.

    .. math::
        T_{j,s,t}^{w} = \frac{h^{w}p_{w}b_{j,s,t}}{h^{w}b_{j,s,t} + m^{w}}

    Args:
        b (Numpy array): savings
        h_wealth (scalar): parameter of wealth tax function
        p_wealth (scalar): parameter of wealth tax function
        m_wealth (scalar): parameter of wealth tax function

    Returns:
        tau_w (Numpy array): effective tax rate on wealth, size = SxJ

    """
    tau_w = (p_wealth * h_wealth * b) / (h_wealth * b + m_wealth)

    return tau_w


def MTR_wealth(b, h_wealth, m_wealth, p_wealth):
    r"""
    Calculates the marginal tax rate on wealth from the wealth tax.

    .. math::
        \frac{\partial T_{j,s,t}^{w}}{\partial b_{j,s,t}} =
        \frac{h^{w}p_{w}b_{j,s,t}}{(b_{j,s,t}h^{w}+m^{w})}\left[2 -
        \frac{h^{w}p_{w}b_{j,s,t}}{(b_{j,s,t}h^{w}+m^{w})}\right]

    Args:
        b (Numpy array): savings
        h_wealth (scalar): parameter of wealth tax function
        p_wealth (scalar): parameter of wealth tax function
        m_wealth (scalar): parameter of wealth tax function

    Returns:
        tau_prime (Numpy array): marginal tax rate on wealth, size = SxJ

    """
    tau_prime = ETR_wealth(b, h_wealth, m_wealth, p_wealth) * 2 - (
        (h_wealth**2 * p_wealth * b**2) / ((b * h_wealth + m_wealth) ** 2)
    )

    return tau_prime


def ETR_income(
    r,
    w,
    b,
    n,
    factor,
    e,
    etr_params,
    labor_noncompliance_rate,
    capital_noncompliance_rate,
    p,
):
    """
    Calculates effective personal income tax rate.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        factor (scalar): scaling factor converting model units to
            dollars
        e (Numpy array): effective labor units
        etr_params (list): list of effective tax rate function
            parameters or nonparametric function
        labor_noncompliance_rate (Numpy array): income tax noncompliance rate for labor income
        capital_noncompliance_rate (Numpy array): income tax noncompliance rate for capital income
        p (OG-Core Specifications object): model parameters

    Returns:
        tau (Numpy array): effective tax rate on total income

    """
    X = (w * e * n) * factor
    Y = (r * b) * factor
    noncompliance_rate = (
        (X * labor_noncompliance_rate) + (Y * capital_noncompliance_rate)
    ) / (X + Y)

    tau = get_tax_rates(
        etr_params, X, Y, None, p.tax_func_type, "etr", for_estimation=False
    )

    return tau * (1 - noncompliance_rate)


def MTR_income(
    r,
    w,
    b,
    n,
    factor,
    mtr_capital,
    e,
    etr_params,
    mtr_params,
    noncompliance_rate,
    p,
):
    r"""
    Generates the marginal tax rate on labor income for households.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        factor (scalar): scaling factor converting model units to
            dollars
        mtr_capital (bool): whether to compute the marginal tax rate on
            capital income or labor income
        e (Numpy array): effective labor units
        etr_params (list): list of effective tax rate function
            parameters or nonparametric function
        mtr_params (list): list of marginal tax rate function
            parameters or nonparametric function
        noncompliance_rate (Numpy array): income tax noncompliance rate
        p (OG-Core Specifications object): model parameters

    Returns:
        tau (Numpy array): marginal tax rate on income source

    """
    X = (w * e * n) * factor
    Y = (r * b) * factor

    if p.analytical_mtrs:
        tau = get_tax_rates(
            etr_params,
            X,
            Y,
            None,
            p.tax_func_type,
            "mtr",
            p.analytical_mtrs,
            mtr_capital,
            for_estimation=False,
        )
    else:
        tau = get_tax_rates(
            mtr_params,
            X,
            Y,
            None,
            p.tax_func_type,
            "mtr",
            p.analytical_mtrs,
            mtr_capital,
            for_estimation=False,
        )

    return tau * (1 - noncompliance_rate)


def get_biz_tax(w, Y, L, K, p_m, p, m, method):
    r"""
    Finds total business income tax revenue.

    .. math::
        R_{t}^{b} = \sum_{m=1}^{M}\tau_{m,t}^{b}(Y_{m,t} - w_{t}L_{m,t}) -
        \tau_{m,t}^{b}\delta_{m,t}^{\tau}K_{m,t}^{\tau} - \tau^{inv}_{m,t}I_{m,t}
    Args:
        r (array_like): real interest rate
        Y (array_like): aggregate output for each industry
        L (array_like): aggregate labor demand for each industry
        K (array_like): aggregate capital demand for each industry
        p_m (array_like): output prices
        p (OG-Core Specifications object): model parameters
        m (int or None): index for production industry, if None, then
            compute for all industries
    Returns:
        business_revenue (array_like): aggregate business tax revenue

    """
    if m is not None:
        if method == "SS":
            delta_tau = p.delta_tau[-1, m]
            tau_b = p.tau_b[-1, m]
            tau_inv = p.inv_tax_credit[-1, m]
            price = p_m[m]
            Inv = p.delta * K[m]  # compute gross investment
        else:
            delta_tau = p.delta_tau[: p.T, m].reshape(p.T)
            tau_b = p.tau_b[: p.T, m].reshape(p.T)
            tau_inv = p.inv_tax_credit[: p.T, m].reshape(p.T)
            price = p_m[: p.T, m].reshape(p.T)
            w = w.reshape(p.T)
            Inv = p.delta * K
    else:
        if method == "SS":
            delta_tau = p.delta_tau[-1, :]
            tau_b = p.tau_b[-1, :]
            tau_inv = p.inv_tax_credit[-1, :]
            price = p_m
            Inv = p.delta * K
        else:
            delta_tau = p.delta_tau[: p.T, :].reshape(p.T, p.M)
            tau_b = p.tau_b[: p.T, :].reshape(p.T, p.M)
            tau_inv = p.inv_tax_credit[: p.T, :].reshape(p.T, p.M)
            price = p_m[: p.T, :].reshape(p.T, p.M)
            w = w.reshape(p.T, 1)
            Inv = p.delta * K

    business_revenue = (
        tau_b * (price * Y - w * L) - tau_b * delta_tau * K - tau_inv * Inv
    )
    return business_revenue


def net_taxes(
    r,
    w,
    b,
    n,
    bq,
    factor,
    tr,
    ubi,
    theta,
    t,
    j,
    shift,
    method,
    e,
    etr_params,
    p,
):
    """
    Calculate net taxes paid for each household.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        bq (Numpy array): bequests received
        factor (scalar): scaling factor converting model units to
            dollars
        tr (Numpy array): government transfers to the household
        ubi (Numpy array): universal basic income payments to households
        theta (Numpy array): social security replacement rate value for
            lifetime income group j
        t (int): time period
        j (int): index of lifetime income group
        shift (bool): whether computing for periods 0--s or 1--(s+1),
            =True for 1--(s+1)
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        e (Numpy array): effective labor units
        etr_params (list): list of effective tax rate function parameters
        p (OG-Core Specifications object): model parameters

    Returns:
        net_tax (Numpy array): net taxes paid for each household

    """
    T_I = income_tax_liab(r, w, b, n, factor, t, j, method, e, etr_params, p)
    # TODO: replace "1" with Y in the args below when want NDC functions
    pension = pensions.pension_amount(
        r, w, n, 1, theta, t, j, shift, method, e, factor, p
    )
    T_BQ = bequest_tax_liab(r, b, bq, t, j, method, p)
    T_W = wealth_tax_liab(r, b, t, j, method, p)

    net_tax = T_I - pension + T_BQ + T_W - tr - ubi

    return net_tax


def income_tax_liab(r, w, b, n, factor, t, j, method, e, etr_params, p):
    """
    Calculate income and payroll tax liability for each household

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        factor (scalar): scaling factor converting model units to
            dollars
        t (int): time period
        j (int): index of lifetime income group
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        e (Numpy array): effective labor units
        etr_params (list): effective tax rate function parameters
        p (OG-Core Specifications object): model parameters

    Returns:
        T_I (Numpy array): total income and payroll taxes paid for each
            household

    """
    if j is not None:
        if method == "TPI":
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
                w = w.reshape(w.shape[0], 1)
            labor_income_tax_compliance_rate = (
                p.labor_income_tax_noncompliance_rate[t, j]
            )
            capital_income_tax_compliance_rate = (
                p.capital_income_tax_noncompliance_rate[t, j]
            )
        else:
            labor_income_tax_compliance_rate = (
                p.labor_income_tax_noncompliance_rate[-1, j]
            )
            capital_income_tax_compliance_rate = (
                p.capital_income_tax_noncompliance_rate[-1, j]
            )
    else:
        if method == "TPI":
            r = utils.to_timepath_shape(r)
            w = utils.to_timepath_shape(w)
            labor_income_tax_compliance_rate = (
                p.labor_income_tax_noncompliance_rate[t, :]
            )
            capital_income_tax_compliance_rate = (
                p.capital_income_tax_noncompliance_rate[t, :]
            )
        else:
            labor_income_tax_compliance_rate = (
                p.labor_income_tax_noncompliance_rate[-1, :]
            )
            capital_income_tax_compliance_rate = (
                p.capital_income_tax_noncompliance_rate[-1, :]
            )
    income = r * b + w * e * n
    labor_income = w * e * n
    T_I = (
        ETR_income(
            r,
            w,
            b,
            n,
            factor,
            e,
            etr_params,
            labor_income_tax_compliance_rate,
            capital_income_tax_compliance_rate,
            p,
        )
        * income
    )
    if method == "SS":
        T_P = p.tau_payroll[-1] * labor_income
    elif method == "TPI":
        length = w.shape[0]
        if len(b.shape) == 1:
            T_P = p.tau_payroll[t : t + length] * labor_income
        elif len(b.shape) == 2:
            T_P = (
                p.tau_payroll[t : t + length].reshape(length, 1) * labor_income
            )
        else:
            T_P = (
                p.tau_payroll[t : t + length].reshape(length, 1, 1)
                * labor_income
            )
    elif method == "TPI_scalar":
        T_P = p.tau_payroll[0] * labor_income

    income_payroll_tax_liab = T_I + T_P

    return income_payroll_tax_liab


def wealth_tax_liab(r, b, t, j, method, p):
    """
    Calculate wealth tax liability for each household.

    Args:
        r (array_like): real interest rate
        b (Numpy array): savings
        t (int): time period
        j (int): index of lifetime income group
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        p (OG-Core Specifications object): model parameters

    Returns:
        T_W (Numpy array): wealth tax liability for each household

    """
    if j is not None:
        if method == "TPI":
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
    else:
        if method == "TPI":
            r = utils.to_timepath_shape(r)

    if method == "SS":
        T_W = ETR_wealth(b, p.h_wealth[-1], p.m_wealth[-1], p.p_wealth[-1]) * b
    elif method == "TPI":
        length = r.shape[0]
        if len(b.shape) == 1:
            T_W = (
                ETR_wealth(
                    b,
                    p.h_wealth[t : t + length],
                    p.m_wealth[t : t + length],
                    p.p_wealth[t : t + length],
                )
                * b
            )
        elif len(b.shape) == 2:
            T_W = (
                ETR_wealth(
                    b,
                    p.h_wealth[t : t + length],
                    p.m_wealth[t : t + length],
                    p.p_wealth[t : t + length],
                )
                * b
            )
        else:
            T_W = (
                ETR_wealth(
                    b,
                    p.h_wealth[t : t + length].reshape(length, 1, 1),
                    p.m_wealth[t : t + length].reshape(length, 1, 1),
                    p.p_wealth[t : t + length].reshape(length, 1, 1),
                )
                * b
            )
    elif method == "TPI_scalar":
        T_W = ETR_wealth(b, p.h_wealth[0], p.m_wealth[0], p.p_wealth[0]) * b

    return T_W


def bequest_tax_liab(r, b, bq, t, j, method, p):
    """
    Calculate liability due from taxes on bequests for each household.

    Args:
        r (array_like): real interest rate
        b (Numpy array): savings
        bq (Numpy array): bequests received
        t (int): time period
        j (int): index of lifetime income group
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        p (OG-Core Specifications object): model parameters

    Returns:
        T_BQ (Numpy array): bequest tax liability for each household

    """
    if j is not None:
        lambdas = p.lambdas[j]
        if method == "TPI":
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
    else:
        lambdas = np.transpose(p.lambdas)
        if method == "TPI":
            r = utils.to_timepath_shape(r)

    if method == "SS":
        T_BQ = p.tau_bq[-1] * bq
    elif method == "TPI":
        length = r.shape[0]
        if len(b.shape) == 1:
            T_BQ = p.tau_bq[t : t + length] * bq
        elif len(b.shape) == 2:
            T_BQ = p.tau_bq[t : t + length].reshape(length, 1) * bq / lambdas
        else:
            T_BQ = p.tau_bq[t : t + length].reshape(length, 1, 1) * bq
    elif method == "TPI_scalar":
        # The above methods won't work if scalars are used.  This option
        # is only called by the SS_TPI_firstdoughnutring function in TPI.
        T_BQ = p.tau_bq[0] * bq

    return T_BQ
