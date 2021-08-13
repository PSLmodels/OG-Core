'''
------------------------------------------------------------------------
Functions for taxes in the steady state and along the transition path.
------------------------------------------------------------------------
'''

# Packages
import numpy as np
from ogcore import utils
from ogcore.txfunc import get_tax_rates

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def replacement_rate_vals(nssmat, wss, factor_ss, j, p):
    '''
    Calculates replacement rate values for the social security system.

    Args:
        nssmat (Numpy array): initial guess at labor supply, size = SxJ
        new_w (scalar): steady state real wage rate
        factor_ss (scalar): scaling factor converting model units to
            dollars
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        theta (Numpy array): social security replacement rate value for
            lifetime income group j

    '''
    if j is not None:
        e = p.e[:, j]
    else:
        e = p.e
    # adjust number of calendar years AIME computed from int model periods
    equiv_periods = int(round((p.S / 80.0) * p.AIME_num_years)) - 1
    if e.ndim == 2:
        dim2 = e.shape[1]
    else:
        dim2 = 1
    earnings = (e * (wss * nssmat * factor_ss)).reshape(p.S, dim2)
    # get highest earning years for number of years AIME computed from
    highest_earn =\
        (-1.0 * np.sort(-1.0 * earnings[:p.retire[-1], :],
                        axis=0))[:equiv_periods]
    AIME = highest_earn.sum(0) / ((12.0 * (p.S / 80.0)) * equiv_periods)
    PIA = np.zeros(dim2)
    # Compute level of replacement using AIME brackets and PIA rates
    for j in range(dim2):
        if AIME[j] < p.AIME_bkt_1:
            PIA[j] = p.PIA_rate_bkt_1 * AIME[j]
        elif AIME[j] < p.AIME_bkt_2:
            PIA[j] = (p.PIA_rate_bkt_1 * p.AIME_bkt_1 +
                      p.PIA_rate_bkt_2 * (AIME[j] - p.AIME_bkt_1))
        else:
            PIA[j] = (p.PIA_rate_bkt_1 * p.AIME_bkt_1 +
                      p.PIA_rate_bkt_2 * (p.AIME_bkt_2 - p.AIME_bkt_1) +
                      p.PIA_rate_bkt_3 * (AIME[j] - p.AIME_bkt_2))
    # Set the maximum monthly replacment rate from SS benefits tables
    PIA[PIA > p.PIA_maxpayment] = p.PIA_maxpayment
    if p.PIA_minpayment != 0.0:
        PIA[PIA < p.PIA_minpayment] = p.PIA_minpayment
    theta = (PIA * (12.0 * p.S / 80.0)) / (factor_ss * wss)
    return theta


def ETR_wealth(b, h_wealth, m_wealth, p_wealth):
    r'''
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

    '''
    tau_w = (p_wealth * h_wealth * b) / (h_wealth * b + m_wealth)
    return tau_w


def MTR_wealth(b, h_wealth, m_wealth, p_wealth):
    r'''
    Calculates the marginal tax rate on wealth from the wealth tax.

    .. math::
        \frac{\partial T_{j,s,t}^{w}}{\partial b_{j,s,t}} =
        \frac{h^{w}m^{w}p_{w}}{(b_{j,s,t}h^{w}m^{w})^{2}}

    Args:
        b (Numpy array): savings
        h_wealth (scalar): parameter of wealth tax function
        p_wealth (scalar): parameter of wealth tax function
        m_wealth (scalar): parameter of wealth tax function

    Returns:
        tau_prime (Numpy array): marginal tax rate on wealth, size = SxJ

    '''
    tau_prime = ((b * h_wealth * m_wealth * p_wealth) /
                 ((b * h_wealth + m_wealth) ** 2) +
                 ETR_wealth(b, h_wealth, m_wealth, p_wealth))
    return tau_prime


def ETR_income(r, w, b, n, factor, e, etr_params, p):
    '''
    Calculates effective personal income tax rate.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        factor (scalar): scaling factor converting model units to
            dollars
        e (Numpy array): effective labor units
        etr_params (Numpy array): effective tax rate function parameters
        p (OG-Core Specifications object): model parameters

    Returns:
        tau (Numpy array): effective tax rate on total income

    '''
    X = (w * e * n) * factor
    Y = (r * b) * factor

    tau = get_tax_rates(etr_params, X, Y, None, p.tax_func_type, 'etr',
                        for_estimation=False)

    return tau


def MTR_income(r, w, b, n, factor, mtr_capital, e, etr_params,
               mtr_params, p):
    r'''
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
        etr_params (Numpy array): effective tax rate function parameters
        p (OG-Core Specifications object): model parameters

    Returns:
        tau (Numpy array): marginal tax rate on income source

    '''
    X = (w * e * n) * factor
    Y = (r * b) * factor

    if p.analytical_mtrs:
        tau = get_tax_rates(
            etr_params, X, Y, None, p.tax_func_type, 'mtr',
            p.analytical_mtrs, mtr_capital, for_estimation=False)
    else:
        tau = get_tax_rates(
            mtr_params, X, Y, None, p.tax_func_type, 'mtr',
            p.analytical_mtrs, mtr_capital, for_estimation=False)

    return tau


def get_biz_tax(w, Y, L, K, p, method):
    r'''
    Finds total business income tax revenue.

    .. math::
        R_{t}^{b} = \tau_{t}^{b}(Y_{t} - w_{t}L_{t}) -
        \tau_{t}^{b}\delta_{t}^{\tau}K_{t}^{\tau}
    Args:
        r (array_like): real interest rate
        Y (array_like): aggregate output
        L (array_like): aggregate labor demand
        K (array_like): aggregate capital demand

    Returns:
        business_revenue (array_like): aggregate business tax revenue

    '''
    if method == 'SS':
        delta_tau = p.delta_tau[-1]
        tau_b = p.tau_b[-1]
    else:
        delta_tau = p.delta_tau[:p.T]
        tau_b = p.tau_b[:p.T]
    business_revenue = tau_b * (Y - w * L) - tau_b * delta_tau * K
    return business_revenue


def net_taxes(r, w, b, n, bq, factor, tr, ubi, theta, t, j, shift, method,
              e, etr_params, p):
    '''
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
        etr_params (Numpy array): effective tax rate function parameters
        p (OG-Core Specifications object): model parameters

    Returns:
        net_tax (Numpy array): net taxes paid for each household

    '''
    T_I = income_tax_liab(r, w, b, n, factor, t, j, method, e, etr_params, p)
    pension = pension_amount(w, n, theta, t, j, shift, method, e, p)
    T_BQ = bequest_tax_liab(r, b, bq, t, j, method, p)
    T_W = wealth_tax_liab(r, b, t, j, method, p)

    net_tax = T_I - pension + T_BQ + T_W - tr - ubi

    return net_tax


def income_tax_liab(r, w, b, n, factor, t, j, method, e, etr_params, p):
    '''
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
        etr_params (Numpy array): effective tax rate function parameters
        p (OG-Core Specifications object): model parameters

    Returns:
        T_I (Numpy array): total income and payroll taxes paid for each
            household

    '''
    if j is not None:
        if method == 'TPI':
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
                w = w.reshape(w.shape[0], 1)
    else:
        if method == 'TPI':
            r = utils.to_timepath_shape(r)
            w = utils.to_timepath_shape(w)

    income = r * b + w * e * n
    labor_income = w * e * n
    T_I = ETR_income(r, w, b, n, factor, e, etr_params, p) * income
    if method == 'SS':
        T_P = p.tau_payroll[-1] * labor_income
    elif method == 'TPI':
        length = w.shape[0]
        if len(b.shape) == 1:
            T_P = p.tau_payroll[t: t + length] * labor_income
        elif len(b.shape) == 2:
            T_P = (p.tau_payroll[t: t + length].reshape(length, 1) *
                   labor_income)
        else:
            T_P = (p.tau_payroll[t:t + length].reshape(length, 1, 1) *
                   labor_income)
    elif method == 'TPI_scalar':
        T_P = p.tau_payroll[0] * labor_income

    income_payroll_tax_liab = T_I + T_P

    return income_payroll_tax_liab


def pension_amount(w, n, theta, t, j, shift, method, e, p):
    '''
    Calculate public pension benefit amounts for each household.

    Args:
        w (array_like): real wage rate
        n (Numpy array): labor supply
        theta (Numpy array): social security replacement rate value for
            lifetime income group j
        t (int): time period
        j (int): index of lifetime income group
        shift (bool): whether computing for periods 0--s or 1--(s+1),
            =True for 1--(s+1)
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        e (Numpy array): effective labor units
        p (OG-Core Specifications object): model parameters

    Returns:
        pension (Numpy array): pension amount for each household

    '''
    if j is not None:
        if method == 'TPI':
            if n.ndim == 2:
                w = w.reshape(w.shape[0], 1)
    else:
        if method == 'TPI':
            w = utils.to_timepath_shape(w)

    pension = np.zeros_like(n)
    if method == 'SS':
        # Depending on if we are looking at b_s or b_s+1, the
        # entry for retirement will change (it shifts back one).
        # The shift boolean makes sure we start replacement rates
        # at the correct age.
        if shift is False:
            pension[p.retire[-1]:] = theta * w
        else:
            pension[p.retire[-1] - 1:] = theta * w
    elif method == 'TPI':
        length = w.shape[0]
        if not shift:
            # retireTPI is different from retire, because in TP income
            # we are counting backwards with different length lists.
            # This will always be the correct location of retirement,
            # depending on the shape of the lists.
            retireTPI = (p.retire[t: t + length] - p.S)
        else:
            retireTPI = (p.retire[t: t + length] - 1 - p.S)
        if len(n.shape) == 1:
            if not shift:
                retireTPI = p.retire[t] - p.S
            else:
                retireTPI = p.retire[t] - 1 - p.S
            pension[retireTPI:] = (
                theta[j] * p.replacement_rate_adjust[t] * w[retireTPI:])
        elif len(n.shape) == 2:
            for tt in range(pension.shape[0]):
                pension[tt, retireTPI[tt]:] = (
                    theta * p.replacement_rate_adjust[t + tt] * w[tt])
        else:
            for tt in range(pension.shape[0]):
                pension[tt, retireTPI[tt]:, :] = (
                    theta.reshape(1, p.J) *
                    p.replacement_rate_adjust[t + tt] * w[tt])
    elif method == 'TPI_scalar':
        # The above methods won't work if scalars are used.  This option
        # is only called by the SS_TPI_firstdoughnutring function in TPI.
        pension = theta * p.replacement_rate_adjust[0] * w

    return pension


def wealth_tax_liab(r, b, t, j, method, p):
    '''
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

    '''
    if j is not None:
        if method == 'TPI':
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
    else:
        if method == 'TPI':
            r = utils.to_timepath_shape(r)

    if method == 'SS':
        T_W = (ETR_wealth(b, p.h_wealth[-1], p.m_wealth[-1],
                          p.p_wealth[-1]) * b)
    elif method == 'TPI':
        length = r.shape[0]
        if len(b.shape) == 1:
            T_W = (ETR_wealth(b, p.h_wealth[t:t + length],
                              p.m_wealth[t:t + length],
                              p.p_wealth[t:t + length]) * b)
        elif len(b.shape) == 2:
            T_W = (ETR_wealth(b, p.h_wealth[t:t + length],
                              p.m_wealth[t:t + length],
                              p.p_wealth[t:t + length]) * b)
        else:
            T_W = (ETR_wealth(
                b, p.h_wealth[t:t + length].reshape(length, 1, 1),
                p.m_wealth[t:t + length].reshape(length, 1, 1),
                p.p_wealth[t:t + length].reshape(length, 1, 1)) * b)
    elif method == 'TPI_scalar':
        T_W = (ETR_wealth(b, p.h_wealth[0], p.m_wealth[0],
                          p.p_wealth[0]) * b)

    return T_W


def bequest_tax_liab(r, b, bq, t, j, method, p):
    '''
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

    '''
    if j is not None:
        lambdas = p.lambdas[j]
        if method == 'TPI':
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
    else:
        lambdas = np.transpose(p.lambdas)
        if method == 'TPI':
            r = utils.to_timepath_shape(r)

    if method == 'SS':
        T_BQ = p.tau_bq[-1] * bq
    elif method == 'TPI':
        length = r.shape[0]
        if len(b.shape) == 1:
            T_BQ = p.tau_bq[t:t + length] * bq
        elif len(b.shape) == 2:
            T_BQ = p.tau_bq[t:t + length].reshape(length, 1) * bq / lambdas
        else:
            T_BQ = p.tau_bq[t:t + length].reshape(length, 1, 1) * bq
    elif method == 'TPI_scalar':
        # The above methods won't work if scalars are used.  This option
        # is only called by the SS_TPI_firstdoughnutring function in TPI.
        T_BQ = p.tau_bq[0] * bq

    return T_BQ
