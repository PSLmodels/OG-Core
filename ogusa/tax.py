'''
------------------------------------------------------------------------
Functions for taxes in the steady state and along the transition path.
------------------------------------------------------------------------
'''

# Packages
import numpy as np
from ogusa import utils, household

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
        p (OG-USA Specifications object): model parameters

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
        \frac{\partial T_{j,s,t}^{w}}{\partial b_{j,s,t}} = \frac{h^{w}m^{w}p_{w}}{(b_{j,s,t}h^{w}m^{w})^{2}}

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
        p (OG-USA Specifications object): model parameters

    Returns:
        tau (Numpy array): effective tax rate on total income

    '''
    X = (w * e * n) * factor
    Y = (r * b) * factor
    X2 = X ** 2
    Y2 = Y ** 2
    income = X + Y
    income2 = income ** 2

    if p.tax_func_type == 'GS':
        phi0 = np.squeeze(etr_params[..., 0])
        phi1 = np.squeeze(etr_params[..., 1])
        phi2 = np.squeeze(etr_params[..., 2])
        tau = ((phi0 * (income - ((income ** -phi1) + phi2) **
                        (-1 / phi1))) / income)
    elif p.tax_func_type == 'DEP_totalinc':
        A = np.squeeze(etr_params[..., 0])
        B = np.squeeze(etr_params[..., 1])
        max_income = np.squeeze(etr_params[..., 4])
        min_income = np.squeeze(etr_params[..., 5])
        shift_income = np.squeeze(etr_params[..., 8])
        shift = np.squeeze(etr_params[..., 10])
        tau_income = (((max_income - min_income) *
                       (A * income2 + B * income) /
                       (A * income2 + B * income + 1)) + min_income)
        tau = tau_income + shift_income + shift
    else:  # DEP or linear
        A = np.squeeze(etr_params[..., 0])
        B = np.squeeze(etr_params[..., 1])
        C = np.squeeze(etr_params[..., 2])
        D = np.squeeze(etr_params[..., 3])
        max_x = np.squeeze(etr_params[..., 4])
        min_x = np.squeeze(etr_params[..., 5])
        max_y = np.squeeze(etr_params[..., 6])
        min_y = np.squeeze(etr_params[..., 7])
        shift_x = np.squeeze(etr_params[..., 8])
        shift_y = np.squeeze(etr_params[..., 9])
        shift = np.squeeze(etr_params[..., 10])
        share = np.squeeze(etr_params[..., 11])

        tau_x = ((max_x - min_x) * (A * X2 + B * X) /
                 (A * X2 + B * X + 1) + min_x)
        tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
                 (C * Y2 + D * Y + 1) + min_y)
        tau = (((tau_x + shift_x) ** share) *
               ((tau_y + shift_y) ** (1 - share))) + shift

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
        p (OG-USA Specifications object): model parameters

    Returns:
        tau (Numpy array): marginal tax rate on income source

    '''
    X = (w * e * n) * factor
    Y = (r * b) * factor
    X2 = X ** 2
    Y2 = Y ** 2
    income = X + Y
    income2 = income ** 2

    if p.tax_func_type == 'GS':
        if p.analytical_mtrs:
            phi0 = np.squeeze(etr_params[..., 0])
            phi1 = np.squeeze(etr_params[..., 1])
            phi2 = np.squeeze(etr_params[..., 2])
        else:
            phi0 = np.squeeze(mtr_params[..., 0])
            phi1 = np.squeeze(mtr_params[..., 1])
            phi2 = np.squeeze(mtr_params[..., 2])
        tau = (phi0*(1 - (income ** (-phi1 - 1) *
                          ((income ** -phi1) + phi2) **
                          ((-1 - phi1) / phi1))))
    elif p.tax_func_type == 'DEP_totalinc':
        if p.analytical_mtrs:
            A = np.squeeze(etr_params[..., 0])
            B = np.squeeze(etr_params[..., 1])
            max_income = np.squeeze(etr_params[..., 4])
            min_income = np.squeeze(etr_params[..., 5])
            shift_income = np.squeeze(etr_params[..., 8])
            shift = np.squeeze(etr_params[..., 10])
            d_etr = ((max_income - min_income) * ((2 * A * income + B) /
                     ((A * income2 + B * income + 1) ** 2)))
            etr = (((max_income - min_income) *
                    ((A * income2 + B * income) /
                     (A * income2 + B * income + 1)) + min_income) +
                   shift_income + shift)
            tau = (d_etr * income) + (etr)
        else:
            A = np.squeeze(mtr_params[..., 0])
            B = np.squeeze(mtr_params[..., 1])
            max_income = np.squeeze(mtr_params[..., 4])
            min_income = np.squeeze(mtr_params[..., 5])
            shift_income = np.squeeze(mtr_params[..., 8])
            shift = np.squeeze(mtr_params[..., 10])
            tau_income = (((max_income - min_income) *
                           (A * income2 + B * income) /
                           (A * income2 + B * income + 1)) + min_income)
            tau = tau_income + shift_income + shift
    else:  # DEP or linear
        if p.analytical_mtrs:
            A = np.squeeze(etr_params[..., 0])
            B = np.squeeze(etr_params[..., 1])
            C = np.squeeze(etr_params[..., 2])
            D = np.squeeze(etr_params[..., 3])
            max_x = np.squeeze(etr_params[..., 4])
            min_x = np.squeeze(etr_params[..., 5])
            max_y = np.squeeze(etr_params[..., 6])
            min_y = np.squeeze(etr_params[..., 7])
            shift_x = np.squeeze(etr_params[..., 8])
            shift_y = np.squeeze(etr_params[..., 9])
            shift = np.squeeze(etr_params[..., 10])
            share = np.squeeze(etr_params[..., 11])

            tau_x = ((max_x - min_x) * (A * X2 + B * X) /
                     (A * X2 + B * X + 1) + min_x)
            tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
                     (C * Y2 + D * Y + 1) + min_y)
            etr = (((tau_x + shift_x) ** share) *
                   ((tau_y + shift_y) ** (1 - share))) + shift
            if mtr_capital:
                d_etr = ((1-share) * ((tau_y + shift_y) ** (-share)) *
                         (max_y - min_y) * ((2 * C * Y + D) /
                                            ((C * Y2 + D * Y + 1)
                                             ** 2)) *
                         ((tau_x + shift_x) ** share))
                tau = d_etr * income + etr
            else:
                d_etr = (share * ((tau_x + shift_x) ** (share - 1)) *
                         (max_x - min_x) * ((2 * A * X + B) /
                                            ((A * X2 + B * X + 1)
                                             ** 2)) *
                         ((tau_y + shift_y) ** (1 - share)))
                tau = d_etr * income + etr
        else:
            A = np.squeeze(mtr_params[..., 0])
            B = np.squeeze(mtr_params[..., 1])
            C = np.squeeze(mtr_params[..., 2])
            D = np.squeeze(mtr_params[..., 3])
            max_x = np.squeeze(mtr_params[..., 4])
            min_x = np.squeeze(mtr_params[..., 5])
            max_y = np.squeeze(mtr_params[..., 6])
            min_y = np.squeeze(mtr_params[..., 7])
            shift_x = np.squeeze(mtr_params[..., 8])
            shift_y = np.squeeze(mtr_params[..., 9])
            shift = np.squeeze(mtr_params[..., 10])
            share = np.squeeze(mtr_params[..., 11])

            tau_x = ((max_x - min_x) * (A * X2 + B * X) /
                     (A * X2 + B * X + 1) + min_x)
            tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
                     (C * Y2 + D * Y + 1) + min_y)
            tau = (((tau_x + shift_x) ** share) *
                   ((tau_y + shift_y) ** (1 - share))) + shift

    return tau


def get_biz_tax(w, Y, L, K, p, method):
    r'''
    Finds total business income tax revenue.

    .. math::
        R_{t}^{b} = \tau_{t}^{b}(Y_{t} - w_{t}L_{t}) - \tau_{t}^{b}\delta_{t}^{\tau}K_{t}^{\tau}
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


def total_taxes(r, w, b, n, bq, factor, tr, theta, t, j, shift, method,
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
        p (OG-USA Specifications object): model parameters

    Returns:
        total_taxes (Numpy array): net taxes paid for each household

    '''
    if j is not None:
        lambdas = p.lambdas[j]
        if method == 'TPI':
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
                w = w.reshape(w.shape[0], 1)
    else:
        lambdas = np.transpose(p.lambdas)
        if method == 'TPI':
            r = utils.to_timepath_shape(r)
            w = utils.to_timepath_shape(w)

    income = r * b  + w * e * n
    T_I = ETR_income(r, w, b, n, factor, e, etr_params, p) * income

    if method == 'SS':
        # Depending on if we are looking at b_s or b_s+1, the
        # entry for retirement will change (it shifts back one).
        # The shift boolean makes sure we start replacement rates
        # at the correct age.
        T_P = p.tau_payroll[-1] * w * e * n
        if shift is False:
            T_P[p.retire[-1]:] -= theta * w
        else:
            T_P[p.retire[-1] - 1:] -= theta * w
        T_BQ = p.tau_bq[-1] * bq
        T_W = (ETR_wealth(b, p.h_wealth[-1], p.m_wealth[-1],
                          p.p_wealth[-1]) * b)
    elif method == 'TPI':
        length = w.shape[0]
        if not shift:
            # retireTPIis different from retire, because in TPincomewe are
            # counting backwards with different length lists.  This will
            # always be the correct location of retirement, depending
            # on the shape of the lists.
            retireTPI = (p.retire[t: t + length] - p.S)
        else:
            retireTPI = (p.retire[t: t + length] - 1 - p.S)
        if len(b.shape) == 1:
            T_P = p.tau_payroll[t: t + length] * w * e * n
            if not shift:
                retireTPI = p.retire[t] - p.S
            else:
                retireTPI = p.retire[t] - 1 - p.S
            T_P[retireTPI:] -= (theta[j] * p.replacement_rate_adjust[t]
                                * w[retireTPI:])
            T_W = (ETR_wealth(b, p.h_wealth[t:t + length],
                              p.m_wealth[t:t + length],
                              p.p_wealth[t:t + length]) * b)
            T_BQ = p.tau_bq[t:t + length] * bq
        elif len(b.shape) == 2:
            T_P = p.tau_payroll[t: t + length].reshape(length, 1) * w * e * n
            for tt in range(T_P.shape[0]):
                T_P[tt, retireTPI[tt]:] -= (
                    theta * p.replacement_rate_adjust[t + tt] * w[tt])
            T_W = (ETR_wealth(b, p.h_wealth[t:t + length],
                              p.m_wealth[t:t + length],
                              p.p_wealth[t:t + length]) * b)
            T_BQ = p.tau_bq[t:t + length].reshape(length, 1) * bq / lambdas
        else:
            T_P = p.tau_payroll[t:t + length].reshape(length, 1, 1) * w * e * n
            for tt in range(T_P.shape[0]):
                T_P[tt, retireTPI[tt]:, :] -= (
                    theta.reshape(1, p.J) *
                    p.replacement_rate_adjust[t + tt] * w[tt])
            T_W = (ETR_wealth(
                b, p.h_wealth[t:t + length].reshape(length, 1, 1),
                p.m_wealth[t:t + length].reshape(length, 1, 1),
                p.p_wealth[t:t + length].reshape(length, 1, 1)) * b)
            T_BQ = p.tau_bq[t:t + length].reshape(length, 1, 1) * bq
    elif method == 'TPI_scalar':
        # The above methods won't work if scalars are used.  This option
        # is only called by the SS_TPI_firstdoughnutring function in TPI.
        T_P = p.tau_payroll[0] * w * e * n
        T_P -= theta * p.replacement_rate_adjust[0] * w
        T_BQ = p.tau_bq[0] * bq
        T_W = (ETR_wealth(b, p.h_wealth[0], p.m_wealth[0],
                          p.p_wealth[0]) * b)
    total_tax = T_I + T_P + T_BQ + T_W - tr

    return total_tax
