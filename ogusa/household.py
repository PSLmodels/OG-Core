'''
------------------------------------------------------------------------
Household functions for taxes in the steady state and along the
transition path..

This file calls the following files:
    tax.py
------------------------------------------------------------------------
'''

# Packages
import numpy as np
from ogusa import tax, utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def marg_ut_cons(c, sigma):
    '''
    Computation of marginal utility of consumption.
    Inputs:
        c     = [T,S,J] array, household consumption
        sigma = scalar, coefficient of relative risk aversion
    Functions called: None
    Objects in function:
        output = [T,S,J] array, marginal utility of consumption
    Returns: output
    '''
    if np.ndim(c) == 0:
        c = np.array([c])
    epsilon = 0.003
    cvec_cnstr = c < epsilon
    MU_c = np.zeros(c.shape)
    MU_c[~cvec_cnstr] = c[~cvec_cnstr] ** (-sigma)
    b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
    b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
    MU_c[cvec_cnstr] = 2 * b2 * c[cvec_cnstr] + b1
    output = MU_c
    output = np.squeeze(output)

    return output


def marg_ut_labor(n, chi_n, p):
    '''
    Computation of marginal disutility of labor.
    Inputs:
        n         = [T,S,J] array, household labor supply
        params    = length 4 tuple (b_ellipse, upsilon, ltilde, chi_n)
        b_ellipse = scalar, scaling parameter in elliptical utility function
        upsilon   = curvature parameter in elliptical utility function
        ltilde    = scalar, upper bound of household labor supply
        chi_n     = [S,] vector, utility weights on disutility of labor
    Functions called: None
    Objects in function:
        output = [T,S,J] array, marginal disutility of labor supply
    Returns: output
    '''
    nvec = n
    if np.ndim(nvec) == 0:
        nvec = np.array([nvec])
    eps_low = 0.000001
    eps_high = p.ltilde - 0.000001
    nvec_low = nvec < eps_low
    nvec_high = nvec > eps_high
    nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
    MDU_n = np.zeros(nvec.shape)
    MDU_n[nvec_uncstr] = (
        (p.b_ellipse / p.ltilde) *
        ((nvec[nvec_uncstr] / p.ltilde) ** (p.upsilon - 1)) *
        ((1 - ((nvec[nvec_uncstr] / p.ltilde) ** p.upsilon)) **
         ((1 - p.upsilon) / p.upsilon)))
    b2 = (0.5 * p.b_ellipse * (p.ltilde ** (-p.upsilon)) * (p.upsilon - 1) *
          (eps_low ** (p.upsilon - 2)) *
          ((1 - ((eps_low / p.ltilde) ** p.upsilon)) **
          ((1 - p.upsilon) / p.upsilon)) *
          (1 + ((eps_low / p.ltilde) ** p.upsilon) *
          ((1 - ((eps_low / p.ltilde) ** p.upsilon)) ** (-1))))
    b1 = ((p.b_ellipse / p.ltilde) * ((eps_low / p.ltilde) **
                                      (p.upsilon - 1)) *
          ((1 - ((eps_low / p.ltilde) ** p.upsilon)) **
          ((1 - p.upsilon) / p.upsilon)) - (2 * b2 * eps_low))
    MDU_n[nvec_low] = 2 * b2 * nvec[nvec_low] + b1
    d2 = (0.5 * p.b_ellipse * (p.ltilde ** (-p.upsilon)) * (p.upsilon - 1) *
          (eps_high ** (p.upsilon - 2)) *
          ((1 - ((eps_high / p.ltilde) ** p.upsilon)) **
          ((1 - p.upsilon) / p.upsilon)) *
          (1 + ((eps_high / p.ltilde) ** p.upsilon) *
          ((1 - ((eps_high / p.ltilde) ** p.upsilon)) ** (-1))))
    d1 = ((p.b_ellipse / p.ltilde) * ((eps_high / p.ltilde) **
          (p.upsilon - 1)) * ((1 - ((eps_high / p.ltilde) ** p.upsilon)) **
          ((1 - p.upsilon) / p.upsilon)) - (2 * d2 * eps_high))
    MDU_n[nvec_high] = 2 * d2 * nvec[nvec_high] + d1
    output = MDU_n * np.squeeze(chi_n)
    output = np.squeeze(output)
    return output


def get_bq(BQ, j, p, method):
    '''
    Calculation of bequests to each lifetime income group.

    Inputs:
        r           = [T,] vector, interest rates
        b_splus1    = [T,S,J] array, distribution of wealth/capital
                      holdings one period ahead
        params      = length 5 tuple, (omega, lambdas, rho, g_n, method)
        omega       = [S,T] array, population weights
        lambdas     = [J,] vector, fraction in each lifetime income group
        rho         = [S,] vector, mortality rates
        g_n         = scalar, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        BQ_presum = [T,S,J] array, weighted distribution of
                    wealth/capital holdings one period ahead
        BQ        = [T,J] array, aggregate bequests by lifetime income group

    Returns: BQ
    '''
    if p.use_zeta:
        if j is not None:
            if method == 'SS':
                bq = (p.zeta[:, j] * BQ) / (p.lambdas[j] * p.omega_SS)
            else:
                len_T = BQ.shape[0]
                bq = ((np.reshape(p.zeta[:, j], (1, p.S)) *
                      BQ.reshape((len_T, 1))) /
                      (p.lambdas[j] * p.omega[:len_T, :]))
        else:
            if method == 'SS':
                bq = ((p.zeta * BQ) / (p.lambdas.reshape((1, p.J)) *
                                       p.omega_SS.reshape((p.S, 1))))
            else:
                len_T = BQ.shape[0]
                bq = ((np.reshape(p.zeta, (1, p.S, p.J)) *
                      utils.to_timepath_shape(BQ, p)) /
                      (p.lambdas.reshape((1, 1, p.J)) *
                       p.omega[:len_T, :].reshape((len_T, p.S, 1))))
    else:
        if j is not None:
            if method == 'SS':
                bq = np.tile(BQ[j], p.S) / p.lambdas[j]
            if method == 'TPI':
                len_T = BQ.shape[0]
                bq = np.tile(np.reshape(BQ[:, j] / p.lambdas[j],
                                        (len_T, 1)), (1, p.S))
        else:
            if method == 'SS':
                BQ_per = BQ / np.squeeze(p.lambdas)
                bq = np.tile(np.reshape(BQ_per, (1, p.J)), (p.S, 1))
            if method == 'TPI':
                len_T = BQ.shape[0]
                BQ_per = BQ / p.lambdas.reshape(1, p.J)
                bq = np.tile(np.reshape(BQ_per, (len_T, 1, p.J)),
                             (1, p.S, 1))
    return bq


def get_cons(r, w, b, b_splus1, n, bq, net_tax, e, tau_c, p):
    '''
    Calculation of household consumption.

    Inputs:
        r        = [T,] vector, interest rates
        w        = [T,] vector, wage rates
        b        = [T,S,J] array, distribution of wealth/capital
        b_splus1 = [T,S,J] array, distribution of wealth/capital,
                    one period ahead
        n        = [T,S,J] array, distribution of labor supply
        BQ       = [T,J] array, bequests by lifetime income group
        net_tax  = [T,S,J] array, distribution of net taxes
        params    = length 3 tuple (e, lambdas, g_y)
        e        = [S,J] array, effective labor units by age and
                    lifetime income group
        lambdas  = [S,] vector, fraction of population in each lifetime
                    income group
        g_y      = scalar, exogenous labor augmenting technological growth

    Functions called: None

    Objects in function:
        cons = [T,S,J] array, household consumption

    Returns: cons
    '''
    cons = ((1 + r) * b + w * e * n + bq - b_splus1 * np.exp(p.g_y) -
            net_tax) / (1 + tau_c)
    return cons


def FOC_savings(r, w, b, b_splus1, n, bq, factor, T_H, theta, e, rho,
                tau_c, etr_params, mtry_params, t, j, p, method):
    '''
    Computes Euler errors for the FOC for savings in the steady state.
    This function is usually looped through over J, so it does one
    lifetime income group at a time.

    Inputs:
        r           = scalar, interest rate
        w           = scalar, wage rate
        b           = [S,J] array, distribution of wealth/capital
        b_splus1    = [S,J] array, distribution of wealth/capital,
                        one period ahead
        b_splus2    = [S,J] array, distribution of wealth/capital, two
                        periods ahead
        n           = [S,J] array, distribution of labor supply
        BQ          = [J,] vector, aggregate bequests by lifetime income
                        group
        factor      = scalar, scaling factor to convert model income to
                        dollars
        T_H         = scalar, lump sum transfer
        params      = length 18 tuple (e, sigma, beta, g_y, chi_b,
                                       theta, tau_bq, rho, lambdas, J,
                                       S, etr_params, mtry_params,
                                       h_wealth, p_wealth, m_wealth,
                                       tau_payroll, tau_bq)
        e           = [S,J] array, effective labor units
        sigma       = scalar, coefficient of relative risk aversion
        beta        = scalar, discount factor
        g_y         = scalar, exogenous labor augmenting technological
                        growth
        chi_b       = [J,] vector, utility weight on bequests for each
                        lifetime income group
        theta       = [J,] vector, replacement rate for each lifetime
                        income group
        tau_bq      = scalar, bequest tax rate (scalar)
        rho         = [S,] vector, mortality rates
        lambdas     = [J,] vector, ability weights
        J           = integer, number of lifetime income groups
        S           = integer, number of economically active periods in
                        lifetime
        etr_params  = [S,12] array, parameters of effective income tax
                        rate function
        mtry_params = [S,12] array, parameters of marginal tax rate on
                        capital income function
        h_wealth    = scalar, parameter in wealth tax function
        p_wealth    = scalar, parameter in wealth tax function
        m_wealth    = scalar, parameter in wealth tax function
        tau_payroll = scalar, payroll tax rate
        tau_bq      = scalar, bequest tax rate

    Functions called:
        get_cons
        marg_ut_cons
        tax.total_taxes
        tax.MTR_income

    Objects in function:
        tax1 = [S,J] array, net taxes in the current period
        tax2 = [S,J] array, net taxes one period ahead
        cons1 = [S,J] array, consumption in the current period
        cons2 = [S,J] array, consumption one period ahead
        deriv = [S,J] array, after-tax return on capital
        savings_ut = [S,J] array, marginal utility from savings
        euler = [S,J] array, Euler error from FOC for savings

    Returns: euler
    '''
    if j is not None:
        chi_b = p.chi_b[j]
    else:
        chi_b = p.chi_b
        if method == 'TPI':
            r = utils.to_timepath_shape(r, p)
            w = utils.to_timepath_shape(w, p)
            T_H = utils.to_timepath_shape(T_H, p)

    taxes = tax.total_taxes(r, w, b, n, bq, factor, T_H, theta, t, j,
                            False, method, e, etr_params, p)
    cons = get_cons(r, w, b, b_splus1, n, bq, taxes, e, tau_c, p)
    deriv = ((1 + r) - r * (tax.MTR_income(r, w, b, n, factor, True, e,
                                           etr_params, mtry_params, p)))
    savings_ut = (rho * np.exp(-p.sigma * p.g_y) * chi_b *
                  b_splus1 ** (-p.sigma))
    euler_error = np.zeros_like(n)
    if n.shape[0] > 1:
        euler_error[:-1] = (marg_ut_cons(cons[:-1], p.sigma) *
                            (1 / (1 + tau_c[:-1])) - p.beta *
                            (1 - rho[:-1]) * deriv[1:] *
                            marg_ut_cons(cons[1:], p.sigma) *
                            (1 / (1 + tau_c[1:])) * np.exp(-p.sigma * p.g_y)
                            - savings_ut[:-1])
        euler_error[-1] = (marg_ut_cons(cons[-1], p.sigma) *
                           (1 / (1 + tau_c[-1])) - savings_ut[-1])
    else:
        euler_error[-1] = (marg_ut_cons(cons[-1], p.sigma) *
                           (1 / (1 + tau_c[-1])) - savings_ut[-1])

    return euler_error


def FOC_labor(r, w, b, b_splus1, n, bq, factor, T_H, theta, chi_n, e,
              tau_c, etr_params, mtrx_params, t, j, p, method):
    '''
    Computes Euler errors for the FOC for labor supply in the steady
    state.  This function is usually looped through over J, so it does
    one lifetime income group at a time.

    Inputs:
        r           = scalar, interest rate
        w           = scalar, wage rate
        b           = [S,J] array, distribution of wealth/capital
                        holdings
        b_splus1    = [S,J] array, distribution of wealth/capital
                        holdings one period ahead
        n           = [S,J] array, distribution of labor supply
        BQ          = [J,] vector, aggregate bequests by lifetime
                        income group
        factor      = scalar, scaling factor to convert model income to
                        dollars
        T_H         = scalar, lump sum transfer
        params      = length 19 tuple (e, sigma, g_y, theta, b_ellipse,
                                       upsilon, ltilde, chi_n, tau_bq,
                                       lambdas, J, S, etr_params,
                                       mtrx_params, h_wealth, p_wealth,
                                       m_wealth, tau_payroll, tau_bq)
        e           = [S,J] array, effective labor units
        sigma       = scalar, coefficient of relative risk aversion
        g_y         = scalar, exogenous labor augmenting technological
                        growth
        theta       = [J,] vector, replacement rate for each lifetime
                        income group
        b_ellipse   = scalar, scaling parameter in elliptical utility
                        function
        upsilon     = curvature parameter in elliptical utility function
        chi_n       = [S,] vector, utility weights on disutility of labor
        ltilde      = scalar, upper bound of household labor supply
        tau_bq      = scalar, bequest tax rate (scalar)
        lambdas     = [J,] vector, ability weights
        J           = integer, number of lifetime income groups
        S           = integer, number of economically active periods in
                        lifetime
        etr_params  = [S,10] array, parameters of effective income tax
                        rate function
        mtrx_params = [S,10] array, parameters of marginal tax rate on
                        labor income function
        h_wealth    = scalar, parameter in wealth tax function
        p_wealth    = scalar, parameter in wealth tax function
        m_wealth    = scalar, parameter in wealth tax function
        tau_payroll = scalar, payroll tax rate
        tau_bq      = scalar, bequest tax rate

    Functions called:
        get_cons
        marg_ut_cons
        marg_ut_labor
        tax.total_taxes
        tax.MTR_income

    Objects in function:
        tax = [S,J] array, net taxes in the current period
        cons = [S,J] array, consumption in the current period
        deriv = [S,J] array, net of tax share of labor income
        euler = [S,J] array, Euler error from FOC for labor supply

    Returns: euler

        if j is not None:
            chi_b = p.chi_b[j]
            if method == 'TPI':
                r = r.reshape(r.shape[0], 1)
                w = w.reshape(w.shape[0], 1)
                T_H = T_H.reshape(T_H.shape[0], 1)
        else:
            chi_b = p.chi_b
            if method == 'TPI':
                r = utils.to_timepath_shape(r, p)
                w = utils.to_timepath_shape(w, p)
                T_H = utils.to_timepath_shape(T_H, p)
    '''
    if method == 'SS':
        tau_payroll = p.tau_payroll[-1]
    elif method == 'TPI_scalar':  # for 1st donut ring onlye
        tau_payroll = p.tau_payroll[0]
    else:
        length = r.shape[0]
        tau_payroll = p.tau_payroll[t:t + length]
    if method == 'TPI':
        if b.ndim == 2:
            r = r.reshape(r.shape[0], 1)
            w = w.reshape(w.shape[0], 1)
            T_H = T_H.reshape(T_H.shape[0], 1)
            tau_payroll = tau_payroll.reshape(tau_payroll.shape[0], 1)
        elif b.ndim == 3:
            r = utils.to_timepath_shape(r, p)
            w = utils.to_timepath_shape(w, p)
            T_H = utils.to_timepath_shape(T_H, p)
            tau_payroll = utils.to_timepath_shape(tau_payroll, p)

    taxes = tax.total_taxes(r, w, b, n, bq, factor, T_H, theta, t, j,
                            False, method, e, etr_params, p)
    cons = get_cons(r, w, b, b_splus1, n, bq, taxes, e, tau_c, p)
    deriv = (1 - tau_payroll - tax.MTR_income(r, w, b, n, factor,
                                              False, e, etr_params,
                                              mtrx_params, p))
    FOC_error = (marg_ut_cons(cons, p.sigma) * (1 / (1 + tau_c)) * w *
                 deriv * e - marg_ut_labor(n, chi_n, p))

    return FOC_error


def constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    '''
    Checks constraints on consumption, savings, and labor supply in the
    steady state.

    Inputs:
        bssmat = [S,J] array, steady state distribution of capital
        nssmat = [S,J] array, steady state distribution of labor
        cssmat = [S,J] array, steady state distribution of consumption
        ltilde = scalar, upper bound of household labor supply

    Functions called: None

    Objects in function:
        flag2 = boolean, indicates if labor supply constraints violated
                (=False if not)

    Returns:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    print('Checking constraints on capital, labor, and consumption.')

    if (bssmat < 0).any():
        print('\tWARNING: There is negative capital stock')
    flag2 = False
    if (nssmat < 0).any():
        print('\tWARNING: Labor supply violates nonnegativity ',
              'constraints.')
        flag2 = True
    if (nssmat > ltilde).any():
        print('\tWARNING: Labor suppy violates the ltilde constraint.')
        flag2 = True
    if flag2 is False:
        print('\tThere were no violations of the constraints on labor',
              ' supply.')
    if (cssmat < 0).any():
        print('\tWARNING: Consumption violates nonnegativity',
              ' constraints.')
    else:
        print('\tThere were no violations of the constraints on',
              ' consumption.')


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, ltilde):
    '''
    Checks constraints on consumption, savings, and labor supply along
    the transition path. Does this for each period t separately.

    Inputs:
        b_dist = [S,J] array, distribution of capital
        n_dist = [S,J] array, distribution of labor
        c_dist = [S,J] array, distribution of consumption
        t      = integer, time period
        ltilde = scalar, upper bound of household labor supply

    Functions called: None

    Objects in function: None

    Returns:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    if (b_dist <= 0).any():
        print('\tWARNING: Aggregate capital is less than or equal to ',
              'zero in period %.f.' % t)
    if (n_dist < 0).any():
        print('\tWARNING: Labor supply violates nonnegativity',
              ' constraints in period %.f.' % t)
    if (n_dist > ltilde).any():
        print('\tWARNING: Labor suppy violates the ltilde constraint',
              ' in period %.f.' % t)
    if (c_dist < 0).any():
        print('\tWARNING: Consumption violates nonnegativity',
              ' constraints in period %.f.' % t)
