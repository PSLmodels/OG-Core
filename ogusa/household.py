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
import tax

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
    output = c**(-sigma)
    return output


def marg_ut_labor(n, params):
    '''
    Computation of marginal disutility of labor.

    Inputs:
        n         = [T,S,J] array, household labor supply
        params    = length 4 tuple (b_ellipse, upsilon, ltilde, chi_n)
        b_ellipse = scalar, scaling parameter in elliptical utility
                    function
        upsilon   = curvature parameter in elliptical utility function
        ltilde    = scalar, upper bound of household labor supply
        chi_n     = [S,] vector, utility weights on disutility of labor

    Functions called: None

    Objects in function:
        output = [T,S,J] array, marginal disutility of labor supply

    Returns: output
    '''
    b_ellipse, upsilon, ltilde, chi_n = params

    deriv = (b_ellipse * (1.0 / ltilde) *
             ((1.0 - (n / ltilde) ** upsilon) **
              ((1.0 / upsilon) - 1.0)) *
             (n / ltilde) ** (upsilon - 1.0))

    output = chi_n * deriv
    return output


def get_cons(r, w, b, b_splus1, n, BQ, net_tax, params):
    '''
    Calculation of househld consumption.

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
    e, lambdas, g_y = params

    cons = ((1 + r) * b + w * e * n + BQ / lambdas - b_splus1 *
            np.exp(g_y) - net_tax)
    return cons


def FOC_savings(r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H,
                params):
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
        tax.MTR_capital

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
    (e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas, j, J, S,
     analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth,
     m_wealth, tau_payroll, retire, method) = params

    # In order to not have 2 savings euler equations (one that solves
    # the first S-1 equations, and one that solves the last one), we
    # combine them.  In order to do this, we have to compute a
    # consumption term in period t+1, which requires us to have a shifted
    # e and n matrix.  We append a zero on the end of both of these so
    # they will be the right size.  We could append any value to them,
    # since in the euler equation, the coefficient on the marginal
    # utility of consumption for this term will be zero (since rho is
    # one).
    e_extended = np.array(list(e) + [0])
    n_extended = np.array(list(n) + [0])
    etr_params_extended = np.append(etr_params,
                                    np.reshape(etr_params[-1, :],
                                               (1, etr_params.shape[1])),
                                    axis=0)[1:, :]
    mtry_params_extended = np.append(mtry_params,
                                     np.reshape(mtry_params[-1, :],
                                                (1, mtry_params.shape[1])),
                                     axis=0)[1:, :]
    if method == 'TPI':
        r_extended = np.append(r, r[-1])
        w_extended = np.append(w, w[-1])
        BQ_extended = np.append(BQ, BQ[-1])
        T_H_extended = np.append(T_H, T_H[-1])
    elif method == 'SS':
        r_extended = np.array([r, r])
        w_extended = np.array([w, w])
        BQ_extended = np.array([BQ, BQ])
        T_H_extended = np.array([T_H, T_H])

    tax1_params = (e, lambdas, method, retire, etr_params, h_wealth,
                   p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, False,
                           tax1_params)
    tax2_params = (e_extended[1:], lambdas, method, retire,
                   etr_params_extended, h_wealth, p_wealth, m_wealth,
                   tau_payroll, theta, tau_bq, J, S)
    tax2 = tax.total_taxes(r_extended[1:], w_extended[1:], b_splus1,
                           n_extended[1:], BQ_extended[1:], factor,
                           T_H_extended[1:], j, True, tax2_params)
    cons1_params = (e, lambdas, g_y)
    cons1 = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons1_params)
    cons2_params = (e_extended[1:], lambdas, g_y)
    cons2 = get_cons(r_extended[1:], w_extended[1:], b_splus1, b_splus2,
                     n_extended[1:], BQ_extended[1:], tax2,
                     cons2_params)
    cons2[-1] = 0.01  # set to small positive number to avoid exception
    # errors when negative b/c this period value doesn't matter -
    # it's consumption after the last period of life
    mtr_cap_params = (e_extended[1:], etr_params_extended,
                      mtry_params_extended, analytical_mtrs)
    deriv = ((1 + r_extended[1:]) - r_extended[1:] *
             (tax.MTR_capital(r_extended[1:], w_extended[1:], b_splus1,
                              n_extended[1:], factor, mtr_cap_params)))

    savings_ut = (rho * np.exp(-sigma * g_y) * chi_b * b_splus1 **
                  (-sigma))

    euler_error = (marg_ut_cons(cons1, sigma) - beta * (1 - rho) *
                   deriv * marg_ut_cons(cons2, sigma) *
                   np.exp(-sigma * g_y) - savings_ut)

    return euler_error


def FOC_labor(r, w, b, b_splus1, n, BQ, factor, T_H, params):
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
        tax.MTR_labor

    Objects in function:
        tax = [S,J] array, net taxes in the current period
        cons = [S,J] array, consumption in the current period
        deriv = [S,J] array, net of tax share of labor income
        euler = [S,J] array, Euler error from FOC for labor supply

    Returns: euler
    '''
    (e, sigma, g_y, theta, b_ellipse, upsilon, chi_n, ltilde, tau_bq,
     lambdas, j, J, S, analytical_mtrs, etr_params, mtrx_params,
     h_wealth, p_wealth, m_wealth, tau_payroll, retire, method) = params

    tax1_params = (e, lambdas, method, retire, etr_params, h_wealth,
                   p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, False,
                           tax1_params)
    cons_params = (e, lambdas, g_y)
    cons = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons_params)
    mtr_lab_params = (e, etr_params, mtrx_params, analytical_mtrs)
    deriv = (1 - tau_payroll - tax.MTR_labor(r, w, b, n, factor,
                                             mtr_lab_params))

    lab_params = (b_ellipse, upsilon, ltilde, chi_n)
    FOC_error = (marg_ut_cons(cons, sigma) * w * deriv * e -
                 marg_ut_labor(n, lab_params))

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
    print 'Checking constraints on capital, labor, and consumption.'

    if (bssmat < 0).any():
        print '\tWARNING: There is negative capital stock'
    flag2 = False
    if (nssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity ' +\
            'constraints.'
        flag2 = True
    if (nssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
        flag2 = True
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor'\
         + ' supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Consumption violates nonnegativity' +\
            ' constraints.'
    else:
        print '\tThere were no violations of the constraints on' +\
            ' consumption.'


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
        print '\tWARNING: Aggregate capital is less than or equal to '\
            'zero in period %.f.' % t
    if (n_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity' +\
            ' constraints in period %.f.' % t
    if (n_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint' +\
            ' in period %.f.' % t
    if (c_dist < 0).any():
        print '\tWARNING: Consumption violates nonnegativity' +\
            ' constraints in period %.f.' % t
