'''
------------------------------------------------------------------------
Last updated 4/7/2016

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


def get_K(b, params):
    '''
    Calculates aggregate capital supplied.

    Inputs:
        b           = [T,S,J] array, distribution of wealth/capital holdings 
        params      = length 4 tuple, (omega, lambdas, g_n, method)
        omega       = [S,T] array, population weights 
        lambdas     = [J,] vector, fraction in each lifetime income group 
        g_n         = [T,] vector, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        K_presum = [T,S,J] array, weighted distribution of wealth/capital holdings
        K        = [T,] vector, aggregate capital supply

    Returns: K
    '''

    omega, lambdas, g_n, method = params 

    K_presum = b * omega * lambdas
    if method == 'SS':
        K = K_presum.sum()
    elif method == 'TPI':
        K = K_presum.sum(1).sum(1)
    K /= (1.0 + g_n)
    return K


def get_BQ(r, b_splus1, params):
    '''
    Calculation of bequests to each lifetime income group.

    Inputs:
        r           = [T,] vector, interest rates
        b_splus1    = [T,S,J] array, distribution of wealth/capital holdings one period ahead
        params      = length 5 tuple, (omega, lambdas, rho, g_n, method)
        omega       = [S,T] array, population weights 
        lambdas     = [J,] vector, fraction in each lifetime income group 
        rho         = [S,] vector, mortality rates
        g_n         = scalar, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        BQ_presum = [T,S,J] array, weighted distribution of wealth/capital holdings one period ahead
        BQ        = [T,J] array, aggregate bequests by lifetime income group

    Returns: BQ
    '''
    omega, lambdas, rho, g_n, method = params 

    BQ_presum = b_splus1 * omega * rho * lambdas
    if method == 'SS':
        BQ = BQ_presum.sum(0)
    elif method == 'TPI':
        BQ = BQ_presum.sum(1)
    BQ *= (1.0 + r) / (1.0 + g_n)
    return BQ


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
        b_ellipse = scalar, scaling parameter in elliptical utility function
        upsilon   = curvature parameter in elliptical utility function 
        ltilde    = scalar, upper bound of household labor supply
        chi_n     = [S,] vector, utility weights on disutility of labor

    Functions called: None

    Objects in function:
        output = [T,S,J] array, marginal disutility of labor supply

    Returns: output
    '''
    b_ellipse, upsilon, ltilde, chi_n = params

    try:
        deriv = b_ellipse * (1.0 / ltilde) * ((1.0 - (n / ltilde) ** upsilon) ** (
            (1.0 / upsilon) - 1.0)) * (n / ltilde) ** (upsilon - 1.0)
    except ValueError as ve:
        # I think TJ added this in.  We need to be careful with doing stuff like this --
        # it could lead to incorrect output, if we are setting deriv to be something that
        # is actually close to the solution (or a possible solution).  If anything,
        # we might want to set deriv to be some huge number (ie, 1e12).  That would almost
        # certainly be far from the true value, which would force the euler error to be quite large,
        # and so the fsolve will not pick this solution.
        deriv = 1e12

    output = chi_n * deriv
    return output


def get_cons(r, w, b, b_splus1, n, BQ, net_tax, params):
    '''
    Calculation of househld consumption.

    Inputs:
        r        = [T,] vector, interest rates
        w        = [T,] vector, wage rates
        b        = [T,S,J] array, distribution of wealth/capital holdings
        b_splus1 = [T,S,J] array, distribution of wealth/capital holdings one period ahead
        n        = [T,S,J] array, distribution of labor supply
        BQ       = [T,J] array, bequests by lifetime income group
        net_tax  = [T,S,J] array, distribution of net taxes
        params    = length 3 tuple (e, lambdas, g_y)
        e        = [S,J] array, effective labor units by age and lifetime income group
        lambdas  = [S,] vector, fraction of population in each lifetime income group
        g_y      = scalar, exogenous labor augmenting technological growth
    
    Functions called: None

    Objects in function:
        cons = [T,S,J] array, household consumption

    Returns: cons
    '''
    e, lambdas, g_y = params
    
    cons = (1 + r) * b + w * e * n + BQ / \
        lambdas - b_splus1 * np.exp(g_y) - net_tax
    return cons


def get_C(c, params):
    '''
    Calculation of aggregate consumption.

    Inputs:
        cons        = [T,S,J] array, household consumption
        params      = length 3 tuple (omega, lambdas, method)
        omega       = [S,T] array, population weights by age (Sx1 array)
        lambdas     = [J,1] vector, lifetime income group weights
        method      = string, 'SS' or 'TPI' 

    Functions called: None

    Objects in function:
        aggC_presum = [T,S,J] array, weighted consumption by household
        aggC        = [T,] vector, aggregate consumption

    Returns: aggC
    '''

    omega, lambdas, method = params

    aggC_presum = c * omega * lambdas
    if method == 'SS':
        aggC = aggC_presum.sum()
    elif method == 'TPI':
        aggC = aggC_presum.sum(1).sum(1)
    return aggC


def FOC_savings(r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H, params):
    '''
    Computes Euler errors for the FOC for savings in the steady state.  
    This function is usually looped through over J, so it does one lifetime income group at a time.
    
    Inputs:
        r           = scalar, interest rate
        w           = scalar, wage rate
        b           = [S,J] array, distribution of wealth/capital holdings
        b_splus1    = [S,J] array, distribution of wealth/capital holdings one period ahead
        b_splus2    = [S,J] array, distribution of wealth/capital holdings two periods ahead
        n           = [S,J] array, distribution of labor supply
        BQ          = [J,] vector, aggregate bequests by lifetime income group
        factor      = scalar, scaling factor to convert model income to dollars
        T_H         = scalar, lump sum transfer
        params      = length 18 tuple (e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas, 
                                    J, S, etr_params, mtry_params, h_wealth, p_wealth, 
                                    m_wealth, tau_payroll, tau_bq)
        e           = [S,J] array, effective labor units
        sigma       = scalar, coefficient of relative risk aversion
        beta        = scalar, discount factor
        g_y         = scalar, exogenous labor augmenting technological growth
        chi_b       = [J,] vector, utility weight on bequests for each lifetime income group
        theta       = [J,] vector, replacement rate for each lifetime income group
        tau_bq      = scalar, bequest tax rate (scalar)
        rho         = [S,] vector, mortality rates
        lambdas     = [J,] vector, ability weights
        J           = integer, number of lifetime income groups
        S           = integer, number of economically active periods in lifetime
        etr_params  = [S,10] array, parameters of effective income tax rate function
        mtry_params = [S,10] array, parameters of marginal tax rate on capital income function
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
    e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas, J, S, \
        analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, method = params

    # In order to not have 2 savings euler equations (one that solves the first S-1 equations, and one that solves the last one),
    # we combine them.  In order to do this, we have to compute a consumption term in period t+1, which requires us to have a shifted
    # e and n matrix.  We append a zero on the end of both of these so they will be the right size.  We could append any value to them,
    # since in the euler equation, the coefficient on the marginal utility of
    # consumption for this term will be zero (since rho is one).
    if method == 'TPI_scalar':
        e_extended = np.array([e] + [0])
        n_extended = np.array([n] + [0])
        etr_params_to_use = etr_params
        mtry_params_to_use = mtry_params
    else:
        e_extended = np.array(list(e) + [0])
        n_extended = np.array(list(n) + [0])
        etr_params_to_use = np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:]
        mtry_params_to_use = np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:]
    
    # tax1_params = (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    # tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, None, False, tax1_params)
    # tax2_params = (e_extended[1:], lambdas, method, retire, 
    #                np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:], 
    #                h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    # tax2 = tax.total_taxes(r, w, b_splus1, n_extended[1:], BQ, factor, T_H, None, True, tax2_params)
    # cons1_params = (e, lambdas, g_y)
    # cons1 = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons1_params)
    # cons2_params = (e_extended[1:], lambdas, g_y)
    # cons2 = get_cons(r, w, b_splus1, b_splus2, n_extended[1:], BQ, tax2, cons2_params)

    # mtr_cap_params = (e_extended[1:], np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:],
    #                   np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:],analytical_mtrs)
    # deriv = (1+r) - r*(tax.MTR_capital(r, w, b_splus1, n_extended[1:], factor, mtr_cap_params))

    tax1_params = (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, None, False, tax1_params)
    tax2_params = (e_extended[1:], lambdas, method, retire, 
                   etr_params_to_use, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax2 = tax.total_taxes(r, w, b_splus1, n_extended[1:], BQ, factor, T_H, None, True, tax2_params)
    cons1_params = (e, lambdas, g_y)
    cons1 = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons1_params)
    cons2_params = (e_extended[1:], lambdas, g_y)
    cons2 = get_cons(r, w, b_splus1, b_splus2, n_extended[1:], BQ, tax2, cons2_params)

    mtr_cap_params = (e_extended[1:], etr_params_to_use,
                      mtry_params_to_use,analytical_mtrs)
    deriv = (1+r) - r*(tax.MTR_capital(r, w, b_splus1, n_extended[1:], factor, mtr_cap_params))

    savings_ut = rho * np.exp(-sigma * g_y) * chi_b * b_splus1 ** (-sigma)

    # Again, note timing in this equation, the (1-rho) term will zero out in the last period, so the last entry of cons2 can be complete
    # gibberish (which it is).  It just has to exist so cons2 is the right
    # size to match all other arrays in the equation.
    euler = marg_ut_cons(cons1, sigma) - beta * (1 - rho) * deriv * marg_ut_cons(
        cons2, sigma) * np.exp(-sigma * g_y) - savings_ut


    return euler


def FOC_labor(r, w, b, b_splus1, n, BQ, factor, T_H, params):
    '''
    Computes Euler errors for the FOC for labor supply in the steady state.  
    This function is usually looped through over J, so it does one lifetime income group at a time.

    Inputs:
        r           = scalar, interest rate
        w           = scalar, wage rate
        b           = [S,J] array, distribution of wealth/capital holdings
        b_splus1    = [S,J] array, distribution of wealth/capital holdings one period ahead
        n           = [S,J] array, distribution of labor supply
        BQ          = [J,] vector, aggregate bequests by lifetime income group
        factor      = scalar, scaling factor to convert model income to dollars
        T_H         = scalar, lump sum transfer
        params      = length 19 tuple (e, sigma, g_y, theta, b_ellipse, upsilon, ltilde, 
                                    chi_n, tau_bq, lambdas, J, S,
                                    etr_params, mtrx_params, h_wealth, p_wealth, 
                                    m_wealth, tau_payroll, tau_bq)
        e           = [S,J] array, effective labor units
        sigma       = scalar, coefficient of relative risk aversion
        g_y         = scalar, exogenous labor augmenting technological growth
        theta       = [J,] vector, replacement rate for each lifetime income group
        b_ellipse   = scalar, scaling parameter in elliptical utility function
        upsilon     = curvature parameter in elliptical utility function 
        chi_n       = [S,] vector, utility weights on disutility of labor
        ltilde      = scalar, upper bound of household labor supply
        tau_bq      = scalar, bequest tax rate (scalar)
        lambdas     = [J,] vector, ability weights
        J           = integer, number of lifetime income groups
        S           = integer, number of economically active periods in lifetime
        etr_params  = [S,10] array, parameters of effective income tax rate function
        mtrx_params = [S,10] array, parameters of marginal tax rate on labor income function
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
    e, sigma, g_y, theta, b_ellipse, upsilon, chi_n, ltilde, tau_bq, lambdas, J, S, \
        analytical_mtrs, etr_params, mtrx_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, method  = params

    tax1_params = (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, 
                  m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b, n, BQ, factor, T_H, None, False, tax1_params)
    cons_params = (e, lambdas, g_y)
    cons = get_cons(r, w, b, b_splus1, n, BQ, tax1, cons_params)  
    mtr_lab_params = (e, etr_params, mtrx_params, analytical_mtrs)
    deriv = (1 - tau_payroll - tax.MTR_labor(r, b, w, n, factor, mtr_lab_params))
        
    lab_params = (b_ellipse, upsilon, ltilde, chi_n)
    euler = marg_ut_cons(cons, sigma) * w * deriv * e - \
        marg_ut_labor(n, lab_params)

    return euler


def constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    '''
    Checks constraints on consumption, savings, and labor supply in the steady state.

    Inputs:
        bssmat = [S,J] array, steady state distribution of capital
        nssmat = [S,J] array, steady state distribution of labor
        cssmat = [S,J] array, steady state distribution of consumption
        ltilde = scalar, upper bound of household labor supply

    Functions called: None

    Objects in function:
        flag2 = boolean, indicates if labor supply constraints violated (=False if not)

    Returns: 
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    print 'Checking constraints on capital, labor, and consumption.' 

    if (bssmat < 0).any():
        print '\tWARNING: There is negative capital stock'
    flag2 = False
    if (nssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (nssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
        flag2 = True
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Consumption violates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, ltilde):
    '''
    Checks constraints on consumption, savings, and labor supply along the transition path.
    Does this for each period t separately.

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
        print '\tWARNING: Aggregate capital is less than or equal to ' \
            'zero in period %.f.' % t
    if (n_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints ' \
            'in period %.f.' % t
    if (n_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint in '\
            'period %.f.' % t
    if (c_dist < 0).any():
        print '\tWARNING: Consumption violates nonnegativity constraints in ' \
            'period %.f.' % t