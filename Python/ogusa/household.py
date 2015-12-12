'''
------------------------------------------------------------------------
Last updated 7/16/2015

Household functions for taxes in SS and TPI.

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


def get_K(b, pop_weights, ability_weights, g_n, method):
    '''
    Inputs:
        b = distribution of capital (SxJ array)
        pop_weights = population weights (Sx1 array)
        ability_weights = ability percentile groups (Jx1 array)
        g_n = population growth rate (scalar)
        method = 'SS' or 'TPI' (string)
    Output:
        K_now = Aggregate Capital (scalar or array)
    '''
    K_presum = b * pop_weights * ability_weights
    if method == 'SS':
        K_now = K_presum.sum()
    elif method == 'TPI':
        K_now = K_presum.sum(1).sum(1)
    K_now /= (1.0 + g_n)
    return K_now


def get_BQ(r, b_splus1, pop_weights, ability_weights, rho, g_n, method):
    '''
    Inputs:
        r = interest rate (scalar)
        b_splus1 = wealth holdings for the next period (SxJ array)
        pop_weights = population weights by age (Sx1 array)
        ability_weights = ability weights (Jx1 array)
        rho = mortality rates (Sx1 array)
        g_n = population growth rate (scalar)
        method = 'SS' or 'TPI' (string)
    Output:
        BQ = aggregate BQ (Jx1 array or TxJ array)
    '''
    BQ_presum = b_splus1 * pop_weights * rho * ability_weights
    if method == 'SS':
        BQ = BQ_presum.sum(0)
    elif method == 'TPI':
        BQ = BQ_presum.sum(1)
    BQ *= (1.0 + r) / (1.0 + g_n)
    return BQ


def marg_ut_cons(c, params):
    '''
    Inputs:
        c = Consumption (any array or scalar)
        params = list of parameters (list)
    Outputs:
        output = Marginal Utility of Consumption (same shape as c)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    output = c**(-sigma)
    return output


def marg_ut_labor(n, chi_n, params):
    '''
    Inputs:
        n = labor particpation distribution (various length array or scalar)
        chi_n = chi^n_s (various length array or scalar)
        params = list of parameters (list)
    Output:
        output = Marginal Utility of Labor (same shape as n)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
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
        deriv = 1e-5

    output = chi_n * deriv
    return output


def get_cons(r, b_s, w, e, n, BQ, lambdas, b_splus1, params, net_tax):
    '''
    Inputs:
        r = interest rate (scalar)
        b_s = wealth holdings at the start of a period (SxJ array or Sx1 array)
        w = wage rate (scalar)
        e = ability levels (SxJ array or Sx1 array)
        n = labor rate distribution (SxJ array or Sx1 array)
        BQ = aggregate bequests (Jx1 array)
        lambdas = ability weights (Jx1 array)
        b_splus1 = wealth holdings for the next period (SxJ or Sx1 array)
        params = list of paramters (list)
        net_tax = net tax (SxJ array or Sx1 array)
    Output:
        cons = Consumption (SxJ or Sx1 array)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    cons = (1 + r) * b_s + w * e * n + BQ / \
        lambdas - b_splus1 * np.exp(g_y) - net_tax
    return cons


def get_C(individ_cons, pop_weights, ability_weights, method):
    '''
    Inputs:
        individ_cons = distribution of consumption (SxJ array)
        pop_weights = population weights by age (Sx1 array)
        ability_weights = ability weights (Jx1 array)
        method = 'SS' or 'TPI' (string)
    Output:
        aggC = aggregate consumption (scalar or array)
    '''
    aggC_presum = individ_cons * pop_weights * ability_weights
    if method == 'SS':
        aggC = aggC_presum.sum()
    elif method == 'TPI':
        aggC = aggC_presum.sum(1).sum(1)
    return aggC


def euler_savings_func(w, r, e, n_guess, b_s, b_splus1, b_splus2, BQ, factor, T_H, chi_b, tax_params, params, theta, tau_bq, rho, lambdas):
    '''
    This function is usually looped through over J, so it does one ability group at a time.
    Inputs:
        w = wage rate (scalar)
        r = rental rate (scalar)
        e = ability levels (Sx1 array)
        n_guess = labor distribution (Sx1 array)
        b_s = wealth holdings at the start of a period (Sx1 array)
        b_splus1 = wealth holdings for the next period (Sx1 array)
        b_splus2 = wealth holdings for 2 periods ahead (Sx1 array)
        BQ = aggregate bequests for a certain ability (scalar)
        factor = scaling factor to convert to dollars (scalar)
        T_H = lump sum tax (scalar)
        chi_b = chi^b_j for a certain ability (scalar)
        params = parameter list (list)
        theta = replacement rate for a certain ability (scalar)
        tau_bq = bequest tax rate (scalar)
        rho = mortality rate (Sx1 array)
        lambdas = ability weight (scalar)
    Output:
        euler = Value of savings euler error (Sx1 array)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params

    a_etr_income, b_etr_income, \
        c_etr_income, d_etr_income, e_etr_income, f_etr_income, \
        min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income,\
        a_mtrx_income, b_mtrx_income, \
        c_mtrx_income, d_mtrx_income, e_mtrx_income, f_mtrx_income, \
        min_x_mtrx_income, max_x_mtrx_income, min_y_mtrx_income, max_y_mtrx_income,\
        a_mtry_income, b_mtry_income, \
        c_mtry_income, d_mtry_income, e_mtry_income, f_mtry_income, \
        min_x_mtry_income, max_x_mtry_income, min_y_mtry_income, max_y_mtry_income = tax_params
    # In order to not have 2 savings euler equations (one that solves the first S-1 equations, and one that solves the last one),
    # we combine them.  In order to do this, we have to compute a consumption term in period t+1, which requires us to have a shifted
    # e and n matrix.  We append a zero on the end of both of these so they will be the right size.  We could append any value to them,
    # since in the euler equation, the coefficient on the marginal utility of
    # consumption for this term will be zero (since rho is one).
    e_extended = np.array(list(e) + [0])
    n_extended = np.array(list(n_guess) + [0])
    tax1_params = (J, S, retire, a_etr_income, b_etr_income, c_etr_income, d_etr_income, e_etr_income, f_etr_income,
                   min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income, h_wealth, p_wealth, m_wealth, tau_payroll)
    tax1 = tax.total_taxes(r, b_s, w, e, n_guess, BQ, lambdas,
                           factor, T_H, None, 'SS', False, tax1_params, theta, tau_bq)

    tax2_params = (J, S, retire, np.array(list(a_etr_income) + [a_etr_income[-1]])[1:], np.array(list(b_etr_income) + [b_etr_income[-1]])[1:], 
                   np.array(list(c_etr_income) + [c_etr_income[-1]])[1:], np.array(list(d_etr_income) + [d_etr_income[-1]])[1:], 
                   np.array(list(e_etr_income) + [e_etr_income[-1]])[1:], np.array(list(f_etr_income) + [f_etr_income[-1]])[1:],
                   np.array(list(min_x_etr_income) + [min_x_etr_income[-1]])[1:], np.array(list(max_x_etr_income) + [max_x_etr_income[-1]])[1:], 
                   np.array(list(min_y_etr_income) + [min_y_etr_income[-1]])[1:], np.array(list(max_y_etr_income) + [max_y_etr_income[-1]])[1:], 
                   h_wealth, p_wealth, m_wealth, tau_payroll)
    tax2 = tax.total_taxes(r, b_splus1, w, e_extended[1:], n_extended[
                           1:], BQ, lambdas, factor, T_H, None, 'SS', True, tax2_params, theta, tau_bq)
    cons1 = get_cons(r, b_s, w, e, n_guess, BQ,
                     lambdas, b_splus1, params, tax1)
    cons2 = get_cons(r, b_splus1, w, e_extended[1:], n_extended[
                     1:], BQ, lambdas, b_splus2, params, tax2)
    income = (r * b_splus1 + w * e_extended[1:] * n_extended[1:]) * factor

    #tau_inc_params = (np.array(list(a_tax_income) + [a_tax_income[-1]])[1:], np.array(list(b_tax_income) + [b_tax_income[-1]])[1:], 
    #               np.array(list(c_tax_income) + [c_tax_income[-1]])[1:], np.array(list(d_tax_income) + [d_tax_income[-1]])[1:], 
    #               np.array(list(e_tax_income) + [e_tax_income[-1]])[1:], np.array(list(f_tax_income) + [f_tax_income[-1]])[1:],
    #               np.array(list(min_x_tax_income) + [min_x_tax_income[-1]])[1:], np.array(list(max_x_tax_income) + [max_x_tax_income[-1]])[1:], 
    #               np.array(list(min_y_tax_income) + [min_y_tax_income[-1]])[1:], np.array(list(max_y_tax_income) + [max_y_tax_income[-1]])[1:]) 
    mtr_cap_params = (np.array(list(a_mtry_income) + [a_mtry_income[-1]])[1:], np.array(list(b_mtry_income) + [b_mtry_income[-1]])[1:], 
                   np.array(list(c_mtry_income) + [c_mtry_income[-1]])[1:], np.array(list(d_mtry_income) + [d_mtry_income[-1]])[1:], 
                   np.array(list(e_mtry_income) + [e_mtry_income[-1]])[1:], np.array(list(f_mtry_income) + [f_mtry_income[-1]])[1:],
                   np.array(list(min_x_mtry_income) + [min_x_mtry_income[-1]])[1:], np.array(list(max_x_mtry_income) + [max_x_mtry_income[-1]])[1:], 
                   np.array(list(min_y_mtry_income) + [min_y_mtry_income[-1]])[1:], np.array(list(max_y_mtry_income) + [max_y_mtry_income[-1]])[1:]) 
    
    deriv = (1+r) - r*(tax.MTR_capital(r, b_splus1, w, e_extended[1:], n_extended[1:], factor, mtr_cap_params))


    #deriv = (1+r) - r*(tax.tau_capital_deriv(r, b_splus1, w, e_extended[1:], n_extended[1:], factor, tau_inc_params)
    #               - tax.tau_income(r, b_splus1, w, e_extended[1:], n_extended[1:], factor, tau_inc_params))

    #deriv = (
    #    1 + r * (1 - tax.tau_income(r, b_splus1, w, e_extended[1:], n_extended[1:], factor, params) - tax.tau_income_deriv(
    #        r, b_splus1, w, e_extended[1:], n_extended[1:], factor, params) * income) - tax.tau_w_prime(b_splus1, params) 
    #         * b_splus1 - tax.tau_wealth(b_splus1, params))
    savings_ut = rho * np.exp(-sigma * g_y) * chi_b * b_splus1 ** (-sigma)

    # Again, note timing in this equation, the (1-rho) term will zero out in the last period, so the last entry of cons2 can be complete
    # gibberish (which it is).  It just has to exist so cons2 is the right
    # size to match all other arrays in the equation.
    euler = marg_ut_cons(cons1, params) - beta * (1 - rho) * deriv * marg_ut_cons(
        cons2, params) * np.exp(-sigma * g_y) - savings_ut


    return euler


def euler_labor_leisure_func(w, r, e, n_guess, b_s, b_splus1, BQ, factor, T_H, chi_n, tax_params, params, theta, tau_bq, lambdas):
    '''
    This function is usually looped through over J, so it does one ability group at a time.
    Inputs:
        w = wage rate (scalar)
        r = rental rate (scalar)
        e = ability levels (Sx1 array)
        n_guess = labor distribution (Sx1 array)
        b_s = wealth holdings at the start of a period (Sx1 array)
        b_splus1 = wealth holdings for the next period (Sx1 array)
        BQ = aggregate bequests for a certain ability (scalar)
        factor = scaling factor to convert to dollars (scalar)
        T_H = lump sum tax (scalar)
        chi_n = chi^n_s (Sx1 array)
        params = parameter list (list)
        theta = replacement rate for a certain ability (scalar)
        tau_bq = bequest tax rate (scalar)
        lambdas = ability weight (scalar)
    Output:
        euler = Value of labor leisure euler error (Sx1 array)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params

    a_etr_income, b_etr_income, \
        c_etr_income, d_etr_income, e_etr_income, f_etr_income, \
        min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income,\
        a_mtrx_income, b_mtrx_income, \
        c_mtrx_income, d_mtrx_income, e_mtrx_income, f_mtrx_income, \
        min_x_mtrx_income, max_x_mtrx_income, min_y_mtrx_income, max_y_mtrx_income,\
        a_mtry_income, b_mtry_income, \
        c_mtry_income, d_mtry_income, e_mtry_income, f_mtry_income, \
        min_x_mtry_income, max_x_mtry_income, min_y_mtry_income, max_y_mtry_income = tax_params

    tax1_params = (J, S, retire, a_etr_income, b_etr_income, c_etr_income, d_etr_income, e_etr_income, f_etr_income,
                   min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income, h_wealth, p_wealth, m_wealth, tau_payroll)
    tax1 = tax.total_taxes(r, b_s, w, e, n_guess, BQ, lambdas,
                           factor, T_H, None, 'SS', False, tax1_params, theta, tau_bq)
    cons = get_cons(r, b_s, w, e, n_guess, BQ, lambdas, b_splus1, params, tax1)
    income = (r * b_s + w * e * n_guess) * factor
    
    #tau_inc_params = (a_tax_income, b_tax_income, c_tax_income, d_tax_income, e_tax_income, f_tax_income,
    #                  min_x_tax_income, max_x_tax_income, min_y_tax_income, max_y_tax_income)
    #deriv = (1 - tau_payroll - tax.tau_labor_deriv(r, b_s, w, e, n_guess, factor, tau_inc_params) 
    #         - tax.tau_income(r, b_s, w, e, n_guess, factor, tau_inc_params))

    mtr_lab_params = (a_mtrx_income, b_mtrx_income, c_mtrx_income, d_mtrx_income, e_mtrx_income, f_mtrx_income,
                      min_x_mtrx_income, max_x_mtrx_income, min_y_mtrx_income, max_y_mtrx_income)
    deriv = (1 - tau_payroll - tax.MTR_labor(r, b_s, w, e, n_guess, factor, mtr_lab_params))

    #deriv = 1 - tau_payroll - tax.tau_income(r, b_s, w, e, n_guess, factor, params) - tax.tau_income_deriv(
    #    r, b_s, w, e, n_guess, factor, params) * income

    euler = marg_ut_cons(cons, params) * w * deriv * e - \
        marg_ut_labor(n_guess, chi_n, params)

    return euler


def constraint_checker_SS(bssmat, nssmat, cssmat, params):
    '''
    Inputs:
        bssmat = steady state distribution of capital ((S-1)xJ array)
        nssmat = steady state distribution of labor (SxJ array)
        cssmat = steady state distribution of consumption (SxJ array)
        params = list of parameters (list)
    Output:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    print 'Checking constraints on capital, labor, and consumption.'
    
    ltilde = params 

    if (bssmat < 0).any():
        print '\tWARNING: There is negative capital stock'
    flag2 = False
    if (nssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (nssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Consumption violates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, params):
    '''
    Inputs:
        b_dist = distribution of capital (SxJ array)
        n_dist = distribution of labor (SxJ array)
        c_dist = distribution of consumption (SxJ array)
        t = time period (scalar)
        params = list of parameters (list)
    Output:
        Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    ltilde = params
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
