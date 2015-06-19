'''
------------------------------------------------------------------------
Last updated 6/4/2015

Household functions for taxes in SS and TPI.

------------------------------------------------------------------------
'''

# Packages
import numpy as np

import tax_funcs as tax

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_K(b, weights):
    '''
    Inputs: b, weights

    Returns:    Aggregate Capital
    '''
    K_now = np.sum(b * weights)
    return K_now


def marg_ut_cons(c, params):
    '''
    Inputs: Consumption, parameters

    Returns:    Marginal Utility of Consumption
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    output = c**(-sigma)
    return output


def marg_ut_labor(n, chi_n, params):
    '''
    Inputs: Labor, chi^n_s, parameters

    Returns:    Marginal Utility of Labor
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    deriv = b_ellipse * (1/ltilde) * ((1 - (n / ltilde) ** upsilon) ** (
        (1/upsilon)-1)) * (n / ltilde) ** (upsilon - 1)
    output = chi_n * deriv
    return output


def get_cons(r, b_s, w, e, n, BQ, lambdas, b_splus1, params, net_tax):
    '''
    Inputs: rental rate, capital stock (s), wage, e, labor stock,
                bequests, lambdas, capital stock (s+1), parameters, taxes

    Returns:    Consumption
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    cons = (1 + r)*b_s + w*e*n + BQ / lambdas - b_splus1*np.exp(g_y) - net_tax
    return cons


def euler_savings_func(w, r, e, n_guess, b_s, b_splus1, b_splus2, BQ, factor, T_H, chi_b, params, theta, tau_bq, rho, lambdas):
    '''
    Inputs:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b_s       = distribution of capital in period t (S x J array)
        b_splus1       = distribution of capital in period t+1 (S x J array)
        b_splus2       = distribution of capital in period t+2 (S x J array)
        BQ        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        T_H  = lump sum transfer from the government to the households
        chi_b    = discount factor of savings
        params = parameters
        theta = replacement rates
        tau_bq = bequest tax parameters
        rho = mortality rates
        lambdas = bin weights

    Returns:
        Value of Euler error.
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    e_extended = np.array(list(e) + [0])
    n_extended = np.array(list(n_guess) + [0])
    tax1 = tax.total_taxes(r, b_s, w, e, n_guess, BQ, lambdas, factor, T_H, None, 'SS', False, params, theta, tau_bq)
    tax2 = tax.total_taxes(r, b_splus1, w, e_extended[1:], n_extended[1:], BQ, lambdas, factor, T_H, None, 'SS', True, params, theta, tau_bq)
    cons1 = get_cons(r, b_s, w, e, n_guess, BQ, lambdas, b_splus1, params, tax1)
    cons2 = get_cons(r, b_splus1, w, e_extended[1:], n_extended[1:], BQ, lambdas, b_splus2, params, tax2)
    income = (r * b_splus1 + w * e_extended[1:] * n_extended[1:]) * factor
    deriv = (
        1 + r*(1-tax.tau_income(r, b_s, w, e_extended[1:], n_extended[1:], factor, params)-tax.tau_income_deriv(
            r, b_s, w, e_extended[1:], n_extended[1:], factor, params)*income)-tax.tau_w_prime(b_splus1, params)*b_splus1-tax.tau_wealth(b_splus1, params))
    savings_ut = rho * np.exp(-sigma * g_y) * chi_b * b_splus1 ** (-sigma)
    euler = marg_ut_cons(cons1, params) - beta * (1-rho) * deriv * marg_ut_cons(
        cons2, params) * np.exp(-sigma * g_y) - savings_ut
    return euler


def euler_labor_leisure_func(w, r, e, n_guess, b_s, b_splus1, BQ, factor, T_H, chi_n, params, theta, tau_bq, lambdas):
    '''
    Inputs:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b_s       = distribution of capital in period t (S x J array)
        b_splus1       = distribution of capital in period t+1 (S x J array)
        BQ        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        T_H  = lump sum transfer from the government to the households
        chi_n    = discount factor of labor
        params = parameters
        theta = replacement rates
        tau_bq = bequest tax parameters
        lambdas = bin weights

    Returns:
        Value of Euler error.
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    tax1 = tax.total_taxes(r, b_s, w, e, n_guess, BQ, lambdas, factor, T_H, None, 'SS', False, params, theta, tau_bq)
    cons = get_cons(r, b_s, w, e, n_guess, BQ, lambdas, b_splus1, params, tax1)
    income = (r * b_s + w * e * n_guess) * factor
    deriv = 1 - tau_payroll - tax.tau_income(r, b_s, w, e, n_guess, factor, params) - tax.tau_income_deriv(
        r, b_s, w, e, n_guess, factor, params) * income
    euler = marg_ut_cons(cons, params) * w * deriv * e - marg_ut_labor(n_guess, chi_n, params)
    return euler


def constraint_checker_SS(bssmat, nssmat, cssmat, params):
    '''
    Inputs:
        bssmat = steady state distribution of capital ((S-1)xJ array)
        nssmat = steady state distribution of labor (SxJ array)
        cssmat = steady state distribution of consumption (SxJ array)

    Created Variables:
        flag1 = False if all borrowing constraints are met, true
               otherwise.
        flag2 = False if all labor constraints are met, true otherwise

    Returns:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    print 'Checking constraints on capital, labor, and consumption.'
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    flag1 = False
    if bssmat.sum() <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to zero.'
        flag1 = True
    if (bssmat < 0).any():
        print '\tWARNING: There is negative capital stock'
    if flag1 is False:
        print '\tThere were no violations of the borrowing constraints.'
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


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, params, N_tilde):
    '''
    Inputs:
        b_dist = distribution of capital ((S-1)xJ array)
        n_dist = distribution of labor (SxJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        c_dist = distribution of consumption (SxJ array)

    Returns:
        Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    if b_dist.sum() / N_tilde[t] <= 0:
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
