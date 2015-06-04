'''
------------------------------------------------------------------------
Last updated 6/3/2015

Firm functions for taxes in SS and TPI.

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


def get_r(Y_now, K_now, params):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_Y(K_now, L_now, params):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    Y_now = Z * (K_now ** alpha) * ((L_now) ** (1 - alpha))
    return Y_now


def get_w(Y_now, L_now, params):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    w_now = (1 - alpha) * Y_now / L_now
    return w_now


def get_L(e, n, weights):
    '''
    Parameters: e, n

    Returns:    Aggregate labor
    '''
    L_now = np.sum(e * weights * n)
    return L_now
