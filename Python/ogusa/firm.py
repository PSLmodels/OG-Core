'''
------------------------------------------------------------------------
Last updated 7/16/2015

Firm functions for firms in SS and TPI.

------------------------------------------------------------------------
'''

# Packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_r(Y_now, K_now, params):
    '''
    Inputs:
        Y_now = Aggregate output ((T+S)x1 array or scalar)
        K_now = Aggregate capital (same shape as Y_now)
        params = list of parameters (list)
    Output:
        r_now = rental rate (same shape as Y_now)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_Y(K_now, L_now, params):
    '''
    Inputs:
        K_now = Aggregate capital ((T+S)x1 array or scalar)
        L_now = Aggregate labor (same shape as K_now)
        params = list of parameters (list)
    Output:
        Y_now = Aggregate output (same shape as K_now)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    Y_now = Z * (K_now ** alpha) * ((L_now) ** (1 - alpha))
    return Y_now


def get_w(Y_now, L_now, params):
    '''
    Inputs:
        Y_now = Aggregate output ((T+S)x1 array or scalar)
        L_now = Aggregate labor (same shape as Y_now)
        params = list of parameters (list)
    Output:
        w_now = wage rate (same shape as Y_now)
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    w_now = (1 - alpha) * Y_now / L_now
    return w_now


def get_L(e, n, pop_weights, ability_weights, method):
    '''
    Inputs:
        e = ability levels (SxJ array)
        n = labor participation array (SxJ array)
        pop_weights = population weights (Sx1 array)
        ability_weights = ability weights (Jx1 array)
        method = 'SS' or 'TPI'
    Output:
        L_now = Aggregate labor (scalar)
    '''
    L_presum = e * pop_weights * ability_weights * n
    if method == 'SS':
        L_now = L_presum.sum()
    elif method == 'TPI':
        L_now = L_presum.sum(1).sum(1)
    return L_now


def get_I(Knext, Know, delta, g_y, g_n):
    '''
    Inputs:
        Knext = K_t+1 (scalar or Tx1 array)
        Know = K_t (scalar or Tx1 array)
        delta = depreciation rate of capital (scalar)
        g_y = production growth rate (scalar)
        g_n = population growth rate (scalar or Tx1 array)
    Output:
        aggI = aggregate investment (scalar or Tx1 array)
    '''
    aggI = (np.exp(g_y) + g_n * np.exp(g_y)) * Knext - (1.0 - delta) * Know
    return aggI
