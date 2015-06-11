'''
------------------------------------------------------------------------
Last updated 6/4/2015

Miscellaneous functions for taxes in SS and TPI.

------------------------------------------------------------------------
'''

# Packages
import numpy as np



'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data and
        simulated data
    '''
    frac = (simul - data)/data
    output = np.abs(frac)
    return output


def convex_combo(var1, var2, params):
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    combo = nu * var1 + (1-nu)*var2
    return combo


def check_wealth_calibration(wealth_model, factor_model, wealth_data, params):
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    wealth_model_dollars = wealth_model * factor_model
    wealth_fits = np.zeros(2*J)
    wealth_fits[0::2] = perc_dif_func(np.mean(wealth_model_dollars[:24], axis=0), np.mean(wealth_data[2:26], axis=0))
    wealth_fits[1::2] = perc_dif_func(np.mean(wealth_model_dollars[24:45], axis=0), np.mean(wealth_data[26:47], axis=0))
    return wealth_fits
