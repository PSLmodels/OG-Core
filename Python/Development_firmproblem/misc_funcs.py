'''
------------------------------------------------------------------------
Last updated 6/19/2015

Miscellaneous functions for SS and TPI.

------------------------------------------------------------------------
'''

# Packages
import numpy as np
import cPickle as pickle



'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data
    moments and model moments
    '''
    frac = (simul - data)/data
    output = np.abs(frac)
    return output


def convex_combo(var1, var2, params):
    '''
    Takes the convex combination of two variables, where nu is the value
    between 0 and 1 in params.
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    combo = nu * var1 + (1-nu)*var2
    return combo


def check_wealth_calibration(wealth_model, factor_model, params):
    '''
    Creates a vector of the percent differences between the
    model and data wealth moments for the two age groups for
    each J group.
    '''
    wealth_dict = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments.pkl", "r"))
    # Set lowest ability group's wealth to be a positive, not negative, number for the calibration
    wealth_dict['wealth_data_array'][2:26, 0] = 500.0
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    wealth_model_dollars = wealth_model * factor_model
    wealth_fits = np.zeros(2*J)
    wealth_fits[0::2] = perc_dif_func(np.mean(wealth_model_dollars[:24], axis=0), np.mean(wealth_dict['wealth_data_array'][2:26], axis=0))
    wealth_fits[1::2] = perc_dif_func(np.mean(wealth_model_dollars[24:45], axis=0), np.mean(wealth_dict['wealth_data_array'][26:47], axis=0))
    return wealth_fits
