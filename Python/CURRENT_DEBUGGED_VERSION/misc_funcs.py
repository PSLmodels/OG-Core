'''
------------------------------------------------------------------------
Last updated 7/16/2015

Miscellaneous functions for SS and TPI.

This python files calls:
    OUTPUT/Saved_moments/wealth_data_moments.pkl

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
    Inputs:
        simul = model moments (any shape)
        data  = data moments (same shape as simul)
    Output:
        output = absolute percent difference between data and model moments (same shape as simul)
    '''
    frac = (simul - data)/data
    output = np.abs(frac)
    return output


def convex_combo(var1, var2, params):
    '''
    Takes the convex combination of two variables, where nu is the value
    between 0 and 1 in params.
    Inputs:
        var1 = (any shape)
        var2 = (same shape as var1)
        params = parameters list from model (list) (only nu is needed...perhaps it should just take that as an input)
    Outputs:
        combo = convex combination of var1 and var2 (same shape as var1)
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    combo = nu * var1 + (1-nu)*var2
    return combo


def check_wealth_calibration(wealth_model, factor_model, params):
    '''
    Creates a vector of the percent differences between the
    model and data wealth moments for the two age groups for
    each J group.
    Inputs:
        wealth_model = model wealth levels (SxJ array)
        factor_model = factor to convert wealth levels to dollars (scalar)
        params = parameters list from model (list)
    Outputs:
        wealth_fits = Fits for how well the model wealth levels match the data wealth levels ((2*J)x1 array)
    '''
    # Import the wealth data moments
    wealth_dict = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments.pkl", "r"))
    # Set lowest ability group's wealth to be a positive, not negative, number for the calibration
    wealth_dict['wealth_data_array'][2:26, 0] = 500.0
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    # Convert wealth levels from model to dollar terms
    wealth_model_dollars = wealth_model * factor_model
    wealth_fits = np.zeros(2*J)
    # Look at the percent difference between the fits for the first age group (20-44) and second age group (45-65)
    #   The wealth data moment indices are shifted because they start at age 18
    wealth_fits[0::2] = perc_dif_func(np.mean(wealth_model_dollars[:24], axis=0), np.mean(wealth_dict['wealth_data_array'][2:26], axis=0))
    wealth_fits[1::2] = perc_dif_func(np.mean(wealth_model_dollars[24:45], axis=0), np.mean(wealth_dict['wealth_data_array'][26:47], axis=0))
    return wealth_fits
