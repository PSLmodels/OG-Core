
'''
This module should be organized as follows:

Main function:
chi_estimate() = returns chi_n, chi_b
    - calls: 
        wealth.get_wealth_data() - returns data moments on wealth distribution
        labor.labor_data_moments() - returns data moments on labor supply
        minstat() - returns min of statistical objective function
            model_moments() - returns model moments
                SS.run_SS() - return SS distributions

'''

'''
------------------------------------------------------------------------
Last updated: 4/11/2016

Uses a simulated method of moments to calibrate the chi_n adn chi_b 
parameters of OG-USA.

This py-file calls the following other file(s):
    wealth.get_wealth_data()
    labor.labor_data_moments()
    SS.run_SS

This py-file creates the following other file(s): None
------------------------------------------------------------------------
'''

import numpy as np
import scipy.optimize as opt
import wealth
import labor
import SS

def chi_estimate(income_tax_params, ss_params, iterative_params, chi_guesses, baseline_dir="./OUTPUT"):

    # unpack tuples of parameters
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, \
                  imm_rates, e, retire, mean_income_data, h_wealth, p_wealth,\
                  m_wealth, b_ellipse, upsilon = ss_params
    chi_b_guess, chi_n_guess = chi_guesses

    flag_graphs = False

    # Generate Wealth data moments
    wealth_moments = wealth.get_wealth_data(lambdas, J, flag_graphs)

    # Generate labor data moments
    labor_moments = labor.labor_data_moments(flag_graphs,baseline_dir)

    # combine moments
    data_moments = list(wealth_moments.flatten()) + list(labor_moments.flatten())

    # call minimizer
    bnds = np.tile(np.array([1e-12, None]),(S+J,1)) # Need (1e-12, None) S+J times
    #min_args = () data_moments
    chi_guesses_flat = list(chi_b_guess.flatten()) + list(chi_n_guess.flatten())
    min_args = data_moments, income_tax_params, ss_params, iterative_params, chi_guesses_flat, baseline_dir
    print 'size of chi: ', len(chi_guesses_flat)
    print 'bnds size: ', bnds.shape
    est_output = opt.minimize(minstat, chi_guesses_flat, args=(min_args), method="L-BFGS-B", bounds=bnds,
                    tol=1e-15)
    chi_params = est_output.x
    objective_func_min = est_out.fun 

    return chi_params




def minstat(chi_guesses, *args):
    '''
    --------------------------------------------------------------------
    This function generates the weighted sum of squared differences
    between the model and data moments.


    RETURNS: wssqdev
    --------------------------------------------------------------------
    '''

    data_moments, income_tax_params, ss_params, iterative_params, chi_params, baseline_dir = args
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, \
                  imm_rates, e, retire, mean_income_data, h_wealth, p_wealth,\
                  m_wealth, b_ellipse, upsilon = ss_params
    chi_b = chi_guesses[:J]
    chi_n = chi_guesses[J:]
    chi_params = (chi_b, chi_n)
    ss_output = SS.run_SS(income_tax_params, ss_params, iterative_params, chi_params, True, baseline_dir)

    model_moments = calc_moments(ss_output, lambdas, S, J)

    # distance with levels
    distance = ((np.array(model_moments) - np.array(data_moments))**2).sum()
    print 'DATA and MODEL DISTANCE: ', distance

    # # distance with percentage diffs
    # distance = (((model_moments - data_moments)/data_moments)**2).sum()

    return distance


def calc_moments(ss_output, lambdas, S, J):
    '''
    --------------------------------------------------------------------
    This function calculates moments from the SS output that correspond
    to the data moments used for estimation.

    RETURNS: model_moments
    --------------------------------------------------------------------
    '''

    wealth_model = ss_output['bssmat']
    factor_model = ss_output['factor_ss']

    # wealth moments
    # Convert wealth levels from model to dollar terms
    wealth_model_dollars = wealth_model * factor_model
    model_wealth_moments = np.zeros(2 * J)
    # Look at the percent difference between the fits for the first age group (20-44) and second age group (45-65)
    #   The wealth_data_array moment indices are shifted because they start at age 18
    # The :: indices is so that we look at the 2 moments for the lowest group,
    # the 2 moments for the second lowest group, etc in order
    model_wealth_moments[0::2] = np.mean(wealth_model_dollars[:24],axis=0)
    model_wealth_moments[1::2] = np.mean(wealth_model_dollars[24:45],axis=0)

    # labor moments
    n = ss_output['nssmat']
    model_labor_moments = (n.reshape(S, J) * lambdas.reshape(1, J)).sum(axis=1)

    # combine moments
    model_moments = list(model_wealth_moments.flatten()) + list(model_labor_moments.flatten())


    return model_moments

