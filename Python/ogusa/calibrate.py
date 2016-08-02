
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
Last updated: 7/27/2016

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
import pandas as pd
import os
import pickle
import wealth
import labor
import SS
import utils

def chi_estimate(income_tax_params, ss_params, iterative_params, chi_guesses, baseline_dir="./OUTPUT"):
    '''
    --------------------------------------------------------------------
    This function calls others to obtain the data momements and then
    runs the simulated method of moments estimation by calling the
    minimization routine.

    INPUTS:
    income_tax_parameters = length 4 tuple, (analytical_mtrs, etr_params, mtrx_params, mtry_params)
    ss_parameters         = length 21 tuple, (J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                            g_n_ss, tau_payroll, retire, mean_income_data,\
                            h_wealth, p_wealth, m_wealth, b_ellipse, upsilon)
    iterative_params      = [2,] vector, vector with max iterations and tolerance
                             for SS solution
    chi_guesses           = [J+S,] vector, initial guesses of chi_b and chi_n stacked together
    baseline_dir          = string, path where baseline results located


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    wealth.compute_wealth_moments()
    labor.labor_data_moments()
    minstat()

    OBJECTS CREATED WITHIN FUNCTION:
    wealth_moments     = [J+2,] array, wealth moments from data
    labor_moments      = [S,] array, labor moments from data
    data_moments       = [J+2+S,] array, wealth and labor moments stacked
    bnds               = [S+J,] array, bounds for parameter estimates
    chi_guesses_flat   =  [J+S,] vector, initial guesses of chi_b and chi_n stacked
    min_arg            = length 6 tuple, variables needed for minimizer
    est_output         = dictionary, output from minimizer
    chi_params         = [J+S,] vector, parameters estimates for chi_b and chi_n stacked
    objective_func_min = scalar, minimum of statistical objective function


    OUTPUT:
    ./baseline_dir/Calibration/chi_estimation.pkl


    RETURNS: chi_params
    --------------------------------------------------------------------
    '''

    # unpack tuples of parameters
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, \
                  imm_rates, e, retire, mean_income_data, h_wealth, p_wealth,\
                  m_wealth, b_ellipse, upsilon = ss_params
    chi_b_guess, chi_n_guess = chi_guesses

    flag_graphs = False

    # specify bootstrap iterations
    n = 10000

    # Generate Wealth data moments
    scf, data = wealth.get_wealth_data()
    wealth_moments = wealth.compute_wealth_moments(scf, lambdas, J)


    # Generate labor data moments
    cps = labor.get_labor_data()
    labor_moments = labor.compute_labor_moments(cps, S)


    # combine moments
    data_moments = list(wealth_moments.flatten()) + list(labor_moments.flatten())


    # determine weighting matrix
    optimal_weight = False
    if optimal_weight:
        VCV_wealth_moments = wealth.VCV_moments(scf, n, lambdas, J)
        VCV_labor_moments = labor.VCV_moments(cps, n, lambdas, S)
        VCV_data_moments = np.zeros((J+2+S,J+2+S))
        VCV_data_moments[:J+2,:J+2] = VCV_wealth_moments
        VCV_data_moments[J+2:,J+2:] = VCV_labor_moments
        W = np.linalg.inv(VCV_data_moments)
        #np.savetxt('VCV_data_moments.csv',VCV_data_moments)
    else:
        W = np.identity(J+2+S)


    # call minimizer
    bnds = np.tile(np.array([1e-12, None]),(S+J,1)) # Need (1e-12, None) S+J times
    chi_guesses_flat = list(chi_b_guess.flatten()) + list(chi_n_guess.flatten())

    min_args = data_moments, W, income_tax_params, ss_params, \
               iterative_params, chi_guesses_flat, baseline_dir
    # est_output = opt.minimize(minstat, chi_guesses_flat, args=(min_args), method="L-BFGS-B", bounds=bnds,
    #                 tol=1e-15, options={'maxfun': 1, 'maxiter': 1, 'maxls': 1})
    # est_output = opt.minimize(minstat, chi_guesses_flat, args=(min_args), method="L-BFGS-B", bounds=bnds,
    #                 tol=1e-15)
    # chi_params = est_output.x
    # objective_func_min = est_output.fun
    #
    # # pickle output
    # utils.mkdirs(os.path.join(baseline_dir, "Calibration"))
    # est_dir = os.path.join(baseline_dir, "Calibration/chi_estimation.pkl")
    # pickle.dump(est_output, open(est_dir, "wb"))
    #
    # # save data and model moments and min stat to csv
    # # to then put in table of paper
    chi_params = chi_guesses_flat
    chi_b = chi_params[:J]
    chi_n = chi_params[J:]
    chi_params_list = (chi_b, chi_n)

    ss_output = SS.run_SS(income_tax_params, ss_params, iterative_params, chi_params_list, True, baseline_dir)
    model_moments = calc_moments(ss_output, omega_SS, lambdas, S, J)

    # # make dataframe for results
    # columns = ['data_moment', 'model_moment', 'minstat']
    # moment_fit = pd.DataFrame(index=range(0,J+2+S), columns=columns)
    # moment_fit = moment_fit.fillna(0) # with 0s rather than NaNs
    # moment_fit['data_moment'] = data_moments
    # moment_fit['model_moment'] = model_moments
    # moment_fit['minstat'] = objective_func_min
    # est_dir = os.path.join(baseline_dir, "Calibration/moment_results.pkl")s
    # moment_fit.to_csv(est_dir)

    # calculate std errors
    h = 0.0001  # pct change in parameter
    model_moments_low = np.zeros((len(chi_params),len(model_moments)))
    model_moments_high = np.zeros((len(chi_params),len(model_moments)))
    chi_params_low = chi_params
    chi_params_high = chi_params
    for i in range(len(chi_params)):
        chi_params_low[i] = chi_params[i]*(1+h)
        chi_b = chi_params_low[:J]
        chi_n = chi_params_low[J:]
        chi_params_list = (chi_b, chi_n)
        ss_output = SS.run_SS(income_tax_params, ss_params, iterative_params, chi_params_list, True, baseline_dir)
        model_moments_low[i,:] = calc_moments(ss_output, omega_SS, lambdas, S, J)

        chi_params_high[i] = chi_params[i]*(1+h)
        chi_b = chi_params_high[:J]
        chi_n = chi_params_high[J:]
        chi_params_list = (chi_b, chi_n)
        ss_output = SS.run_SS(income_tax_params, ss_params, iterative_params, chi_params_list, True, baseline_dir)
        model_moments_high[i,:] = calc_moments(ss_output, omega_SS, lambdas, S, J)

    deriv_moments = (np.asarray(model_moments_high) - np.asarray(model_moments_low)).T/(2.*h*np.asarray(chi_params))
    VCV_params = np.linalg.inv(np.dot(np.dot(deriv_moments.T,W),deriv_moments))
    std_errors_chi = (np.diag(VCV_params))**(1/2.)
    sd_dir = os.path.join(baseline_dir, "Calibration/chi_std_errors.pkl")
    pickle.dump(std_errors_chi, open(sd_dir, "wb"))

    np.savetxt('chi_std_errors.csv',std_errors_chi)


    return chi_params




def minstat(chi_guesses, *args):
    '''
    --------------------------------------------------------------------
    This function generates the weighted sum of squared differences
    between the model and data moments.

    INPUTS:
    chi_guesses = [J+S,] vector, initial guesses of chi_b and chi_n stacked together
    arg         = length 6 tuple, variables needed for minimizer


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    SS.run_SS()
    calc_moments()

    OBJECTS CREATED WITHIN FUNCTION:
    ss_output     = dictionary, variables from SS of model
    model_moments = [J+2+S,] array, moments from the model solution
    distance      = scalar, weighted, squared deviation between data and model moments

    RETURNS: distance
    --------------------------------------------------------------------
    '''

    data_moments, W, income_tax_params, ss_params, iterative_params, chi_params, baseline_dir = args
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, \
                  imm_rates, e, retire, mean_income_data, h_wealth, p_wealth,\
                  m_wealth, b_ellipse, upsilon = ss_params
    chi_b = chi_guesses[:J]
    chi_n = chi_guesses[J:]
    chi_params = (chi_b, chi_n)
    ss_output = SS.run_SS(income_tax_params, ss_params, iterative_params, chi_params, True, baseline_dir)

    model_moments = calc_moments(ss_output, omega_SS, lambdas, S, J)

    # distance with levels
    distance = np.dot(np.dot((np.array(model_moments) - np.array(data_moments)).T,W),
                   np.array(model_moments) - np.array(data_moments))
    #distance = ((np.array(model_moments) - np.array(data_moments))**2).sum()
    print 'DATA and MODEL DISTANCE: ', distance

    # # distance with percentage diffs
    # distance = (((model_moments - data_moments)/data_moments)**2).sum()

    return distance


def calc_moments(ss_output, omega_SS, lambdas, S, J):
    '''
    --------------------------------------------------------------------
    This function calculates moments from the SS output that correspond
    to the data moments used for estimation.

    INPUTS:
    ss_output = dictionary, variables from SS of model
    omega_SS  = [S,] array, SS population distribution over age
    lambdas   = [J,] array, proportion of population of each ability type
    S         = integer, number of ages
    J         = integer, number of ability types

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    the_inequalizer()

    OBJECTS CREATED WITHIN FUNCTION:
    model_wealth_moments = [J+2,] array, wealth moments from the model
    model_labor_moments  = [S,] array, labor moments from the model
    model_moments        = [J+2+S,] array, wealth and data moments from the model solution
    distance             = scalar, weighted, squared deviation between data and model moments

    RETURNS: distance

    RETURNS: model_moments
    --------------------------------------------------------------------
    '''
    # unpack relevant SS variables
    bssmat = ss_output['bssmat']
    factor = ss_output['factor_ss']
    n = ss_output['nssmat']

    # wealth moments
    model_wealth_moments = the_inequalizer(bssmat, omega_SS, lambdas, factor, S, J)

    # labor moments
    model_labor_moments = (n.reshape(S, J) * lambdas.reshape(1, J)).sum(axis=1)

    # combine moments
    model_moments = list(model_wealth_moments.flatten()) + list(model_labor_moments.flatten())

    return model_moments


def the_inequalizer(dist, pop_weights, ability_weights, factor, S, J):
    '''
    --------------------------------------------------------------------
    Generates three measures of inequality.

    Inputs:
        dist            = [S,J] array, distribution of endogenous variables over age and lifetime income group
        pop_weights     = [S,] vector, fraction of population by each age
        ability_weights = [J,] vector, fraction of population for each lifetime income group
        factor          = scalar, factor relating model units to dollars
        S               = integer, number of economically active periods in lifetime
        J               = integer, number of ability types

    Functions called: None

    Objects in function:
        weights           = [S,J] array, fraction of population for each age and lifetime income group
        flattened_dist    = [S*J,] vector, vectorized dist
        flattened_weights = [S*J,] vector, vectorized weights
        sort_dist         = [S*J,] vector, ascending order vector of dist
        loc_90th          = integer, index of 90th percentile
        loc_10th          = integer, index of 10th percentile
        loc_99th          = integer, index of 99th percentile

    Returns: measure of inequality
    --------------------------------------------------------------------
    '''

    weights = np.tile(pop_weights.reshape(S, 1), (1, J)) * \
    ability_weights.reshape(1, J)
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]
    cum_weights = np.cumsum(sort_weights)

    # gini
    p = cum_weights/cum_weights.sum()
    nu = np.cumsum(sort_dist*sort_weights)
    nu = nu/nu[-1]
    gini_coeff = (nu[1:]*p[:-1]).sum() - (nu[:-1] * p[1:]).sum()


    # variance
    ln_dist = np.log(sort_dist*factor) # not scale invariant
    weight_mean = (ln_dist*sort_weights).sum()/sort_weights.sum()
    var_ln_dist = ((sort_weights*((ln_dist-weight_mean)**2)).sum())*(1./(sort_weights.sum()))


    # 90/10 ratio
    loc_90th = np.argmin(np.abs(cum_weights - .9))
    loc_10th = np.argmin(np.abs(cum_weights - .1))
    ratio_90_10 = sort_dist[loc_90th] / sort_dist[loc_10th]

    # top 10% share
    top_10_share= (sort_dist[loc_90th:] * sort_weights[loc_90th:]
           ).sum() / (sort_dist * sort_weights).sum()

    # top 1% share
    loc_99th = np.argmin(np.abs(cum_weights - .99))
    top_1_share = (sort_dist[loc_99th:] * sort_weights[loc_99th:]
           ).sum() / (sort_dist * sort_weights).sum()

    # calculate percentile shares (percentiles based on lambdas input)
    dist_weight = (sort_weights*sort_dist)
    total_dist_weight = dist_weight.sum()
    cumsum = np.cumsum(sort_weights)
    dist_sum = np.zeros((J,))
    cum_weights = ability_weights.cumsum()
    for i in range(J):
        cutoff = sort_weights.sum() / (1./cum_weights[i])
        dist_sum[i] = ((dist_weight[cumsum < cutoff].sum())/total_dist_weight)


    dist_share = np.zeros((J,))
    dist_share[0] = dist_sum[0]
    dist_share[1:] = dist_sum[1:]-dist_sum[0:-1]

    return np.append([dist_share], [gini_coeff,var_ln_dist])
