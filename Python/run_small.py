'''
A 'smoke test' for the dynamic package. Uses a fake data set to run the
baseline
'''

import cPickle as pickle
import os
from glob import glob
from subprocess import call
import numpy as np
import time

import dynamic
dynamic.parameters.DATASET = 'SMALL'

import dynamic.SS
import dynamic.TPI
from dynamic import parameters, wealth, labor, demographics, income, SS, TPI

globals().update(dynamic.parameters.get_parameters())

def runner():

    # Flag to prevent graphing from occuring in demographic, income, wealth, and labor files
    flag_graphs = False
    # Generate Income and Demographic parameters
    omega, g_n, omega_SS, surv_rate = demographics.get_omega(S, J, T, lambdas, starting_age, ending_age, E, flag_graphs)
    #      S, J, T, lambdas, starting_age, ending_age, E, flag_graphs)
    #e = income.get_test_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)
    # Create a fake 'e'
    e = np.array([[0.25, 1.25]] * 10)
    rho = 1-surv_rate
    rho[-1] = 1.0

    # Generate Wealth data moments
    output_dir = "./TESTED_DONT_TOUCH/REF/"
    wealth.get_wealth_data(lambdas, J, flag_graphs, output_dir)

    # Generate labor data moments
    labor.labor_data_moments(flag_graphs)

    # Remove pickle of altered parameters -- reset the experiment
    if os.path.exists("OUTPUT/Saved_moments/params_changed.pkl"):
        os.remove("OUTPUT/Saved_moments/params_changed.pkl")

    get_baseline = True

    # List of parameter names that will not be changing (unless we decide to
    # change them for a tax experiment.
    param_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
                'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
                'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
                'b_ellipse', 'k_ellipse', 'upsilon',
                'a_tax_income',
                'b_tax_income', 'c_tax_income', 'd_tax_income',
                'tau_payroll', 'tau_bq',
                'retire', 'mean_income_data',
                'h_wealth', 'p_wealth', 'm_wealth', 'get_baseline',
                'omega', 'g_n', 'omega_SS', 'surv_rate', 'e', 'rho']

    '''
    ------------------------------------------------------------------------
        Run SS with minimization to fit chi_b and chi_n
    ------------------------------------------------------------------------
    '''

    # This is the simulation before getting the replacement rate values
    sim_params = {}
    glbs = globals()
    lcls = locals()
    for key in param_names:
        if key in glbs:
            sim_params[key] = glbs[key]
        else:
            sim_params[key] = lcls[key]

    income_tax_params, wealth_tax_params, ellipse_params, ss_parameters, iterative_params = SS.create_steady_state_parameters(**sim_params)

    print "got here"

    before = time.time()
    ss_outputs = SS.run_steady_state(ss_parameters, iterative_params, get_baseline)

    '''
    ------------------------------------------------------------------------
        Run the baseline TPI simulation
    ------------------------------------------------------------------------
    '''

    ss_outputs['get_baseline'] = get_baseline
    income_tax_params, wealth_tax_params, ellipse_params, parameters, N_tilde, omega_stationary, K0, b_sinit, b_splus1init, L0, Y0, w0, r0, BQ0, T_H_0, tax0, c0, initial_b, initial_n = TPI.create_tpi_params(**sim_params)
    ss_outputs['income_tax_params'] = income_tax_params
    ss_outputs['wealth_tax_params'] = wealth_tax_params
    ss_outputs['ellipse_params'] = ellipse_params
    ss_outputs['parameters'] = parameters
    ss_outputs['N_tilde'] = N_tilde
    ss_outputs['omega_stationary'] = omega_stationary
    ss_outputs['K0'] = K0
    ss_outputs['b_sinit'] = b_sinit
    ss_outputs['b_splus1init'] = b_splus1init
    ss_outputs['L0'] = L0
    ss_outputs['Y0'] = Y0
    ss_outputs['r0'] = r0
    ss_outputs['BQ0'] = BQ0
    ss_outputs['T_H_0'] = T_H_0
    ss_outputs['tax0'] = tax0
    ss_outputs['c0'] = c0
    ss_outputs['initial_b'] = initial_b
    ss_outputs['initial_n'] = initial_n
    ss_outputs['tau_bq'] = tau_bq

    with open("ss_outputs.pkl", 'wb') as fp:
        pickle.dump(ss_outputs, fp)

    TPI.run_time_path_iteration(**ss_outputs)

    print "took {0} seconds to get that part done.".format(time.time() - before)

if __name__ == "__main__":
    runner()
