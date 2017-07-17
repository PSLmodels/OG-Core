'''
------------------------------------------------------------------------
A 'driver' script to use the ogusa package for an Overlapping Generations
macroeconomic model of the United States

This will run the steady state solver as well as time path iteration.

This py-file creates the following other file(s):
        OUTPUT/Saved_moments/params_given.pkl
        OUTPUT/Saved_moments/params_changed.pkl
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
Import Packages
------------------------------------------------------------------------
'''

import numpy as np
import parameters as pmtr

# import cPickle as pickle
# import os

# import ogusa
# ogusa.parameters.DATASET = 'REAL'

# import ogusa.SS
# import ogusa.TPI
# from ogusa import parameters, wealth, labor, demographics, income, SS, TPI

'''
------------------------------------------------------------------------
Setup
------------------------------------------------------------------------
get_baseline = Flag to run baseline or tax experiments (bool)
calibrate_model = Flag to run calibration of chi values or not (bool)
------------------------------------------------------------------------
'''
params_all = pmtr.get_full_parameters()

# globals().update(ogusa.parameters.get_parameters())

# # Generate Wealth data moments
# output_dir = "./OUTPUT"
# wealth.get_wealth_data(lambdas, J, flag_graphs, output_dir)

# # Generate labor data moments
# labor.labor_data_moments(flag_graphs)

# get_baseline = True
# calibrate_model = False

# # List of parameter names that will not be changing (unless we decide to
# # change them for a tax experiment)
# param_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
#              'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
#              'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
#              'b_ellipse', 'k_ellipse', 'upsilon',
#              'a_tax_income', 'chi_b_guess', 'chi_n_guess',
#              'b_tax_income', 'c_tax_income', 'd_tax_income',
#              'tau_payroll', 'tau_bq', 'calibrate_model',
#              'retire', 'mean_income_data', 'g_n_vector',
#              'h_wealth', 'p_wealth', 'm_wealth', 'get_baseline',
#              'omega', 'g_n_ss', 'omega_SS', 'surv_rate', 'e', 'rho']

# '''
# ------------------------------------------------------------------------
#     Run SS with minimization to fit chi_b and chi_n
# ------------------------------------------------------------------------
# '''

# # This is the simulation before getting the replacement rate values
# sim_params = {}
# for key in param_names:
#     sim_params[key] = globals()[key]

# #pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))
# #call(['python', 'SS.py'])
# income_tax_params, wealth_tax_params, ellipse_params, ss_parameters, iterative_params = SS.create_steady_state_parameters(**sim_params)


# ss_outputs = SS.run_steady_state(ss_parameters, iterative_params, get_baseline, calibrate_model)


# '''
# ------------------------------------------------------------------------
#     Run the baseline TPI simulation
# ------------------------------------------------------------------------
# '''

# ss_outputs['get_baseline'] = get_baseline
# income_tax_params, wealth_tax_params, ellipse_params, parameters, N_tilde, omega_stationary, K0, b_sinit, \
# b_splus1init, L0, Y0, w0, r0, BQ0, T_H_0, tax0, c0, initial_b, initial_n = TPI.create_tpi_params(**sim_params)
# ss_outputs['income_tax_params'] = income_tax_params
# ss_outputs['wealth_tax_params'] = wealth_tax_params
# ss_outputs['ellipse_params'] = ellipse_params
# ss_outputs['parameters'] = parameters
# ss_outputs['N_tilde'] = N_tilde
# ss_outputs['omega_stationary'] = omega_stationary
# ss_outputs['K0'] = K0
# ss_outputs['b_sinit'] = b_sinit
# ss_outputs['b_splus1init'] = b_splus1init
# ss_outputs['L0'] = L0
# ss_outputs['Y0'] = Y0
# ss_outputs['r0'] = r0
# ss_outputs['BQ0'] = BQ0
# ss_outputs['T_H_0'] = T_H_0
# ss_outputs['tax0'] = tax0
# ss_outputs['c0'] = c0
# ss_outputs['initial_b'] = initial_b
# ss_outputs['initial_n'] = initial_n
# ss_outputs['tau_bq'] = tau_bq
# ss_outputs['g_n_vector'] = g_n_vector
# TPI.run_time_path_iteration(**ss_outputs)

# '''
# ------------------------------------------------------------------------
#     Alter desired tax parameters
# ------------------------------------------------------------------------
# '''

# # get_baseline = False
# # dictionary = {}
# # for key in param_names:
# #     dictionary[key] = globals()[key]
# # pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))

# # # Altered parameters
# # d_tax_income = .42

# # # List of all parameters that have been changed for the tax experiment
# # params_changed_names = ['d_tax_income']
# # dictionary = {}
# # for key in params_changed_names:
# #     dictionary[key] = globals()[key]
# # pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_changed.pkl", "w"))

# '''
# ------------------------------------------------------------------------
#     Run TPI for tax experiment
# ------------------------------------------------------------------------
# '''

# # call(['python', 'TPI.py'])
# # import TPI
