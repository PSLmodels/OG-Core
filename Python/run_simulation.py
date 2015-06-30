'''
------------------------------------------------------------------------
A 'driver' script to use the dynamic package for an Overlapping Generations
macroeconomic model

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

import cPickle as pickle
import os
from glob import glob
from subprocess import call
from dynamic import parameters, wealth, labor, demographics, income, SS, TPI
import numpy as np

#import income_polynomials as income
#import demographics
#import wealth_data
#import labor_data


'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
lambdas  = desired percentiles of ability groups
scal         = scalar multiplier used in SS files to make the initial value work
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta_annual  = discount factor for one year
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
Z            = total factor productivity parameter in firms' production
               function
delta_annual = depreciation rate of capital for one year
delta        = depreciation rate of capital for each cohort
ltilde       = measure of time each individual is endowed with each
               period
g_y_annual   = annual growth rate of technology
g_y          = growth rate of technology for one cohort
slow_work    = time at which chi_n starts increasing from 1
chi_n_multiplier = scalar which is increased to force the labor
               distribution to 0
maxiter   = Maximum number of iterations that TPI will undergo
mindist_SS   = Cut-off distance between iterations for SS
mindist_TPI   = Cut-off distance between iterations for TPI
nu           = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_nu
b_ellipse    = value of b for elliptical fit of utility function
k_ellipse    = value of k for elliptical fit of utility function
upsilon      = value of omega for elliptical fit of utility function
mean_income_data  = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income = used to calibrate income tax (scalar)
b_tax_income = used to calibrate income tax (scalar)
c_tax_income = used to calibrate income tax (scalar)
d_tax_income = used to calibrate income tax (scalar)
retire       = age in which individuals retire(scalar)
h_wealth     = wealth tax parameter h
m_wealth     = wealth tax parameter m
p_wealth     = wealth tax parameter p
tau_bq       = bequest tax (scalar)
tau_payroll  = payroll tax (scalar)
theta    = payback value for payroll tax (scalar)
e            = S x J matrix of age dependent possible working abilities
               e_s
omega        = T x S x J array of demographics
g_n          = steady state population growth rate
omega_SS     = steady state population distribution
surv_rate    = S x 1 array of survival rates
rho    = S x 1 array of mortality rates
------------------------------------------------------------------------
'''

from dynamic.parameters import *

# Flag to prevent graphing from occuring in demographic, income, wealth, and labor files
flag_graphs = False
# Generate Income and Demographic parameters
omega, g_n, omega_SS, surv_rate = demographics.get_omega(S, J, T, lambdas, starting_age, ending_age, E, flag_graphs)
#      S, J, T, lambdas, starting_age, ending_age, E, flag_graphs)
#e = income.get_e()
e = income.get_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)
rho = 1-surv_rate
rho[-1] = 1.0

# Generate Wealth data moments
output_dir = "./OUTPUT"
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
for key in param_names:
    sim_params[key] = globals()[key]

#pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))
#call(['python', 'SS.py'])
income_tax_params, wealth_tax_params, ellipse_params, ss_parameters, iterative_params = SS.create_steady_state_parameters(**sim_params)


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
TPI.run_time_path_iteration(**ss_outputs)

'''
------------------------------------------------------------------------
    Alter desired tax parameters
------------------------------------------------------------------------
'''

# New Tax Parameters

# get_baseline = False
# d_tax_income = .42


# var_names = ['get_baseline', 'd_tax_income']
# dictionary = {}
# for key in var_names:
#     dictionary[key] = globals()[key]
# pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_changed.pkl", "w"))

'''
------------------------------------------------------------------------
    Run TPI for tax experiment
------------------------------------------------------------------------
'''

# call(['python', 'TPI.py'])
# import TPI
