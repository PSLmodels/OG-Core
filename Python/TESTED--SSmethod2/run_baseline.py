'''
------------------------------------------------------------------------
Last updated 6/4/2015

This will run the steady state solver as well as time path iteration.

This py-file calls the following other file(s):
            wealth_data.py
            labor_data.py
            SS.py
            payroll.py
            TPI.py

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Saved_moments/params_given.pkl
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
import sys
from subprocess import call
import time

import numpy as np

import wealth_data
import labor_data


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
mindist   = Cut-off distance between iterations for TPI
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
------------------------------------------------------------------------
'''

# Parameters
S = 80
J = 7
T = int(2 * S)
lambdas = np.array([.25, .25, .2, .1, .1, .09, .01])
starting_age = 20
ending_age = 100
E = int(starting_age * (S / float(ending_age-starting_age)))
beta_annual = .96
beta = beta_annual ** (float(ending_age-starting_age) / S)
sigma = 3.0
alpha = .35
Z = 1.0
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (float(ending_age-starting_age) / S))
ltilde = 1.0
g_y_annual = 0.03
g_y = (1 + g_y_annual)**(float(ending_age-starting_age)/S) - 1
# TPI parameters
maxiter = 150
mindist = 3 * 1e-6
nu = .40
TPI_initial_run = True
# Ellipse parameters
b_ellipse = 25.6594
k_ellipse = -26.4902
upsilon = 3.0542
# Tax parameters:
mean_income_data = 84377.0
a_tax_income = 3.03452713268985e-06
b_tax_income = .222
c_tax_income = 133261.0
d_tax_income = .219
retire = np.round(9.0 * S / 16.0) - 1
# Wealth tax params
# These won't be used for the wealth tax, h and m just need
# need to be nonzero to avoid errors
h_wealth = 0.1
m_wealth = 1.0
p_wealth = 0.0
# Tax parameters that are zeroed out for SS
# Initial taxes below
tau_bq = np.zeros(J)
tau_payroll = 0.15
theta = np.zeros(J)
# Other parameters
scal = np.ones(J)

# Generate Wealth data moments
wealth_data.get_wealth_data(lambdas, J)

# Remove pickle of altered parameters -- reset the experiment
if os.path.isfile("OUTPUT/Saved_moments/params_changed.pkl"):
    os.remove("OUTPUT/Saved_moments/params_changed.pkl")

'''
------------------------------------------------------------------------
    Run SS without calibration, to get initial values
------------------------------------------------------------------------
'''

start_time = time.time()

SS_stage = 'first_run_for_guesses'


print 'Getting initial SS distribution, not calibrating bequests, to speed up SS.'

var_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
             'ltilde', 'g_y', 'maxiter',
             'mindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income',
             'tau_payroll', 'tau_bq',
             'theta', 'retire', 'mean_income_data',
             'h_wealth', 'p_wealth', 'm_wealth', 'scal',
             'SS_stage', 'TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))
call(['python', 'SS.py'])

print '\tFinished'

initialrun_end = time.time()


'''
------------------------------------------------------------------------
    Run SS with minimization to fit chi_b and chi_n
------------------------------------------------------------------------
'''

os.remove("OUTPUT/Saved_moments/chi_b_fits.pkl")

# This is the simulation to get the replacement rate values

SS_stage = 'constrained_minimization'

thetas_simulation = True
var_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
             'ltilde', 'g_y', 'maxiter',
             'mindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income',
             'tau_payroll', 'tau_bq',
             'theta', 'retire', 'mean_income_data',
             'h_wealth', 'p_wealth', 'm_wealth', 'scal',
             'SS_stage', 'TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))

print 'Getting Thetas'
call(['python', 'SS.py'])

minend = time.time()
'''
------------------------------------------------------------------------
    Get replacement rates
------------------------------------------------------------------------
'''

import tax_funcs
theta = tax_funcs.replacement_rate_vals()
del sys.modules['tax_funcs']
print '\tFinished.'

'''
------------------------------------------------------------------------
    Run SS with replacement rates, and baseline taxes
------------------------------------------------------------------------
'''

print 'Getting initial distribution.'

SS_stage = 'SS_init'

var_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
             'ltilde', 'g_y', 'maxiter',
             'mindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income',
             'tau_payroll', 'tau_bq',
             'theta', 'retire', 'mean_income_data',
             'h_wealth', 'p_wealth', 'm_wealth', 'scal',
             'SS_stage', 'TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))
call(['python', 'SS.py'])
print '\tFinished'


'''
------------------------------------------------------------------------
    Run the baseline TPI simulation
------------------------------------------------------------------------
'''

# call(['python', 'TPI.py'])
import TPI

'''
------------------------------------------------------------------------
Delete all .pyc files that have been generated
------------------------------------------------------------------------
'''

files = glob('*.pyc')
for i in files:
    os.remove(i)

final_end = time.time()

print 'SS method 2 times:'
print 'intialtime = ', initialrun_end - start_time
print 'minimize time = ', minend - initialrun_end
print 'SSinit time = ', final_end - minend
print 'overall time = ', final_end - start_time
