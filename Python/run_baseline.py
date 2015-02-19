'''
------------------------------------------------------------------------
Last updated 2/13/2015

This will run the steady state solver as well as time path iteration.

This py-file calls the following other file(s):
            wealth_data.py
            SS_old_income.py
            SS_init_init.py
            payroll.py
            SS.py
            TPI.py

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/given_params.pkl
            OUTPUT/Nothing/tpi_var.pkl
------------------------------------------------------------------------
'''

'''
Import Packages
'''

import numpy as np
import pickle
import os
from glob import glob
import sys
import wealth_data


'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
bin_weights  = desires percentiles of ability groups
bin_weights_init = simple bin_weights (all equal sizes) to get initial
                   guesses
bin_weights_array = array of bin weights, each one being used as initial
                    values for the next
which_iterations = array of strings that label the bin weights in
                   bin_weights_array, to be used for saving files
init_vals_scalars = scalar multipliers used in SS files to make the
                    initial values work
start_point_iter = at what element of bin_weights_array to start
                   SS_init_init
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta_annual  = discount factor for one year
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
A            = total factor productivity parameter in firms' production
               function
delta_annual = depreciation rate of capital for one year
delta        = depreciation rate of capital for each cohort
ltilde       = measure of time each individual is endowed with each
               period
g_y_annual   = annual growth rate of technology
g_y          = growth rate of technology for one cohort
ctilde       = minimum value amount of consumption
bqtilde      = minimum bequest value
chi_n        = discount factor of labor that changes with S (Sx1 array)
slow_work    = time at which chi_n starts increasing from 1
chi_n_multiplier = scalar which is increased to force the labor
               distribution to 0
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
nu           = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_nu
b_ellipse    = value of b for elliptical fit of utility function
k_ellipse    = value of k for elliptical fit of utility function
upsilon= value of omega for elliptical fit of utility function
mean_income  = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income = used to calibrate income tax (scalar)
b_tax_income = used to calibrate income tax (scalar)
c_tax_income = used to calibrate income tax (scalar)
d_tax_income = used to calibrate income tax (scalar)
retire       = age in which individuals retire(scalar)
h_wealth     = wealth tax parameter h
m_wealth     = wealth tax parameter m
p_wealth     = wealth tax parameter p
tau_sales    = sales tax (scalar)
tau_bq       = bequest tax (scalar)
tau_lump     = lump sum tax (scalar)
tau_payroll  = payroll tax (scalar)
theta_tax    = payback value for payroll tax (scalar)
bin_weights  = percent of each age cohort in each ability group
------------------------------------------------------------------------
'''
# Parameters
S = 80
J = 7
T = int(2 * S)
bin_weights = np.array([.25, .25, .2, .1, .1, .09, .01])
bin_weights_init = np.array([1.0/J] * J)
bin_weights_array = np.array([[.25, .25, .1, .1, .1, .1, .1], [
    .25, .25, .2, .1, .1, .05, .05], [.25, .25, .2, .1, .1, .06, .04], [
    .25, .25, .2, .1, .1, .07, .03], [.25, .25, .2, .1, .1, .08, .02]])
which_iterations = np.array([
    'tens', 'fives', 'six_four', 'seven_three', 'eight_two'])
init_vals_scalars = np.array([.5, .5, .5, .5, .7, .7])
start_point_iter = 6
starting_age = 20
ending_age = 100
E = int(starting_age * (S / float(ending_age-starting_age)))
beta_annual = .96
beta = beta_annual ** (float(ending_age-starting_age) / S)
sigma = 3.0
alpha = .35
A = 1.0
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (float(ending_age-starting_age) / S))
ltilde = 1.0
g_y_annual = 0.03
g_y = (1 + g_y_annual)**(float(ending_age-starting_age)/S) - 1
# Constraint parameters
ctilde = .000001
bqtilde = .000001
# Chi_n callibration parameters
# chi_n = np.ones(S) * 1.0
# slow_work = np.round(7.0 * S / 16.0)
# chi_n_multiplier = 30
# TPI parameters
TPImaxiter = 100
TPImindist = 3 * 1e-6
nu = .20
# Ellipse parameters
b_ellipse = 25.6594
k_ellipse = -26.4902
upsilon = 3.0542
# Tax parameters:
mean_income = 84377.0
a_tax_income = 3.03452713268985e-06
b_tax_income = .222
c_tax_income = 133261.0
d_tax_income = .219
retire = np.round(9.0 * S / 16.0) - 1
# Wealth tax params
h_wealth = 1.06719316335025
m_wealth = 1.55165028571888
p_wealth = 0.025
# Tax parameters that are zeroed out for SS_init_init
# Initial taxes below
d_tax_income = 0.0
tau_sales = 0.0
tau_bq = 0.0
tau_lump = 0.0
tau_payroll = 0.0
theta_tax = 0.0
p_wealth = 0.0

print 'Getting initial SS distribution, without calibrating bequests, to speed up SS.'
print '\tAttempt using old income distribution.'
var_names = ['S', 'J', 'T', 'bin_weights_init', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth']
dictionary = {}
if os.path.isfile("/OUTPUT/given_params.pkl"):
    os.remove("OUTPUT/given_params.pkl")
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))
import SS_old_income
del sys.modules['tax_funcs']
del sys.modules['income_old']
del sys.modules['demographics']
del sys.modules['SS_old_income']

# Tax Parameters Initially
d_tax_income = .219
# tau_sales = 0.02
tau_bq = np.zeros(J)
tau_wealth = np.zeros(J)
tau_lump = 0.0
tau_payroll = 0.15
theta_tax = np.zeros(J)


print 'Getting initial SS distribution, calibrating bequests, to speed up SS.'
thetas_simulation = False
for i in xrange(start_point_iter, len(bin_weights_array)):
    print '\tAttempt', i+1
    bin_weights_init = bin_weights_array[i, :]
    scal = init_vals_scalars[i]
    if i == 0:
        name_of_last = 'init_init_sols'
    else:
        name_of_last = which_iterations[i-1]
    name_of_it = which_iterations[i]
    var_names = ['S', 'J', 'T', 'bin_weights_init', 'starting_age', 'ending_age',
                 'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
                 'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
                 'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
                 'a_tax_income',
                 'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
                 'tau_payroll', 'tau_bq', 'tau_lump',
                 'theta_tax', 'retire', 'mean_income',
                 'h_wealth', 'p_wealth', 'm_wealth', 'scal', 'name_of_it',
                 'name_of_last', 'thetas_simulation']
    dictionary = {}
    os.remove("OUTPUT/given_params.pkl")
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))
    import SS_init_init
    del sys.modules['tax_funcs']
    del sys.modules['demographics']
    del sys.modules['income']
    del sys.modules['SS_init_init']
print '\tFinished'

# #Tax Parameters Initially
# d_tax_income = .225298
# # tau_sales = 0.02
# tau_bq = np.zeros(J)
# tau_wealth = np.zeros(J)
# tau_lump = 0.0
# tau_payroll = 0.15
# theta_tax = np.zeros(J)


scal = init_vals_scalars[-1]
name_of_last = which_iterations[-2]
bin_weights_init = bin_weights
# This is the last simulation, using the desired bin weights, to get the replacement rate values
thetas_simulation = True
name_of_it = which_iterations[-1]
var_names = ['S', 'J', 'T', 'bin_weights_init', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth', 'scal', 'name_of_last',
             'thetas_simulation', 'name_of_it']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))

print 'Getting Thetas'
import SS_init_init
del sys.modules['tax_funcs']
del sys.modules['demographics']
del sys.modules['income']
del sys.modules['SS_init_init']

import payroll
theta_tax_orig = payroll.vals()
del sys.modules['payroll']
print '\tFinished.'

# Run SS with replacement rates, and baseline taxes
SS_initial_run = True
print 'Getting initial distribution.'

# Due to difficulty in finding the Steady State, the replacement rates must be added in slowly
theta_tax_tenth = theta_tax_orig / 15.0

for i in xrange(15):
    theta_tax = theta_tax_tenth * (i+1)
    var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
                 'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
                 'bqtilde', 'ltilde', 'g_y', 'TPImaxiter',
                 'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
                 'a_tax_income',
                 'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
                 'tau_payroll', 'tau_bq', 'tau_lump',
                 'theta_tax', 'retire', 'mean_income',
                 'h_wealth', 'p_wealth', 'm_wealth', 'SS_initial_run']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))

    print 'Rep rate ', i+1
    import SS
    del sys.modules['tax_funcs']
    del sys.modules['demographics']
    del sys.modules['SS']

print '\tFinished'

# Run the baseline TPI simulation
TPI_initial_run = True
var_names = ['TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/tpi_var.pkl", "w"))
import TPI
del sys.modules['tax_funcs']
del sys.modules['TPI']



'''
Delete all .pyc files that have been generated
'''

files = glob('*.pyc')
for i in files:
    os.remove(i)
