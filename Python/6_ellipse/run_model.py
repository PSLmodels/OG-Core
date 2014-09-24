'''
------------------------------------------------------------------------
Last updated 9/24/2014

This will run the steady state solver as well as time path iteration.
------------------------------------------------------------------------
'''

'''
Import Packages
'''

import numpy as np
import pickle
import os
from glob import glob

# Run steady state
run_SS = True
# Run TPI
run_TPI = True

'''
Delete all .png and .pkl files that have been generated
'''

files = glob('OUTPUT/*.png')
for i in files:
    os.remove(i)
files = glob('OUTPUT/*.pkl')
for i in files:
    os.remove(i)

'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
bin_weights  = percent of each age cohort in each ability group
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta_annual  = discount factor for one year
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
nu_init      = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_new
A            = total factor productivity parameter in firms' production
               function
delta_annual = depreciation rate of capital for one year
delta        = depreciation rate of capital for each cohort
ctilde       = minimum value amount of consumption
bqtilde      = minimum bequest value
ltilde       = measure of time each individual is endowed with each
               period
chi_n        = discount factor of labor that changes with S (Sx1 array)
chi_b        = discount factor of incidental bequests
g_y_annual   = annual growth rate of technology
g_y          = growth rate of technology for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
b_ellipse    = value of b for elliptical fit of utility function
k_ellipse    = value of k for elliptical fit of utility function
omega_ellipse= value of omega for elliptical fit of utility function
slow_work    = time at which chi_n starts increasing from 1
chi_n_multiplier = scalar which is increased to force the labor
               distribution to 0
------------------------------------------------------------------------
'''


# Parameters
S = 80
J = 7
T = int(1.5 * S)
bin_weights = np.array([1.0/J] * J)
starting_age = 20
ending_age = 100
E = int(starting_age * (S / float(ending_age-starting_age)))
beta_annual = .96
beta = beta_annual ** (float(ending_age-starting_age) / S)
sigma = 3.0
alpha = .35
nu = .20
A = 1.0
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (float(ending_age-starting_age) / S))
ctilde = .000001
bqtilde = .000001
ltilde = 1.0
chi_n = np.ones(S) * 1.0
chi_b = 1.0
g_y_annual = 0.03
g_y = (1 + g_y_annual)**(float(ending_age-starting_age)/S) - 1
TPImaxiter = 100
TPImindist = 3 * 1e-6
slow_work = np.round(7.0 * S / 16.0)
chi_n_multiplier = 30
b_ellipse = 25.0
k_ellipse = -25.0
omega_ellipse = 3.0

'''
Pickle parameter values
'''

print 'Saving user given parameter values.'
var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'chi_n', 'chi_b', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'omega_ellipse',
             'slow_work', 'chi_n_multiplier']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))
print '\tFinished.'


'''
Run steady state solver and TPI (according to run_SS and run_TPI
    variables)
'''

if run_SS:
    import SS

if run_TPI:
    import TPI

'''
See the pickles for results:
    OUTPUT\SS_vars.pkl
    OUTPUT\TPI_vars.pkl
'''

'''
Delete all .pyc files that have been generated
'''

files = glob('*.pyc')
for i in files:
    os.remove(i)
