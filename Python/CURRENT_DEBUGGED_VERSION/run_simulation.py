'''
------------------------------------------------------------------------
Last updated 7/16/2015

This will run the steady state solver as well as time path iteration.
First this file generates demographic and ability variables, then it saves
the needed labor and wealth data moments, and then it saves all the
baseline parameters.  After this, it runs the SS and TPI simulations.
If tax experiments are desired, the changed parameters are saved,
and then SS and TPI simulations are repeated.  Finally, all .pyc files
are deleted.

This py-file calls the following other file(s):
            income_polynomials.py
            demographics.py
            wealth_data.py
            labor_data.py
            SS.py
            TPI.py

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
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

import numpy as np

import income_polynomials as income
import demographics
import wealth_data
import labor_data


'''
------------------------------------------------------------------------
Parameters
------------------------------------------------------------------------
Model Parameters:
------------------------------------------------------------------------
S            = number of periods an individual lives (scalar)
J            = number of different ability groups (scalar)
T            = number of time periods until steady state is reached (scalar)
lambdas      = percentiles for ability groups (Jx1 array)
starting_age = age of first members of cohort (scalar)
ending age   = age of the last members of cohort (scalar)
E            = number of cohorts before S=1 (scalar)
beta_annual  = discount factor for one year (scalar)
beta         = discount factor for each age cohort (scalar)
sigma        = coefficient of relative risk aversion (scalar)
alpha        = capital share of income (scalar)
Z            = total factor productivity parameter in firms' production
               function (scalar)
delta_annual = depreciation rate of capital for one year (scalar)
delta        = depreciation rate of capital for each cohort (scalar)
ltilde       = measure of time each individual is endowed with each
               period (scalar)
g_y_annual   = annual growth rate of technology (scalar)
g_y          = growth rate of technology for one cohort (scalar)
b_ellipse    = value of b for elliptical fit of utility function (scalar)
k_ellipse    = value of k for elliptical fit of utility function (scalar)
upsilon      = value of omega for elliptical fit of utility function (scalar)
------------------------------------------------------------------------
Tax Parameters:
------------------------------------------------------------------------
mean_income_data  = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income = used to calibrate income tax (scalar)
b_tax_income = used to calibrate income tax (scalar)
c_tax_income = used to calibrate income tax (scalar)
d_tax_income = used to calibrate income tax (scalar)
h_wealth     = wealth tax parameter h (scalar)
m_wealth     = wealth tax parameter m (scalar)
p_wealth     = wealth tax parameter p (scalar)
tau_bq       = bequest tax (Jx1 array)
tau_payroll  = payroll tax (scalar)
retire       = age at which individuals retire (scalar)
------------------------------------------------------------------------
Simulation Parameters:
------------------------------------------------------------------------
maxiter      = Maximum number of iterations that SS and TPI will undergo (scalar)
mindist_SS   = Cut-off distance between iterations for SS (scalar)
mindist_TPI  = Cut-off distance between iterations for TPI (scalar)
nu           = contraction parameter in SS and TPI iteration process
               representing the weight on the new distribution (scalar)
flag_graphs  = Flag to prevent graphing from occuring in demographic, income,
               wealth, and labor files (True=graph) (bool)
get_baseline = Flag to run baseline or tax experiments (bool)
calibrate_model = Flag to run calibration of chi values or not (bool)
chi_b_guess  = Chi^b_j initial guess for model (Jx1 array)
               (if no calibration occurs, these are the values that will be used for chi^b_j)
chi_n_guess  = Chi^n_s initial guess for model (Sx1 array)
               (if no calibration occurs, these are the values that will be used for chi^n_s)
------------------------------------------------------------------------
Demographics and Ability variables:
------------------------------------------------------------------------
omega        =  Time path of of population size for each age across T ((T+S)xS array)
g_n_ss       = steady state population growth rate (scalar)
omega_SS     = stationarized steady state population distribution (Sx1 array)
surv_rate    = survival rates (Sx1 array)
rho          = mortality rates (Sx1 array)
g_n_vector   = population size for each T ((T+S)x1 array)
e            = age dependent possible working abilities (SxJ array)
------------------------------------------------------------------------
'''

# Model Parameters
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
#   Ellipse parameters
b_ellipse = 25.6594
k_ellipse = -26.4902
upsilon = 3.0542

# Tax parameters:
#   Income Tax Parameters
mean_income_data = 84377.0
a_tax_income = 3.03452713268985e-06
b_tax_income = .222
c_tax_income = 133261.0
d_tax_income = .219
#   Wealth tax params
#       These are non-calibrated values, h and m just need
#       need to be nonzero to avoid errors. When p_wealth
#       is zero, there is no wealth tax.
h_wealth = 0.1
m_wealth = 1.0
p_wealth = 0.0
#   Bequest and Payroll Taxes
tau_bq = np.zeros(J)
tau_payroll = 0.15
retire = np.round(9.0 * S / 16.0) - 1

# Simulation Parameters
maxiter = 250
mindist_SS = 1e-9
mindist_TPI = 1e-6
nu = .4
flag_graphs = False
get_baseline = True
#   Calibration parameters
calibrate_model = False
# These guesses are close to the calibrated values
chi_b_guess = np.array([2, 10, 90, 350, 1700, 22000, 120000])
chi_n_guess = np.array([47.12000874 , 22.22762421 , 14.34842241 , 10.67954008 ,  8.41097278
                         ,  7.15059004 ,  6.46771332 ,  5.85495452 ,  5.46242013 ,  5.00364263
                         ,  4.57322063 ,  4.53371545 ,  4.29828515 ,  4.10144524 ,  3.8617942  ,  3.57282
                         ,  3.47473172 ,  3.31111347 ,  3.04137299 ,  2.92616951 ,  2.58517969
                         ,  2.48761429 ,  2.21744847 ,  1.9577682  ,  1.66931057 ,  1.6878927
                         ,  1.63107201 ,  1.63390543 ,  1.5901486  ,  1.58143606 ,  1.58005578
                         ,  1.59073213 ,  1.60190899 ,  1.60001831 ,  1.67763741 ,  1.70451784
                         ,  1.85430468 ,  1.97291208 ,  1.97017228 ,  2.25518398 ,  2.43969757
                         ,  3.21870602 ,  4.18334822 ,  4.97772026 ,  6.37663164 ,  8.65075992
                         ,  9.46944758 , 10.51634777 , 12.13353793 , 11.89186997 , 12.07083882
                         , 13.2992811  , 14.07987878 , 14.19951571 , 14.97943562 , 16.05601334
                         , 16.42979341 , 16.91576867 , 17.62775142 , 18.4885405  , 19.10609921
                         , 20.03988031 , 20.86564363 , 21.73645892 , 22.6208256  , 23.37786072
                         , 24.38166073 , 25.22395387 , 26.21419653 , 27.05246704 , 27.86896121
                         , 28.90029708 , 29.83586775 , 30.87563699 , 31.91207845 , 33.07449767
                         , 34.27919965 , 35.57195873 , 36.95045988 , 38.62308152])

# Demographic and Ability variables
omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = demographics.get_omega(
    S, T, starting_age, ending_age, E, flag_graphs)
e = income.get_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)


# List of parameter names that will not be changing (unless we decide to
# change them for a tax experiment)
param_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
             'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
             'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income', 'chi_b_guess', 'chi_n_guess',
             'b_tax_income', 'c_tax_income', 'd_tax_income',
             'tau_payroll', 'tau_bq', 'calibrate_model',
             'retire', 'mean_income_data', 'g_n_vector',
             'h_wealth', 'p_wealth', 'm_wealth', 'get_baseline',
             'omega', 'g_n_ss', 'omega_SS', 'surv_rate', 'e', 'rho']

'''
------------------------------------------------------------------------
    Obtain wealth and labor distributions from data
------------------------------------------------------------------------
'''

# Generate Wealth data moments
wealth_data.get_wealth_data(lambdas, J, flag_graphs)
# Generate labor data moments
labor_data.labor_data_moments(flag_graphs)

'''
------------------------------------------------------------------------
    Run baseline SS
------------------------------------------------------------------------
'''

dictionary = {}
for key in param_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))

import SS

'''
------------------------------------------------------------------------
    Run baseline TPI
------------------------------------------------------------------------
'''

# import TPI

'''
------------------------------------------------------------------------
    Alter desired tax parameters
------------------------------------------------------------------------
'''

get_baseline = False
dictionary = {}
for key in param_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_given.pkl", "w"))

# Altered parameters
d_tax_income = .42

# List of all parameters that have been changed for the tax experiment
params_changed_names = ['d_tax_income']
dictionary = {}
for key in params_changed_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_changed.pkl", "w"))

'''
------------------------------------------------------------------------
    Run SS with tax experiment
------------------------------------------------------------------------
'''

# import SS

'''
------------------------------------------------------------------------
    Run TPI for tax experiment
------------------------------------------------------------------------
'''

# import TPI


'''
------------------------------------------------------------------------
Delete all .pyc files that have been generated
------------------------------------------------------------------------
'''

files = glob('*.pyc')
for i in files:
    os.remove(i)
