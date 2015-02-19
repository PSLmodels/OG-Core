'''
------------------------------------------------------------------------
Last updated 2/13/2015

This will run the steady state solver as well as time path iteration,
given that these have already run with run_model.py, with new tax
policies.
------------------------------------------------------------------------
'''

'''
Import Packages
'''

import numpy as np
import pickle
from glob import glob
import os
import sys
import scipy.optimize as opt
import shutil

# Import Parameters from initial simulations
variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

# New Tax Parameters
p_wealth = 0.025
SS_initial_run = False

print 'Getting SS distribution for wealth tax.'
var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'chi_n', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'slow_work', 'chi_n_multiplier', 'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth', 'SS_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))


'''
Run steady state solver and TPI (according to given variables) for wealth tax
'''
import SS
del sys.modules['tax_funcs']
del sys.modules['demographics']
del sys.modules['SS']


TPI_initial_run = False
var_names = ['TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/tpi_var.pkl", "w"))

import TPI
del sys.modules['TPI']
del sys.modules['tax_funcs']


shutil.rmtree('OUTPUT_wealth_tax')
shutil.copytree('OUTPUT', 'OUTPUT_wealth_tax')

'''
Run Steady State Solver and TPI for wealth tax
'''
p_wealth = 0.0

var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'chi_n', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'slow_work', 'chi_n_multiplier', 'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth', 'SS_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))

lump_to_match = pickle.load(open("OUTPUT/SS/Tss_var.pkl", "r"))


def matcher(d_inc_guess):
    pickle.dump(d_inc_guess, open("OUTPUT/SS/d_inc_guess.pkl", "w"))
    import SS
    del sys.modules['tax_funcs']
    del sys.modules['demographics']
    del sys.modules['SS']
    lump_new = pickle.load(open("OUTPUT/SS/Tss_var.pkl", "r"))
    error = abs(lump_to_match - lump_new)
    # print error
    return error

print 'Computing new income tax to match wealth tax'
new_d_inc = opt.fsolve(matcher, d_tax_income, xtol=1e-13)
print '\tOld income tax:', d_tax_income
print '\tNew income tax:', new_d_inc

os.remove("OUTPUT/SS/d_inc_guess.pkl")
os.remove("OUTPUT/SS/Tss_var.pkl")

d_tax_income = new_d_inc
var_names = ['S', 'J', 'T', 'bin_weights', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'A', 'delta', 'ctilde', 'E',
             'bqtilde', 'ltilde', 'chi_n', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'slow_work', 'chi_n_multiplier', 'a_tax_income',
             'b_tax_income', 'c_tax_income', 'd_tax_income', 'tau_sales',
             'tau_payroll', 'tau_bq', 'tau_lump',
             'theta_tax', 'retire', 'mean_income',
             'h_wealth', 'p_wealth', 'm_wealth', 'SS_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))
print 'Getting SS distribution for income tax.'
import SS
del sys.modules['tax_funcs']
del sys.modules['demographics']
del sys.modules['SS']

import TPI
del sys.modules['tax_funcs']

shutil.rmtree('OUTPUT_income_tax')
shutil.copytree('OUTPUT', 'OUTPUT_income_tax')

files = glob('*.pyc')
for i in files:
    os.remove(i)
