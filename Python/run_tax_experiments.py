'''
------------------------------------------------------------------------
Last updated 5/21/2015

This will run the steady state solver as well as time path iteration,
given that these have already run with run_model.py, with new tax
policies.

This py-file calls the following other file(s):
            OUTPUT/given_params.pkl

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/given_params.pkl
            OUTPUT/Saved_moments/tpi_var.pkl
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Import Packages
------------------------------------------------------------------------
'''

import numpy as np
import pickle
from glob import glob
import os
import sys
import scipy.optimize as opt
import shutil
from subprocess import call

'''
------------------------------------------------------------------------
    Import parameters from baseline, and alter desired tax parameters
------------------------------------------------------------------------
'''

# Import Parameters from initial simulations
variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

# New Tax Parameters
p_wealth = 0.025
h_wealth = 0.305509008443123
m_wealth = 2.16050687852062

scal = np.ones(J) * 1.1
scal[-1] = .5
scal[-2] = .7
chi_b_scal = np.zeros(J)
d_tax_income = .219


'''
------------------------------------------------------------------------
    Run SS with tax experiment
------------------------------------------------------------------------
'''

SS_stage = 'SS_tax'

print 'Getting SS distribution for wealth tax.'
var_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
             'ltilde', 'g_y', 'TPImaxiter',
             'TPImindist', 'b_ellipse', 'k_ellipse', 'upsilon',
             'a_tax_income', 'scal',
             'b_tax_income', 'c_tax_income', 'd_tax_income',
             'tau_payroll', 'tau_bq',
             'theta', 'retire', 'mean_income_data',
             'h_wealth', 'p_wealth', 'm_wealth', 'chi_b_scal', 'SS_stage']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/given_params.pkl", "w"))

call(['python', 'SS.py'])

'''
------------------------------------------------------------------------
    Run TPI for tax experiment
------------------------------------------------------------------------
'''

TPI_initial_run = False
var_names = ['TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/tpi_var.pkl", "w"))

call(['python', 'TPI.py'])


'''
------------------------------------------------------------------------
Delete all .pyc files that have been generated
------------------------------------------------------------------------
'''

files = glob('*.pyc')
for i in files:
    os.remove(i)
