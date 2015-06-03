'''
------------------------------------------------------------------------
Last updated 5/21/2015

This will run the steady state solver as well as time path iteration,
given that these have already run with run_model.py, with new tax
policies.

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Saved_moments/params_changed.pkl
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

# New Tax Parameters
p_wealth = 0.025
h_wealth = 0.305509008443123
m_wealth = 2.16050687852062

scal = np.ones(J) * 1.1
scal[-1] = .5
scal[-2] = .7
chi_b_scal = np.zeros(J)
d_tax_income = .219

SS_stage = 'SS_tax'
TPI_initial_run = False

var_names = ['p_wealth', 'h_wealth', 'm_wealth', 'scal', 'chi_b_scal',
             'd_tax_income', 'TPI_initial_run']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/params_changed.pkl", "w"))

'''
------------------------------------------------------------------------
    Run SS with tax experiment
------------------------------------------------------------------------
'''


print 'Getting SS distribution for wealth tax.'
call(['python', 'SS.py'])

'''
------------------------------------------------------------------------
    Run TPI for tax experiment
------------------------------------------------------------------------
'''

call(['python', 'TPI.py'])


'''
------------------------------------------------------------------------
Delete all .pyc files that have been generated
------------------------------------------------------------------------
'''

files = glob('*.pyc')
for i in files:
    os.remove(i)
