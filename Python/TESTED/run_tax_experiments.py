'''
------------------------------------------------------------------------
Last updated 6/4/2015

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
import cPickle as pickle
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

SS_stage = 'SS_tax'
TPI_initial_run = False

d_tax_income = .9 * .219

var_names = ['SS_stage', 'TPI_initial_run', 'd_tax_income']
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
