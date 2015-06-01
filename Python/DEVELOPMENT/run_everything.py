'''
------------------------------------------------------------------------
Last updated 5/21/2015

This will run the entire wealth tax project

This py-file calls the following other file(s):
            run_baseline.py
            run_tax_experiments.py
------------------------------------------------------------------------
'''

from subprocess import call

call(['python', 'run_baseline.py'])

# Note: Unless nothing in run_baseline.py has been changed, the wealth tax will
# need to be re-calibrated (see note in run_tax_experiments.py) before
# running run_tax_experiments.py, or the wealth tax parameters will be wrong

call(['python', 'run_tax_experiments.py'])
