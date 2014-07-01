'''
------------------------------------------------------------------------
Last updated 7/1/2014

Functions for created the matrix of ability levels, e, and the 
probabilities, f, to be used in OLG_fastversion.py

This py-file calls the following other file(s):
            data/income_data.asc
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd

'''
------------------------------------------------------------------------
    Read Data
------------------------------------------------------------------------
The data comes from **************
The data has the age and hourly wage for each observation
------------------------------------------------------------------------
'''

data = pd.read_table("data/income_data.asc", sep=',', header=0)
data = data.query("19 < PRTAGE < 80")
data['age'], data['wage'] = data['PRTAGE'], data['PTERNHLY']
del data['HRHHID'], data['OCCURNUM'], data['YYYYMM'], data[
    'HRHHID2'], data['PRTAGE'], data['PTERNHLY']


def get_e(S, J):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age

    Returns:    e - S x J matrix of J working ability levels for each 
                    age cohort measured by hourly wage, normalized so 
                    the mean is one
    '''
    age_groups = np.linspace(20, 80, S+1)
    e = np.zeros((S, J))
    for i in xrange(S):
        incomes = data.query('age_groups[i]<=age<age_groups[i+1]')
        inc = np.array(incomes['wage'])
        inc.sort()
        for j in xrange(J):
            e[i, j] = inc[len(inc)*(j+.5)/J]
    e /= e.mean()
    return e


def get_f(S, J):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age

    Returns:    f - S x J matrix of probabilities for each ability level
                    by age group
    '''
    f = np.ones((S, J))*(1.0/J)
    return f
