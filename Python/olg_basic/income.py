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
    Read Data for e
------------------------------------------------------------------------
The data comes from the Consumer Population Survey.  The variables used
are age (PRTAGE) and hourly wage (PTERNHLY).  Because the sample size
for each month is small, we used data from May 2014 and September 2013
(the samples must be at least 8 months apart so that observations do not
appear in both samples).  
------------------------------------------------------------------------
'''

data = pd.read_table("data/income_data.asc", sep=',', header=0)
data = data[(19 < data.PRTAGE) & (data.PRTAGE < 80)]
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
        incomes = data[(age_groups[i] <= data.age) & (
            data.age < age_groups[i+1])]
        inc = np.array(incomes['wage'])
        inc.sort()
        for j in xrange(J):
            e[i, j] = inc[len(inc)*(j+.5)/J]
    e /= e.mean()
    return e



'''
------------------------------------------------------------------------
    Read Data for Markov Matrix
------------------------------------------------------------------------
The data comes from the Panel Study of Income Dynamics (PSID).  The
variables come from samples taken in 1999 and 2001.  The variables used
are the age of the individual and the hourly wage of the individual.
------------------------------------------------------------------------
'''

markov_dta = pd.read_csv("data/PSIDdata.csv", sep=',', header=0)
markov_dta['age_99'], markov_dta['wage_99'] = markov_dta[
    'ER33504'], markov_dta['ER33537O']
markov_dta['age_01'], markov_dta['wage_01'] = markov_dta[
    'ER33604'], markov_dta['ER33628O']
del markov_dta['ER33504'], markov_dta['ER33537O'], markov_dta[
    'ER33604'], markov_dta['ER33628O'], markov_dta['ER30001'], markov_dta[
    'ER30002'], markov_dta['ER33501'], markov_dta['ER33502'], markov_dta[
    'ER33503'], markov_dta['ER33601'], markov_dta['ER33602'], markov_dta[
    'ER33603']
markov_dta = markov_dta[(19 < markov_dta.age_99) & (
    markov_dta.age_99 < 80) & (19 < markov_dta.age_01) & (
    markov_dta.age_01 < 80)]
markov_dta = markov_dta[(markov_dta.wage_99 != -99) & (
    markov_dta.wage_99 != 0) & (markov_dta.wage_99 != 999)]
markov_dta = markov_dta[(markov_dta.wage_01 != -99) & (
    markov_dta.wage_01 != 0) & (markov_dta.wage_01 != 999)]
markov_dta = markov_dta.reset_index()
del markov_dta['index']


def get_f_equalprobabilities(S, J):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age

    Returns:    f - S x J matrix of probabilities for each ability level
                    by age group
    '''
    f = np.ones((J, J))*(1.0/J)
    f = np.tile(f.reshape(1, J, J), (S, 1, 1))
    return f


def get_f_noswitch(S, J):
    f = np.identity(J)
    f = np.tile(f.reshape(1, J, J), (S, 1, 1))
    return f


def get_f_markov(S, J):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age

    Returns:    f - S x J x J matrix of probabilities for each ability level
                    by age group

    Note:       Because the data is separated by 2 years, but the Markov
                    matrix is for a transition to the next year, we find
                    the "square root" of f.
    '''
    f = np.zeros((J, J))
    sort99 = np.array(markov_dta['wage_99'])
    sort01 = np.array(markov_dta['wage_01'])
    sort99.sort()
    sort01.sort()
    percentiles99 = np.zeros(J-1)
    percentiles01 = np.zeros(J-1)
    # Generate J percentile groups for ability
    for j in xrange(J-1):
        percentiles99[j] = sort99[len(sort99)*(j+1)/J]
        percentiles01[j] = sort01[len(sort01)*(j+1)/J]
    for ind99 in xrange(J):
        # Count the number of individuals in each percentile group in 1999
        if ind99 == 0:
            num_in_99df = markov_dta[
                markov_dta.wage_99 <= percentiles99[ind99]]
            num_in_99 = np.array(num_in_99df.count()[0])
        elif ind99 < J-1:
            num_in_99df = markov_dta[
                (percentiles99[ind99-1] < markov_dta.wage_99) & (
                    markov_dta.wage_99 <= percentiles99[ind99])]
            num_in_99 = np.array(num_in_99df.count()[0])
        else:
            num_in_99df = markov_dta[percentiles99[
                ind99-1] < markov_dta.wage_99]
            num_in_99 = np.array(num_in_99df.count()[0])
        for ind01 in xrange(J):
            # Count the number of individuals that were in a certain
                # percentile group in 1999 who remain in that group
            if ind01 == 0:
                num_in_both = num_in_99df[
                    num_in_99df.wage_01 <= percentiles01[ind01]]
                num_in_both = np.array(num_in_both.count()[0])
            elif ind01 < J-1:
                num_in_both = num_in_99df[(
                    percentiles01[ind01-1] < num_in_99df.wage_01) & (
                    num_in_99df.wage_01 <= percentiles01[ind01])]
                num_in_both = np.array(num_in_both.count()[0])
            else:
                num_in_both = num_in_99df[
                    percentiles01[ind01-1] < num_in_99df.wage_01]
                num_in_both = np.array(num_in_both.count()[0])
            # Each entry of the Markov matrix is the number of individuals who
                # were in a percentile group and stayed in that percentile
                # group, divided by the number of individuals originally in
                # that percentile group
            f[ind99, ind01] = float(num_in_both) / num_in_99
    eigvals, eigvecs = np.linalg.eig(f)
    Diagonal = np.diag(eigvals)
    eigvecs_inv = np.linalg.inv(eigvecs)
    Diagonal_squareroot = np.sqrt(Diagonal)
    # Generate the value of f for a one year interval, not 2
    f_root = eigvecs.dot(Diagonal_squareroot).dot(eigvecs_inv)
    f_root = np.real(f_root)
    # Multiply f by itself 60/S times, where 60/S is the number of ages
        # in each age group
    f_root = np.linalg.matrix_power(f_root, 60/S)
    f_root = np.tile(f_root.reshape(1, J, J), (S, 1, 1))
    return f_root
