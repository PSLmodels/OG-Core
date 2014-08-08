'''
------------------------------------------------------------------------
Last updated 7/11/2014

Functions for created the matrix of ability levels, e, and the
probabilities, f, to be used in OLG_fastversion.py

This py-file calls the following other file(s):
            data/e_vec_data/jan2014.asc
            data/e_vec_data/feb2014.asc
            data/e_vec_data/march2014.asc
            data/e_vec_data/april2014.asc
            data/e_vec_data/may2014.asc
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
------------------------------------------------------------------------
    Read Data for e
------------------------------------------------------------------------
The data comes from the Consumer Population Survey.  The variables used
are age (PRTAGE) and hourly wage (PTERNHLY).  Because the sample size
for each month is small, we used data from January, February, March,
April, and May 2014.  The matrix of ability levels was created for each
month, and then the average of the 5 matrices was taken for the final
matrix to return.
------------------------------------------------------------------------
'''

jan_dat = pd.read_table("data/e_vec_data/jan2014.asc", sep=',', header=0)
jan_dat['wgt'] = jan_dat['PWCMPWGT']
jan_dat['age'], jan_dat['wage'] = jan_dat['PRTAGE'], jan_dat['PTERNHLY']
del jan_dat['HRHHID'], jan_dat['OCCURNUM'], jan_dat['YYYYMM'], jan_dat[
    'HRHHID2'], jan_dat['PRTAGE'], jan_dat['PTERNHLY'], jan_dat['PWCMPWGT']

feb_dat = pd.read_table("data/e_vec_data/feb2014.asc", sep=',', header=0)
feb_dat['wgt'] = feb_dat['PWCMPWGT']
feb_dat['age'], feb_dat['wage'] = feb_dat['PRTAGE'], feb_dat['PTERNHLY']
del feb_dat['HRHHID'], feb_dat['OCCURNUM'], feb_dat['YYYYMM'], feb_dat[
    'HRHHID2'], feb_dat['PRTAGE'], feb_dat['PTERNHLY'], feb_dat['PWCMPWGT']

mar_dat = pd.read_table("data/e_vec_data/mar2014.asc", sep=',', header=0)
mar_dat['wgt'] = mar_dat['PWCMPWGT']
mar_dat['age'], mar_dat['wage'] = mar_dat['PRTAGE'], mar_dat['PTERNHLY']
del mar_dat['HRHHID'], mar_dat['OCCURNUM'], mar_dat['YYYYMM'], mar_dat[
    'HRHHID2'], mar_dat['PRTAGE'], mar_dat['PTERNHLY'], mar_dat['PWCMPWGT']

apr_dat = pd.read_table("data/e_vec_data/apr2014.asc", sep=',', header=0)
apr_dat['wgt'] = apr_dat['PWCMPWGT']
apr_dat['age'], apr_dat['wage'] = apr_dat['PRTAGE'], apr_dat['PTERNHLY']
del apr_dat['HRHHID'], apr_dat['OCCURNUM'], apr_dat['YYYYMM'], apr_dat[
    'HRHHID2'], apr_dat['PRTAGE'], apr_dat['PTERNHLY'], apr_dat['PWCMPWGT']

may_dat = pd.read_table("data/e_vec_data/may2014.asc", sep=',', header=0)
may_dat['age'], may_dat['wage'] = may_dat['PRTAGE'], may_dat['PTERNHLY']
may_dat['wgt'] = may_dat['PWCMPWGT']
del may_dat['HRHHID'], may_dat['OCCURNUM'], may_dat['YYYYMM'], may_dat[
    'HRHHID2'], may_dat['PRTAGE'], may_dat['PTERNHLY'], may_dat['PWCMPWGT']


def get_e_indiv(S, J, data):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age

    Returns:    e - S x J matrix of J working ability levels for each
                    age cohort measured by hourly wage, normalized so
                    the mean is one
    '''
    age_groups = np.linspace(16, 76, S+1)
    e = np.zeros((S, J))
    data = data[(15 < data.age) & (data.age < 77)]
    for i in xrange(S):
        incomes = data[(age_groups[i] <= data.age) & (
            data.age < age_groups[i+1])]
        incomes = incomes.sort(['wage'])
        inc = np.array(incomes.wage)
        wgt_ar = np.array(incomes.wgt)
        wgt_cum = np.zeros(inc.shape[0])
        cum_weight_scalar = 0
        for k in xrange(inc.shape[0]):
            cum_weight_scalar += wgt_ar[k]
            wgt_cum[k] = cum_weight_scalar
        total_wgts = wgt_cum[-1]
        for j in xrange(J):
            percentile = total_wgts*(j+.5)/J
            ind = 0
            while wgt_cum[ind] < percentile:
                ind += 1
            e[i, j] = inc[ind]
    e /= e.mean()
    return e


def get_e(S, J):
    e = np.zeros((S, J))
    e += get_e_indiv(S, J, jan_dat)
    e += get_e_indiv(S, J, feb_dat)
    e += get_e_indiv(S, J, mar_dat)
    e += get_e_indiv(S, J, apr_dat)
    e += get_e_indiv(S, J, may_dat)
    e /= 5
    return e


