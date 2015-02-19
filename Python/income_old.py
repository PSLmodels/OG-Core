'''
------------------------------------------------------------------------
Last updated 1/29/2015

Functions for created the matrix of ability levels, e.

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
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import scipy.optimize as opt

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


def fit_exp_right(params, point1, point2):
    a, b = params
    x1, y1 = point1
    x2, y2 = point2
    error1 = a*b**(-x1) - y1
    error2 = a*b**(-x2) - y2
    return [error1, error2]


def exp_int(points, a, b):
    top = a * ((1.0/(b**70)) - b**(-points))
    bottom = np.log(b)
    return top / bottom


def integrate(func, points, j):
    params_guess = [1, 1]
    # fit_to = j/2.0
    fit_to = poly.polyval(70, func) * .5
    a, b = opt.fsolve(fit_exp_right, params_guess, args=(
        [70, poly.polyval(70, func)], [100, fit_to]))
    func_int = poly.polyint(func)
    integral = np.empty(points.shape)
    integral[points <= 70] = poly.polyval(points[points <= 70], func_int)
    integral[points > 70] = poly.polyval(70, func_int) + exp_int(
        points[points > 70], a, b)
    vals = np.diff(integral)
    # vals[50:] = np.ones(30) * vals[50]
    return vals


def get_e_indiv(S, J, data, starting_age, ending_age, bin_weights):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age

    Returns:    e - S x J matrix of ability levels for each
                    age cohort measured by hourly wage, normalized so
                    the mean is one
    '''
    temp_ending_age = starting_age + 50
    age_groups = np.linspace(starting_age, temp_ending_age, 51)
    e = np.zeros((50, J))
    data = data[(starting_age <= data.age) & (data.age <= temp_ending_age)]
    for i in xrange(50):
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
        percentile = np.zeros(J)
        indicies = np.zeros(J+1)
        for j, weight in enumerate(bin_weights):
            percentile[j:] += weight
            ind = 0
            while (ind < len(wgt_cum)) and (
                    wgt_cum[ind] < total_wgts * percentile[j]):
                ind += 1
            indicies[j+1] = ind
        for j in xrange(J):
            e[i, j] = np.mean(inc[indicies[j]:indicies[j+1]])
    e /= e.mean()
    new_e = np.empty((S, J))
    for j in xrange(J):
        func = poly.polyfit(np.arange(50)+starting_age, e[:50, j], deg=2)
        new_e[:, j] = integrate(func, np.linspace(
            starting_age, ending_age, S+1), percentile[j])
    new_e /= new_e.mean()
    return new_e


def graph_income(S, J, e, starting_age, ending_age, bin_weights):
    domain = np.linspace(starting_age, ending_age, S)
    Jgrid = np.zeros(J)
    for j in xrange(J):
        Jgrid[j:] += bin_weights[j]
    X, Y = np.meshgrid(domain, Jgrid)
    if J == 1:
        plt.figure()
        plt.plot(domain, e)
        plt.savefig('OUTPUT/Demographics/ability')
    else:
        # 3D Graph
        cmap2 = matplotlib.cm.get_cmap('summer')
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, e.T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability-$j$')
        ax10.set_zlabel(r'Income Level $e_j(s)$')
        # ax10.set_title('Income Levels')
        plt.savefig('OUTPUT/Demographics/ability')


def get_e(S, J, starting_age, ending_age, bin_weights):
    e = np.zeros((S, J))
    e += get_e_indiv(S, J, jan_dat, starting_age, ending_age, bin_weights)
    e += get_e_indiv(S, J, feb_dat, starting_age, ending_age, bin_weights)
    e += get_e_indiv(S, J, mar_dat, starting_age, ending_age, bin_weights)
    e += get_e_indiv(S, J, apr_dat, starting_age, ending_age, bin_weights)
    e += get_e_indiv(S, J, may_dat, starting_age, ending_age, bin_weights)
    e /= 5
    return e
