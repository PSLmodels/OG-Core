'''
------------------------------------------------------------------------
Last updated 2/16/2014

Functions for created the matrix of ability levels, e.

This py-file calls the following other file(s):
            data/e_vec_data/cwhs_earn_rate_age_profile.csv

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Demographics/ability_log
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
import numpy.polynomial.polynomial as poly
import scipy.optimize as opt
from scipy import interpolate


'''
------------------------------------------------------------------------
    Read Data for Ability Types
------------------------------------------------------------------------
The data comes from the IRS.  We can either use wage or earnings data,
in this version we are using earnings data.  The data is for individuals
in centile groups for each age from 20 to 70.
------------------------------------------------------------------------
'''

earn_rate = pd.read_table(
    "data/e_vec_data/cwhs_earn_rate_age_profile.csv", sep=',', header=0)
del earn_rate['obs_earn']
piv = earn_rate.pivot(index='age', columns='q_earn', values='mean_earn_rate')

emat_basic = np.array(piv)

'''
------------------------------------------------------------------------
    Generate ability type matrix
------------------------------------------------------------------------
Given desired starting and stopping ages, as well as the values for S
and J, the ability matrix is created.
------------------------------------------------------------------------
'''


def fit_exp_right(params, pt1):
    a, b = params
    x1, y1, slope = pt1
    error1 = -a*b**(-x1)*np.log(b) - slope
    error2 = a*b**(-x1) - y1
    return [error1, error2]


def exp_funct(points, a, b):
    y = a*b**(-points)
    return y


def exp_fit(e_input, S, J):
    params_guess = [20, 1]
    e_output = np.zeros((S, J))
    e_output[:50, :] = e_input
    for j in xrange(J):
        meanslope = np.mean([e_input[-1, j]-e_input[-2, j], e_input[
            -2, j]-e_input[-3, j], e_input[-3, j]-e_input[-4, j]])
        slope = np.min([meanslope, -.01])
        a, b = opt.fsolve(fit_exp_right, params_guess, args=(
            [70, e_input[-1, j], slope]))
        e_output[50:, j] = exp_funct(np.linspace(70, 100, 30), a, b)
    return e_output


def graph_income(S, J, e, starting_age, ending_age, bin_weights):
    '''
    Graphs the log of the ability matrix.
    '''
    e_tograph = np.log(e)
    domain = np.linspace(starting_age, ending_age, S)
    Jgrid = np.zeros(J)
    for j in xrange(J):
        Jgrid[j:] += bin_weights[j]
    X, Y = np.meshgrid(domain, Jgrid)
    cmap2 = matplotlib.cm.get_cmap('summer')
    if J == 1:
        plt.figure()
        plt.plot(domain, e_tograph)
        plt.savefig('OUTPUT/Demographics/ability_log')
    else:
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, e_tograph.T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'log ability $log(e_j(s))$')
        plt.savefig('OUTPUT/Demographics/ability_log')
    if J == 1:
        plt.figure()
        plt.plot(domain, e)
        plt.savefig('OUTPUT/Demographics/ability')
    else:
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, e.T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'ability $e_j(s)$')
        plt.savefig('OUTPUT/Demographics/ability')


def get_e(S, J, starting_age, ending_age, bin_weights, omega_SS):
    '''
    Parameters: S - Number of age cohorts
                J - Number of ability levels by age
                starting_age - age of first age cohort
                ending_age - age of last age cohort
                bin_weights - what fraction of each age is in each
                              abiility type

    Returns:    e - S x J matrix of ability levels for each
                    age cohort, normalized so
                    the mean is one
    '''
    emat_trunc = emat_basic[:50, :]
    cum_bins = 100 * np.array(bin_weights)
    for j in xrange(J-1):
        cum_bins[j+1] += cum_bins[j]
    emat_collapsed = np.zeros((50, J))
    for s in xrange(50):
        for j in xrange(J):
            if j == 0:
                emat_collapsed[s, j] = emat_trunc[s, :cum_bins[j]].mean()
            else:
                emat_collapsed[s, j] = emat_trunc[
                    s, cum_bins[j-1]:cum_bins[j]].mean()
    e_fitted = np.zeros((50, J))
    for j in xrange(J):
        func = poly.polyfit(
            np.arange(50)+starting_age, emat_collapsed[:50, j], deg=2)

        e_fitted[:, j] = poly.polyval(np.arange(50)+starting_age, func)
    emat_extended = exp_fit(e_fitted, S, J)
    for j in xrange(1, J):
        emat_extended[:, j] = np.max(np.array(
            [emat_extended[:, j], emat_extended[:, j-1]]), axis=0)
    graph_income(S, J, emat_extended, starting_age, ending_age, bin_weights)
    emat_normed = emat_extended/(omega_SS * emat_extended).sum()
    return emat_normed
