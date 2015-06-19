'''
------------------------------------------------------------------------
Last updated 6/19/2015

Functions for created the matrix of ability levels, e.  This can
    only be used for looking at the 25, 50, 70, 80, 90, 99, and 100th
    percentiles, as it uses fitted polynomials to those percentiles.
    For a more generic version, see income_nopoly.py.


This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Demographics/ability_log
            OUTPUT/Demographics/ability
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import numpy as np
import scipy.optimize as opt


'''
------------------------------------------------------------------------
    Generate Polynomials
------------------------------------------------------------------------
The following coefficients are for polynomials which fit ability data
for the 25, 50, 70, 80, 90, 99, and 100 percentiles.  The data comes from
the following file: 

data/ability/FR_wage_profile_tables.xlsx

the polynomials are of the form
log(ability) = constant + (one)(age) + (two)(age)^2 + (three)(age)^3
------------------------------------------------------------------------
'''

# Vals for: .25 .25 .2 .1 .1 .09 .01
one = np.array([-0.09720122, 0.05995294, 0.17654618, 0.21168263, 0.21638731, 0.04500235, 0.09229392])                 
two = np.array([0.00247639, -0.00004086, -0.00240656, -0.00306555, -0.00321041, 0.00094253, 0.00012902])                   
three = np.array([-0.00001842, -0.00000521, 0.00001039, 0.00001438, 0.00001579, -0.00001470, -0.00001169])             
constant = np.array([3.41e+00, 0.69689692, -0.78761958, -1.11e+00, -0.93939272, 1.60e+00, 1.89e+00])
ages = np.linspace(21, 80, 60)
ages = np.tile(ages.reshape(60, 1), (1, 7))
income_profiles = constant + one * ages + two * ages ** 2 + three * ages ** 3
income_profiles = np.exp(income_profiles)


'''
------------------------------------------------------------------------
    Generate ability type matrix
------------------------------------------------------------------------
Given desired starting and stopping ages, as well as the values for S
and J, the ability matrix is created.  An arctan function is used
to extrapolate ability for ages 80-100.
------------------------------------------------------------------------
'''


def graph_income(S, J, e, starting_age, ending_age, bin_weights):
    '''
    Graphs the log of the ability matrix.
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    domain = np.linspace(starting_age, ending_age, S)
    Jgrid = np.zeros(J)
    for j in xrange(J):
        Jgrid[j:] += bin_weights[j]
    X, Y = np.meshgrid(domain, Jgrid)
    cmap2 = matplotlib.cm.get_cmap('winter')
    if J == 1:
        plt.figure()
        plt.plot(domain, np.log(e))
        plt.savefig('OUTPUT/Demographics/ability_log')
    else:
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, np.log(e).T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'log ability $log(e_j(s))$')
        # plt.show()
        plt.savefig('OUTPUT/Demographics/ability_log')
        # 2D Version
        fig112 = plt.figure()
        ax = plt.subplot(111)
        ax.plot(domain, np.log(e[:, 0]), label='0 - 24%', linestyle='-', color='black')
        ax.plot(domain, np.log(e[:, 1]), label='25 - 49%', linestyle='--', color='black')
        ax.plot(domain, np.log(e[:, 2]), label='50 - 69%', linestyle='-.', color='black')
        ax.plot(domain, np.log(e[:, 3]), label='70 - 79%', linestyle=':', color='black')
        ax.plot(domain, np.log(e[:, 4]), label='80 - 89%', marker='x', color='black')
        ax.plot(domain, np.log(e[:, 5]), label='90 - 99%', marker='v', color='black')
        ax.plot(domain, np.log(e[:, 6]), label='99 - 100%', marker='1', color='black')
        ax.axvline(x=80, color='black', linestyle='--')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.show()
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'log ability $log(e_j(s))$')
        plt.savefig('OUTPUT/Demographics/ability_log_2D')
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
        # plt.show()


def arc_tan_func(points, a, b, c):
    y = (-a / np.pi) * np.arctan(b*points + c) + a / 2
    return y


def arc_tan_deriv_func(points, a, b, c):
    y = -a * b / (np.pi * (1+(b*points+c)**2))
    return y


def arc_error(guesses, params):
    a, b, c = guesses
    first_point, coef1, coef2, coef3, ability_depreciation = params
    error1 = first_point - arc_tan_func(80, a, b, c)
    if (3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1) < 0:
        error2 = (3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1)*first_point - arc_tan_deriv_func(80, a, b, c)
    else:
        error2 = -.02 * first_point - arc_tan_deriv_func(80, a, b, c)
    # print (3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1) * first_point
    error3 = ability_depreciation * first_point - arc_tan_func(100, a, b, c)
    error = [np.abs(error1)] + [np.abs(error2)] + [np.abs(error3)]
    # print np.array(error).max()
    return error


def arc_tan_fit(first_point, coef1, coef2, coef3, ability_depreciation, init_guesses):
    guesses = init_guesses
    params = [first_point, coef1, coef2, coef3, ability_depreciation]
    a, b, c = opt.fsolve(arc_error, guesses, params)
    # print a, b, c
    # print np.array(arc_error([a, b, c], params)).max()
    old_ages = np.linspace(81, 100, 20)
    return arc_tan_func(old_ages, a, b, c)


def get_e(S, J, starting_age, ending_age, bin_weights, omega_SS, flag_graphs):
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
    e_short = income_profiles
    e_final = np.ones((S, J))
    e_final[:60, :] = e_short
    e_final[60:, :] = 0.0
    # This following variable is what percentage of ability at age 80 ability falls to at age 100
    ability_depreciation = np.array([.47, .5, .5, .5, .5, .7, .5])
    init_guesses = np.array([[58, 0.0756438545595, -5.6940142786],
                             [27, 0.069, -5],
                             [35, .06, -5],
                             [37, 0.339936555352, -33.5987329144],
                             [70.5229181668, 0.0701993896947, -6.37746859905],
                             [35, .06, -5],
                             [35, .06, -5]])
    for j in xrange(J):
        e_final[60:, j] = arc_tan_fit(e_final[59, j], one[j], two[j], three[j], ability_depreciation[j], init_guesses[j])
    if flag_graphs:
        graph_income(S, J, e_final, starting_age, ending_age, bin_weights)
    e_final /= (e_final * omega_SS).sum()
    return e_final
  