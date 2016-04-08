'''
------------------------------------------------------------------------
Last updated 4/7/2016

Functions for created the matrix of ability levels, e.  This can
    only be used for looking at the 25, 50, 70, 80, 90, 99, and 100th
    percentiles, as it uses fitted polynomials to those percentiles.
    For a more generic version, see income_nopoly.py.

This file calls the following files:
    utils.py

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
import utils


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
# Values come from regression analysis using IRS CWHS with
# hours imputed from the CPS
one = np.array([-0.09720122, 0.05995294, 0.17654618,
                0.21168263, 0.21638731, 0.04500235, 0.09229392])
two = np.array([0.00247639, -0.00004086, -0.00240656, -
                0.00306555, -0.00321041, 0.00094253, 0.00012902])
three = np.array([-0.00001842, -0.00000521, 0.00001039,
                  0.00001438, 0.00001579, -0.00001470, -0.00001169])
constant = np.array([3.41e+00, 0.69689692, -0.78761958, -
                     1.11e+00, -0.93939272, 1.60e+00, 1.89e+00])
ages = np.linspace(21, 80, 60)
ages = np.tile(ages.reshape(60, 1), (1, 7))
income_profiles = constant + (one * ages) + (two * (ages ** 2)) + (three * (ages ** 3))
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


def graph_income(S, J, e, starting_age, ending_age, bin_weights,
                 output_dir='./OUTPUT'):
    '''
    Graphs the ability matrix (and it's log)

    Inputs:
        S            = number of age groups (scalar)
        J            = number of ability types (scalar)
        e            = ability matrix (SxJ array)
        starting_age = initial age (scalar)
        ending_age   = end age (scalar)
        bin_weights  = ability weights (Jx1 array)

    Outputs:
        OUTPUT/Demographics/ability_log.png
        OUTPUT/Demographics/ability.png
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
        ability = os.path.join(output_dir, "Demographics/ability_log")
        plt.savefig(ability)
    else:
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, np.log(e).T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'log ability $log(e_j(s))$')
        # plt.show()
        ability = os.path.join(output_dir, "Demographics/ability_log")
        plt.savefig(ability)
        # 2D Version
        fig112 = plt.figure()
        ax = plt.subplot(111)
        ax.plot(domain, np.log(e[:, 0]),
                label='0 - 24%', linestyle='-', color='black')
        ax.plot(domain, np.log(e[:, 1]), label='25 - 49%',
                linestyle='--', color='black')
        ax.plot(domain, np.log(e[:, 2]), label='50 - 69%',
                linestyle='-.', color='black')
        ax.plot(domain, np.log(e[:, 3]),
                label='70 - 79%', linestyle=':', color='black')
        ax.plot(domain, np.log(e[:, 4]),
                label='80 - 89%', marker='x', color='black')
        ax.plot(domain, np.log(e[:, 5]),
                label='90 - 99%', marker='v', color='black')
        ax.plot(domain, np.log(e[:, 6]),
                label='99 - 100%', marker='1', color='black')
        ax.axvline(x=80, color='black', linestyle='--')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'log ability $log(e_j(s))$')
        ability_2d = os.path.join(output_dir, "Demographics/ability_log_2D")
        plt.savefig(ability_2d)
    if J == 1:
        plt.figure()
        plt.plot(domain, e)
        ability = os.path.join(output_dir, "Demographics/ability")
        plt.savefig(ability)
    else:
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, e.T, rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'ability $e_j(s)$')
        ability = os.path.join(output_dir, "Demographics/ability")
        plt.savefig(ability)


def arc_tan_func(points, a, b, c):
    '''
    Functional form for a generic arctan function

    Inputs:
        points    = any length vector, grid on which to fit arctan function
        a         = scalar, scale parameter for arctan function
        b         = scalar, curvature parameter for arctan function
        c         = scalar, shift parameter for arctan function

    Functions called: None

    Objects in function:
        y = any length vector (same length as points), fitted values of arctan function

    Returns: y
    '''
    y = (-a / np.pi) * np.arctan(b * points + c) + a / 2
    return y


def arc_tan_deriv_func(points, a, b, c):
    '''
    Functional form for the derivative of a generic arctan function

    Inputs:
        points    = any length vector, grid on which to fit arctan function
        a         = scalar, scale parameter for arctan function
        b         = scalar, curvature parameter for arctan function
        c         = scalar, shift parameter for arctan function

    Functions called: None

    Objects in function:
        y = any length vector (same length as points), fitted values of arctan deriv function

    Returns: y
    '''
    y = -a * b / (np.pi * (1 + (b * points + c)**2))
    return y


def arc_error(guesses, params):
    '''
    How well the arctan function fits the slope of ability matrix at age 80, the level at age 80, and the level of age 80 times a constant

    Inputs:
        guesses    = length 3 tuple, (a,b,c)
        a         = scalar, scale parameter for arctan function
        b         = scalar, curvature parameter for arctan function
        c         = scalar, shift parameter for arctan function
        params    = length 5 tuple, (first_point, coef1, coef2, coef3, ability_depreciation)
        first_point = 
        coef1 = 
        coef2 = 
        coef3 = 
        ability_depreciation = 

    Functions called: 
        arc_tan_deriv_func
        arc_tan_func

    Objects in function:
        error = 

    Returns: error
    '''

    a, b, c = guesses
    first_point, coef1, coef2, coef3, ability_depreciation = params
    error1 = first_point - arc_tan_func(80, a, b, c)
    if (3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1) < 0:
        error2 = (3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1) * \
            first_point - arc_tan_deriv_func(80, a, b, c)
    else:
        error2 = -.02 * first_point - arc_tan_deriv_func(80, a, b, c)
    error3 = ability_depreciation * first_point - arc_tan_func(100, a, b, c)
    error = [np.abs(error1)] + [np.abs(error2)] + [np.abs(error3)]
    return error


def arc_tan_fit(first_point, coef1, coef2, coef3, ability_depreciation, init_guesses):
    '''
    Fits an arctan function to the last 20 years of the ability levels

    Inputs:
        first_point = 
        coef1 = 
        coef2 = 
        coef3 = 
        ability_depreciation = 
        init_guesses = 

    Functions called: 
        arc_error
        arc_tan_func

    Objects in function:
        a         = scalar, scale parameter for arctan function
        b         = scalar, curvature parameter for arctan function
        c         = scalar, shift parameter for arctan function
        old_ages  = [20,] vector, grid of ages 81-100

    Returns: arc_tan_func over ages 81-100
    '''
    guesses = init_guesses
    params = [first_point, coef1, coef2, coef3, ability_depreciation]
    a, b, c = opt.fsolve(arc_error, guesses, params)
    old_ages = np.linspace(81, 100, 20)
    return arc_tan_func(old_ages, a, b, c)


def get_e(S, J, starting_age, ending_age, bin_weights, omega_SS, flag_graphs):
    '''
    Inputs:
        S            = Number of age cohorts (scalar)
        J            = Number of ability levels by age (scalar)
        starting_age = age of first age cohort (scalar)
        ending_age   = age of last age cohort (scalar)
        bin_weights  = ability weights (Jx1 array)
        omega_SS     = population weights (Sx1 array)
        flag_graphs  = Graph flags or not (bool)

    Functions called: 
        arc_tan_fit
        graph_income

    Objects in function:
        e_short              = [S-20,J] array, ability levels for each age (except last 20
                                    years) and ability type
        e_final              = [S,J] array, ability levels for each age and ability type,
                                    normalize so the weighted sum is one
        ability_depreciation = [J,] vector, depreciaton rate for effective labor
                                    unites for each ability group
        init_guesses         = [3,J] array, initial guesses for parameters of the arctan 
                                    functional fit for the last 20 years of life
    
    Returns: e_final
    '''
    e_short = income_profiles
    e_final = np.ones((S, J))
    e_final[:60, :] = e_short
    e_final[60:, :] = 0.0
    # This following variable is what percentage of ability at age 80 ability falls to at age 100.
    # In general, we wanted people to lose half of their ability over a 20 year period.  The first
    # entry is .47, though, because nothing higher would converge.  The second to last is .7 because this group
    # actually has a slightly higher ability at age 80 then the last group, so this makes it decrease more so it
    # ends monotonic.
    ability_depreciation = np.array([.47, .5, .5, .5, .5, .7, .5])
    # Initial guesses for the arctan.  They're pretty sensitive.
    init_guesses = np.array([[58, 0.0756438545595, -5.6940142786],
                             [27, 0.069, -5],
                             [35, .06, -5],
                             [37, 0.339936555352, -33.5987329144],
                             [70.5229181668, 0.0701993896947, -6.37746859905],
                             [35, .06, -5],
                             [35, .06, -5]])
    for j in xrange(J):
        e_final[60:, j] = arc_tan_fit(e_final[59, j], one[j], two[j], three[
                                      j], ability_depreciation[j], init_guesses[j])
    if flag_graphs:
        graph_income(S, J, e_final, starting_age, ending_age, bin_weights)
    e_final /= (e_final * omega_SS.reshape(S, 1)
                * bin_weights.reshape(1, J)).sum()
    return e_final
