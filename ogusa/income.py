'''
------------------------------------------------------------------------
Functions for created the matrix of ability levels, e.  This can
only be used for looking at the 25, 50, 70, 80, 90, 99, and 100th
percentiles, as it uses fitted polynomials to those percentiles.
For a more generic version, see income_nopoly.py.
------------------------------------------------------------------------
'''
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
from . import utils
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os


def graph_income(ages, abil_midp, abil_pcts, emat, filesuffix=""):
    '''
    This function graphs ability matrix in 3D, 2D, log, and nolog

    Args:
        ages (Numpy array) ages represented in sample, length S
        abil_midp (Numpy array): midpoints of income percentile bins in
            each ability group
        abil_pcts (Numpy array): percent of population in each lifetime
            income group, length J
        emat (Numpy array): effective labor units by age and lifetime
            income group, size SxJ
        filesuffix (str): suffix to be added to plot files

    Returns:
        None

    '''
    J = abil_midp.shape[0]
    abil_mesh, age_mesh = np.meshgrid(abil_midp, ages)
    cmap1 = matplotlib.cm.get_cmap('summer')
    # Make sure that "./OUTPUT/ability" directory is created
    output_dir = "./OUTPUT/ability"
    utils.mkdirs(output_dir)
    if J == 1:
        # Plot of 2D, J=1 in levels
        plt.figure()
        plt.plot(ages, emat)
        filename = "ability_2D_lev" + filesuffix
        fullpath = os.path.join(output_dir, filename)
        plt.savefig(fullpath)
        plt.close()

        # Plot of 2D, J=1 in logs
        plt.figure()
        plt.plot(ages, np.log(emat))
        filename = "ability_2D_log" + filesuffix
        fullpath = os.path.join(output_dir, filename)
        plt.savefig(fullpath)
        plt.close()
    else:
        # Plot of 3D, J>1 in levels
        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(
            age_mesh, abil_mesh, emat, rstride=8, cstride=1,
            cmap=cmap1)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'ability $e_{j,s}$')
        filename = "ability_3D_lev" + filesuffix
        fullpath = os.path.join(output_dir, filename)
        plt.savefig(fullpath)
        plt.close()

        # Plot of 3D, J>1 in logs
        fig11 = plt.figure()
        ax11 = fig11.gca(projection='3d')
        ax11.plot_surface(
            age_mesh, abil_mesh, np.log(emat), rstride=8, cstride=1,
            cmap=cmap1)
        ax11.set_xlabel(r'age-$s$')
        ax11.set_ylabel(r'ability type -$j$')
        ax11.set_zlabel(r'log ability $log(e_{j,s})$')
        filename = "ability_3D_log" + filesuffix
        fullpath = os.path.join(output_dir, filename)
        plt.savefig(fullpath)
        plt.close()

        if J <= 10:  # Restricted because of line and marker types
            # Plot of 2D lines from 3D version in logs
            ax = plt.subplot(111)
            linestyles = np.array(["-", "--", "-.", ":", ])
            markers = np.array(["x", "v", "o", "d", ">", "|"])
            pct_lb = 0
            for j in range(J):
                this_label = (
                    str(int(np.rint(pct_lb))) + " - " +
                    str(int(np.rint(pct_lb + 100*abil_pcts[j]))) + "%")
                pct_lb += 100*abil_pcts[j]
                if j <= 3:
                    ax.plot(ages, np.log(emat[:, j]), label=this_label,
                            linestyle=linestyles[j], color='black')
                elif j > 3:
                    ax.plot(ages, np.log(emat[:, j]), label=this_label,
                            marker=markers[j-4], color='black')
            ax.axvline(x=80, color='black', linestyle='--')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_xlabel(r'age-$s$')
            ax.set_ylabel(r'log ability $log(e_{j,s})$')
            filename = "ability_2D_log" + filesuffix
            fullpath = os.path.join(output_dir, filename)
            plt.savefig(fullpath)
            plt.close()


def arctan_func(xvals, a, b, c):
    r'''
    This function generates predicted ability levels given data (xvals)
    and parameters a, b, and c, from the following arctan function:

    .. math::
        y = (-a / \pi) * \arctan(b * x + c) + (a / 2)

    Args:
        xvals (Numpy array): data inputs to arctan function
        a (scalar): scale parameter for arctan function
        b (scalar): curvature parameter for arctan function
        c (scalar): shift parameter for arctan function

    RETURNS:
        yvals (Numpy array): predicted values (output) of arctan
            function

    '''
    yvals = (-a / np.pi) * np.arctan(b * xvals + c) + (a / 2)
    return yvals


def arctan_deriv_func(xvals, a, b, c):
    r'''
    This function generates predicted derivatives of arctan function
    given data (xvals) and parameters a, b, and c. The functional form
    of the derivative of the function is the following:

    .. math::
        y = - (a * b) / (\pi * (1 + (b * xvals + c)^2))

    Args:
        xvals (Numpy array): data inputs to arctan derivative function
        a (scalar): scale parameter for arctan function
        b (scalar): curvature parameter for arctan function
        c (scalar): shift parameter for arctan function

    RETURNS:
        yvals (Numpy array): predicted values (output) of arctan
            derivative function

    '''
    yvals = -(a * b) / (np.pi * (1 + (b * xvals + c) ** 2))
    return yvals


def arc_error(abc_vals, params):
    '''
    This function returns a vector of errors in the three criteria on
    which the arctan function is fit to predict extrapolated ability in
    ages 81 to 100.::

        1) The arctan function value at age 80 must match the estimated
           original function value at age 80.
        2) The arctan function slope at age 80 must match the estimated
           original function slope at age 80.
        3) The level of ability at age 100 must be a given fraction
           (abil_deprec) below the ability level at age 80.

    Args:
        abc_vals (tuple): contains (a,b,c)
        a (scalar): scale parameter for arctan function
        b (scalar): curvature parameter for arctan function
        c (scalar): shift parameter for arctan function
        params (tuple): contains (first_point, coef1, coef2, coef3,
            abil_deprec)
        first_point (scalar): ability level at age 80, > 0
        coef1 (scalar): coefficient in log ability equation on linear
            term in age
        coef2 (scalar): coefficient in log ability equation on
            quadratic term in age
        coef3 (scalar): coefficient in log ability equation on cubic
            term in age
        abil_deprec (scalar): ability depreciation rate between
            ages 80 and 100, in (0, 1).

    Returns:
        error_vec (Numpy array): errors ([error1, error2, error3])
        error1 (scalar): error between ability level at age 80 from
            original function minus the predicted ability at age 80 from
            the arctan function given a, b, and c
        error2 (scalar): error between the slope of the original
            function at age 80 minus the slope of the arctan function at
            age 80 given a, b, and c
        error3 (scalar): error between the ability level at age 100
            predicted by the original model value times abil_deprec
            minus the ability predicted by the arctan function at age
            100 given a, b, and c

    '''
    a, b, c = abc_vals
    first_point, coef1, coef2, coef3, abil_deprec = params
    error1 = first_point - arctan_func(80, a, b, c)
    if (3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1) < 0:
        error2 = ((3 * coef3 * 80 ** 2 + 2 * coef2 * 80 + coef1) *
                  first_point - arctan_deriv_func(80, a, b, c))
    else:
        error2 = -.02 * first_point - arctan_deriv_func(80, a, b, c)
    error3 = abil_deprec * first_point - arctan_func(100, a, b, c)
    error_vec = np.array([error1, error2, error3])

    return error_vec


def arctan_fit(first_point, coef1, coef2, coef3, abil_deprec,
               init_guesses):
    '''
    This function fits an arctan function to the last 20 years of the
    ability levels of a particular ability group to extrapolate
    abilities by trying to match the slope in the 80th year and the
    ability depreciation rate between years 80 and 100.

    Args:
        first_point (scalar): ability level at age 80, > 0
        coef1 (scalar): coefficient in log ability equation on linear
            term in age
        coef2 (scalar): coefficient in log ability equation on
            quadratic term in age
        coef3 (scalar): coefficient in log ability equation on cubic
            term in age
        abil_deprec (scalar): ability depreciation rate between
            ages 80 and 100, in (0, 1)
        init_guesses (Numpy array): initial guesses

    Returns:
        abil_last (Numpy array): extrapolated ability levels for ages
            81 to 100, length 20

    '''
    params = [first_point, coef1, coef2, coef3, abil_deprec]
    solution = opt.root(arc_error, init_guesses,
                        args=params, method='lm')
    [a, b, c] = solution.x
    old_ages = np.linspace(81, 100, 20)
    abil_last = arctan_func(old_ages, a, b, c)
    return abil_last


def get_e_interp(S, age_wgts, age_wgts_80, abil_wgts, plot=False):
    '''
    This function takes a source matrix of lifetime earnings profiles
    (abilities, emat) of size (80, 7), where 80 is the number of ages
    and 7 is the number of ability types in the source matrix, and
    interpolates new values of a new S x J sized matrix of abilities
    using linear interpolation. [NOTE: For this application, cubic
    spline interpolation introduces too much curvature.]

    Args:
        S (int): number of ages to interpolate. This method assumes that
            ages are evenly spaced between the beginning of the 21st
            year and the end of the 100th year, >= 3
        age_wgts (Numpy array): distribution of population in each age
            for the interpolated ages, length S
        age_wgts_80 (Numpy array): percent of population in each
            one-year age from 21 to 100, length 80
        abil_wgts (Numpy array): distribution of population in each
            ability group, length J
        plot (bool): if True, creates plots of emat_orig and the new
            interpolated emat_new

    Returns:
        emat_new_scaled (Numpy array): interpolated ability matrix scaled
            so that population-weighted average is 1, size SxJ

    '''
    # Get original 80 x 7 ability matrix
    abil_wgts_orig = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
    emat_orig = get_e_orig(age_wgts_80, abil_wgts_orig, plot)

    # Return emat_orig if S = 80 and abil_wgts = abil_wgts_orig
    if S == 80 and np.array_equal(abil_wgts,
                                  np.array([0.25, 0.25, 0.2, 0.1, 0.1,
                                            0.09, 0.01])) is True:
        emat_new_scaled = emat_orig
    else:
        # generate abil_midp vector
        J = abil_wgts.shape[0]
        abil_midp = np.zeros(J)
        pct_lb = 0.0
        for j in range(J):
            abil_midp[j] = pct_lb + 0.5 * abil_wgts[j]
            pct_lb += abil_wgts[j]

        # Make sure that values in abil_midp are within interpolating
        # bounds set by the hard coded abil_wgts_orig
        if abil_midp.min() < 0.125 or abil_midp.max() > 0.995:
            err = ("One or more entries in abils vector is outside the "
                   + "allowable bounds.")
            raise RuntimeError(err)

        emat_j_midp = np.array([0.125, 0.375, 0.600, 0.750, 0.850,
                                0.945, 0.995])
        emat_s_midp = np.linspace(20.5, 99.5, 80)
        emat_j_mesh, emat_s_mesh = np.meshgrid(emat_j_midp, emat_s_midp)
        newstep = 80 / S
        new_s_midp = np.linspace(
            20 + 0.5 * newstep, 100 - 0.5 * newstep, S)
        new_j_mesh, new_s_mesh = np.meshgrid(abil_midp, new_s_midp)
        newcoords = np.hstack((emat_s_mesh.reshape((80*7, 1)),
                               emat_j_mesh.reshape((80*7, 1))))
        emat_new = si.griddata(newcoords, emat_orig.flatten(),
                               (new_s_mesh, new_j_mesh), method='linear')
        emat_new_scaled = emat_new / (emat_new * age_wgts.reshape(S, 1)
                                      * abil_wgts.reshape(1, J)).sum()

        if plot:
            kwargs = {'filesuffix': '_intrp_scaled'}
            graph_income(new_s_midp, abil_midp, abil_wgts,
                         emat_new_scaled, **kwargs)

    return emat_new_scaled


def get_e_orig(age_wgts, abil_wgts, plot=False):
    r'''
    This function generates the 80 x 7 matrix of lifetime earnings
    ability profiles, corresponding to annual ages from 21 to 100 and to
    paths based on income percentiles 0-25, 25-50, 50-70, 70-80, 80-90,
    90-99, 99-100. The ergodic population distribution is an input in
    order to rescale the paths so that the weighted average equals 1.

    The data come from the following file:

        `data/ability/FR_wage_profile_tables.xlsx`

    The polynomials are of the form

    .. math::
        \ln(abil) = \alpha + \beta_{1}\text{age} + \beta_{2}\text{age}^2 + \beta_{3}\text{age}^3

    Values come from regression analysis using IRS CWHS with hours
    imputed from the CPS.

    Args:
        age_wgts (Numpy array): ergodic age distribution, length S
        abil_wgts (Numpy array): population weights in each lifetime
            earnings group, length J
        plot (bool): if True, generates 3D plots of ability paths

    Returns:
        e_orig_scaled (Numpy array): = lifetime ability profiles scaled
            so that population-weighted average is 1, size SxJ

    '''
    # Return and error if age_wgts is not a vector of size (80,)
    if age_wgts.shape[0] != 80:
        err = "Vector age_wgts does not have 80 elements."
        raise RuntimeError(err)
    # Return and error if abil_wgts is not a vector of size (7,)
    if abil_wgts.shape[0] != 7:
        err = "Vector abil_wgts does not have 7 elements."
        raise RuntimeError(err)

    # 1) Generate polynomials and use them to get income profiles for
    #    ages 21 to 80.
    one = np.array([-0.09720122, 0.05995294, 0.17654618,
                    0.21168263, 0.21638731, 0.04500235, 0.09229392])
    two = np.array([0.00247639, -0.00004086, -0.00240656, -
                    0.00306555, -0.00321041, 0.00094253, 0.00012902])
    three = np.array([-0.00001842, -0.00000521, 0.00001039,
                      0.00001438, 0.00001579, -0.00001470, -0.00001169])
    const = np.array([3.41e+00, 0.69689692, -0.78761958, -1.11e+00,
                      -0.93939272, 1.60e+00, 1.89e+00])
    ages_short = np.tile(np.linspace(21, 80, 60).reshape((60, 1)),
                         (1, 7))
    log_abil_paths = (const + (one * ages_short) +
                      (two * (ages_short ** 2)) +
                      (three * (ages_short ** 3)))
    abil_paths = np.exp(log_abil_paths)
    e_orig = np.zeros((80, 7))
    e_orig[:60, :] = abil_paths
    e_orig[60:, :] = 0.0

    # 2) Forecast (with some art) the path of the final 20 years of
    #    ability types. This following variable is what percentage of
    #    ability at age 80 ability falls to at age 100. In general, we
    #    wanted people to lose half of their ability over a 20-year
    #    period. The first entry is 0.47, though, because nothing higher
    #    would converge. The second-to-last is 0.7 because this group
    #    actually has a slightly higher ability at age 80 than the last
    #    group, so this value makes it decrease more so it ends up being
    #    monotonic.
    abil_deprec = np.array([0.47, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5])
    #     Initial guesses for the arctan. They're pretty sensitive.
    init_guesses = np.array(
        [[58, 0.0756438545595, -5.6940142786], [27, 0.069, -5],
         [35, .06, -5], [37, 0.339936555352, -33.5987329144],
         [70.5229181668, 0.0701993896947, -6.37746859905],
         [35, .06, -5], [35, .06, -5]])
    for j in range(7):
        e_orig[60:, j] = arctan_fit(
            e_orig[59, j], one[j], two[j], three[j], abil_deprec[j],
            init_guesses[j])

    # 3) Rescale the lifetime earnings path matrix so that the
    #    population weighted average equals 1.
    e_orig_scaled = e_orig / (e_orig * age_wgts.reshape(80, 1)
                              * abil_wgts.reshape(1, 7)).sum()

    if plot:
        ages_long = np.linspace(21, 100, 80)
        abil_midp = np.array([12.5, 37.5, 60.0, 75.0, 85.0, 94.5, 99.5])
        # Plot original unscaled 80 x 7 ability matrix
        kwargs = {'filesuffix': '_orig_unscaled'}
        graph_income(ages_long, abil_midp, abil_wgts, e_orig, **kwargs)

        # Plot original scaled 80 x 7 ability matrix
        kwargs = {'filesuffix': '_orig_scaled'}
        graph_income(ages_long, abil_midp, abil_wgts, e_orig_scaled,
                     **kwargs)

    return e_orig_scaled
