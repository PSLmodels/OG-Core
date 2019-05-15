'''
------------------------------------------------------------------------
This script reads in data generated from the OSPC Tax Calculator and
the 2009 IRS PUF. It then estimates tax functions tau_{s,t}(x,y), where
tau_{s,t} is the effective tax rate, marginal tax rate on labor income,
or the marginal tax rate on capital income, for a given age (s) in a
particular year (t). x is total labor income, and y is total capital
income.

This module defines the following functions:
    get_tax_rates()
    wsumsq()
    find_outliers()
    replace_outliers()

    tax_func_estimate()
    get_tax_func_estimate()

This Python script calls the following modules:
    get_micro_data.py
    utils.py

This Python script outputs the following:
    ./TAX_ESTIMATE_PATH/TxFuncEst_baseline{}.pkl
    ./TAX_ESTIMATE_PATH/TxFuncEst_policy{}.pkl
------------------------------------------------------------------------
'''
# Import packages
import time
import os
import numpy as np
import scipy.optimize as opt
from dask.distributed import Client
from dask import compute, delayed
import dask.multiprocessing
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from ogusa import get_micro_data
from ogusa.utils import DEFAULT_START_YEAR

TAX_ESTIMATE_PATH = os.environ.get("TAX_ESTIMATE_PATH", ".")

'''
------------------------------------------------------------------------
Define Functions
------------------------------------------------------------------------
'''


def gen_3Dscatters_hist(df, s, t, output_dir):
    '''
    --------------------------------------------------------------------
    Create 3-D scatterplots and corresponding 3D histogram of ETR, MTRx,
    and MTRy as functions of labor income and capital income with
    truncated data in the income dimension
    --------------------------------------------------------------------
    INPUTS:
    df         = (N1, 11) DataFrame, 11 variables with N observations
    s          = integer >= 21, age of individual
    t          = integer >= 2016, year of analysis
    output_dir = string, output directory for saving plot files

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    df_trnc   = (N2 x 6) DataFrame, truncated data for 3D graph
    inc_lab   = (N2 x 1) vector, total labor income for 3D graph
    inc_cap   = (N2 x 1) vector, total capital income for 3D graph
    etr_data  = (N2 x 1) vector, effective tax rate data
    mtrx_data = (N2 x 1) vector, marginal tax rate of labor income data
    mtry_data = (N2 x 1) vector, marginal tax rate of capital income
                data
    filename  = string, name of image file
    fullpath  = string, full path of file
    bin_num   = integer >= 2, number of bins along each axis for 3D
                histogram
    hist      = (bin_num, bin_num) matrix, bin percentages
    xedges    = (bin_num+1,) vector, bin edge values in x-dimension
    yedges    = (bin_num+1,) vector, bin edge values in y-dimension
    x_midp    = (bin_num,) vector, midpoints of bins in x-dimension
    y_midp    = (bin_num,) vector, midpoints of bins in y-dimension
    elements  = integer, total number of 3D histogram bins
    xpos      = (bin_num * bin_num) vector, x-coordinates of each bin
    ypos      = (bin_num * bin_num) vector, y-coordinates of each bin
    zpos      = (bin_num * bin_num) vector, zeros or z-coordinates of
                origin of each bin
    dx        = (bin_num,) vector, x-width of each bin
    dy        = (bin_num,) vector, y-width of each bin
    dz        = (bin_num * bin_num) vector, height of each bin

    FILES SAVED BY THIS FUNCTION:
        output_dir/ETR_Age_[age]_Year_[year]_data.png
        output_dir/MTRx_Age_[age]_Year_[year]_data.png
        output_dir/MTRy_Age_[age]_Year_[year]_data.png
        output_dir/Hist_Age_[age]_Year_[year].png

    RETURNS: None
    --------------------------------------------------------------------
    '''
    # Truncate the data
    df_trnc = df[(df['Total labor income'] > 5) &
                 (df['Total labor income'] < 500000) &
                 (df['Total capital income'] > 5) &
                 (df['Total capital income'] < 500000)]
    inc_lab = df_trnc['Total labor income']
    inc_cap = df_trnc['Total capital income']
    etr_data = df_trnc['ETR']
    mtrx_data = df_trnc['MTR labor income']
    mtry_data = df_trnc['MTR capital income']

    # Plot 3D scatterplot of ETR data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inc_lab, inc_cap, etr_data, c='r', marker='o')
    ax.set_xlabel('Total labor income')
    ax.set_ylabel('Total capital income')
    ax.set_zlabel('ETR')
    plt.title('ETR, Lab. Inc., and Cap. Inc., Age=' + str(s) +
              ', Year=' + str(t))
    filename = ("ETR_Age_" + str(s) + "_Year_" + str(t) + "_data.png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

    # Plot 3D histogram for all data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    bin_num = int(30)
    hist, xedges, yedges = np.histogram2d(inc_lab, inc_cap,
                                          bins=bin_num)
    hist = hist / hist.sum()
    x_midp = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
    y_midp = yedges[:-1] + 0.5 * (yedges[1] - yedges[0])
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    ypos, xpos = np.meshgrid(y_midp, x_midp)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements)
    dx = (xedges[1] - xedges[0]) * np.ones_like(bin_num)
    dy = (yedges[1] - yedges[0]) * np.ones_like(bin_num)
    dz = hist.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    ax.set_xlabel('Total labor income')
    ax.set_ylabel('Total capital income')
    ax.set_zlabel('Percent of obs.')
    plt.title('Histogram by lab. inc., and cap. inc., Age=' + str(s) +
              ', Year=' + str(t))
    filename = ("Hist_Age_" + str(s) + "_Year_" + str(t) + ".png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

    # Plot 3D scatterplot of MTRx data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inc_lab, inc_cap, mtrx_data, c='r', marker='o')
    ax.set_xlabel('Total labor income')
    ax.set_ylabel('Total capital income')
    ax.set_zlabel('Marginal Tax Rate, Labor Inc.)')
    plt.title("MTR labor income Income, Lab. Inc., and Cap. Inc., Age="
              + str(s) + ", Year=" + str(t))
    filename = ("MTRx_Age_" + str(s) + "_Year_" + str(t) + "_data.png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

    # Plot 3D scatterplot of MTRy data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inc_lab, inc_cap, mtry_data, c='r', marker='o')
    ax.set_xlabel('Total labor income')
    ax.set_ylabel('Total capital income')
    ax.set_zlabel('Marginal Tax Rate (Capital Inc.)')
    plt.title("MTR Capital Income, Cap. Inc., and Cap. Inc., Age=" +
              str(s) + ", Year=" + str(t))
    filename = ("MTRy_Age_" + str(s) + "_Year_" + str(t) + "_data.png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

    # Garbage collection
    del df, df_trnc, inc_lab, inc_cap, etr_data, mtrx_data, mtry_data


def plot_txfunc_v_data(tx_params, data, params):  # This isn't in use yet
    '''
    --------------------------------------------------------------------
    This function plots a single estimated tax function against its
    corresponding data
    --------------------------------------------------------------------
    cmap1       = color map object for matplotlib 3D plots
    tx_label    = string, text representing type of tax rate
    gridpts     = scalar > 2, number of grid points in X and Y
                  dimensions
    X_vec       = (gridpts,) vector, discretized log support of X
    Y_vec       = (gridpts,) vector, discretized log support of Y
    X_grid      = (gridpts, gridpts) matrix, ?
    Y_grid      = (gridpts, gridpts) matrix, ?
    txrate_grid = (gridpts, gridpts) matrix, ?
    filename    = string, name of plot to be saved
    fullpath    = string, full path name of file to be saved
    df_trnc_gph = (Nb, 11) DataFrame, truncated data for plotting
    X_gph       = (Nb,) Series, truncated labor income data
    Y_gph       = (Nb,) Series, truncated capital income data
    txrates_gph = (Nb,) Series, truncated tax rate (ETR, MTRx, or MTRy)
                  data
    --------------------------------------------------------------------
    '''
    X_data = data['Total labor income']
    Y_data = data['Total capital income']
    (s, t, rate_type, plot_full, plot_trunc, show_plots, save_plots,
        output_dir) = params

    cmap1 = matplotlib.cm.get_cmap('summer')

    if plot_full:
        if rate_type == 'etr':
            txrate_data = data['ETR']
            tx_label = 'ETR'
        elif rate_type == 'mtrx':
            txrate_data = data['MTR labor income']
            tx_label = 'MTRx'
        elif rate_type == 'mtry':
            txrate_data = data['MTR capital income']
            tx_label = 'MTRy'
        # Make comparison plot with full income domains
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_data, Y_data, txrate_data, c='r', marker='o')
        ax.set_xlabel('Total labor income')
        ax.set_ylabel('Total capital income')
        ax.set_zlabel(tx_label)
        plt.title(tx_label + ' vs. Predicted ' + tx_label + ': Age=' +
                  str(s) + ', Year=' + str(t))

        gridpts = 50
        X_vec = np.exp(np.linspace(np.log(1), np.log(X_data.max()),
                                   gridpts))
        Y_vec = np.exp(np.linspace(np.log(1), np.log(Y_data.max()),
                                   gridpts))
        X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
        txrate_grid = get_tax_rates(tx_params, X_grid, Y_grid, None,
                                    tax_func_type, rate_type,
                                    for_estimation=False)
        ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1,
                        linewidth=0)

        if save_plots:
            filename = (tx_label + '_Age_' + str(s) + '_Year_' + str(t)
                        + '_vsPred.png')
            fullpath = os.path.join(output_dir, filename)
            fig.savefig(fullpath, bbox_inches='tight')

        if show_plots:
            plt.show()

        plt.close()

    if plot_trunc:
        # Make comparison plot with truncated income domains
        data_trnc = data[(data['Total labor income'] > 5) &
                         (data['Total labor income'] < 800000) &
                         (data['Total capital income'] > 5) &
                         (data['Total capital income'] < 800000)]
        X_trnc = data_trnc['Total labor income']
        Y_trnc = data_trnc['Total capital income']
        if rate_type == 'etr':
            txrates_trnc = data_trnc['ETR']
            tx_label = 'ETR'
        elif rate_type == 'mtrx':
            txrates_trnc = data_trnc['MTR labor income']
            tx_label = 'MTRx'
        elif rate_type == 'mtry':
            txrates_trnc = data_trnc['MTR capital income']
            tx_label = 'MTRy'

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_trnc, Y_trnc, txrates_trnc, c='r', marker='o')
        ax.set_xlabel('Total labor income')
        ax.set_ylabel('Total capital income')
        ax.set_zlabel(tx_label)
        plt.title('Truncated ' + tx_label + ', Lab. Inc., and Cap. ' +
                  'Inc., Age=' + str(s) + ', Year=' + str(t))

        gridpts = 50
        X_vec = np.exp(np.linspace(np.log(1), np.log(X_trnc.max()),
                                   gridpts))
        Y_vec = np.exp(np.linspace(np.log(1), np.log(Y_trnc.max()),
                                   gridpts))
        X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
        txrate_grid = get_tax_rates(tx_params, X_grid, Y_grid, None,
                                    tax_func_type, rate_type,
                                    for_estimation=False)
        ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1,
                        linewidth=0)

        if save_plots:
            filename = (tx_label + 'trunc_Age_' + str(s) + '_Year_' +
                        str(t) + '_vsPred.png')
            fullpath = os.path.join(output_dir, filename)
            fig.savefig(fullpath, bbox_inches='tight')

        if show_plots:
            plt.show()

        plt.close()


def get_tax_rates(params, X, Y, wgts, tax_func_type, rate_type,
                  for_estimation=True):
    X2 = X ** 2
    Y2 = Y ** 2
    income = X + Y
    if tax_func_type == 'GS':
        phi0, phi1, phi2 = params[:3]
        if rate_type == 'etr':
            txrates = (
                (phi0 * (income - ((income ** -phi1) + phi2) **
                         (-1 / phi1))) / income)
        else:  # marginal tax rate function
            txrates = (phi0*(1 - (income ** (-phi1 - 1) *
                                  ((income ** -phi1) + phi2)
                                  ** ((-1 - phi1) / phi1))))
    elif tax_func_type == 'DEP':
        A, B, C, D, max_x, max_y, share, min_x, min_y, shift = params
        shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
        shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
        Etil = A + B
        Ftil = C + D
        if for_estimation:
            X2bar = (X2 * wgts).sum() / wgts.sum()
            Xbar = (X * wgts).sum() / wgts.sum()
            Y2bar = (Y2 * wgts).sum() / wgts.sum()
            Ybar = (Y * wgts).sum() / wgts.sum()
            X2til = (X2 - X2bar) / X2bar
            Xtil = (X - Xbar) / Xbar
            Y2til = (Y2 - Y2bar) / Y2bar
            Ytil = (Y - Ybar) / Ybar
            tau_x = (((max_x - min_x) * (A * X2til + B * Xtil + Etil) /
                      (A * X2til + B * Xtil + Etil + 1)) + min_x)
            tau_y = (((max_y - min_y) * (C * Y2til + D * Ytil + Ftil) /
                      (C * Y2til + D * Ytil + Ftil + 1)) + min_y)
            txrates = (((tau_x + shift_x) ** share) *
                       ((tau_y + shift_y) ** (1 - share))) + shift
        else:
            tau_x = (((max_x - min_x) * (A * X2 + B * X) /
                      (A * X2 + B * X + 1)) + min_x)
            tau_y = (((max_y - min_y) * (C * Y2 + D * Y) /
                      (C * Y2 + D * Y + 1)) + min_y)
            txrates = (((tau_x + shift_x) ** share) *
                       ((tau_y + shift_y) ** (1 - share))) + shift
    elif tax_func_type == 'DEP_totalinc':
        A, B, max_income, min_income, shift = params
        shift_income = (np.maximum(-min_income, 0.0) + 0.01 *
                        (max_income - min_income))
        Etil = A + B
        income2 = income ** 2
        if for_estimation:
            income2bar = (income2 * wgts).sum() / wgts.sum()
            Ibar = (income * wgts).sum() / wgts.sum()
            income2til = (income2 - income2bar) / income2bar
            Itil = (income - Ibar) / Ibar
            tau_income = (((max_income - min_income) *
                           (A * income2til + B * Itil + Etil) /
                           (A * income2til + B * Itil + Etil + 1)) +
                          min_income)
            txrates = tau_income + shift_income + shift
        else:
            tau_income = (((max_income - min_income) *
                           (A * income2 + B * income) /
                           (A * income2 + B * income + 1)) + min_income)
            txrates = tau_income + shift_income + shift

    return txrates


def wsumsq(params, *args):
    '''
    --------------------------------------------------------------------
    This function generates the weighted sum of squared deviations of
    predicted values of tax rates (ETR, MTRx, or MTRy) from the tax
    rates from the data for the Cobb-Douglas functional form of the tax
    function.
    --------------------------------------------------------------------
    INPUTS:
    params  = (7,) vector, guesses for (A, B, C, D,
              max_x, max_y, share)
    A   = scalar > 0, adjusted coefficient on \hat{X^2} term
    B   = scalar > 0, adjusted coefficient on \hat{X} term
    C   = scalar > 0, adjusted coefficient on \hat{Y^2} term
    D   = scalar > 0, adjusted coefficient on \hat{Y} term
    max_x   = scalar > 0, maximum asymptotic tax rate when y=0
    max_y   = scalar > 0, maximum asymptotic tax rate when x=0
    share   = scalar in [0,1], share parameter in Cobb-Douglas function
    args    = length 7 tuple, (X, Y, min_x, min_y, shift, txrates, wgts)
    X       = (N,) Series, X (labor income) data
    Y       = (N,) Series, Y (capital income) data
    min_x   = scalar < max_x, minimum value of tax rate when y=0
    min_y   = scalar < max_y, minimum value of tax rate when x=0
    shift   = scalar, shifts the entire tax rate function
    txrates = (N,) Series, tax rate data (ETR, MTRx, or MTRy)
    wgts    = (N,) Series, population weights for each observation

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    X2          = (N,) Series, X^2 (labor income ** 2) data
    Y2          = (N,) Series, Y^2 (capital income ** 2) data
    X2bar       = scalar > 0, weighted average of X2 (labor income^2)
    Xbar        = scalar > 0, weighted average of X (labor income)
    Y2bar       = scalar > 0, weighted average of Y2 (capital income^2)
    Ybar        = scalar > 0, weighted average of Y (capital income)
    shift_x     = scalar, shifter to make tau(x) in CES positive
    shift_y     = scalar, shifter to make tau(y) in CES positive
    X2til       = (N,) Series, X2 percent deviation from weighted mean
    Xtil        = (N,) Series, X percent deviation from weighted mean
    Y2til       = (N,) Series, Y2 percent deviation from weighted mean
    Ytil        = (N,) Series, Y percent deviation from weighted mean
    Etil        = scalar > 0, constant term in adjusted X polynomial
    Ftil        = scalar > 0, constant term in adjusted Y polynomial
    tau_x       = (N,) Series, ratio of polynomials function tau(X)
                  evaluated at points X
    tau_y       = (N,) Series, ratio of polynomials function tau(Y)
                  evaluated at points Y
    txrates_est = (N,) Series, predicted tax rates (ETR, MTRx, MTRy) for
                  each observation
    errors      = (N,) Series, difference between predicted tax rates
                  and the tax rates from the data
    wssqdev     = scalar > 0, weighted sum of squared deviations

    RETURNS: wssqdev
    --------------------------------------------------------------------
    '''
    (fixed_tax_func_params, X, Y, txrates, wgts, tax_func_type,
     rate_type) = args
    params_all = np.append(params, fixed_tax_func_params)
    txrates_est = get_tax_rates(params_all, X, Y, wgts, tax_func_type,
                                rate_type)
    errors = txrates_est - txrates
    wssqdev = (wgts * (errors ** 2)).sum()

    return wssqdev


def find_outliers(sse_mat, age_vec, se_mult, start_year, varstr,
                  graph=False):
    '''
    --------------------------------------------------------------------
    This function takes a matrix of sum of squared errors (SSE) from
    tax function estimations for each age (s) in each year of the budget
    window (t) and marks estimations that have outlier SSE.
    --------------------------------------------------------------------
    INPUTS:
    sse_mat    = [S,BW] array, SSE for each estimated tax function
    age_vec    = [S,] vector, vector of ages
    se_mult    = scalar, multiple of standard deviations before consider
                  estimate an outlier
    start_year = integer, first year of budget window
    varstr     = string, name of tax function being evaluated
    graph      = boolean, flag to output graphs
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    OBJECTS CREATED WITHIN FUNCTION:
    thresh      = [S,BW] array, threshold values for SSE before consider
                   tax function outlier
    sse_big_mat = [S,BW] array, indicators of weither tax function is
                  outlier
    RETURNS: sse_big_mat
    --------------------------------------------------------------------
    '''
    # Mark outliers from estimated MTRx functions
    thresh = (sse_mat[sse_mat > 0].mean() +
              se_mult * sse_mat[sse_mat > 0].std())
    sse_big_mat = sse_mat > thresh
    print(varstr, ": ", str(sse_big_mat.sum()),
          " observations tagged as outliers.")
    if graph:
        # Plot sum of squared errors of tax functions over age for each
        # year of budget window
        fig, ax = plt.subplots()
        plt.plot(age_vec, sse_mat[:, 0], label=str(start_year))
        plt.plot(age_vec, sse_mat[:, 1], label=str(start_year + 1))
        plt.plot(age_vec, sse_mat[:, 2], label=str(start_year + 2))
        plt.plot(age_vec, sse_mat[:, 3], label=str(start_year + 3))
        plt.plot(age_vec, sse_mat[:, 4], label=str(start_year + 4))
        plt.plot(age_vec, sse_mat[:, 5], label=str(start_year + 5))
        plt.plot(age_vec, sse_mat[:, 6], label=str(start_year + 6))
        plt.plot(age_vec, sse_mat[:, 7], label=str(start_year + 7))
        plt.plot(age_vec, sse_mat[:, 8], label=str(start_year + 8))
        plt.plot(age_vec, sse_mat[:, 9], label=str(start_year + 9))
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.legend(loc='upper left')
        titletext = "Sum of Squared Errors by Age and Tax Year: " + varstr
        plt.title(titletext)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'SSE')
        # Create directory if OUTPUT directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "OUTPUT/TaxFunctions"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        graphname = "SSE_" + varstr
        output_path = os.path.join(output_dir, graphname)
        plt.savefig(output_path)
        # plt.show()
    if sse_big_mat.sum() > 0:
        # Mark the outliers from the first sweep above. Then mark the
        # new outliers in a second sweep
        sse_mat_new = sse_mat.copy()
        sse_mat_new[sse_big_mat] = np.nan
        thresh2 = (sse_mat_new[sse_mat_new > 0].mean() + se_mult *
                   sse_mat_new[sse_mat_new > 0].std())
        sse_big_mat += sse_mat_new > thresh2
        print(varstr, ": ", "After second round, ",
              str(sse_big_mat.sum()),
              " observations tagged as outliers (cumulative).")
        if graph:
            # Plot sum of squared errors of tax functions over age for
            # each year of budget window
            fig, ax = plt.subplots()
            plt.plot(age_vec, sse_mat_new[:, 0], label=str(start_year))
            plt.plot(age_vec, sse_mat_new[:, 1],
                     label=str(start_year + 1))
            plt.plot(age_vec, sse_mat_new[:, 2],
                     label=str(start_year + 2))
            plt.plot(age_vec, sse_mat_new[:, 3],
                     label=str(start_year + 3))
            plt.plot(age_vec, sse_mat_new[:, 4],
                     label=str(start_year + 4))
            plt.plot(age_vec, sse_mat_new[:, 5],
                     label=str(start_year + 5))
            plt.plot(age_vec, sse_mat_new[:, 6],
                     label=str(start_year + 6))
            plt.plot(age_vec, sse_mat_new[:, 7],
                     label=str(start_year + 7))
            plt.plot(age_vec, sse_mat_new[:, 8],
                     label=str(start_year + 8))
            plt.plot(age_vec, sse_mat_new[:, 9],
                     label=str(start_year + 9))
            # for the minor ticks, use no labels; default NullFormatter
            minorLocator = MultipleLocator(1)
            ax.xaxis.set_minor_locator(minorLocator)
            plt.grid(b=True, which='major', color='0.65', linestyle='-')
            plt.legend(loc='upper left')
            titletext = ("Sum of Squared Errors by Age and Tax Year" +
                         " minus outliers (round 1): " + varstr)
            plt.title(titletext)
            plt.xlabel(r'Age $s$')
            plt.ylabel(r'SSE')
            graphname = "SSE_" + varstr + "_NoOut1"
            output_path = os.path.join(output_dir, graphname)
            plt.savefig(output_path)
            # plt.show()
        if (sse_mat_new > thresh2).sum() > 0:
            # Mark the outliers from the second sweep above
            sse_mat_new2 = sse_mat_new.copy()
            sse_mat_new2[sse_big_mat] = np.nan
            if graph:
                # Plot sum of squared errors of tax functions over age
                # for each year of budget window
                fig, ax = plt.subplots()
                plt.plot(age_vec, sse_mat_new2[:, 0], label=str(start_year))
                plt.plot(age_vec, sse_mat_new2[:, 1],
                         label=str(start_year + 1))
                plt.plot(age_vec, sse_mat_new2[:, 2],
                         label=str(start_year + 2))
                plt.plot(age_vec, sse_mat_new2[:, 3],
                         label=str(start_year + 3))
                plt.plot(age_vec, sse_mat_new2[:, 4],
                         label=str(start_year + 4))
                plt.plot(age_vec, sse_mat_new2[:, 5],
                         label=str(start_year + 5))
                plt.plot(age_vec, sse_mat_new2[:, 6],
                         label=str(start_year + 6))
                plt.plot(age_vec, sse_mat_new2[:, 7],
                         label=str(start_year + 7))
                plt.plot(age_vec, sse_mat_new2[:, 8],
                         label=str(start_year + 8))
                plt.plot(age_vec, sse_mat_new2[:, 9],
                         label=str(start_year + 9))
                # for the minor ticks, use no labels; default NullFormatter
                minorLocator = MultipleLocator(1)
                ax.xaxis.set_minor_locator(minorLocator)
                plt.grid(b=True, which='major', color='0.65',
                         linestyle='-')
                plt.legend(loc='upper left')
                titletext = ("Sum of Squared Errors by Age and Tax Year"
                             + " minus outliers (round 2): " + varstr)
                plt.title(titletext)
                plt.xlabel(r'Age $s$')
                plt.ylabel(r'SSE')
                graphname = "SSE_" + varstr + "_NoOut2"
                output_path = os.path.join(output_dir, graphname)
                plt.savefig(output_path)
                # plt.show()

    return sse_big_mat


def replace_outliers(param_arr, sse_big_mat):
    '''
    --------------------------------------------------------------------
    This function replaces outlier estimated tax function parameters
    with linearly interpolated tax function tax function parameters
    --------------------------------------------------------------------
    INPUTS:
    sse_big_mat = [S,BW] array, indicators of weither tax function is outlier
    param_arr   = [S,BW,#tax params] array, estimated tax function parameters
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    OBJECTS CREATED WITHIN FUNCTION:
    age_ind       = [S,] list, list of ages
    param_arr_adj = [S,BW,#tax params] array, estimated and interpolated
                    tax function parameters
    big_cnt       = integer, number of outliers replaced
    slopevec      = [1,1,#tax params] array, slope used for linear
                    interpolation of outliers
    interceptvec  = [1,1,#tax params] array, intercept used for linear
                    interpolation of outliers
    RETURNS: param_arr_adj
    --------------------------------------------------------------------
    '''
    numparams = param_arr.shape[2]
    age_ind = np.arange(0, sse_big_mat.shape[0])
    param_arr_adj = param_arr.copy()
    for t in range(sse_big_mat.shape[1]):
        big_cnt = 0
        for s in age_ind:
            # Smooth out ETR tax function outliers
            if sse_big_mat[s, t] and s < sse_big_mat.shape[0]-1:
                # For all outlier observations, increase the big_cnt by
                # 1 and set the param_arr_adj equal to nan
                big_cnt += 1
                param_arr_adj[s, t, :] = np.nan
            if not sse_big_mat[s, t] and big_cnt > 0 and s == big_cnt:
                # When the current function is not an outlier but the last
                # one was and this string of outliers is at the beginning
                # ages, set the outliers equal to this period's tax function
                reshaped = param_arr_adj[s, t, :].reshape((1, 1,
                                                           numparams))
                param_arr_adj[:big_cnt, t, :] = np.tile(reshaped,
                                                        (big_cnt, 1))
                big_cnt = 0

            if not sse_big_mat[s, t] and big_cnt > 0 and s > big_cnt:
                # When the current function is not an outlier but the last
                # one was and this string of outliers is in the interior of
                # ages, set the outliers equal to a linear interpolation
                # between the two bounding non-outlier functions
                diff = (param_arr_adj[s, t, :] -
                        param_arr_adj[s-big_cnt-1, t, :])
                slopevec = diff / (big_cnt + 1)
                slopevec = slopevec.reshape(1, numparams)
                tiled_slopevec = np.tile(slopevec, (big_cnt, 1))

                interceptvec = \
                    param_arr_adj[s-big_cnt-1, t, :].reshape(1,
                                                             numparams)
                tiled_intvec = np.tile(interceptvec, (big_cnt, 1))

                reshaped_arange =\
                    np.arange(1, big_cnt+1).reshape(big_cnt, 1)
                tiled_reshape_arange =\
                    np.tile(reshaped_arange, (1, numparams))

                param_arr_adj[s-big_cnt:s, t, :] = (
                    tiled_intvec + tiled_slopevec * tiled_reshape_arange
                )

                big_cnt = 0
            if sse_big_mat[s, t] and s == sse_big_mat.shape[0] - 1:
                # When the last ages are outliers, set the parameters equal
                # to the most recent non-outlier tax function
                big_cnt += 1
                param_arr_adj[s, t, :] = np.nan
                reshaped = \
                    param_arr_adj[s-big_cnt, t, :].reshape(1, 1,
                                                           numparams)
                param_arr_adj[s-big_cnt+1:, t, :] = np.tile(reshaped,
                                                            (big_cnt,
                                                             1))

    return param_arr_adj


def txfunc_est(df, s, t, rate_type, tax_func_type, numparams,
               output_dir, graph):
    '''
    --------------------------------------------------------------------
    This function uses tax tax rate and income data for individuals of a
    particular age (s) and a particular year (t) to estimate the
    parameters of a Cobb-Douglas aggregation function of two ratios of
    polynomials in labor income and capital income, respectively.
    --------------------------------------------------------------------
    INPUTS:
    df         = (N, 11) DataFrame, data variables indexed by 11
                 variable names
    s          = integer >= 21, age
    t          = integer >= 2016, year
    rate_type  = string, either 'etr', 'mtrx', or 'mtry'
    output_dir = string, output directory in which to save plots
    graph      = Boolean, =True graphs the estimated functions compared
                 to the data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        wsumsq()
        utils.mkdirs()
        get_tax_rates()

    OBJECTS CREATED WITHIN FUNCTION:
    X           = (N,) Series, labor income data
    Y           = (N,) Series, capital income data
    wgts        = (N,) Series, population weights on the data
    X2          = (N,) Series, labor income squared (X^2)
    Y2          = (N,) Series, capital income squared (Y^2)
    X2bar       = scalar > 0, population weighted mean of X2
    Xbar        = scalar > 0, population weighted mean of X
    Y2bar       = scalar > 0, population weighted mean of Y2
    Ybar        = scalar > 0, population weighted mean of Y
    txrates     = (N,) vector, tax rates from data (ETR, MTRx, or MTRy)
    x_10pctl    = scalar > 0, 10th percentile of labor income data
    y_10pctl    = scalar > 0, 10th percentile of capital income data
    x_20pctl    = scalar > 0, 20th percentile of labor income data
    y_20pctl    = scalar > 0, 20th percentile of capital income data
    min_x       = scalar, minimum tax rate for X given Y=0
    min_y       = scalar, minimum tax rate for Y given X=0
    Atil_init   = scalar > 0, initial guess for coefficient on \hat{X^2}
    Btil_init   = scalar > 0, initial guess for coefficient on \hat{X}
    Ctil_init   = scalar > 0, initial guess for coefficient on \hat{Y^2}
    Dtil_init   = scalar > 0, initial guess for coefficient on \hat{Y}
    ub_max_x    = scalar > 0, maximum amount of capital income at the
                  low range of capital income used to calculate the
                  asymptotic maximum tax rate when capital income is
                  close to zero
    ub_max_y    = scalar > 0, maximum amount of labor income at the low
                  range of labor income used to calculate the asymptotic
                  maximum tax rate when labor income is close to zero
    max_x_init  = scalar > 0, initial guess for maximum tax rate for X
                  given Y=0
    max_y_init  = scalar > 0, initial guess for maximum tax rate for Y
                  given X=0
    shift       = scalar, adds to the Cobb-Douglas function to capture
                  negative tax rates
    share_init  = scalar in [0,1], share parameter in Cobb-Douglas
                  function
    numparams   = integer > 1, number of parameters to characterize
                  function
    params_init = (7,) vector, parameters for minimization function
                  (Atil_init, Btil_init, Ctil_init, Dtil_init,
                  max_x_init, max_y_init, share_init)
    tx_objs     = length 7 tuple, arguments to be passed in to minimizer
                  (X, Y, min_x, min_y, shift, txrates, wgts)
    lb_max_x    = scalar > 0, lower bound for max_x. Must be greater
                  than min_x
    lb_max_y    = scalar > 0, lower bound for max_y. Must be greater
                  than min_y
    bnds        = length 7 tuple, max and min parameter bounds
    params_til  = dictionary, output from minimization
    Atil        = scalar, estimated coefficient on \hat{X^2} term
    Btil        = scalar, estimated coefficient on \hat{X} term
    Ctil        = scalar, estimated coefficient on \hat{Y^2} term
    Dtil        = scalar, estimated coefficient on \hat{Y} term
    max_x       = scalar > 0, estimated value for max_x (labor income)
    max_y       = scalar > 0, estimated value for max_y (capital income)
    share       = scalar in [0, 1], estimated Cobb-Douglas share param
    wsse        = scalar > 0, weighted sum of squared deviations from
                  minimization
    obs         = integer > 600, number of obervations in the data (N)
    shift_x     = scalar, adds to tau(Y) to assure value in rate(X,Y)
                  term is strictly positive
    shift_y     = scalar, adds to tau(Y) to assure value in rate(X,Y)
                  term is strictly positive
    params      = (12,) vector, all parameters for Cobb-Douglas
                  functional form, both estimated and calibrated (A, B,
                  C, D, max_x, min_x, max_y, min_y, shift_x, shift_y,
                  shift, share)

    RETURNS: params, wsse, obs
    --------------------------------------------------------------------
    '''
    X = df['Total labor income']
    Y = df['Total capital income']
    wgts = df['Weights']
    X2 = X ** 2
    Y2 = Y ** 2
    X2bar = (X2 * wgts).sum() / wgts.sum()
    Xbar = (X * wgts).sum() / wgts.sum()
    Y2bar = (Y2 * wgts).sum() / wgts.sum()
    Ybar = (Y * wgts).sum() / wgts.sum()
    income = X + Y
    income2 = income ** 2
    Ibar = (income * wgts).sum() / wgts.sum()
    income2bar = (income2 * wgts).sum() / wgts.sum()
    if rate_type == 'etr':
        txrates = df['ETR']
    elif rate_type == 'mtrx':
        txrates = df['MTR labor income']
    elif rate_type == 'mtry':
        txrates = df['MTR capital income']
    x_10pctl = df['Total labor income'].quantile(0.1)
    y_10pctl = df['Total capital income'].quantile(0.1)
    x_20pctl = df['Total labor income'].quantile(.2)
    y_20pctl = df['Total capital income'].quantile(.2)
    min_x = txrates[(df['Total capital income'] < y_10pctl)].min()
    min_y = txrates[(df['Total labor income'] < x_10pctl)].min()

    if tax_func_type == 'DEP':
        '''
        Estimate DeBacker, Evans, Phillips (2018) ratio of polynomial
        tax functions.
        '''
        Atil_init = 1.0
        Btil_init = 1.0
        Ctil_init = 1.0
        Dtil_init = 1.0
        max_x_init = np.minimum(
            txrates[(df['Total capital income'] < y_20pctl)].max(), 0.7)
        max_y_init = np.minimum(
            txrates[(df['Total labor income'] < x_20pctl)].max(), 0.7)
        shift = txrates[(df['Total labor income'] < x_20pctl) |
                        (df['Total capital income'] < y_20pctl)].min()
        share_init = 0.5
        params_init = np.array([Atil_init, Btil_init, Ctil_init,
                                Dtil_init, max_x_init, max_y_init,
                                share_init])
        tx_objs = (np.array([min_x, min_y, shift]), X, Y, txrates, wgts,
                   tax_func_type, rate_type)
        lb_max_x = np.maximum(min_x, 0.0) + 1e-4
        lb_max_y = np.maximum(min_y, 0.0) + 1e-4
        bnds = ((1e-12, None), (1e-12, None), (1e-12, None), (1e-12, None),
                (lb_max_x, 0.8), (lb_max_y, 0.8), (0, 1))
        params_til = opt.minimize(wsumsq, params_init, args=(tx_objs),
                                  method="L-BFGS-B", bounds=bnds, tol=1e-15)
        Atil, Btil, Ctil, Dtil, max_x, max_y, share = params_til.x
        # message = ("(max_x, min_x)=(" + str(max_x) + ", " + str(min_x) +
        #     "), (max_y, min_y)=(" + str(max_y) + ", " + str(min_y) + ")")
        # print(message)
        wsse = params_til.fun
        obs = df.shape[0]
        shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
        shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
        params = np.zeros(numparams)
        params[:4] = (np.array([Atil, Btil, Ctil, Dtil]) /
                      np.array([X2bar, Xbar, Y2bar, Ybar]))
        params[4:] = np.array([max_x, min_x, max_y, min_y, shift_x, shift_y,
                               shift, share])
        params_to_plot = np.append(params[:4],
                                   np.array([max_x, max_y, share, min_x,
                                             min_y, shift]))
    elif tax_func_type == 'DEP_totalinc':
        '''
        Estimate DeBacker, Evans, Phillips (2018) ratio of polynomial
        tax functions as a function of total income.
        '''
        Atil_init = 1.0
        Btil_init = 1.0
        max_x_init = np.minimum(
            txrates[(df['Total capital income'] < y_20pctl)].max(), 0.7)
        max_y_init = np.minimum(
            txrates[(df['Total labor income'] < x_20pctl)].max(), 0.7)
        max_income_init = max(max_x_init, max_y_init)
        min_income = min(min_x, min_y)
        shift = txrates[(df['Total labor income'] < x_20pctl) |
                        (df['Total capital income'] < y_20pctl)].min()
        share_init = 0.5
        params_init = np.array([Atil_init, Btil_init, max_income_init])
        tx_objs = (np.array([min_income, shift]), X, Y, txrates, wgts,
                   tax_func_type, rate_type)
        lb_max_income = np.maximum(min_income, 0.0) + 1e-4
        bnds = ((1e-12, None), (1e-12, None), (lb_max_income, 0.8))
        params_til = opt.minimize(wsumsq, params_init, args=(tx_objs),
                                  method="L-BFGS-B", bounds=bnds, tol=1e-15)
        Atil, Btil, max_income = params_til.x
        wsse = params_til.fun
        obs = df.shape[0]
        shift_income = (np.maximum(-min_income, 0.0) + 0.01 *
                        (max_income - min_income))
        params = np.zeros(numparams)
        params[:4] = (np.array([Atil, Btil, 0.0, 0.0]) /
                      np.array([income2bar, Ibar, Y2bar, Ybar]))
        params[4:] = np.array([max_income, min_income, 0.0, 0.0,
                               shift_income, 0.0, shift, 1.0])
        params_to_plot = np.append(params[:4],
                                   np.array([max_x, max_y, share, min_x,
                                             min_y, shift]))
    elif tax_func_type == "GS":
        '''
        Estimate Gouveia-Strauss parameters via least squares.
        Need to use a different functional form than for DEP function.
        '''
        phi0_init = 1.0
        phi1_init = 1.0
        phi2_init = 1.0
        params_init = np.array([phi0_init, phi1_init, phi2_init])
        tx_objs = (np.array([None]), X, Y, txrates, wgts, tax_func_type,
                   rate_type)
        bnds = ((1e-12, None), (1e-12, None), (1e-12, None))
        params_til = opt.minimize(wsumsq, params_init, args=(tx_objs),
                                  method="L-BFGS-B", bounds=bnds, tol=1e-15)
        phi0til, phi1til, phi2til = params_til.x
        wsse = params_til.fun
        obs = df.shape[0]
        params = np.zeros(numparams)
        params[:3] = np.array([phi0til, phi1til, phi2til])
        params_to_plot = params
    elif tax_func_type == "linear":
        '''
        For linear rates, just take the mean ETR or MTR by age-year.
        Can use DEP form and set all parameters except for the shift
        parameter to zero.
        '''
        params = np.zeros(numparams)
        wsse = 0.0
        obs = df.shape[0]
        params[10] = txrates.mean()
        params_to_plot = params[1:11]
    else:
        raise RuntimeError("Choice of tax function is not in the set of"
                           + " possible tax functions.  Please select"
                           + " from: DEP, DEP_totalinc, GS, linear.")
    if graph:
        '''
        ----------------------------------------------------------------
        cmap1       = color map object for matplotlib 3D plots
        tx_label    = string, text representing type of tax rate
        gridpts     = scalar > 2, number of grid points in X and Y
                      dimensions
        X_vec       = (gridpts,) vector, discretized log support of X
        Y_vec       = (gridpts,) vector, discretized log support of Y
        X_grid      = (gridpts, gridpts) matrix, ?
        Y_grid      = (gridpts, gridpts) matrix, ?
        txrate_grid = (gridpts, gridpts) matrix, ?
        filename    = string, name of plot to be saved
        fullpath    = string, full path name of file to be saved
        df_trnc_gph = (Nb, 11) DataFrame, truncated data for plotting
        X_gph       = (Nb,) Series, truncated labor income data
        Y_gph       = (Nb,) Series, truncated capital income data
        txrates_gph = (Nb,) Series, truncated tax rate (ETR, MTRx, or
                      MTRy) data
        ----------------------------------------------------------------
        '''
        cmap1 = matplotlib.cm.get_cmap('summer')

        # Make comparison plot with full income domains
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, txrates, c='r', marker='o')
        ax.set_xlabel('Total labor income')
        ax.set_ylabel('Total capital income')
        if rate_type == 'etr':
            tx_label = 'ETR'
        elif rate_type == 'mtrx':
            tx_label = 'MTRx'
        elif rate_type == 'mtry':
            tx_label = 'MTRy'
        ax.set_zlabel(tx_label)
        plt.title(tx_label + ' vs. Predicted ' + tx_label + ': Age=' +
                  str(s) + ', Year=' + str(t))

        gridpts = 50
        X_vec = np.exp(np.linspace(np.log(5), np.log(X.max()), gridpts))
        Y_vec = np.exp(np.linspace(np.log(5), np.log(Y.max()), gridpts))
        X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
        txrate_grid = get_tax_rates(params_to_plot, X_grid, Y_grid, None,
                                    tax_func_type, rate_type,
                                    for_estimation=False)
        ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1,
                        linewidth=0)
        filename = (tx_label + '_Age_' + str(s) + '_Year_' + str(t) +
                    '_vsPred.png')
        fullpath = os.path.join(output_dir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        # Make comparison plot with truncated income domains
        df_trnc_gph = df[(df['Total labor income'] > 5) &
                         (df['Total labor income'] < 800000) &
                         (df['Total capital income'] > 5) &
                         (df['Total capital income'] < 800000)]
        X_gph = df_trnc_gph['Total labor income']
        Y_gph = df_trnc_gph['Total capital income']
        if rate_type == 'etr':
            txrates_gph = df_trnc_gph['ETR']
        elif rate_type == 'mtrx':
            txrates_gph = df_trnc_gph['MTR labor income']
        elif rate_type == 'mtry':
            txrates_gph = df_trnc_gph['MTR capital income']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_gph, Y_gph, txrates_gph, c='r', marker='o')
        ax.set_xlabel('Total labor income')
        ax.set_ylabel('Total capital income')
        ax.set_zlabel(tx_label)
        plt.title('Truncated ' + tx_label + ', Lab. Inc., and Cap. ' +
                  'Inc., Age=' + str(s) + ', Year=' + str(t))

        gridpts = 50
        X_vec = np.exp(np.linspace(np.log(5), np.log(X_gph.max()),
                                   gridpts))
        Y_vec = np.exp(np.linspace(np.log(5), np.log(Y_gph.max()),
                                   gridpts))
        X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
        txrate_grid = get_tax_rates(params_to_plot, X_grid, Y_grid, None,
                                    tax_func_type, rate_type,
                                    for_estimation=False)
        ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1,
                        linewidth=0)
        filename = (tx_label + 'trunc_Age_' + str(s) + '_Year_' +
                    str(t) + '_vsPred.png')
        fullpath = os.path.join(output_dir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        del df_trnc_gph

    # Garbage collection
    del df, txrates

    return params, wsse, obs


def tax_func_loop(t, micro_data, beg_yr, s_min, s_max, age_specific,
                  tax_func_type, analytical_mtrs, desc_data, graph_data,
                  graph_est, output_dir, numparams, tpers):
    '''
    ----------------------------------------------------------------
    Clean up the data
    ----------------------------------------------------------------
    data_orig  = (N1, 11) DataFrame, original micro tax data from
                 Tax-Calculator for particular year
    data       = (N1, 8) DataFrame, new variables dataset
    data_trnc  = (N2, 8) DataFrame, truncated observations dataset
    min_age    = integer >= 1, minimum age in micro data that is
                 relevant to model
    max_age    = integer >= min_age, maximum age in micro data that
                 is relevant to model
    NoData_cnt = integer >= 0, number of consecutive ages with
                 insufficient data to estimate tax functions
    ----------------------------------------------------------------
    '''
    # initialize arrays for output
    etrparam_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
    mtrxparam_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
    mtryparam_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
    etr_wsumsq_arr = np.zeros((s_max - s_min + 1, tpers))
    etr_obs_arr = np.zeros((s_max - s_min + 1, tpers))
    mtrx_wsumsq_arr = np.zeros((s_max - s_min + 1, tpers))
    mtrx_obs_arr = np.zeros((s_max - s_min + 1, tpers))
    mtry_wsumsq_arr = np.zeros((s_max - s_min + 1, tpers))
    mtry_obs_arr = np.zeros((s_max - s_min + 1, tpers))
    AvgInc = np.zeros(tpers)
    AvgETR = np.zeros(tpers)
    AvgMTRx = np.zeros(tpers)
    AvgMTRy = np.zeros(tpers)
    TotPop_yr = np.zeros(tpers)
    PopPct_age = np.zeros((s_max - s_min + 1, tpers))

    micro_data['Total labor income'] = \
        (micro_data['Wage income'] + micro_data['SE income'])
    micro_data['ETR'] = \
        (micro_data['Total tax liability'] /
         micro_data["Adjusted total income"])
    micro_data["Total capital income"] = \
        (micro_data['Adjusted total income'] -
         micro_data['Total labor income'])
    # use weighted avg for MTR labor - abs value because
    # SE income may be negative
    micro_data['MTR labor income'] = (
        micro_data['MTR wage income'] * (micro_data['Wage income'] /
                                         (micro_data['Wage income'].abs()
                                          +
                                          micro_data['SE income'].abs()))
        + micro_data['MTR SE income'] * (micro_data['SE income'].abs() /
                                         (micro_data['Wage income'].abs()
                                          +
                                          micro_data['SE income'].abs())))

    data = micro_data[['Age', 'MTR labor income', 'MTR capital income',
                       'Total labor income', 'Total capital income',
                       'Adjusted total income', 'ETR', 'Weights']].copy()

    del micro_data

    # Calculate average total income in each year
    AvgInc[t-beg_yr] = \
        (((data['Adjusted total income'] * data['Weights']).sum())
         / data['Weights'].sum())

    # Calculate average ETR and MTRs (weight by population weights
    #    and income) for each year
    AvgETR[t-beg_yr] = \
        (((data['ETR']*data['Adjusted total income']
           * data['Weights']).sum()) /
         (data['Adjusted total income']*data['Weights']).sum())

    AvgMTRx[t-beg_yr] = \
        (((data['MTR labor income']*data['Adjusted total income'] *
           data['Weights']).sum()) /
         (data['Adjusted total income']*data['Weights']).sum())

    AvgMTRy[t-beg_yr] = \
        (((data['MTR capital income'] *
           data['Adjusted total income'] * data['Weights']).sum()) /
         (data['Adjusted total income']*data['Weights']).sum())

    # Calculate total population in each year
    TotPop_yr[t-beg_yr] = data['Weights'].sum()

    # Clean up the data by dropping outliers
    # drop all obs with ETR > 0.65
    data.drop(data[data['ETR'] > 0.65].index, inplace=True)
    # drop all obs with ETR < -0.15
    data.drop(data[data['ETR'] < -0.15].index, inplace=True)
    # drop all obs with ATI, TLI, TCincome< $5
    data.drop(data[(data['Adjusted total income'] < 5) |
                   (data['Total labor income'] < 5) |
                   (data['Total capital income'] < 5)].index,
              inplace=True)
    # drop all obs with MTR on capital income > 0.99
    data.drop(data[data['MTR capital income'] > 0.99].index,
              inplace=True)
    # drop all obs with MTR on capital income < -0.45
    data.drop(data[data['MTR capital income'] < -0.45].index,
              inplace=True)
    # drop all obs with MTR on labor income > 0.99
    data.drop(data[data['MTR labor income'] > 0.99].index, inplace=True)
    # drop all obs with MTR on labor income < -0.45
    data.drop(data[data['MTR labor income'] < -0.45].index, inplace=True)

    # Create an array of the different ages in the data
    min_age = int(np.maximum(data['Age'].min(), s_min))
    max_age = int(np.minimum(data['Age'].max(), s_max))
    if age_specific:
        ages_list = np.arange(min_age, max_age + 1)
    else:
        ages_list = np.arange(0, 1)

    NoData_cnt = np.min(min_age - s_min, 0)

    # Each age s must be done in serial
    for s in ages_list:
        if age_specific:
            print("year=", t, "Age=", s)
            df = data[data['Age'] == s]
            PopPct_age[s-min_age, t-beg_yr] = \
                df['Weights'].sum() / TotPop_yr[t-beg_yr]

        else:
            print("year=", t, "Age= all ages")
            df = data
            PopPct_age[0, t-beg_yr] = \
                df['Weights'].sum() / TotPop_yr[t-beg_yr]
        df_etr = df.loc[df[
            (np.isfinite(df['ETR'])) &
            (np.isfinite(df['Total labor income'])) &
            (np.isfinite(df['Total capital income'])) &
            (np.isfinite(df['Weights']))].index,
                        ['MTR labor income', 'MTR capital income',
                         'Total labor income', 'Total capital income',
                         'ETR', 'Weights']].copy()
        df_mtrx = df.loc[df[
            (np.isfinite(df['MTR labor income'])) &
            (np.isfinite(df['Total labor income'])) &
            (np.isfinite(df['Total capital income'])) &
            (np.isfinite(df['Weights']))].index,
                         ['MTR labor income', 'Total labor income',
                          'Total capital income', 'Weights']].copy()
        df_mtry = df.loc[df[
            (np.isfinite(df['MTR capital income'])) &
            (np.isfinite(df['Total labor income'])) &
            (np.isfinite(df['Total capital income'])) &
            (np.isfinite(df['Weights']))].index,
                         ['MTR capital income', 'Total labor income',
                          'Total capital income', 'Weights']].copy()
        df_minobs = np.min([df_etr.shape[0], df_mtrx.shape[0],
                            df_mtry.shape[0]])
        del df
        # 240 is 8 parameters to estimate times 30 obs per parameter
        if df_minobs < 240 and s < max_age:
            '''
            --------------------------------------------------------
            Don't estimate function on this iteration if obs < 500.
            Will fill in later with interpolated values
            --------------------------------------------------------
            '''
            message = ("Insuff. sample size for age " + str(s) +
                       " in year " + str(t))
            print(message)
            NoData_cnt += 1
            etrparam_arr[s-s_min, t-beg_yr, :] = np.nan
            mtrxparam_arr[s-s_min, t-beg_yr, :] = np.nan
            mtryparam_arr[s-s_min, t-beg_yr, :] = np.nan

        elif df_minobs < 240 and s == max_age:
            '''
            --------------------------------------------------------
            If last period does not have sufficient data, fill in
            final missing age data with last positive year
            --------------------------------------------------------
            lastp_etr  = (numparams,) vector, vector of parameter
                         estimates from previous age with sufficient
                         observations
            lastp_mtrx = (numparams,) vector, vector of parameter
                         estimates from previous age with sufficient
                         observations
            lastp_mtry = (numparams,) vector, vector of parameter
                         estimates from previous age with sufficient
                         observations
            --------------------------------------------------------
            '''
            message = ("Max age (s=" + str(s) + ") insuff. data in"
                       + " year " + str(t) +
                       ". Fill in final ages with " +
                       "insuff. data with most recent successful " +
                       "estimate.")
            print(message)
            NoData_cnt += 1
            lastp_etr = \
                etrparam_arr[s-NoData_cnt-s_min, t-beg_yr, :]
            etrparam_arr[s-NoData_cnt-s_min+1:, t-beg_yr, :] = \
                np.tile(lastp_etr.reshape((1, numparams)),
                        (NoData_cnt+s_max-max_age, 1))
            lastp_mtrx = \
                mtrxparam_arr[s-NoData_cnt-s_min, t-beg_yr, :]
            mtrxparam_arr[s-NoData_cnt-s_min+1:, t-beg_yr, :] = \
                np.tile(lastp_mtrx.reshape((1, numparams)),
                        (NoData_cnt+s_max-max_age, 1))
            lastp_mtry = \
                mtryparam_arr[s-NoData_cnt-s_min, t-beg_yr, :]
            mtryparam_arr[s-NoData_cnt-s_min+1:, t-beg_yr, :] = \
                np.tile(lastp_mtry.reshape((1, numparams)),
                        (NoData_cnt+s_max-max_age, 1))

        else:
            # Estimate parameters for age with sufficient data
            if desc_data:
                # print some desciptive stats
                message = ("Descriptive ETR statistics for age=" +
                           str(s) + " in year " + str(t))
                print(message)
                print(df_etr.describe())
                message = ("Descriptive MTRx statistics for age=" +
                           str(s) + " in year " + str(t))
                print(message)
                print(df_mtrx.describe())
                message = ("Descriptive MTRy statistics for age=" +
                           str(s) + " in year " + str(t))
                print(message)
                print(df_mtry.describe())

            if graph_data:
                gen_3Dscatters_hist(df_etr, s, t, output_dir)

            # Estimate effective tax rate function ETR(x,y)
            (etrparams, etr_wsumsq_arr[s-s_min, t-beg_yr],
                etr_obs_arr[s-s_min, t-beg_yr]) = \
                txfunc_est(df_etr, s, t, 'etr', tax_func_type,
                           numparams, output_dir, graph_est)
            etrparam_arr[s-s_min, t-beg_yr, :] = etrparams
            del df_etr

            # Estimate marginal tax rate of labor income function
            # MTRx(x,y)
            (mtrxparams, mtrx_wsumsq_arr[s-s_min, t-beg_yr],
                mtrx_obs_arr[s-s_min, t-beg_yr]) = \
                txfunc_est(df_mtrx, s, t, 'mtrx', tax_func_type,
                           numparams, output_dir, graph_est)
            mtrxparam_arr[s-s_min, t-beg_yr, :] = mtrxparams
            del df_mtrx
            # Estimate marginal tax rate of capital income function
            # MTRy(x,y)
            (mtryparams, mtry_wsumsq_arr[s-s_min, t-beg_yr],
                mtry_obs_arr[s-s_min, t-beg_yr]) = \
                txfunc_est(df_mtry, s, t, 'mtry', tax_func_type,
                           numparams, output_dir, graph_est)
            mtryparam_arr[s-s_min, t-beg_yr, :] = mtryparams

            del df_mtry

            if NoData_cnt > 0 & NoData_cnt == s-s_min:
                '''
                ----------------------------------------------------
                Fill in initial blanks with first positive data
                estimates. This includes the case in which
                min_age > s_min
                ----------------------------------------------------
                '''
                message = "Fill in all previous blank ages"
                print(message)
                etrparam_arr[:s-s_min, t-beg_yr, :] = \
                    np.tile(etrparams.reshape((1, numparams)),
                            (s-s_min, 1))
                mtrxparam_arr[:s-s_min, t-beg_yr, :] = \
                    np.tile(mtrxparams.reshape((1, numparams)),
                            (s-s_min, 1))
                mtryparam_arr[:s-s_min, t-beg_yr, :] = \
                    np.tile(mtryparams.reshape((1, numparams)),
                            (s-s_min, 1))

            elif NoData_cnt > 0 & NoData_cnt < s-s_min:
                '''
                ----------------------------------------------------
                Fill in interior data gaps with linear interpolation
                between bracketing positive data ages. In all of
                these cases min_age < s <= max_age.
                ----------------------------------------------------
                tvals        = (NoData_cnt+2,) vector, linearly
                               space points between 0 and 1
                x0_etr       = (NoData_cnt x 10) matrix, positive
                               estimates at beginning of no data
                               spell
                x1_etr       = (NoData_cnt x 10) matrix, positive
                               estimates at end (current period) of
                               no data spell
                lin_int_etr  = (NoData_cnt x 10) matrix, linearly
                               interpolated etr parameters between
                               x0_etr and x1_etr
                x0_mtrx      = (NoData_cnt x 10) matrix, positive
                               estimates at beginning of no data
                               spell
                x1_mtrx      = (NoData_cnt x 10) matrix, positive
                               estimates at end (current period) of
                               no data spell
                lin_int_mtrx = (NoData_cnt x 10) matrix, linearly
                               interpolated mtrx parameters between
                               x0_mtrx and x1_mtrx
                ----------------------------------------------------
                '''
                message = ("Linearly interpolate previous blank " +
                           "tax functions")
                print(message)
                tvals = np.linspace(0, 1, NoData_cnt+2)
                x0_etr = np.tile(
                    etrparam_arr[s-NoData_cnt-s_min-1,
                                 t-beg_yr, :].reshape((1, numparams)),
                    (NoData_cnt, 1))
                x1_etr = np.tile(etrparams.reshape((1, numparams)),
                                 (NoData_cnt, 1))
                lin_int_etr = \
                    (x0_etr + tvals[1:-1].reshape((NoData_cnt, 1)) *
                     (x1_etr - x0_etr))
                etrparam_arr[s-NoData_cnt-min_age:s-min_age,
                             t-beg_yr, :] = lin_int_etr
                x0_mtrx = np.tile(
                    mtrxparam_arr[s-NoData_cnt-s_min-1,
                                  t-beg_yr, :].reshape((1, numparams)),
                    (NoData_cnt, 1))
                x1_mtrx = np.tile(
                    mtrxparams.reshape((1, numparams)),
                    (NoData_cnt, 1))
                lin_int_mtrx = \
                    (x0_mtrx + tvals[1:-1].reshape((NoData_cnt, 1)) *
                     (x1_mtrx - x0_mtrx))
                mtrxparam_arr[s-NoData_cnt-min_age:s-min_age,
                              t-beg_yr, :] = lin_int_mtrx
                x0_mtry = np.tile(
                    mtryparam_arr[s-NoData_cnt-s_min-1,
                                  t-beg_yr, :].reshape((1, numparams)),
                    (NoData_cnt, 1))
                x1_mtry = np.tile(
                    mtryparams.reshape((1, numparams)),
                    (NoData_cnt, 1))
                lin_int_mtry = \
                    (x0_mtry + tvals[1:-1].reshape((NoData_cnt, 1)) *
                     (x1_mtry - x0_mtry))
                mtryparam_arr[s-NoData_cnt-min_age:s-min_age,
                              t-beg_yr, :] = lin_int_mtry

            NoData_cnt == 0

            if s == max_age and max_age < s_max:
                '''
                ----------------------------------------------------
                If the last age estimates, and max_age< s_max, fill
                in the remaining ages with these last estimates
                ----------------------------------------------------
                '''
                message = "Fill in all old tax functions."
                print(message)
                etrparam_arr[s-s_min+1:, t-beg_yr, :] = \
                    np.tile(etrparams.reshape((1, numparams)),
                            (s_max-max_age, 1))
                mtrxparam_arr[s-s_min+1:, t-beg_yr, :] = \
                    np.tile(mtrxparams.reshape((1, numparams)),
                            (s_max-max_age, 1))
                mtryparam_arr[s-s_min+1:, t-beg_yr, :] = \
                    np.tile(mtryparams.reshape((1, numparams)),
                            (s_max-max_age, 1))

    return (TotPop_yr[t-beg_yr], PopPct_age[:, t-beg_yr],
            AvgInc[t-beg_yr], AvgETR[t-beg_yr], AvgMTRx[t-beg_yr],
            AvgMTRy[t-beg_yr], etrparam_arr[:, t-beg_yr, :],
            etr_wsumsq_arr[:, t-beg_yr], etr_obs_arr[:, t-beg_yr],
            mtrxparam_arr[:, t-beg_yr, :], mtrx_wsumsq_arr[:, t-beg_yr],
            mtrx_obs_arr[:, t-beg_yr], mtryparam_arr[:, t-beg_yr, :],
            mtry_wsumsq_arr[:, t-beg_yr], mtry_obs_arr[:, t-beg_yr])


def tax_func_estimate(BW, S, starting_age, ending_age,
                      beg_yr=DEFAULT_START_YEAR, baseline=True,
                      analytical_mtrs=False, tax_func_type='DEP',
                      age_specific=False, reform={}, data=None,
                      client=None, num_workers=1):
    '''
    --------------------------------------------------------------------
    This function performs analysis on the source data from Tax-
    Calculator and estimates functions for the effective tax rate (ETR),
    marginal tax rate on labor income (MTRx), and marginal tax rate on
    capital income (MTRy).
    --------------------------------------------------------------------
    INPUTS:
    beg_yr          = integer >= 2016, current year for analysis
    baseline        = Boolean, =True performs baseline analysis in
                      getting tax micro data from Tax-Calculator
    analytical_mtrs = Boolean, =True if use analytical_mtrs, =False if
                      use estimated MTRs
    age_specific    = Boolean, =True calculates tax functions for each
                      age, =False calculates an average over all ages
    reform          = dictionary, Tax-Calculator reform values to be
                      passed in to get_micro_data.get_data() function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        utils.mkdirs()
        get_micro_data.get_data()

    OBJECTS CREATED WITHIN FUNCTION:
    (See comments within this function)
    dict_params = dictionary,

    RETURNS: dict_params
    --------------------------------------------------------------------
    Set parameters and create objects for output
    --------------------------------------------------------------------
    S               = integer >= 3, number of periods an individual can
                      live. S represents the lifespan in years between
                      s_min and s_max
    tpers           = integer >= 1, number of years in budget window to
                      estimate tax functions
    s_min           = integer > 0, minimum economically active age
                      (years)
    s_max           = integer > s_min, maximum age (years)
    end_yr          = integer >= beg_yr, ending year for analysis
    numparams       = integer > 0, number of parameters to estimate for
                      the tax function for each age s and year t
    desc_data       = Boolean, =True if print descriptive stats for data
                      each age (s) and year (t)
    graph_data      = Boolean, =True if print 3D scatterplots of data
                      for each age (s) and year (t)
    graph_est       = Boolean, =True if print 3D plot of data and
                      estimated tax function for each age s and year t
    etrparam_arr    = (s_max-s_min+1, tpers, numparams) array, parameter
                      values for the estimated effective tax rate
                      functions for each age s and year t
    mtrxparam_arr   = (s_max-s_min+1, tpers, numparams) array, parameter
                      values for the estimated marginal tax rate funcs
                      of labor income for each age s and year t
    mtryparam_arr   = (s_max-s_min+1, tpers, numparams) array, parameter
                      values for the estimated marginal tax rate funcs
                      of capital income for each age s and year t
    etr_wsumsq_arr  = (s_max-s_min+1, tpers) matrix, weighted sum of
                      squared deviations from ETR estimations for each
                      year (t) and age (s)
    etr_obs_arr     = (s_max-s_min+1, tpers) matrix, number of
                      observations for ETR estimations for each year (t)
                      and age (s)
    mtrx_wsumsq_arr = (s_max-s_min+1, tpers) matrix, weighted sum of
                      squared deviations from MTRx estimations for each
                      year (t) and age (s)
    mtrx_obs_arr    = (s_max-s_min+1, tpers) matrix, number of
                      observations for MTRx estimations for each year
                      (t) and age (s)
    mtry_wsumsq_arr = (s_max-s_min+1, tpers) matrix, weighted sum of
                      squared deviations from MTRy estimations for each
                      year (t) and age (s)
    mtry_obs_arr    = (s_max-s_min+1, tpers) matrix, number of
                      observations for MTRy estimations for each year
                      (t) and age (s)
    AvgInc          = (tpers,) vector, average income in each year
    AvgETR          = (tpers,) vector, average ETR in each year
    AvgMTRx         = (tpers,) vector, average MTRx in each year
    AvgMTRy         = (tpers,) vector, average MTRy in each year
    TotPop_yr       = (tpers,) vector, total population according to
                      weights variable in each year
    PopPct_age      = (s_max-s_min+1, tpers) matrix, population percent
                      of each age in each year
    years_list      = [beg_yr-end_yr+1,] vector, iterable list of years
                      to be forecast
    --------------------------------------------------------------------
    '''
    tpers = BW
    s_min = starting_age + 1
    s_max = ending_age
    beg_yr = int(beg_yr)
    end_yr = int(beg_yr + tpers - 1)
    print('BW = ', BW, "begin year = ", beg_yr, "end year = ", end_yr)
    numparams = int(12)
    desc_data = False
    graph_data = False
    graph_est = False
    years_list = np.arange(beg_yr, end_yr + 1)
    if age_specific:
        ages_list = np.arange(s_min, s_max+1)
    else:
        ages_list = np.arange(0, 1)
    # initialize arrays for output
    etrparam_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
    mtrxparam_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
    mtryparam_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
    etr_wsumsq_arr = np.zeros((s_max - s_min + 1, tpers))
    etr_obs_arr = np.zeros((s_max - s_min + 1, tpers))
    mtrx_wsumsq_arr = np.zeros((s_max - s_min + 1, tpers))
    mtrx_obs_arr = np.zeros((s_max - s_min + 1, tpers))
    mtry_wsumsq_arr = np.zeros((s_max - s_min + 1, tpers))
    mtry_obs_arr = np.zeros((s_max - s_min + 1, tpers))
    AvgInc = np.zeros(tpers)
    AvgETR = np.zeros(tpers)
    AvgMTRx = np.zeros(tpers)
    AvgMTRy = np.zeros(tpers)
    TotPop_yr = np.zeros(tpers)
    PopPct_age = np.zeros((s_max-s_min+1, tpers))

    '''
    --------------------------------------------------------------------
    Solve for tax functions for each year (t) and each age (s)
    --------------------------------------------------------------------
    start_time = scalar, current processor time in seconds (float)
    output_dir = string, directory to which plots will be saved
    micro_data = dictionary, tpers (one for each year) DataFrames,
                 each of which has variables with observations from
                 Tax-Calculator
    t          = integer >= beg_yr, index for year of analysis
    --------------------------------------------------------------------
    '''
    start_time = time.time()
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "OUTPUT/TaxFunctions"
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # call tax caculator and get microdata
    micro_data, taxcalc_version = get_micro_data.get_data(
        baseline=baseline, start_year=beg_yr, reform=reform, data=data,
        client=client, num_workers=num_workers)

    lazy_values = []
    for t in years_list:
        lazy_values.append(
            delayed(tax_func_loop)(t, micro_data[str(t)], beg_yr, s_min,
                                   s_max, age_specific, tax_func_type,
                                   analytical_mtrs, desc_data,
                                   graph_data, graph_est, output_dir,
                                   numparams, tpers))
    results = compute(*lazy_values, scheduler=dask.multiprocessing.get,
                      num_workers=num_workers)

    # Garbage collection
    del micro_data

    # for i, result in results.items():
    for i, result in enumerate(results):
        (TotPop_yr[i], PopPct_age[:, i], AvgInc[i],
         AvgETR[i], AvgMTRx[i], AvgMTRy[i],
         etrparam_arr[:, i, :], etr_wsumsq_arr[:, i],
         etr_obs_arr[:, i], mtrxparam_arr[:, i, :],
         mtrx_wsumsq_arr[:, i], mtrx_obs_arr[:, i],
         mtryparam_arr[:, i, :], mtry_wsumsq_arr[:, i],
         mtry_obs_arr[:, i]) = result

    message = ("Finished tax function loop through " +
               str(len(years_list)) + " years and " + str(len(ages_list)) +
               " ages per year.")
    print(message)
    elapsed_time = time.time() - start_time

    # Print tax function computation time
    if elapsed_time < 60:  # less than a minute
        secs = round(elapsed_time, 3)
        message = "Tax function estimation time: " + str(secs) + " sec"
        print(message)
    elif elapsed_time >= 60 and elapsed_time < 3600:  # less than hour
        mins = int(elapsed_time / 60)
        secs = round(((elapsed_time / 60) - mins) * 60, 1)
        message = ("Tax function estimation time: " + str(mins) +
                   " min, " + str(secs) + " sec")
        print(message)
    elif elapsed_time >= 3600 and elapsed_time < 86400:  # less than day
        hours = int(elapsed_time / (60 * 60))
        mins = int((elapsed_time - (hours * 60 * 60)) / 60)
        secs = round(elapsed_time - (hours * 60 * 60) - (mins * 60), 1)
        message = ("Tax function estimation time: " + str(hours) +
                   " hour(s), " + str(mins) + " min(s), " + str(secs) +
                   " sec(s)")
        print(message)

    '''
    --------------------------------------------------------------------
    Replace outlier tax functions (SSE>mean+2.5*std) with linear
    linear interpolation. We make two passes (filtering runs).
    --------------------------------------------------------------------
    '''
    if age_specific:
        age_sup = np.linspace(s_min, s_max, s_max-s_min+1)
        se_mult = 3.5
        etr_sse_big = find_outliers(etr_wsumsq_arr / etr_obs_arr,
                                    age_sup, se_mult, beg_yr, "ETR")
        if etr_sse_big.sum() > 0:
            etrparam_arr_adj = replace_outliers(etrparam_arr,
                                                etr_sse_big)
        elif etr_sse_big.sum() == 0:
            etrparam_arr_adj = etrparam_arr

        mtrx_sse_big = find_outliers(mtrx_wsumsq_arr / mtrx_obs_arr,
                                     age_sup, se_mult, beg_yr, "MTRx")
        if mtrx_sse_big.sum() > 0:
            mtrxparam_arr_adj = replace_outliers(mtrxparam_arr,
                                                 mtrx_sse_big)
        elif mtrx_sse_big.sum() == 0:
            mtrxparam_arr_adj = mtrxparam_arr

        mtry_sse_big = find_outliers(mtry_wsumsq_arr / mtry_obs_arr,
                                     age_sup, se_mult, beg_yr, "MTRy")
        if mtry_sse_big.sum() > 0:
            mtryparam_arr_adj = replace_outliers(mtryparam_arr,
                                                 mtry_sse_big)
        elif mtry_sse_big.sum() == 0:
            mtryparam_arr_adj = mtryparam_arr

    '''
    --------------------------------------------------------------------
    Generate tax function parameters for S < s_max - s_min + 1
    --------------------------------------------------------------------
    etrparam_arr_S  = S x tpers x 10 array, this is an array in which S
                      is less-than-or-equal-to s_max-s_min+1. We use
                      weighted averages of parameters in relevant age
                      groups
    mtrxparam_arr_S = S x tpers x 10 array, this is an array in which S
                      is less-than-or-equal-to s_max-s_min+1. We use
                      weighted averages of parameters in relevant age
                      groups
    age_cuts     = (S+1,) vector, linspace of age cutoffs of S+1 points
                   between 0 and S+1
    yrcut_lb     = integer >= 0, index of lower bound age for S bin
    yrcut_ub     = integer >= 0, index of upper bound age for S bin
    rmndr_pct_lb = scalar in [0,1], discounted weight on lower bound age
    rmndr_pct_ub = scalar in [0,1], discounted weight on upper bound age
    age_wgts     = ages x tpers x 10 array, age weights for each age in
                   each year copied back 10 times in the 3rd dimension
    --------------------------------------------------------------------
    '''
    if age_specific:
        if S == s_max - s_min + 1:
            etrparam_arr_S = etrparam_arr_adj
            mtrxparam_arr_S = mtrxparam_arr_adj
            mtryparam_arr_S = mtryparam_arr_adj
        elif S < s_max - s_min + 1:
            etrparam_arr_S = etrparam_arr_adj
            mtrxparam_arr_S = mtrxparam_arr_adj
            mtryparam_arr_S = mtryparam_arr_adj
            etrparam_arr_S = np.zeros((S, tpers, numparams))
            mtrxparam_arr_S = np.zeros((S, tpers, numparams))
            mtryparam_arr_S = np.zeros((S, tpers, numparams))
            age_cuts = np.linspace(0, s_max - s_min + 1, S + 1)
            yrcut_lb = int(age_cuts[0])
            rmndr_pct_lb = 1.
            for s in np.arange(S):
                yrcut_ub = int(np.floor(age_cuts[s + 1]))
                rmndr_pct_ub = (age_cuts[s + 1] -
                                np.floor(age_cuts[s + 1]))
                if rmndr_pct_ub == 0.:
                    rmndr_pct_ub = 1.
                    yrcut_ub -= 1
                age_wgts = np.dstack(
                    [PopPct_age[yrcut_lb:yrcut_ub + 1, :]] * numparams)
                age_wgts[0, :, :] *= rmndr_pct_lb
                age_wgts[yrcut_ub-yrcut_lb, :, :] *= rmndr_pct_ub
                etrparam_arr_S[s, :, :] = (
                    etrparam_arr_adj[yrcut_lb:yrcut_ub + 1, :, :] *
                    age_wgts).sum(axis=0)
                mtrxparam_arr_S[s, :, :] = (
                    mtrxparam_arr_adj[yrcut_lb:yrcut_ub + 1, :, :] *
                    age_wgts).sum(axis=0)
                mtryparam_arr_S[s, :, :] = (
                    mtryparam_arr_adj[yrcut_lb:yrcut_ub + 1, :, :] *
                    age_wgts).sum(axis=0)
                yrcut_lb = yrcut_ub
                rmndr_pct_lb = 1 - rmndr_pct_ub
        else:
            print('S is larger than the difference between the minimum'
                  + ' age and the maximum age specified.  Please choose'
                  + ' and S such that a model period equals at least'
                  + ' one calendar year.')

        print('Big S: ', S)
        print('max age, min age: ', s_max, s_min)
    else:
        etrparam_arr_S = np.tile(np.reshape(etrparam_arr[0-s_min, :, :],
                                            (1, tpers,
                                             etrparam_arr.shape[2])),
                                 (S, 1, 1))
        mtrxparam_arr_S = np.tile(
            np.reshape(mtrxparam_arr[0-s_min, :, :],
                       (1, tpers, mtrxparam_arr.shape[2])), (S, 1, 1))
        mtryparam_arr_S = np.tile(
            np.reshape(mtryparam_arr[0-s_min, :, :],
                       (1, tpers, mtryparam_arr.shape[2])), (S, 1, 1))

    # Save tax function parameters array and computation time in
    # dictionary
    dict_params = dict([('tfunc_etr_params_S', etrparam_arr_S),
                        ('tfunc_mtrx_params_S', mtrxparam_arr_S),
                        ('tfunc_mtry_params_S', mtryparam_arr_S),
                        ('tfunc_avginc', AvgInc),
                        ('tfunc_avg_etr', AvgETR),
                        ('tfunc_avg_mtrx', AvgMTRx),
                        ('tfunc_avg_mtry', AvgMTRy),
                        ('tfunc_etr_sumsq', etr_wsumsq_arr),
                        ('tfunc_mtrx_sumsq', mtrx_wsumsq_arr),
                        ('tfunc_mtry_sumsq', mtry_wsumsq_arr),
                        ('tfunc_etr_obs', etr_obs_arr),
                        ('tfunc_mtrx_obs', mtrx_obs_arr),
                        ('tfunc_mtry_obs', mtry_obs_arr),
                        ('tfunc_time', elapsed_time),
                        ('tax_func_type', tax_func_type),
                        ('taxcalc_version', taxcalc_version)])

    return dict_params


def get_tax_func_estimate(BW, S, starting_age, ending_age,
                          baseline=False, analytical_mtrs=False,
                          tax_func_type='DEP', age_specific=False,
                          start_year=DEFAULT_START_YEAR, reform={},
                          guid='', tx_func_est_path=None, data=None,
                          client=None, num_workers=1):
    '''
    --------------------------------------------------------------------
    This function calls the tax function estimation routine and saves
    the resulting dictionary in pickle files corresponding to the
    baseline or reform policy.
    --------------------------------------------------------------------

    INPUTS:
    baseline        = boolean, =True if baseline tax policy, =False if
                      reform
    analytical_mtrs = boolean, =True if use analytical_mtrs, =False if
                      use estimated MTRs
    age_specific    = boolean, =True if estimate tax functions
                      separately for each age, =False if estimate a
                      single tax function to represent all ages in a
                      given budget year
    start_year      = integer, first year of budget window
    reform          = dictionary, reform parameters
    guid            = string, id for reform run
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    tax_func_estimate()
    OBJECTS CREATED WITHIN FUNCTION:
    RETURNS: N/A

    OUTPUT:
    ./TAX_ESTIMATE_PATH/TxFuncEst_baseline{}.pkl
    ./TAX_ESTIMATE_PATH/TxFuncEst_policy{}.pkl
    --------------------------------------------------------------------
    '''
    # Code to run manually from here:
    dict_params = tax_func_estimate(BW, S, starting_age, ending_age,
                                    start_year, baseline,
                                    analytical_mtrs, tax_func_type,
                                    age_specific, reform, data=data,
                                    client=client,
                                    num_workers=num_workers)
    if baseline:
        baseline_pckl = (tx_func_est_path or
                         "TxFuncEst_baseline{}.pkl".format(guid))
        pkl_path = os.path.join(baseline_pckl)
    else:
        policy_pckl = (tx_func_est_path or
                       "TxFuncEst_policy{}.pkl".format(guid))
        pkl_path = os.path.join(policy_pckl)

    pickle.dump(dict_params, open(pkl_path, "wb"))
