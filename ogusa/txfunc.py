'''
------------------------------------------------------------------------
This script reads in data generated from the OSPC Tax Calculator and
the 2009 IRS PUF. It then estimates tax functions tau_{s,t}(x,y), where
tau_{s,t} is the effective tax rate, marginal tax rate on labor income,
or the marginal tax rate on capital income, for a given age (s) in a
particular year (t). x is total labor income, and y is total capital
income.

This module defines the following functions:
    gen_rate_grid()
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
import numpy.random as rnd
import scipy.optimize as opt
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import get_micro_data
import utils

TAX_ESTIMATE_PATH = os.environ.get("TAX_ESTIMATE_PATH", ".")

'''
------------------------------------------------------------------------
Define Functions
------------------------------------------------------------------------
'''

def gen_rate_grid(X, Y, params):
    '''
    --------------------------------------------------------------------
    This function generates a grid of tax rates (ETR, MTRx, or MTRy)
    from a grid of total labor income (X) and a grid of total capital
    income (Y).

    tau(X) = (max_x - min_x) * ((AX^2 + BX) / (AX^2 + BX + 1)) + min_x

    tau(Y) = (max_y - min_y) * ((CY^2 + DY) / (CY^2 + DY + 1)) + min_y

    rate(X,Y) =
    ((tau(X) + shift_x)^share) * ((tau(Y) + shift_y)^(1-share))) + shift
    --------------------------------------------------------------------
    INPUTS:
    X       = (N, N) matrix, discretized support (N elements) of labor
              income as row vector copied down N rows
    Y       = (N, N) matrix, discretized support (N elements) of capital
              income as column vector copied across N columns
    params  = (12,) vector, estimated parameters (A, B, C, D, max_x,
              min_x, max_y, min_y, shift_x, shift_y, shift, share)
    A       = scalar > 0, polynomial coefficient on X**2
    B       = scalar > 0, polynomial coefficient on X
    C       = scalar > 0, polynomial coefficient on Y**2
    D       = scalar > 0, polynomial coefficient on Y
    max_x   = scalar > 0, maximum tax rate for X given Y=0
    min_x   = scalar, minimum effective tax rate for X given Y=0
    max_y   = scalar > 0, maximum effective tax rate for Y given X=0
    min_y   = scalar, minimum effective tax rate for Y given X=0
    shift_x = scalar, adds to tau(X) to assure value in rate(X,Y) term
              is strictly positive
    shift_y = scalar, adds to tau(Y) to assure value in rate(X,Y) term
              is strictly positive
    shift   = scalar, adds to the Cobb-Douglas function to capture
              negative tax rates
    share   = scalar in [0, 1], share parameter in Cobb-Douglas function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    X2        = (N, N) matrix, X squared
    Y2        = (N, N) matrix, Y squared
    tau_x     = (N, N) matrix, ratio of polynomials function tau(X)
                evaluated at grid points X
    tau_x     = (N, N) matrix, ratio of polynomials function tau(Y)
                evaluated at grid points Y
    rate_grid = (N, N) matrix, predicted tax rates given labor income
                grid (X) and capital income grid (Y)

    RETURNS: rate_grid
    --------------------------------------------------------------------
    '''
    (A, B, C, D, max_x, min_x, max_y, min_y, shift_x, shift_y, shift,
        share) = params
    X2 = X ** 2
    Y2 = Y ** 2
    tau_x = (((max_x - min_x) * (A * X2 + B * X) / (A * X2 + B * X + 1))
        + min_x)
    tau_y = (((max_y - min_y) * (C * Y2 + D * Y) / (C * Y2 + D * Y + 1))
        + min_y)
    rate_grid = (((tau_x + shift_x) ** share) *
        ((tau_y + shift_y) ** (1 - share))) + shift

    return rate_grid


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
    df_trnc = df[(df['Total Labor Income'] > 5) &
        (df['Total Labor Income'] < 500000) &
        (df['Total Capital Income'] > 5) &
        (df['Total Capital Income'] < 500000)]
    inc_lab = df_trnc['Total Labor Income']
    inc_cap = df_trnc['Total Capital Income']
    etr_data = df_trnc['Effective Tax Rate']
    mtrx_data = df_trnc['MTR Labor']
    mtry_data = df_trnc['MTR capital income']

    # Plot 3D scatterplot of ETR data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    ax.scatter(inc_lab, inc_cap, etr_data, c='r', marker='o')
    ax.set_xlabel('Total Labor Income')
    ax.set_ylabel('Total Capital Income')
    ax.set_zlabel('Effective Tax Rate')
    plt.title('ETR, Lab. Inc., and Cap. Inc., Age=' + str(s) + ', Year='
        + str(t))
    filename = ("ETR_Age_" + str(s) + "_Year_" + str(t) + "_data.png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

    # Plot 3D histogram for all data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
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
    ax.set_xlabel('Total Labor Income')
    ax.set_ylabel('Total Capital Income')
    ax.set_zlabel('Percent of obs.')
    plt.title('Histogram by lab. inc., and cap. inc., Age=' + str(s) +
        ', Year=' + str(t))
    filename = ("Hist_Age_" + str(s) + "_Year_" + str(t) + ".png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

    # Plot 3D scatterplot of MTRx data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    ax.scatter(inc_lab, inc_cap, mtrx_data, c='r', marker='o')
    ax.set_xlabel('Total Labor Income')
    ax.set_ylabel('Total Capital Income')
    ax.set_zlabel('Marginal Tax Rate, Labor Inc.)')
    plt.title("MTR Labor Income, Lab. Inc., and Cap. Inc., Age=" +
        str(s) + ", Year=" + str(t))
    filename = ("MTRx_Age_"+ str(s) + "_Year_" + str(t) + "_data.png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

    # Plot 3D scatterplot of MTRy data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    ax.scatter(inc_lab, inc_cap, mtry_data, c='r', marker='o')
    ax.set_xlabel('Total Labor Income')
    ax.set_ylabel('Total Capital Income')
    ax.set_zlabel('Marginal Tax Rate (Capital Inc.)')
    plt.title("MTR Capital Income, Cap. Inc., and Cap. Inc., Age=" +
        str(s) + ", Year=" + str(t))
    filename = ("MTRy_Age_"+ str(s) + "_Year_" + str(t) + "_data.png")
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()


def plot_txfunc_v_data(tx_params, data, params): #This isn't in use yet
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
    X_data = data['Total Labor Income']
    Y_data = data['Total Capital Income']
    (s, t, rate_type, plot_full, plot_trunc, show_plots, save_plots,
        output_dir) = params

    cmap1 = matplotlib.cm.get_cmap('summer')

    if plot_full:
        if rate_type == 'etr':
            txrate_data = data['Effective Tax Rate']
            tx_label = 'ETR'
        elif rate_type == 'mtrx':
            txrate_data = data['MTR Labor']
            tx_label = 'MTRx'
        elif rate_type == 'mtry':
            txrate_data = data['MTR capital income']
            tx_label = 'MTRy'
        # Make comparison plot with full income domains
        fig = plt.figure()
        ax = fig.add_subplot(111, projection ='3d')
        ax.scatter(X_data, Y_data, txrate_data, c='r', marker='o')
        ax.set_xlabel('Total Labor Income')
        ax.set_ylabel('Total Capital Income')
        ax.set_zlabel(tx_label)
        plt.title(tx_label + ' vs. Predicted ' + tx_label + ': Age=' +
            str(s) + ', Year=' + str(t))

        gridpts = 50
        X_vec = np.exp(np.linspace(np.log(1), np.log(X_data.max()),
            gridpts))
        Y_vec = np.exp(np.linspace(np.log(1), np.log(Y_data.max()),
            gridpts))
        X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
        txrate_grid = gen_rate_grid(X_grid, Y_grid, tx_params)
        ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1,
            linewidth=0)

        if save_plots:
            filename = (tx_label + '_Age_' + str(s) + '_Year_' + str(t) +
                '_vsPred.png')
            fullpath = os.path.join(output_dir, filename)
            fig.savefig(fullpath, bbox_inches='tight')

        if show_plots:
            plt.show()

        plt.close()

    if plot_trunc:
        # Make comparison plot with truncated income domains
        data_trnc = data[(data['Total Labor Income'] > 5) &
            (data['Total Labor Income'] < 800000) &
            (data['Total Capital Income'] > 5) &
            (data['Total Capital Income'] < 800000)]
        X_trnc = data_trnc['Total Labor Income']
        Y_trnc = data_trnc['Total Capital Income']
        if rate_type == 'etr':
            txrates_trnc = data_trnc['Effective Tax Rate']
            tx_label = 'ETR'
        elif rate_type == 'mtrx':
            txrates_trnc = data_trnc['MTR Labor']
            tx_label = 'MTRx'
        elif rate_type == 'mtry':
            txrates_trnc = data_trnc['MTR capital income']
            tx_label = 'MTRy'

        fig = plt.figure()
        ax = fig.add_subplot(111, projection ='3d')
        ax.scatter(X_trnc, Y_trnc, txrates_trnc, c='r', marker='o')
        ax.set_xlabel('Total Labor Income')
        ax.set_ylabel('Total Capital Income')
        ax.set_zlabel(tx_label)
        plt.title('Truncated ' + tx_label + ', Lab. Inc., and Cap. ' +
            'Inc., Age=' + str(s) + ', Year=' + str(t))

        gridpts = 50
        X_vec = np.exp(np.linspace(np.log(1), np.log(X_trnc.max()),
            gridpts))
        Y_vec = np.exp(np.linspace(np.log(1), np.log(Y_trnc.max()),
            gridpts))
        X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
        txrate_grid = gen_rate_grid(X_grid, Y_grid, tx_params)
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


def wsumsq(params, *args):
    '''
    --------------------------------------------------------------------
    This function generates the weighted sum of squared deviations of
    predicted values of tax rates (ETR, MTRx, or MTRy) from the tax
    rates from the data for the Cobb-Douglas functional form of the tax
    function.
    --------------------------------------------------------------------
    INPUTS:
    params  = (7,) vector, guesses for (coef1, coef2, coef3, coef4,
              max_x, max_y, share)
    coef1   = scalar > 0, adjusted coefficient on \hat{X^2} term
    coef2   = scalar > 0, adjusted coefficient on \hat{X} term
    coef3   = scalar > 0, adjusted coefficient on \hat{Y^2} term
    coef4   = scalar > 0, adjusted coefficient on \hat{Y} term
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
    coef1, coef2, coef3, coef4, max_x, max_y, share = params
    X, Y, min_x, min_y, shift, txrates, wgts = args
    X2 = X ** 2
    Y2 = Y ** 2
    X2bar = (X2 * wgts).sum() / wgts.sum()
    Xbar = (X * wgts).sum() / wgts.sum()
    Y2bar = (Y2 * wgts).sum() / wgts.sum()
    Ybar = (Y * wgts).sum() / wgts.sum()
    shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
    shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
    X2til = (X2 - X2bar) / X2bar
    Xtil = (X - Xbar) / Xbar
    Y2til = (Y2 - Y2bar) / Y2bar
    Ytil = (Y - Ybar) / Ybar
    Etil = coef1 + coef2
    Ftil = coef3 + coef4
    tau_x = (((max_x - min_x) * (coef1 * X2til + coef2 * Xtil + Etil) /
        (coef1 * X2til + coef2 * Xtil + Etil + 1)) + min_x)
    tau_y = (((max_y - min_y) * (coef3 * Y2til + coef4 * Ytil + Ftil) /
        (coef3 * Y2til + coef4 * Ytil + Ftil + 1)) + min_y)
    txrates_est = (((tau_x + shift_x) ** share) *
                  ((tau_y + shift_y) ** (1 - share))) + shift
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
    se_mult    = scalar, multiple of standard devitiosn before consider
                  estimate an outlier
    start_year = integer, first year of budget window
    varstr     = string, name of tax function being evaluated
    graph      = boolean, flag to output graphs
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    OBJECTS CREATED WITHIN FUNCTION:
    thresh      = [S,BW] array, threshold values for SSE before consider
                   tax function outlier
    sse_big_mat = [S,BW] array, indicators of weither tax function is outlier
    RETURNS: sse_big_mat
    --------------------------------------------------------------------
    '''
    # Mark outliers from estimated MTRx functions
    thresh = (sse_mat[sse_mat>0].mean() +
             se_mult * sse_mat[sse_mat>0].std())
    sse_big_mat = sse_mat > thresh
    print varstr, ": ", str(sse_big_mat.sum()), \
          " observations tagged as outliers."
    if graph == True:
        # Plot sum of squared errors of tax functions over age for each
        # year of budget window
        fig, ax = plt.subplots()
        plt.plot(age_vec, sse_mat[:,0], label=str(start_year))
        plt.plot(age_vec, sse_mat[:,1], label=str(start_year+1))
        plt.plot(age_vec, sse_mat[:,2], label=str(start_year+2))
        plt.plot(age_vec, sse_mat[:,3], label=str(start_year+3))
        plt.plot(age_vec, sse_mat[:,4], label=str(start_year+4))
        plt.plot(age_vec, sse_mat[:,5], label=str(start_year+5))
        plt.plot(age_vec, sse_mat[:,6], label=str(start_year+6))
        plt.plot(age_vec, sse_mat[:,7], label=str(start_year+7))
        plt.plot(age_vec, sse_mat[:,8], label=str(start_year+8))
        plt.plot(age_vec, sse_mat[:,9], label=str(start_year+9))
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator   = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.legend(loc='upper left')
        titletext = "Sum of Squared Errors by Age and Tax Year: " + varstr
        plt.title(titletext)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'SSE')
        # Create directory if OUTPUT directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "OUTPUT/TaxFunctions"
        output_dir = os.path.join(cur_path, output_fldr)
        if os.access(output_dir, os.F_OK) == False:
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
        thresh2 = (sse_mat_new[sse_mat_new>0].mean() +
                  se_mult * sse_mat_new[sse_mat_new>0].std())
        sse_big_mat += sse_mat_new > thresh2
        print varstr, ": ", "After second round, ", \
            str(sse_big_mat.sum()), \
            " observations tagged as outliers (cumulative)."
        if graph == True:
            # Plot sum of squared errors of tax functions over age for
            # each year of budget window
            fig, ax = plt.subplots()
            plt.plot(age_vec, sse_mat_new[:,0], label=str(start_year))
            plt.plot(age_vec, sse_mat_new[:,1], label=str(start_year+1))
            plt.plot(age_vec, sse_mat_new[:,2], label=str(start_year+2))
            plt.plot(age_vec, sse_mat_new[:,3], label=str(start_year+3))
            plt.plot(age_vec, sse_mat_new[:,4], label=str(start_year+4))
            plt.plot(age_vec, sse_mat_new[:,5], label=str(start_year+5))
            plt.plot(age_vec, sse_mat_new[:,6], label=str(start_year+6))
            plt.plot(age_vec, sse_mat_new[:,7], label=str(start_year+7))
            plt.plot(age_vec, sse_mat_new[:,8], label=str(start_year+8))
            plt.plot(age_vec, sse_mat_new[:,9], label=str(start_year+9))
            # for the minor ticks, use no labels; default NullFormatter
            minorLocator   = MultipleLocator(1)
            ax.xaxis.set_minor_locator(minorLocator)
            plt.grid(b=True, which='major', color='0.65',linestyle='-')
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
            if graph == True:
                # Plot sum of squared errors of tax functions over age
                # for each year of budget window
                fig, ax = plt.subplots()
                plt.plot(age_vec, sse_mat_new2[:,0], label=str(start_year))
                plt.plot(age_vec, sse_mat_new2[:,1], label=str(start_year+1))
                plt.plot(age_vec, sse_mat_new2[:,2], label=str(start_year+2))
                plt.plot(age_vec, sse_mat_new2[:,3], label=str(start_year+3))
                plt.plot(age_vec, sse_mat_new2[:,4], label=str(start_year+4))
                plt.plot(age_vec, sse_mat_new2[:,5], label=str(start_year+5))
                plt.plot(age_vec, sse_mat_new2[:,6], label=str(start_year+6))
                plt.plot(age_vec, sse_mat_new2[:,7], label=str(start_year+7))
                plt.plot(age_vec, sse_mat_new2[:,8], label=str(start_year+8))
                plt.plot(age_vec, sse_mat_new2[:,9], label=str(start_year+9))
                # for the minor ticks, use no labels; default NullFormatter
                minorLocator   = MultipleLocator(1)
                ax.xaxis.set_minor_locator(minorLocator)
                plt.grid(b=True, which='major', color='0.65',linestyle='-')
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
    param_arr_adj = [S,BW,#tax params] array, estimated and interpolated tax function parameters
    big_cnt       = integer, number of outliers replaced
    slopevec      = [1,1,#tax params] array, slope used for linear interpolation of outliers
    interceptvec  = [1,1,#tax params] array, intercept used for linear interpolation of outliers
    RETURNS: param_arr_adj
    --------------------------------------------------------------------
    '''
    numparams = param_arr.shape[2]
    age_ind = np.arange(0, sse_big_mat.shape[0])
    param_arr_adj = param_arr.copy()
    for t in xrange(0, sse_big_mat.shape[1]):
        big_cnt = 0
        for s in age_ind:
            # Smooth out ETR tax function outliers
            if sse_big_mat[s, t] == True and s < sse_big_mat.shape[0]-1:
                # For all outlier observations, increase the big_cnt by
                # 1 and set the param_arr_adj equal to nan
                big_cnt += 1
                param_arr_adj[s, t, :] = np.nan
            if (sse_big_mat[s, t] == False and big_cnt > 0 and
              s == big_cnt):
                # When the current function is not an outlier but the last
                # one was and this string of outliers is at the beginning
                # ages, set the outliers equal to this period's tax function
                param_arr_adj[:big_cnt, t, :] = \
                    np.tile(param_arr_adj[s, t, :].reshape((1, 1, numparams)),
                    (big_cnt, 1, 1))
                big_cnt = 0
            if (sse_big_mat[s, t] == False and big_cnt > 0 and
              s > big_cnt):
                # When the current function is not an outlier but the last
                # one was and this string of outliers is in the interior of
                # ages, set the outliers equal to a linear interpolation
                # between the two bounding non-outlier functions
                slopevec = ((param_arr_adj[s, t, :] -
                    param_arr_adj[s-big_cnt-1, t, :]) / (big_cnt + 1))
                interceptvec = (param_arr_adj[s-big_cnt-1, t, :])
                param_arr_adj[s-big_cnt:s, t, :] = (np.tile(interceptvec.reshape(1,numparams),(big_cnt,1)) +
                    np.tile(slopevec.reshape(1,numparams),(big_cnt,1))*np.tile(np.reshape(np.arange(1,big_cnt+1),(big_cnt,1)),(1,numparams)))
                big_cnt = 0
            if sse_big_mat[s, t] == True and s == sse_big_mat.shape[0]-1:
                # When the last ages are outliers, set the parameters equal
                # to the most recent non-outlier tax function
                big_cnt += 1
                param_arr_adj[s, t, :] = np.nan
                param_arr_adj[s-big_cnt+1:, t, :] = \
                    np.tile(param_arr_adj[s-big_cnt, t, :].reshape((1, 1, numparams)),
                    (big_cnt, 1, 1))

    return param_arr_adj


def txfunc_est(df, s, t, rate_type, output_dir, graph):
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
        gen_rate_grid()

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
    X = df['Total Labor Income']
    Y = df['Total Capital Income']
    wgts = df['Weights']
    X2 = X ** 2
    Y2 = Y ** 2
    X2bar = (X2 * wgts).sum() / wgts.sum()
    Xbar = (X * wgts).sum() / wgts.sum()
    Y2bar = (Y2 * wgts).sum() / wgts.sum()
    Ybar = (Y * wgts).sum() / wgts.sum()
    if rate_type == 'etr':
        txrates = df['Effective Tax Rate']
    elif rate_type == 'mtrx':
        txrates = df['MTR Labor']
    elif rate_type == 'mtry':
        txrates = df['MTR capital income']
    x_10pctl = df['Total Labor Income'].quantile(0.1)
    y_10pctl = df['Total Capital Income'].quantile(0.1)
    x_20pctl = df['Total Labor Income'].quantile(.2)
    y_20pctl = df['Total Capital Income'].quantile(.2)
    min_x = txrates[(df['Total Capital Income'] < y_10pctl)].min()
    min_y = txrates[(df['Total Labor Income'] < x_10pctl)].min()
    Atil_init = 1.0
    Btil_init = 1.0
    Ctil_init = 1.0
    Dtil_init = 1.0
    max_x_init = np.minimum(
        txrates[(df['Total Capital Income'] < y_20pctl)].max(), 0.7)
    max_y_init = np.minimum(
        txrates[(df['Total Labor Income'] < x_20pctl)].max(), 0.7)
    shift = txrates[(df['Total Labor Income'] < x_20pctl) |
        (df['Total Capital Income'] < y_20pctl)].min()
    share_init = 0.5
    numparams = int(12)
    params_init = np.array([Atil_init, Btil_init, Ctil_init,
        Dtil_init, max_x_init, max_y_init, share_init])
    tx_objs = (X, Y, min_x, min_y, shift, txrates, wgts)
    lb_max_x = np.maximum(min_x, 0.0) + 1e-4
    lb_max_y = np.maximum(min_y, 0.0) + 1e-4
    bnds = ((1e-12, None), (1e-12, None), (1e-12, None), (1e-12, None),
        (lb_max_x, 0.8), (lb_max_y, 0.8), (0, 1))
    params_til = opt.minimize(wsumsq, params_init,
        args=(tx_objs), method="L-BFGS-B", bounds=bnds, tol=1e-15)
    Atil, Btil, Ctil, Dtil, max_x, max_y, share = params_til.x
    # message = ("(max_x, min_x)=(" + str(max_x) + ", " + str(min_x) +
    #     "), (max_y, min_y)=(" + str(max_y) + ", " + str(min_y) + ")")
    # print message
    wsse = params_til.fun
    obs = df.shape[0]
    shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
    shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
    params = np.zeros(numparams)
    params[:4] = (np.array([Atil, Btil, Ctil, Dtil]) /
        np.array([X2bar, Xbar, Y2bar, Ybar]))
    params[4:] = np.array([max_x, min_x, max_y, min_y, shift_x,
        shift_y, shift, share])

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
        ax = fig.add_subplot(111, projection ='3d')
        ax.scatter(X, Y, txrates, c='r', marker='o')
        ax.set_xlabel('Total Labor Income')
        ax.set_ylabel('Total Capital Income')
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
        txrate_grid = gen_rate_grid(X_grid, Y_grid, params)
        ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1,
            linewidth=0)
        filename = (tx_label + '_Age_' + str(s) + '_Year_' + str(t) +
            '_vsPred.png')
        fullpath = os.path.join(output_dir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        # Make comparison plot with truncated income domains
        df_trnc_gph = df[(df['Total Labor Income'] > 5) &
            (df['Total Labor Income'] < 800000) &
            (df['Total Capital Income'] > 5) &
            (df['Total Capital Income'] < 800000)]
        X_gph = df_trnc_gph['Total Labor Income']
        Y_gph = df_trnc_gph['Total Capital Income']
        if rate_type == 'etr':
            txrates_gph = df_trnc_gph['Effective Tax Rate']
        elif rate_type == 'mtrx':
            txrates_gph = df_trnc_gph['MTR Labor']
        elif rate_type == 'mtry':
            txrates_gph = df_trnc_gph['MTR capital income']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection ='3d')
        ax.scatter(X_gph, Y_gph, txrates_gph, c='r', marker='o')
        ax.set_xlabel('Total Labor Income')
        ax.set_ylabel('Total Capital Income')
        ax.set_zlabel(tx_label)
        plt.title('Truncated ' + tx_label + ', Lab. Inc., and Cap. ' +
            'Inc., Age=' + str(s) + ', Year=' + str(t))

        gridpts = 50
        X_vec = np.exp(np.linspace(np.log(5), np.log(X_gph.max()),
            gridpts))
        Y_vec = np.exp(np.linspace(np.log(5), np.log(Y_gph.max()),
            gridpts))
        X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
        txrate_grid = gen_rate_grid(X_grid, Y_grid, params)
        ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1,
            linewidth=0)
        filename = (tx_label + 'trunc_Age_' + str(s) + '_Year_' + str(t) +
            '_vsPred.png')
        fullpath = os.path.join(output_dir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

    return params, wsse, obs


def tax_func_estimate(beg_yr=2016, baseline=True, analytical_mtrs=False,
  age_specific=True, reform={}):
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
    S = int(80)
    tpers = int(10)
    s_min = int(21)
    s_max = int(100)
    beg_yr = int(beg_yr)
    end_yr = int(beg_yr + tpers - 1)
    numparams = int(12)
    desc_data = False
    graph_data = False
    graph_est = False
    # A, B, C, D, maxx, minx, maxy, miny, shift_x, shift_y, shift, share
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
    years_list = np.arange(beg_yr, end_yr + 1)

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
    start_time = time.clock()
    output_dir = "./OUTPUT/txfuncs"
    utils.mkdirs(output_dir)

    # call tax caculator and get microdata
    micro_data = get_micro_data.get_data(baseline=baseline,
        start_year=beg_yr, reform=reform)
    # if reform:
    #     micro_data = pickle.load(open("micro_data_policy.pkl", "rb"))
    # else:
    #     micro_data = pickle.load(open("micro_data_baseline.pkl", "rb"))

    for t in years_list: #for t in np.arange(2016, 2017):
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
        data_orig = micro_data[str(t)]
        data_orig['Total Labor Income'] = \
            (data_orig['Wage and Salaries'] +
            data_orig['Self-Employed Income'])
        data_orig['Effective Tax Rate'] = \
            (data_orig['Total Tax Liability'] /
            data_orig["Adjusted Total income"])
        data_orig["Total Capital Income"] = \
            (data_orig['Adjusted Total income'] -
            data_orig['Total Labor Income'])
        # use weighted avg for MTR labor - abs value because
        # SE income may be negative
        data_orig['MTR Labor'] = \
            (data_orig['MTR wage'] * (data_orig['Wage and Salaries'] /
            (data_orig['Wage and Salaries'].abs() +
            data_orig['Self-Employed Income'].abs())) +
            data_orig['MTR self-employed Wage'] *
            (data_orig['Self-Employed Income'].abs() /
            (data_orig['Wage and Salaries'].abs() +
            data_orig['Self-Employed Income'].abs())))
        data = data_orig[['Age', 'MTR Labor', 'MTR capital income',
            'Total Labor Income', 'Total Capital Income',
            'Adjusted Total income', 'Effective Tax Rate', 'Weights']]

        # Calculate average total income in each year
        AvgInc[t-beg_yr] = \
            (((data['Adjusted Total income'] * data['Weights']).sum())
            / data['Weights'].sum())

        # Calculate average ETR and MTRs (weight by population weights
        #    and income) for each year
        AvgETR[t-beg_yr] = \
            (((data['Effective Tax Rate']*data['Adjusted Total income']
            * data['Weights']).sum()) /
            (data['Adjusted Total income']*data['Weights']).sum())

        AvgMTRx[t-beg_yr] = \
            (((data['MTR Labor']*data['Adjusted Total income'] *
            data['Weights']).sum()) /
            (data['Adjusted Total income']*data['Weights']).sum())

        AvgMTRy[t-beg_yr] = \
            (((data['MTR capital income'] *
            data['Adjusted Total income'] * data['Weights']).sum()) /
            (data['Adjusted Total income']*data['Weights']).sum())

        # Calculate total population in each year
        TotPop_yr[t-beg_yr] = data['Weights'].sum()

        # Clean up the data by dropping outliers
        # drop all obs with ETR > 0.65
        data_trnc = \
            data.drop(data[data['Effective Tax Rate'] >0.65].index)
        # drop all obs with ETR < -0.15
        data_trnc = \
            data_trnc.drop(data_trnc[data_trnc['Effective Tax Rate']
            < -0.15].index)
        # drop all obs with ATI, TLI, TCI < $5
        data_trnc = data_trnc[(data_trnc['Adjusted Total income'] >= 5)
            & (data_trnc['Total Labor Income'] >= 5) &
            (data_trnc['Total Capital Income'] >= 5)]

        if analytical_mtrs==False:
            # drop all obs with MTR on capital income > 10.99
            data_trnc = \
                data_trnc.drop(data_trnc[data_trnc['MTR capital income']
                > 0.99].index)
            # drop all obs with MTR on capital income < -0.45
            data_trnc = \
                data_trnc.drop(data_trnc[data_trnc['MTR capital income']
                < -0.45].index)
            # drop all obs with MTR on labor income > 10.99
            data_trnc = data_trnc.drop(data_trnc[data_trnc['MTR Labor']
                        > 0.99].index)
            # drop all obs with MTR on labor income < -0.45
            data_trnc = data_trnc.drop(data_trnc[data_trnc['MTR Labor']
                        < -0.45].index)

        # Create an array of the different ages in the data
        min_age = int(np.maximum(data_trnc['Age'].min(), s_min))
        max_age = int(np.minimum(data_trnc['Age'].max(), s_max))
        if age_specific:
            ages_list = np.arange(min_age, max_age+1)
        else:
            ages_list = np.arange(0,1)

        NoData_cnt = np.min(min_age - s_min, 0)

        # Each age s must be done in serial, but each year can be done
        # in parallel

        for s in ages_list: # for s in np.array([23, 24, 60, 63, 64, 65, 66, 67, 70, 71, 74, 79]):
            if age_specific:
                print "year=", t, "Age=", s
                df = data_trnc[data_trnc['Age'] == s]
                PopPct_age[s-min_age, t-beg_yr] = \
                    df['Weights'].sum() / TotPop_yr[t-beg_yr]

            else:
                print "year=", t, "Age= all ages"
                df = data_trnc
                PopPct_age[0, t-beg_yr] = \
                    df['Weights'].sum() / TotPop_yr[t-beg_yr]

            df_etr = df[['MTR Labor', 'MTR capital income',
                'Total Labor Income', 'Total Capital Income',
                'Effective Tax Rate', 'Weights']]
            df_etr = df_etr[
                (np.isfinite(df_etr['Effective Tax Rate'])) &
                (np.isfinite(df_etr['Total Labor Income'])) &
                (np.isfinite(df_etr['Total Capital Income'])) &
                (np.isfinite(df_etr['Weights']))]
            df_mtrx = df[['MTR Labor', 'Total Labor Income',
                'Total Capital Income', 'Weights']]
            df_mtrx = df_mtrx[
                (np.isfinite(df_etr['MTR Labor'])) &
                (np.isfinite(df_etr['Total Labor Income'])) &
                (np.isfinite(df_etr['Total Capital Income'])) &
                (np.isfinite(df_etr['Weights']))]
            df_mtry = df[['MTR capital income', 'Total Labor Income',
                'Total Capital Income', 'Weights']]
            df_mtry = df_mtry[
                (np.isfinite(df_etr['MTR capital income'])) &
                (np.isfinite(df_etr['Total Labor Income'])) &
                (np.isfinite(df_etr['Total Capital Income'])) &
                (np.isfinite(df_etr['Weights']))]
            df_minobs = np.min([df_etr.shape[0], df_mtrx.shape[0],
                df_mtry.shape[0]])

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
                print message
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
                    + " year " + str(t) + ". Fill in final ages with " +
                    "insuff. data with most recent successful " +
                    "estimate.")
                print message
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
                    print message
                    print df_etr.describe()
                    message = ("Descriptive MTRx statistics for age=" +
                        str(s) + " in year " + str(t))
                    print message
                    print df_mtrx.describe()
                    message = ("Descriptive MTRy statistics for age=" +
                        str(s) + " in year " + str(t))
                    print message
                    print df_mtry.describe()

                if graph_data:
                    gen_3Dscatters_hist(df, s, t, output_dir)

                # Estimate effective tax rate function ETR(x,y)
                (etrparams, etr_wsumsq_arr[s-s_min, t-beg_yr],
                    etr_obs_arr[s-s_min, t-beg_yr]) = \
                    txfunc_est(df_etr, s, t, 'etr', output_dir,
                        graph_est)
                etrparam_arr[s-s_min, t-beg_yr, :] = etrparams

                # Estimate marginal tax rate of labor income function
                # MTRx(x,y)
                (mtrxparams, mtrx_wsumsq_arr[s-s_min, t-beg_yr],
                    mtrx_obs_arr[s-s_min, t-beg_yr]) = \
                    txfunc_est(df_mtrx, s, t, 'mtrx', output_dir,
                        graph_est)
                mtrxparam_arr[s-s_min, t-beg_yr, :] = mtrxparams

                # Estimate marginal tax rate of capital income function
                # MTRy(x,y)
                (mtryparams, mtry_wsumsq_arr[s-s_min, t-beg_yr],
                    mtry_obs_arr[s-s_min, t-beg_yr]) = \
                    txfunc_est(df_mtry, s, t, 'mtry', output_dir,
                        graph_est)
                mtryparam_arr[s-s_min, t-beg_yr, :] = mtryparams

                if NoData_cnt > 0 & NoData_cnt == s-s_min:
                    '''
                    ----------------------------------------------------
                    Fill in initial blanks with first positive data
                    estimates. This includes the case in which
                    min_age > s_min
                    ----------------------------------------------------
                    '''
                    message = "Fill in all previous blank ages"
                    print message
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
                    print message
                    tvals = np.linspace(0, 1, NoData_cnt+2)
                    x0_etr = np.tile(etrparam_arr[s-NoData_cnt-s_min-1,
                        t-beg_yr, :].reshape((1, numparams)),
                        (NoData_cnt, 1))
                    x1_etr = np.tile(etrparams.reshape((1, numparams)),
                             (NoData_cnt, 1))
                    lin_int_etr = \
                        (x0_etr + tvals[1:-1].reshape((NoData_cnt, 1))
                        * (x1_etr - x0_etr))
                    etrparam_arr[s-NoData_cnt-min_age:s-min_age,
                        t-beg_yr, :] = lin_int_etr
                    x0_mtrx = np.tile(
                        mtrxparam_arr[s-NoData_cnt-s_min-1, t-beg_yr,
                        :].reshape((1, numparams)),(NoData_cnt, 1))
                    x1_mtrx = np.tile(
                        mtrxparams.reshape((1, numparams)),
                        (NoData_cnt, 1))
                    lin_int_mtrx = \
                        (x0_mtrx + tvals[1:-1].reshape((NoData_cnt, 1))
                        * (x1_mtrx - x0_mtrx))
                    mtrxparam_arr[s-NoData_cnt-min_age:s-min_age,
                        t-beg_yr, :] = lin_int_mtrx
                    x0_mtry = np.tile(
                        mtryparam_arr[s-NoData_cnt-s_min-1, t-beg_yr,
                        :].reshape((1, numparams)), (NoData_cnt, 1))
                    x1_mtry = np.tile(
                        mtryparams.reshape((1, numparams)),
                        (NoData_cnt, 1))
                    lin_int_mtry = \
                        (x0_mtry + tvals[1:-1].reshape((NoData_cnt, 1))
                        * (x1_mtry - x0_mtry))
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
                    print message
                    etrparam_arr[s-s_min+1:, t-beg_yr, :] = \
                        np.tile(etrparams.reshape((1, numparams)),
                        (s_max-max_age, 1))
                    mtrxparam_arr[s-s_min+1:, t-beg_yr, :] = \
                        np.tile(mtrxparams.reshape((1, numparams)),
                        (s_max-max_age, 1))
                    mtryparam_arr[s-s_min+1:, t-beg_yr, :] = \
                        np.tile(mtryparams.reshape((1, numparams)),
                        (s_max-max_age, 1))

    message = ("Finished tax function loop through " +
        str(len(years_list)) + " years and " + str(len(ages_list)) +
        " ages per year.")
    print message
    elapsed_time = time.clock() - start_time

    # Print tax function computation time
    if elapsed_time < 60: # less than a minute
        secs = round(elapsed_time, 3)
        message = "Tax function estimation time: " + str(secs) + " sec"
        print message
    elif elapsed_time >= 60 and elapsed_time < 3600: # less than hour
        mins = int(elapsed_time / 60)
        secs = round(((elapsed_time / 60) - mins) * 60, 1)
        message = ("Tax function estimation time: " + str(mins) +
            " min, " + str(secs) + " sec")
        print message
    elif elapsed_time >= 3600 and elapsed_time < 86400: # less than day
        hours = int(elapsed_time / (60 * 60))
        mins = int((elapsed_time  - (hours * 60 * 60))/ 60)
        secs = round(elapsed_time - (hours * 60 * 60) - (mins * 60), 1)
        message = ("Tax function estimation time: " + str(hours) +
            " hour(s), "+ str(mins) + " min(s), " + str(secs) + " sec(s)")
        print message

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
            etrparam_arr_adj = replace_outliers(etrparam_arr, etr_sse_big)
        elif etr_sse_big.sum() == 0:
            etrparam_arr_adj = etrparam_arr

        mtrx_sse_big = find_outliers(mtrx_wsumsq_arr / mtrx_obs_arr,
            age_sup, se_mult, beg_yr, "MTRx")
        if mtrx_sse_big.sum() > 0:
            mtrxparam_arr_adj = replace_outliers(mtrxparam_arr, mtrx_sse_big)
        elif mtrx_sse_big.sum() == 0:
            mtrxparam_arr_adj = mtrxparam_arr

        mtry_sse_big = find_outliers(mtry_wsumsq_arr / mtry_obs_arr,
            age_sup, se_mult, beg_yr, "MTRy")
        if mtry_sse_big.sum() > 0:
            mtryparam_arr_adj = replace_outliers(mtryparam_arr, mtry_sse_big)
        elif mtry_sse_big.sum() == 0:
            mtryparam_arr_adj = mtryparam_arr

    '''
    --------------------------------------------------------------------
    Generate tax function parameters for S < s_max - s_min + 1
    --------------------------------------------------------------------
    etrparam_arr_S  = S x tpers x 10 array, this is an array in which S is
                      less-than-or-equal-to s_max-s_min+1. We use weighted
                      averages of parameters in relevant age groups
    mtrxparam_arr_S = S x tpers x 10 array, this is an array in which S is
                      less-than-or-equal-to s_max-s_min+1. We use weighted
                      averages of parameters in relevant age groups
    age_cuts     = (S+1,) vector, linspace of age cutoffs of S+1 points
                   between 0 and S+1
    yrcut_lb     = integer >= 0, index of lower bound age for S bin
    yrcut_ub     = integer >= 0, index of upper bound age for S bin
    rmndr_pct_lb = scalar in [0,1], discounted weight on lower bound age
    rmndr_pct_ub = scalar in [0,1], discounted weight on upper bound age
    age_wgts     = ages x tpers x 10 array, age weights for each age in each
                   year copied back 10 times in the 3rd dimension
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
            age_cuts = np.linspace(0, s_max-s_min+1, S+1)
            yrcut_lb = int(age_cuts[0])
            rmndr_pct_lb = 1.
            for s in np.arange(S):
                yrcut_ub = int(np.floor(age_cuts[s+1]))
                rmndr_pct_ub = age_cuts[s+1] - np.floor(age_cuts[s+1])
                if rmndr_pct_ub == 0.:
                    rmndr_pct_ub = 1.
                    yrcut_ub -= 1
                age_wgts = np.dstack([PopPct_age[yrcut_lb:yrcut_ub+1, :]]*10)
                # print yrcut_lb, yrcut_ub, rmndr_pct_lb, rmndr_pct_ub, age_wgts.shape
                age_wgts[0, :, :] *= rmndr_pct_lb
                age_wgts[yrcut_ub-yrcut_lb, :, :] *= rmndr_pct_ub
                etrparam_arr_S[s, :, :] = (etrparam_arr_adj[yrcut_lb:yrcut_ub+1, :, :] * age_wgts).sum(axis=0)
                mtrxparam_arr_S[s, :, :] = (mtrxparam_arr_adj[yrcut_lb:yrcut_ub+1, :, :] * age_wgts).sum(axis=0)
                mtryparam_arr_S[s, :, :] = (mtryparam_arr_adj[yrcut_lb:yrcut_ub+1, :, :] * age_wgts).sum(axis=0)
                yrcut_lb = yrcut_ub
                rmndr_pct_lb = 1 - rmndr_pct_ub

        print 'Big S: ', S
        print 'max age, min age: ', s_max, s_min
    else:
        etrparam_arr_S = np.tile(np.reshape(etrparam_arr[s-s_min, :, :],
            (1, tpers, etrparam_arr.shape[2])), (S, 1, 1))
        mtrxparam_arr_S = np.tile(
            np.reshape(mtrxparam_arr[s-s_min, :, :],
            (1, tpers, mtrxparam_arr.shape[2])), (S, 1, 1))
        mtryparam_arr_S = np.tile(
            np.reshape(mtryparam_arr[s-s_min, :, :],
            (1, tpers, mtryparam_arr.shape[2])), (S, 1, 1))


    # Save tax function parameters array and computation time in
    # dictionary
    dict_params = dict([('tfunc_etr_params_S', etrparam_arr_S),
        ('tfunc_mtrx_params_S', mtrxparam_arr_S),
        ('tfunc_mtry_params_S', mtryparam_arr_S),
        ('tfunc_avginc', AvgInc), ('tfunc_avg_etr', AvgETR),
        ('tfunc_avg_mtrx', AvgMTRx), ('tfunc_avg_mtry', AvgMTRy),
        ('tfunc_etr_sumsq', etr_wsumsq_arr),
        ('tfunc_mtrx_sumsq', mtrx_wsumsq_arr),
        ('tfunc_mtry_sumsq', mtry_wsumsq_arr),
        ('tfunc_etr_obs', etr_obs_arr),
        ('tfunc_mtrx_obs', mtrx_obs_arr),
        ('tfunc_mtry_obs', mtry_obs_arr),
        ('tfunc_time', elapsed_time)])

    return dict_params


def get_tax_func_estimate(baseline=False, analytical_mtrs=False,
  age_specific=False, start_year=2016, reform={}, guid=''):
    '''
    --------------------------------------------------------------------
    This function calls the tax function estimation routine and saves
    the resulting dictionary in pickle files corresponding to the
    baseline or reform policy.
    --------------------------------------------------------------------

    INPUTS:
    baseline        = boolean, =True if baseline tax policy, =False if reform
    analytical_mtrs = boolean, =True if use analytical_mtrs, =False if
                      use estimated MTRs
    age_specific    = boolean, =True if estimate tax functions separately
                      for each age, =False if estimate a single tax function
                      to represent all ages in a given budget year
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
    dict_params = tax_func_estimate(start_year, baseline,
        analytical_mtrs, age_specific, reform)
    if baseline:
        baseline_pckl = "TxFuncEst_baseline{}.pkl".format(guid)
        pkl_path = os.path.join(TAX_ESTIMATE_PATH, baseline_pckl)
    else:
        policy_pckl = "TxFuncEst_policy{}.pkl".format(guid)
        pkl_path = os.path.join(TAX_ESTIMATE_PATH, policy_pckl)

    pickle.dump(dict_params, open(pkl_path, "wb"))
