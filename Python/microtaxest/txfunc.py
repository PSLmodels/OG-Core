'''
------------------------------------------------------------------------
This script reads in data generated from the OSPC Tax Calcultor and
the 2009 IRS PUF. It then estimates tax functions tau_{s,t}(x,y), where
tau_{s,t} is the effective tax rate for a given age (s) in a particular
year (t), x is total labor income, and y is total capital income.

This Python script calls the following functions:
    gen_etr_grid:   generates summary grid points for effective tax rate
    gen_dmtrx_grid: generates summary grid points for derivative of
                    marginal tax rate with respect to labor income
    gen_dmtry_grid: generates summary grid points for derivative of
                    marginal tax rate with respect to capital income
    wsumsq:         generates weighted sum of squared residuals


This Python script outputs the following:

------------------------------------------------------------------------
'''

# Import packages
import time
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
------------------------------------------------------------------------
Set parameters and create objects for output
------------------------------------------------------------------------
s_min       = integer > 0, minimum age relevant to the model
s_max       = integer > s_min, maximum age relevant to the model
tpers       = integer > 0, number of years to forecast
numparams   = integer > 0, number of parameters to estimate for the tax
              function for each age s and year t
param_arr   = (s_max-s_min+1 x tpers x numparams) array, parameter
              values for the estimated tax function for each age s and
              year t
desc_data   = boolean, =True if print descriptive stats for each age s
              and year t
graph_data  = boolean, =True if print 3D graph of data for each age s
              and year t
graph_est   = boolean, =True if print 3D graph of data and estimated tax
              function for each age s and year t
dmtrgr_est  = boolean, =True if print 3D graph of derivative of
              estimated marginal tax rates for each age s and year t
cmap1       = matplotlib setting, set color map for 3D surface plots
beg_yr      = integer >= 2015, beginning year for forecasts
end_yr      = integer >= beg_yr, ending year for forecasts
years_list  = [beg_yr-end_yr+1,] vector, iterable list of years to be
              forecast
data_folder = string, path of hard drive folder where data files reside
start_time  = scalar, current processor time in seconds (float)
------------------------------------------------------------------------
'''
s_min = int(21)
s_max = int(100)
tpers = int(10)
numparams = int(10)
param_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
desc_data = False
graph_data = False
graph_est = False
dmtrgr_est = False
cmap1 = matplotlib.cm.get_cmap('summer')
# cmap1 = matplotlib.cm.get_cmap('jet')
# cmap1 = matplotlib.cm.get_cmap('coolwarm')

beg_yr = int(2015)
end_yr = int(2024)
years_list = np.arange(beg_yr, end_yr + 1)
data_folder = '/Users/rwe2/Documents/Economics/OSPC/Data/micro-dynamic/'
start_time = time.clock()

'''
------------------------------------------------------------------------
Define Functions
------------------------------------------------------------------------
'''

def gen_etr_grid(X, Y, params):
    '''
    --------------------------------------------------------------------
    This function generates a grid of effective tax rates from a grid of
    total labor income and a grid of total capital income.
    --------------------------------------------------------------------
    '''
    A, B, C, D, E, F, max_x, min_x, max_y, min_y = params
    phi = X / (X + Y)
    P_num = A * (X ** 2) + B * (Y ** 2) + C * (X * Y) + D * X + E * Y
    P_den = (A * (X ** 2) + B * (Y ** 2) + C * (X * Y) + D * X + E * Y
            + F)
    etr_grid = ((phi * (max_x - min_x) + (1 - phi) * (max_y - min_y)) *
        (P_num / P_den) + (phi * min_x + (1 - phi) * min_y))
    return etr_grid


def gen_dmtrx_grid(X, Y, params):
    '''
    --------------------------------------------------------------------
    This function generates a grid of derivatives of marginal tax rates
    with respect to labor income from a grid of total labor income and a
    grid of total capital income
    --------------------------------------------------------------------
    '''
    A, B, C, D, E, F, max_x, min_x, max_y, min_y = params
    MTRx = ((F * ((2 * A * X) + (C * Y) + D)) /
           (((A * X ** 2) + (B * Y ** 2) + (C * X * Y) + (D * X) +
           (E * Y) + F) **2))
    dMTRx = ((2 * F * ((-3 * (A ** 2) * (X ** 2)) +
        ((A * B - C ** 2) * (Y ** 2)) - (3 * A * C * X * Y) -
        (3 * A * D * X) + ((A * E - 2 * C * D) * Y) + (A * F - D ** 2)))
        / (((A * X ** 2) + (B * Y ** 2) + (C * X * Y) + (D * X) +
        (E * Y) + F) ** 3))
    detrx_grid = (2 * (max_x - min_x) * MTRx +
                 ((max_x - min_x) * X + (max_y - min_y) * Y) * dMTRx)
    return dmtrx_grid


def gen_dmtry_grid(X, Y, params):
    '''
    --------------------------------------------------------------------
    This function generates a grid of derivatives of marginal tax rates
    from a grid of total labor income and
    a grid of total capital income
    --------------------------------------------------------------------
    '''
    A, B, C, D, E, F, max_x, min_x, max_y, min_y = params
    MTRy = ((F * ((2 * B * Y) + (C * X) + E)) /
           (((A * X ** 2) + (B * Y ** 2) + (C * X * Y) + (D * X) +
           (E * Y) + F) **2))
    dMTRy = ((2 * F * ((-3 * (B ** 2) * (Y ** 2)) +
        ((A * B - C ** 2) * (X ** 2)) - (3 * B * C * X * Y) -
        (3 * B * E * Y) + ((B * D - 2 * C * E) * X) + (B * F - E ** 2)))
        / (((A * X ** 2) + (B * Y ** 2) + (C * X * Y) + (D * X) +
        (E * Y) + F) ** 3))
    detry_grid = (2 * (max_y - min_y) * MTRy +
                 ((max_x - min_x) * X + (max_y - min_y) * Y) * dMTRy)
    return dmtry_grid



def wsumsq(params, *objs):
    '''
    --------------------------------------------------------------------
    This function generates the sum of squared percent deviations of
    predicted values of effective tax rates as a function of income and
    functional form parameters.

    tau(y) = (maxt - mint)*(A*y^2 + B*y)/(A*y^2 + B*y + C) + mint
    --------------------------------------------------------------------
    params     = [5,] vector, guesses for maxt, mint, A, B, C
    maxt       = scalar > 0, guess for maximum value of tax rate
                 function
    mint       = scalar, guess for minimum value of tax rate function
    A          = scalar > 0, tax function parameter A
    B          = scalar > 0, tax function parameter B
    C          = scalar > 0, tax function parameter B
    objs       = (3,) tuple, array objects passed in to function
    avinc      = [n,] vector, average AGI of income bins
    avgtax_dta = [n,] vector, average effective tax rate for each
                 income bin
    incwgts    = [n,] vector, weights on each (income, tax) point for
                 estimation
    avgtax_est = [n,] vector, average estimated effective tax rate for
                 each income bin
    pctdev     = [n,] vector, weighted percent deviation (times 100) of
                 estimated tax rates from data tax rates
    wssqdev    = scalar > 0, weighted sum of squared percent deviations

    returns: wssqdev
    --------------------------------------------------------------------
    '''
    varmat_hat, etr, wgts, varmat_bar, phi = objs
    max_x, min_x, max_y, min_y = params[-4:]
    Gtil = params[:5].sum()
    P_num = np.dot(varmat_hat[:,:-1], params[:5]) + Gtil
    P_den = np.dot(varmat_hat, params[:6]) + Gtil
    etr_est = ((phi * (max_x - min_x) + (1 - phi) * (max_y - min_y)) *
        (P_num / P_den) + (phi * min_x + (1 - phi) * min_y))
    errors = etr_est - etr
    wssqdev = (wgts * (errors ** 2)).sum()
    return wssqdev


'''
------------------------------------------------------------------------
Solve for tax functions for each year (t) and each age (s)
------------------------------------------------------------------------
'''

for t in years_list:
    '''
    --------------------------------------------------------------------
    Load OSPC Tax Calculator Data into a Dataframe
    --------------------------------------------------------------------
    t         = integer >= 2015, current year of tax functions being
                estimated
    data_file = string, name of data file for current year tax data
    data_path = string, path to directory address of data file
    data_orig = (I x 12) DataFrame, I observations, 12 variables
    data      = (I x 6) DataFrame, I observations, 6 variables
    data_trnc = (I2 x 6) DataFrame, I2 observations (I2<I), 6 variables
    min_age   = integer, minimum age in data_trnc that is at least s_min
    max_age   = integer, maximum age in data_trnc that is at most s_max
    ages_list = [age_max-age_min+1,] vector, iterable list of relevant
                ages in data_trnc
    --------------------------------------------------------------------
    '''
    data_file = str(t) + '_tau_n.csv'
    data_path = data_folder + data_file
    data_orig = pd.read_csv(data_path)
    data_orig['Total Labor Income'] = (data_orig['Wage and Salaries'] +
        data_orig['Self-Employed Income'])
    data_orig['Effective Tax Rate'] = (data_orig['Total Tax Liability']/
        data_orig["Adjusted Total income"])
    data_orig["Total Capital Income"] = \
        (data_orig['Adjusted Total income'] -
        data_orig['Total Labor Income'])
    data = data_orig[['Age', 'Total Labor Income',
        'Total Capital Income', 'Adjusted Total income',
        'Effective Tax Rate', 'Weights']]

    # Clean up the data by dropping outliers
    # drop all obs with AETR > 0.5
    data_trnc = data.drop(data[data['Effective Tax Rate'] >0.5].index)
    # drop all obs with AETR < -0.15
    data_trnc = data_trnc.drop(data_trnc[data_trnc['Effective Tax Rate']
                < -0.15].index)
    # drop all obs with ATI < $5
    data_trnc = \
        data_trnc.drop(data_trnc[data_trnc['Adjusted Total income'] < 5]
        .index)

    # Create an array of the different ages in the data
    min_age = int(np.maximum(data_trnc['Age'].min(), s_min))
    max_age = int(np.minimum(data_trnc['Age'].max(), s_max))
    ages_list = np.arange(min_age, max_age+1)

    for s in ages_list:
        '''
        ----------------------------------------------------------------
        Load OSPC Tax Calculator Data into a Dataframe
        ----------------------------------------------------------------
        s            = integer >= s_min, current age of tax function
                       estimation
        df           = (I3 x 6) DataFrame, data_trnc for age=s
        df_trnc_gph  = (I4 x 6) DataFrame, truncated data for 3D graph
        inc_lab_gph  = (I4 x 1) vector, total labor income for 3D graph
        inc_cap_gph  = (I4 x 1) vector, total capital income for 3D
                       graph
        etr_data_gph = (I4 x 1) vector, effective tax rate data for 3D
                       graph
        df_trnc      = (I5 x 6) DataFrame, truncated data for parameter
                       estimation
        inc_lab      = (I5 x 1) vector, total labor income for parameter
                       estimation
        inc_cap      = (I5 x 1) vector, total capital income for
                       parameter estimation
        etr          = (I5 x 1) vector, effective tax rate data for
                       parameter estimation
        wgts         = (I5 x 1) vector, population weights for each
                       for each observation for parameter estimation
        Obs          = integer, number of observations in sample
        X1           = (Obs x 1) vector, ?
        X2           = (Obs x 1) vector, ?
        X3           = (Obs x 1) vector, ?
        X4           = (Obs x 1) vector, ?
        X5           = (Obs x 1) vector, ?
        X6           = (Obs x 1) vector, ?
        ----------------------------------------------------------------
        '''
        print "year=", t, "Age=", s
        df = data_trnc[data_trnc['Age'] == s]
        # Don't estimate function if obs < 600
        if df.shape[0] < 600:
            param_arr[s-21, t-beg_yr, :] = np.nan
        else:
            if desc_data == True:
                # print some desciptive stats
                print 'Descriptive Statistics for age == ', s
                print df.describe()

            if graph_data == True:
                # Create 3-D scatterplot of effective tax rate as a
                # function of labor income and capital income
                df_trnc_gph = df[(df['Total Labor Income'] > 5) &
                              (df['Total Labor Income'] < 500000) &
                              (df['Total Capital Income'] > 5) &
                              (df['Total Capital Income'] < 500000)]
                inc_lab_gph = df_trnc_gph['Total Labor Income']
                inc_cap_gph = df_trnc_gph['Total Capital Income']
                etr_data_gph = df_trnc_gph['Effective Tax Rate']

                fig = plt.figure()
                ax = fig.add_subplot(111, projection ='3d')
                ax.scatter(inc_lab_gph, inc_cap_gph, etr_data_gph,
                    c='r', marker='o')
                ax.set_xlabel('Total Labor Income')
                ax.set_ylabel('Total Capital Income')
                ax.set_zlabel('Effective Tax Rate')
                plt.title('ETR, Lab. inc., and Cap. inc., Age=' + str(s) + ', Year=' + str(t))
                plt.show()

            df_trnc = df[(df['Total Labor Income'] > 5) &
                      (df['Total Capital Income'] > 5)]
            inc_lab = df_trnc['Total Labor Income']
            inc_cap = df_trnc['Total Capital Income']
            etr = df_trnc['Effective Tax Rate']
            wgts = df_trnc['Weights']
            Obs = inc_lab.shape[0]
            X1 = (inc_lab ** 2).reshape((Obs, 1))
            X2 = (inc_cap ** 2).reshape((Obs, 1))
            X3 = (inc_lab * inc_cap).reshape((Obs, 1))
            X4 = inc_lab.reshape((Obs, 1))
            X5 = inc_cap.reshape((Obs, 1))
            X6 = np.ones((Obs, 1))
            varmat = np.hstack((X1, X2, X3, X4, X5, X6))
            varmat_bar = varmat.mean(axis=0)
            varmat_hat = (varmat - varmat_bar)/varmat_bar
            varmat_hat[:, 5] = np.ones(Obs)
            Atil_init = 0.5
            Btil_init = 0.5
            Ctil_init = 0.5
            Dtil_init = 0.5
            Etil_init = 0.5
            F_init = 0.5
            max_x_init = etr[(df_trnc['Total Capital Income']
                         <3000)].max()
            min_x_init = etr[(df_trnc['Total Capital Income']
                         <3000)].min()
            max_y_init = etr[(df_trnc['Total Labor Income']
                         <3000)].max()
            min_y_init = etr[(df_trnc['Total Labor Income']
                         <3000)].min()
            params_init = np.array([Atil_init, Btil_init, Ctil_init,
                Dtil_init, Etil_init, F_init, max_x_init, min_x_init,
                max_y_init, min_y_init])
            varmat_barm1 = varmat_bar[:-1]
            phi = (X4 / (X4 + X5)).reshape(Obs)
            tau_objs = (varmat_hat, etr, wgts, varmat_barm1, phi)
            bnds = ((1e-12, None), (1e-12, None), (1e-12, None),
                   (1e-12, None), (1e-12, None), (1e-12, None),
                   (1e-12, None), (None, None), (1e-12, None),
                   (None, None))
            params_til = opt.minimize(wsumsq, params_init,
                args=(tau_objs), method="L-BFGS-B", bounds=bnds,
                tol=1e-15)
            (Atil, Btil, Ctil, Dtil, Etil, F, max_x, min_x, max_y,
                min_y) = params_til.x
            Gtil = params_til.x[:5].sum()
            P_num = np.dot(varmat_hat[:, :-1], params_til.x[:5]) + Gtil
            P_den = np.dot(varmat_hat, params_til.x[:6]) + Gtil
            etr_est = \
                ((phi * (max_x - min_x) + (1 - phi) * (max_y - min_y)) *
                (P_num / P_den) + (phi * min_x + (1 - phi) * min_y))
            params = np.zeros(numparams)
            params[:5] = params_til.x[:5] / varmat_bar[:5]
            A, B, C, D, E = params[:5]
            params[5] = F
            params[6:] = [max_x, min_x, max_y, min_y]
            param_arr[s-21, t-beg_yr, :] = params

            if graph_est == True:
                # Generate 3-D graph of predicted surface and
                # scatterplot of data
                fig = plt.figure()
                ax = fig.add_subplot(111, projection ='3d')
                ax.scatter(inc_lab, inc_cap, etr, c='r', marker='o')
                ax.set_xlabel('Total Labor Income')
                ax.set_ylabel('Total Capital Income')
                ax.set_zlabel('Effective Tax Rate')
                plt.title('ETR vs. Predicted ETR: Age=' + str(s) + ', Year=' + str(t))

                gridpts = 50
                inc_lab_vec = np.exp(np.linspace(np.log(5),
                              np.log(inc_lab.max()), gridpts))
                inc_cap_vec = np.exp(np.linspace(np.log(5),
                              np.log(inc_cap.max()), gridpts))
                inc_lab_grid, inc_cap_grid = np.meshgrid(inc_lab_vec,
                                             inc_cap_vec)
                etr_grid = gen_etr_grid(inc_lab_grid, inc_cap_grid,
                           params)
                ax.plot_surface(inc_lab_grid, inc_cap_grid, etr_grid,
                    cmap=cmap1, linewidth=0)
                plt.show()

            if dmtrgr_est == True:
                # Generate 3-D graph of predicted derivative of marginal
                # tax rates
                gridpts = 50
                inc_lab_vec = np.exp(np.linspace(np.log(5),
                              np.log(inc_lab.max()), gridpts))
                inc_cap_vec = np.exp(np.linspace(np.log(5),
                              np.log(inc_cap.max()), gridpts))
                inc_lab_grid, inc_cap_grid = np.meshgrid(inc_lab_vec,
                                             inc_cap_vec)
                dmtrx_grid = gen_dmtrx_grid(inc_lab_grid, inc_cap_grid,
                             params)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection ='3d')
                ax.plot_surface(inc_lab_grid, inc_cap_grid, dmtrx_grid,
                    cmap=cmap1, linewidth=0)
                ax.set_xlabel('Total Labor Income')
                ax.set_ylabel('Total Capital Income')
                ax.set_zlabel('d MTR labor inc.')
                plt.title('d MTR labor inc.: Age=' + str(s) + ', Year=' + str(t))
                print dmtrx_grid.min(), dmtrx_grid.max()
                plt.show()

                dmtry_grid = gen_dmtry_grid(inc_lab_grid, inc_cap_grid,
                             params)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection ='3d')
                ax.plot_surface(inc_lab_grid, inc_cap_grid, dmtry_grid,
                    cmap=cmap1, linewidth=0)
                ax.set_xlabel('Total Labor Income')
                ax.set_ylabel('Total Capital Income')
                ax.set_zlabel('d MTR capital inc.')
                plt.title('d MTR capital inc.: Age=' + str(s) + ', Year=' + str(t))
                print dmtry_grid.min(), dmtry_grid.max()
                plt.show()

elapsed_time = time.clock() - start_time

# Print tax function computation time
if elapsed_time < 60: # seconds
    secs = round(elapsed_time, 3)
    print 'Tax function estimation time: ', secs, ' sec.'
elif elapsed_time >= 60 and elapsed_time < 3600: # minutes
    mins = int(elapsed_time / 60)
    secs = round(((elapsed_time / 60) - mins) * 60, 1)
    print 'Tax function estimation time: ', mins, ' min, ', secs, ' sec'

# Save tax function parameters array and computation time in pickle
dict_params = dict([('tfunc_params', param_arr), ('tfunc_time', elapsed_time)])
pkl_path = "TxFuncEst.pkl"
pickle.dump(dict_params, open(pkl_path, "wb"))
