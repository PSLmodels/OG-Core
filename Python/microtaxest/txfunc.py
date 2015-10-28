'''
------------------------------------------------------------------------
This script reads in data generated from the OSPC Tax Calcultor and
the 2009 IRS PUF.  It then estimates tax functions tau_{s,t}(x,y), where
tau_{s,t} is the effective tax rate for a given age (s) in a particular
year (t), x is total labor income, and y is total capital income.

This Python script calls the following functions:


This Python script outputs the following:

------------------------------------------------------------------------
'''

# Import packages
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
------------------------------------------------------------------------
Set parameters and create objects for output
------------------------------------------------------------------------
s_min     = integer > 0, minimum age relevant to the model
s_max     = integer > s_min, maximum age relevant to the model
taxparams =
------------------------------------------------------------------------
'''
s_min = int(21)
s_max = int(100)
tpers = int(10)
numparams = int(7)
param_arr = np.zeros((s_max - s_min + 1, tpers, numparams))
desc_data = True
graph_data = True
graph_est = True
cmap1 = matplotlib.cm.get_cmap('summer')
# cmap1 = matplotlib.cm.get_cmap('jet')
# cmap1 = matplotlib.cm.get_cmap('coolwarm')

beg_yr = int(2015)
end_yr = int(2015)
years_list = np.arange(beg_yr, end_yr + 1)
data_folder = '/Users/rwe2/Documents/OSPC/Data/micro-dynamic/'

'''
------------------------------------------------------------------------
Define Functions
------------------------------------------------------------------------
'''

def gen_etr_grid(X, Y, params, G):
    '''
    --------------------------------------------------------------------
    This function generates a grid of effective tax rates from a grid of
    total labor income and a grid of total capital income.
    --------------------------------------------------------------------
    '''
    A, B, C, D, E, F = params
    P_num = (A * (X ** 2) + B * (Y ** 2) + C * (X * Y) + D * X + E * Y
            + A + B + C + D + E)
    P_den = (A * (X ** 2) + B * (Y ** 2) + C * (X * Y) + D * X + E * Y
            + F + A + B + C + D + E)
    etr_grid = P_num / P_den - G
    return etr_grid


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
    varmat_hat, etr, wgts, G, varmat_bar = objs
    Htil = np.dot(varmat_bar, params[:-1])
    P_num = np.dot(varmat_hat[:,:-1], params[:-1]) + Htil
    P_den = np.dot(varmat_hat, params) + Htil
    etr_est = P_num / P_den - G
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
    data_path = string, path to directory address of data
    data_orig = (I x 12) DataFrame, I observations, 12 variables
    data      = (I x 6) DataFrame, I observations, 6 variables
    data_trnc = (I2 x 6) DataFrame, I2 observations (I2<I), 6 variables
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
    # min_age = int(np.maximum(data_trnc['Age'].min(), s_min))
    # max_age = int(np.minimum(data_trnc['Age'].max(), s_max))
    min_age = int(45)
    max_age = int(45)
    ages_list = np.arange(min_age, max_age+1)

    for s in ages_list:
        df = data_trnc[data_trnc['Age'] == s]
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
                inc_lab = df_trnc_gph['Total Labor Income']
                inc_cap = df_trnc_gph['Total Capital Income']
                etr_data = df_trnc_gph['Effective Tax Rate']


                fig = plt.figure()
                ax = fig.add_subplot(111, projection ='3d')
                ax.scatter(inc_lab, inc_cap, etr_data, c='r', marker='o')
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
            G = np.maximum(0, -etr.min() + 0.1)
            Atil_init = 0.1
            Btil_init = 0.1
            Ctil_init = 0.5
            Dtil_init = 0.5
            Etil_init = 0.5
            Ftil_init = 0.5
            params_init = np.array([Atil_init, Btil_init, Ctil_init,
                Dtil_init, Etil_init, Ftil_init])
            varmat_barm1 = varmat_bar[:-1]
            tau_objs = (varmat_hat, etr, wgts, G, varmat_barm1)
            bnds = ((1e-12, None), (1e-12, None), (1e-12, None),
                   (1e-12, None), (1e-12, None), (1e-12, None))
            params_til = opt.minimize(wsumsq, params_init,
                args=(tau_objs), method="L-BFGS-B", bounds=bnds,
                tol=1e-15)
            Atil, Btil, Ctil, Dtil, Etil, Ftil = params_til.x
            Htil = (params_til.x[:-1] * varmat_bar[:-1]).sum()
            P_num = np.dot(varmat_hat[:,:-1], params_til.x[:-1]) + Htil
            P_den = np.dot(varmat_hat, params_til.x) + Htil
            etr_est = P_num / P_den - G
            # etr_est = np.exp(-np.dot(varmat_hat, params_til.x)) - G
            params = np.zeros(numparams - 1)
            params[:-1] = params_til.x[:-1] / varmat_bar[:-1]
            A, B, C, D, E = params[:-1]
            F = params_til.x[-1]
            params[-1] = F
            param_arr[s-21, t-beg_yr, :] = np.hstack((params, G))

            if graph_est == True:
                # Generate 3-D graph here of predicted surface and
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
                           params, G)
                ax.plot_surface(inc_lab_grid, inc_cap_grid, etr_grid,
                    cmap=cmap1, linewidth=0)
                plt.show()


