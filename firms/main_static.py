'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
solver for the OG model with S-period lived agents, exogenous labor, 
M industries, and I goods.

This Python script calls the following other file(s) with the associated
functions:
    sfuncs.py
        feasible
        SS
    tpfuncs.py
        TPI
------------------------------------------------------------------------
'''

# Import packages
import time
import numpy as np
import scipy.optimize as opt
import pandas as pd
import ssfuncs_static as ssf
reload(ssf)
import tpfuncs_static as tpf
reload(tpf)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S            = integer in [3,80], number of periods an individual lives
T            = integer > S, number of time periods until steady state
I            = integer, number of consumption goods
alpha        = [I,] vector, ith element is the expenditure share on 
              good i (elements must sum to one)
c_bar     = [I,] vector, ith element is the minimum consumption 
              amount for good i
beta_ann     = scalar in [0,1), discount factor for one year
beta         = scalar in [0,1), discount factor for each model period
sigma        = scalar > 0, coefficient of relative risk aversion
n         = [S,] vector, exogenous labor supply n_{s,t}
M            = integer, number of production industries
A            = [M,] vector, mth element is the total factor productivity 
              values for the mth industry
gamma        = [M,] vector, mth element is capital's share of income 
              for the mth industry 
epsilon      = [M,] vector, mth element is the elasticity of substitution 
              between capital and labor for the mth industry
delta_annual = [M,] vector, mth element is the one-year physical depreciation 
              rate of capital in the mth industry
delta        = [M,] vector, mth element is the model-period physical depreciation 
              rate of capital in the mth industry
xi           = [M,M] matrix, element i,j gives the fraction of capital used by 
               industry j that comes from the output of industry i
pi           = [I,M] matrix, element i,j gives the fraction of consumption
               good i that comes from the output of industry j
ss_tol       = scalar > 0, tolerance level for steady-state fsolve
ss_graphs    = boolean, =True if want graphs of steady-state objects
tp_solve     = boolean, =True if want to solve TPI after solving SS
tp_tol       = scalar > 0, tolerance level for fsolve's in TP 
tp_graphs    = boolean, =True if want graphs of TP objects
------------------------------------------------------------------------
'''
# Household parameters
S = int(80)
T = 220
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 3.0
n = np.zeros(S)
n[:int(round(2 * S / 3))] = 1.
n[int(round(2 * S / 3)):] = 0.9

# Model calibation parameters
FR1993_calib = False # if True, then calibration firm params to 
      # Fullerton and Rogers (Brookings, 1993)

if FR1993_calib == True:
  # Specify model dimensions
  I = 17 # number of consumption goods
  M = 19 # number of production industries

  # Read in parameters from Fullerton and Rogers (1993) from excel file
  xi_sheet = pd.read_excel('Firm_Parameters_FullertonRogers.xlsx', sheetname='xi')
  xi = xi_sheet.as_matrix() # turn into numpy array
  xi=xi[0:19,1:20] # keep only cells interested in
  xi=xi.astype(float) # make sure type is float
  xi = (xi/np.tile(xi.sum(0),(M,1))).transpose() # make xi so fractions and so rows are capital used in and columns are capital supplied in (MxM)
  #print 'xi sum check: ', xi.sum(1)

  pi_sheet = pd.read_excel('Firm_Parameters_FullertonRogers.xlsx', sheetname='pi')
  pi = pi_sheet.as_matrix()
  pi=pi[0:19,1:18]
  pi=pi.astype(float)
  pi = (pi/np.tile(pi.sum(0),(M,1))).transpose() # make pi so fractions and so rows are consumption goods and columns are output industries in (IxM)
  #print 'pi sum check: ', pi.sum(1)

  delta_sheet = pd.read_excel('Firm_Parameters_FullertonRogers.xlsx', sheetname='delta')
  delta = delta_sheet.as_matrix()
  delta=delta[0:19,1]
  delta=delta.astype(float)
  #print 'delta shape: ', delta.shape

  gamma_sheet = pd.read_excel('Firm_Parameters_FullertonRogers.xlsx', sheetname='gamma')
  gamma = gamma_sheet.as_matrix()
  gamma=gamma[0:19,1]
  gamma=gamma.astype(float)

  epsilon_sheet = pd.read_excel('Firm_Parameters_FullertonRogers.xlsx', sheetname='epsilon')
  epsilon = epsilon_sheet.as_matrix()
  epsilon=epsilon[0:19,1]
  epsilon=epsilon.astype(float)

  alpha_sheet = pd.read_excel('Firm_Parameters_FullertonRogers.xlsx', sheetname='alpha')
  alpha = alpha_sheet.as_matrix()
  alpha =alpha[0:17,1]
  alpha =alpha.astype(float)
  alpha = alpha/alpha.sum() # ensure the alpha vector sums to one 

  cbar_sheet = pd.read_excel('Firm_Parameters_FullertonRogers.xlsx', sheetname='cbar')
  c_bar = cbar_sheet.as_matrix()
  c_bar =c_bar[0:17,1]
  c_bar =c_bar.astype(float)

  # No TFP from FR1993, so just set to one for all
  A = np.ones((M,))

else: 
# Firm/consumption parameters
  I = 2
  alpha = np.array([0.4,0.6])
  #c_bar = np.array([0.6, 0.6])
  c_bar = np.array([0.0, 0.0])
  
  M = 2
  A = np.array([1, 1.2])
  gamma = np.array([0.15, 0.2])
  epsilon = np.array([0.6, 0.6])
  delta_annual = np.array([0.04,0.05])
  delta = 1 - ((1-delta_annual)**(80/S))
  xi = np.array([[1.0, 0.0],[0.0, 1.0] ]) 
  pi = np.array([[1.0, 0.0],[0.0, 1.0] ]) 


  # M = 3
  # A = np.array([1, 1.2, 0.9])
  # gamma = np.array([0.3, 0.25, 0.4])
  # delta = np.array([0.1, 0.12, 0.15])
  # epsilon = np.array([0.55, 0.6, 0.62])
  # pi = np.array([[0.4, 0.3, 0.3],[0.1, 0.8, 0.1]]) 
  # xi = np.array([[0.2, 0.6, 0.2],[0.0, 0.2, 0.8], [0.6, 0.2, 0.2] ])

# SS parameters
ss_tol = 1e-13
ss_graphs = False
# TP parameters
tp_solve = True
tp_graphs = False
tp_tol = 1e-9 # tolerance for fsolve for TP and for HH prob along time path

'''
------------------------------------------------------------------------
Compute the steady state
------------------------------------------------------------------------
rbar_init    = scalar > 1, initial guess for steady-state model period
               interest rate
wbar_init    = scalar > 1, initial guess for steady-state real wage
rwbar_init   = [2,] vector, initial guesses for steady-state r and w
feas_params  = length 5 tuple, parameters for feasible function:
               (S, alpha, beta, sigma, ss_tol)
b_guess      = [S-1,] vector, initial guess for savings to use in fsolve
               in ssf.get_cbess
GoodGuess    = boolean, =True if initial steady-state guess is feasible
r_cstr_ss    = boolean, =True if initial r + delta <= 0
w_cstr_ss    = boolean, =True if initial w <= 0
c_tilde_cstr_ss    = [S,] boolean vector, =True if c_tilde_{s}<=0 for initial r and w
c_cstr_ss   = [I, S] boolean matrix, =True if c_{i,s}<=0 for initial r
               and w
K_cstr_ss = boolean, =True if sum of K_{m}<=0 for initial r and w
ss_params    = length 5 tuple, parameters for SS function:
               (S, alpha, beta, sigma, ss_tol)
r_ss         = scalar, steady-state interest rate
w_ss         = scalar > 0, steady-state wage
p_c_ss        = [I,] vector, steady-state prices for each consumption good
p_tilde_ss   = scalar > 0, steady-state composite good price
b_ss         = [S-1,] vector, steady-state savings
c_tilde_ss         = [S,] vector, steady-state composite consumption
c_ss        = [I,S] matrix, steady-state consumption of each good
eul_ss       = [S-1,] vector, steady-state Euler errors
Cm_ss        = [M,] vector, total demand for goods from each industry
X_ss        = [M,] vector, steady-state total output for each industry
K_ss        = [M,] vector, steady-state capital demand for each industry
L_ss        = [M,] vector, steady-state labor demand for each industry
MCK_err_ss   = scalar, steady-state capital market clearing error
MCL_err_ss   = scalar, steady-state labor market clearing error
ss_time      = scalar, number of seconds to compute SS solution
rcdiff_ss   = [M,] vector, steady-state difference in goods market
               clearing (resource constraint) in each industry
------------------------------------------------------------------------
'''

# Make sure initial guess of r and w is feasible
rbar_init = ((1 + 0.04) ** (80 / S)) - 1
wbar_init = 1.
rwbar_init = np.array([rbar_init, wbar_init])
feas_params = (S, alpha, beta, sigma, ss_tol)
b_guess = np.zeros(S-1)
b_guess[:int(round(2 * S / 3))] = \
    (np.linspace(0.003, 0.3, int(round(2 * S / 3))))
b_guess[int(round(2 * S / 3)):] = \
    (np.linspace(0.3, 0.003, S - 1 - int(round(2 * S / 3))))
GoodGuess, r_cstr_ss, w_cstr_ss, c_tilde_cstr_ss, c_cstr_ss, K_cstr_ss \
    = ssf.feasible(feas_params, rwbar_init, b_guess, c_bar, A,
    gamma, epsilon, delta, pi, I, S, n)

if r_cstr_ss == True and w_cstr_ss == True:
    print 'Initial guess is not feasible because both r + delta, w <= 0.'
elif r_cstr_ss == True and w_cstr_ss == False:
    print 'Initial guess is not feasible because r + delta <= 0.'
elif r_cstr_ss == False and w_cstr_ss == True:
    print 'Initial guess is not feasible because w <= 0.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_tilde_cstr_ss.max() == 1
  and K_cstr_ss == False):
    print 'Initial guess is not feasible because c_tilde_{s}<=0 for some s.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_tilde_cstr_ss.max() == 1
  and K_cstr_ss == True):
    print 'Initial guess is not feasible because c_tilde_{s}<=0 for some s and sum of K_{m}<=0.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_tilde_cstr_ss.max() == 0
  and c_cstr_ss.max() == 1 and K_cstr_ss == False):
    print 'Initial guess is not feasible because c_{i,s}<=0 for some i and s.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_tilde_cstr_ss.max() == 0
  and c_cstr_ss.max() == 1 and K_cstr_ss == True):
    print 'Initial guess is not feasible because c_{i,s}<=0 for some i and s and sum of K_{m}<=0.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_tilde_cstr_ss.max() == 0
  and c_cstr_ss.max() == 0 and K_cstr_ss == True):
    print 'Initial guess is not feasible because sum of K_{m}<=0.'
elif GoodGuess == True:
    print 'Initial guess is feasible.'

    # Compute steady state
    print 'BEGIN STEADY STATE COMPUTATION'
    ss_params = (S, alpha, beta, sigma, ss_tol)
    (r_ss, w_ss, p_c_ss, p_tilde_ss, b_ss, c_tilde_ss, c_ss, eul_ss, C_ss, X_ss,
        K_ss, L_ss, MCK_err_ss, MCL_err_ss, ss_time) = \
        ssf.SS(ss_params, rwbar_init, b_guess, c_bar, A,
        gamma, epsilon, delta, xi, pi, I, M, S, n, ss_graphs)

    # Print diagnostics
    print 'The maximum absolute steady-state Euler error is: ', \
        np.absolute(eul_ss).max()
    print 'The capital and labor market clearing errors are: ', \
        (MCK_err_ss, MCL_err_ss)
    print 'The steady-state distribution of capital is:'
    print b_ss
    print 'The steady-state distribution of composite consumption is:'
    print c_tilde_ss
    print 'The steady-state distribution of goods consumption is:'
    print c_ss
    print 'The steady-state interest rate and wage:'
    print np.array([r_ss, w_ss])
    print 'Steady-state consumption good prices and composite price are:'
    print p_c_ss, p_tilde_ss
    print 'Aggregate output, capital stock and consumption for each industry/consumption good are:'
    print np.array([[X_ss], [K_ss], [C_ss]])
    RCdiff_ss = X_ss - (np.dot(np.reshape(C_ss,(1,I)),pi)) - (np.dot(delta*K_ss,xi)) 
    print 'The difference in the resource constraints are: ', RCdiff_ss

    # Print SS computation time
    if ss_time < 60: # seconds
        secs = round(ss_time, 3)
        print 'SS computation time: ', secs, ' sec'
    elif ss_time >= 60 and ss_time < 3600: # minutes
        mins = int(ss_time / 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', mins, ' min, ', secs, ' sec'
    elif ss_time >= 3600 and ss_time < 86400: # hours
        hrs = int(ss_time / 3600)
        mins = int(((ss_time / 3600) - hrs) * 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'
    elif ss_time >= 86400: # days
        days = int(ss_time / 86400)
        hrs = int(((ss_time / 86400) - days) * 24)
        mins = int(((ss_time / 3600) - hrs) * 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', days, ' days,', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'

    '''
    --------------------------------------------------------------------
    Compute the equilibrium time path by TPI
    --------------------------------------------------------------------
    Gamma1      = [S-1,] vector, initial period savings distribution
    rpath_init  = [T+S-1,] vector, initial guess for the time path of
                  the interest rate
    r1          = scalar > 0, guess for period 1 value of r
    cc_r        = scalar, parabola coefficient for rpath_init
    bb_r        = scalar, parabola coefficient for rpath_init
    aa_r        = scalar, parabola coefficient for rpath_init
    wpath_init  = [T+S-1,] vector, initial guess for the time path of
                  the wage
    w1          = scalar > 0, guess for period 1 value of w
    cc_w        = scalar, parabola coefficient for wpath_init
    bb_w        = scalar, parabola coefficient for wpath_init
    aa_w        = scalar, parabola coefficient for wpath_init
    tp_params   = length 11 tuple, parameters to pass into TP function:
                  (S, T, alpha_path, beta, sigma, r_ss, w_ss, tp_tol)
    alpha_path  = [I,T+S-2] matrix, consumption good shares in each 
                  period along time path
    c_bar_path = [I,T+S-2] matrix, minimum consumption amounts in each 
                     period along time path
    A_path        = [M,T+S-2] matrix, TFP for each industry in each 
                     period along time path
    gamma_path    = [M,T+S-2] matrix, capital's share of output for each industry in each 
                     period along time path
    epsilon_path  = [M,T+S-2] matrix, elasticity of substitution for each industry in each 
                     period along time path
    delta_path    = [M,T+S-2] matrix, physical depreciation for each industry in each 
                     period along time path


    r_path      = [T+S-2,] vector, equilibrium time path of the interest
                  rate
    w_path      = [T+S-2,] vector, equilibrium time path of the wage
    pm_path     = [M, T+S-2] matrix, equilibrium time path of industry
                  output prices
    pc_path     = [I, T+S-2] matrix, equilibrium time path of consumption
                  good prices
    p_tilde_path      = [T+S-2,] vector, equilibrium time path of the
                  composite good price
    b_path      = [S-1, T+S-2] matrix, equilibrium time path of the
                  distribution of savings. Period 1 is the initial
                  exogenous distribution
    c_tilde_path      = [S, T+S-2] matrix, equilibrium time path of the
                  distribution of composite good consumption
    c_path     = [S, T+S-2, I] array, equilibrium time path of the
                  distribution of individual consumption goods
    eul_path    = [S-1, T+S-2] matrix, equilibrium time path of the
                  euler errors associated with the distribution of
                  savings. Period 1 is a column of zeros
    C_path     = [I, T+S-2] matrix, equilibrium time path of total
                  demand for each consumption good
    X_path     = [M, T+S-2] matrix, equilibrium time path of total
                  output from each industry
    K_path     = [M, T+S-2] matrix, equilibrium time path of capital
                  demand for each industry
    L_path     = [M, T+S-2] matrix, equilibrium time path of labor
                  demand for each industry
    Inv_path   = [M,T+S-2] matrix, equilibrium time path for investment
                   demand from each industry
    X_c_path    = [M,T+S-2] matrix, equlibirum time path for demand
                   for output from each industry from consumption demand
    X_inv_path  = [M,T+S-2] matrix, equlibirum time path for demand
                   for output from each industry from investment demand
    MCKerr_path = [T+S-2,] vector, equilibrium time path of capital
                  market clearing errors
    MCLerr_path = [T+S-2,] vector, equilibrium time path of labor market
                  clearing errors
    tpi_time    = scalar, number of seconds to solve for transition path
    ResmDiff    = [M, T-1] matrix, errors in the resource constraint
                  from period 1 to T-1. We don't use T because we are
                  missing one individual's consumption in that period
    --------------------------------------------------------------------
    '''
    if tp_solve == True:
        print 'BEGIN EQUILIBRIUM TIME PATH COMPUTATION'
        #Gamma1 = b_ss
        Gamma1 = 0.95 * b_ss
        # Make sure initial savings distr. is feasible (sum of b_{s}>0)
        if Gamma1.sum() <= 0:
            print 'Initial savings distribution is not feasible (sum of b_{s}<=0)'
        else:
            # Choose initial guesses of path of interest rate and wage.
            # Use parabola specification aa*x^2 + bb*x + cc
            # rpath_init = r_ss * np.ones(T+S-1)
            rpath_init = np.zeros(T+S-1)
            r1 = 1.02 * r_ss
            cc_r = r1
            bb_r = - 2 * (r1 - r_ss) / (T - S)
            aa_r = -bb_r / (2 * (T - S))
            rpath_init[:T-S+1] = (aa_r * (np.arange(0, T-S+1) ** 2) +
                             (bb_r * np.arange(0, T-S+1)) + cc_r)
            rpath_init[T-S+1:] = r_ss
            #rpath_init[:] = r_ss


            wpath_init = np.zeros(T+S-1)
            w1 = 0.98 * w_ss
            cc_w = w1
            bb_w = - 2 * (w1 - w_ss) / (T - S)
            aa_w = -bb_w / (2 * (T - S))
            wpath_init[:T-S+1] = (aa_w * (np.arange(0, T-S+1) ** 2) +
                             (bb_w * np.arange(0, T-S+1)) + cc_w)
            wpath_init[T-S+1:] = w_ss
            #wpath_init[:] = w_ss

            # Solve for time path
            # Tile arrays of time path parameters so easy to handle in 
            # TP functions
            alpha_path = np.tile(np.reshape(alpha,(I,1)),(1,len(rpath_init)))
            c_bar_path = np.tile(np.reshape(c_bar,(I,1)),(1,len(rpath_init)))
            A_path = np.tile(np.reshape(A,(M,1)),(1,len(rpath_init)))
            gamma_path = np.tile(np.reshape(gamma,(M,1)),(1,len(rpath_init)))
            epsilon_path = np.tile(np.reshape(epsilon,(M,1)),(1,len(rpath_init)))
            delta_path = np.tile(np.reshape(delta,(M,1)),(1,len(rpath_init)))

            tp_params = (S, T, alpha_path, beta, sigma, r_ss, w_ss, tp_tol)

            guesses = np.append(rpath_init[:T], wpath_init[:T])
            start_time = time.clock()
            solutions = opt.fsolve(tpf.TP_fsolve, guesses, args=(tp_params, K_ss, X_ss,
               Gamma1, c_bar_path, A_path, gamma_path, epsilon_path, delta_path, xi, pi, I, M, S, n,
               tp_graphs), xtol=tp_tol, col_deriv=1)
            #solutions = tpf.TP_fsolve(guesses, tp_params, K_ss, X_ss,
            #   Gamma1, c_bar_path, A_path, gamma_path, epsilon_path, delta_path, xi, pi, I, M, S, n,
            #   tp_graphs)
            tpi_time = time.clock() - start_time
            rpath = solutions[:T].reshape(T)
            wpath = solutions[T:].reshape(T)


            # run one iteration of TP with fsolve solution to get other output
            tp_params = (S, T, alpha_path, beta, sigma, r_ss, w_ss, tp_tol)
            (r_path, w_path, pc_path, p_tilde_path, b_path, c_tilde_path, c_path,
                eul_path, C_path, X_path, K_path, L_path,
                MCKerr_path, MCLerr_path, RCdiff_path) = \
                tpf.TP(tp_params, rpath, wpath, K_ss, X_ss,
                Gamma1, c_bar_path, A_path, gamma_path, epsilon_path, delta_path, xi, pi, I, 
                M, S, n, tp_graphs)
                


            # Print diagnostics
            print 'The max. absolute difference in the resource constraints are:'
            print np.absolute(RCdiff_path).max(axis=1)
            print 'The max. absolute error in the market clearing conditions are:'
            print np.absolute(MCKerr_path).max(), np.absolute(MCLerr_path).max()




            # Print TPI computation time
            if tpi_time < 60: # seconds
                secs = round(tpi_time, 3)
                print 'TPI computation time: ', secs, ' sec'
            elif tpi_time >= 60 and tpi_time < 3600: # minutes
                mins = int(tpi_time / 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', mins, ' min, ', secs, ' sec'
            elif tpi_time >= 3600 and tpi_time < 86400: # hours
                hrs = int(tpi_time / 3600)
                mins = int(((tpi_time / 3600) - hrs) * 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'
            elif tpi_time >= 86400: # days
                days = int(tpi_time / 86400)
                hrs = int(((tpi_time / 86400) - days) * 24)
                mins = int(((tpi_time / 3600) - hrs) * 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', days, ' days,', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'


