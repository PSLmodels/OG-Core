'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents, exogenous
labor, and two industries and two goods.

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
import numpy as np
import scipy.optimize as opt
import ssfuncs as ssf
reload(ssf)
import tpfuncs as tpf
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
ci_tilde     = [I,] vector, ith element is the minimum consumption 
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
I = 2
alpha = np.array([0.4,0.6])
ci_tilde = np.array([0.6, 0.6])
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 3.0
n = np.zeros(S)
n[:int(round(2 * S / 3))] = 1.
n[int(round(2 * S / 3)):] = 0.9

# Firm parameters
M = 2 
A = np.array([1, 1.2])
gamma = np.array([0.15, 0.2])
epsilon = np.array([0.6, 0.6])
delta_annual = np.array([0.04,0.05])
delta = 1 - ((1-delta_annual)**(80/S))
xi = np.array([[1.0, 0.0],[0.0, 1.0] ]) 
pi = np.array([[1.0, 0.0],[0.0, 1.0] ]) 

# SS parameters
ss_tol = 1e-13
ss_graphs = False
# TP parameters
tp_solve = True
tp_graphs = True
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
c_cstr_ss    = [S,] boolean vector, =True if c_s<=0 for initial r and w
cm_cstr_ss   = [2, S] boolean matrix, =True if c_{m,s}<=0 for initial r
               and w
K_cstr_ss = boolean, =True if sum of K_{m}<=0 for initial r and w
ss_params    = length 5 tuple, parameters for SS function:
               (S, alpha, beta, sigma, ss_tol)
r_ss         = scalar, steady-state interest rate
w_ss         = scalar > 0, steady-state wage
pm_ss        = [2,] vector, steady-state prices in each industry
p_ss         = scalar > 0, steady-state composite good price
b_ss         = [S-1,] vector, steady-state savings
c_ss         = [S,] vector, steady-state composite consumption
cm_ss        = [2,S] matrix, steady-state consumption of each good
eul_ss       = [S-1,] vector, steady-state Euler errors
Cm_ss        = [2,] vector, total demand for goods from each industry
Ym_ss        = [2,] vector, steady-state total output for each industry
Km_ss        = [2,] vector, steady-state capital demand for each industry
Lm_ss        = [2,] vector, steady-state labor demand for each industry
MCK_err_ss   = scalar, steady-state capital market clearing error
MCL_err_ss   = scalar, steady-state labor market clearing error
ss_time      = scalar, number of seconds to compute SS solution
rcmdiff_ss   = [2,] vector, steady-state difference in goods market
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
GoodGuess, r_cstr_ss, w_cstr_ss, c_cstr_ss, cm_cstr_ss, K_cstr_ss \
    = ssf.feasible(feas_params, rwbar_init, b_guess, ci_tilde, A,
    gamma, epsilon, delta, pi, I, S, n)

if r_cstr_ss == True and w_cstr_ss == True:
    print 'Initial guess is not feasible because both r + delta, w <= 0.'
elif r_cstr_ss == True and w_cstr_ss == False:
    print 'Initial guess is not feasible because r + delta <= 0.'
elif r_cstr_ss == False and w_cstr_ss == True:
    print 'Initial guess is not feasible because w <= 0.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_cstr_ss.max() == 1
  and K1K2_cstr_ss == False):
    print 'Initial guess is not feasible because c_s<=0 for some s.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_cstr_ss.max() == 1
  and K1K2_cstr_ss == True):
    print 'Initial guess is not feasible because c_s<=0 for some s and sum of K_{m}<=0.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_cstr_ss.max() == 0
  and cm_cstr_ss.max() == 1 and K1K2_cstr_ss == False):
    print 'Initial guess is not feasible because c_{i,s}<=0 for some m and s.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_cstr_ss.max() == 0
  and cm_cstr_ss.max() == 1 and K1K2_cstr_ss == True):
    print 'Initial guess is not feasible because c_{i,s}<=0 for some m and s and sum of K_{m}<=0.'
elif (r_cstr_ss == False and w_cstr_ss == False and c_cstr_ss.max() == 0
  and cm_cstr_ss.max() == 0 and K_cstr_ss == True):
    print 'Initial guess is not feasible because sum of K_{m}<=0.'
elif GoodGuess == True:
    print 'Initial guess is feasible.'

    # Compute steady state
    print 'BEGIN STEADY STATE COMPUTATION'
    ss_params = (S, alpha, beta, sigma, ss_tol)
    (r_ss, w_ss, pm_ss, p_ss, b_ss, c_ss, cm_ss, eul_ss, Cm_ss, Ym_ss,
        Km_ss, Lm_ss, MCK_err_ss, MCL_err_ss, ss_time) = \
        ssf.SS(ss_params, rwbar_init, b_guess, ci_tilde, A,
        gamma, epsilon, delta, xi, pi, I, M, S, n, ss_graphs)

    # Print diagnostics
    print 'The maximum absolute steady-state Euler error is: ', \
        np.absolute(eul_ss).max()
    print 'The capital and labor market clearing errors are: ', \
        (MCK_err_ss, MCL_err_ss)
    print 'The steady-state distribution of capital is:'
    print b_ss
    print 'The steady-state distribution of composite consumption is:'
    print c_ss
    print 'The steady-state distribution of goods consumption is:'
    print cm_ss
    print 'The steady-state interest rate and wage:'
    print np.array([r_ss, w_ss])
    print 'Steady-state industry prices and composite price are:'
    print pm_ss, p_ss
    print 'Aggregate output, capital stock and consumption for each industry are:'
    print np.array([[Ym_ss], [Km_ss], [Cm_ss]])
    rcmdiff_ss = Ym_ss - Cm_ss - delta * Km_ss
    print 'The difference Ym_ss - Cm_ss - delta_m * Km_ss is: ', rcmdiff_ss

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
    ci_tilde_path = [I,T+S-2] matrix, minimum consumption amounts in each 
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
    p_path      = [T+S-2,] vector, equilibrium time path of the
                  composite good price
    b_path      = [S-1, T+S-2] matrix, equilibrium time path of the
                  distribution of savings. Period 1 is the initial
                  exogenous distribution
    c_path      = [S, T+S-2] matrix, equilibrium time path of the
                  distribution of composite good consumption
    ci_path     = [S, T+S-2, I] array, equilibrium time path of the
                  distribution of individual consumption goods
    eul_path    = [S-1, T+S-2] matrix, equilibrium time path of the
                  euler errors associated with the distribution of
                  savings. Period 1 is a column of zeros
    Ci_path     = [I, T+S-2] matrix, equilibrium time path of total
                  demand for each consumption good
    Ym_path     = [M, T+S-2] matrix, equilibrium time path of total
                  output from each industry
    Km_path     = [M, T+S-2] matrix, equilibrium time path of capital
                  demand for each industry
    Lm_path     = [M, T+S-2] matrix, equilibrium time path of labor
                  demand for each industry
    MCKerr_path = [T+S-2,] vector, equilibrium time path of capital
                  market clearing errors
    MCLerr_path = [T+S-2,] vector, equilibrium time path of labor market
                  clearing errors
    tpi_time    = scalar, number of seconds to compute TPI solution
    ResmDiff    = [M, T-1] matrix, errors in the resource constraint
                  from period 1 to T-1. We don't use T because we are
                  missing one individual's consumption in that period
    --------------------------------------------------------------------
    '''
    if tp_solve == True:
        print 'BEGIN EQUILIBRIUM TIME PATH COMPUTATION'
        #Gamma1 = b_ss
        Gamma1 = 0.95 * b_ss
        # Make sure initial savings distr. is feasible (sum of K_{m}>0)
        if Gamma1.sum() <= 0:
            print 'Initial savings distribution is not feasible (K1+K2<=0)'
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


            # wpath_init = w_ss * np.ones(T+S-1)
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
            alpha_path = np.tile(alpha,(1,len(rpath_init)))
            ci_tilde_path = np.tile(ci_tilde,(1,len(rpath_init)))
            A_path = np.tile(A,(1,len(rpath_init)))
            gamma_path = np.tile(gamma,(1,len(rpath_init)))
            epsilon_path = np.tile(epsilon,(1,len(rpath_init)))
            delta_path = np.tile(delta,(1,len(rpath_init)))

            tp_params = (S, T, alpha_path, beta, sigma, r_ss, w_ss, tp_tol)

            guesses = np.append(rpath_init[:T], wpath_init[:T])
            #solutions = opt.fsolve(tpf.TP_fsolve, guesses, args=(tp_params, Km_ss, Ym_ss,
            #   Gamma1, ci_tilde, A, gamma, epsilon, delta, n,
            #   tp_graphs), xtol=tp_tol, col_deriv=1)
            solutions = tpf.TP_fsolve(guesses, tp_params, Km_ss, Ym_ss,
               Gamma1, ci_tilde_path, A_path, gamma_path, epsilon_path, delta_path, xi, pi, I, M, S, n,
               tp_graphs)
            rpath = solutions[:T].reshape(T)
            wpath = solutions[T:].reshape(T)


            # run one iteration of TP with fsolve solution to get other output
            tp_params = (S, T, alpha, beta, sigma, r_ss, w_ss, tp_tol)
            (r_path, w_path, pc_path, p_path, b_path, c_path, ci_path,
                eul_path, Ci_path, Ym_path, Km_path, Lm_path,
                MCKerr_path, MCLerr_path, tpi_time) = \
                tpf.TP(tp_params, rpath, wpath, Km_ss, Ym_ss,
                Gamma1, ci_tilde, A, gamma, epsilon, delta, xi, pi, I, 
                M, S, n, tp_graphs)
                


            # Print diagnostics
            print 'The max. absolute difference in the resource constraints are:'
            delmat = np.tile(delta.reshape((I, 1)), T-2)
            ResmDiff = (Ym_path[:, :T-2] - Cm_path[:, :T-2] - Km_path[:, 1:T-1] +
                      (1 - delmat) * Km_path[:, :T-2])
            print np.absolute(ResmDiff).max(axis=1)
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


