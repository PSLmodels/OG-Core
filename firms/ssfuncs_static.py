'''
------------------------------------------------------------------------
This file contains the functions specific to solving for the steady state
of the OG model with S-period lived agents, exogenous labor, and M industries 
and I goods.

These functions include:
    feasible
    get_p
    get_p
    get_cbess
    EulerSys_b
    get_c_tilde
    get_c
    get_b_errors
    get C
    get_XK
    get_Lvec
    MCerrs
    SS

This Python script calls the following other file(s) with the associated
functions:
    firm_funcs.py
        get_p
        get_p_tilde
        get_c_tilde
        get_c
        get_C
        get_K
        get_L

------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import firm_funcs_static as firm
reload(firm)

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def feasible(params, rw_init, b_guess, c_bar, A, gamma, epsilon,
  delta, pi, I, S, n):
    '''
    Determines whether a particular guess for the steady-state values
    or r and w are feasible. Feasibility means that
    r + delta > 0, w > 0, implied c_s>0, c_{i,s}>0 for all i and
    all s and implied sum of K_{m} > 0

    Inputs:
        params   = length 5 tuple, (S, alpha, beta, sigma, ss_tol)
        S        = integer in [3,80], number of periods in a life
        alpha    = scalar in (0,1), expenditure share on good 1
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve
        rw_init  = [2,] vector, initial guesses for steady-state r and w
        b_guess  = [S-1,] vector, initial guess for savings vector
        c_bar = [I,] vector, minimum consumption values for all goods
        A        = [M,] vector, total factor productivity values for all
                   industries
        gamma   = [M,] vector, capital shares of income for all
                   industries
        epsilon   = [M,] vector, elasticities of substitution between
                   capital and labor for all industries
        delta   = [M,] vector, model period depreciation rates for all
                   industries
        xi      = [M,M] matrix, element i,j gives the fraction of capital used by 
               industry j that comes from the output of industry i
        pi      = [I,M] matrix, element i,j gives the fraction of consumption
               good i that comes from the output of industry j
        n     = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        firm.get_p    = generates vector of industry prices
        firm.get_p_tilde     = generates composite goods price
        get_cbess = generates savings, consumption, Euler, and
                    constraint vectors

    Objects in function:
        r          = scalar > 0, initial guess for interest rate
        w          = scalar > 0, initial guess for real wage
        GoodGuess  = boolean, =True if initial steady-state guess is
                     feasible
        r_cstr     = boolean, =True if r + delta <= 0
        w_cstr     = boolean, =True if w <= 0
        c_cstr     = [S,] boolean vector, =True if c_s<=0 for some s
        K_cstr     = boolean, =True if sum of K_{m} <= 0
        p_params  = length 4 tuple, vectors to be passed in to get_p
                     (A, gamma, epsilon, delta)
        p      = [M,] vector, output prices from each industry
        p_c      = [I,] vector, consumption goods prices from each industry
        p          = scalar > 0, composite good price
        cbe_params = length 7 tuple, parameters for get_cbess function
                     (alpha, beta, sigma, r, w, p, ss_tol)
        b       = [S-1,] vector, optimal savings given prices
        c       = [S,] vector, optimal composite consumption given
                     prices
        c      = [I,S] matrix, optimal consumption of each good
                     given prices
        c_cstr    = [2,S] boolean matrix, =True if c_{i,s}<=0 for
                     given c_s
        euler_errors     = [S-1,] vector, Euler equations from optimal savings
        K_s        = scalar, sum of all savings (capital supply)

    Returns: GoodGuess, r_cstr, w_cstr, c_cstr, c_cstr, K_cstr
    '''
    S, alpha, beta, sigma, ss_tol = params
    r, w = rw_init
    GoodGuess = True
    r_cstr = False
    w_cstr = False
    c_cstr = np.zeros(S, dtype=bool)
    K_cstr = False
    if (r + delta).min() <= 0 and w <= 0:
        r_cstr = True
        w_cstr = True
        GoodGuess = False
    elif (r + delta).min() <= 0 and w > 0:
        r_cstr = True
        GoodGuess = False
    elif (r + delta).min() > 0 and w <= 0:
        w_cstr = True
        GoodGuess = False
    elif (r + delta).min() > 0 and w > 0:
        p_params = (A, gamma, epsilon, delta)
        p = firm.get_p(p_params, r, w)
        p_c = np.dot(pi,p)
        p_tilde = firm.get_p_tilde(alpha, p_c)
        cbe_params = (alpha, beta, sigma, r, w, p_tilde, ss_tol)
        b, c, c_cstr, c, c_cstr, euler_errors = \
            get_cbess(cbe_params, b_guess, p_c, c_bar, I, S, n)
        # Check K1 + K2
        K_s = b.sum()
        K_cstr = K_s <= 0
        if K_cstr == True or c_cstr.max() == 1 or c_cstr.max() == 1:
            GoodGuess = False

    return GoodGuess, r_cstr, w_cstr, c_cstr, c_cstr, K_cstr


def get_cbess(params, b_guess, p_c, c_bar, I, S, n):
    '''
    Generates vectors for individual savings, composite consumption,
    industry-specific consumption, constraint vectors, and Euler errors
    for a given set of prices (r, w, p, p).

    Inputs:
        params   = length 7 tuple, (alpha, beta, sigma, r, w, p ss_tol)
        alpha    = [I,] vector, expenditure shares on each good
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p_tilde        = scalar > 0, composite good price
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve
        b_guess  = [S-1,] vector, initial guess for savings vector
        p_c    = [I,] vector, prices in each industry
        c_bar = [I,] vector, minimum consumption values for all goods
        n     = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        EulerSys_b
        firm.get_c_tilde
        firm.get_c
        firm.get_b_errors

    Objects in function:
        eulb_objs  = length 9 tuple, objects to be passed in to
                     EulerSys_b: (alpha, beta, sigma, r, w, p, p,
                     c_bar, n)
        b       = [S-1,] vector, optimal lifetime savings decisions
        c_tidle_params   = length 3 tuple, parameters for get_c_tilde(r, w, p_tilde)
        c_tilde       = [S,] vector, optimal lifetime consumption
        c_tilde_cstr     = [S,] boolean vector, =True if c_s<=0
        c_params  = length 2 tuple, parameters for get_c (alpha, p)
        c        = [I,S] matrix,  optimal consumption of each good
                     given prices
        c_cstr    = [I,S] boolean matrix, =True if c_{i,s}<=0 for given
                     c_s
        eul_params = length 2 tuple, parameters for get_b_errors
                     (beta, sigma)
        euler_errors     = [S-1,] vector, Euler errors from savings decisions

    Returns: b, c, c_cstr, c, c_cstr, euler_errors
    '''
    alpha, beta, sigma, r, w, p_tilde, ss_tol = params
    eulb_objs = (alpha, beta, sigma, r, w, p_tilde, p_c, c_bar, I, S, n)
    b = opt.fsolve(EulerSys_b, b_guess, args=(eulb_objs), xtol=ss_tol)
    c_tilde, c_tilde_cstr = firm.get_c_tilde(c_bar, r, w, p_c, p_tilde, n, np.append([0], b))
    c_params = (alpha, p_tilde)
    c, c_cstr = firm.get_c(np.tile(np.reshape(alpha,(I,1)),(1,S)), np.tile(np.reshape(c_bar,(I,1)),(1,S)), 
                           np.tile(c_tilde,(I,1)), np.tile(np.reshape(p_c,(I,1)),(1,S)) , p_tilde)
    eul_params = (beta, sigma)
    euler_errors = firm.get_b_errors(eul_params, r, c_tilde, c_tilde_cstr, diff=True)

    return b, c, c_cstr, c, c_cstr, euler_errors


def EulerSys_b(b, *objs):
    '''
    Generates vector of all Euler errors for a given b, which errors
    characterize all optimal lifetime savings decisions

    Inputs:
        b     = [S-1,] vector, lifetime savings decisions
        objs     = length 9 tuple,
                   (alpha, beta, sigma, r, w, p, p, c_bar, n)
        alpha    = [I,] vector, expenditure shares on each good
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p_tilde        = scalar > 0, composite good price
        p_c    = [I,] vector, price for each consumption good
        c_bar = [I,] vector, minimum consumption values for all goods
        n     = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        firm.get_c_tilde
        firm.get_c
        firm.get_b_errors

    Objects in function:
        c_params     = length 3 tuple, parameters for get_c (r, w, p)
        c_tilde         = [S,] vector, remaining lifetime consumption
                       levels implied by b
        c_tilde_cstr       = [S, ] boolean vector, =True if c_{s,t}<=0
        c_params    = length 2 tuple, parameters for get_c
                       (alpha, p)
        c        = [I,S] matrix, consumption values for each good
                       and age c_{i,s}
        c_cstr      = [I,S] boolean matrix, =True if c_{i,s}<=0
        b_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_err_vec    = [S-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_err_vec
    '''
    alpha, beta, sigma, r, w, p_tilde, p_c, c_bar, I, S, n = objs

    c_tilde, c_tilde_cstr = firm.get_c_tilde(c_bar, r, w, p_c, p_tilde, n, np.append([0], b))
    c, c_cstr = firm.get_c(np.tile(np.reshape(alpha,(I,1)),(1,S)), np.tile(np.reshape(c_bar,(I,1)),(1,S)), 
                           np.tile(c_tilde,(I,1)), np.tile(np.reshape(p_c,(I,1)),(1,S)) , p_tilde)
    b_err_params = (beta, sigma)
    b_err_vec = firm.get_b_errors(b_err_params, r, c_tilde, c_tilde_cstr, diff=True)
    return b_err_vec



def solve_X(X_init, params, C, A, gamma, epsilon, delta, xi, pi, I, M):
    '''
    Generates vector of aggregate output X_m of good m given r and w
    and consumption demands

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        C    = [I,] vector, aggregate consumption of all goods
        A   = [M,] vector, total factor productivity values for all
                 industries
        gamma = [M,] vector, capital shares of income for all
                 industries
        epsilon = [M,] vector, elasticities of substitution between
                 capital and labor for all industries
        delta = [M,] vector, model period depreciation rates for all
                 industries
        xi      = [M,M] matrix, element i,j gives the fraction of capital used by 
               industry j that comes from the output of industry i
        pi      = [I,M] matrix, element i,j gives the fraction of consumption
               good i that comes from the output of industry j

    Functions called: 
        firm.get_K

    Objects in function:
        X    = [M,] vector, aggregate output of all industries
        X_c   = [M,] vector, demand for output from industry M from consumption demand
        Inv   = [M,] vector, investment demand from each industry M
        rc_ errors = [M,] vector, differences in resource constriant for each industry

    Returns: rc_errors
    '''
    r, w = params
    X = X_init
    
    X_c = np.dot(np.reshape(C,(1,I)),pi)

    Inv = np.reshape(delta*firm.get_K(r, w, X, A, gamma, epsilon, delta),(1,M))
    
    rc_errors = np.reshape(X_c  + np.dot(Inv,xi) - X,(M))

    return rc_errors


def MCerrs(rwvec, *objs):
    '''
    Returns capital and labor market clearing condition errors given
    particular values of r and w

    Inputs:
        rwvec    = [2,] vector, given values of r and w
        objs     = length 12 tuple, (S, alpha, beta, sigma, b_guess,
                   c_bar, A, gamma, epsilon, delta, n, ss_tol)
        S        = integer in [3,80], number of periods an individual
                   lives
        alpha    = [I,] vector, expenditure shares on all consumption goods
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        b_guess  = [S-1,] vector, initial guess for savings to use in
                   fsolve in get_cbess
        c_bar = [I,] vector, minimum consumption values for all goods
        A     = [I,] vector, total factor productivity values for all
                   industries
        gamma   = [I,] vector, capital shares of income for all
                   industries
        epsilon   = [I,] vector, elasticities of substitution between
                   capital and labor for all industries
        delta   = [I,] vector, model period depreciation rates for all
                   industries
        n     = [S,] vector, exogenous labor supply n_{s}
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve

    Functions called:
        firm.get_p
        firm.get_p_tilde
        get_cbess
        firm.get_C
        firm.get_K
        firm.get_L

    Objects in function:
        r          = scalar > 0, interest rate
        w          = scalar > 0, real wage
        MCKerr     = scalar, error in capital market clearing condition
                     given r and w
        MCLerr     = scalar, error in labor market clearing condition
                     given r and w
        p_c_params  = length 4 tuple, vectors to be passed in to get_p_c
                     (A, gamma, epsilon, delta)
        p      = [2,] vector, prices in each industry
        p_tilde          = scalar > 0, composite good price
        cbe_params = length 7 tuple, parameters for get_cbess function
                     (alpha, beta, sigma, r, w, p, ss_tol)
        b       = [S-1,] vector, optimal savings given prices
        c       = [S,] vector, optimal composite consumption given
                     prices
        c_cstr     = [S,] boolean vector, =True if c_s<=0 for some s
        c      = [2,S] matrix, optimal consumption of each good
                     given prices
        c_cstr    = [2,S] boolean matrix, =True if c_{m,s}<=0 for
                     given c_s
        euler_errors     = [S-1,] vector, Euler equations from optimal savings
        C      = [2,] vector, total consumption demand for each
                     industry
        X_params  = length 2 tuple, parameters for get_XK function:
                     (r, w)
        X      = [2,] vector, total output for each industry
        K      = [2,] vector, capital demand for each industry
        L_params  = length 2 tuple, parameters for get_Lvec function:
                     (r, w)
        Lvec      = [2,] vector, labor demand for each industry
        MC_errs    = [2,] vector, capital and labor market clearing
                     errors given r ans w

    Returns: MC_errs
    '''
    (S, alpha, beta, sigma, b_guess, c_bar, A, gamma, epsilon,
        delta, xi, pi, I, M, S, n, ss_tol) = objs
    r, w = rwvec
    if (r + delta).min() <= 0 or w <=0:
        MCKerr = 9999.
        MCLerr = 9999.
        MC_errs = np.array((MCKerr, MCLerr))
    elif (r + delta).min() > 0 and w > 0:
        p_params = (A, gamma, epsilon, delta)
        p = firm.get_p(p_params, r, w)   
        p_c = np.dot(pi,p)
        p_tilde = firm.get_p_tilde(alpha, p_c)
        cbe_params = (alpha, beta, sigma, r, w, p_tilde, ss_tol)
        b, c_tilde, c_tilde_cstr, c, c_cstr, euler_errors = \
            get_cbess(cbe_params, b_guess, p_c, c_bar, I, S, n)
        C = firm.get_C(c.transpose())
        X_params = (r, w)
        X_init = (np.dot(np.reshape(C,(1,I)),pi))/I
        X = opt.fsolve(solve_X, X_init, args=(X_params, C, A, gamma, epsilon, delta, xi, pi, I, M), xtol=ss_tol, col_deriv=1)
        K = firm.get_K(r, w, X, A, gamma, epsilon, delta)
        L = firm.get_L(r, w, K, gamma, epsilon, delta)
        MCKerr = K.sum() - b.sum()
        MCLerr = L.sum() - n.sum()
        MC_errs = np.array((MCKerr, MCLerr))

    return MC_errs


def SS(params, rw_init, b_guess, c_bar, A, gamma, epsilon, delta, xi, pi, I,
    M, S, n, graphs):
    '''
    Generates all endogenous steady-state objects

    Inputs:
        params   = length 5 tuple, (S, alpha, beta, sigma, ss_tol)
        S        = integer in [3,80], number of periods an individual
                   lives
        alpha    = [I,] vectors, expenditure share on all consumption goods
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve
        rw_init  = [2,] vector, initial guesses for steady-state r and w
        b_guess  = [S-1,] vector, initial guess for savings to use in
                   fsolve in get_cbess
        c_bar = [I,] vector, minimum consumption values for all goods
        A     = [M,] vector, total factor productivity values for all
                   industries
        gamma   = [M,] vector, capital shares of income for all
                   industries
        epsilon   = [M,] vector, elasticities of substitution between
                   capital and labor for all industries
        delta   = [M,] vector, model period depreciation rates for all
                   industries
        xi      = [M,M] matrix, element i,j gives the fraction of capital used by 
               industry j that comes from the output of industry i
        pi      = [I,M] matrix, element i,j gives the fraction of consumption
        I       = number of consumption goods
        n     = [S,] vector, exogenous labor supply n_{s}
        graphs   = boolean, =True if want graphs of steady-state objects

    Functions called:
        MCerrs
        firm.get_p
        firm.get_p_tilde
        get_cbess
        firm.get_C
        firm.get_K
        firm.get_L

    Objects in function:
        start_time  = scalar, current processor time in seconds (float)
        MCerrs_objs = length 12 tuple, objects to be passed in to
                      MCerrs function: (S, alpha, beta, sigma, b_guess,
                      c_bar, A, gamma, epsilon, delta, n,
                      ss_tol)
        rw_ss       = [2,] vector, steady-state r and w
        r_ss        = scalar, steady-state interest rate
        w_ss        = scalar > 0, steady-state wage
        p_c_params   = length 4 tuple, vectors to be passed in to get_p_c
                      (A, gamma, epsilon, delta)
        p_ss       = [M,] vector, steady-state output prices for each industry
        p_c_ss       = [I,] vector, steady-state consumption good prices
        p_tilde_ss        = scalar > 0, steady-state composite good price
        cbe_params  = length 7 tuple, parameters for get_cbess function
                      (alpha, beta, sigma, r_ss, w_ss, p_ss, ss_tol)
        b_ss        = [S-1,] vector, steady-state savings
        c_ss        = [S,] vector, steady-state composite consumption
        c_cstr      = [S,] boolean vector, =True if cbar_s<=0 for some s
        c_ss       = [I,S] matrix, steady-state consumption of each
                      good
        c_cstr     = [I,S] boolean matrix, =True if c_ss{i,s}<=0 for
                      given c_ss
        EulErr_ss   = [S-1,] vector, steady-state Euler errors
        C_ss       = [I,] vector, total demand for goods from each
                      industry
        X_params   = length 2 tuple, parameters for get_XK
                      function: (r_ss, w_ss)
        X_ss       = [M,] vector, steady-state total output for each
                      industry
        K_ss       = [M,] vector, steady-state capital demand for each
                      industry
        L_params   = length 2 tuple, parameters for get_Lvec function:
                      (r_ss, w_ss)
        L_ss       = [M,] vector, steady-state labor demand for each
                      industry
        MCK_err_ss  = scalar, steady-state capital market clearing error
        MCL_err_ss  = scalar, steady-state labor market clearing error
        MCerr_ss    = [2,] vector, steady-state capital and labor market
                      clearing errors
        ss_time     = scalar, time to compute SS solution (seconds)
        svec         = [S,] vector, age-s indices from 1 to S
        b_ss0        = [S,] vector, age-s wealth levels including b_1=0

    Returns: r_ss, w_ss, p_ss, p_ss, b_ss, c_ss, c_ss, EulErr_ss,
             Cm_ss, X_ss, K_ss, L_ss, MCK_err_ss, MCL_err_ss, ss_time
    '''
    start_time = time.clock()
    S, alpha, beta, sigma, ss_tol = params
    MCerrs_objs = (S, alpha, beta, sigma, b_guess, c_bar, A,
                  gamma, epsilon, delta, xi, pi, I, M, S, n, ss_tol)
    rw_ss = opt.fsolve(MCerrs, rw_init, args=(MCerrs_objs),
                xtol=ss_tol)
    r_ss, w_ss = rw_ss
    p_params = (A, gamma, epsilon, delta)
    p_ss = firm.get_p(p_params, r_ss, w_ss)

    p_c_ss = np.dot(pi,p_ss)
    p_tilde_ss = firm.get_p_tilde(alpha, p_c_ss)
    cbe_params = (alpha, beta, sigma, r_ss, w_ss, p_tilde_ss, ss_tol)
    b_ss, c_tilde_ss, c_tilde_cstr, c_ss, c_cstr, EulErr_ss = \
        get_cbess(cbe_params, b_guess, p_c_ss, c_bar, I, S, n)
    C_ss = firm.get_C(c_ss.transpose())
    X_params = (r_ss, w_ss)
    X_init = (np.dot(np.reshape(C_ss,(1,I)),pi))/I
    X_ss = opt.fsolve(solve_X, X_init, args=(X_params, C_ss, A, gamma, epsilon, delta, xi, pi, I, M), xtol=ss_tol, col_deriv=1)
    K_ss = firm.get_K(r_ss, w_ss, X_ss, A, gamma, epsilon, delta)
    L_ss = firm.get_L(r_ss, w_ss, K_ss, gamma, epsilon, delta)

    MCK_err_ss = K_ss.sum() - b_ss.sum()
    MCL_err_ss = L_ss.sum() - n.sum()
    MCerr_ss = np.array([MCK_err_ss, MCL_err_ss])
    ss_time = time.clock() - start_time

    if graphs == True:
        # Plot steady-state distribution of savings
        svec = np.linspace(1, S, S)
        b_ss0 = np.append([0], b_ss)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(svec, b_ss0)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Steady-state distribution of savings')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual savings $\bar{b}_{s}$')
        # plt.savefig('b_ss_Chap11')
        plt.show()

        # Plot steady-state distribution of composite consumption
        fig, ax = plt.subplots()
        plt.plot(svec, c_tilde_ss)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Steady-state distribution of composite consumption')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual consumption $\tilde{c}_{s}$')
        # plt.savefig('c_ss_Chap11')
        plt.show()

        # Plot steady-state distribution of individual good consumption
        fig, ax = plt.subplots()
        plt.plot(svec, c_ss[0,:], 'r--', label='Good 1')
        plt.plot(svec, c_ss[1,:], 'b', label='Good 2')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.legend(loc='center right')
        plt.title('Steady-state distribution of goods consumption')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual consumption $\bar{c}_{m,s}$')
        # plt.savefig('c_ss_Chap11')
        plt.show()

    return (r_ss, w_ss, p_c_ss, p_tilde_ss, b_ss, c_tilde_ss, c_ss, EulErr_ss,
           C_ss, X_ss, K_ss, L_ss, MCK_err_ss, MCL_err_ss, ss_time)
