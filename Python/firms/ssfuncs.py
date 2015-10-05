'''
------------------------------------------------------------------------
All the functions for the SS computation from Chapter 11 of the OG
textbook
    feasible
    get_pm
    get_p
    get_cbess
    EulerSys_b
    get_c
    get_c_i
    get_b_errors
    get C_i
    get_YKm
    get_Lmvec
    MCerrs
    SS
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

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def feasible(params, rw_init, b_guess, ci_tilde, A, gamma, epsilon,
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
        ci_tilde = [I,] vector, minimum consumption values for all goods
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
        get_pm    = generates vector of industry prices
        get_p     = generates composite goods price
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
        pm_params  = length 4 tuple, vectors to be passed in to get_pm
                     (A, gamma, epsilon, delta)
        p_m      = [M,] vector, output prices from each industry
        p_c      = [I,] vector, consumption goods prices from each industry
        p          = scalar > 0, composite good price
        cbe_params = length 7 tuple, parameters for get_cbess function
                     (alpha, beta, sigma, r, w, p, ss_tol)
        b       = [S-1,] vector, optimal savings given prices
        c       = [S,] vector, optimal composite consumption given
                     prices
        c_i      = [I,S] matrix, optimal consumption of each good
                     given prices
        ci_cstr    = [2,S] boolean matrix, =True if c_{i,s}<=0 for
                     given c_s
        euler_errors     = [S-1,] vector, Euler equations from optimal savings
        K_s        = scalar, sum of all savings (capital supply)

    Returns: GoodGuess, r_cstr, w_cstr, c_cstr, ci_cstr, K_cstr
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
        pm_params = (A, gamma, epsilon, delta)
        p_m = get_pm(pm_params, r, w)
        p_c = np.dot(pi,p_m)
        p = get_p(alpha, p_c)
        cbe_params = (alpha, beta, sigma, r, w, p, ss_tol)
        b, c, c_cstr, c_i, ci_cstr, euler_errors = \
            get_cbess(cbe_params, b_guess, p_c, ci_tilde, I, S, n)
        # Check K1 + K2
        K_s = b.sum()
        K_cstr = K_s <= 0
        if K_cstr == True or c_cstr.max() == 1 or ci_cstr.max() == 1:
            GoodGuess = False

    return GoodGuess, r_cstr, w_cstr, c_cstr, ci_cstr, K_cstr


def get_pm(params, r, w):
    '''
    Generates vector of industry prices p_m from r and w

    Inputs:
        params = length 4 tuple, (A, gamma, epsilon, delta)
        A   = [M,] vector, total factor productivity for each
                 industry
        gamma = [M,] vector, capital share of income for each industry
        epsilon = [M,] vector, elasticity of substitution between capital
                 and labor for each industry
        delta = [M,] vector, capital depreciation rate for each
                 industry
        r      = scalar > 0, interest rate
        w      = scalar > 0, wage

    Functions called: None

    Objects in function:
        p_m = [M,] vector, industry prices

    Returns: p_m
    '''
    A, gamma, epsilon, delta = params
    p_m = (1 / A) * ((gamma * ((r + delta) ** (1 - epsilon)) +
            (1 - gamma) * (w ** (1 - epsilon))) ** (1 / (1 - epsilon)))

    return p_m


def get_p(alpha, p_c):
    '''
    Generates vector of industry prices p_m from r and w

    Inputs:
        alpha = [I,] vector, expenditure shares on each good
        p_c = [I,] vector, consumption good prices

    Functions called: None

    Objects in function:
        p = scalar > 0, price of composite consumption good

    Returns: p
    '''
    p = ((p_c/alpha)**alpha).prod()

    return p


def get_cbess(params, b_guess, p_c, ci_tilde, I, S, n):
    '''
    Generates vectors for individual savings, composite consumption,
    industry-specific consumption, constraint vectors, and Euler errors
    for a given set of prices (r, w, p, p_m).

    Inputs:
        params   = length 7 tuple, (alpha, beta, sigma, r, w, p ss_tol)
        alpha    = [I,] vector, expenditure shares on each good
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p        = scalar > 0, composite good price
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve
        b_guess  = [S-1,] vector, initial guess for savings vector
        p_c    = [I,] vector, prices in each industry
        ci_tilde = [I,] vector, minimum consumption values for all goods
        n     = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        EulerSys_b
        get_c
        get_c_i
        get_b_errors

    Objects in function:
        eulb_objs  = length 9 tuple, objects to be passed in to
                     EulerSys_b: (alpha, beta, sigma, r, w, p, p_m,
                     ci_tilde, n)
        b       = [S-1,] vector, optimal lifetime savings decisions
        c_params   = length 3 tuple, parameters for get_c (r, w, p)
        c       = [S,] vector, optimal lifetime consumption
        c_cstr     = [S,] boolean vector, =True if c_s<=0
        cm_params  = length 2 tuple, parameters for get_c_i (alpha, p)
        c_i        = [I,S] matrix,  optimal consumption of each good
                     given prices
        ci_cstr    = [I,S] boolean matrix, =True if c_{i,s}<=0 for given
                     c_s
        eul_params = length 2 tuple, parameters for get_b_errors
                     (beta, sigma)
        euler_errors     = [S-1,] vector, Euler errors from savings decisions

    Returns: b, c, c_cstr, c_i, ci_cstr, euler_errors
    '''
    alpha, beta, sigma, r, w, p, ss_tol = params
    eulb_objs = (alpha, beta, sigma, r, w, p, p_c, ci_tilde, I, S, n)
    b = opt.fsolve(EulerSys_b, b_guess, args=(eulb_objs), xtol=ss_tol)
    c_params = (r, w, p)
    c, c_cstr = get_c(c_params, n, b, p_c, ci_tilde)
    ci_params = (alpha, p)
    c_i, ci_cstr = get_c_i(ci_params, c, p_c, ci_tilde, I, S)
    eul_params = (beta, sigma)
    euler_errors = get_b_errors(eul_params, r, c, c_cstr, diff=True)

    return b, c, c_cstr, c_i, ci_cstr, euler_errors


def EulerSys_b(b, *objs):
    '''
    Generates vector of all Euler errors for a given b, which errors
    characterize all optimal lifetime savings decisions

    Inputs:
        b     = [S-1,] vector, lifetime savings decisions
        objs     = length 9 tuple,
                   (alpha, beta, sigma, r, w, p, p_m, ci_tilde, n)
        alpha    = [I,] vector, expenditure shares on each good
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p        = scalar > 0, composite good price
        p_c    = [I,] vector, price for each consumption good
        ci_tilde = [I,] vector, minimum consumption values for all goods
        n     = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        get_c
        get_c_i
        get_b_errors

    Objects in function:
        c_params     = length 3 tuple, parameters for get_c (r, w, p)
        c         = [S,] vector, remaining lifetime consumption
                       levels implied by b
        c_cstr       = [S, ] boolean vector, =True if c_{s,t}<=0
        ci_params    = length 2 tuple, parameters for get_c_i
                       (alpha, p)
        c_i        = [I,S] matrix, consumption values for each good
                       and age c_{i,s}
        ci_cstr      = [I,S] boolean matrix, =True if c_{i,s}<=0
        b_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_err_vec    = [S-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_err_vec
    '''
    alpha, beta, sigma, r, w, p, p_c, ci_tilde, I, S, n = objs
    c_params = (r, w, p)
    c, c_cstr = get_c(c_params, n, b, p_c, ci_tilde)
    cm_params = (alpha, p)
    c_i, ci_cstr = get_c_i(cm_params, c, p_c, ci_tilde, I, S)
    b_err_params = (beta, sigma)
    b_err_vec = get_b_errors(b_err_params, r, c, c_cstr, diff=True)
    return b_err_vec


def get_c(params, n, b, p_c, ci_tilde):
    '''
    Generate lifetime consumption given prices and savings decisions

    Inputs:
        params   = length 3 tuple, (r, w, p)
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p        = scalar > 0, composite good price
        n     = [S,] vector, exogenous labor supply n_{s}
        b     = [S-1,] vector, distribution of savings b_{s+1}
        p_c      = [I,] vector, prices for each consumption good
        ci_tilde = [I,] vector, minimum consumption values for all goods

    Functions called: None

    Objects in function:
        b_s    = [S,] vector, 0 in first element and b in last S-1
                 elements
        b_sp1  = [S,] vector, b in first S-1 elements and 0 in last
                 element
        c   = [S,] vector, composite consumption by age c_s
        c_cstr = [S,] boolean vector, =True if element c_s <= 0

    Returns: c, c_constr
    '''
    r, w, p = params
    b_s = np.append([0], b)
    b_sp1 = np.append(b, [0])
    c = (1 / p) * ((1 + r) * b_s + w * n -
           (p_c * ci_tilde).sum() - b_sp1)
    c_cstr = c <= 0
    return c, c_cstr


def get_c_i(params, c, p_c, ci_tilde, I, S):
    '''
    Generates matrix of consumptions of each type of good given prices
    and composite consumption

    Inputs:
        params   = length 2 tuple, (alpha, p)
        alpha    = [I,] vector, expenditure share on each consumption good
        p        = scalar > 0, composite good price
        c     = [S,] vector, composite consumption by age c_s
        p_c    = [I,] vector, prices for each consumption good
        ci_tilde = [I,] vector, minimum consumption values for all goods

    Functions called: None

    Objects in function:
        c_i   = [I,S] matrix, consumption of each good by age c_{I,s}
        ci_cstr = [S,] boolean vector, =True if element c_s <= 0

    Returns: c_i, ci_cstr
    '''
    alpha, p = params
    c_i = ((p*np.tile(c,(I,1))*np.tile(np.reshape(alpha,(I,1)),(1,S)))/np.tile(np.reshape(p_c,(I,1)),(1,S)) 
                + np.tile(np.reshape(ci_tilde,(I,1)),(1,S)))
    ci_cstr = c_i <= 0
    return c_i, ci_cstr


def get_b_errors(params, r, c, c_cstr, diff):
    '''
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings

    Inputs:
        params = length 2 tuple, (beta, sigma)
        beta   = scalar in [0,1), discount factor
        sigma  = scalar > 0, coefficient of relative risk aversion
        r      = scalar > 0, interest rate
        c   = [S,] vector, distribution of consumption by age c_s
        c_cstr = [S,] boolean vector, =True if c_s<=0 for given b
        diff   = boolean, =True if use simple difference Euler errors.
                 Use percent difference errors otherwise.

    Functions called: None

    Objects in function:
        mu_c     = [S-1,] vector, marginal utility of current
                   consumption
        mu_cp1   = [S-1,] vector, marginal utility of next period
                   consumption
        b_errors = [S-1,] vector, Euler errors with errors = 0
                   characterizing optimal savings b

    Returns: b_errors
    '''
    beta, sigma = params
    c[c_cstr] = 9999. # Each consumption must be positive to
                         # generate marginal utilities
    mu_c = c[:-1] ** (-sigma)
    mu_cp1 = c[1:] ** (-sigma)
    if diff == True:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
        b_errors[c_cstr[:-1]] = 9999.
        b_errors[c_cstr[1:]] = 9999.
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        b_errors[c_cstr[:-1]] = 9999. / 100
        b_errors[c_cstr[1:]] = 9999. / 100
    return b_errors


def get_C_i(c_i):
    '''
    Generates vector of aggregate consumption C_i of good i

    Inputs:
        c_i    = [I,S] matrix, distribution of individual consumption
                   c_{i,s}

    Functions called: None

    Objects in function:
        C_i = [I,] vector, aggregate consumption of all goods

    Returns: C_i
    '''
    C_i = c_i.sum(axis=1)
    return C_i


def solve_Ym(Ym_init, params, C_i, A, gamma, epsilon, delta, xi, pi, I, M):
    '''
    Generates vector of aggregate output Y_m of good m given r and w
    and consumption demands

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        C_i    = [I,] vector, aggregate consumption of all goods
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
        get_Km

    Objects in function:
        Ym    = [M,] vector, aggregate output of all industries
        Y_c   = [M,] vector, demand for output from industry M from consumption demand
        Inv   = [M,] vector, investment demand from each industry M
        rc_ errors = [M,] vector, differences in resource constriant for each industry

    Returns: rc_errors
    '''
    r, w = params
    Ym = Ym_init
    
    Y_c = np.dot(np.reshape(C_i,(1,I)),pi)

    K_params = (r, w)
    Inv = np.reshape(delta*get_Km(K_params, Ym, A, gamma, epsilon, delta),(1,M))
    
    rc_errors = np.reshape(Y_c  + np.dot(Inv,xi) - Ym,(M))

    return rc_errors

def get_Km(params, Ym, A, gamma, epsilon, delta):
    '''
    Generates vector of capital demand, Km, from industry m 
    for a given Ym, r, and w

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        Ym  = [M,] vector, output from each industry
        A   = [M,] vector, total factor productivity values for all
                 industries
        gamma = [M,] vector, capital shares of income for all
                 industries
        epsilon = [M,] vector, elasticities of substitution between
                 capital and labor for all industries
        delta = [M,] vector, model period depreciation rates for all
                 industries

    Functions called: None

    Objects in function:
        aa    = [M,] vector, gamma
        bb    = [M,] vector, 1 - gamma
        cc    = [M,] vector, (1 - gamma) / gamma
        dd    = [M,] vector, (r + delta) / w
        ee    = [M,] vector, 1 / epsilon
        ff    = [M,] vector, (epsilon - 1) / epsilon
        gg    = [M,] vector, epsilon - 1
        hh    = [M,] vector, epsilon / (1 - epsilon)
        ii    = [M,] vector, ((1 / A) * (((aa ** ee) + (bb ** ee) *
                (cc ** ff) * (dd ** gg)) ** hh))
        Km = [M,] vector, capital demand of all industries

    Returns: Km
    '''
    r, w = params
    aa = gamma
    bb = 1 - gamma
    cc = (1 - gamma) / gamma
    dd = (r + delta) / w
    ee = 1 / epsilon
    ff = (epsilon - 1) / epsilon
    gg = epsilon - 1
    hh = epsilon / (1 - epsilon)
    Km = ((Ym / A) *
         (((aa ** ee) + (bb ** ee) * (cc ** ff) * (dd ** gg)) ** hh))
    return Km


def get_Lm(params, Km, gamma, epsilon, delta):
    '''
    Generates vector of labor demand L_m for each industry m

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        Km  = [M,] vector, capital demand in all industries
        gamma = [M,] vector, capital shares of income for all
                 industries
        epsilon = [M,] vector, elasticities of substitution between
                 capital and labor for all industries
        delta = [M,] vector, model period depreciation rates for all
                 industries

    Functions called: None

    Objects in function:
        Lmvec = [2,] vector, aggregate output of all goods

    Returns: Lmvec
    '''
    r, w = params
    Lm = Km*((1-gamma)/gamma)*(((r+delta)/w)**epsilon)

    return Lm


def MCerrs(rwvec, *objs):
    '''
    Returns capital and labor market clearing condition errors given
    particular values of r and w

    Inputs:
        rwvec    = [2,] vector, given values of r and w
        objs     = length 12 tuple, (S, alpha, beta, sigma, b_guess,
                   ci_tilde, A, gamma, epsilon, delta, n, ss_tol)
        S        = integer in [3,80], number of periods an individual
                   lives
        alpha    = [I,] vector, expenditure shares on all consumption goods
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        b_guess  = [S-1,] vector, initial guess for savings to use in
                   fsolve in get_cbess
        ci_tilde = [I,] vector, minimum consumption values for all goods
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
        get_pm
        get_p
        get_cbess
        get_C_i
        get_YKm
        get_Lmvec

    Objects in function:
        r          = scalar > 0, interest rate
        w          = scalar > 0, real wage
        MCKerr     = scalar, error in capital market clearing condition
                     given r and w
        MCLerr     = scalar, error in labor market clearing condition
                     given r and w
        pm_params  = length 4 tuple, vectors to be passed in to get_pm
                     (A, gamma, epsilon, delta)
        p_m      = [2,] vector, prices in each industry
        p          = scalar > 0, composite good price
        cbe_params = length 7 tuple, parameters for get_cbess function
                     (alpha, beta, sigma, r, w, p, ss_tol)
        b       = [S-1,] vector, optimal savings given prices
        c       = [S,] vector, optimal composite consumption given
                     prices
        c_cstr     = [S,] boolean vector, =True if c_s<=0 for some s
        c_i      = [2,S] matrix, optimal consumption of each good
                     given prices
        ci_cstr    = [2,S] boolean matrix, =True if c_{m,s}<=0 for
                     given c_s
        euler_errors     = [S-1,] vector, Euler equations from optimal savings
        C_i      = [2,] vector, total consumption demand for each
                     industry
        Ym_params  = length 2 tuple, parameters for get_YKm function:
                     (r, w)
        Ym      = [2,] vector, total output for each industry
        Km      = [2,] vector, capital demand for each industry
        Lm_params  = length 2 tuple, parameters for get_Lmvec function:
                     (r, w)
        Lmvec      = [2,] vector, labor demand for each industry
        MC_errs    = [2,] vector, capital and labor market clearing
                     errors given r ans w

    Returns: MC_errs
    '''
    (S, alpha, beta, sigma, b_guess, ci_tilde, A, gamma, epsilon,
        delta, xi, pi, I, M, S, n, ss_tol) = objs
    r, w = rwvec
    if (r + delta).min() <= 0 or w <=0:
        MCKerr = 9999.
        MCLerr = 9999.
        MC_errs = np.array((MCKerr, MCLerr))
    elif (r + delta).min() > 0 and w > 0:
        pm_params = (A, gamma, epsilon, delta)
        p_m = get_pm(pm_params, r, w)   
        p_c = np.dot(pi,p_m)
        p = get_p(alpha, p_c)
        cbe_params = (alpha, beta, sigma, r, w, p, ss_tol)
        b, c, c_cstr, c_i, ci_cstr, euler_errors = \
            get_cbess(cbe_params, b_guess, p_c, ci_tilde, I, S, n)
        C_i = get_C_i(c_i)
        Ym_params = (r, w)
        Ym_init = (np.dot(np.reshape(C_i,(1,I)),pi))/I
        Ym = opt.fsolve(solve_Ym, Ym_init, args=(Ym_params, C_i, A, gamma, epsilon, delta, xi, pi, I, M), xtol=ss_tol, col_deriv=1)
        Km_params = (r,w)
        Km = get_Km(Km_params, Ym, A, gamma, epsilon, delta)
        Lm_params = (r, w)
        Lm = get_Lm(Lm_params, Km, gamma, epsilon, delta)
        MCKerr = Km.sum() - b.sum()
        MCLerr = Lm.sum() - n.sum()
        MC_errs = np.array((MCKerr, MCLerr))

    return MC_errs


def SS(params, rw_init, b_guess, ci_tilde, A, gamma, epsilon, delta, xi, pi, I,
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
        ci_tilde = [I,] vector, minimum consumption values for all goods
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
        get_pm
        get_p
        get_cbess
        get_C_i
        get_YKm
        get_Lmvec

    Objects in function:
        start_time  = scalar, current processor time in seconds (float)
        MCerrs_objs = length 12 tuple, objects to be passed in to
                      MCerrs function: (S, alpha, beta, sigma, b_guess,
                      ci_tilde, A, gamma, epsilon, delta, n,
                      ss_tol)
        rw_ss       = [2,] vector, steady-state r and w
        r_ss        = scalar, steady-state interest rate
        w_ss        = scalar > 0, steady-state wage
        pm_params   = length 4 tuple, vectors to be passed in to get_pm
                      (A, gamma, epsilon, delta)
        pm_ss       = [M,] vector, steady-state output prices for each industry
        pc_ss       = [I,] vector, steady-state consumption good prices
        p_ss        = scalar > 0, steady-state composite good price
        cbe_params  = length 7 tuple, parameters for get_cbess function
                      (alpha, beta, sigma, r_ss, w_ss, p_ss, ss_tol)
        b_ss        = [S-1,] vector, steady-state savings
        c_ss        = [S,] vector, steady-state composite consumption
        c_cstr      = [S,] boolean vector, =True if cbar_s<=0 for some s
        ci_ss       = [I,S] matrix, steady-state consumption of each
                      good
        ci_cstr     = [I,S] boolean matrix, =True if c_ss{i,s}<=0 for
                      given c_ss
        EulErr_ss   = [S-1,] vector, steady-state Euler errors
        Ci_ss       = [I,] vector, total demand for goods from each
                      industry
        Ym_params   = length 2 tuple, parameters for get_YKm
                      function: (r_ss, w_ss)
        Ym_ss       = [M,] vector, steady-state total output for each
                      industry
        Km_ss       = [M,] vector, steady-state capital demand for each
                      industry
        Lm_params   = length 2 tuple, parameters for get_Lmvec function:
                      (r_ss, w_ss)
        Lm_ss       = [M,] vector, steady-state labor demand for each
                      industry
        MCK_err_ss  = scalar, steady-state capital market clearing error
        MCL_err_ss  = scalar, steady-state labor market clearing error
        MCerr_ss    = [2,] vector, steady-state capital and labor market
                      clearing errors
        ss_time     = scalar, time to compute SS solution (seconds)
        svec         = [S,] vector, age-s indices from 1 to S
        b_ss0        = [S,] vector, age-s wealth levels including b_1=0

    Returns: r_ss, w_ss, pm_ss, p_ss, b_ss, c_ss, cm_ss, EulErr_ss,
             Cm_ss, Ym_ss, Km_ss, Lm_ss, MCK_err_ss, MCL_err_ss, ss_time
    '''
    start_time = time.clock()
    S, alpha, beta, sigma, ss_tol = params
    MCerrs_objs = (S, alpha, beta, sigma, b_guess, ci_tilde, A,
                  gamma, epsilon, delta, xi, pi, I, M, S, n, ss_tol)
    rw_ss = opt.fsolve(MCerrs, rw_init, args=(MCerrs_objs),
                xtol=ss_tol)
    r_ss, w_ss = rw_ss
    pm_params = (A, gamma, epsilon, delta)
    pm_ss = get_pm(pm_params, r_ss, w_ss)

    pc_ss = np.dot(pi,pm_ss)
    p_ss = get_p(alpha, pc_ss)
    cbe_params = (alpha, beta, sigma, r_ss, w_ss, p_ss, ss_tol)
    b_ss, c_ss, c_cstr, ci_ss, ci_cstr, EulErr_ss = \
        get_cbess(cbe_params, b_guess, pc_ss, ci_tilde, I, S, n)
    Ci_ss = get_C_i(ci_ss)
    Ym_params = (r_ss, w_ss)
    Ym_init = (np.dot(np.reshape(Ci_ss,(1,I)),pi))/I
    Ym_ss = opt.fsolve(solve_Ym, Ym_init, args=(Ym_params, Ci_ss, A, gamma, epsilon, delta, xi, pi, I, M), xtol=ss_tol, col_deriv=1)
    Km_params = (r_ss,w_ss)
    Km_ss = get_Km(Km_params, Ym_ss, A, gamma, epsilon, delta)
    Lm_params = (r_ss, w_ss)
    Lm_ss = get_Lm(Lm_params, Km_ss, gamma, epsilon, delta)

    MCK_err_ss = Km_ss.sum() - b_ss.sum()
    MCL_err_ss = Lm_ss.sum() - n.sum()
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
        plt.plot(svec, c_ss)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Steady-state distribution of composite consumption')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual consumption $\bar{c}_{s}$')
        # plt.savefig('c_ss_Chap11')
        plt.show()

        # Plot steady-state distribution of individual good consumption
        fig, ax = plt.subplots()
        plt.plot(svec, ci_ss[0,:], 'r--', label='Good 1')
        plt.plot(svec, ci_ss[1,:], 'b', label='Good 2')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.legend(loc='center right')
        plt.title('Steady-state distribution of goods consumption')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual consumption $\bar{c}_{m,s}$')
        # plt.savefig('cm_ss_Chap11')
        plt.show()

    return (r_ss, w_ss, pc_ss, p_ss, b_ss, c_ss, ci_ss, EulErr_ss,
           Ci_ss, Ym_ss, Km_ss, Lm_ss, MCK_err_ss, MCL_err_ss, ss_time)
