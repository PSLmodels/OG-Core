'''
------------------------------------------------------------------------
All the functions for the SS computation from Chapter 11 of the OG
textbook
    feasible
    get_pm
    get_p
    get_cbess
    EulerSys_b
    get_cvec
    get_cmmat
    get_b_errors
    get Cmvec
    get_YKmvec
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

def feasible(params, rw_init, b_guess, cm_tilde, A, gamma, epsilon,
  delta, nvec):
    '''
    Determines whether a particular guess for the steady-state values
    or r and w are feasible. Feasibility means that
    r + delta > 0, w > 0, implied c_s>0, c_{1,s}>0, and c_{2,s}>0 for
    all s and implied K1 + K2 > 0

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
        cm_tilde = [2,] vector, minimum consumption values for all goods
        A     = [2,] vector, total factor productivity values for all
                   industries
        gamma   = [2,] vector, capital shares of income for all
                   industries
        epsilon   = [2,] vector, elasticities of substitution between
                   capital and labor for all industries
        delta   = [2,] vector, model period depreciation rates for all
                   industries
        nvec     = [S,] vector, exogenous labor supply n_{s}

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
        K1K2_cstr  = boolean, =True if K1 + K2 <= 0
        pm_params  = length 4 tuple, vectors to be passed in to get_pm
                     (A, gamma, epsilon, delta)
        pmvec      = [2,] vector, prices in each industry
        p          = scalar > 0, composite good price
        cbe_params = length 7 tuple, parameters for get_cbess function
                     (alpha, beta, sigma, r, w, p, ss_tol)
        bvec       = [S-1,] vector, optimal savings given prices
        cvec       = [S,] vector, optimal composite consumption given
                     prices
        cmmat      = [2,S] matrix, optimal consumption of each good
                     given prices
        cm_cstr    = [2,S] boolean matrix, =True if c_{m,s}<=0 for
                     given c_s
        eulvec     = [S-1,] vector, Euler equations from optimal savings
        K1pK2      = scalar, sum of all savings

    Returns: GoodGuess, r_cstr, w_cstr, c_cstr, cm_cstr, K1K2_cstr
    '''
    S, alpha, beta, sigma, ss_tol = params
    r, w = rw_init
    GoodGuess = True
    r_cstr = False
    w_cstr = False
    c_cstr = np.zeros(S, dtype=bool)
    K1K2_cstr = False
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
        pmvec = get_pm(pm_params, r, w)
        p = get_p(alpha, pmvec)
        cbe_params = (alpha, beta, sigma, r, w, p, ss_tol)
        bvec, cvec, c_cstr, cmmat, cm_cstr, eulvec = \
            get_cbess(cbe_params, b_guess, pmvec, cm_tilde, nvec)
        # Check K1 + K2
        K1pK2 = bvec.sum()
        K1K2_cstr = K1pK2 <= 0
        if K1K2_cstr == True or c_cstr.max() == 1 or cm_cstr.max() == 1:
            GoodGuess = False

    return GoodGuess, r_cstr, w_cstr, c_cstr, cm_cstr, K1K2_cstr


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
        pmvec = [M,] vector, industry prices

    Returns: pmvec
    '''
    A, gamma, epsilon, delta = params
    pmvec = (1 / A) * ((gamma * ((r + delta) ** (1 - epsilon)) +
            (1 - gamma) * (w ** (1 - epsilon))) ** (1 / (1 - epsilon)))
    return pmvec


def get_p(alpha, pmvec):
    '''
    Generates vector of industry prices p_m from r and w

    Inputs:
        alpha = scalar in (0,1), expenditure share on good 1
        pmvec = [2,] vector, normalized industry prices

    Functions called: None

    Objects in function:
        p1 = scalar > 0, price in industry 1
        p2 = scalar > 0, price in industry 2
        p = scalar > 0, price of composite good

    Returns: p
    '''
    p1, p2 = pmvec
    p = (((p1 / alpha) ** alpha) *
        ((p2 / (1 - alpha)) ** (1 - alpha)))
    return p


def get_cbess(params, b_guess, pmvec, cm_tilde, nvec):
    '''
    Generates vectors for individual savings, composite consumption,
    industry-specific consumption, constraint vectors, and Euler errors
    for a given set of prices (r, w, p, p_m).

    Inputs:
        params   = length 7 tuple, (alpha, beta, sigma, r, w, p ss_tol)
        alpha    = scalar in (0,1), expenditure share on good 1
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p        = scalar > 0, composite good price
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve
        b_guess  = [S-1,] vector, initial guess for savings vector
        pmvec    = [2,] vector, prices in each industry
        cm_tilde = [2,] vector, minimum consumption values for all goods
        nvec     = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        EulerSys_b
        get_cvec
        get_cmmat
        get_b_errors

    Objects in function:
        eulb_objs  = length 9 tuple, objects to be passed in to
                     EulerSys_b: (alpha, beta, sigma, r, w, p, pmvec,
                     cm_tilde, nvec)
        bvec       = [S-1,] vector, optimal lifetime savings decisions
        c_params   = length 3 tuple, parameters for get_cvec (r, w, p)
        cvec       = [S,] vector, optimal lifetime consumption
        c_cstr     = [S,] boolean vector, =True if c_s<=0
        cm_params  = length 2 tuple, parameters for get_cmmat (alpha, p)
        cmmat      = [2,S] matrix,  optimal consumption of each good
                     given prices
        cm_cstr    = [2,S] boolean matrix, =True if c_{m,s}<=0 for given
                     c_s
        eul_params = length 2 tuple, parameters for get_b_errors
                     (beta, sigma)
        eulvec     = [S-1,] vector, Euler errors from savings decisions

    Returns: bvec, cvec, c_cstr, cmmat, cm_cstr, eulvec
    '''
    alpha, beta, sigma, r, w, p, ss_tol = params
    eulb_objs = (alpha, beta, sigma, r, w, p, pmvec, cm_tilde, nvec)
    bvec = opt.fsolve(EulerSys_b, b_guess, args=(eulb_objs), xtol=ss_tol)
    c_params = (r, w, p)
    cvec, c_cstr = get_cvec(c_params, nvec, bvec, pmvec, cm_tilde)
    cm_params = (alpha, p)
    cmmat, cm_cstr = get_cmmat(cm_params, cvec, pmvec, cm_tilde)
    eul_params = (beta, sigma)
    eulvec = get_b_errors(eul_params, r, cvec, c_cstr, diff=True)

    return bvec, cvec, c_cstr, cmmat, cm_cstr, eulvec


def EulerSys_b(bvec, *objs):
    '''
    Generates vector of all Euler errors for a given bvec, which errors
    characterize all optimal lifetime savings decisions

    Inputs:
        bvec     = [S-1,] vector, lifetime savings decisions
        objs     = length 9 tuple,
                   (alpha, beta, sigma, r, w, p, pmvec, cm_tilde, nvec)
        alpha    = scalar in (0,1), expenditure share on good 1
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p        = scalar > 0, composite good price
        pmvec    = [2,] vector, prices in each industry
        cm_tilde = [2,] vector, minimum consumption values for all goods
        nvec     = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        get_cvec
        get_cmmat
        get_b_errors

    Objects in function:
        c_params     = length 3 tuple, parameters for get_cvec (r, w, p)
        cvec         = [S,] vector, remaining lifetime consumption
                       levels implied by bvec
        c_cstr       = [S, ] boolean vector, =True if c_{s,t}<=0
        cm_params    = length 2 tuple, parameters for get_cmmat
                       (alpha, p)
        cmmat        = [2,S] matrix, consumption values for each good
                       and age c_{m,s}
        cm_cstr      = [2,S] boolean matrix, =True if c_{m,s}<=0
        b_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_err_vec    = [S-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_err_vec
    '''
    alpha, beta, sigma, r, w, p, pmvec, cm_tilde, nvec = objs
    c_params = (r, w, p)
    cvec, c_cstr = get_cvec(c_params, nvec, bvec, pmvec, cm_tilde)
    cm_params = (alpha, p)
    cmmat, cm_cstr = get_cmmat(cm_params, cvec, pmvec, cm_tilde)
    b_err_params = (beta, sigma)
    b_err_vec = get_b_errors(b_err_params, r, cvec, c_cstr, diff=True)
    return b_err_vec


def get_cvec(params, nvec, bvec, pmvec, cm_tilde):
    '''
    Generate lifetime consumption given prices and savings decisions

    Inputs:
        params   = length 3 tuple, (r, w, p)
        r        = scalar > 0, interest rate
        w        = scalar > 0, real wage
        p        = scalar > 0, composite good price
        nvec     = [S,] vector, exogenous labor supply n_{s}
        bvec     = [S-1,] vector, distribution of savings b_{s+1}
        pmvec    = [2,] vector, prices in each industry
        cm_tilde = [2,] vector, minimum consumption values for all goods

    Functions called: None

    Objects in function:
        b_s    = [S,] vector, 0 in first element and bvec in last S-1
                 elements
        b_sp1  = [S,] vector, bvec in first S-1 elements and 0 in last
                 element
        cvec   = [S,] vector, composite consumption by age c_s
        c_cstr = [S,] boolean vector, =True if element c_s <= 0

    Returns: cvec, c_constr
    '''
    r, w, p = params
    b_s = np.append([0], bvec)
    b_sp1 = np.append(bvec, [0])
    cvec = (1 / p) * ((1 + r) * b_s + w * nvec -
           (pmvec * cm_tilde).sum() - b_sp1)
    c_cstr = cvec <= 0
    return cvec, c_cstr


def get_cmmat(params, cvec, pmvec, cm_tilde):
    '''
    Generates matrix of consumptions of each type of good given prices
    and composite consumption

    Inputs:
        params   = length 2 tuple, (alpha, p)
        alpha    = scalar in (0,1), expenditure share on good 1
        p        = scalar > 0, composite good price
        cvec     = [S,] vector, composite consumption by age c_s
        pmvec    = [2,] vector, prices in each industry
        cm_tilde = [2,] vector, minimum consumption values for all goods

    Functions called: None

    Objects in function:
        c1vec   = [S,] vector, consumption values of good 1 by age
                  c_{1,s}
        c2vec   = [S,] vector, consumption values of good 2 by age
                  c_{2,s}
        cmmat   = [2,S] matrix, consumption of each good by age c_{m,s}
        cm_cstr = [S,] boolean vector, =True if element c_s <= 0

    Returns: cmmat, cm_cstr
    '''
    alpha, p = params
    c1vec = ((alpha * p * cvec) / pmvec[0]) + cm_tilde[0]
    c2vec = (((1 - alpha) * p * cvec) / pmvec[1]) + cm_tilde[1]
    cmmat = np.vstack((c1vec, c2vec))
    cm_cstr = cmmat <= 0
    return cmmat, cm_cstr


def get_b_errors(params, r, cvec, c_cstr, diff):
    '''
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings

    Inputs:
        params = length 2 tuple, (beta, sigma)
        beta   = scalar in [0,1), discount factor
        sigma  = scalar > 0, coefficient of relative risk aversion
        r      = scalar > 0, interest rate
        cvec   = [S,] vector, distribution of consumption by age c_S
        c_cstr = [S,] boolean vector, =True if c_s<=0 for given bvec
        diff   = boolean, =True if use simple difference Euler errors.
                 Use percent difference errors otherwise.

    Functions called: None

    Objects in function:
        mu_c     = [S-1,] vector, marginal utility of current
                   consumption
        mu_cp1   = [S-1,] vector, marginal utility of next period
                   consumption
        b_errors = [S-1,] vector, Euler errors with errors = 0
                   characterizing optimal savings bvec

    Returns: b_errors
    '''
    beta, sigma = params
    cvec[c_cstr] = 9999. # Each consumption must be positive to
                         # generate marginal utilities
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    if diff == True:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
        b_errors[c_cstr[:-1]] = 9999.
        b_errors[c_cstr[1:]] = 9999.
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        b_errors[c_cstr[:-1]] = 9999. / 100
        b_errors[c_cstr[1:]] = 9999. / 100
    return b_errors


def get_Cmvec(cmmat):
    '''
    Generates vector of aggregate consumption C_m of good m

    Inputs:
        cmmat    = [2,S] matrix, distribution of individual consumption
                   c_{m,s}

    Functions called: None

    Objects in function:
        Cmvec = [2,] vector, aggregate consumption of all goods

    Returns: Cmvec
    '''
    Cmvec = cmmat.sum(axis=1)
    return Cmvec


def get_YKmvec(params, Cmvec, pmvec, A, gamma, epsilon, delta):
    '''
    Generates vector of aggregate output Y_m of good m and capital
    demand K_m for good m given r and w

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        Cmvec  = [2,] vector, aggregate consumption of all goods
        pmvec  = [2,] vector, prices in each industry
        A   = [2,] vector, total factor productivity values for all
                 industries
        gamma = [2,] vector, capital shares of income for all
                 industries
        epsilon = [2,] vector, elasticities of substitution between
                 capital and labor for all industries
        delta = [2,] vector, model period depreciation rates for all
                 industries

    Functions called: None

    Objects in function:
        aa    = [2,] vector, gamma
        bb    = [2,] vector, 1 - gamma
        cc    = [2,] vector, (1 - gamma) / gamma
        dd    = [2,] vector, (r + delta) / w
        ee    = [2,] vector, 1 / epsilon
        ff    = [2,] vector, (epsilon - 1) / epsilon
        gg    = [2,] vector, epsilon - 1
        hh    = [2,] vector, epsilon / (1 - epsilon)
        ii    = [2,] vector, ((1 / A) * (((aa ** ee) + (bb ** ee) *
                (cc ** ff) * (dd ** gg)) ** hh))
        Ymvec = [2,] vector, aggregate output of all industries
        Kmvec = [2,] vector, capital demand of all industries

    Returns: Ymvec, Kmvec
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
    ii = ((1 / A) *
         (((aa ** ee) + (bb ** ee) * (cc ** ff) * (dd ** gg)) ** hh))
    Ymvec = Cmvec / (1 - delta * ii)
    Kmvec = Ymvec * ii
    return Ymvec, Kmvec


def get_Lmvec(params, Ymvec, Kmvec, pmvec, delta):
    '''
    Generates vector of labor demand L_m for each industry m

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        Ymvec  = [2,] vector, aggregate output of all goods
        Kmvec  = [2,] vector, capital demand in all industries
        pmvec  = [2,] vector, prices in each industry
        delta = [2,] vector, model period depreciation rates for all
                 industries

    Functions called: None

    Objects in function:
        Lmvec = [2,] vector, aggregate output of all goods

    Returns: Lmvec
    '''
    r, w = params
    Lmvec = (pmvec * Ymvec - (r + delta) * Kmvec) / w

    return Lmvec

def get_Lmvec_alt(params, Kmvec, gamma, epsilon, delta):
    '''
    Generates vector of labor demand L_m for each industry m

    Inputs:
        params = length 2 tuple, (r, w)
        r      = scalar > 0, interest rate
        w      = scalar > 0, real wage
        Ymvec  = [2,] vector, aggregate output of all goods
        Kmvec  = [2,] vector, capital demand in all industries
        pmvec  = [2,] vector, prices in each industry
        delta = [2,] vector, model period depreciation rates for all
                 industries

    Functions called: None

    Objects in function:
        Lmvec = [2,] vector, aggregate output of all goods

    Returns: Lmvec
    '''
    r, w = params
    Lmvec = ((1-gamma)/gamma)*Kmvec*(((r+delta)/w)**epsilon)

    return Lmvec


def MCerrs(rwvec, *objs):
    '''
    Returns capital and labor market clearing condition errors given
    particular values of r and w

    Inputs:
        rwvec    = [2,] vector, given values of r and w
        objs     = length 12 tuple, (S, alpha, beta, sigma, b_guess,
                   cm_tilde, A, gamma, epsilon, delta, nvec, ss_tol)
        S        = integer in [3,80], number of periods an individual
                   lives
        alpha    = scalar in (0,1), expenditure share on good 1
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        b_guess  = [S-1,] vector, initial guess for savings to use in
                   fsolve in get_cbess
        cm_tilde = [2,] vector, minimum consumption values for all goods
        A     = [2,] vector, total factor productivity values for all
                   industries
        gamma   = [2,] vector, capital shares of income for all
                   industries
        epsilon   = [2,] vector, elasticities of substitution between
                   capital and labor for all industries
        delta   = [2,] vector, model period depreciation rates for all
                   industries
        nvec     = [S,] vector, exogenous labor supply n_{s}
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve

    Functions called:
        get_pm
        get_p
        get_cbess
        get_Cmvec
        get_YKmvec
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
        pmvec      = [2,] vector, prices in each industry
        p          = scalar > 0, composite good price
        cbe_params = length 7 tuple, parameters for get_cbess function
                     (alpha, beta, sigma, r, w, p, ss_tol)
        bvec       = [S-1,] vector, optimal savings given prices
        cvec       = [S,] vector, optimal composite consumption given
                     prices
        c_cstr     = [S,] boolean vector, =True if c_s<=0 for some s
        cmmat      = [2,S] matrix, optimal consumption of each good
                     given prices
        cm_cstr    = [2,S] boolean matrix, =True if c_{m,s}<=0 for
                     given c_s
        eulvec     = [S-1,] vector, Euler equations from optimal savings
        Cmvec      = [2,] vector, total consumption demand for each
                     industry
        Ym_params  = length 2 tuple, parameters for get_YKmvec function:
                     (r, w)
        Ymvec      = [2,] vector, total output for each industry
        Kmvec      = [2,] vector, capital demand for each industry
        Lm_params  = length 2 tuple, parameters for get_Lmvec function:
                     (r, w)
        Lmvec      = [2,] vector, labor demand for each industry
        MC_errs    = [2,] vector, capital and labor market clearing
                     errors given r ans w

    Returns: MC_errs
    '''
    (S, alpha, beta, sigma, b_guess, cm_tilde, A, gamma, epsilon,
        delta, nvec, ss_tol) = objs
    r, w = rwvec
    if (r + delta).min() <= 0 or w <=0:
        MCKerr = 9999.
        MCLerr = 9999.
        MC_errs = np.array((MCKerr, MCLerr))
    elif (r + delta).min() > 0 and w > 0:
        pm_params = (A, gamma, epsilon, delta)
        pmvec = get_pm(pm_params, r, w)
        p = get_p(alpha, pmvec)
        cbe_params = (alpha, beta, sigma, r, w, p, ss_tol)
        bvec, cvec, c_cstr, cmmat, cm_cstr, eulvec = \
            get_cbess(cbe_params, b_guess, pmvec, cm_tilde, nvec)
        Cmvec = get_Cmvec(cmmat)
        Ym_params = (r, w)
        Ymvec, Kmvec = get_YKmvec(Ym_params, Cmvec, pmvec, A, gamma,
                       epsilon, delta)
        Lm_params = (r, w)
        #Lmvec = get_Lmvec(Lm_params, Ymvec, Kmvec, pmvec, delta)
        Lmvec = get_Lmvec_alt(Lm_params, Kmvec, gamma, epsilon, delta)
        MCKerr = Kmvec.sum() - bvec.sum()
        MCLerr = Lmvec.sum() - nvec.sum()
        MC_errs = np.array((MCKerr, MCLerr))

    return MC_errs


def SS(params, rw_init, b_guess, cm_tilde, A, gamma, epsilon, delta,
    nvec, graphs):
    '''
    Generates all endogenous steady-state objects

    Inputs:
        params   = length 5 tuple, (S, alpha, beta, sigma, ss_tol)
        S        = integer in [3,80], number of periods an individual
                   lives
        alpha    = scalar in (0,1), expenditure share on good 1
        beta     = scalar in [0,1), discount factor for each model
                   period
        sigma    = scalar > 0, coefficient of relative risk aversion
        ss_tol   = scalar > 0, tolerance level for steady-state fsolve
        rw_init  = [2,] vector, initial guesses for steady-state r and w
        b_guess  = [S-1,] vector, initial guess for savings to use in
                   fsolve in get_cbess
        cm_tilde = [2,] vector, minimum consumption values for all goods
        A     = [2,] vector, total factor productivity values for all
                   industries
        gamma   = [2,] vector, capital shares of income for all
                   industries
        epsilon   = [2,] vector, elasticities of substitution between
                   capital and labor for all industries
        delta   = [2,] vector, model period depreciation rates for all
                   industries
        nvec     = [S,] vector, exogenous labor supply n_{s}
        graphs   = boolean, =True if want graphs of steady-state objects

    Functions called:
        MCerrs
        get_pm
        get_p
        get_cbess
        get_Cmvec
        get_YKmvec
        get_Lmvec

    Objects in function:
        start_time  = scalar, current processor time in seconds (float)
        MCerrs_objs = length 12 tuple, objects to be passed in to
                      MCerrs function: (S, alpha, beta, sigma, b_guess,
                      cm_tilde, A, gamma, epsilon, delta, nvec,
                      ss_tol)
        rw_ss       = [2,] vector, steady-state r and w
        r_ss        = scalar, steady-state interest rate
        w_ss        = scalar > 0, steady-state wage
        pm_params   = length 4 tuple, vectors to be passed in to get_pm
                      (A, gamma, epsilon, delta)
        pm_ss       = [2,] vector, steady-state prices in each industry
        p_ss        = scalar > 0, steady-state composite good price
        cbe_params  = length 7 tuple, parameters for get_cbess function
                      (alpha, beta, sigma, r_ss, w_ss, p_ss, ss_tol)
        b_ss        = [S-1,] vector, steady-state savings
        c_ss        = [S,] vector, steady-state composite consumption
        c_cstr      = [S,] boolean vector, =True if cbar_s<=0 for some s
        cm_ss       = [2,S] matrix, steady-state consumption of each
                      good
        cm_cstr     = [2,S] boolean matrix, =True if cbar_{m,s}<=0 for
                      given cbar_s
        EulErr_ss   = [S-1,] vector, steady-state Euler errors
        Cm_ss       = [2,] vector, total demand for goods from each
                      industry
        Ym_params   = length 2 tuple, parameters for get_YKmvec
                      function: (r_ss, w_ss)
        Ym_ss       = [2,] vector, steady-state total output for each
                      industry
        Km_ss       = [2,] vector, steady-state capital demand for each
                      industry
        Lm_params   = length 2 tuple, parameters for get_Lmvec function:
                      (r_ss, w_ss)
        Lm_ss       = [2,] vector, steady-state labor demand for each
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
    MCerrs_objs = (S, alpha, beta, sigma, b_guess, cm_tilde, A,
                  gamma, epsilon, delta, nvec, ss_tol)
    rw_ss = opt.fsolve(MCerrs, rw_init, args=(MCerrs_objs),
                xtol=ss_tol)
    r_ss, w_ss = rw_ss
    pm_params = (A, gamma, epsilon, delta)
    pm_ss = get_pm(pm_params, r_ss, w_ss)
    p_ss = get_p(alpha, pm_ss)
    cbe_params = (alpha, beta, sigma, r_ss, w_ss, p_ss, ss_tol)
    b_ss, c_ss, c_cstr, cm_ss, cm_cstr, EulErr_ss = \
        get_cbess(cbe_params, b_guess, pm_ss, cm_tilde, nvec)
    Cm_ss = get_Cmvec(cm_ss)
    Ym_params = (r_ss, w_ss)
    Ym_ss, Km_ss = get_YKmvec(Ym_params, Cm_ss, pm_ss, A, gamma, epsilon, delta)
    Lm_params = (r_ss, w_ss)
    #Lm_ss = get_Lmvec(Lm_params, Ym_ss, Km_ss, pm_ss, delta)
    Lm_ss = get_Lmvec_alt(Lm_params, Km_ss, gamma, epsilon, delta)
    MCK_err_ss = Km_ss.sum() - b_ss.sum()
    MCL_err_ss = Lm_ss.sum() - nvec.sum()
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
        plt.plot(svec, cm_ss[0,:], 'r--', label='Good 1')
        plt.plot(svec, cm_ss[1,:], 'b', label='Good 2')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.legend(loc='center right')
        plt.title('Steady-state distribution of goods consumption')
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Individual consumption $\bar{c}_{m,s}$')
        # plt.savefig('cm_ss_Chap11')
        plt.show()

    return (r_ss, w_ss, pm_ss, p_ss, b_ss, c_ss, cm_ss, EulErr_ss,
           Cm_ss, Ym_ss, Km_ss, Lm_ss, MCK_err_ss, MCL_err_ss, ss_time)
