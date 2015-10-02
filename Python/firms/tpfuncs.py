'''
------------------------------------------------------------------------
All the functions for the TPI computation for the model with S-period
lived agents, exogenous labor, and two industries and two goods.
    get_pmpath
    get_ppath
    get_cbepath

    get_cvec_lf
    LfEulerSys
    paths_life

    TPI
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import ssfuncs as ssf
reload(ssf)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import sys

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def get_pmpath(params, rpath, wpath):
    '''
    Generates time path of industry prices p_m from rpath and wpath

    Inputs:
        params = length 4 tuple, (A, gamma, epsilon, delta)
        A   = [2,] vector, total factor productivity for each
                 industry
        gamma = [2,] vector, capital share of income for each industry
        epsilon = [2,] vector, elasticity of substitution between capital
                 and labor for each industry
        delta = [2,] vector, capital depreciation rate for each
                 industry
        rpath  = [T+S-2,] vector, time path of interest rate
        w      = [T+S-2,] vector, time path of wage

    Functions called: None

    Objects in function:
        pmpath = [2, T+S-2] matrix, time path of industry prices

    Returns: pmpath
    '''
    A, gamma, epsilon, delta = params
    pmpath = np.zeros((len(A), len(rpath)))
    pmpath[0, :] = (1 / A[0]) * ((gamma[0] * ((rpath + delta[0]) **
                   (1 - epsilon[0])) + (1 - gamma[0]) * (wpath **
                   (1 - epsilon[0]))) ** (1 / (1 - epsilon[0])))
    pmpath[1, :] = (1 / A[1]) * ((gamma[1] * ((rpath + delta[1]) **
                   (1 - epsilon[1])) + (1 - gamma[1]) * (wpath **
                   (1 - epsilon[1]))) ** (1 / (1 - epsilon[1])))
    return pmpath


def get_ppath(alpha, pmpath):
    '''
    Generates time path of composite price p from pmpath

    Inputs:
        alpha = scalar in (0,1), expenditure share on good 1
        pmpath = [2, T+S-2] matrix, time path of industry prices

    Functions called: None

    Objects in function:
        ppath = [T+S-2] vector, time path of price of composite good

    Returns: ppath
    '''
    ppath = (((pmpath[0, :] / alpha) ** alpha) *
        ((pmpath[1, :] / (1 - alpha)) ** (1 - alpha)))
    return ppath


def get_cbepath(params, Gamma1, rpath_init, wpath_init, pmpath, ppath,
  cm_tilde, nvec):
    '''
    Generates matrices for the time path of the distribution of
    individual savings, individual composite consumption, individual
    consumption of each type of good, and the Euler errors associated
    with the savings decisions.

    Inputs:
        params     = length 6 tuple, (S, T, alpha, beta, sigma, tpi_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        T          = integer > S, number of time periods until steady
                     state
        alpha      = scalar in (0,1), expenditure share on good 1
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        tpi_tol    = scalar > 0, tolerance level for fsolve's in TPI
        Gamma1     = [S-1,] vector, initial period savings distribution
        rpath_init = [T+S-1,] vector, initial guess for the time path of
                     the interest rate
        wpath_init = [T+S-1,] vector, initial guess for the time path of
                     the wage
        pmpath     = [2, T+S-1] matrix, time path of industry prices
        ppath      = [T+S-1] vector, time path of composite price
        cm_tilde   = [2,] vector, minimum consumption values for all
                     goods
        nvec       = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        paths_life

    Objects in function:
        bpath      = [S-1, T+S-1] matrix,
        cpath      = [S, T+S-1] matrix,
        cmpath     = [S, T+S-1, 2] array,
        eulerrpath = [S-1, T+S-1] matrix,
        pl_params  = length 4 tuple, parameters to pass into paths_life
                     (S, beta, sigma, TPI_tol)
        p          = integer >= 2, represents number of periods
                     remaining in a lifetime, used to solve incomplete
                     lifetimes
        b_guess    = [p-1,] vector, initial guess for remaining lifetime
                     savings, taken from previous cohort's choices
        bveclf     = [p-1,] vector, optimal remaining lifetime savings
                     decisions
        cveclf     = [p,] vector, optimal remaining lifetime consumption
                     decisions
        b_err_veclf = [p-1,] vector, Euler errors associated with
                      optimal remaining lifetime savings decisions
        DiagMaskb   = [p-1, p-1] boolean identity matrix
        DiagMaskc   = [p, p] boolean identity matrix

    Returns: bpath, cpath, cmpath, eulerrpath
    '''
    S, T, alpha, beta, sigma, tpi_tol = params
    bpath = np.append(Gamma1.reshape((S-1,1)), np.zeros((S-1, T+S-2)),
            axis=1)
    cpath = np.zeros((S, T+S-1))
    cmpath = np.zeros((S, T+S-1, 2))
    eulerrpath = np.zeros((S-1, T+S-1))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    cpath[S-1, 0] = (1 / ppath[0]) * ((1 + rpath_init[0]) * Gamma1[S-2]
        + wpath_init[0] * nvec[S-1] - (pmpath[:, 0] * cm_tilde).sum())
    cmpath[S-1, 0, 0] = alpha * ((ppath[0] * cpath[S-1, 0]) /
                        pmpath[0, 0]) + cm_tilde[0]
    cmpath[S-1, 0, 1] = (1 - alpha) * ((ppath[0] * cpath[S-1, 0]) /
                        pmpath[1, 0]) + cm_tilde[1]
    pl_params = (S, alpha, beta, sigma, tpi_tol)
    for p in xrange(2, S):
        # b_guess = b_ss[-p+1:]
        b_guess = np.diagonal(bpath[S-p:, :p-1])
        bveclf, cveclf, cmmatlf, b_err_veclf = paths_life(pl_params,
            S-p+1, Gamma1[S-p-1], cm_tilde, nvec[-p:], rpath_init[:p],
            wpath_init[:p], pmpath[:, :p], ppath[:p], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        DiagMaskb = np.eye(p-1, dtype=bool)
        DiagMaskc = np.eye(p, dtype=bool)
        bpath[S-p:, 1:p] = DiagMaskb * bveclf + bpath[S-p:, 1:p]
        cpath[S-p:, :p] = DiagMaskc * cveclf + cpath[S-p:, :p]
        cmpath[S-p:, :p, 0] = (DiagMaskc * cmmatlf[0, :] +
                              cmpath[S-p:, :p, 0])
        cmpath[S-p:, :p, 1] = (DiagMaskc * cmmatlf[1, :] +
                              cmpath[S-p:, :p, 1])
        eulerrpath[S-p:, 1:p] = (DiagMaskb * b_err_veclf +
                                eulerrpath[S-p:, 1:p])
    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    DiagMaskb = np.eye(S-1, dtype=bool)
    DiagMaskc = np.eye(S, dtype=bool)
    for t in xrange(1, T+1): # Go from periods 1 to T
        # b_guess = b_ss
        b_guess = np.diagonal(bpath[:, t-1:t+S-2])
        bveclf, cveclf, cmmatlf, b_err_veclf = paths_life(pl_params, 1,
            0, cm_tilde, nvec, rpath_init[t-1:t+S-1],
            wpath_init[t-1:t+S-1], pmpath[:, t-1:t+S-1],
            ppath[t-1:t+S-1], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        bpath[:, t:t+S-1] = DiagMaskb * bveclf + bpath[:, t:t+S-1]
        cpath[:, t-1:t+S-1] = DiagMaskc * cveclf + cpath[:, t-1:t+S-1]
        cmpath[:, t-1:t+S-1, 0] = (DiagMaskc * cmmatlf[0, :] +
                                  cmpath[:, t-1:t+S-1, 0])
        cmpath[:, t-1:t+S-1, 1] = (DiagMaskc * cmmatlf[1, :] +
                                  cmpath[:, t-1:t+S-1, 1])
        eulerrpath[:, t:t+S-1] = (DiagMaskb * b_err_veclf +
                                 eulerrpath[:, t:t+S-1])

    return bpath, cpath, cmpath, eulerrpath


def paths_life(params, beg_age, beg_wealth, cm_tilde, nvec, rpath,
               wpath, pmpath, ppath, b_init):
    '''
    Solve for the remaining lifetime savings decisions of an individual
    who enters the model at age beg_age, with corresponding initial
    wealth beg_wealth.

    Inputs:
        params     = length 5 tuple, (S, alpha, beta, sigma, tpi_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        alpha      = scalar in (0,1), expenditure share on good 1
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        tpi_tol    = scalar > 0, tolerance level for fsolve's in TPI
        beg_age    = integer in [1,S-1], beginning age of remaining life
        beg_wealth = scalar, beginning wealth at beginning age
        nvec       = [S-beg_age+1,] vector, remaining exogenous labor
                     supplies
        rpath      = [S-beg_age+1,] vector, remaining lifetime interest
                     rates
        wpath      = [S-beg_age+1,] vector, remaining lifetime wages
        pmpath     = [2, S-beg_age+1] matrix, remaining lifetime
                     industry prices
        ppath      = [S-beg_age+1,] vector, remaining lifetime composite
                     goods prices
        b_init     = [S-beg_age,] vector, initial guess for remaining
                     lifetime savings

    Functions called:
        LfEulerSys
        get_cvec_lf
        c4ssf.get_b_errors

    Objects in function:
        p            = integer in [2,S], remaining periods in life
        b_guess      = [p-1,] vector, initial guess for lifetime savings
                       decisions
        eullf_objs   = length 9 tuple, objects to be passed in to
                       LfEulerSys: (p, beta, sigma, beg_wealth, nvec,
                       rpath, wpath, pmpath, ppath)
        bpath        = [p-1,] vector, optimal remaining lifetime savings
                       decisions
        cpath        = [p,] vector, optimal remaining lifetime
                       consumption decisions
        c_constr     = [p,] boolean vector, =True if c_{p}<=0,
        b_err_params = length 2 tuple, parameters to pass into
                       c4ssf.get_b_errors (beta, sigma)
        b_err_vec    = [p-1,] vector, Euler errors associated with
                       optimal savings decisions

    Returns: bpath, cpath, cmpath, b_err_vec
    '''
    S, alpha, beta, sigma, tpi_tol = params
    p = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(rpath) != p:
        #print len(rpath), S-beg_age+1
        sys.exit("Beginning age and length of rpath do not match.")
    if len(wpath) != p:
        sys.exit("Beginning age and length of wpath do not match.")
    if len(nvec) != p:
        sys.exit("Beginning age and length of nvec do not match.")
    b_guess = 1.01 * b_init
    eullf_objs = (p, beta, sigma, beg_wealth, cm_tilde, nvec, rpath,
                  wpath, pmpath, ppath)
    bpath = opt.fsolve(LfEulerSys, b_guess, args=(eullf_objs),
                       xtol=tpi_tol)
    cpath, c_cstr = get_cvec_lf(cm_tilde, rpath, wpath, pmpath, ppath,
                    nvec, np.append(beg_wealth, bpath))
    cmpath, cm_cstr = get_cmmat_lf(alpha, cm_tilde, cpath, pmpath, ppath)
    b_err_params = (beta, sigma)
    b_err_vec = ssf.get_b_errors(b_err_params, rpath[1:], cpath,
                                   c_cstr, diff=True)
    return bpath, cpath, cmpath, b_err_vec


def LfEulerSys(bvec, *objs):
    '''
    Generates vector of all Euler errors for a given bvec, which errors
    characterize all optimal lifetime decisions

    Inputs:
        bvec       = [p-1,] vector, remaining lifetime savings decisions
                     where p is the number of remaining periods
        objs       = length 9 tuple, (p, beta, sigma, beg_wealth, nvec,
                     rpath, wpath, pmpath, ppath)
        p          = integer in [2,S], remaining periods in life
        beta       = scalar in [0,1), discount factor
        sigma      = scalar > 0, coefficient of relative risk aversion
        beg_wealth = scalar, wealth at the beginning of first age
        nvec       = [p,] vector, remaining exogenous labor supply
        rpath      = [p,] vector, interest rates over remaining life
        wpath      = [p,] vector, wages rates over remaining life

    Functions called:
        get_cvec_lf
        ssf.get_b_errors

    Objects in function:
        bvec2        = [p, ] vector, remaining savings including initial
                       savings
        cvec         = [p, ] vector, remaining lifetime consumption
                       levels implied by bvec2
        c_constr     = [p, ] boolean vector, =True if c_{s,t}<=0
        b_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_err_vec    = [p-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_err_vec
    '''
    (p, beta, sigma, beg_wealth, cm_tilde, nvec, rpath, wpath, pmpath,
        ppath) = objs
    bvec2 = np.append(beg_wealth, bvec)
    cvec, c_cstr = get_cvec_lf(cm_tilde, rpath, wpath, pmpath, ppath,
                               nvec, bvec2)
    b_err_params = (beta, sigma)
    b_err_vec = ssf.get_b_errors(b_err_params, rpath[1:], cvec,
                                   c_cstr, diff=True)
    return b_err_vec


def get_cvec_lf(cm_tilde, rpath, wpath, pmpath, ppath, nvec, bvec):
    '''
    Generates vector of remaining lifetime consumptions from individual
    savings, and the time path of interest rates and the real wages

    Inputs:
        p      = integer in [2,80], number of periods remaining in
                 individual life
        rpath  = [p,] vector, remaining interest rates
        wpath  = [p,] vector, remaining wages
        pmpath = [2, p] matrix, remaining industry prices
        ppath  = [p,] vector, remaining composite prices
        nvec   = [p,] vector, remaining exogenous labor supply
        bvec   = [p,] vector, remaining savings including initial
                 savings

    Functions called: None

    Objects in function:
        c_cstr = [p,] boolean vector, =True if element c_s <= 0
        b_s    = [p,] vector, bvec
        b_sp1  = [p,] vector, last p-1 elements of bvec and 0 in last
                 element
        cvec   = [p,] vector, remaining consumption by age c_s

    Returns: cvec, c_constr
    '''
    b_s = bvec
    b_sp1 = np.append(bvec[1:], [0])
    cvec = (1 / ppath) *((1 + rpath) * b_s + wpath * nvec -
           pmpath[0, :] * cm_tilde[0] - pmpath[1, :] * cm_tilde[1]
           - b_sp1)
    c_cstr = cvec <= 0
    return cvec, c_cstr


def get_cmmat_lf(alpha, cm_tilde, cpath, pmpath, ppath):
    '''
    Generates matrix of remaining lifetime consumptions of individual
    goods

    Inputs:
        p      = integer in [2,80], number of periods remaining in
                 individual life
        rpath  = [p,] vector, remaining interest rates
        wpath  = [p,] vector, remaining wages
        pmpath = [2, p] matrix, remaining industry prices
        ppath  = [p,] vector, remaining composite prices
        nvec   = [p,] vector, remaining exogenous labor supply
        bvec   = [p,] vector, remaining savings including initial
                 savings

    Functions called: None

    Objects in function:
        c_cstr = [p,] boolean vector, =True if element c_s <= 0
        b_s    = [p,] vector, bvec
        b_sp1  = [p,] vector, last p-1 elements of bvec and 0 in last
                 element
        cvec   = [p,] vector, remaining consumption by age c_s

    Returns: cvec, c_constr
    '''
    c1vec = alpha * ((ppath * cpath) / pmpath[0, :]) + cm_tilde[0]
    c2vec = (1 - alpha) * ((ppath * cpath) / pmpath[1, :]) + cm_tilde[1]
    cmmat = np.vstack((c1vec, c2vec))
    cm_cstr = cmmat <= 0
    return cmmat, cm_cstr


def get_Cmpath(cmpath):
    '''
    Generates vector of aggregate consumption C_m of good m

    Inputs:
        cmpath = [S, S+T-1, 2] array, time path of distribution of
                 individual consumption of each good c_{m,s,t}

    Functions called: None

    Objects in function:
        Cmvec = [2,] vector, aggregate consumption of all goods

    Returns: Cmvec
    '''
    C1path = cmpath[:, :, 0].sum(axis=0)
    C2path = cmpath[:, :, 1].sum(axis=0)
    Cmpath = np.vstack((C1path, C2path))
    return Cmpath



def get_YKmpath(params, rpath, wpath, Km_ss, Cmpath, A, gamma,
  epsilon, delta):
    '''
    Generate matrix (vectors) of time path of aggregate output Y_{m,t}
    by industry given r_t, w_t, and C_{m,t}
    '''
    T, r_ss, w_ss = params
    Ympath = np.zeros(Cmpath.shape)
    Kmpath = np.zeros(Cmpath.shape)
    rtp1 = r_ss
    wtp1 = w_ss
    Kmtp1 = Km_ss
    aa = gamma
    bb = 1 - gamma
    cc = (1 - gamma) / gamma
    dd = 1 / epsilon
    ee = epsilon -1
    ff = (epsilon - 1) / epsilon
    gg = epsilon / (1 - epsilon)

    for t in range(T, 0, -1): # Go from periods T to 1
        hh = (rtp1 + delta) / wtp1
        ii = (rpath[t-1] + delta) / wpath[t-1]

        numerator = Cmpath[:,t-1] + Kmtp1
        denominator = 1 + ((1 - delta) / A) * (((aa ** dd) +
                    (bb ** dd) * (ii ** ee) * (cc ** ff)) ** gg)
        Ymt = numerator / denominator
        Ympath[:, t-1] = Ymt
        Kmt =  (Ymt/A) * (((aa ** dd) +
                    (bb ** dd) * (ii ** ee) * (cc ** ff)) ** gg)
        Kmpath[:, t-1] = Kmt
        Ytp1 = Ymt
        Kmtp1 = Kmt
        rtp1 = rpath[t-1]
        wtp1 = wpath[t-1]

    return Ympath, Kmpath



def get_Lmpath(Kmpath, rpath, wpath, gamma, epsilon, delta):
    '''
    Generates vector of labor demand L_m for good m given Y_m, p_m and w

    Inputs:
        Ympath = [2, T] matrix, time path of aggregate output by
                 industry
        wpath  = [T, ] vector, time path of real wage
        pmpath = [2, T] matrix, time path of industry prices
        A   = [2,] vector, total factor productivity values for all
                 industries
        gamma = [2,] vector, capital shares of income for all
                 industries
        epsilon = [2,] vector, elasticities of substitution between
                 capital and labor for all industries

    Functions called: None

    Objects in function:
        asdf

    Returns: Lmpath
    '''
    L1path = Kmpath[0,:]*((1-gamma[0])/gamma[0])*(((rpath+delta[0])/wpath)**epsilon[0])
    L2path = Kmpath[1,:]*((1-gamma[1])/gamma[1])*(((rpath+delta[1])/wpath)**epsilon[1])

    Lmpath = np.vstack((L1path, L2path))
    return Lmpath




def TP(params, rpath_init, wpath_init, Km_ss, Ym_ss, Gamma1, cm_tilde, A,
  gamma, epsilon, delta, nvec, graphs):
    '''
    Generates equilibrium time path for all endogenous objects from
    initial state (Gamma1) to the steady state using initial guesses
    rpath_init and wpath_init.

    Inputs:
        params     = length 11 tuple, (S, T, alpha, beta, sigma, r_ss,
                     w_ss, maxiter, mindist, xi, tpi_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        T          = integer > S, number of time periods until steady
                     state
        alpha      = scalar in (0,1), expenditure share on good 1
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        r_ss       = scalar > 0, steady-state interest rate
        w_ss       = scalar > 0, steady-state wage
        maxiter    = integer >= 1, Maximum number of iterations for TPI
        mindist    = scalar > 0, Convergence criterion for TPI
        xi         = scalar in (0,1], TPI path updating parameter
        tpi_tol    = scalar > 0, tolerance level for fsolve's in TPI
        rpath_init = [T+S-1,] vector, initial guess for the time path of
                     the interest rate
        wpath_init = [T+S-1,] vector, initial guess for the time path of
                     the wage
        Ym_ss      = [2,] vector, steady-state industry output levels
        Gamma1     = [S-1,] vector, initial period savings distribution
        cm_tilde   = [2,] vector, minimum consumption values for all
                     goods
        A       = [2,] vector, total factor productivity values for
                     all industries
        gamma     = [2,] vector, capital shares of income for all
                     industries
        epsilon     = [2,] vector, elasticities of substitution between
                     capital and labor for all industries
        delta     = [2,] vector, model period depreciation rates for
                     all industries
        nvec       = [S,] vector, exogenous labor supply n_{s}
        graphs     = boolean, =True if want graphs of TPI objects

    Functions called:
        get_pmpath
        get_ppath
        get_cbepath

    Objects in function:
        start_time   = scalar, current processor time in seconds (float)
        iter_tpi     = integer >= 0, current iteration of TPI
        dist_tpi     = scalar >= 0, distance measure for fixed point
        rpath_new    = [T+S-2,] vector, new time path of the interest
                       rate implied by household and firm optimization
        wpath_new    = [T+S-2,] vector, new time path of the wage
                       implied by household and firm optimization
        pm_params    = length 4 tuple, objects to be passed to
                       get_pmpath function:
                       (A, gamma, epsilon, delta)
        pmpath       = [2, T+S-1] matrix, time path of industry prices
        ppath        = [T+S-1] vector, time path of composite price

        r_params     = length 3 tuple, parameters passed in to get_r
        w_params     = length 2 tuple, parameters passed in to get_w
        cbe_params   = length 5 tuple. parameters passed in to
                       get_cbepath
        rpath        = [T+S-2,] vector, equilibrium time path of the
                       interest rate
        wpath        = [T+S-2,] vector, equilibrium time path of the
                       real wage
        cpath        = [S, T+S-2] matrix, equilibrium time path values
                       of individual consumption c_{s,t}
        bpath        = [S-1, T+S-2] matrix, equilibrium time path values
                       of individual savings b_{s+1,t+1}
        EulErrPath   = [S-1, T+S-2] matrix, equilibrium time path values
                       of Euler errors corresponding to individual
                       savings b_{s+1,t+1} (first column is zeros)
        Kpath_constr = [T+S-2,] boolean vector, =True if K_t<=0
        Kpath        = [T+S-2,] vector, equilibrium time path of the
                       aggregate capital stock
        Y_params     = length 2 tuple, parameters to be passed to get_Y
        Ypath        = [T+S-2,] vector, equilibrium time path of
                       aggregate output (GDP)
        Cpath        = [T+S-2,] vector, equilibrium time path of
                       aggregate consumption
        elapsed_time = scalar, time to compute TPI solution (seconds)

    Returns: bpath, cpath, wpath, rpath, Kpath, Ypath, Cpath,
             EulErrpath, elapsed_time
    '''
    start_time = time.clock()
    (S, T, alpha, beta, sigma, r_ss, w_ss,tp_tol) = params

    pm_params = (A, gamma, epsilon, delta)
    pmpath = get_pmpath(pm_params, rpath_init, wpath_init)
    ppath = get_ppath(alpha, pmpath)
    cbe_params = (S, T, alpha, beta, sigma, tp_tol)
    bpath, cpath, cmpath, eulerrpath = get_cbepath(cbe_params,
        Gamma1, rpath_init, wpath_init, pmpath, ppath, cm_tilde,
        nvec)
    Cmpath = get_Cmpath(cmpath[:, :T, :])
    Ym_params = (T, r_ss, w_ss)
    Ympath, Kmpath = get_YKmpath(Ym_params, rpath_init[:T],
        wpath_init[:T], Km_ss, Cmpath, A, gamma, epsilon, delta)


    Lmpath = get_Lmpath(Kmpath, rpath_init[:T], wpath_init[:T], gamma, epsilon, delta)

    
    rpath = rpath_init
    wpath = wpath_init
    MCKerrpath = bpath[:, :T].sum(axis=0) - Kmpath.sum(axis=0)
    MCLerrpath = nvec.sum() - Lmpath.sum(axis=0)
    elapsed_time = time.clock() - start_time

    if graphs == True:
        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Kmpath[0,:])
        plt.plot(tvec, Kmpath[1,:])
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate capital stock')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate capital $K_{t}$')
        # plt.savefig('Kt_Sec2')
        plt.show()

        # Plot time path of aggregate output (GDP)
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Ympath[0,:])
        plt.plot(tvec, Ympath[1,:])
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate output (GDP)')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate output $Y_{t}$')
        # plt.savefig('Yt_Sec2')
        plt.show()

        # Plot time path of aggregate consumption
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Cmpath[0,:])
        plt.plot(tvec, Cmpath[1,:])
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate consumption')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate consumption $C_{t}$')
        # plt.savefig('Ct_Sec2')
        plt.show()

        
        # Plot time path of real wage
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, wpath)
        plt.plot(tvec, np.ones(T)*w_ss)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for real wage')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real wage $w_{t}$')
        # plt.savefig('wt_Sec2')
        plt.show()

        # Plot time path of real interest rate
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, rpath)
        plt.plot(tvec, np.ones(T)*r_ss)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for real interest rate')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real interest rate $r_{t}$')
        # plt.savefig('rt_Sec2')
        plt.show()

        # Plot time path of the differences in the resource constraint
        tvec = np.linspace(1, T-2, T-2)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, ResmDiff[0,:T-2])
        plt.plot(tvec, ResmDiff[1,:T-2])
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for resource constraint')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'RC Difference')
        # plt.savefig('wt_Sec2')
        plt.show()

        # Plot time path of individual savings distribution
        tgrid = np.linspace(1, T, T)
        sgrid = np.linspace(2, S, S - 1)
        tmat, smat = np.meshgrid(tgrid, sgrid)
        cmap_bp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual savings $b_{s,t}$')
        strideval = max(int(1), int(round(S/10)))
        ax.plot_surface(tmat, smat, bpath[:, :T], rstride=strideval,
            cstride=strideval, cmap=cmap_bp)
        # plt.savefig('bpath')
        plt.show()

        # Plot time path of individual savings distribution
        tgrid = np.linspace(1, T-1, T-1)
        sgrid = np.linspace(1, S, S)
        tmat, smat = np.meshgrid(tgrid, sgrid)
        cmap_cp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual consumption $c_{s,t}$')
        strideval = max(int(1), int(round(S/10)))
        ax.plot_surface(tmat, smat, cpath[:, :T-1], rstride=strideval,
            cstride=strideval, cmap=cmap_cp)
        # plt.savefig('bpath')
        plt.show()

    return (rpath, wpath, pmpath, ppath, bpath, cpath, cmpath,
        eulerrpath, Cmpath, Ympath, Kmpath, Lmpath, MCKerrpath,
        MCLerrpath, elapsed_time)


def TP_fsolve(guesses, params, Km_ss, Ym_ss, Gamma1, cm_tilde, A,
  gamma, epsilon, delta, nvec, graphs):
    '''
    Generates equilibrium time path for all endogenous objects from
    initial state (Gamma1) to the steady state using initial guesses
    rpath_init and wpath_init.

    Inputs:
        params     = length 11 tuple, (S, T, alpha, beta, sigma, r_ss,
                     w_ss, maxiter, mindist, xi, tpi_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        T          = integer > S, number of time periods until steady
                     state
        alpha      = scalar in (0,1), expenditure share on good 1
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        r_ss       = scalar > 0, steady-state interest rate
        w_ss       = scalar > 0, steady-state wage
        maxiter    = integer >= 1, Maximum number of iterations for TPI
        mindist    = scalar > 0, Convergence criterion for TPI
        xi         = scalar in (0,1], TPI path updating parameter
        tpi_tol    = scalar > 0, tolerance level for fsolve's in TPI
        rpath_init = [T+S-1,] vector, initial guess for the time path of
                     the interest rate
        wpath_init = [T+S-1,] vector, initial guess for the time path of
                     the wage
        Ym_ss      = [2,] vector, steady-state industry output levels
        Gamma1     = [S-1,] vector, initial period savings distribution
        cm_tilde   = [2,] vector, minimum consumption values for all
                     goods
        A       = [2,] vector, total factor productivity values for
                     all industries
        gamma     = [2,] vector, capital shares of income for all
                     industries
        epsilon     = [2,] vector, elasticities of substitution between
                     capital and labor for all industries
        delta     = [2,] vector, model period depreciation rates for
                     all industries
        nvec       = [S,] vector, exogenous labor supply n_{s}
        graphs     = boolean, =True if want graphs of TPI objects

    Functions called:
        get_pmpath
        get_ppath
        get_cbepath

    Objects in function:
        start_time   = scalar, current processor time in seconds (float)
        iter_tpi     = integer >= 0, current iteration of TPI
        dist_tpi     = scalar >= 0, distance measure for fixed point
        rpath_new    = [T+S-2,] vector, new time path of the interest
                       rate implied by household and firm optimization
        wpath_new    = [T+S-2,] vector, new time path of the wage
                       implied by household and firm optimization
        pm_params    = length 4 tuple, objects to be passed to
                       get_pmpath function:
                       (A, gamma, epsilon, delta)
        pmpath       = [2, T+S-1] matrix, time path of industry prices
        ppath        = [T+S-1] vector, time path of composite price

        r_params     = length 3 tuple, parameters passed in to get_r
        w_params     = length 2 tuple, parameters passed in to get_w
        cbe_params   = length 5 tuple. parameters passed in to
                       get_cbepath
        rpath        = [T+S-2,] vector, equilibrium time path of the
                       interest rate
        wpath        = [T+S-2,] vector, equilibrium time path of the
                       real wage
        cpath        = [S, T+S-2] matrix, equilibrium time path values
                       of individual consumption c_{s,t}
        bpath        = [S-1, T+S-2] matrix, equilibrium time path values
                       of individual savings b_{s+1,t+1}
        EulErrPath   = [S-1, T+S-2] matrix, equilibrium time path values
                       of Euler errors corresponding to individual
                       savings b_{s+1,t+1} (first column is zeros)
        Kpath_constr = [T+S-2,] boolean vector, =True if K_t<=0
        Kpath        = [T+S-2,] vector, equilibrium time path of the
                       aggregate capital stock
        Y_params     = length 2 tuple, parameters to be passed to get_Y
        Ypath        = [T+S-2,] vector, equilibrium time path of
                       aggregate output (GDP)
        Cpath        = [T+S-2,] vector, equilibrium time path of
                       aggregate consumption
        elapsed_time = scalar, time to compute TPI solution (seconds)

    Returns: bpath, cpath, wpath, rpath, Kpath, Ypath, Cpath,
             EulErrpath, elapsed_time
    '''
    start_time = time.clock()
    (S, T, alpha, beta, sigma, r_ss, w_ss, tp_tol) = params

    rpath = np.zeros(T+S-1)
    wpath = np.zeros(T+S-1)
    rpath[:T] = guesses[0:T]
    wpath[:T] = guesses[T:]
    rpath[T:] = r_ss
    wpath[T:] = w_ss

    pm_params = (A, gamma, epsilon, delta)
    pmpath = get_pmpath(pm_params, rpath, wpath)
    ppath = get_ppath(alpha, pmpath)
    cbe_params = (S, T, alpha, beta, sigma, tp_tol)
    bpath, cpath, cmpath, eulerrpath = get_cbepath(cbe_params,
        Gamma1, rpath, wpath, pmpath, ppath, cm_tilde,
        nvec)
    Cmpath = get_Cmpath(cmpath[:, :T, :])
    Ym_params = (T, r_ss, w_ss)

    Ympath, Kmpath = get_YKmpath(Ym_params, rpath[:T], wpath[:T], Km_ss,
                      Cmpath, A, gamma, epsilon, delta)

    Lmpath = get_Lmpath(Kmpath, rpath[:T], wpath[:T], gamma, epsilon, delta)

    # Check market clearing in each period
    K_market_error = bpath[:, :T].sum(axis=0) - Kmpath[:, :].sum(axis=0)
    L_market_error = nvec.sum() - Lmpath[:, :].sum(axis=0)

    # Check and punish constraing violations
    mask1 = rpath[:T] <= 0
    mask2 = wpath[:T] <= 0
    mask3 = np.isnan(rpath[:T])
    mask4 = np.isnan(wpath[:T])
    K_market_error[mask1] += 1e14
    L_market_error[mask2] += 1e14
    K_market_error[mask3] += 1e14
    L_market_error[mask4] += 1e14


    print 'max capital market clearing distance: ', np.absolute(K_market_error).max()
    print 'max labor market clearing distance: ', np.absolute(L_market_error).max()
    print 'min capital market clearing distance: ', np.absolute(K_market_error).min()
    print 'min labor market clearing distance: ', np.absolute(L_market_error).min()

    errors = np.append(K_market_error, L_market_error)

    return errors


