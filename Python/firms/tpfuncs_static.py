'''
------------------------------------------------------------------------
This file contains the functions specific to solving for the time path of
the OG model with S-period lived agents, exogenous labor, and M industries 
and I goods.

These functions include:
    get_p_path
    get_p_tilde_path
    get_cbepath

    get_c_tilde_lf
    get_c_lf
    LfEulerSys
    paths_life

    TP
    TP_fsolve

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
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import sys
import firm_funcs_static as firm
reload(firm)

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def get_p_c_path(p_path, pi, I):
    '''
    Generates time path of consumption good prices from
    industry output prices and fixed coefficient matrix
    relating output goods to consumption goods.

    Inputs:
        p_path    = [M,T+S-2] matrix of params = length 4 tuple, (A, gamma, epsilon, delta)
        pi        = [I,M] matrix, element i,j gives the fraction of consumption
        T         = integer > S, number of time periods until steady
                     state

    Functions called: None

    Objects in function:
        p_c_path = [I, T+S-2] matrix, time path of consumption good prices

    Returns: p_c_path
    '''

    p_c_path = np.zeros((I,(p_path.shape)[1]))
    for t in range(0,(p_path.shape)[1]): 
        p_c_path[:,t] = np.dot(pi,p_path[:,t])


    return p_c_path


def get_cbepath(params, Gamma1, r_path, w_path, p_c_path, p_tilde_path,
  c_bar, I, n):
    '''
    Generates matrices for the time path of the distribution of
    individual savings, individual composite consumption, individual
    consumption of each type of good, and the Euler errors associated
    with the savings decisions.

    Inputs:
        params     = length 6 tuple, (S, T, alpha, beta, sigma, tp_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        T          = integer > S, number of time periods until steady
                     state
        I          = integer, number unique consumption goods
        alpha      = [I,T+S-1] matrix, expenditure shares on each good along time path
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        tp_tol    = scalar > 0, tolerance level for fsolve's in TPI
        Gamma1     = [S-1,] vector, initial period savings distribution
        r_path = [T+S-1,] vector, the time path of
                     the interest rate
        w_path = [T+S-1,] vector, the time path of
                     the wage
        p_c_path     = [I, T+S-1] matrix, time path of consumption goods prices
        p_tilde_path      = [T+S-1] vector, time path of composite price
        c_bar   = [I,T+S-1] matrix, minimum consumption values for all
                     goods along time path
        n          = [S,] vector, exogenous labor supply n_{s}

    Functions called:
        paths_life

    Objects in function:
        b_path      = [S-1, T+S-1] matrix, distribution of savings along time path
        c_tilde_path      = [S, T+S-1] matrix, distribution of composite consumption along time path
        c_path     = [S, T+S-1, I] array, distribution of consumption of each cons good along time path
        eulerr_path = [S-1, T+S-1] matrix, Euler equation errors along the time path
        pl_params  = length 4 tuple, parameters to pass into paths_life
                     (S, beta, sigma, TP_tol)
        u          = integer >= 2, represents number of periods
                     remaining in a lifetime, used to solve incomplete
                     lifetimes
        b_guess    = [u-1,] vector, initial guess for remaining lifetime
                     savings, taken from previous cohort's choices
        b_lf     = [u-1,] vector, optimal remaining lifetime savings
                     decisions
        c_tilde_lf     = [u,] vector, optimal remaining lifetime composite consumption
                     decisions
        c_lf    = [u,I] matrix, optimal remaining lifetime invididual consumption decisions 
        b_err_vec_lf = [u-1,] vector, Euler errors associated with
                      optimal remaining lifetime savings decisions
        DiagMaskb   = [u-1, u-1] boolean identity matrix
        DiagMaskc   = [u, u] boolean identity matrix

    Returns: b_path, c_tilde_path, c_path, eulerr_path
    '''
    S, T, alpha, beta, sigma, tp_tol = params
    b_path = np.append(Gamma1.reshape((S-1,1)), np.zeros((S-1, T+S-2)),
            axis=1)
    c_tilde_path = np.zeros((S, T+S-1))
    c_path = np.zeros((S, T+S-1, I))
    eulerr_path = np.zeros((S-1, T+S-1))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    c_tilde_path[S-1, 0], c_tilde_cstr = firm.get_c_tilde(c_bar[:,0],r_path[0],w_path[0],
                                p_c_path[:,0], p_tilde_path[0],n[S-1],Gamma1[S-2])
    c_path[S-1, 0, :], c_cstr = firm.get_c(alpha[:,0],c_bar[:,0],c_tilde_path[S-1,0],
                                     p_c_path[:,0],p_tilde_path[0])
    for u in xrange(2, S):
        # b_guess = b_ss[-u+1:]
        b_guess = np.diagonal(b_path[S-u:, :u-1])
        pl_params = (S, alpha[:,:u], beta, sigma, tp_tol)
        b_lf, c_tilde_lf, c_lf, b_err_vec_lf = paths_life(pl_params,
            S-u+1, Gamma1[S-u-1], c_bar[:,:u], n[-u:], r_path[:u],
            w_path[:u], p_c_path[:, :u], p_tilde_path[:u], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the c_tilde_path, b_path, and EulErrPath matrices
        DiagMaskb = np.eye(u-1, dtype=bool)
        DiagMaskc = np.eye(u, dtype=bool)
        b_path[S-u:, 1:u] = DiagMaskb * b_lf + b_path[S-u:, 1:u]
        c_tilde_path[S-u:, :u] = DiagMaskc * c_tilde_lf + c_tilde_path[S-u:, :u]
        DiagMaskc_tiled = np.tile(np.expand_dims(np.eye(u, dtype=bool),axis=2),(1,1,I))
        c_lf = np.tile(np.expand_dims(c_lf.transpose(),axis=0),((c_lf.shape)[1],1,1))
        c_path[S-u:, :u, :] = (DiagMaskc_tiled * c_lf +
                              c_path[S-u:, :u, :])
        eulerr_path[S-u:, 1:u] = (DiagMaskb * b_err_vec_lf +
                                eulerr_path[S-u:, 1:u])
    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the c_tilde_path, b_path, and EulErrPath matrices
    
    DiagMaskb = np.eye(S-1, dtype=bool)
    DiagMaskc = np.eye(S, dtype=bool)
    DiagMaskc_tiled = np.tile(np.expand_dims(np.eye(S, dtype=bool),axis=2),(1,1,I))

    for t in xrange(1, T+1): # Go from periods 1 to T
        # b_guess = b_ss
        b_guess = np.diagonal(b_path[:, t-1:t+S-2])
        pl_params = (S, alpha[:,t-1:t+S-1], beta, sigma, tp_tol)
        b_lf, c_lf, c_i_lf, b_err_vec_lf = paths_life(pl_params, 1,
            0, c_bar[:,t-1:t+S-1], n, r_path[t-1:t+S-1],
            w_path[t-1:t+S-1], p_c_path[:, t-1:t+S-1],
            p_tilde_path[t-1:t+S-1], b_guess)
        c_i_lf = np.tile(np.expand_dims(c_i_lf.transpose(),axis=0),((c_i_lf.shape)[1],1,1))
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the c_tilde_path, b_path, and EulErrPath matrices
        b_path[:, t:t+S-1] = DiagMaskb * b_lf + b_path[:, t:t+S-1]
        c_tilde_path[:, t-1:t+S-1] = DiagMaskc * c_lf + c_tilde_path[:, t-1:t+S-1]
        c_path[:, t-1:t+S-1, :] = (DiagMaskc_tiled * c_i_lf +
                                  c_path[:, t-1:t+S-1, :])
        eulerr_path[:, t:t+S-1] = (DiagMaskb * b_err_vec_lf +
                                 eulerr_path[:, t:t+S-1])
    return b_path, c_tilde_path, c_path, eulerr_path


def paths_life(params, beg_age, beg_wealth, c_bar, n, r_path,
               w_path, p_c_path, p_tilde_path, b_init):
    '''
    Solve for the remaining lifetime savings decisions of an individual
    who enters the model at age beg_age, with corresponding initial
    wealth beg_wealth.

    Inputs:
        params     = length 5 tuple, (S, alpha, beta, sigma, tp_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        alpha      = [I,S-beg_age+1], expenditure share on good for remaing lifetime
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        tp_tol    = scalar > 0, tolerance level for fsolve's in TPI
        beg_age    = integer in [1,S-1], beginning age of remaining life
        beg_wealth = scalar, beginning wealth at beginning age
        n       = [S-beg_age+1,] vector, remaining exogenous labor
                     supplies
        r_path      = [S-beg_age+1,] vector, remaining lifetime interest
                     rates
        w_path      = [S-beg_age+1,] vector, remaining lifetime wages
        p_c_path     = [I, S-beg_age+1] matrix, remaining lifetime
                     consumption good prices
        p_tilde_path      = [S-beg_age+1,] vector, remaining lifetime composite
                     goods prices
        b_init     = [S-beg_age,] vector, initial guess for remaining
                     lifetime savings

    Functions called:
        LfEulerSys
        firm.get_c_tilde
        firm.get_c
        firm.get_b_errors

    Objects in function:
        u            = integer in [2,S], remaining periods in life
        b_guess      = [u-1,] vector, initial guess for lifetime savings
                       decisions
        eullf_objs   = length 9 tuple, objects to be passed in to
                       LfEulerSys: (p, beta, sigma, beg_wealth, n,
                       r_path, w_path, p_path, p_tilde_path)
        b_path        = [u-1,] vector, optimal remaining lifetime savings
                       decisions
        c_tilde_path        = [u,] vector, optimal remaining lifetime
                       consumption decisions
        c_path       = [u,I] martrix, remaining lifetime consumption
                        decisions by consumption good
        c_constr     = [u,] boolean vector, =True if c_{u}<=0,
        b_err_params = length 2 tuple, parameters to pass into
                       firm.get_b_errors (beta, sigma)
        b_err_vec    = [u-1,] vector, Euler errors associated with
                       optimal savings decisions

    Returns: b_path, c_tilde_path, c_path, b_err_vec
    '''
    S, alpha, beta, sigma, tp_tol = params
    u = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(r_path) != u:
        #print len(r_path), S-beg_age+1
        sys.exit("Beginning age and length of r_path do not match.")
    if len(w_path) != u:
        sys.exit("Beginning age and length of w_path do not match.")
    if len(n) != u:
        sys.exit("Beginning age and length of n do not match.")
    b_guess = 1.01 * b_init
    eullf_objs = (u, beta, sigma, beg_wealth, c_bar, n, r_path,
                  w_path, p_c_path, p_tilde_path)
    b_path = opt.fsolve(LfEulerSys, b_guess, args=(eullf_objs),
                       xtol=tp_tol)
    c_tilde_path, c_tilde_cstr = firm.get_c_tilde(c_bar, r_path, w_path, p_c_path, p_tilde_path,
                    n, np.append(beg_wealth, b_path))
    c_path, c_cstr = firm.get_c(alpha[:,:u], c_bar, c_tilde_path, p_c_path, p_tilde_path)
    b_err_params = (beta, sigma)
    b_err_vec = firm.get_b_errors(b_err_params, r_path[1:], c_tilde_path,
                                   c_tilde_cstr, diff=True)
    return b_path, c_tilde_path, c_path, b_err_vec


def LfEulerSys(b, *objs):
    '''
    Generates vector of all Euler errors for a given b, which errors
    characterize all optimal lifetime decisions

    Inputs:
        b       = [u-1,] vector, remaining lifetime savings decisions
                     where p is the number of remaining periods
        objs       = length 9 tuple, (u, beta, sigma, beg_wealth, n,
                     r_path, w_path, p_path, p_tilde_path)
        u          = integer in [2,S], remaining periods in life
        beta       = scalar in [0,1), discount factor
        sigma      = scalar > 0, coefficient of relative risk aversion
        beg_wealth = scalar, wealth at the beginning of first age
        n       = [u,] vector, remaining exogenous labor supply
        r_path      = [u,] vector, interest rates over remaining life
        w_path      = [u,] vector, wages rates over remaining life
        p_c_path     = [I, u] matrix, remaining lifetime
                     consumption good prices
        p_tilde_path      = [u,] vector, remaining lifetime composite
                     goods prices
        p_tilde_path      = [u,] vector of composite good prices

    Functions called:
        firm.get_c_tilde
        firm.get_b_errors

    Objects in function:
        b2        = [u, ] vector, remaining savings including initial
                       savings
        c         = [u, ] vector, remaining lifetime consumption
                       levels implied by b2
        c_constr     = [u, ] boolean vector, =True if c_{s,t}<=0
        b_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_err_vec    = [u-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_err_vec
    '''
    (u, beta, sigma, beg_wealth, c_bar, n, r_path, w_path, p_c_path,
        p_tilde_path) = objs
    b2 = np.append(beg_wealth, b)
    c_tilde, c_tilde_cstr = firm.get_c_tilde(c_bar, r_path, w_path, p_c_path, p_tilde_path,
                               n, b2)
    b_err_params = (beta, sigma)
    b_err_vec = firm.get_b_errors(b_err_params, r_path[1:], c_tilde,
                                   c_tilde_cstr, diff=True)
    return b_err_vec


def get_K_over_X_path(r, w, A, gamma, epsilon, delta):
    '''
    Generates vector of K/X for each industry along the time path.

    Inputs:
        r      = [T,] vector, real interest rates
        w      = [T,] vector, real wage rates
        A       = [M,T] matrix, total factor productivity values for all
                   industries
        gamma = [M,T] matrix, capital shares of income for all
                 industries
        epsilon = [M,T] matrix, elasticities of substitution between
                 capital and labor for all industries
        delta = [M,T] matrix, model period depreciation rates for all
                 industries

    Functions called: None

    Objects in function:
        aa    = [M,T] matrix, gamma
        bb    = [M,T] matrix, 1 - gamma
        cc    = [M,T] matrix, (1 - gamma) / gamma
        dd    = [M,T] matrix, (r + delta) / w
        ee    = [M,T] matrix, 1 / epsilon
        ff    = [M,T] matrix, (epsilon - 1) / epsilon
        gg    = [M,T] matrix, epsilon - 1
        hh    = [M,T] matrix, epsilon / (1 - epsilon)
        ii    = [M,T] matrix, ((1 / A) * (((aa ** ee) + (bb ** ee) *
                (cc ** ff) * (dd ** gg)) ** hh))
        K_over_X_path = [M,T] matrix, capital demand of all industries

    Returns: K_over_X_path
    '''
    aa = gamma
    bb = 1 - gamma
    cc = (1 - gamma) / gamma
    dd = (r + delta) / w
    ee = 1 / epsilon
    ff = (epsilon - 1) / epsilon
    gg = epsilon - 1
    hh = epsilon / (1 - epsilon)

    K_over_X_path = ((1 / A) *
         (((aa ** ee) + (bb ** ee) * (cc ** ff) * (dd ** gg)) ** hh))

    return K_over_X_path


def get_X_path(params, r_path, w_path, C_path, A, gamma,
  epsilon, delta, xi, pi, I, M):

    '''
    Generate matrix (vectors) of time path of aggregate output X_{m,t}
    by industry given r_t, w_t, and C_{m,t}
    
    Inputs:
        K_ss             = [M,] vector, steady-state capital stock by industry 
        r_path             = [T,] vector, real interest rates
        w_path             = [T,] vector, real wage rates
        C_path            = [I,T] matrix, aggregate consumption of each good
                             along the time path     
        A                 = [M,T] matrix, total factor productivity values for all
                            industries
        gamma             = [M,T] matrix, capital shares of income for all
                            industries
        epsilon           = [M,T] matrix, elasticities of substitution between
                            capital and labor for all industries
        delta             = [M,T] matrix, model period depreciation rates for all
                            industries
        xi                = [M,M] matrix, element i,j gives the fraction of capital used by 
                            industry j that comes from the output of industry i
        pi                = [I,M] matrix, element i,j gives the fraction of consumption
        T                 = integer > S, number of time periods until steady
                           state
        I                 = integer, number unique consumption goods
        M                 = integer, number unique production industires 


    Functions called: 
        get_K_over_X_path

    Objects in function:
        Inv = [M,T] matrix, investment demand from each industry
        K_p1 = [M,] vector, demand for capital from each industry 
                in the next period
        X_kp1 = [M,] vector, demand for output for capital from
                 each industry in the next period
        X_c   = [M,T] matrix, demand for output from each industry due to 
                 consumption demand
        b_coeffs = coeffients in the linear problem: aX=b
        a_coeffs = coefficients in the linear problem: aX=b
        X_path  = [M,T] matrix, output from each industry
    

    Returns: rc_errors
    '''

    T, K_ss = params

    aa = get_K_over_X_path(r_path, w_path, A, gamma, epsilon, delta)
    bb = (1-delta)*aa

    X_path = np.zeros((M,T))
    K_p1 = K_ss
    for t in range(T-1, -1, -1): # Go from periods T to 1
        X_kp1 = np.dot(np.reshape(K_p1,(1,M)),xi)
        X_c = np.dot(np.reshape(C_path[:,t],(1,I)),pi)
        b_coeffs = (X_c + X_kp1).transpose()
        a_coeffs = np.eye(M) + (np.tile(bb[:,t],(M,1))*xi.transpose())
        X_path[:,t] = np.reshape(np.linalg.solve(a_coeffs, b_coeffs),(M,))
        #K = X_path[:,t]*aa[:,t]
        #Inv = K_p1 - (1-delta[:,t])*K
        #print 'Check RC: ', X_path[:,t]- X_c - np.dot(Inv,xi)
        #print(a_coeffs)
        #print(b_coeffs)
        K_p1 = X_path[:,t]*aa[:,t]


    return X_path
    

def TP(params, r_path_init, w_path_init, K_ss, X_ss, Gamma1, c_bar, A,
  gamma, epsilon, delta, xi, pi, I, M, S, n, graphs):

    '''
    Generates equilibrium time path for all endogenous objects from
    initial state (Gamma1) to the steady state using initial guesses
    r_path_init and w_path_init.

    Inputs:
        params     = length 11 tuple, (S, T, alpha, beta, sigma, r_ss,
                     w_ss, maxiter, mindist, xi, tp_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        T          = integer > S, number of time periods until steady
                     state
        I          = integer, number unique consumption goods
        M          = integer, number unique production industires
        alpha      = [I,T+S-1] matrix, expenditure share on each good
                      along the time path
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        r_ss       = scalar > 0, steady-state interest rate
        w_ss       = scalar > 0, steady-state wage
        tp_tol    = scalar > 0, tolerance level for fsolve's in TP solution
        r_path_init = [T+S-1,] vector, initial guess for the time path of
                     the interest rate
        w_path_init = [T+S-1,] vector, initial guess for the time path of
                     the wage
        X_ss      = [M,] vector, steady-state industry output levels
        Gamma1     = [S-1,] vector, initial period savings distribution
        c_bar   = [I,T+S-1] matrix, minimum consumption values for all
                     goods
        A       = [M,T+S-1] matrix, total factor productivity values for
                     all industries
        gamma     = [M,T+S-1] matrix, capital shares of income for all
                     industries
        epsilon     = [M,T+S-1] matrix, elasticities of substitution between
                     capital and labor for all industries
        delta     = [M,T+S-1] matrix, model period depreciation rates for
                     all industries
        xi      = [M,M] matrix, element i,j gives the fraction of capital used by 
               industry j that comes from the output of industry i
        pi      = [I,M] matrix, element i,j gives the fraction of consumption
        n       = [S,] vector, exogenous labor supply n_{s}
        graphs     = boolean, =True if want graphs of TPI objects

    Functions called:
        firm.get_p
        firm.get_p_tilde
        get_cbepath

    Objects in function:
        start_time   = scalar, current processor time in seconds (float)
        p_params    = length 4 tuple, objects to be passed to
                       get_p_path function:
                       (A, gamma, epsilon, delta)
        p_path       = [M, T+S-1] matrix, time path of industry output prices
        p_c_path       = [I, T+S-1] matrix, time path of consumption good prices
        p_tilde_path        = [T+S-1] vector, time path of composite price

        r_params     = length 3 tuple, parameters passed in to get_r
        w_params     = length 2 tuple, parameters passed in to get_w
        cbe_params   = length 5 tuple. parameters passed in to
                       get_cbepath
        r_path        = [T+S-2,] vector, equilibrium time path of the
                       interest rate
        w_path        = [T+S-2,] vector, equilibrium time path of the
                       real wage
        c_tilde_path        = [S, T+S-2] matrix, equilibrium time path values
                       of individual consumption c_{s,t}
        b_path        = [S-1, T+S-2] matrix, equilibrium time path values
                       of individual savings b_{s+1,t+1}
        EulErrPath   = [S-1, T+S-2] matrix, equilibrium time path values
                       of Euler errors corresponding to individual
                       savings b_{s+1,t+1} (first column is zeros)
        K_path_constr = [T+S-2,] boolean vector, =True if K_t<=0
        K_path        = [T+S-2,] vector, equilibrium time path of the
                       aggregate capital stock
        X_params     = length 2 tuple, parameters to be passed to get_X
        X_path        = [M,T+S-2] matrix, equilibrium time path of
                       industry output 
        C_path        = [I, T+S-2] matrix, equilibrium time path of
                       aggregate consumption
        elapsed_time = scalar, time to compute TPI solution (seconds)

    Returns: b_path, c_tilde_path, w_path, r_path, K_path, X_path, Cpath,
             EulErr_path, elapsed_time
    '''
    (S, T, alpha, beta, sigma, r_ss, w_ss, tp_tol) = params

    r_path = np.zeros(T+S-1)
    w_path = np.zeros(T+S-1)
    r_path[:T] = r_path_init[:T]
    w_path[:T] = w_path_init[:T]
    r_path[T:] = r_ss
    w_path[T:] = w_ss


    p_params = (A, gamma, epsilon, delta)
    p_path = firm.get_p(p_params, r_path, w_path)
    p_c_path = get_p_c_path(p_path, pi, I)
    p_tilde_path = firm.get_p_tilde(alpha, p_c_path)
    cbe_params = (S, T, alpha, beta, sigma, tp_tol)
    b_path, c_tilde_path, c_path, eulerr_path = get_cbepath(cbe_params,
        Gamma1, r_path, w_path, p_c_path, p_tilde_path, c_bar, I,
        n)
    C_path = firm.get_C(c_path[:, :T, :])

    X_params = (T, K_ss)
    X_path = get_X_path(X_params, r_path[:T], w_path[:T], C_path[:,:T], A[:,:T], gamma[:,:T],
                            epsilon[:,:T], delta[:,:T], xi, pi, I, M)

    K_path = firm.get_K(r_path[:T], w_path[:T], X_path, A[:,:T], gamma[:,:T], epsilon[:,:T], delta[:,:T])

    L_path = firm.get_L(r_path[:T], w_path[:T], K_path, gamma[:,:T], epsilon[:,:T], delta[:,:T])
    
    # Checking resource constraint along the path:
    Inv_path = np.zeros((M,T))
    X_inv_path = np.zeros((M,T))
    X_c_path = np.zeros((M,T))
    Inv_path[:,:T-1] = K_path[:,1:] - (1-delta[:,:T-1])*K_path[:,:T-1]
    Inv_path[:,T-1] = K_ss - (1-delta[:,T-1])*K_path[:,T-1]
    for t in range(0,T):
        X_inv_path[:,t] = np.dot(Inv_path[:,t],xi)
        X_c_path[:,t] = np.dot(np.reshape(C_path[:,t],(1,I)),pi)
    RCdiff_path = (X_path - X_c_path - X_inv_path) 
    
    MCKerr_path = b_path[:, :T].sum(axis=0) - K_path.sum(axis=0)
    MCLerr_path = n.sum() - L_path.sum(axis=0)
    

    if graphs == True:
        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        #plt.plot(tvec, K_path[0,:T])
        plt.plot(tvec, K_path[1,:T])
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
        plt.plot(tvec, X_path[0,:T])
        plt.plot(tvec, X_path[1,:T])
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate output (GDP)')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate output $X_{t}$')
        # plt.savefig('Yt_Sec2')
        plt.show()

        # Plot time path of aggregate consumption
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, C_path[0,:T])
        plt.plot(tvec, C_path[1,:T])
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
        plt.plot(tvec, w_path[:T])
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
        plt.plot(tvec, r_path[:T])
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
        tvec = np.linspace(1, T-1, T-1)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, ResmDiff[0,:T-1])
        plt.plot(tvec, ResmDiff[1,:T-1])
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for resource constraint')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'RC Difference')
        # plt.savefig('wt_Sec2')
        plt.show()

        # Plot time path of the differences in the market clearing conditions
        tvec = np.linspace(1, T, T)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, MCKerr_path[:T])
        plt.plot(tvec, MCLerr_path[:T])
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
        ax.plot_surface(tmat, smat, b_path[:, :T], rstride=strideval,
            cstride=strideval, cmap=cmap_bp)
        # plt.savefig('b_path')
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
        ax.plot_surface(tmat, smat, c_tilde_path[:, :T-1], rstride=strideval,
            cstride=strideval, cmap=cmap_cp)
        # plt.savefig('b_path')
        plt.show()

    return (r_path, w_path, p_path, p_tilde_path, b_path, c_tilde_path, c_path,
        eulerr_path, C_path, X_path, K_path, L_path, MCKerr_path,
        MCLerr_path, RCdiff_path)


def TP_fsolve(guesses, params, K_ss, X_ss, Gamma1, c_bar, A,
  gamma, epsilon, delta, xi, pi, I, M, S, n, graphs):

    '''
    Generates equilibrium time path for all endogenous objects from
    initial state (Gamma1) to the steady state using initial guesses
    r_path_init and w_path_init.

    Inputs:
        params     = length 11 tuple, (S, T, alpha, beta, sigma, r_ss,
                     w_ss, maxiter, mindist, xi, tp_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        T          = integer > S, number of time periods until steady
                     state
        I          = integer, number unique consumption goods
        M          = integer, number unique production industires
        alpha      = [I,T+S-1] matrix, expenditure share on each good
                      along the time path
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        r_ss       = scalar > 0, steady-state interest rate
        w_ss       = scalar > 0, steady-state wage
        tp_tol    = scalar > 0, tolerance level for fsolve's in TP solution
        r_path_init = [T+S-1,] vector, initial guess for the time path of
                     the interest rate
        w_path_init = [T+S-1,] vector, initial guess for the time path of
                     the wage
        X_ss      = [M,] vector, steady-state industry output levels
        Gamma1     = [S-1,] vector, initial period savings distribution
        c_bar   = [M,T+S-1] matrix, minimum consumption values for all
                     goods
        A       = [M,T+S-1] matrix, total factor productivity values for
                     all industries
        gamma     = [M,T+S-1] matrix, capital shares of income for all
                     industries
        epsilon     = [M,T+S-1] matrix, elasticities of substitution between
                     capital and labor for all industries
        delta     = [M,T+S-1] matrix, model period depreciation rates for
                     all industries
        xi      = [M,M] matrix, element i,j gives the fraction of capital used by 
               industry j that comes from the output of industry i
        pi      = [I,M] matrix, element i,j gives the fraction of consumption
        n       = [S,] vector, exogenous labor supply n_{s}
        graphs     = boolean, =True if want graphs of TPI objects

    Functions called:
        firm.get_p
        firm.get_p_tilde
        get_cbepath

    Objects in function:
        start_time   = scalar, current processor time in seconds (float)
        r_path_new    = [T+S-2,] vector, new time path of the interest
                       rate implied by household and firm optimization
        w_path_new    = [T+S-2,] vector, new time path of the wage
                       implied by household and firm optimization
        p_params    = length 4 tuple, objects to be passed to
                       get_p_path function:
                       (A, gamma, epsilon, delta)
        p_path       = [M, T+S-1] matrix, time path of industry output prices
        p_c_path       = [I, T+S-1] matrix, time path of consumption good prices
        p_tilde_path        = [T+S-1] vector, time path of composite price

        r_params     = length 3 tuple, parameters passed in to get_r
        w_params     = length 2 tuple, parameters passed in to get_w
        cbe_params   = length 5 tuple. parameters passed in to
                       get_cbepath
        r_path        = [T+S-2,] vector, equilibrium time path of the
                       interest rate
        w_path        = [T+S-2,] vector, equilibrium time path of the
                       real wage
        c_tilde_path        = [S, T+S-2] matrix, equilibrium time path values
                       of individual consumption c_{s,t}
        b_path        = [S-1, T+S-2] matrix, equilibrium time path values
                       of individual savings b_{s+1,t+1}
        EulErrPath   = [S-1, T+S-2] matrix, equilibrium time path values
                       of Euler errors corresponding to individual
                       savings b_{s+1,t+1} (first column is zeros)
        K_path_constr = [T+S-2,] boolean vector, =True if K_t<=0
        K_path        = [T+S-2,] vector, equilibrium time path of the
                       aggregate capital stock
        X_params     = length 2 tuple, parameters to be passed to get_X
        X_path        = [M,T+S-2] matrix, equilibrium time path of
                       industry output 
        C_path        = [I, T+S-2] matrix, equilibrium time path of
                       aggregate consumption

    Returns: b_path, c_tilde_path, w_path, r_path, K_path, X_path, Cpath,
             EulErr_path
    '''
    (S, T, alpha, beta, sigma, r_ss, w_ss, tp_tol) = params

    r_path = np.zeros(T+S-1)
    w_path = np.zeros(T+S-1)
    r_path[:T] = guesses[0:T]
    w_path[:T] = guesses[T:]
    r_path[T:] = r_ss
    w_path[T:] = w_ss


    p_params = (A, gamma, epsilon, delta)
    p_path = firm.get_p(p_params, r_path, w_path)
    p_c_path = get_p_c_path(p_path, pi, I)
    p_tilde_path = firm.get_p_tilde(alpha, p_c_path)
    cbe_params = (S, T, alpha, beta, sigma, tp_tol)
    b_path, c_tilde_path, c_path, eulerr_path = get_cbepath(cbe_params,
        Gamma1, r_path, w_path, p_c_path, p_tilde_path, c_bar, I,
        n)
    C_path = firm.get_C(c_path[:, :T, :])

    X_params = (T, K_ss)
    X_path = get_X_path(X_params, r_path[:T], w_path[:T], C_path[:,:T], A[:,:T], gamma[:,:T],
                            epsilon[:,:T], delta[:,:T], xi, pi, I, M)

    K_path = firm.get_K(r_path[:T], w_path[:T], X_path, A[:,:T], gamma[:,:T], epsilon[:,:T], delta[:,:T])

    L_path = firm.get_L(r_path[:T], w_path[:T], K_path, gamma[:,:T], epsilon[:,:T], delta[:,:T])

    # Check market clearing in each period
    K_market_error = b_path[:, :T].sum(axis=0) - K_path[:, :].sum(axis=0)
    L_market_error = n.sum() - L_path[:, :].sum(axis=0)

    # Checking resource constraint along the path:
    Inv_path = np.zeros((M,T))
    X_inv_path = np.zeros((M,T))
    X_c_path = np.zeros((M,T))
    Inv_path[:,:T-1] = K_path[:,1:] - (1-delta[:,:T-1])*K_path[:,:T-1]
    Inv_path[:,T-1] = K_ss - (1-delta[:,T-1])*K_path[:,T-1]
    for t in range(0,T):
        X_inv_path[:,t] = np.dot(Inv_path[:,t],xi)
        X_c_path[:,t] = np.dot(np.reshape(C_path[:,t],(1,I)),pi)
    RCdiff_path = (X_path - X_c_path - X_inv_path) 
    print 'the max RC diff is: ', np.absolute(RCdiff_path).max(axis=1)

    # Check and punish constraing violations
    mask1 = r_path[:T] <= 0
    mask2 = w_path[:T] <= 0
    mask3 = np.isnan(r_path[:T])
    mask4 = np.isnan(w_path[:T])
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


