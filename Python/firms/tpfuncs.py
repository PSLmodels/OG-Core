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
import ssfuncs as ssf
reload(ssf)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import sys
import firm_funcs as firm

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

# def get_p_path(params, r_path, w_path):
#     '''
#     Generates time path of industry prices p_m from r_path and w_path

#     Inputs:
#         params = length 4 tuple, (A, gamma, epsilon, delta)
#         A   = [M,T+S-2] matrix, total factor productivity for each
#                  industry
#         gamma = [M,T+S-2] matrix, capital share of income for each industry
#         epsilon = [M,T+S-2] matrix, elasticity of substitution between capital
#                  and labor for each industry
#         delta = [M,T+S-2] matrix, capital depreciation rate for each
#                  industry
#         r_path  = [T+S-2,] matrix, time path of interest rate
#         w      = [T+S-2,] matrix, time path of wage

#     Functions called: None

#     Objects in function:
#         p_path = [M, T+S-2] matrix, time path of industry prices

#     Returns: p_path
#     '''
#     A, gamma, epsilon, delta = params

#     p_path = (1 / A) * ((gamma * ((r_path + delta) **
#                    (1 - epsilon)) + (1 - gamma) * (w_path **
#                    (1 - epsilon))) ** (1 / (1 - epsilon)))

#     return p_path

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


# def get_p_tilde_path(alpha, p_c_path):
#     '''
#     Generates time path of composite price p from p_path

#     Inputs:
#         alpha = [I, T+S-2], expenditure share on each good along time path
#         p_c_path = [I, T+S-2] matrix, time path of industry prices

#     Functions called: None

#     Objects in function:
#         p_tilde_path = [T+S-2,] vector, time path of price of composite good

#     Returns: p_tilde_path
#     '''

#     p_tilde_path = ((p_c_path/alpha)**alpha).prod(axis=0)
#     return p_tilde_path


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
        p          = integer >= 2, represents number of periods
                     remaining in a lifetime, used to solve incomplete
                     lifetimes
        b_guess    = [p-1,] vector, initial guess for remaining lifetime
                     savings, taken from previous cohort's choices
        b_lf     = [p-1,] vector, optimal remaining lifetime savings
                     decisions
        c_lf     = [p,] vector, optimal remaining lifetime consumption
                     decisions
        c_i_lf    = [p,I] matrix, optimal remaining lifetime consumption decisions 
        b_err_vec_lf = [p-1,] vector, Euler errors associated with
                      optimal remaining lifetime savings decisions
        DiagMaskb   = [p-1, p-1] boolean identity matrix
        DiagMaskc   = [p, p] boolean identity matrix

    Returns: b_path, c_tilde_path, c_path, eulerr_path
    '''
    S, T, alpha, beta, sigma, tp_tol = params
    b_path = np.append(Gamma1.reshape((S-1,1)), np.zeros((S-1, T+S-2)),
            axis=1)
    c_tilde_path = np.zeros((S, T+S-1))
    c_path = np.zeros((S, T+S-1, 2))
    eulerr_path = np.zeros((S-1, T+S-1))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    #c_tilde_path[S-1, 0] = (1 / p_tilde_path[0]) * ((1 + r_path[0]) * Gamma1[S-2]
    #    + w_path[0] * n[S-1] - (p_c_path[:, 0] * c_bar[:,0]).sum(axis=0))
    c_tilde_path[S-1, 0], c_tilde_cstr = firm.get_c_tilde_lf(c_bar[:,0],r_path[0],w_path[0],
                                p_c_path[:,0], p_tilde_path[0],n[S-1],Gamma1[S-2])

    c_path[S-1, 0, :], ci_cstr = firm.get_c(alpha[:,0],c_bar[:,0],c_tilde_path[S-1,0],
                                     p_c_path[:,0],p_tilde_path[0])
    #c_path[S-1, 0, :] = alpha[:,0] * ((p_tilde_path[0] * c_tilde_path[S-1, 0]) /
    #                    p_c_path[:, 0]) + c_bar[:,0]
    for p in xrange(2, S):
        # b_guess = b_ss[-p+1:]
        b_guess = np.diagonal(b_path[S-p:, :p-1])
        pl_params = (S, alpha[:,:p], beta, sigma, tp_tol)
        b_lf, c_lf, c_i_lf, b_err_vec_lf = paths_life(pl_params,
            S-p+1, Gamma1[S-p-1], c_bar[:,:p], n[-p:], r_path[:p],
            w_path[:p], p_c_path[:, :p], p_tilde_path[:p], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the c_tilde_path, b_path, and EulErrPath matrices
        DiagMaskb = np.eye(p-1, dtype=bool)
        DiagMaskc = np.eye(p, dtype=bool)
        b_path[S-p:, 1:p] = DiagMaskb * b_lf + b_path[S-p:, 1:p]
        c_tilde_path[S-p:, :p] = DiagMaskc * c_lf + c_tilde_path[S-p:, :p]
        DiagMaskc_tiled = np.tile(np.expand_dims(np.eye(p, dtype=bool),axis=2),(1,1,I))
        c_i_lf = np.tile(np.expand_dims(c_i_lf.transpose(),axis=0),((c_i_lf.shape)[1],1,1))
        c_path[S-p:, :p, :] = (DiagMaskc_tiled * c_i_lf +
                              c_path[S-p:, :p, :])
        eulerr_path[S-p:, 1:p] = (DiagMaskb * b_err_vec_lf +
                                eulerr_path[S-p:, 1:p])
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
        p            = integer in [2,S], remaining periods in life
        b_guess      = [p-1,] vector, initial guess for lifetime savings
                       decisions
        eullf_objs   = length 9 tuple, objects to be passed in to
                       LfEulerSys: (p, beta, sigma, beg_wealth, n,
                       r_path, w_path, p_path, p_tilde_path)
        b_path        = [p-1,] vector, optimal remaining lifetime savings
                       decisions
        c_tilde_path        = [p,] vector, optimal remaining lifetime
                       consumption decisions
        c_path       = [p,I] martrix, remaining lifetime consumption
                        decisions by consumption good
        c_constr     = [p,] boolean vector, =True if c_{p}<=0,
        b_err_params = length 2 tuple, parameters to pass into
                       firm.get_b_errors (beta, sigma)
        b_err_vec    = [p-1,] vector, Euler errors associated with
                       optimal savings decisions

    Returns: b_path, c_tilde_path, c_path, b_err_vec
    '''
    S, alpha, beta, sigma, tp_tol = params
    p = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(r_path) != p:
        #print len(r_path), S-beg_age+1
        sys.exit("Beginning age and length of r_path do not match.")
    if len(w_path) != p:
        sys.exit("Beginning age and length of w_path do not match.")
    if len(n) != p:
        sys.exit("Beginning age and length of n do not match.")
    b_guess = 1.01 * b_init
    eullf_objs = (p, beta, sigma, beg_wealth, c_bar, n, r_path,
                  w_path, p_c_path, p_tilde_path)
    b_path = opt.fsolve(LfEulerSys, b_guess, args=(eullf_objs),
                       xtol=tp_tol)
    c_tilde_path, c_tilde_cstr = firm.get_c_tilde(c_bar, r_path, w_path, p_c_path, p_tilde_path,
                    n, np.append(beg_wealth, b_path))
    c_path, ci_cstr = firm.get_c(alpha[:,:p], c_bar, c_tilde_path, p_c_path, p_tilde_path)
    b_err_params = (beta, sigma)
    b_err_vec = ssf.get_b_errors(b_err_params, r_path[1:], c_tilde_path,
                                   c_tilde_cstr, diff=True)
    return b_path, c_tilde_path, c_path, b_err_vec


def LfEulerSys(b, *objs):
    '''
    Generates vector of all Euler errors for a given b, which errors
    characterize all optimal lifetime decisions

    Inputs:
        b       = [p-1,] vector, remaining lifetime savings decisions
                     where p is the number of remaining periods
        objs       = length 9 tuple, (p, beta, sigma, beg_wealth, n,
                     r_path, w_path, p_path, p_tilde_path)
        p          = integer in [2,S], remaining periods in life
        beta       = scalar in [0,1), discount factor
        sigma      = scalar > 0, coefficient of relative risk aversion
        beg_wealth = scalar, wealth at the beginning of first age
        n       = [p,] vector, remaining exogenous labor supply
        r_path      = [p,] vector, interest rates over remaining life
        w_path      = [p,] vector, wages rates over remaining life
        p_c_path     = [I, p] matrix, remaining lifetime
                     consumption good prices
        p_tilde_path      = [p,] vector, remaining lifetime composite
                     goods prices
        p_tilde_path      = 

    Functions called:
        firm.get_c_tilde
        firm.get_b_errors

    Objects in function:
        b2        = [p, ] vector, remaining savings including initial
                       savings
        c         = [p, ] vector, remaining lifetime consumption
                       levels implied by b2
        c_constr     = [p, ] boolean vector, =True if c_{s,t}<=0
        b_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_err_vec    = [p-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_err_vec
    '''
    (p, beta, sigma, beg_wealth, c_bar, n, r_path, w_path, p_c_path,
        p_tilde_path) = objs
    b2 = np.append(beg_wealth, b)
    c_tilde, c_tilde_cstr = firm.get_c_tilde(c_bar, r_path, w_path, p_c_path, p_tilde_path,
                               n, b2)
    b_err_params = (beta, sigma)
    b_err_vec = firm.get_b_errors(b_err_params, r_path[1:], c_tilde,
                                   c_tilde_cstr, diff=True)
    return b_err_vec


# def get_c_tilde_lf(c_bar, r_path, w_path, p_c_path, p_tilde_path, n, b):
#     '''
#     Generates vector of remaining lifetime consumptions from individual
#     savings, and the time path of interest rates and the real wages

#     Inputs:
#         p      = integer in [2,80], number of periods remaining in
#                  individual life
#         r_path  = [p,] vector, remaining interest rates
#         w_path  = [p,] vector, remaining wages
#         p_c_path = [I, p] matrix, remaining industry prices
#         p_tilde_path  = [p,] vector, remaining composite prices
#         n   = [p,] vector, remaining exogenous labor supply
#         b   = [p,] vector, remaining savings including initial
#                  savings

#     Functions called: None

#     Objects in function:
#         c_cstr = [p,] boolean vector, =True if element c_s <= 0
#         b_s    = [p,] vector, b
#         b_sp1  = [p,] vector, last p-1 elements of b and 0 in last
#                  element
#         c   = [p,] vector, remaining consumption by age c_s

#     Returns: c, c_constr
#     '''

#     if np.isscalar(b): # check if scalar - if so, then in last period of life and savings = 0
#         c_tilde = (1 / p_tilde_path) *((1 + r_path) * b + w_path * n -
#            (p_c_path * c_bar).sum(axis=0))
#         c_tilde_cstr = c_tilde <= 0
#     else:
#         b_s = b
#         b_sp1 = np.append(b[1:], [0])
#         c_tilde = (1 / p_tilde_path) *((1 + r_path) * b_s + w_path * n -
#            (p_c_path * c_bar).sum(axis=0) - b_sp1)
#         c_tilde_cstr = c_tilde <= 0
#     return c_tilde, c_tilde_cstr


# def get_c_lf(alpha, c_bar, c_tilde_path, p_c_path, p_tilde_path):
#     '''
#     Generates matrix of remaining lifetime consumptions of individual
#     goods

#     Inputs:
#         p      = integer in [2,80], number of periods remaining in
#                  individual life
#         r_path  = [p,] vector, remaining interest rates
#         w_path  = [p,] vector, remaining wages
#         p_c_path = [I, p] matrix, remaining industry prices
#         p_tilde_path  = [p,] vector, remaining composite prices
#         n   = [p,] vector, remaining exogenous labor supply
#         b   = [p,] vector, remaining savings including initial
#                  savings

#     Functions called: None

#     Objects in function:
#         c_cstr = [p,] boolean vector, =True if element c_s <= 0
#         b_s    = [p,] vector, b
#         b_sp1  = [p,] vector, last p-1 elements of b and 0 in last
#                  element
#         c   = [p,] vector, remaining consumption by age c_s

#     Returns: c, c_constr
#     '''

#     c = alpha * ((p_tilde_path * c_tilde_path) / p_c_path) + c_bar

#     c_cstr = c <= 0
#     return c, c_cstr


# def get_C_path(c_path):
#     '''
#     Generates vector of aggregate consumption C_m of good m

#     Inputs:
#         c_path = [S, S+T-1, I] array, time path of distribution of
#                  individual consumption of each good c_{m,s,t}

#     Functions called: None

#     Objects in function:
#         C_path = [I,S+T-1] matrix, aggregate consumption of all goods

#     Returns: C_path
#     '''

#     C_path = (c_path.sum(axis=0)).transpose()

#     return C_path



def solve_X_path(X_path_init_guess, params, r_path, w_path, C_path, A, gamma,
  epsilon, delta, xi, pi, I, M):

    '''
    Generate matrix (vectors) of time path of aggregate output X_{m,t}
    by industry given r_t, w_t, and C_{m,t}
    
    Inputs:
        X_path_init_guess = [M*T,] vector, initial guess of X_path
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
        firm.get_K

    Objects in function:
        Inv = [M,T] matrix, investment demand from each industry
        X_inv = [M,T] matrix, demand for output from each industry due to 
                 investment demand
        X_c   = [M,T] matrix, demand for output from each industry due to 
                 consumption demand
        K_path  = [M,T] matrix, capital demand of all industries
        X_path  = [M,T] matrix, output from each industry
        rc_errors = [M*T,] vector, errors in resource constraint
                    for each production industry along time path

    Returns: rc_errors
    '''

    T, K_ss = params

    #unpack guesses, which needed to be a vector
    X_path = np.zeros((M,T))
    for m in range(0,M):
        X_path[m,:] = X_path_init_guess[(m*T):((m+1)*T)]
    
    K_path = firm.get_K_path(r_path, w_path, X_path, A, gamma, epsilon, delta)
    Inv = np.zeros((M,T))
    Inv[:,:-1] = K_path[:,1:] - (1-delta[:,:-1])*K_path[:,:-1]
    Inv[:,T-1] = K_ss - (1-delta[:,T-1])*K_path[:,T-1]

    X_inv = np.zeros((M,T))
    X_c = np.zeros((M,T))
    for t in range(0,T):
        X_inv[:,t] = np.dot(Inv[:,t],xi)
        X_c[:,t] = np.dot(np.reshape(C_path[:,t],(1,I)),pi)

    rc_errors = np.reshape(X_c  + X_inv - X_path,(M*T,))

    return rc_errors
    

# def get_K_path(r_path, w_path, X_path, A, gamma, epsilon, delta):
#     '''
#     Generates vector of capital demand from production industry m 
#     along the time path for a given X_path, r_path, w_path.

#     Inputs:
#         r_path      = [T,] vector, real interest rates
#         w_path      = [T,] vector, real wage rates
#         X_path  = [M,T] matrix, output from each industry
#         A       = [M,T] matrix, total factor productivity values for all
#                    industries
#         gamma = [M,T] matrix, capital shares of income for all
#                  industries
#         epsilon = [M,T] matrix, elasticities of substitution between
#                  capital and labor for all industries
#         delta = [M,T] matrix, model period depreciation rates for all
#                  industries

#     Functions called: None

#     Objects in function:
#         aa    = [M,T] matrix, gamma
#         bb    = [M,T] matrix, 1 - gamma
#         cc    = [M,T] matrix, (1 - gamma) / gamma
#         dd    = [M,T] matrix, (r + delta) / w
#         ee    = [M,T] matrix, 1 / epsilon
#         ff    = [M,T] matrix, (epsilon - 1) / epsilon
#         gg    = [M,T] matrix, epsilon - 1
#         hh    = [M,T] matrix, epsilon / (1 - epsilon)
#         ii    = [M,T] matrix, ((1 / A) * (((aa ** ee) + (bb ** ee) *
#                 (cc ** ff) * (dd ** gg)) ** hh))
#         K_path = [M,T] matrix, capital demand of all industries

#     Returns: K_path
#     '''
#     aa = gamma
#     bb = 1 - gamma
#     cc = (1 - gamma) / gamma
#     dd = (r_path + delta) / w_path
#     ee = 1 / epsilon
#     ff = (epsilon - 1) / epsilon
#     gg = epsilon - 1
#     hh = epsilon / (1 - epsilon)

#     K_path = ((X_path / A) *
#          (((aa ** ee) + (bb ** ee) * (cc ** ff) * (dd ** gg)) ** hh))

#     return K_path



# def get_L_path(K_path, r_path, w_path, gamma, epsilon, delta):
#     '''
#     Generates vector of labor demand L_m for good m given X_m, p_m and w

#     Inputs:
#         K_path = [M, T] matrix, time path of aggregate output by
#                  industry
#         r_path  = [T, ] matrix, time path of real interest rate
#         w_path  = [T, ] matrix, time path of real wage
#         gamma = [M,T] matrix, capital shares of income for all
#                  industries
#         epsilon = [M,T] matrix, elasticities of substitution between
#                  capital and labor for all industries
#         delta = [M,T] matrix, rate of phyical depreciation for all industries

#     Functions called: None

#     Objects in function:
#         L_path = [M,T] matrix, labor demand from each industry

#     Returns: L_path
#     '''
#     L_path = K_path*((1-gamma)/gamma)*(((r_path+delta)/w_path)**epsilon)

#     return L_path




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
    p_path = firm.get_p_path(p_params, r_path, w_path)
    p_c_path = get_p_c_path(p_path, pi, I)
    p_tilde_path = firm.get_p_tilde_path(alpha, p_c_path)
    cbe_params = (S, T, alpha, beta, sigma, tp_tol)
    b_path, c_tilde_path, c_path, eulerr_path = get_cbepath(cbe_params,
        Gamma1, r_path, w_path, p_c_path, p_tilde_path, c_bar, I,
        n)
    C_path = firm.get_C_path(c_path[:, :T, :])

    X_path_params = (T, K_ss)
    X_path_init = np.zeros((M,T))
    for t in range(0,T):
        X_path_init[:,t] = np.reshape((np.dot(np.reshape(C_path[:,t],(1,I)),pi))/I,(M,))

    X_path_init_guess = np.reshape(X_path_init,(T*M,)) #need a vector going into fsolve

    X_path_sol = opt.fsolve(solve_X_path, X_path_init_guess, args=(X_path_params, r_path[:T], w_path[:T], 
             C_path, A[:,:T], gamma[:,:T], epsilon[:,:T], delta[:,:T], xi, pi, I, 
             M), xtol=tp_tol, col_deriv=1)
    X_path = np.zeros((M,T))
    for m in range(0,M): # unpack vector of X_path solved by fsolve
        X_path[m,:] = X_path_sol[(m*T):((m+1)*T)]

    K_path = firm.get_K_path(r_path[:T], w_path[:T], X_path, A[:,:T], gamma[:,:T], epsilon[:,:T], delta[:,:T])

    L_path = firm.get_L_path(K_path, r_path[:T], w_path[:T], gamma[:,:T], epsilon[:,:T], delta[:,:T])
    
    # Checking resource constraint along the path:
    print 'L_path shape: ', L_path.shape
    Inv_path = np.zeros((M,T))
    X_inv_path = np.zeros((M,T))
    X_c_path = np.zeros((M,T))
    Inv_path[:,:-1] = K_path[:,1:] - (1-delta[:,:T-1])*K_path[:,:-1]
    Inv_path[:,T-1] = K_ss - (1-delta[:,T-1])*K_path[:,T-1]
    for t in range(0,T):
        X_inv_path[:,t] = np.dot(Inv_path[:,t],xi)
        X_c_path[:,t] = np.dot(np.reshape(C_path[:,t],(1,I)),pi)
    RCdiff_path = (X_path[:, :T-1] - X_c_path[:, :T-1] - X_inv_path[:, 1:T]) 

    
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
        MCLerr_path)


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
    p_path = firm.get_p_path(p_params, r_path, w_path)
    p_c_path = get_p_c_path(p_path, pi, I)
    p_tilde_path = firm.get_p_tilde_path(alpha, p_c_path)
    cbe_params = (S, T, alpha, beta, sigma, tp_tol)
    b_path, c_tilde_path, c_path, eulerr_path = get_cbepath(cbe_params,
        Gamma1, r_path, w_path, p_c_path, p_tilde_path, c_bar, I,
        n)
    C_path = firm.get_C_path(c_path[:, :T, :])
    X_path_params = (T, K_ss)
    X_path_init = np.zeros((M,T))
    for t in range(0,T):
        X_path_init[:,t] = np.reshape((np.dot(np.reshape(C_path[:,t],(1,I)),pi))/I,(M,))

    X_path_init_guess = np.reshape(X_path_init,(T*M,)) #need a vector going into fsolve

    X_path_sol = opt.fsolve(solve_X_path, X_path_init_guess, args=(X_path_params, r_path[:T], w_path[:T], 
             C_path, A[:,:T], gamma[:,:T], epsilon[:,:T], delta[:,:T], xi, pi, I, 
             M), xtol=tp_tol, col_deriv=1)
    X_path = np.zeros((M,T))
    for m in range(0,M): # unpack vector of X_path solved by fsolve
        X_path[m,:] = X_path_sol[(m*T):((m+1)*T)]

    K_path = firm.get_K_path(r_path[:T], w_path[:T], X_path, A[:,:T], gamma[:,:T], epsilon[:,:T], delta[:,:T])

    L_path = firm.get_L_path(K_path, r_path[:T], w_path[:T], gamma[:,:T], epsilon[:,:T], delta[:,:T])

    #print 'K_path: ', K_path
    #print 'L_path: ', L_path

    # Check market clearing in each period
    K_market_error = b_path[:, :T].sum(axis=0) - K_path[:, :].sum(axis=0)
    L_market_error = n.sum() - L_path[:, :].sum(axis=0)

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


