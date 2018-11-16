from __future__ import print_function
'''
------------------------------------------------------------------------
Firm functions for firms in the steady state and along the transition
path

This Python module imports the following module(s): None

This Python module defines the following functions:
    get_Y()
    get_r()
    get_w()
    get_KLrat_from_r()
    get_w_from_r()
    get_K()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_Y(K, L, p):
    '''
    --------------------------------------------------------------------
    Generates aggregate output (GDP) from aggregate capital stock,
    aggregate labor, and CES production function parameters
    --------------------------------------------------------------------
    INPUTS:
    K = scalar or D dimensional array, aggregate capital stock
    L = scalar or D dimensional array, aggregate labor
    p = model parameters object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon = scalar >= 0, elasticity of substitution between capital
              and labor
    Z       = scalar > 0, total factor productivity
    gamma   = scalar in [0, 1], CES production function share parameter
    Y       = scalar or D dimensional array, aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    if p.epsilon == 1:
        # Unit elasticity, Cobb-Douglas
        Y = p.Z * (K ** p.gamma) * (L ** (1 - p.gamma))
    else:
        # General case
        Y = (p.Z * (((p.gamma ** (1 / p.epsilon)) *
                     (K ** ((p.epsilon - 1) / p.epsilon))) +
                    (((1 - p.gamma) ** (1 / p.epsilon)) *
                     (L ** ((p.epsilon - 1) / p.epsilon)))) **
             (p.epsilon / (p.epsilon - 1)))

    return Y


def get_r(Y, K, p):
    '''
    --------------------------------------------------------------------
    This function computes the interest rate as a function of Y, K, and
    parameters using the firm's first order condition for capital demand
    --------------------------------------------------------------------
    INPUTS:
    Y = scalar or (T+S,) vector, aggregate output
    K = scalar or (T+S,) vector, aggregate capital
    p = model parameters object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    tau_b     = scalar in [0, 1], corporate income tax rate
    Z         = scalar > 0, total factor productivity
    epsilon   = scalar > 0, elasticity of substitution between capital
                and labor
    gamma     = scalar in [0, 1], share parameter in CES production
                function (capital share of income in Cobb-Douglas case)
    delta     = scalar in [0, 1], per-period capital depreciation rate
    delta_tau = scalar >= 0, percent of capital depreciation rate
                refunded at the corporate income tax rate
    r         = scalar or (T+S,) vector, interest rate on (rental rate
                of) capital

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: r
    --------------------------------------------------------------------
    '''
    r = ((1 - p.tau_b) * (p.Z ** ((p.epsilon - 1) / p.epsilon)) *
         (((p.gamma * Y) / K) ** (1 / p.epsilon)) -
         p.delta + p.tau_b * p.delta_tau)

    return r


def get_w(Y, L, p):
    '''
    --------------------------------------------------------------------
    This function computes the wage as a function of Y, L, and
    parameters using the firm's first order condition for labor demand
    --------------------------------------------------------------------
    INPUTS:
    Y = scalar or (T+S,) vector, aggregate output
    L = scalar or (T+S,) vector, aggregate labor
    p = model parameters object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Z       = scalar > 0, total factor productivity
    epsilon = scalar > 0, elasticity of substitution between capital and
              labor
    gamma   = scalar in [0, 1], share parameter in CES production
              function (capital share of income in Cobb-Douglas case)
    w       = scalar or (T+S,) vector, wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: w
    --------------------------------------------------------------------
    '''
    w = ((p.Z ** ((p.epsilon - 1) / p.epsilon)) *
         ((((1 - p.gamma) * Y) / L) ** (1 / p.epsilon)))

    return w


def get_KLrat_from_r(r, p):
    '''
    --------------------------------------------------------------------
    This function solves for the capital-labor ratio given the interest
    rate r and parameters
    --------------------------------------------------------------------
    INPUTS:
    r = scalar or D dimensional array, aggregate labor
    p = model parameters object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    tau_b     = scalar in [0, 1], corporate income tax rate
    Z         = scalar > 0, total factor productivity
    epsilon   = scalar > 0, elasticity of substitution between capital
                and labor
    gamma     = scalar in [0, 1], share parameter in CES production
                function (capital share of income in Cobb-Douglas case)
    delta     = scalar in [0, 1], per-period capital depreciation rate
    delta_tau = scalar >= 0, percent of capital depreciation rate
                refunded at the corporate income tax rate
    bracket   = scalar or D dimensional array, value in bracket in
                equation for capital-labor ratio

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: KLratio
    --------------------------------------------------------------------
    '''
    if p.epsilon == 1:
        # Cobb-Douglas case
        bracket = (((1 - p.tau_b) * p.gamma * p.Z) /
                   (r + p.delta - p.tau_b * p.delta_tau))
        KLratio = bracket ** (1 / (1 - p.gamma))
    else:
        # General CES case
        bracket = ((r + p.delta - (p.delta_tau * p.tau_b)) /
                   ((1 - p.tau_b) * p.Z * (p.gamma ** (1 / p.epsilon))))
        KLratio = \
            ((((1 - p.gamma) ** (1 / p.epsilon)) /
              ((bracket ** (p.epsilon - 1)) -
               (p.gamma ** (1 / p.epsilon)))) ** (p.epsilon /
                                                  (p.epsilon - 1)))
    return KLratio


def get_w_from_r(r, p):
    '''
    --------------------------------------------------------------------
    Solve for steady-state wage w or time path of wages w_t given
    interest rate
    --------------------------------------------------------------------
    INPUTS:
    r = scalar or D dimensional array, aggregate labor
    p = model parameters object

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_KLrat_from_r()

    OBJECTS CREATED WITHIN FUNCTION:
    KLratio = scalar or D dimensional array, capital-labor ratio
              computed from interest rate and parameters
    epsilon = scalar > 0, elasticity of substitution between capital and
              labor
    eps     = scalar > 0, epsilon
    gamma   = scalar in [0, 1], share parameter in CES production
              function (capital share of income in Cobb-Douglas case)
    Z       = scalar > 0, total factor productivity
    w       = scalar or D dimensional array, wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: w
    --------------------------------------------------------------------
    '''
    KLratio = get_KLrat_from_r(r, p)
    eps = p.epsilon
    if eps == 1:
        # Cobb-Douglas case
        w = (1 - p.gamma) * p.Z * (KLratio ** p.gamma)
    else:
        # General CES case
        w = (((1 - p.gamma) ** (1 / eps)) * p.Z *
             (((p.gamma ** (1 / eps)) * (KLratio ** ((eps - 1) / eps)) +
               ((1 - p.gamma) ** (1 / eps))) ** (1 / (eps - 1))))
    return w


def get_K(L, r, p):
    '''
    --------------------------------------------------------------------
    Generates vector of aggregate capital. Use with small open economy
    option.
    --------------------------------------------------------------------
    Inputs:
        L      = [T+S,] vector, aggregate labor
        r      = [T+S,] vector, exogenous rental rate of the firm
        params = length 3 tuple, (alpha, delta, z)
        alpha  = scalar, capital's share of output
        delta  = scalar, rate of depreciation of capital
        Z      = scalar, total factor productivity

    Functions called: None

    Objects in function:
        K = [T+S,] vector, aggregate capital

    Returns: r
    --------------------------------------------------------------------
    '''
    if p.epsilon == 1:
        K = (((1 - p.tau_b) * p.gamma * p.Z /
              (r + p.delta - (p.tau_b * p.delta_tau))) **
             (1 / (1 - p.gamma)) * L)
    elif p.epsilon == 0:
        K = (1 - ((1 - p.gamma) * L)) / p.gamma
    else:
        K = (L * ((1 - p.gamma) ** (1 / (p.epsilon - 1))) *
             (((
                (((r + p.delta - (p.tau_b * p.delta_tau)) /
                  (1 - p.tau_b)) ** (p.epsilon - 1)) *
                (p.gamma ** ((1 - p.epsilon) / p.epsilon)) *
                (p.Z ** (1 - p.epsilon))) - p.gamma **
               (1 / p.epsilon))) ** (p.epsilon / (1 - p.epsilon)))

    print('USING firm.getK()')

    return K
