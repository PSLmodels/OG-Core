'''
------------------------------------------------------------------------
Last updated 8/15/2017

Firm functions for firms in the steady state and along the transition
path, including functions for small open economy firms which take r as given.

------------------------------------------------------------------------
'''

# Packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_r(Y, K, params):
    '''
    Generates vector of interest rates.

    Inputs:
        Y      = [T+S,] vector, aggregate output
        K      = [T+S,] vector, aggregate capital
        params = length 2 tuple, (alpha, delta)
        alpha  = scalar, capital's share of output
        delta  = scalar, rate of depreciation of capital

    Functions called: None

    Objects in function:
        r = [T+S,] vector, rental rate

    Returns: r
    '''
    Z, gamma, epsilon, delta, tau_b, delta_tau = params
    if epsilon == 0:
        r = (1-tau_b)*gamma - delta
    else:
        r = ((1-tau_b)*((Z**((epsilon-1)/epsilon)) *
             (((gamma*Y)/K)**(1/epsilon)))
             - delta + tau_b*delta_tau)

    return r


def get_w(Y, L, params):
    '''
    Generates vector of aggregate output.

    Inputs:
        Y      = [T+S,] vector, aggregate output
        L      = [T+S,] vector, aggregate labor
        params = length 1 tuple, (alpha)
        alpha  = scalar, capital's share of output

    Functions called: None

    Objects in function:
        w = [T+S,] vector, rental rate

    Returns: w
    '''
    # alpha = params
    # w = (1 - alpha) * Y / L
    Z, gamma, epsilon = params
    if epsilon == 0:
        w = 1-gamma
    else:
        w = ((Z**((epsilon-1)/epsilon))*((((1-gamma)*Y)/L)**(1/epsilon)))

    return w


def get_Y(K, L, params):
    '''
    Generates vector of aggregate output.

    Inputs:
        K      = [T+S,] vector, aggregate capital
        L      = [T+S,] vector, aggregate labor
        params = length 2 tuple, (alpha, Z)
        alpha  = scalar, capital's share of output
        Z      = scalar, total factor productivity

    Functions called: None

    Objects in function:
        Y = [T+S,] vector, aggregate output

    Returns: Y
    '''
    # alpha, Z = params
    # Y = Z * (K ** alpha) * (L ** (1 - alpha))
    Z, gamma, epsilon = params
    if epsilon == 1:
        Y = Z*(K**gamma)*(L**(1-gamma))
    elif epsilon == 0:
        Y = Z*(gamma*K + (1-gamma)*L)
    else:
        Y = (Z * (((gamma**(1/epsilon))*(K**((epsilon-1)/epsilon))) +
             (((1-gamma)**(1/epsilon)) *
             (L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))

    return Y


def get_K(L, r, params):
    '''
    Generates vector of aggregate capital. Use with small open economy option.

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
    '''

    Z, gamma, epsilon, delta, tau_b, delta_tau = params
    if epsilon == 1:
        K = ((1-tau_b)*gamma*Z/(r+delta-(tau_b*delta_tau)))**(1/(1-gamma)) * L
    elif epsilon == 0:
        K = (1-((1-gamma)*L))/gamma
    else:
        K = (L *
             ((1-gamma)**(1/(epsilon-1))) *
             (((
                (((r+delta-(tau_b*delta_tau))/(1 - tau_b))**(epsilon-1)) *
                (gamma**((1-epsilon)/epsilon)) * (Z**(1-epsilon))
                ) -
              gamma**(1/epsilon))) ** (epsilon/(1-epsilon))
             )

    print 'USING firm.getK()'

    return K
