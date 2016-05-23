'''
------------------------------------------------------------------------
Last updated 4/7/2015

Firm functions for firms in the steady state and along the transition 
path.

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

    alpha, delta = params
    r = (alpha * Y / K) - delta
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
    alpha = params
    w = (1 - alpha) * Y / L
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
    alpha, Z = params
    Y = Z * (K ** alpha) * (L ** (1 - alpha))
    return Y


def get_L(n, params):
    '''
    Generates vector of aggregate labor supply.

    Inputs:
        n               = [T,S,J] array, labor supply
        params          = length 4 tuple, (e, omega, lambdas, method)
        e               = [T,S,J] array, effective labor units
        omega     = [T,S,1] array, population weights
        lambdas = [1,1,J] array, ability weights
        method          = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        L_presum = [T,S,J] array, weighted labor supply
        L = [T+S,] vector, aggregate labor 

    Returns: L

    '''
    e, omega, lambdas, method = params

    L_presum = e * omega * lambdas * n
    if method == 'SS':
        L = L_presum.sum()
    elif method == 'TPI':
        L = L_presum.sum(1).sum(1)
    return L


def get_I(b_splus1, K_p1, K, params):
    '''
    Generates vector of aggregate investment.

    Inputs:
        K_p1   = [T,] vector, aggregate capital, one period ahead        
        K      = [T,] vector, aggregate capital
        params = length 3 tuple, (delta, g_y, g_n)
        delta  = scalar, depreciation rate of capital 
        g_y    = scalar, production growth rate 
        g_n    = [T,] vector, population growth rate

    Functions called: None

    Objects in function:
        aggI = [T,] vector, aggregate investment 

    Returns: aggI

    '''
    delta, g_y, omega, lambdas, imm_rates, g_n = params

    #1)
    # aggI = ((1. + g_n) * np.exp(g_y)) * K_p1 - (1.0 - delta) * K

    #2)
    # aggI = ((((1.0 + g_n) * np.exp(g_y)) * K_p1) - (1.0 - delta) * (K + 
    #         ((1./(1+g_n))*(((lambdas*b_splus1)*(imm_rates*omega).reshape((np.shape(omega)[0],1))).sum()))))

    #3)
    # part1 = (lambdas*b_splus1)*omega.reshape((np.shape(omega)[0],1))
    # imm_mask = imm_rates<0.0
    # part2 = (lambdas*b_splus1)*(imm_rates*omega).reshape((np.shape(omega)[0],1))
    # aggI =   np.exp(g_y)*(part1 - part2).sum() - (1.0 - delta) * K

    # part1 = (lambdas*b_splus1)*omega.reshape((np.shape(omega)[0],1))
    # imm_mask = imm_rates<0.0
    # part2 = (lambdas*b_splus1)*(imm_rates*omega*imm_mask).reshape((np.shape(omega)[0],1))
    # aggI =   np.exp(g_y)*(part1 - part2).sum() - (1.0 - delta) * K

    #4)
    # part2 = (((lambdas*b_splus1)*(omega.reshape((omega.shape[0],))*imm_rates).reshape((np.shape(omega)[0],1))).sum())/(1+g_n)
    # aggI =   (1+g_n)*np.exp(g_y)*(K_p1 -  part2).sum() - (1.0 - delta) * K

    omega_extended = np.append(omega[1:],[0.0])
    imm_extended = np.append(imm_rates[1:],[0.0])
    part2 = (((lambdas*b_splus1)*(omega_extended.reshape((omega.shape[0],))*imm_extended).reshape((np.shape(omega)[0],1))).sum())/(1+g_n)
    aggI =   (1+g_n)*np.exp(g_y)*(K_p1 -  part2).sum() - (1.0 - delta) * K


    #5)
    # part2 = (((lambdas*b_splus1)*(omega.reshape((omega.shape[0],))*imm_rates).reshape((np.shape(omega)[0],1))).sum())/(1+g_n)
    # aggI =   np.exp(g_y)*((1+g_n)*K_p1 -  part2).sum() - (1.0 - delta) * K
    

    print 'imm rates:', imm_rates
    print 'pop growth: ', g_n
    print 'omega: ', omega
    return aggI


    return aggI