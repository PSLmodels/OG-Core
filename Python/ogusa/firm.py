'''
------------------------------------------------------------------------
Last updated 9/8/2016

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
        r = (1-tau_b)*((Z**((epsilon-1)/epsilon))*(((gamma*Y)/K)**(1/epsilon))) - delta + tau_b*delta_tau

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
          (((1-gamma)**(1/epsilon))*(L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))

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
    delta, g_y, omega, lambdas, imm_rates, g_n, method = params

    if method == 'SS':
        omega_extended = np.append(omega[1:],[0.0])
        imm_extended = np.append(imm_rates[1:],[0.0])
        part2 = (((b_splus1*(omega_extended*imm_extended).reshape(omega.shape[0],1))*lambdas).sum())/(1+g_n)
        aggI =   (1+g_n)*np.exp(g_y)*(K_p1 -  part2) - (1.0 - delta) * K
    elif method == 'TPI':
        # omega_extended = np.append(omega[1:,:,:],np.zeros((1,omega.shape[1],omega.shape[2])),axis=0)
        # imm_extended = np.append(imm_rates[1:,:,:],np.zeros((1,imm_rates.shape[1],imm_rates.shape[2])),axis=0)
        # part2 = ((b_splus1*omega_extended*imm_extended*lambdas).sum(1).sum(1))/(1+g_n)
        omega_shift = np.append(omega[:,1:,:],np.zeros((omega.shape[0],1,omega.shape[2])),axis=1)
        imm_shift = np.append(imm_rates[:,1:,:],np.zeros((imm_rates.shape[0],1,imm_rates.shape[2])),axis=1)
        part2 = ((b_splus1*imm_shift*omega_shift*lambdas).sum(1).sum(1))/(1+g_n)
        aggI =   (1+g_n)*np.exp(g_y)*(K_p1 -  part2) - (1.0 - delta) * K

    return aggI

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
    print 'USING firm.getK()'
    if epsilon == 1:
        K = ((1-tau_b)*gamma*Z/(r+delta-(tau_b*delta_tau)))**(1/(1-gamma)) * L
    elif epsilon == 0:
        K = (1-((1-gamma)*L))/gamma
    else:
        K = (((1-gamma)**(1/(epsilon-1)))*((((((r+delta-(tau_b*delta_tau))/(1-tau_b))**(epsilon-1))*(gamma**((1-epsilon)/epsilon))
             *(Z**(1-epsilon)))-(gamma**(1/epsilon)))**(epsilon/(1-epsilon)))*L)

    print 'USING firm.getK()'

    return K
