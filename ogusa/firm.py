from __future__ import print_function
'''
------------------------------------------------------------------------

Firm functions for firms in the steady state and along the transition
path, including functions for small open economy firms which take r as given.

------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_r(Y, K, p):
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
    if p.epsilon == 0:
        r = (1 - p.tau_b) * p.gamma - p.delta
    else:
        r = ((1 - p.tau_b) * ((p.Z ** ((p.epsilon - 1) / p.epsilon)) *
             (((p.gamma * Y) / K) ** (1 / p.epsilon)))
             - p.delta + p.tau_b * p.delta_tau)

    return r


def get_w(Y, L, p):
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
    if p.epsilon == 0:
        w = 1 - p.gamma
    else:
        w = ((p.Z ** ((p.epsilon - 1) / p.epsilon)) * ((((1 - p.gamma)
                                                         * Y) / L) **
                                                       (1 / p.epsilon)))

    return w


def get_w_from_r(r, p):
    '''
    --------------------------------------------------------------------
    Solve for steady-state wage w or time path of wages w_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 4 tuple, (Z, alpha, delta, tau_c)
    Z      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate
    r      = scalar or (T+S-2) vector, real interest rate
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    OBJECTS CREATED WITHIN FUNCTION:
    w = scalar > 0 or (T+S-2) vector, steady-state wage or time path of
        wage
    FILES CREATED BY THIS FUNCTION: None
    RETURNS: w
    --------------------------------------------------------------------
    '''
    if p.epsilon == 0:
        w = 1 - p.gamma
    else:
        w = ((1 - p.gamma) * p.Z * ((p.gamma * p.Z * (1 - p.tau_b)) /
                                    (r + p.delta - (p.tau_b *
                                                    p.delta_tau))) **
             (p.gamma / (1 - p.gamma)))

    return w


def get_Y(K, L, p):
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
    if p.epsilon == 1:
        Y = p.Z * (K ** p.gamma) * (L ** (1 - p.gamma))
    elif p.epsilon == 0:
        Y = p.Z * (p.gamma * K + (1 - p.gamma) * L)
    else:
        Y = (p.Z * (((p.gamma ** (1 / p.epsilon)) *
                     (K ** ((p.epsilon - 1) / p.epsilon))) +
                    (((1 - p.gamma) ** (1 / p.epsilon)) *
                     (L ** ((p.epsilon - 1) / p.epsilon)))) **
             (p.epsilon / (p.epsilon - 1)))

    return Y


def get_K(L, r, p):
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
    if p.epsilon == 1:
        K = (((1 - p.tau_b) * p.gamma * p.Z /
              (r + p.delta - (p.tau_b * p.delta_tau))) **
             (1 / (1 - p.gamma)) * L)
    elif p.epsilon == 0:
        K = (1 - ((1 - p.gamma) * L)) / p.gamma
    else:
        K = (L *
             ((1-p.gamma) ** (1 / (p.epsilon - 1))) *
             (((
                (((r + p.delta - (p.tau_b * p.delta_tau)) /
                  (1 - p.tau_b)) ** (p.epsilon - 1)) *
                (p.gamma ** ((1 - p.epsilon) / p.epsilon)) *
                (p.Z ** (1 - p.epsilon))) - p.gamma **
               (1 / p.epsilon))) ** (p.epsilon / (1 - p.epsilon)))

    print('USING firm.getK()')

    return K
