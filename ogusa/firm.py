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
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_Y(K, L, p, method):
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
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    if p.epsilon == 1:
        # Unit elasticity, Cobb-Douglas
        Y = Z * (K ** p.gamma) * (L ** (1 - p.gamma))
    else:
        # General case
        Y = (Z * (((p.gamma ** (1 / p.epsilon)) *
                   (K ** ((p.epsilon - 1) / p.epsilon))) +
                  (((1 - p.gamma) ** (1 / p.epsilon)) *
                   (L ** ((p.epsilon - 1) / p.epsilon)))) **
             (p.epsilon / (p.epsilon - 1)))

    return Y


def get_MPK(Y, K, p, method):
    '''
    Compute the marginal product of capital.
    '''
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    MPK = ((Z ** ((p.epsilon - 1) / p.epsilon)) *
           (((p.gamma * Y) / K) ** (1 / p.epsilon)))

    return MPK


def get_MPL(Y, L, p, method):
    '''
    Compute the marginal product of labor.
    '''
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    MPL = ((Z ** ((p.epsilon - 1) / p.epsilon)) *
           ((((1 - p.gamma) * Y) / L) ** (1 / p.epsilon)))

    return MPL


def get_r(Y, K, p, method):
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
    if method == 'SS':
        delta_tau = p.delta_tau[-1]
        tau_b = p.tau_b[-1]
    else:
        delta_tau = p.delta_tau[:p.T]
        tau_b = p.tau_b[:p.T]
    MPK = get_MPK(Y, K, p, method)
    r = (1 - tau_b) * MPK - p.delta + tau_b * delta_tau

    return r


def get_w(Y, L, p, method):
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
    w = get_MPL(Y, L, p, method)

    return w


def get_KLratio_from_r(r, p, method):
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
    if method == 'SS':
        Z = p.Z[-1]
        delta_tau = p.delta_tau[-1]
        tau_b = p.tau_b[-1]
    else:
        length = r.shape[0]
        Z = p.Z[:length]
        delta_tau = p.delta_tau[:length]
        tau_b = p.tau_b[:length]
    if p.epsilon == 1:
        # Cobb-Douglas case
        bracket = (((1 - tau_b) * p.gamma * Z) /
                   (r + p.delta - tau_b * delta_tau))
        KLratio = bracket ** (1 / (1 - p.gamma))
    else:
        # General CES case
        bracket = ((r + p.delta - (delta_tau * tau_b)) /
                   ((1 - tau_b) * Z * (p.gamma ** (1 / p.epsilon))))
        KLratio = \
            ((((1 - p.gamma) ** (1 / p.epsilon)) /
              ((bracket ** (p.epsilon - 1)) -
               (p.gamma ** (1 / p.epsilon)))) ** (p.epsilon /
                                                  (p.epsilon - 1)))
    return KLratio


def get_w_from_r(r, p, method):
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
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    KLratio = get_KLratio_from_r(r, p, method)
    if p.epsilon == 1:
        # Cobb-Douglas case
        w = (1 - p.gamma) * Z * (KLratio ** p.gamma)
    else:
        # General CES case
        w = (((1 - p.gamma) ** (1 / p.epsilon)) * Z *
             (((p.gamma ** (1 / p.epsilon)) *
               (KLratio ** ((p.epsilon - 1) / p.epsilon)) +
               ((1 - p.gamma) ** (1 / p.epsilon))) **
              (1 / (p.epsilon - 1))))
    return w


def get_K(L, r, p, method):
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
    KLratio = get_KLratio_from_r(r, p, method)
    K = KLratio * L

    return K


def get_K_from_Y(Y, r, p, method):
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
    KLratio = get_KLratio_from_r(r, p, method)
    LKratio = KLratio ** -1
    YKratio = get_Y(1, LKratio, p, method)
    K = Y / YKratio

    return K


def adj_cost(K, Kp1, p, method):
    '''
    Firm capital adjstment costs

    ..math::
        \Psi(K_{t}, K_{t+1}) = \frac{\psi}{2}\biggr(\frac{\biggr(\frac{I_{t}}{K_{t}}-\mu\biggl)^{2}}{\frac{I_{t}}{K_{t}}}\biggl)

    Args:
        K (array-like): Current period capital stock
        Kp1 (array-like): One-period ahead capital stock
        p (OG-USA Parameters class object): Model parameters

    Returns
        Psi (array-like): Capital adjstment costs
    '''
    if method == 'SS':
        g_n = p.g_n_ss
    else:
        length = K.shape[0]
        g_n = p.g_n[:length]
    I = Kp1 * np.exp(p.g_y) * (1 + g_n) - (1 - p.delta) * K
    Psi = ((p.psi / 2) * (I / K - p.mu) ** 2) / (I / K)

    return Psi


def adj_cost_dK(K, Kp1, p, method):
    '''
    Derivative of firm capital adjstment costs with respect to current
    period capital stock.

    ..math::
        \frac{\partial \Psi(K_{t}, K_{t}+1)}{\partial K_{t}} = -\psi \biggr(\frac{I_{t}}{K_{t}} - \mu\biggl)\frac{K_{t+1}}{I_{t}} - \frac{\psi}{2}\frac{\biggr(\frac{I_{t}}{K_{t}} - \mu\biggl)^{2}}{\frac{I_{t}^{2}K_{t+1}}{K_{t}^{3}}}

    Args:
        K (array-like): Current period capital stock
        Kp1 (array-like): One-period ahead capital stock
        p (OG-USA Parameters class object): Model parameters

    Returns
        dPsi (array-like): Derivative of capital adjstment costs
    '''
    if method == 'SS':
        g_n = p.g_n_ss
    else:
        length = K.shape[0]
        g_n = p.g_n[:length]
    I = Kp1 * np.exp(p.g_y) * (1 + g_n) - (1 - p.delta) * K
    dPsi = (((p.psi * (I / K - p.mu) * Kp1) / (I * K)) *
            ((I / K - p.mu) / (2 * I) - 1))

    return dPsi


def adj_cost_dKp1(K, Kp1, p, method):
    '''
    Derivative of firm capital adjstment costs with respect to one
    period-ahead capital stock.

    ..math::
        \frac{\partial \Psi(K_{t}, K_{t}+1)}{\partial K_{t}} = \psi \frac{\biggr(\frac{I_{t}}{K_{t}} - \mu\biggl)}{I_{t}} - \frac{\psi}{2}\frac{\biggr(\frac{I_{t}}{K_{t}} - \mu\biggl)^{2}}{\frac{I_{t}^{2}}{K_{t}^{3}}}

    Args:
        K (array-like): Current period capital stock
        Kp1 (array-like): One-period ahead capital stock
        p (OG-USA Parameters class object): Model parameters

    Returns
        dPsi (array-like): Derivative of capital adjstment costs
    '''
    if method == 'SS':
        g_n = p.g_n_ss
    else:
        length = K.shape[0]
        g_n = p.g_n[:length]
    I = Kp1 * np.exp(p.g_y) * (1 + g_n) - (1 - p.delta) * K
    dPsi = (((p.psi * (I / K - p.mu)) / I) *
            (1 - ((I / K - p.mu) / (2 * I / K))))

    return dPsi
