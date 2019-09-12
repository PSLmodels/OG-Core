'''
------------------------------------------------------------------------
Firm functions for firms in the steady state and along the transition
path
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_Y(K, L, p, method):
    r'''
    Generates aggregate output (GDP) from aggregate capital stock,
    aggregate labor, and CES production function parameters.

    .. math::
        Y_{t} = Z_{t}\left[\gamma^{\frac{1}{\varepsilon}}K_{t}^{\frac{\varepsilon - 1}{\varepsilon}} + (1 - \gamma)^{\frac{1}{\varepsilon}}L_{t}^{\frac{\varepsilon - 1}{\varepsilon}}\right]^{\frac{\varepsilon}{\varepsilon - 1}}

    Args:
        K (array_like): aggregate capital
        L (array_like): aggregate labor
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        Y (array_like): aggregate output

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


def get_r(Y, K, p, method):
    r'''
    This function computes the interest rate as a function of Y, K, and
    parameters using the firm's first order condition for capital
    demand.

    .. math::
        r_{t} = (1 - \tau^{corp})(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{Y_t}{K_t}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau

    Args:
        Y (array_like): aggregate output
        K (array_like): aggregate capital
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        r (array_like): the real interest rate

    '''
    if method == 'SS':
        Z = p.Z[-1]
        delta_tau = p.delta_tau[-1]
        tau_b = p.tau_b[-1]
    else:
        Z = p.Z[:p.T]
        delta_tau = p.delta_tau[:p.T]
        tau_b = p.tau_b[:p.T]
    r = ((1 - tau_b) * (Z ** ((p.epsilon - 1) / p.epsilon)) *
         (((p.gamma * Y) / K) ** (1 / p.epsilon)) -
         p.delta + tau_b * delta_tau)

    return r


def get_w(Y, L, p, method):
    r'''
    This function computes the wage as a function of Y, L, and
    parameters using the firm's first order condition for labor demand.

    .. math::
        w_t = (Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma)\frac{\hat{Y}_t}{\hat{L}_t}\right]^\frac{1}{\varepsilon}

    Args:
        Y (array_like): aggregate output
        L (array_like): aggregate labor
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        w (array_like): the real wage rate

    '''
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    w = ((Z ** ((p.epsilon - 1) / p.epsilon)) *
         ((((1 - p.gamma) * Y) / L) ** (1 / p.epsilon)))

    return w


def get_KLratio_from_r(r, p, method):
    r'''
    This function solves for the capital-labor ratio given the interest
    rate r and parameters.

    .. math::
        \frac{K}{L} = \left(\frac{(1-\gamma)^\frac{1}{\varepsilon}}{\left[\frac{r + \delta - \tau^{corp}\delta^\tau}{(1 - \tau^{corp})\gamma^\frac{1}{\varepsilon}Z}\right]^{\varepsilon-1} - \gamma^\frac{1}{\varepsilon}}\right)^\frac{\varepsilon}{\varepsilon-1}

    Args:
        r (array_like): the real interest rate
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        KLratio (array_like): the capital-labor ratio

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
    r'''
    Solve for steady-state wage w or time path of wages w_t given
    interest rate.

    .. math::
        w = (1-\gamma)^\frac{1}{\varepsilon}Z\left[(\gamma)^\frac{1}{\varepsilon}\left(\frac{(1-\gamma)^\frac{1}{\varepsilon}}{\left[\frac{r + \delta - \tau^{corp}\delta^\tau}{(1 - \tau^{corp})\gamma^\frac{1}{\varepsilon}Z}\right]^{\varepsilon-1} - \gamma^\frac{1}{\varepsilon}}\right) + (1-\gamma)^\frac{1}{\varepsilon}\right]^\frac{1}{\varepsilon-1}

    Args:
        r (array_like): the real interest rate
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        w (array_like): the real wage rate

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
    r'''
    Generates vector of aggregate capital. Use with the open economy
    options.

    .. math::
        K_{t} = \frac{K_{t}}{L_{t}} \times L_{t}

    Inputs:
        L (array_like): aggregate labor
        r (array_like): the real interest rate
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        K (array_like): aggregate capital demand
    --------------------------------------------------------------------
    '''
    KLratio = get_KLratio_from_r(r, p, method)
    K = KLratio * L

    return K


def get_K_from_Y(Y, r, p, method):
    r'''
    Generates vector of aggregate capital. Use with the open economy
    options.

    .. math::
        K_{t} = \frac{Y_{t}}{Y_{t}/K_{t}}

    Args:
        Y (array_like): aggregate output
        r (array_like): the real interest rate
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        r (array_like): the real interest rate

    '''
    KLratio = get_KLratio_from_r(r, p, method)
    LKratio = KLratio ** -1
    YKratio = get_Y(1, LKratio, p, method)  # can use get_Y because CRS
    K = Y / YKratio

    return K
