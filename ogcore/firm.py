import numpy as np

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


def get_Y(K, K_g, L, p, method):
    r'''
    Generates aggregate output (GDP) from aggregate capital stock,
    aggregate labor, and CES production function parameters.

    .. math::
        \hat{Y}_t &= F(\hat{K}_t, \hat{K}_{g,t}, \hat{L}_t) \\
        &\equiv Z_t\biggl[(\gamma)^\frac{1}{\varepsilon}(\hat{K}_t)^\frac{\varepsilon-1}{\varepsilon} +
          (\gamma_{g})^\frac{1}{\varepsilon}(\hat{K}_{g,t})^\frac{\varepsilon-1}{\varepsilon} +
          (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(\hat{L}_t)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}
          \quad\forall t

    Args:
        K (array_like): aggregate private capital
        K_g (array_like): aggregate government capital
        L (array_like): aggregate labor
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        Y (array_like): aggregate output

    '''
    if method == 'SS':
        Z = p.Z[-1]
        # Set gamma_g to 0 when K_g=0 and eps=1 to remove K_g from prod func
        if K_g == 0 and p.epsilon <= 1:
            gamma_g = 0
            K_g = 1
        else:
            gamma_g = p.gamma_g
    else:
        Z = p.Z[:p.T]
        # Change values of K_g=0 to 1 when eps=1 to remove K_g from prod func
        if np.any(K_g == 0) and p.epsilon == 1:
            K_g[K_g == 0] = 1.0
            gamma_g = 0
        else:
            gamma_g = p.gamma_g
    if p.epsilon == 1:
        # Unit elasticity, Cobb-Douglas
        Y = (Z * (K ** p.gamma) * (K_g ** gamma_g) *
             (L ** (1 - p.gamma - gamma_g)))
    else:
        # General CES
        Y = (Z * (((p.gamma ** (1 / p.epsilon)) *
                   (K ** ((p.epsilon - 1) / p.epsilon))) +
                  ((gamma_g ** (1 / p.epsilon)) *
                   (K_g ** ((p.epsilon - 1) / p.epsilon))) +
                  (((1 - p.gamma - gamma_g) ** (1 / p.epsilon)) *
                   (L ** ((p.epsilon - 1) / p.epsilon)))) **
             (p.epsilon / (p.epsilon - 1)))

    return Y


def get_r(Y, K, p, method):
    r'''
    This function computes the interest rate as a function of Y, K, and
    parameters using the firm's first order condition for capital
    demand.

    .. math::
        r_{t} = (1 - \tau^{corp}_t)Z_t^\frac{\varepsilon-1}{\varepsilon}
        \left[\gamma\frac{Y_t}{K_t}\right]^\frac{1}{\varepsilon} -
        \delta + \tau^{corp}_t\delta^\tau_t

    Args:
        Y (array_like): aggregate output
        K (array_like): aggregate capital
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        r (array_like): the real interest rate

    '''
    if method == 'SS':
        delta_tau = p.delta_tau[-1]
        tau_b = p.tau_b[-1]
    else:
        delta_tau = p.delta_tau[:p.T]
        tau_b = p.tau_b[:p.T]
    MPK = get_MPx(Y, K, p.gamma, p, method)
    r = (1 - tau_b) * MPK - p.delta + tau_b * delta_tau

    return r


def get_w(Y, L, p, method):
    r'''
    This function computes the wage as a function of Y, L, and
    parameters using the firm's first order condition for labor demand.

    .. math::
        w_t = Z_t^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma-\gamma_g)
        \frac{\hat{Y}_t}{\hat{L}_t}\right]^\frac{1}{\varepsilon}

    Args:
        Y (array_like): aggregate output
        L (array_like): aggregate labor
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        w (array_like): the real wage rate

    '''
    w = get_MPx(Y, L, 1 - p.gamma - p.gamma_g, p, method)

    return w


def get_KLratio_KLonly(r, p, method):
    r'''
    This function solves for the capital-labor ratio given the interest
    rate, r, and parameters when the production function is only a
    function of K and L.  This is used in the get_w_from_r function.

    .. math::
        \frac{K}{L} = \left(\frac{(1-\gamma)^\frac{1}{\varepsilon}}
        {\left[\frac{r_t + \delta - \tau_t^{corp}\delta_t^\tau}
        {(1 - \tau_t^{corp})\gamma^\frac{1}
        {\varepsilon}Z_t}\right]^{\varepsilon-1} -
        \gamma^\frac{1}{\varepsilon}}\right)^\frac{\varepsilon}{\varepsilon-1}

    Args:
        r (array_like): the real interest rate
        p (OG-Core Specifications object): model parameters
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


def get_KLratio(r, w, p, method):
    r'''
    This function solves for the capital-labor ratio given the interest
    rate r wage w and parameters.

    .. math::
        \frac{K}{L} = \left(\frac{\gamma}{1 - \gamma - \gamma_g}\right)
            \left(\frac{w_t}{\frac{r_t + \delta -
            \tau_t^{corp}\delta_t^{\tau}}{1 -
            \tau_t^{corp}}}\right)^\varepsilon

    Args:
        r (array_like): the real interest rate
        w (array_like): the wage rate
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        KLratio (array_like): the capital-labor ratio

    '''
    if method == 'SS':
        tau_b = p.tau_b[-1]
        delta_tau = p.delta_tau[-1]
    else:
        tau_b = p.tau_b[:p.T]
        delta_tau = p.delta_tau[:p.T]
    cost_of_capital = (r + p.delta - tau_b * delta_tau) / (1 - tau_b)
    KLratio = (
        (p.gamma / (1 - p.gamma - p.gamma_g)) *
        (w / cost_of_capital) ** p.epsilon)
    return KLratio


def get_MPx(Y, x, share, p, method):
    r'''
    Compute the marginal product of x (where x is K, L, or K_g)

    .. math::
        MPx = Z_t^\frac{\varepsilon-1}{\varepsilon}\left[(share)
        \frac{\hat{Y}_t}{\hat{x}_t}\right]^\frac{1}{\varepsilon}

    Args:
        Y (array_like): output
        x (array_like): input to production function
        share (scalar): share of output paid to factor x
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        MPx (array_like): the marginal product of x
    '''
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    if np.any(x) == 0:
        MPx = np.zeros_like(Y)
    else:
        MPx = (
            Z ** ((p.epsilon - 1) / p.epsilon) * ((share * Y) / x)
            ** (1 / p.epsilon)
        )

    return MPx


def get_w_from_r(r, p, method):
    r'''
    Solve for a wage rate from a given interest rate.  N.B. this is only
    appropriate if the production function only uses capital and labor
    as inputs.  As such, this is not used for determining the domestic
    wage rate due to the presense of public capital in the production
    function.  It is used only to determine the wage rate that affects
    the open economy demand for capital.

    .. math::
        w = (1-\gamma)^\frac{1}{\varepsilon}Z\left[(\gamma)^\frac{1}
        {\varepsilon}\left(\frac{(1-\gamma)^\frac{1}{\varepsilon}}
        {\left[\frac{r + \delta - \tau^{corp}\delta^\tau}{(1 - \tau^{corp})
        \gamma^\frac{1}{\varepsilon}Z}\right]^{\varepsilon-1} -
        \gamma^\frac{1}{\varepsilon}}\right) +
        (1-\gamma)^\frac{1}{\varepsilon}\right]^\frac{1}{\varepsilon-1}

    Args:
        r (array_like): the real interest rate
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        w (array_like): the real wage rate

    '''
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    KLratio = get_KLratio_KLonly(r, p, method)
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


def get_K_KLonly(L, r, p, method):
    r'''
    Generates vector of aggregate capital when the production function
    uses only K and L as inputs. Use with the open economy options.

    .. math::
        K_{t} = \frac{K_{t}}{L_{t}} \times L_{t}

    Args:
        L (array_like): aggregate labor
        r (array_like): the real interest rate
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        K (array_like): aggregate capital demand

    '''
    KLratio = get_KLratio_KLonly(r, p, method)
    K = KLratio * L

    return K


def get_K_from_Y(Y, r, p, method):
    r'''
    Generates vector of aggregate capital. Use with the open economy
    options.

    .. math::
        K_{t} = \frac{Y_{t}}{Y_{t}/K_{t}} \\
        K_{t} = \frac{\gamma Z_t^{\varepsilon -1} Y_t}{
            \left(\frac{r_t + \delta - \tau_t^{corp}\delta_t^\tau}
            {1 - \tau_{t}^{corp}}\right)^\varepsilon}

    Args:
        Y (array_like): aggregate output
        r (array_like): the real interest rate
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        r (array_like): the real interest rate

    '''
    if method == 'SS':
        Z = p.Z[-1]
        tau_b = p.tau_b[-1]
        delta_tau = p.delta_tau[-1]
    else:
        Z = p.Z[:p.T]
        tau_b = p.tau_b[:p.T]
        delta_tau = p.delta_tau[:p.T]
    numerator = p.gamma * Z ** (p.epsilon - 1) * Y
    denominator = (
        ((r + p.delta - tau_b * delta_tau) / (1 - tau_b)) ** p.epsilon
        )
    K = numerator / denominator

    return K


def get_L_from_Y(w, Y, p, method):
    r'''
    Find aggregate labor L from output Y and wages w

    .. math::
        L_{t} = \frac{(1 - \gamma - \gamma_g) Z_{t}^{\varepsilon-1}
        Y_{t}}{w_{t}^{\varepsilon}}

    Args:
        w (array_like): the wage rate
        Y (array_like): aggregate output
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        L (array_like): firm labor demand

    '''
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    L = (
        ((1 - p.gamma - p.gamma_g) * Z ** (p.epsilon - 1) * Y) /
        (w ** p.epsilon)
        )

    return L


def get_K_from_Y_and_L(Y, L, K_g, p, method):
    r'''
    Find aggregate private capital K from output Y, aggregate labor L,
    and public capital K_g

    .. math::
        K_{t} = \left(\frac{\left(\frac{Y_t}{Z_t}\right)^{\frac{\varepsilon-1}
        {\varepsilon}} -
        (1-\gamma-\gamma_g)L_t^{\frac{\varepsilon-1}{\varepsilon}} -
        \gamma_g^{\frac{1}{\varepsilon}}K_{g,t}^{\frac{\varepsilon-1}{\varepsilon}}}
        {\gamma^{\frac{1}{\varepsilon}}}\right)^{\frac{\varepsilon}{\varepsilon-1}}

    Args:
        w (array_like): the wage rate
        Y (array_like): aggregate output
        L (array_like): aggregate labor
        K_g (array_like): aggregate public capital
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        K (array_like): firm capital demand

    '''
    if method == 'SS':
        Z = p.Z[-1]
    else:
        Z = p.Z[:p.T]
    K = (
        (((Y / Z) ** ((p.epsilon - 1) / p.epsilon) -
            (1 - p.gamma - p.gamma_g) *
            L ** ((p.epsilon - 1) / p.epsilon) -
            (p.gamma_g ** (1 / p.epsilon)) *
            (K_g ** ((p.epsilon - 1) / p.epsilon))) /
            (p.gamma ** (1 / p.epsilon))) ** (p.epsilon / (p.epsilon - 1)))

    return K


def get_K(r, w, L, p, method):
    r'''
    Get K from r, w, L.  For determining capital demand for open
    economy case.

    .. math::
        K_{t} = \frac{K_{t}}{L_{t}} \times L_{t}

    Args:
        r (array_like): the real interest rate
        w (array_like): the wage rate
        L (array_like): aggregate labor
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        K (array_like): aggregate capital demand

    '''
    KLratio = get_KLratio(r, w, p, method)
    K = KLratio * L

    return K
