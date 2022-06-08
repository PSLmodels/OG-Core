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


def get_Y(K, K_g, L, p, method, m=-1):
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
        m (int or None): industry index

    Returns:
        Y (array_like): aggregate output

    '''
    # TODO: Generalize for T x M
    # in this case, follow example of household functions that allow
    # one to pass j or not (if not, then do for all j at once)

    if method == 'SS':
        if m is not None:
            # Set gamma_g to 0 when K_g=0 and eps=1 to remove K_g from prod func
            if K_g == 0 and p.epsilon[m] <= 1:
                gamma_g = 0
                K_g = 1
            else:
                gamma_g = p.gamma_g[m]
            gamma = p.gamma[m]
            epsilon = p.epsilon[m]
            Z = p.Z[-1, m]
            if epsilon == 1:
                Y = (Z * (K ** gamma) * (K_g ** gamma_g) *
                      (L ** (1 -  gamma - gamma_g)))
            else:
                Y = (Z * (((gamma ** (1 / epsilon)) *
                    (K ** ((epsilon - 1) / epsilon))) +
                    ((gamma_g ** (1 / epsilon)) *
                    (K_g ** ((epsilon - 1) / epsilon))) +
                    (((1 - gamma - gamma_g) ** (1 / epsilon)) *
                    (L ** ((epsilon - 1) / epsilon)))) **
                    (epsilon / (epsilon - 1)))
        else:
            # Set gamma_g to 0 when K_g=0 and eps=1 to remove K_g from prod func
            if K_g == 0 and np.any(p.epsilon) <= 1:
                gamma_g = p.gamma_g
                gamma_g[p.epsilon <= 1] = 0
                K_g = 1.0
            else:
                gamma_g = p.gamma_g
            gamma = p.gamma
            epsilon = p.epsilon
            Z = p.Z[-1, :]
            Y = (Z * (((gamma ** (1 / epsilon)) *
                (K ** ((epsilon - 1) / epsilon))) +
                ((gamma_g ** (1 / epsilon)) *
                (K_g ** ((epsilon - 1) / epsilon))) +
                (((1 - gamma - gamma_g) ** (1 / epsilon)) *
                (L ** ((epsilon - 1) / epsilon)))) **
                (epsilon / (epsilon - 1)))
            Y2 = (Z * (K ** gamma) * (K_g ** gamma_g) *
                (L ** (1 - gamma - gamma_g)))
            Y[epsilon == 1] = Y2[epsilon == 1]
    else: #TPI case
        if m is not None:
            # Set gamma_g to 0 when K_g=0 and eps=1 to remove K_g from prod func
            if np.any(K_g == 0) and p.epsilon[m] == 1:
                gamma_g = 0
                K_g[K_g == 0] = 1.0
            else:
                gamma_g = p.gamma_g[m]
            gamma = p.gamma[m]
            epsilon = p.epsilon[m]
            Z = p.Z[:p.T, m]
            if epsilon == 1:
                Y = (Z * (K ** gamma) * (K_g ** gamma_g) *
                     (L ** (1 - gamma - gamma_g)))
            else:
                Y = (Z * (((gamma ** (1 / epsilon)) *
                    (K ** ((epsilon - 1) / epsilon))) +
                    ((gamma_g ** (1 / epsilon)) *
                    (K_g ** ((epsilon - 1) / epsilon))) +
                    (((1 - gamma - gamma_g) ** (1 / epsilon)) *
                    (L ** ((epsilon - 1) / epsilon)))) **
                    (epsilon / (epsilon - 1)))
        else:
            # Set gamma_g to 0 when K_g=0 and eps=1 to remove K_g from prod func
            if np.any(K_g == 0) and np.any(p.epsilon) == 1:
                gamma_g = p.gamma_g
                K_g[K_g == 0] = 1.0
            else:
                gamma_g = p.gamma_g
            gamma = p.gamma
            epsilon = p.epsilon
            Z = p.Z[:p.T, :]
            Y = (Z * (((gamma ** (1 / epsilon)) *
                (K ** ((epsilon - 1) / epsilon))) +
                ((gamma_g ** (1 / epsilon)) *
                (K_g ** ((epsilon - 1) / epsilon))) +
                (((1 - gamma - gamma_g) ** (1 / epsilon)) *
                (L ** ((epsilon - 1) / epsilon)))) **
                (epsilon / (epsilon - 1)))
            Y2 = (Z * (K ** gamma) * (K_g ** gamma_g) *
                (L ** (1 - gamma - gamma_g)))
            Y[:, epsilon == 1] = Y2[:, epsilon == 1]

    return Y


def get_r(Y, K, p_m, p, method, m=-1):
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
        p_m (array_like): output prices
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        m (int): index of the production industry

    Returns:
        r (array_like): the real interest rate

    '''
    if method == 'SS':
        delta_tau = p.delta_tau[-1, m]
        tau_b = p.tau_b[-1, m]
        p_mm = p_m[m]
    else:
        delta_tau = p.delta_tau[:p.T, m].reshape(p.T, 1)
        tau_b = p.tau_b[:p.T, m].reshape(p.T, 1)
        p_mm = p_m[:, m].reshape(p.T, 1)
    MPK = get_MPx(Y, K, p.gamma[m], p, method, m)
    r = (1 - tau_b) * p_mm * MPK - p.delta + tau_b * delta_tau

    return r


def get_w(Y, L, p_m, p, method, m=-1):
    r'''
    This function computes the wage as a function of Y, L, and
    parameters using the firm's first order condition for labor demand.

    .. math::
        w_t = Z_t^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma-\gamma_g)
        \frac{\hat{Y}_t}{\hat{L}_t}\right]^\frac{1}{\varepsilon}

    Args:
        Y (array_like): aggregate output
        L (array_like): aggregate labor
        p_m (array_like): output prices
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        m (int): index of the production industry

    Returns:
        w (array_like): the real wage rate

    '''
    # mp = get_MPx(Y, L, 1 - p.gamma[m] - p.gamma_g[m], p, method, m)
    # print('MPx size = ', mp.shape)
    if method == 'SS':
        p_mm = p_m[m]
    else:
        p_mm = p_m[:, m].reshape(p.T, 1)
    w = p_mm * get_MPx(Y, L, 1 - p.gamma[m] - p.gamma_g[m], p, method, m)

    return w


def get_KLratio_KLonly(r, p, method, m=-1):
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
        m (int): production industry index

    Returns:
        KLratio (array_like): the capital-labor ratio

    '''
    if method == 'SS':
        Z = p.Z[-1, m]
    else:
        length = r.shape[0]
        Z = p.Z[:length, m]
    gamma = p.gamma[m]
    epsilon = p.epsilon[m]
    if epsilon == 1:
        # Cobb-Douglas case
        cost_of_capital = get_cost_of_capital(r, p, method, m)
        KLratio = ((gamma * Z) / cost_of_capital) ** (1 / (1 - gamma))
    else:
        # General CES case
        cost_of_capital = get_cost_of_capital(r, p, method, m)
        bracket = cost_of_capital * (Z * (gamma ** (1 / epsilon))) ** -1
        KLratio = (
            ((1 - gamma) ** (1 / epsilon))
            / ((bracket ** (epsilon - 1)) - (gamma ** (1 / epsilon)))
        ) ** (epsilon / (epsilon - 1))

    return KLratio


def get_KLratio(r, w, p, method, m=-1):
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
        m (int): production industry index

    Returns:
        KLratio (array_like): the capital-labor ratio

    '''
    cost_of_capital = get_cost_of_capital(r, p, method, m)
    KLratio = (
        (p.gamma[m] / (1 - p.gamma[m] - p.gamma_g[m])) *
        (w / cost_of_capital) ** p.epsilon[m])
    return KLratio


def get_MPx(Y, x, share, p, method, m=-1):
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
        m (int): production industry index

    Returns:
        MPx (array_like): the marginal product of x
    '''
    if method == 'SS':
        Z = p.Z[-1, m]
    else:
        Z = p.Z[:p.T, m].reshape(p.T, 1)
        Y = Y[:p.T].reshape(p.T, 1)
        x = x[:p.T].reshape(p.T, 1)
    if np.any(x) == 0:
        MPx = np.zeros_like(Y)
    else:
        MPx = (
            Z ** ((p.epsilon[m] - 1) / p.epsilon[m]) * ((share * Y) / x)
            ** (1 / p.epsilon[m])
        )

    return MPx


def get_w_from_r(r, p, method, m=-1):
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
        m (int or None): production industry index

    Returns:
        w (array_like): the real wage rate

    '''
    KLratio = get_KLratio_KLonly(r, p, method, m)

    if method == 'TPI':
        Z = p.Z[:p.T, m]
    else:
        Z = p.Z[-1, m]
    gamma = p.gamma[m]
    epsilon = p.epsilon[m]
    if epsilon == 1:
        # Cobb-Douglas case
        w = (1 - gamma) * Z * (KLratio ** gamma)
    else:
        # General CES case
        w = (((1 - gamma) ** (1 / epsilon)) * Z *
            (((gamma ** (1 / epsilon)) *
            (KLratio ** ((epsilon - 1) / epsilon)) +
            ((1 - gamma) ** (1 / epsilon))) **
            (1 / (epsilon - 1))))

    return w


def get_K_KLonly(L, r, p, method, m=-1):
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
        m (int or None): production industry index

    Returns:
        K (array_like): aggregate capital demand

    '''
    KLratio = get_KLratio_KLonly(r, p, method, m)
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


def get_K(r, w, L, p, method, m=-1):
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
        m (int or None): production industry index

    Returns:
        K (array_like): aggregate capital demand

    '''
    KLratio = get_KLratio(r, w, p, method, m)
    K = KLratio * L

    return K


def get_cost_of_capital(r, p, method, m=-1):
    r'''
    Compute the cost of capital.

    .. math::
        \rho = \frac{r + delta - \tau^{b} delta_tau}{1 - \tau^{b}}

    Args:
        r (array_like): the real interest rate
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
        m (int or None): production industry index

    Returns:
        cost_of_capital (array_like): cost of capital
    '''
    if m is None:
        if method == 'SS':
            tau_b = p.tau_b[-1, :]
            delta_tau = p.delta_tau[-1, :]
        else:
            tau_b = p.tau_b[:p.T, :]
            delta_tau = p.delta_tau[:p.T, :]
    else:
        if method == 'SS':
            tau_b = p.tau_b[-1, m]
            delta_tau = p.delta_tau[-1, m]
        else:
            tau_b = p.tau_b[:p.T, m]
            delta_tau = p.delta_tau[:p.T, m]
    cost_of_capital = (r + p.delta - tau_b * delta_tau) / (1 - tau_b)

    return cost_of_capital


def get_pm(w, KL_ratio, p, method):
    r'''
    Find prices for outputs from each industry.

    .. math::
        p_{m,t} = \frac{w_{t}^{\varepsilon}}{K_{t}L_{t}}

    Args:
        w (array_like): the wage rate
        KL_ratio (array_like): ratio of capital to labor
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        p_m (array_like): output prices for each industry
    '''
    if method == 'SS':
        Z = p.Z[-1, :].reshape(p.M)
        gamma = p.gamma
        epsilon = p.epsilon
    else:
        Z = p.Z[:p.T, :].reshape((p.T, p.M))
        gamma = p.gamma.reshape((1, p.M))
        epsilon = p.epsilon.reshape((1, p.M))
        w = w.reshape((p.T, 1))
        KL_ratio = KL_ratio.reshape((p.T, p.M))
    # p_m = (w / ((1 - gamma) * Z)) * (KL_ratio ** (-gamma))
    p_m = (
        (w / (
            Z * (1 - gamma) ** (1 / epsilon))) * (
                gamma ** (1 / epsilon) * KL_ratio **
                ((epsilon - 1) / epsilon) +
                 (1 - gamma) ** (1 / epsilon)) **
        (1 / (1 - epsilon)))
    print('Shapes in get_pm:', p_m.shape, w.shape, Z.shape, gamma.shape, epsilon.shape, KL_ratio.shape)

    return p_m


def get_KY_ratio(r, p_m, p, method, m=-1):
    r'''
    Get capital output ratio from FOC for interest rate.

    .. math::

    Args:
        r (array_like): the real interest rate
        p_m (array_like): output prices for each industry
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        KY_ratio (array_like): capital output ratio
    '''
    cost_of_capital = get_cost_of_capital(r, p, method, m)
    KY_ratio = (p_m[m] * p.gamma[m]) / cost_of_capital

    return KY_ratio
