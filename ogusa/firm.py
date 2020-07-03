'''
------------------------------------------------------------------------
Firm functions for firms in the steady state and along the transition
path
------------------------------------------------------------------------
'''
import numpy as np
import scipy.optimize as opt
from ogusa import aggregates as aggr

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


def get_r(Y, K, Kp1, V, Vp1, X, Xp1, p, method):
    r'''
    This function computes the interest rate as a function of Y, K, and
    parameters using the firm's first order condition for capital
    demand.

    .. math::
        r_{t} = (1 - \tau^{corp})(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{Y_t}{K_t}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau

    Args:
        Y (array_like): aggregate output
        K (array_like): aggregate capital
        Kp1 (array_like): aggregate capital one period ahead
        V (array_like): aggregate firm value
        Vp1 (array_like): aggregate firm value one period ahead
        X (array_like): aggregate value of depreciation deductions on
            existing capital
        Xp1 (array_like): one period ahead aggregate value of
            depreciation deductions on existing capital
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        r (array_like): the real interest rate

    '''
    if method == 'SS':
        tau_b = p.tau_b[-1]
        g_np1 = p.g_nss
    else:
        tau_b = p.tau_b[:p.T]
        g_np1 = p.g_n[1:p.T+1]
    MPK = get_MPK(Y, K, p, method)
    I = aggr.get_I(Kp1, K, g_np1, p.g_y, p.delta)
    q = get_q(K, V, X)
    qp1 = get_q(Kp1, Vp1, Xp1)
    print('MPK = ', MPK)
    print('I = ', I)
    print('q = ', q)
    print('qp1 = ', qp1)
    dPsi_dK = adj_cost_dK(K, Kp1, p, method)
    print('adj costs  = ', dPsi_dK)

    r = ((
        (1 - tau_b) * (MPK - I * dPsi_dK) + (1 - p.delta) * qp1 - q) / q)

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
    w = get_MPL(Y, L, p, method)

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


def adj_cost(K, Kp1, p, method):
    r'''
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
        g_np1 = p.g_n_ss
    else:
        g_np1 = p.g_n[1:p.T+1]
    I = aggr.get_I(Kp1, K, g_np1, p.g_y, p.delta)
    Psi = ((p.psi / 2) * (I / K - p.mu) ** 2) / (I / K)

    return Psi


def adj_cost_dK(K, Kp1, p, method):
    r'''
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
        g_np1 = p.g_n_ss
    else:
        g_np1 = p.g_n[1:p.T+1]
    I = aggr.get_I(Kp1, K, g_np1, p.g_y, p.delta)
    dPsi = (((p.psi * (I / K - p.mu) * Kp1) / (I * K)) *
            ((I / K - p.mu) / (2 * I) - 1))

    return dPsi


def adj_cost_dKp1(K, Kp1, p, method):
    r'''
    Derivative of firm capital adjstment costs with respect to one
    period-ahead capital stock.

    ..math::
        \frac{\partial \Psi(K_{t}, K_{t}+1)}{\partial K_{t}} = \psi \frac{\biggr(\frac{I_{t}}{K_{t}} - \mu\biggl)}{I_{t}} - \frac{\psi}{2}\frac{\biggr(\frac{I_{t}}{K_{t}} - \mu\biggl)^{2}}{\frac{I_{t}^{2}}{K_{t}^{3}}}

    Args:
        K (array-like): Current period capital stock
        Kp1 (array-like): One-period ahead capital stock
        p (OG-USA Parameters class object): Model parameters

    Returns:
        dPsi (array-like): Derivative of capital adjstment costs
    '''
    if method == 'SS':
        g_np1 = p.g_n_ss
    else:
        g_np1 = p.g_n[1:p.T+1]
    I = aggr.get_I(Kp1, K, g_np1, p.g_y, p.delta)
    dPsi = (((p.psi * (I / K - p.mu)) / I) *
            (1 - ((I / K - p.mu) / (2 * I / K))))

    return dPsi


def get_NPV_depr(r, p, method):
    r'''
    Computes the NPV of depreciation deductions per unit of capital.

    ..math::
        z_{t} = \sum_{u=t}^{\infty}\tau^{b}_{u}\delta^{\tau}_{u}\prod_{v=t}^{u}\frac{(1-\delta^{\tau}_{v})}{1+r_{v}}

    Args:
        r (array_like): the real interest rate
        p (OG-USA Parameters class object): Model parameters
        method (str): Whether computing for SS or time path

    Returns:
        z (array_like): Net present value of depreciation deductions

    '''
    if method == 'SS':
        z = p.tau_b[-1] * p.delta_tau[-1] / (r + p.delta_tau[-1])
    else:
        z_ss = p.tau_b[-1] * p.delta_tau[-1] / (r[-1] + p.delta_tau[-1])
        z = np.ones(p.T + p.S - 1) * z_ss
        for t in range(p.T):
            z[t] = 0
            depr_basis = 1
            discount = 1
            for u in range(t + 1, p.T):
                discount *= 1 / (1 + r[u])
                z[t] += p.tau_b[u] * p.delta_tau[u] * depr_basis * discount
                depr_basis *= (1 - p.delta_tau[u])
            z[t] += z_ss * depr_basis * discount

    return z


def get_K_tau_p1(K_tau, I, delta_tau):
    r'''
    Computes the tax basis of the depreciable capital stock using its
    law of motion.

    ..math::
        K^{\tau}_{t+1} = (1 - \delta^{\tau})K^{\tau}_{t} + I_{t}

    Args:
        K_tau (array_like): Tax basis of depreciable stock of capital
            at time t
        I (array_like): Investment at time t
        delta_tau (array_like): Rate of depreciation for tax purposes

    Returns:
        K_tau_p1 (array_like): Tax basis of depreciable stock of capital
            at time t + 1

    '''
    if delta_tau == 0:
        K_tau_p1 = np.zeros_like(I)
    else:
        K_tau_p1 = (1 - delta_tau) * K_tau + I

    return K_tau_p1


def get_K_demand(K0, V, K_tau0, z, delta, psi, mu, tau_b, delta_tau, p):
    '''
    Function to solve for capital demand using the firm's FOC for its
    choice of investment.

    Args:
        K0 (scalar): initial period capital stock
        V (array_like): aggregate firm value
        K_tau0 (scalar): initial period tax depreciable basis of the
            capital stock
        z (array_like): NPV of depreciation deductions on a unit of
            capital
        delta (scalar): per period depreciation rate
        psi (scalar): scale parameter in adjustment cost function
        mu (scalar): shift parameter in adjustment cost function
        delta_tau (array_like): rate of depreciation for tax purposes
        p (OG-USA Parameters class object): Model parameters

    Returns:
        K (array_like): capital demand by the firm

    '''
    K = np.zeros(p.T)
    K_tau = np.zeros(p.T)
    K[0] = K0
    K_tau[0] = K_tau0
    for t in range(p.T - 1):
        Kp1_args = (K[t], V[t+1], K_tau[t], z[t+1], delta, psi,
                    mu, tau_b, delta_tau, p, t)
        results = opt.root(FOC_I, K[t], args=Kp1_args)
        K[t + 1] = results.x
        I = aggr.get_I(K[t+1], K[t], p.g_n[t+1], p.g_y, p.delta)
        K_tau[t + 1] = get_K_tau_p1(K_tau[t], I, delta_tau)


def FOC_I(Kp1, *args):
    '''
    Returns error from the firm's FOC for its choice of
    investment.

    ..math::

    Args:
        Kp1 (scalar): initial guess at one period ahead capital stock
        args (tuple): tuple of length 5, (K, Vp1, delta, psi, mu)
            K (scalar): current period aggregate capital stock
            Vp1 (array_like): one period ahead aggregate firm value
            p (OG-USA Parameters class object): Model parameters
            t (int): period in time path

    Returns:
        error (scalar): error in  FOC

    '''
    K, Vp1, K_tau, z, p, t = args
    I = aggr.get_I(Kp1, K, p.g_n[t+1], p.g_y, p.delta)
    K_tau_p1 = get_K_tau_p1(K_tau, I, p.delta_tau[t])
    Xp1 = get_X(z, K_tau_p1)
    qp1 = get_q(Kp1, Vp1, Xp1)

    error = (qp1 - 1 - (
        (1 - p.tau_b[t]) * p.psi * ((Kp1 / K) - 1 + p.delta - p.mu)) + z)
    # NOTE SURE IF z in above should be discounted by rp1...
    return error


def get_X(z, K_tau):
    r'''
    Computes the NPV of future depreciation deductions on old capital

    ..math::
        X_{t = Z_{t}K^{\tau}_{t}}

    Args:
        z (array_like): NPV of depreciation deductions per unit of capital
        K_tau (array_like): tax basis of the capital stock
    '''
    X = z * K_tau

    return X


def get_q(K, V, X):
    r'''
    Computes Tobin's q, the marginal increase in firm value for an
    additional unit of capital.  Derived from Hayashi's (1982) proof
    that marginal q = average q under quadratic adjustment costs.

    ..math::
        q_{t} = \frac{V_{t} - X_{t}}{K_{t}}

    Args:
        K (array_like): firm capital stock
        V (array_like): firm value
        X (array_like): value of future depreciation deductions on
            existing capital

    Returns:
        q (array_like): Tobin's q

    '''
    q = (V - X) / K

    return q
