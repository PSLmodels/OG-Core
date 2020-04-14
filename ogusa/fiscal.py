'''
------------------------------------------------------------------------

Fiscal policy functions for unbalanced budgeting. In particular, some
functions require time-path calculation.

------------------------------------------------------------------------
'''

# Packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def D_G_path(r_gov, dg_fixed_values, Gbaseline, p):
    r'''
    Calculate the time paths of debt and government spending

    .. math::
        \begin{split}
        &G_t = g_{g,t}\:\alpha_{g}\: Y_t \\
        &\text{where}\quad g_{g,t} =
          \begin{cases}
            1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\:\:\,\text{if}\quad t < T_{G1} \\
            \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{t})D_{t} - TR_{t} + Rev_{t}}{\alpha_g Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
            \frac{\alpha_{D}Y_{t} - (1+r_{t})D_{t} - TR_{t} + Rev_{t}}{\alpha_g Y_t} \qquad\qquad\quad\:\:\:\,\text{if}\quad t \geq T_{G2}
          \end{cases} \\
        &\quad\text{and}\quad g_{tr,t} = 1 \quad\forall t
      \end{split}

    Args:
        r_gov (Numpy array): interest rate on government debt over the
            time path
        dg_fixed_values (tuple): (Y, total_revenue, TR, D0, G0) values
            of variables that are taken as given in the government
            budget constraint
        Gbaseline (Numpy array): government spending over the time path
            in the baseline equilibrium, used only if
            baseline_spending=True
        p (OG-USA Specifications object): model parameters

    Returns:
        (tuple): fiscal variable path output:

            * D (Numpy array): government debt over the time path
            * G (Numpy array): government spending over the time path

    '''
    Y, total_revenue, TR, D0, G0 = dg_fixed_values

    D = np.zeros(p.T + 1)
    D[0] = D0
    if not p.baseline_spending:
        G = p.alpha_G[:p.T] * Y[:p.T]
        G[0] = G0
    else:
        G = Gbaseline[:p.T]
    growth = (1 + p.g_n) * np.exp(p.g_y)

    t = 1
    while t < p.T-1:
        D[t] = ((1 / growth[t]) * ((1 + r_gov[t - 1]) * D[t - 1] +
                                   G[t - 1] + TR[t - 1] -
                                   total_revenue[t - 1]))
        if (t >= p.tG1) and (t < p.tG2):
            G[t] = (growth[t + 1] * (p.rho_G * p.debt_ratio_ss * Y[t] +
                                     (1 - p.rho_G) * D[t]) -
                    (1 + r_gov[t]) * D[t] + total_revenue[t] - TR[t])
        elif t >= p.tG2:
            G[t] = (growth[t + 1] * (p.debt_ratio_ss * Y[t]) -
                    (1 + r_gov[t]) * D[t] + total_revenue[t] - TR[t])
        t += 1

    # in final period, growth rate has stabilized, so we can replace
    # growth[t+1] with growth[t]
    t = p.T - 1
    D[t] = ((1 / growth[t]) * ((1 + r_gov[t - 1]) * D[t - 1] + G[t - 1]
                               + TR[t - 1] - total_revenue[t - 1]))
    G[t] = (growth[t] * (p.debt_ratio_ss * Y[t]) - (1 + r_gov[t]) * D[t]
            + total_revenue[t] - TR[t])
    D[t + 1] = ((1 / growth[t + 1]) * ((1 + r_gov[t]) * D[t] + G[t] +
                                       TR[t] - total_revenue[t]))
    D_ratio_max = np.amax(D[:p.T] / Y[:p.T])
    print('Maximum debt ratio: ', D_ratio_max)

    return D, G


def get_r_gov(r, p):
    r'''
    Determine the interest rate on government debt

    .. math::
        r_{gov,t} = \max\{(1-\tau_{d,t}r_{t} - \mu_d, 0.0\}

    Args:
        r (array_like): interest rate on private capital debt over the
            time path or in the steady state
        p (OG-USA Specifications object): model parameters

    Returns:
        r_gov (array_like): interest rate on government debt over the
            time path or in the steady-state

    '''
    r_gov = np.maximum(p.r_gov_scale * r - p.r_gov_shift, 0.00)

    return r_gov
