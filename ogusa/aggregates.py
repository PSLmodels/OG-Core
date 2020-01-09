B'''
------------------------------------------------------------------------
Functions to compute economic aggregates.
------------------------------------------------------------------------
'''

# Packages
import numpy as np
from ogusa import tax, utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_L(n, p, method):
    r'''
    Calculate aggregate labor supply.

    .. math::
        L_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}\lambda_{j}n_{j,s,t}

    Args:
        n (Numpy array): labor supply of households
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        L (array_like): aggregate labor supply

    '''
    if method == 'SS':
        L_presum = p.e * np.transpose(p.omega_SS * p.lambdas) * n
        L = L_presum.sum()
    elif method == 'TPI':
        L_presum = ((n * (p.e * np.squeeze(p.lambdas))) *
                    np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                            (1, 1, p.J)))
        L = L_presum.sum(1).sum(1)
    return L


def get_I(b_splus1, K_p1, K, p, method):
    r'''
    Calculate aggregate investment.

    .. math::
        I_{t} = (1 + g_{n,t+1})e^{g_{y}}(K_{t+1} - \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s+1,t}i_{s+1,t}\lambda_{j}b_{j,s+1,t+1} \ (1+ g_{n,t+1})) - (1 - \delta)K_{t}

    Args:
        b_splus1 (Numpy array): savings of households
        K_p1 (array_like): aggregate capital, one period ahead
        K (array_like): aggregate capital
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        aggI (array_like): aggregate investment

    '''
    if method == 'SS':
        omega_extended = np.append(p.omega_SS[1:], [0.0])
        imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
        part2 = (((b_splus1 *
                   np.transpose((omega_extended * imm_extended) *
                                p.lambdas)).sum()) / (1 + p.g_n_ss))
        aggI = ((1 + p.g_n_ss) * np.exp(p.g_y) * (K_p1 - part2) -
                (1.0 - p.delta) * K)
    if method == 'BI_SS':
        delta = 0
        omega_extended = np.append(p.omega_SS[1:], [0.0])
        imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
        part2 = (((b_splus1 *
                   np.transpose((omega_extended * imm_extended) *
                                p.lambdas)).sum()) / (1 + p.g_n_ss))
        aggI = ((1 + p.g_n_ss) * np.exp(p.g_y) * (K_p1 - part2) -
                (1.0 - delta) * K)
    elif method == 'TPI':
        omega_shift = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
                                axis=1)
        imm_shift = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
                              axis=1)
        part2 = ((((b_splus1 * np.squeeze(p.lambdas)) *
                   np.tile(np.reshape(imm_shift * omega_shift,
                                      (p.T, p.S, 1)),
                           (1, 1, p.J))).sum(1).sum(1)) /
                 (1 + np.squeeze(np.hstack((p.g_n[1:p.T], p.g_n_ss)))))
        aggI = ((1 + np.squeeze(np.hstack((p.g_n[1:p.T], p.g_n_ss)))) *
                np.exp(p.g_y) * (K_p1 - part2) - (1.0 - p.delta) * K)

    return aggI


def get_B(b, p, method, preTP):
    r'''
    Calculate aggregate savings

    .. math::
        B_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}\lambda_{j}b_{j,s,t}

    Args:
        b (Numpy array): savings of households
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        preTP (bool): whether calculation is for the pre-time path
            period amount of savings.  If True, then need to use
            `omega_S_preTP`.

    Returns:
        B (array_like): aggregate supply of savings

    '''
    if method == 'SS':
        if preTP:
            part1 = b * np.transpose(p.omega_S_preTP * p.lambdas)
            omega_extended = np.append(p.omega_S_preTP[1:], [0.0])
            imm_extended = np.append(p.imm_rates[0, 1:], [0.0])
            pop_growth_rate = p.g_n[0]
        else:
            part1 = b * np.transpose(p.omega_SS * p.lambdas)
            omega_extended = np.append(p.omega_SS[1:], [0.0])
            imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
            pop_growth_rate = p.g_n_ss
        part2 = b * np.transpose(omega_extended * imm_extended * p.lambdas)
        B_presum = part1 + part2
        B = B_presum.sum()
        B /= (1.0 + pop_growth_rate)
    elif method == 'TPI':
        part1 = ((b * np.squeeze(p.lambdas)) *
                 np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                         (1, 1, p.J)))
        omega_shift = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
                                axis=1)
        imm_shift = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
                              axis=1)
        part2 = ((b * np.squeeze(p.lambdas)) *
                 np.tile(np.reshape(imm_shift * omega_shift,
                                    (p.T, p.S, 1)), (1, 1, p.J)))
        B_presum = part1 + part2
        B = B_presum.sum(1).sum(1)
        B /= (1.0 + np.hstack((p.g_n[1:p.T], p.g_n_ss)))
    return B


def get_BQ(r, b_splus1, j, p, method, preTP):
    r'''
    Calculation of aggregate bequests.  If `use_zeta` is False, then
    computes aggregate bequests within each lifetime income group.

    .. math::
        BQ_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\rho_{s}\omega_{s,t}\lambda_{j}b_{j,s+1,1}

    Args:
        r (array_like): the real interest rate
        b_splus1 (numpy array): household savings one period ahead
        j (int): index of lifetime income group
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on SS or
            TPI
        preTP (bool): whether calculation is for the pre-time path
            period amount of savings.  If True, then need to use
            `omega_S_preTP`.

    Returns:
        BQ (array_like): aggregate bequests, overall or by lifetime
            income group, depending on `use_zeta` value.

    '''
    if method == 'SS':
        if preTP:
            omega = p.omega_S_preTP
            pop_growth_rate = p.g_n[0]
        else:
            omega = p.omega_SS
            pop_growth_rate = p.g_n_ss
        if j:
            BQ_presum = omega * p.rho * b_splus1 * p.lambdas[j]
        else:
            BQ_presum = (np.transpose(omega * (p.rho * p.lambdas)) *
                         b_splus1)
        BQ = BQ_presum.sum(0)
        BQ *= (1.0 + r) / (1.0 + pop_growth_rate)
    elif method == 'TPI':
        pop = np.append(p.omega_S_preTP.reshape(1, p.S),
                        p.omega[:p.T - 1, :], axis=0)
        if j:
            BQ_presum = ((b_splus1 * p.lambdas[j]) *
                         (pop * p.rho))
            BQ = BQ_presum.sum(1)
            BQ *= (1.0 + r) / (1.0 + p.g_n[:p.T])
        else:
            BQ_presum = ((b_splus1 * np.squeeze(p.lambdas)) *
                         np.tile(np.reshape(pop * p.rho, (p.T, p.S, 1)),
                                 (1, 1, p.J)))
            BQ = BQ_presum.sum(1)
            BQ *= np.tile(np.reshape((1.0 + r) / (1.0 + p.g_n[:p.T]),
                                     (p.T, 1)), (1, p.J))
    if p.use_zeta:
        if method == 'SS':
            BQ = BQ.sum()
        else:
            if not j:
                BQ = BQ.sum(1)
    return BQ


def get_C(c, p, method):
    r'''
    Calculation of aggregate consumption.

    .. math::
        C_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}\lambda_{j}c_{j,s,t}

    Args:
        c (Numpy array): consumption of households
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        C (array_like): aggregate consumption

    '''

    if method == 'SS':
        aggC = (c * np.transpose(p.omega_SS * p.lambdas)).sum()
    elif method == 'TPI':
        aggC = ((c * np.squeeze(p.lambdas)) *
                np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                        (1, 1, p.J))).sum(1).sum(1)
    return aggC


def revenue(r, w, b, n, bq, c, Y, L, K, factor, theta, etr_params,
            p, method):
    r'''
    Calculate aggregate tax revenue.

    .. math::
        R_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}\lambda_{j}(T_{j,s,t} + \tau^{p}_{t}w_{t}e_{j,s}n_{j,s,t} - \theta_{j}w_{t} + \tau^{bq}bq_{j,s,t} + \tau^{c}_{s,t}c_{j,s,t} + \tau^{w}_{t}b_{j,s,t}) + \tau^{b}_{t}(Y_{t}-w_{t}L_{t}) - \tau^{b}_{t}\delta^{\tau}_{t}K^{\tau}_{t}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        b (Numpy array): household savings
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        c (Numpy array): household consumption
        Y (array_like): aggregate output
        L (array_like): aggregate labor
        K (array_like): aggregate capital
        factor (scalar): factor (scalar): scaling factor converting
            model units to dollars
        theta (Numpy array): social security replacement rate for each
            lifetime income group
        etr_params (Numpy array): paramters of the effective tax rate
            functions
        p (OG-USA Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        REVENUE (array_like): aggregate tax revenue
        T_I (array_like): aggregate income tax revenue
        T_P (array_like): aggregate net payroll tax revenue, revenues
            minus social security benefits paid
        T_BQ (array_like): aggregate bequest tax revenue
        T_W (array_like): aggregate wealth tax revenue
        T_C (array_like): aggregate consumption tax revenue
        business_revenue (array_like): aggregate business tax revenue

    '''
    if method == 'SS':
        I = r * b + w * p.e * n
        T_I = np.zeros_like(I)
        T_I = tax.ETR_income(r, w, b, n, factor, p.e, etr_params, p) * I
        T_P = p.tau_payroll[-1] * w * p.e * n
        T_P[p.retire[-1]:] -= theta * w
        T_W = (tax.ETR_wealth(b, p.h_wealth[-1], p.m_wealth[-1],
                              p.p_wealth[-1]) * b)
        T_BQ = p.tau_bq[-1] * bq
        T_C = p.tau_c[-1, :, :] * c
        business_revenue = tax.get_biz_tax(w, Y, L, K, p, method)
        REVENUE = ((np.transpose(p.omega_SS * p.lambdas) *
                    (T_I + T_P + T_BQ + T_W + T_C)).sum() +
                   business_revenue)
    elif method == 'TPI':
        r_array = utils.to_timepath_shape(r)
        w_array = utils.to_timepath_shape(w)
        I = r_array * b + w_array * n * p.e
        T_I = np.zeros_like(I)
        T_I = tax.ETR_income(r_array, w_array, b, n, factor, p.e,
                             etr_params, p) * I
        T_P = p.tau_payroll[:p.T].reshape(p.T, 1, 1) * w_array * n * p.e
        for t in range(T_P.shape[0]):
            T_P[t, p.retire[t]:, :] -= (theta.reshape(1, p.J) *
                                        p.replacement_rate_adjust[t] *
                                        w_array[t])
        T_W = (tax.ETR_wealth(b, p.h_wealth[:p.T].reshape(p.T, 1, 1),
                              p.m_wealth[:p.T].reshape(p.T, 1, 1),
                              p.p_wealth[:p.T].reshape(p.T, 1, 1)) * b)
        T_BQ = p.tau_bq[:p.T].reshape(p.T, 1, 1) * bq
        T_C = p.tau_c[:p.T, :, :] * c
        business_revenue = tax.get_biz_tax(w, Y, L, K, p, method)
        REVENUE = ((((np.squeeze(p.lambdas)) *
                   np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                           (1, 1, p.J)))
                   * (T_I + T_P + T_BQ + T_W + T_C)).sum(1).sum(1) +
                   business_revenue)

    return REVENUE, T_I, T_P, T_BQ, T_W, T_C, business_revenue


def get_r_hh(r, r_gov, K, D):
    r'''
    Compute the interest rate on the household's portfolio of assets,
    a mix of government debt and private equity.

    .. math::
        r_{hh,t} = \frac{r_{gov,t}D_{t} + r_{t}K_{t}}{D_{t} + K_{t}}

    Args:
        r (array_like): the real interest rate
        r_gov (array_like): the real interest rate on government debt
        K (array_like): aggregate capital
        D (array_like): aggregate government debt

    Returns:
        r_hh (array_like): the real interest rate on the households
            portfolio

    '''
    r_hh = ((r * K) + (r_gov * D)) / (K + D)

    return r_hh


def resource_constraint(Y, C, G, I, K_f, new_borrowing_f,
                        debt_service_f, r, p):
    r'''
    Compute the error in the resource constraint.

    .. math::
        \hat{Y}_{t} = \hat{C}_{t} + (\hat{K}^{d}_{t+1}e^{g_{y}}(1+g_{n,t+1}) - \hat{K}^{d}_{t}) + \delta \hat{K}_{t} +  \hat{G}_{t} + r_{hh, t}\hat{K}^{f}_{t} - (\hat{D}^{f}_{t+1}e^{g_{y}}(1+g_{n,t+1})- \hat{D}^{f}_{t}) + r_{hh,t}\hat{D}^{f}_{t}

    Args:
        Y (array_like): aggregate output
        C (array_like): aggregate consumption
        G (array_like): aggregate government spending
        I (array_like): aggregate investment
        K_f (array_like): aggregate capital that is foreign-owned
        new_borrowing_f (array_like): new borrowing of government debt
            from foreign investors
        debt_service_f (array_like): interest payments on government
            debt owned by foreigners
        r (array_like): the real interest rate
        p (OG-USA Specifications object): model parameters

    Returns:
        rc_error (array_like): error in the resource constraint

    '''
    rc_error = (Y - C - I - G - (r + p.delta) * K_f + new_borrowing_f -
                debt_service_f)

    return rc_error
