'''
------------------------------------------------------------------------
Functions to compute economic aggregates.
------------------------------------------------------------------------
'''

# Packages
import numpy as np
from ogusa import tax, household, utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_L(n, p, method):
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


def get_K(b, p, method, preTP):
    '''
    Calculates aggregate capital supplied.

    Inputs:
        b           = [T,S,J] array, distribution of wealth/capital holdings
        params      = length 4 tuple, (omega, lambdas, g_n, method)
        omega       = [S,T] array, population weights
        lambdas     = [J,] vector, fraction in each lifetime income group
        g_n         = [T,] vector, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        K_presum = [T,S,J] array, weighted distribution of wealth/capital
                   holdings
        K        = [T,] vector, aggregate capital supply

    Returns: K
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
        K_presum = part1 + part2
        K = K_presum.sum()
        K /= (1.0 + pop_growth_rate)
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
        K_presum = part1 + part2
        K = K_presum.sum(1).sum(1)
        K /= (1.0 + np.hstack((p.g_n[1:p.T], p.g_n_ss)))
    return K


def get_BQ(r, b_splus1, j, p, method, preTP):
    '''
    Calculation of bequests to each lifetime income group.

    Inputs:
        r           = [T,] vector, interest rates
        b_splus1    = [T,S,J] array, distribution of wealth/capital
                      holdings one period ahead
        params      = length 5 tuple, (omega, lambdas, rho, g_n, method)
        omega       = [S,T] array, population weights
        lambdas     = [J,] vector, fraction in each lifetime income group
        rho         = [S,] vector, mortality rates
        g_n         = scalar, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        BQ_presum = [T,S,J] array, weighted distribution of
                    wealth/capital holdings one period ahead
        BQ        = [T,J] array, aggregate bequests by lifetime income group

    Returns: BQ
    '''
    if method == 'SS':
        if preTP:
            omega = p.omega_S_preTP
            pop_growth_rate = p.g_n[0]
        else:
            omega = p.omega_SS
            pop_growth_rate = p.g_n_ss
        if j is not None:
            BQ_presum = omega * p.rho * b_splus1 * p.lambdas[j]
        else:
            BQ_presum = (np.transpose(omega * (p.rho * p.lambdas)) *
                         b_splus1)
        BQ = BQ_presum.sum(0)
        BQ *= (1.0 + r) / (1.0 + pop_growth_rate)
    elif method == 'TPI':
        pop = np.append(p.omega_S_preTP.reshape(1, p.S),
                        p.omega[:p.T - 1, :], axis=0)
        if j is not None:
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
            BQ = BQ.sum(1)
    return BQ


def get_C(c, p, method):
    '''
    Calculation of aggregate consumption.

    Inputs:
        cons        = [T,S,J] array, household consumption
        params      = length 3 tuple (omega, lambdas, method)
        omega       = [S,T] array, population weights by age (Sx1 array)
        lambdas     = [J,1] vector, lifetime income group weights
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        aggC_presum = [T,S,J] array, weighted consumption by household
        aggC        = [T,] vector, aggregate consumption

    Returns: aggC
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
    '''
    Gives lump sum transfer value.
    Inputs:
        r           = [T,] vector, interest rate
        w           = [T,] vector, wage rate
        b           = [T,S,J] array, wealth holdings
        n           = [T,S,J] array, labor supply
        BQ          = [T,J] array, bequest amounts
        factor      = scalar, model income scaling factor
        params      = length 12 tuple, (e, lambdas, omega, method, etr_params,
                                        theta, tau_bq, tau_payroll, h_wealth,
                                        p_wealth, m_wealth, retire, T, S, J)
        e           = [T,S,J] array, effective labor units
        lambdas     = [J,] vector, population weights by lifetime income group
        omega       = [T,S] array, population weights by age
        method      = string, 'SS' or 'TPI'
        etr_params  = [T,S,J] array, effective tax rate function parameters
        tax_func_types = string, type of tax function used
        theta       = [J,] vector, replacement rate values by lifetime
                      income group
        tau_bq      = scalar, bequest tax rate
        h_wealth    = scalar, wealth tax function parameter
        p_wealth    = scalar, wealth tax function parameter
        m_wealth    = scalar, wealth tax function parameter
        tau_payroll = scalar, payroll tax rate
        retire      = integer, retirement age
        T           = integer, number of periods in transition path
        S           = integer, number of age groups
        J           = integer, number of lifetime income groups
    Functions called:
        ETR_income
        ETR_wealth
    Objects in function:
        I    = [T,S,J] array, total income
        T_I  = [T,S,J] array, total income taxes
        T_P  = [T,S,J] array, total payroll taxes
        T_W  = [T,S,J] array, total wealth taxes
        T_BQ = [T,S,J] array, total bequest taxes
        T_H  = [T,] vector, lump sum transfer amount(s)
    Returns: T_H

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
        r_array = utils.to_timepath_shape(r, p)
        w_array = utils.to_timepath_shape(w, p)
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
    '''
    Compute the interest rate on the household's portfolio of assets,
    a mix of government debt and private equity
    '''
    r_hh = ((r * K) + (r_gov * D)) / (K + D)

    return r_hh


def resource_constraint(Y, C, G, I, K_f, new_borrowing_f,
                        debt_service_f, r, p):
    '''
    Compute the error in the resource constraint
    '''
    rc_error = (Y - C - I - G - (r + p.delta) * K_f + new_borrowing_f -
                debt_service_f)

    return rc_error
