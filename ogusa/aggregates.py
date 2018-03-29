'''
------------------------------------------------------------------------
Last updated 8/15/2017

Functions to compute economic aggregates.

------------------------------------------------------------------------
'''

# Packages
import numpy as np
from . import tax

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

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
        omega_extended = np.append(omega[1:], [0.0])
        imm_extended = np.append(imm_rates[1:], [0.0])
        part2 = ((((b_splus1 *
                   (omega_extended*imm_extended).reshape(omega.shape[0], 1)) *
                   lambdas).sum())/(1+g_n))
        aggI = (1+g_n)*np.exp(g_y)*(K_p1 - part2) - (1.0 - delta) * K
    elif method == 'TPI':
        # omega_extended = np.append(omega[1:,:,:],np.zeros((1,omega.shape[1],omega.shape[2])),axis=0)
        # imm_extended = np.append(imm_rates[1:,:,:],np.zeros((1,imm_rates.shape[1],imm_rates.shape[2])),axis=0)
        # part2 = ((b_splus1*omega_extended*imm_extended*lambdas).sum(1).sum(1))/(1+g_n)
        omega_shift = np.append(omega[:, 1:, :],
                                np.zeros((omega.shape[0], 1, omega.shape[2]))
                                , axis=1)
        imm_shift = np.append(imm_rates[:, 1:, :],
                              np.zeros((imm_rates.shape[0],
                                       1, imm_rates.shape[2])),
                              axis=1)
        part2 = (((b_splus1*imm_shift*omega_shift*lambdas).sum(1).sum(1)) /
                 (1+g_n))
        aggI = (1+g_n)*np.exp(g_y)*(K_p1 - part2) - (1.0 - delta) * K

    return aggI

def get_K(b, params):
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
        K_presum = [T,S,J] array, weighted distribution of wealth/capital holdings
        K        = [T,] vector, aggregate capital supply

    Returns: K
    '''

    omega, lambdas, imm_rates, g_n, method = params

    if method == 'SS':
        part1 = b* omega * lambdas
        omega_extended = np.append(omega[1:],[0.0])
        imm_extended = np.append(imm_rates[1:],[0.0])
        part2 = b*(omega_extended*imm_extended).reshape(omega.shape[0],1)*lambdas
        K_presum = part1+part2
        K = K_presum.sum()
    elif method == 'TPI':
        part1 = b* omega * lambdas
        #omega_extended = np.append(omega[1:,:,:],np.zeros((1,omega.shape[1],omega.shape[2])),axis=0)
        omega_shift = np.append(omega[:,1:,:],np.zeros((omega.shape[0],1,omega.shape[2])),axis=1)
        #imm_extended = np.append(imm_rates[1:,:,:],np.zeros((1,imm_rates.shape[1],imm_rates.shape[2])),axis=0)
        imm_shift = np.append(imm_rates[:,1:,:],np.zeros((imm_rates.shape[0],1,imm_rates.shape[2])),axis=1)
        #part2 = b*(omega_extended*imm_extended)*lambdas
        part2 = b*imm_shift*omega_shift*lambdas
        K_presum = part1+part2
        K = K_presum.sum(1).sum(1)
    K /= (1.0 + g_n)
    return K


def get_BQ(r, b_splus1, params):
    '''
    Calculation of bequests to each lifetime income group.

    Inputs:
        r           = [T,] vector, interest rates
        b_splus1    = [T,S,J] array, distribution of wealth/capital holdings one period ahead
        params      = length 5 tuple, (omega, lambdas, rho, g_n, method)
        omega       = [S,T] array, population weights
        lambdas     = [J,] vector, fraction in each lifetime income group
        rho         = [S,] vector, mortality rates
        g_n         = scalar, population growth rate
        method      = string, 'SS' or 'TPI'

    Functions called: None

    Objects in function:
        BQ_presum = [T,S,J] array, weighted distribution of wealth/capital holdings one period ahead
        BQ        = [T,J] array, aggregate bequests by lifetime income group

    Returns: BQ
    '''
    omega, lambdas, rho, g_n, method = params

    BQ_presum = b_splus1 * omega * rho * lambdas
    if method == 'SS':
        BQ = BQ_presum.sum(0)
    elif method == 'TPI':
        BQ = BQ_presum.sum(1)
    BQ *= (1.0 + r) / (1.0 + g_n)
    return BQ


def get_C(c, params):
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

    omega, lambdas, method = params

    aggC_presum = c * omega * lambdas
    if method == 'SS':
        aggC = aggC_presum.sum()
    elif method == 'TPI':
        aggC = aggC_presum.sum(1).sum(1)
    return aggC


def revenue(r, w, b, n, BQ, Y, L, K, factor, params):
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
        theta       = [J,] vector, replacement rate values by lifetime income group
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
        tau_income
        tau_wealth
    Objects in function:
        I    = [T,S,J] array, total income
        T_I  = [T,S,J] array, total income taxes
        T_P  = [T,S,J] array, total payroll taxes
        T_W  = [T,S,J] array, total wealth taxes
        T_BQ = [T,S,J] array, total bequest taxes
        T_H  = [T,] vector, lump sum transfer amount(s)
    Returns: T_H

    '''

    e, lambdas, omega, method, etr_params, tax_func_type, theta, tau_bq, \
        tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J,\
        tau_b, delta_tau = params

    I = r * b + w * e * n

    if I.ndim == 2:
        T_I = np.zeros((S,J))
        for j in xrange(J):
            TI_params = (e[:,j], etr_params, tax_func_type)
            T_I[:,j] = tax.tau_income(r, w, b[:,j], n[:,j], factor, TI_params) * I[:,j]
    if I.ndim == 3:
        T_I = np.zeros((T,S,J))
        for j in xrange(J):
            if etr_params.ndim == 3:
                tau_inc_params3D = etr_params[:,j,:]
            if etr_params.ndim == 4:
                tau_inc_params3D = etr_params[:,:,j,:]
            TI_params = (e[:,:,j], tau_inc_params3D, tax_func_type)
            T_I[:,:,j] = tax.tau_income(r[:,:,j], w[:,:,j], b[:,:,j], n[:,:,j], factor, TI_params) * I[:,:,j]
    T_P = tau_payroll * w * e * n
    TW_params = (h_wealth, p_wealth, m_wealth)
    T_W = tax.tau_wealth(b, TW_params) * b
    if method == 'SS':
        T_P[retire:] -= theta * w
        T_BQ = tau_bq * BQ / lambdas
        biz_params = (tau_b, delta_tau)
        business_revenue = tax.get_biz_tax(w, Y, L, K, biz_params)
        REVENUE = (omega * lambdas * (T_I + T_P + T_BQ + T_W)).sum() + business_revenue
    elif method == 'TPI':
        T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w[:,retire:,:]
        T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
        biz_params = (tau_b, delta_tau)
        business_revenue = tax.get_biz_tax(w[:T,0,0], Y, L, K, biz_params)
        REVENUE = (omega * lambdas * (T_I + T_P + T_BQ + T_W)).sum(1).sum(1) + business_revenue
    return REVENUE
