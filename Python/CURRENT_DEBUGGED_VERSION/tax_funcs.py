'''
------------------------------------------------------------------------
Last updated 7/16/2015

Functions for taxes in SS and TPI.

------------------------------------------------------------------------
'''

# Packages
import numpy as np
import cPickle as pickle

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def replacement_rate_vals(nssmat, wss, factor_ss, e, J, omega_SS, lambdas):
    '''
    Calculates replacement rate values for the payroll tax.
    Inputs:
        nssmat = labor participation rate values (SxJ array or Sx1 array)
        wss = wage rate (scalar)
        factor_ss = factor that converts income to dollars (scalar)
        e = ability levels (SxJ array or Sx1 array)
        J = number of ability types (scalar)
        omega_SS = population weights by age (Sx1 array)
        lambdas = ability weights (Jx1 array or scalar)
    Outputs:
        theta = replacement rates for each ability type (Jx1 array)
    '''
    # Do a try/except, depending on whether the arrays are 1 or 2 dimensional 
    try:
        AIME = ((wss * factor_ss * e * nssmat)*omega_SS).sum(0) * lambdas / 12.0
        PIA = np.zeros(J)
        # Bins from data for each level of replacement
        for j in xrange(J):
            if AIME[j] < 749.0:
                PIA[j] = .9 * AIME[j]
            elif AIME[j] < 4517.0:
                PIA[j] = 674.1+.32*(AIME[j] - 749.0)
            else:
                PIA[j] = 1879.86 + .15*(AIME[j] - 4517.0)
        theta = PIA * (e * nssmat).mean(0) / AIME
        # Set the maximum replacment rate to be $30,000
        maxpayment = 30000.0/(factor_ss * wss)
        theta[theta > maxpayment] = maxpayment
    except:
        AIME = ((wss * factor_ss * e * nssmat)*omega_SS).sum() * lambdas / 12.0
        PIA = 0
        if AIME < 749.0:
            PIA = .9 * AIME
        elif AIME < 4517.0:
            PIA = 674.1+.32*(AIME - 749.0)
        else:
            PIA = 1879.86 + .15*(AIME - 4517.0)
        theta = PIA * (e * nssmat).mean(0) / AIME
        # Set the maximum replacment rate to be $30,000
        maxpayment = 30000.0/(factor_ss * wss)
        if theta > maxpayment:
            theta = maxpayment
    return theta


def tau_wealth(b, params):
    '''
    Calculates tau_wealth based on the wealth level for an individual
    Inputs:
        b = wealth holdings of an individual (various length arrays or scalar)
        params = parameter list of model
    Outputs:
        tau_w = tau_wealth (various length arrays or scalar)
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w = p * h * b / (h*b + m)
    return tau_w


def tau_w_prime(b, params):
    '''
    Calculates derivative of tau_wealth based on the wealth level for an individual
    Inputs:
        b = wealth holdings of an individual (various length arrays or scalar)
        params = parameter list of model (list)
    Outputs:
        tau_w_prime = derivative of tau_wealth (various length arrays or scalar)
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w_prime = h * m * p / (b*h + m) ** 2
    return tau_w_prime


def tau_income(r, b, w, e, n, factor, params):
    '''
    Gives income tax value for a certain income level
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        factor = scaling factor (scalar)
        params = parameter list of model (list)
    Output:
        tau = tau_income (various length array or scalar)
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    a = a_tax_income
    b = b_tax_income
    c = c_tax_income
    d = d_tax_income
    I = r * b + w * e * n
    I *= factor
    num = a * (I ** 2) + b * I
    denom = a * (I ** 2) + b * I + c
    tau = d * num / denom
    return tau


def tau_income_deriv(r, b, w, e, n, factor, params):
    '''
    Gives derivative of income tax value at a certain income level
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        factor = scaling factor (scalar)
        params = parameter list of model (list)
    Output:
        tau = derivative of tau_income (various length array or scalar)
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    a = a_tax_income
    b = b_tax_income
    c = c_tax_income
    d = d_tax_income
    I = r * b + w * e * n
    I *= factor
    denom = a * (I ** 2) + b * I + c
    num = (2 * a * I + b)
    tau = d * c * num / (denom ** 2)
    return tau


def get_lump_sum(r, b, w, e, n, BQ, lambdas, factor, weights, method, params, theta, tau_bq):
    '''
    Gives lump sum tax value.
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        BQ = Bequest values (various length array or scalar)
        lambdas = ability levels (Jx1 array or scalar)
        factor = scaling factor (scalar)
        weights = population weights (various length array or scalar)
        method = 'SS' or 'TPI', depending on the shape of arrays
        params = parameter list of model (list)
        theta = replacement rate values (Jx1 array or scalar)
        tau_bq = bequest tax values (Jx1 array or scalar)
    Output:
        T_H = lump sum tax (Tx1 array or scalar)
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor, params) * I
    T_P = tau_payroll * w * e * n
    T_W = tau_wealth(b, params) * b
    if method == 'SS':
        T_P[retire:] -= theta * w
        T_BQ = tau_bq * BQ / lambdas
        T_H = (weights * lambdas * (T_I + T_P + T_BQ + T_W)).sum()
    elif method == 'TPI':
        T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w
        T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
        T_H = (weights * lambdas * (T_I + T_P + T_BQ + T_W)).sum(1).sum(1)
    return T_H


def total_taxes(r, b, w, e, n, BQ, lambdas, factor, T_H, j, method, shift, params, theta, tau_bq):
    '''
    Gives net taxes values.
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        BQ = Bequest values (various length array or scalar)
        lambdas = ability levels (Jx1 array or scalar)
        factor = scaling factor (scalar)
        T_H = net taxes (Tx1 array or scalar)
        j = Which ability level is being computed, if doing one ability level at a time (scalar)
        method = 'SS' or 'TPI' or 'TPI_scalar', depending on the shape of arrays
        shift = Computing for periods 0--s or 1--(s+1) (bool) (True for 1--(s+1))
        params = parameter list of model (list)
        theta = replacement rate values (Jx1 array or scalar)
        tau_bq = bequest tax values (Jx1 array or scalar)
    Output:
        total_taxes = net taxes (various length array or scalar)
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor, params) * I
    T_P = tau_payroll * w * e * n
    T_W = tau_wealth(b, params) * b
    if method == 'SS':
        # Depending on if we are looking at b_s or b_s+1, the 
        # entry for retirement will change (it shifts back one).
        # The shift boolean makes sure we start replacement rates
        # at the correct age.
        if shift is False:
            T_P[retire:] -= theta * w
        else:
            T_P[retire-1:] -= theta * w
        T_BQ = tau_bq * BQ / lambdas
    elif method == 'TPI':
        if shift is False:
            # retireTPI is different from retire, because in TPI we are counting backwards
            # with different length lists.  This will always be the correct location
            # of retirement, depending on the shape of the lists.
            retireTPI = (retire - S)
        else:
            retireTPI = (retire-1 - S)
        if len(b.shape) != 3:
            T_P[retireTPI:] -= theta[j] * w[retireTPI:]
            T_BQ = tau_bq[j] * BQ / lambdas
        else:
            T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w
            T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
    elif method == 'TPI_scalar':
        # The above methods won't work if scalars are used.  This option is only called by the
        # SS_TPI_firstdoughnutring function in TPI.
        T_P -= theta[j] * w
        T_BQ = tau_bq[j] * BQ / lambdas
    total_taxes = T_I + T_P + T_BQ + T_W - T_H
    return total_taxes
