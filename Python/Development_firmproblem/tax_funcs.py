'''
------------------------------------------------------------------------
Last updated 6/19/2015

Functions for taxes in SS and TPI.

------------------------------------------------------------------------
'''

# Packages
import numpy as np
import cPickle as pickle

'''
------------------------------------------------------------------------
Tax functions
------------------------------------------------------------------------
    The first function gets the replacement rate values for the payroll
        tax.  The next 4 functions are the wealth and income tax functions,
        with their derivative functions.  The remaining functions
        are used to get the total amount of taxes and lump sum taxes.
------------------------------------------------------------------------
'''


def replacement_rate_vals(nssmat, wss, factor_ss, e, J, omega_SS, lambdas):
    # Import data need to compute replacement rates, outputed from SS.py
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
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w = p * h * b / (h*b + m)
    return tau_w


def tau_w_prime(b, params):
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w_prime = h * m * p / (b*h + m) ** 2
    return tau_w_prime


def tau_income(r, b, w, e, n, factor, params):
    '''
    Gives income tax value at a
    certain income level
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
    Gives derivative of income tax value at a
    certain income level
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
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, g_n_ss, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor, params) * I
    T_P = tau_payroll * w * e * n
    T_W = tau_wealth(b, params) * b
    if method == 'SS':
        if shift is False:
            T_P[retire:] -= theta * w
        else:
            T_P[retire-1:] -= theta * w
        T_BQ = tau_bq * BQ / lambdas
    elif method == 'TPI':
        if shift is False:
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
        T_P -= theta[j] * w
        T_BQ = tau_bq[j] * BQ / lambdas
    total_taxes = T_I + T_P + T_BQ + T_W - T_H
    return total_taxes
