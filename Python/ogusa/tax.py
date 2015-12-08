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
        AIME = ((wss * factor_ss * e * nssmat) *
                omega_SS).sum(0) * lambdas / 12.0
        PIA = np.zeros(J)
        # Bins from data for each level of replacement
        for j in xrange(J):
            if AIME[j] < 749.0:
                PIA[j] = .9 * AIME[j]
            elif AIME[j] < 4517.0:
                PIA[j] = 674.1 + .32 * (AIME[j] - 749.0)
            else:
                PIA[j] = 1879.86 + .15 * (AIME[j] - 4517.0)
        theta = PIA * (e * nssmat).mean(0) / AIME
        # Set the maximum replacment rate to be $30,000
        maxpayment = 30000.0 / (factor_ss * wss)
        theta[theta > maxpayment] = maxpayment
    except:
        AIME = ((wss * factor_ss * e * nssmat) *
                omega_SS).sum() * lambdas / 12.0
        PIA = 0
        if AIME < 749.0:
            PIA = .9 * AIME
        elif AIME < 4517.0:
            PIA = 674.1 + .32 * (AIME - 749.0)
        else:
            PIA = 1879.86 + .15 * (AIME - 4517.0)
        theta = PIA * (e * nssmat).mean(0) / AIME
        # Set the maximum replacment rate to be $30,000
        maxpayment = 30000.0 / (factor_ss * wss)
        if theta > maxpayment:
            theta = maxpayment
    theta = 0 
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
    h_wealth, p_wealth, m_wealth = params
    
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w = p * h * b / (h * b + m)
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
    h_wealth, p_wealth, m_wealth = params

    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w_prime = h * m * p / (b * h + m) ** 2
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
    a_etr_income, b_etr_income, c_etr_income, d_etr_income, e_etr_income, f_etr_income,\
        min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income = params
    A = a_etr_income
    B = b_etr_income
    C = c_etr_income
    D = d_etr_income
    E = d_etr_income
    F = d_etr_income
    min_x = min_x_etr_income
    max_x = max_x_etr_income
    min_y = min_y_etr_income
    max_y = max_y_etr_income
    x = (w*e*n)*factor
    y = (r*b)*factor
    I = x+y

    phi = x/I
    Phi = phi*(max_x-min_x) + (1-phi)*(max_y-min_y)
    K = phi*min_x + (1-phi)*min_y

    num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
    denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
    tau =  (Phi*(num/denom)) + K
    return tau


def tau_capital_deriv(r, b, w, e, n, factor, params):
    '''
    Gives derivative of AETR function with repect to 
    capital income at a certain income level
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        factor = scaling factor (scalar)
        params = parameter list of model (list)
    Output:
        tau = derivative of tau_income w.r.t. capital income (various length array or scalar)
    '''
    a_tax_income, b_tax_income, c_tax_income, d_tax_income, e_tax_income, f_tax_income, \
        min_x_tax_income, max_x_tax_income, min_y_tax_income, max_y_tax_income = params
    A = a_tax_income
    B = b_tax_income
    C = c_tax_income
    D = d_tax_income
    E = d_tax_income
    F = d_tax_income
    min_x = min_x_tax_income
    max_x = max_x_tax_income
    min_y = min_y_tax_income
    max_y = max_y_tax_income
    x = (w*e*n)*factor
    y = (r*b)*factor

    num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
    denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
    Lambda = num/denom

    Lambda_deriv = ((2*B*y + C*x + E)*F)/(denom**2)

    tau =  ((max_y-min_y)*Lambda) + ((x*(max_x-min_x))+(y*(max_y-min_y)))*Lambda_deriv + min_y 

    return tau


## Note that since when we use the same functional form, one could
# use just one tax function for ATR, MTR_lab, MTR_cap, just with different parameters input
def MTR_capital(r, b, w, e, n, factor, params):
    '''
    Gives derivative of MTR function with repect to 
    labor income at a certain income level
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        factor = scaling factor (scalar)
        params = parameter list of model (list)
    Output:
        tau = derivative of tau_income w.r.t. labor income (various length array or scalar)
    '''
    a_mtry_income, b_mtry_income, c_mtry_income, d_mtry_income, e_mtry_income, f_mtry_income,\
                      min_x_mtry_income, max_x_mtry_income, min_y_mtry_income, max_y_mtry_income = params
    A = a_mtry_income
    B = b_mtry_income
    C = c_mtry_income
    D = d_mtry_income
    E = d_mtry_income
    F = d_mtry_income
    min_x = min_x_mtry_income
    max_x = max_x_mtry_income
    min_y = min_y_mtry_income
    max_y = max_y_mtry_income
    x = (w*e*n)*factor
    y = (r*b)*factor
    I = x+y

    phi = x/I
    Phi = phi*(max_x-min_x) + (1-phi)*(max_y-min_y)
    K = phi*min_x + (1-phi)*min_y

    num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
    denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
    tau =  (Phi*(num/denom)) + K
    return tau


def MTR_labor(r, b, w, e, n, factor, params):
    '''
    Gives derivative of MTR function with repect to 
    labor income at a certain income level
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        factor = scaling factor (scalar)
        params = parameter list of model (list)
    Output:
        tau = derivative of tau_income w.r.t. labor income (various length array or scalar)
    '''
    a_mtrx_income, b_mtrx_income, c_mtrx_income, d_mtrx_income, e_mtrx_income, f_mtrx_income,\
                      min_x_mtrx_income, max_x_mtrx_income, min_y_mtrx_income, max_y_mtrx_income = params
    A = a_mtrx_income
    B = b_mtrx_income
    C = c_mtrx_income
    D = d_mtrx_income
    E = d_mtrx_income
    F = d_mtrx_income
    min_x = min_x_mtrx_income
    max_x = max_x_mtrx_income
    min_y = min_y_mtrx_income
    max_y = max_y_mtrx_income
    x = (w*e*n)*factor
    y = (r*b)*factor
    I = x+y

    phi = x/I
    Phi = phi*(max_x-min_x) + (1-phi)*(max_y-min_y)
    K = phi*min_x + (1-phi)*min_y

    num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
    denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
    tau =  (Phi*(num/denom)) + K
    return tau


def tau_labor_deriv(r, b, w, e, n, factor, params):
    '''
    Gives derivative of AETR function with repect to 
    labor income at a certain income level
    Inputs:
        r = interest rate (various length list or scalar)
        b = wealth holdings (various length array or scalar)
        w = wage (various length list or scalar)
        e = ability level (various size array or scalar)
        n = labor participation rate (various length array or scalar)
        factor = scaling factor (scalar)
        params = parameter list of model (list)
    Output:
        tau = derivative of tau_income w.r.t. labor income (various length array or scalar)
    '''
    a_tax_income, b_tax_income, c_tax_income, d_tax_income, e_tax_income, f_tax_income,\
                      min_x_tax_income, max_x_tax_income, min_y_tax_income, max_y_tax_income = params
    A = a_tax_income
    B = b_tax_income
    C = c_tax_income
    D = d_tax_income
    E = d_tax_income
    F = d_tax_income
    min_x = min_x_tax_income
    max_x = max_x_tax_income
    min_y = min_y_tax_income
    max_y = max_y_tax_income
    x = (w*e*n)*factor
    y = (r*b)*factor

    num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
    denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
    Lambda = num/denom

    Lambda_deriv = ((2*A*x + C*y + D)*F)/(denom**2)

    tau =  ((max_x-min_x)*Lambda) + ((x*(max_x-min_x))+(y*(max_y-min_y)))*Lambda_deriv + min_x 
    return tau



def get_lump_sum(r, b, w, e, n, BQ, lambdas, factor, weights, method, tax_params, params, theta, tau_bq):
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
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params

    a_etr_income, b_etr_income, \
        c_etr_income, d_etr_income, e_etr_income, f_etr_income, \
        min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income = tax_params

    I = r * b + w * e * n
    tau_inc_params = (a_etr_income, b_etr_income, c_etr_income, d_etr_income, e_etr_income, f_etr_income,
                      min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income)
    
    if I.ndim == 2: 
        T_I = np.zeros((S,J))
        for j in xrange(J):
            T_I[:,j] = tau_income(r, b[:,j], w, e[:,j], n[:,j], factor, tau_inc_params) * I[:,j]
    if I.ndim == 3:
        T_I = np.zeros((T,S,J))
        for j in xrange(J):
            tau_inc_params3D = (a_etr_income[:,:,j], b_etr_income[:,:,j], c_etr_income[:,:,j], d_etr_income[:,:,j], 
                                e_etr_income[:,:,j], f_etr_income[:,:,j], min_x_etr_income[:,:,j], max_x_etr_income[:,:,j], 
                                min_y_etr_income[:,:,j], max_y_etr_income[:,:,j])
            T_I[:,:,j] = tau_income(r[:,:,j], b[:,:,j], w[:,:,j], e[:,:,j], n[:,:,j], factor, tau_inc_params3D) * I[:,:,j]  
    T_P = tau_payroll * w * e * n
    TW_params = (h_wealth, p_wealth, m_wealth)
    T_W = tau_wealth(b, TW_params) * b
    if method == 'SS':
        T_P[retire:] -= theta * w
        T_BQ = tau_bq * BQ / lambdas
        T_H = (weights * lambdas * (T_I + T_P + T_BQ + T_W)).sum()
    elif method == 'TPI':
        T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w[:,retire:,:]
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
    J, S, retire, a_etr_income, b_etr_income, c_etr_income, d_etr_income, e_etr_income, f_etr_income,\
                   min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income, h_wealth, p_wealth, m_wealth, tau_payroll = params
    I = r * b + w * e * n
    tau_inc_params = (a_etr_income, b_etr_income, c_etr_income, d_etr_income, e_etr_income, f_etr_income,
                      min_x_etr_income, max_x_etr_income, min_y_etr_income, max_y_etr_income)
    T_I = tau_income(r, b, w, e, n, factor, tau_inc_params) * I

    T_P = tau_payroll * w * e * n
    TW_params = (h_wealth, p_wealth, m_wealth)
    T_W = tau_wealth(b, TW_params) * b
    if method == 'SS':
        # Depending on if we are looking at b_s or b_s+1, the
        # entry for retirement will change (it shifts back one).
        # The shift boolean makes sure we start replacement rates
        # at the correct age.
        if shift is False:
            T_P[retire:] -= theta * w
        else:
            T_P[retire - 1:] -= theta * w
        T_BQ = tau_bq * BQ / lambdas
    elif method == 'TPI':
        if shift is False:
            # retireTPI is different from retire, because in TPI we are counting backwards
            # with different length lists.  This will always be the correct location
            # of retirement, depending on the shape of the lists.
            retireTPI = (retire - S)
        else:
            retireTPI = (retire - 1 - S)
        if len(b.shape) != 3:
            T_P[retireTPI:] -= theta[j] * w[retireTPI:]
            T_BQ = tau_bq[j] * BQ / lambdas
        else:
            T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w[:,retire:,:]
            T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
    elif method == 'TPI_scalar':
        # The above methods won't work if scalars are used.  This option is only called by the
        # SS_TPI_firstdoughnutring function in TPI.
        T_P -= theta[j] * w
        T_BQ = tau_bq[j] * BQ / lambdas
    total_taxes = T_I + T_P + T_BQ + T_W - T_H


    return total_taxes
