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


def tau_income(r, b, w, e, n, factor, etr_params):
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

    if etr_params.ndim == 4: 
        A = etr_params[:,:,:,0]
        B = etr_params[:,:,:,1]
        C = etr_params[:,:,:,2]
        D = etr_params[:,:,:,3]
        E = etr_params[:,:,:,4]
        F = etr_params[:,:,:,5]
        max_x = etr_params[:,:,:,6]
        min_x = etr_params[:,:,:,7]
        max_y = etr_params[:,:,:,8]
        min_y = etr_params[:,:,:,9]
    if etr_params.ndim == 3: 
        A = etr_params[:,:,0]
        B = etr_params[:,:,1]
        C = etr_params[:,:,2]
        D = etr_params[:,:,3]
        E = etr_params[:,:,4]
        F = etr_params[:,:,5]
        max_x = etr_params[:,:,6]
        min_x = etr_params[:,:,7]
        max_y = etr_params[:,:,8]
        min_y = etr_params[:,:,9]
    if etr_params.ndim == 2: 
        A = etr_params[:,0]
        B = etr_params[:,1]
        C = etr_params[:,2]
        D = etr_params[:,3]
        E = etr_params[:,4]
        F = etr_params[:,5]
        max_x = etr_params[:,6]
        min_x = etr_params[:,7]
        max_y = etr_params[:,8]
        min_y = etr_params[:,9]
    if etr_params.ndim == 1: 
        A = etr_params[0]
        B = etr_params[1]
        C = etr_params[2]
        D = etr_params[3]
        E = etr_params[4]
        F = etr_params[5]
        max_x = etr_params[6]
        min_x = etr_params[7]
        max_y = etr_params[8]
        min_y = etr_params[9]

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



## Note that since when we use the same functional form, one could
# use just one tax function for ATR, MTR_lab, MTR_cap, just with different parameters input
def MTR_capital(r, b, w, e, n, factor, analytical_mtrs, etr_params, mtry_params):
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

    if analytical_mtrs:
        if etr_params.ndim == 3: 
            A = etr_params[:,:,0]
            B = etr_params[:,:,1]
            C = etr_params[:,:,2]
            D = etr_params[:,:,3]
            E = etr_params[:,:,4]
            F = etr_params[:,:,5]
            max_x = etr_params[:,:,6]
            min_x = etr_params[:,:,7]
            max_y = etr_params[:,:,8]
            min_y = etr_params[:,:,9]
        if etr_params.ndim == 2: 
            A = etr_params[:,0]
            B = etr_params[:,1]
            C = etr_params[:,2]
            D = etr_params[:,3]
            E = etr_params[:,4]
            F = etr_params[:,5]
            max_x = etr_params[:,6]
            min_x = etr_params[:,7]
            max_y = etr_params[:,8]
            min_y = etr_params[:,9]
        if etr_params.ndim == 1: 
            A = etr_params[0]
            B = etr_params[1]
            C = etr_params[2]
            D = etr_params[3]
            E = etr_params[4]
            F = etr_params[5]
            max_x = etr_params[6]
            min_x = etr_params[7]
            max_y = etr_params[8]
            min_y = etr_params[9]
        
        x = (w*e*n)*factor
        y = (r*b)*factor
        I = x+y

        num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
        denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
        Lambda = num/denom

        d_num = (2*B*y + C*x + E)*F
        d_denom = ((A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F)**2
        d_Lambda = d_num/d_denom

        tau =  (max_y-min_y)*Lambda + (x*(max_x-min_x) + y*(max_y-min_y))*d_Lambda + min_y


    else:
        if mtry_params.ndim == 3: 
            A = mtry_params[:,:,0]
            B = mtry_params[:,:,1]
            C = mtry_params[:,:,2]
            D = mtry_params[:,:,3]
            E = mtry_params[:,:,4]
            F = mtry_params[:,:,5]
            max_x = mtry_params[:,:,6]
            min_x = mtry_params[:,:,7]
            max_y = mtry_params[:,:,8]
            min_y = mtry_params[:,:,9]
        if mtry_params.ndim == 2: 
            A = mtry_params[:,0]
            B = mtry_params[:,1]
            C = mtry_params[:,2]
            D = mtry_params[:,3]
            E = mtry_params[:,4]
            F = mtry_params[:,5]
            max_x = mtry_params[:,6]
            min_x = mtry_params[:,7]
            max_y = mtry_params[:,8]
            min_y = mtry_params[:,9]
        if mtry_params.ndim == 1: 
            A = mtry_params[0]
            B = mtry_params[1]
            C = mtry_params[2]
            D = mtry_params[3]
            E = mtry_params[4]
            F = mtry_params[5]
            max_x = mtry_params[6]
            min_x = mtry_params[7]
            max_y = mtry_params[8]
            min_y = mtry_params[9]
        
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


def MTR_labor(r, b, w, e, n, factor, analytical_mtrs, etr_params, mtrx_params):
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

    if analytical_mtrs:
        if etr_params.ndim == 3: 
            A = etr_params[:,:,0]
            B = etr_params[:,:,1]
            C = etr_params[:,:,2]
            D = etr_params[:,:,3]
            E = etr_params[:,:,4]
            F = etr_params[:,:,5]
            max_x = etr_params[:,:,6]
            min_x = etr_params[:,:,7]
            max_y = etr_params[:,:,8]
            min_y = etr_params[:,:,9]
        if etr_params.ndim == 2: 
            A = etr_params[:,0]
            B = etr_params[:,1]
            C = etr_params[:,2]
            D = etr_params[:,3]
            E = etr_params[:,4]
            F = etr_params[:,5]
            max_x = etr_params[:,6]
            min_x = etr_params[:,7]
            max_y = etr_params[:,8]
            min_y = etr_params[:,9]
        if etr_params.ndim == 1: 
            A = etr_params[0]
            B = etr_params[1]
            C = etr_params[2]
            D = etr_params[3]
            E = etr_params[4]
            F = etr_params[5]
            max_x = etr_params[6]
            min_x = etr_params[7]
            max_y = etr_params[8]
            min_y = etr_params[9]
        
        x = (w*e*n)*factor
        y = (r*b)*factor
        I = x+y

        num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
        denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
        Lambda = num/denom

        d_num = (2*A*x + C*y + D)*F
        d_denom = ((A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F)**2
        d_Lambda = d_num/d_denom

        tau =  (max_x-min_x)*Lambda + (x*(max_x-min_x) + y*(max_y-min_y))*d_Lambda + min_x

    else:
        if mtrx_params.ndim == 3: 
            A = mtrx_params[:,:,0]
            B = mtrx_params[:,:,1]
            C = mtrx_params[:,:,2]
            D = mtrx_params[:,:,3]
            E = mtrx_params[:,:,4]
            F = mtrx_params[:,:,5]
            max_x = mtrx_params[:,:,6]
            min_x = mtrx_params[:,:,7]
            max_y = mtrx_params[:,:,8]
            min_y = mtrx_params[:,:,9]
        if mtrx_params.ndim == 2: 
            A = mtrx_params[:,0]
            B = mtrx_params[:,1]
            C = mtrx_params[:,2]
            D = mtrx_params[:,3]
            E = mtrx_params[:,4]
            F = mtrx_params[:,5]
            max_x = mtrx_params[:,6]
            min_x = mtrx_params[:,7]
            max_y = mtrx_params[:,8]
            min_y = mtrx_params[:,9]
        if mtrx_params.ndim == 1: 
            A = mtrx_params[0]
            B = mtrx_params[1]
            C = mtrx_params[2]
            D = mtrx_params[3]
            E = mtrx_params[4]
            F = mtrx_params[5]
            max_x = mtrx_params[6]
            min_x = mtrx_params[7]
            max_y = mtrx_params[8]
            min_y = mtrx_params[9]
      
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


def get_lump_sum(r, b, w, e, n, BQ, lambdas, factor, weights, method, etr_params, params, theta, tau_bq):
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

    I = r * b + w * e * n
    
    if I.ndim == 2: 
        T_I = np.zeros((S,J))
        for j in xrange(J):
            T_I[:,j] = tau_income(r, b[:,j], w, e[:,j], n[:,j], factor, etr_params) * I[:,j]
    if I.ndim == 3:
        T_I = np.zeros((T,S,J))
        for j in xrange(J):
            if etr_params.ndim == 3:
                tau_inc_params3D = etr_params[:,j,:]
            if etr_params.ndim == 4:
                tau_inc_params3D = etr_params[:,:,j,:]
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
    J, S, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll = params
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor, etr_params) * I

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
