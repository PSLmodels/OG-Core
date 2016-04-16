'''
------------------------------------------------------------------------
Last updated 4/7/2016

Functions for taxes in the steady state and along the transition path.

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


def replacement_rate_vals(nssmat, wss, factor_ss, params):
    '''
    Calculates replacement rate values for the payroll tax.

    Inputs:
        nssmat    = [S,J] array, steady state labor supply
        wss       = scalar, steady state wage rate
        factor_ss = scalar, factor that converts model income to dollars 
        params    = length 4 tuple, (e, J, omega_SS, lambdas)
        e         = [S,J] array, effective labor units 
        J         = integer, number of ability types 
        omega_SS  = [S,] vector, population weights by age 
        lambdas   = [J,] vector, lifetime income group weights

    Functions called: None

    Objects in function:
        AIME       = [J,] vector, average indexed monthly earnings by lifetime income group
        PIA        = [J,] vector, primary insurance amount by lifetime income group
        maxpayment = scalar, maximum replacement rate
        theta      = [J,] vector, replacement rates by lifetime income group

    Returns: theta

    '''
    e, J, omega_SS, lambdas = params

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
    theta = 0  # setting theta = 0 since we are including social security income in capital income (CHECK)
    return theta


def tau_wealth(b, params):
    '''
    Calculates the effective tax rate on wealth.

    Inputs:
        b        = [T,S,J] array, wealth holdings
        params   = length 3 tuple, (h_wealth, p_wealth, m_wealth)
        h_wealth = scalar, parameter of wealth tax function
        p_wealth = scalar, parameter of wealth tax function
        m_wealth = scalar, parameter of wealth tax function

    Functions called: None

    Objects in function:
        tau_w = [T,S,J] array, effective tax rate on wealth 

    Returns: tau_w
        
    '''
    h_wealth, p_wealth, m_wealth = params
    
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w = p * h * b / (h * b + m)
    return tau_w


def tau_w_prime(b, params):
    '''
    Calculates the marginal tax rate on wealth from the wealth tax.

    Inputs:
        b        = [T,S,J] array, wealth holdings
        params   = length 3 tuple, (h_wealth, p_wealth, m_wealth)
        h_wealth = scalar, parameter of wealth tax function
        p_wealth = scalar, parameter of wealth tax function
        m_wealth = scalar, parameter of wealth tax function

    Functions called: None

    Objects in function:
        tau_w_prime = [T,S,J] array, marginal tax rate on wealth from wealth tax

    Returns: tau_w_prime

    '''
    h_wealth, p_wealth, m_wealth = params

    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w_prime = h * m * p / (b * h + m) ** 2
    return tau_w_prime


def tau_income(r, w, b, n, factor, params):
    '''
    Calculate personal income tax liability.
    
    Inputs:
        r          = [T,] vector, interest rate 
        w          = [T,] vector, wage rate 
        b          = [T,S,J] array, wealth holdings 
        n          = [T,S,J] array, labor supply
        factor     = scalar, model income scaling factor
        params     = length 2 tuple, (e, etr_params)
        e          = [T,S,J] array, effective labor units
        etr_params = [T,S,J] array, effective tax rate function parameters

    Functions called: None

    Objects in function:
        A     = [T,S,J] array, polynomial coefficient on x**2
        B     = [T,S,J] array, polynomial coefficient on y**2
        C     = [T,S,J] array, polynomial coefficient on x*y
        D     = [T,S,J] array, polynomial coefficient on x
        E     = [T,S,J] array, polynomial coefficient on y
        F     = [T,S,J] array, polynomial constant
        max_x = [T,S,J] array, maximum effective tax rate for x given y=0
        min_x = [T,S,J] array, minimum effective tax rate for x given y=0
        max_y = [T,S,J] array, maximum effective tax rate for y given x=0
        min_y = [T,S,J] array, minimum effective tax rate for y given x=0
        x     = [T,S,J] array, labor income
        y     = [T,S,J] array, capital income
        I     = [T,S,J] array, total income (capital plus labor income)
        phi   = [T,S,J] array, fraction of total income that is labor income
        tau   = [T,S,J] array, personal income tax liability

    Returns: tau
    '''
    e, etr_params = params

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
def MTR_capital(r, w, b, n, factor, params):
    '''
    Generates the marginal tax rate on capital income for households.
    
    Inputs:
        r               = [T,] vector, interest rate 
        w               = [T,] vector, wage rate 
        b               = [T,S,J] array, wealth holdings 
        n               = [T,S,J] array, labor supply
        factor          = scalar, model income scaling factor
        params          = length 3 tuple, (e, mtry_params, analytical_mtrs)
        e               = [T,S,J] array, effective labor units
        mtry_params     = [T,S,J] array, marginal tax rate on capital income function parameters
        analytical_mtrs = boolean, =True if use analytical mtrs rather than estimated mtrs

    Functions called: None

    Objects in function:
        A     = [T,S,J] array, polynomial coefficient on x**2
        B     = [T,S,J] array, polynomial coefficient on y**2
        C     = [T,S,J] array, polynomial coefficient on x*y
        D     = [T,S,J] array, polynomial coefficient on x
        E     = [T,S,J] array, polynomial coefficient on y
        F     = [T,S,J] array, polynomial constant
        max_x = [T,S,J] array, maximum effective tax rate for x given y=0
        min_x = [T,S,J] array, minimum effective tax rate for x given y=0
        max_y = [T,S,J] array, maximum effective tax rate for y given x=0
        min_y = [T,S,J] array, minimum effective tax rate for y given x=0
        x     = [T,S,J] array, labor income
        y     = [T,S,J] array, capital income
        I     = [T,S,J] array, total income (capital plus labor income)
        phi   = [T,S,J] array, fraction of total income that is labor income
        tau   = [T,S,J] array, marginal tax rate on capital income 

    Returns: tau
    '''

    e, etr_params, mtry_params, analytical_mtrs = params

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


def MTR_labor(r, w, b, n, factor, params):
    '''
    Generates the marginal tax rate on labor income for households.
    
    Inputs:
        r               = [T,] vector, interest rate 
        w               = [T,] vector, wage rate 
        b               = [T,S,J] array, wealth holdings 
        n               = [T,S,J] array, labor supply
        factor          = scalar, model income scaling factor
        params          = length 3 tuple, (e, mtry_params, analytical_mtrs)
        e               = [T,S,J] array, effective labor units
        mtrx_params     = [T,S,J] array, marginal tax rate on capital income function parameters
        analytical_mtrs = boolean, =True if use analytical mtrs rather than estimated mtrs

    Functions called: None

    Objects in function:
        A     = [T,S,J] array, polynomial coefficient on x**2
        B     = [T,S,J] array, polynomial coefficient on y**2
        C     = [T,S,J] array, polynomial coefficient on x*y
        D     = [T,S,J] array, polynomial coefficient on x
        E     = [T,S,J] array, polynomial coefficient on y
        F     = [T,S,J] array, polynomial constant
        max_x = [T,S,J] array, maximum effective tax rate for x given y=0
        min_x = [T,S,J] array, minimum effective tax rate for x given y=0
        max_y = [T,S,J] array, maximum effective tax rate for y given x=0
        min_y = [T,S,J] array, minimum effective tax rate for y given x=0
        x     = [T,S,J] array, labor income
        y     = [T,S,J] array, capital income
        I     = [T,S,J] array, total income (capital plus labor income)
        phi   = [T,S,J] array, fraction of total income that is labor income
        tau   = [T,S,J] array, marginal tax rate on labor income 

    Returns: tau
    '''

    e, etr_params, mtrx_params, analytical_mtrs = params

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


def get_lump_sum(r, w, b, n, BQ, factor, params):
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

    e, lambdas, omega, method, etr_params, theta, tau_bq, \
        tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J = params

    I = r * b + w * e * n
    
    if I.ndim == 2: 
        T_I = np.zeros((S,J))
        for j in xrange(J):
            TI_params = (e[:,j], etr_params)
            T_I[:,j] = tau_income(r, w, b[:,j], n[:,j], factor, TI_params) * I[:,j]
    if I.ndim == 3:
        T_I = np.zeros((T,S,J))
        for j in xrange(J):
            if etr_params.ndim == 3:
                tau_inc_params3D = etr_params[:,j,:]
            if etr_params.ndim == 4:
                tau_inc_params3D = etr_params[:,:,j,:]
            TI_params = (e[:,:,j], tau_inc_params3D)
            T_I[:,:,j] = tau_income(r[:,:,j], w[:,:,j], b[:,:,j], n[:,:,j], factor, TI_params) * I[:,:,j]  
    T_P = tau_payroll * w * e * n
    TW_params = (h_wealth, p_wealth, m_wealth)
    T_W = tau_wealth(b, TW_params) * b
    if method == 'SS':
        T_P[retire:] -= theta * w
        T_BQ = tau_bq * BQ / lambdas
        T_H = (omega * lambdas * (T_I + T_P + T_BQ + T_W)).sum()
    elif method == 'TPI':
        T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w[:,retire:,:]
        T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
        T_H = (omega * lambdas * (T_I + T_P + T_BQ + T_W)).sum(1).sum(1)
    return T_H

    
def total_taxes(r, w, b, n, BQ, factor, T_H, j, shift, params):
    '''
    Gives net taxes paid values.

    Inputs:
        r          = [T,] vector, interest rate 
        w          = [T,] vector, wage rate 
        b          = [T,S,J] array, wealth holdings 
        n          = [T,S,J] array, labor supply
        BQ         = [T,J] vector,  bequest amounts
        factor     = scalar, model income scaling factor
        T_H        = [T,] vector, lump sum transfer amount(s) 
        j          = integer, lifetime incoem group being computed
        shift      = boolean, computing for periods 0--s or 1--(s+1) (bool) (True for 1--(s+1))
        params = length 13 tuple, (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, 
                                   m_wealth, tau_payroll, theta, tau_bq, J, S)
        e           = [T,S,J] array, effective labor units
        lambdas     = [J,] vector, population weights by lifetime income group
        method      = string, 'SS' or 'TPI'
        retire      = integer, retirement age
        etr_params  = [T,S,J] array, effective tax rate function parameters
        h_wealth    = scalar, wealth tax function parameter
        p_wealth    = scalar, wealth tax function parameter
        m_wealth    = scalar, wealth tax function parameter
        tau_payroll = scalar, payroll tax rate
        theta       = [J,] vector, replacement rate values by lifetime income group
        tau_bq      = scalar, bequest tax rate 
        S           = integer, number of age groups
        J           = integer, number of lifetime income groups 

    Functions called: 
        tau_income
        tau_wealth

    Objects in function:
        I           = [T,S,J] array, total income
        T_I         = [T,S,J] array, total income taxes
        T_P         = [T,S,J] array, total payroll taxes
        T_W         = [T,S,J] array, total wealth taxes
        T_BQ        = [T,S,J] array, total bequest taxes
        retireTPI   = integer, =(retire - S)
        total_taxes = [T,] vector, net taxes 

    Returns: total_taxes
    
    '''

    e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S = params

    I = r * b + w * e * n
    TI_params = (e, etr_params)
    T_I = tau_income(r, w, b, n, factor, TI_params) * I

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
        #T_P -= theta[j] * w
        T_P = 0.
        T_BQ = tau_bq[j] * BQ / lambdas
    total_taxes = T_I + T_P + T_BQ + T_W - T_H


    return total_taxes