'''
------------------------------------------------------------------------
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
        params    = length 3 tuple, (e, S, retire)
        e         = [S,J] array, effective labor units
        S         = integer, length of economic life
        retire    = integer, retirement age
    Functions called: None
    Objects in function:
        AIME       = [J,] vector, average indexed monthly earnings by lifetime income group
        PIA        = [J,] vector, primary insurance amount by lifetime income group
        maxpayment = scalar, maximum replacement rate
        theta      = [J,] vector, replacement rates by lifetime income group
    Returns: theta
    '''
    e, S, retire = params
    equiv_35 = int(round((S/80.0)*35)) - 1 # adjusts 35 year work history for any S
    if e.ndim == 2:
        dim2 = e.shape[1]
    else:
        dim2 = 1
    earnings = (e *(wss * nssmat * factor_ss)).reshape(S,dim2)
    # get highest earning 35 years
    highest_35_earn = (-1.0*np.sort(-1.0*earnings[:retire,:] ,axis=0))[:equiv_35]
    AIME = highest_35_earn.sum(0) / ((12.0*(S/80.0))*equiv_35)
    PIA = np.zeros(dim2)
    # Bins from data for each level of replacement
    for j in xrange(dim2):
        if AIME[j] < 749.0:
            PIA[j] = .9 * AIME[j]
        elif AIME[j] < 4517.0:
            PIA[j] = 674.1 + .32 * (AIME[j] - 749.0)
        else:
            PIA[j] = 1879.86 + .15 * (AIME[j] - 4517.0)
    # Set the maximum monthly replacment rate from SS benefits tables
    maxpayment = 3501.00
    PIA[PIA > maxpayment] = maxpayment
    theta = (PIA*(12.0*S/80.0)) / (factor_ss*wss)
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
    --------------------------------------------------------------------
    Calculates effective personal income tax rate.
    --------------------------------------------------------------------
    INPUTS:
    r          = [T,] vector, interest rate
    w          = [T,] vector, wage rate
    b          = [T,S,J] array, wealth holdings
    n          = [T,S,J] array, labor supply
    factor     = scalar, model income scaling factor
    params     = length 2 tuple, (e, etr_params)
    e          = [T,S,J] array, effective labor units
    etr_params = [T,S,J] array, effective tax rate function parameters

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    A       = [T,S,J] array, polynomial coefficient on x**2
    B       = [T,S,J] array, polynomial coefficient on x
    C       = [T,S,J] array, polynomial coefficient on y**2
    D       = [T,S,J] array, polynomial coefficient on y
    max_x   = [T,S,J] array, maximum effective tax rate for x given y=0
    min_x   = [T,S,J] array, minimum effective tax rate for x given y=0
    max_y   = [T,S,J] array, maximum effective tax rate for y given x=0
    min_y   = [T,S,J] array, minimum effective tax rate for y given x=0
    shift_x = (T, S, J) array, shift parameter on labor income in Cobb-
              Douglas function
    shift_y = (T, S, J) array, shift parameter on capital income in
              Cobb-Douglas function
    shift   = (T, S, J) array, shift parameter on total function in
              Cobb-Douglas function
    share   = (T, S, J) array, share parameter (exponent) in Cobb-
              Douglas functions
    X       = [T,S,J] array, labor income
    Y       = [T,S,J] array, capital income
    X2      = [T,S,J] array, labor income squared X**2
    Y2      = [T,S,J] array, capital income squared Y**2
    tau_x   = [T,S,J] array, labor income portion of the function with
              ratio of polynomials
    tau_y   = [T,S,J] array, capital income portion of the function with
              ratio of polynomials
    tau     = [T,S,J] array, effective personal income tax rate

    RETURNS: tau
    --------------------------------------------------------------------
    '''
    e, etr_params = params

    if etr_params.ndim == 4:
        A = etr_params[:,:,:,0]
        B = etr_params[:,:,:,1]
        C = etr_params[:,:,:,2]
        D = etr_params[:,:,:,3]
        max_x = etr_params[:,:,:,4]
        min_x = etr_params[:,:,:,5]
        max_y = etr_params[:,:,:,6]
        min_y = etr_params[:,:,:,7]
        shift_x = etr_params[:,:,:,8]
        shift_y = etr_params[:,:,:,9]
        shift = etr_params[:,:,:,10]
        share = etr_params[:,:,:,11]
    if etr_params.ndim == 3:
        A = etr_params[:,:,0]
        B = etr_params[:,:,1]
        C = etr_params[:,:,2]
        D = etr_params[:,:,3]
        max_x = etr_params[:,:,4]
        min_x = etr_params[:,:,5]
        max_y = etr_params[:,:,6]
        min_y = etr_params[:,:,7]
        shift_x = etr_params[:,:,8]
        shift_y = etr_params[:,:,9]
        shift = etr_params[:,:,10]
        share = etr_params[:,:,11]
    if etr_params.ndim == 2:
        A = etr_params[:,0]
        B = etr_params[:,1]
        C = etr_params[:,2]
        D = etr_params[:,3]
        max_x = etr_params[:,4]
        min_x = etr_params[:,5]
        max_y = etr_params[:,6]
        min_y = etr_params[:,7]
        shift_x = etr_params[:,8]
        shift_y = etr_params[:,9]
        shift = etr_params[:,10]
        share = etr_params[:,11]
    if etr_params.ndim == 1:
        A = etr_params[0]
        B = etr_params[1]
        C = etr_params[2]
        D = etr_params[3]
        max_x = etr_params[4]
        min_x = etr_params[5]
        max_y = etr_params[6]
        min_y = etr_params[7]
        shift_x = etr_params[8]
        shift_y = etr_params[9]
        shift = etr_params[10]
        share = etr_params[11]

    X = (w*e*n)*factor
    Y = (r*b)*factor
    X2 = X ** 2
    Y2 = Y ** 2
    tau_x = ((max_x - min_x) * (A * X2 + B * X) /
        (A * X2 + B * X + 1) + min_x)
    tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
        (C * Y2 + D * Y + 1) + min_y)
    tau = (((tau_x + shift_x) ** share) *
        ((tau_y + shift_y) ** (1 - share))) + shift

    return tau



# Note that since when we use the same functional form, one could use
# just one tax function for ETR, MTR_lab, MTR_cap, just with different
# parameters input

def MTR_capital(r, w, b, n, factor, params):
    '''
    --------------------------------------------------------------------
    Generates the marginal tax rate on capital income for households.
    --------------------------------------------------------------------
    INPUTS:
    r               = [T,] vector, interest rate
    w               = [T,] vector, wage rate
    b               = [T,S,J] array, wealth holdings
    n               = [T,S,J] array, labor supply
    factor          = scalar, model income scaling factor
    params          = length 3 tuple, (e, mtry_params, analytical_mtrs)
    e               = [T,S,J] array, effective labor units
    mtry_params     = [T,S,J] array, marginal tax rate on capital income
                      function parameters
    analytical_mtrs = boolean, =True if use analytical mtrs rather than
                      estimated mtrs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    A       = [T,S,J] array, polynomial coefficient on x**2
    B       = [T,S,J] array, polynomial coefficient on x
    C       = [T,S,J] array, polynomial coefficient on y**2
    D       = [T,S,J] array, polynomial coefficient on y
    max_x   = [T,S,J] array, maximum effective tax rate for x given y=0
    min_x   = [T,S,J] array, minimum effective tax rate for x given y=0
    max_y   = [T,S,J] array, maximum effective tax rate for y given x=0
    min_y   = [T,S,J] array, minimum effective tax rate for y given x=0
    shift_x = (T, S, J) array, shift parameter on labor income in Cobb-
              Douglas function
    shift_y = (T, S, J) array, shift parameter on capital income in
              Cobb-Douglas function
    shift   = (T, S, J) array, shift parameter on total function in
              Cobb-Douglas function
    share   = (T, S, J) array, share parameter (exponent) in Cobb-
              Douglas functions
    X       = [T,S,J] array, labor income
    Y       = [T,S,J] array, capital income
    X2      = [T,S,J] array, labor income squared X**2
    Y2      = [T,S,J] array, capital income squared Y**2
    tau_x   = [T,S,J] array, labor income portion of the function with
              ratio of polynomials
    tau_y   = [T,S,J] array, capital income portion of the function with
              ratio of polynomials
    tau     = [T,S,J] array, marginal tax rate on capital income

    RETURNS: tau
    --------------------------------------------------------------------
    '''
    e, etr_params, mtry_params, analytical_mtrs = params

    if analytical_mtrs:
        if etr_params.ndim == 3:
            A = etr_params[:,:,0]
            B = etr_params[:,:,1]
            C = etr_params[:,:,2]
            D = etr_params[:,:,3]
            max_x = etr_params[:,:,4]
            min_x = etr_params[:,:,5]
            max_y = etr_params[:,:,6]
            min_y = etr_params[:,:,7]
            shift_x = etr_params[:,:,8]
            shift_y = etr_params[:,:,9]
            shift = etr_params[:,:,10]
            share = etr_params[:,:,11]
        if etr_params.ndim == 2:
            A = etr_params[:,0]
            B = etr_params[:,1]
            C = etr_params[:,2]
            D = etr_params[:,3]
            max_x = etr_params[:,4]
            min_x = etr_params[:,5]
            max_y = etr_params[:,6]
            min_y = etr_params[:,7]
            shift_x = etr_params[:,8]
            shift_y = etr_params[:,9]
            shift = etr_params[:,10]
            share = etr_params[:,11]
        if etr_params.ndim == 1:
            A = etr_params[0]
            B = etr_params[1]
            C = etr_params[2]
            D = etr_params[3]
            max_x = etr_params[4]
            min_x = etr_params[5]
            max_y = etr_params[6]
            min_y = etr_params[7]
            shift_x = etr_params[8]
            shift_y = etr_params[9]
            shift = etr_params[10]
            share = etr_params[11]

        X = (w*e*n)*factor
        Y = (r*b)*factor
        X2 = X ** 2
        Y2 = Y ** 2
        tau_x = ((max_x - min_x) * (A * X2 + B * X) /
            (A * X2 + B * X + 1) + min_x)
        tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
            (C * Y2 + D * Y + 1) + min_y)
        tau_x_y = (((tau_x + shift_x) ** share) *
            ((tau_y + shift_y) ** (1 - share))) + shift

        tau = ((X+Y)*((tau_x+shift_x)**share)*(1-share)*(max_y-min_y)*\
               ((2*C*X+D)/((C*X2+D*X+1)**2))*((tau_y+shift_y)**(-share)) + tau_x_y)

    else:
        if mtry_params.ndim == 3:
            A = mtry_params[:,:,0]
            B = mtry_params[:,:,1]
            C = mtry_params[:,:,2]
            D = mtry_params[:,:,3]
            max_x = mtry_params[:,:,4]
            min_x = mtry_params[:,:,5]
            max_y = mtry_params[:,:,6]
            min_y = mtry_params[:,:,7]
            shift_x = mtry_params[:,:,8]
            shift_y = mtry_params[:,:,9]
            shift = mtry_params[:,:,10]
            share = mtry_params[:,:,11]
        if mtry_params.ndim == 2:
            A = mtry_params[:,0]
            B = mtry_params[:,1]
            C = mtry_params[:,2]
            D = mtry_params[:,3]
            max_x = mtry_params[:,4]
            min_x = mtry_params[:,5]
            max_y = mtry_params[:,6]
            min_y = mtry_params[:,7]
            shift_x = mtry_params[:,8]
            shift_y = mtry_params[:,9]
            shift = mtry_params[:,10]
            share = mtry_params[:,11]
        if mtry_params.ndim == 1:
            A = mtry_params[0]
            B = mtry_params[1]
            C = mtry_params[2]
            D = mtry_params[3]
            max_x = mtry_params[4]
            min_x = mtry_params[5]
            max_y = mtry_params[6]
            min_y = mtry_params[7]
            shift_x = mtry_params[8]
            shift_y = mtry_params[9]
            shift = mtry_params[10]
            share = mtry_params[11]

        X = (w*e*n)*factor
        Y = (r*b)*factor
        X2 = X ** 2
        Y2 = Y ** 2
        tau_x = ((max_x - min_x) * (A * X2 + B * X) /
            (A * X2 + B * X + 1) + min_x)
        tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
            (C * Y2 + D * Y + 1) + min_y)
        tau = (((tau_x + shift_x) ** share) *
            ((tau_y + shift_y) ** (1 - share))) + shift

    return tau


def MTR_labor(r, w, b, n, factor, params):
    '''
    --------------------------------------------------------------------
    Generates the marginal tax rate on labor income for households.
    --------------------------------------------------------------------
    INPUTS:
    r               = [T,] vector, interest rate
    w               = [T,] vector, wage rate
    b               = [T,S,J] array, wealth holdings
    n               = [T,S,J] array, labor supply
    factor          = scalar, model income scaling factor
    params          = length 3 tuple, (e, mtry_params, analytical_mtrs)
    e               = [T,S,J] array, effective labor units
    mtrx_params     = [T,S,J] array, marginal tax rate on capital income
                      function parameters
    analytical_mtrs = boolean, =True if use analytical mtrs rather than
                      estimated mtrs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    A       = [T,S,J] array, polynomial coefficient on x**2
    B       = [T,S,J] array, polynomial coefficient on x
    C       = [T,S,J] array, polynomial coefficient on y**2
    D       = [T,S,J] array, polynomial coefficient on y
    max_x   = [T,S,J] array, maximum effective tax rate for x given y=0
    min_x   = [T,S,J] array, minimum effective tax rate for x given y=0
    max_y   = [T,S,J] array, maximum effective tax rate for y given x=0
    min_y   = [T,S,J] array, minimum effective tax rate for y given x=0
    shift_x = (T, S, J) array, shift parameter on labor income in Cobb-
              Douglas function
    shift_y = (T, S, J) array, shift parameter on capital income in
              Cobb-Douglas function
    shift   = (T, S, J) array, shift parameter on total function in
              Cobb-Douglas function
    share   = (T, S, J) array, share parameter (exponent) in Cobb-
              Douglas functions
    X       = [T,S,J] array, labor income
    Y       = [T,S,J] array, capital income
    X2      = [T,S,J] array, labor income squared X**2
    Y2      = [T,S,J] array, capital income squared Y**2
    tau_x   = [T,S,J] array, labor income portion of the function with
              ratio of polynomials
    tau_y   = [T,S,J] array, capital income portion of the function with
              ratio of polynomials
    tau     = [T,S,J] array, marginal tax rate on labor income

    RETURNS: tau
    --------------------------------------------------------------------
    '''
    e, etr_params, mtrx_params, analytical_mtrs = params

    if analytical_mtrs:
        if etr_params.ndim == 3:
            A = etr_params[:,:,0]
            B = etr_params[:,:,1]
            C = etr_params[:,:,2]
            D = etr_params[:,:,3]
            max_x = etr_params[:,:,4]
            min_x = etr_params[:,:,5]
            max_y = etr_params[:,:,6]
            min_y = etr_params[:,:,7]
            shift_x = etr_params[:,:,8]
            shift_y = etr_params[:,:,9]
            shift = etr_params[:,:,10]
            share = etr_params[:,:,11]
        if etr_params.ndim == 2:
            A = etr_params[:,0]
            B = etr_params[:,1]
            C = etr_params[:,2]
            D = etr_params[:,3]
            max_x = etr_params[:,4]
            min_x = etr_params[:,5]
            max_y = etr_params[:,6]
            min_y = etr_params[:,7]
            shift_x = etr_params[:,8]
            shift_y = etr_params[:,9]
            shift = etr_params[:,10]
            share = etr_params[:,11]
        if etr_params.ndim == 1:
            A = etr_params[0]
            B = etr_params[1]
            C = etr_params[2]
            D = etr_params[3]
            max_x = etr_params[4]
            min_x = etr_params[5]
            max_y = etr_params[6]
            min_y = etr_params[7]
            shift_x = etr_params[8]
            shift_y = etr_params[9]
            shift = etr_params[10]
            share = etr_params[11]

        X = (w*e*n)*factor
        Y = (r*b)*factor
        X2 = X ** 2
        Y2 = Y ** 2
        tau_x = ((max_x - min_x) * (A * X2 + B * X) /
            (A * X2 + B * X + 1) + min_x)
        tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
            (C * Y2 + D * Y + 1) + min_y)
        tau_x_y = (((tau_x + shift_x) ** share) *
            ((tau_y + shift_y) ** (1 - share))) + shift

        tau = ((X+Y)*share*((tau_x+shift_x)**(share-1))*(max_x-min_x)*\
               ((2*A*X+B)/((A*X2+B*X+1)**2))*((tau_y+shift_y)**(1-share)) + tau_x_y)

    else:
        if mtrx_params.ndim == 3:
            A = mtrx_params[:,:,0]
            B = mtrx_params[:,:,1]
            C = mtrx_params[:,:,2]
            D = mtrx_params[:,:,3]
            max_x = mtrx_params[:,:,4]
            min_x = mtrx_params[:,:,5]
            max_y = mtrx_params[:,:,6]
            min_y = mtrx_params[:,:,7]
            shift_x = mtrx_params[:,:,8]
            shift_y = mtrx_params[:,:,9]
            shift = mtrx_params[:,:,10]
            share = mtrx_params[:,:,11]
        if mtrx_params.ndim == 2:
            A = mtrx_params[:,0]
            B = mtrx_params[:,1]
            C = mtrx_params[:,2]
            D = mtrx_params[:,3]
            max_x = mtrx_params[:,4]
            min_x = mtrx_params[:,5]
            max_y = mtrx_params[:,6]
            min_y = mtrx_params[:,7]
            shift_x = mtrx_params[:,8]
            shift_y = mtrx_params[:,9]
            shift = mtrx_params[:,10]
            share = mtrx_params[:,11]
        if mtrx_params.ndim == 1:
            A = mtrx_params[0]
            B = mtrx_params[1]
            C = mtrx_params[2]
            D = mtrx_params[3]
            max_x = mtrx_params[4]
            min_x = mtrx_params[5]
            max_y = mtrx_params[6]
            min_y = mtrx_params[7]
            shift_x = mtrx_params[8]
            shift_y = mtrx_params[9]
            shift = mtrx_params[10]
            share = mtrx_params[11]

        X = (w*e*n)*factor
        Y = (r*b)*factor
        X2 = X ** 2
        Y2 = Y ** 2
        tau_x = (((max_x - min_x) * (A * X2 + B * X) /
            (A * X2 + B * X + 1)) + min_x)
        tau_y = (((max_y - min_y) * (C * Y2 + D * Y) /
        (C * Y2 + D * Y + 1)) + min_y)
        tau = (((tau_x + shift_x) ** share) *
            ((tau_y + shift_y) ** (1 - share))) + shift

    return tau


def get_biz_tax(w, Y, L, K, params):
    '''
    Finds total business income tax receipts
    Inputs:
        r           = [T,] vector, interest rate
        Y           = [T,] vector, aggregate output
        L           = [T,] vector, aggregate labor demand
        K           = [T,] vector, aggregate capital demand
    Objects in function:
        business_revenue    = [T,] vector, total revenue from business income taxes
    Returns: T_H

    '''

    tau_b, delta_tau = params
    business_revenue = tau_b*(Y-w*L) - tau_b*delta_tau*K
    return business_revenue

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
        tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J,\
        tau_b, delta_tau = params

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
        biz_params = (tau_b, delta_tau)
        business_revenue = get_biz_tax(w, Y, L, K, biz_params)
        REVENUE = (omega * lambdas * (T_I + T_P + T_BQ + T_W)).sum() + business_revenue
    elif method == 'TPI':
        T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w[:,retire:,:]
        T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
        biz_params = (tau_b, delta_tau)
        business_revenue = get_biz_tax(w[:T,0,0], Y, L, K, biz_params)
        REVENUE = (omega * lambdas * (T_I + T_P + T_BQ + T_W)).sum(1).sum(1) + business_revenue
    return REVENUE


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
