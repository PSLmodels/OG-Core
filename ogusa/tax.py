'''
------------------------------------------------------------------------
Functions for taxes in the steady state and along the transition path.
------------------------------------------------------------------------
'''

# Packages
import numpy as np
from ogusa import utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def replacement_rate_vals(nssmat, wss, factor_ss, j, p):
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
        AIME       = [J,] vector, average indexed monthly earnings by
                          lifetime income group
        PIA        = [J,] vector, primary insurance amount by lifetime
                          income group
        maxpayment = scalar, maximum replacement rate
        theta      = [J,] vector, replacement rates by lifetime income
                          group
    Returns: theta
    '''
    if j is not None:
        e = p.e[:, j]
    else:
        e = p.e
    # adjust 35 yr work history for any S
    equiv_35 = int(round((p.S / 80.0) * 35)) - 1
    if e.ndim == 2:
        dim2 = e.shape[1]
    else:
        dim2 = 1
    earnings = (e * (wss * nssmat * factor_ss)).reshape(p.S, dim2)
    # get highest earning 35 years
    highest_35_earn =\
        (-1.0 * np.sort(-1.0 * earnings[:p.retire, :], axis=0))[:equiv_35]
    AIME = highest_35_earn.sum(0) / ((12.0 * (p.S / 80.0)) * equiv_35)
    PIA = np.zeros(dim2)
    # Bins from data for each level of replacement
    for j in range(dim2):
        if AIME[j] < 749.0:
            PIA[j] = .9 * AIME[j]
        elif AIME[j] < 4517.0:
            PIA[j] = 674.1 + .32 * (AIME[j] - 749.0)
        else:
            PIA[j] = 1879.86 + .15 * (AIME[j] - 4517.0)
    # Set the maximum monthly replacment rate from SS benefits tables
    maxpayment = 3501.00
    PIA[PIA > maxpayment] = maxpayment
    theta = (PIA * (12.0 * p.S / 80.0)) / (factor_ss * wss)
    return theta


def ETR_wealth(b, p):
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
    tau_w = (p.p_wealth * p.h_wealth * b) / (p.h_wealth * b + p.m_wealth)
    return tau_w


def MTR_wealth(b, p):
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
        tau_w_prime = [T,S,J] array, marginal tax rate on wealth from
                                     wealth tax
    Returns: tau_w_prime
    '''
    tau_prime = (p.h_wealth * p.m_wealth * p.p_wealth/
                 (b * p.h_wealth + p.m_wealth) ** 2)
    return tau_prime


def ETR_income(r, w, b, n, factor, e, etr_params, p):
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
    X = (w * e * n) * factor
    Y = (r * b) * factor
    X2 = X ** 2
    Y2 = Y ** 2
    I = X + Y
    I2 = I ** 2

    if p.tax_func_type == 'GS':
        phi0 = np.squeeze(etr_params[..., 0])
        phi1 = np.squeeze(etr_params[..., 1])
        phi2 = np.squeeze(etr_params[..., 2])
        tau = (phi0 * (I - ((I ** -phi1) + phi2) ** (-1 / phi1))) / I
    elif p.tax_func_type == 'DEP_totalinc':
        A = np.squeeze(etr_params[..., 0])
        B = np.squeeze(etr_params[..., 1])
        max_I = np.squeeze(etr_params[..., 4])
        min_I = np.squeeze(etr_params[..., 5])
        shift_I = np.squeeze(etr_params[..., 8])
        shift = np.squeeze(etr_params[..., 10])
        tau_I = (((max_I - min_I) * (A * I2 + B * I) /
                  (A * I2 + B * I + 1)) + min_I)
        tau = tau_I + shift_I + shift
    else:  # DEP or linear
        A = np.squeeze(etr_params[..., 0])
        B = np.squeeze(etr_params[..., 1])
        C = np.squeeze(etr_params[..., 2])
        D = np.squeeze(etr_params[..., 3])
        max_x = np.squeeze(etr_params[..., 4])
        min_x = np.squeeze(etr_params[..., 5])
        max_y = np.squeeze(etr_params[..., 6])
        min_y = np.squeeze(etr_params[..., 7])
        shift_x = np.squeeze(etr_params[..., 8])
        shift_y = np.squeeze(etr_params[..., 9])
        shift = np.squeeze(etr_params[..., 10])
        share = np.squeeze(etr_params[..., 11])

        tau_x = ((max_x - min_x) * (A * X2 + B * X) /
                 (A * X2 + B * X + 1) + min_x)
        tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
                 (C * Y2 + D * Y + 1) + min_y)
        tau = (((tau_x + shift_x) ** share) *
               ((tau_y + shift_y) ** (1 - share))) + shift

    return tau


def MTR_income(r, w, b, n, factor, mtr_capital, e, etr_params,
               mtr_params, p):
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
    params          = length 4 tuple, (e, mtry_params, tax_func_type,
                      analytical_mtrs)
    e               = [T,S,J] array, effective labor units
    mtr_params      = [T,S,J] array, marginal tax rate on labor/capital
                      income function parameters
    tax_func_type   = string, type of tax function used
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
    X = (w * e * n) * factor
    Y = (r * b) * factor
    X2 = X ** 2
    Y2 = Y ** 2
    I = X + Y
    I2 = I ** 2

    if p.tax_func_type == 'GS':
        if p.analytical_mtrs:
            phi0 = np.squeeze(etr_params[..., 0])
            phi1 = np.squeeze(etr_params[..., 1])
            phi2 = np.squeeze(etr_params[..., 2])
        else:
            phi0 = np.squeeze(mtr_params[..., 0])
            phi1 = np.squeeze(mtr_params[..., 1])
            phi2 = np.squeeze(mtr_params[..., 2])
        tau = (phi0*(1 - (I ** (-phi1 - 1) * ((I ** -phi1) + phi2)
                          ** ((-1 - phi1) / phi1))))
    elif p.tax_func_type == 'DEP_totalinc':
        if p.analytical_mtrs:
            A = np.squeeze(etr_params[..., 0])
            B = np.squeeze(etr_params[..., 1])
            max_I = np.squeeze(etr_params[..., 4])
            min_I = np.squeeze(etr_params[..., 5])
            shift_I = np.squeeze(etr_params[..., 8])
            shift = np.squeeze(etr_params[..., 10])
            d_etr = ((max_I - min_I) * ((2 * A * I + B) /
                     ((A * I2 + B * I + 1) ** 2)))
            etr = (((max_I - min_I) * ((A * I2 + B * I) /
                   (A * I2 + B * I + 1)) + min_I) + shift_I + shift)
            tau = (d_etr * I) + (etr)
        else:
            A = np.squeeze(mtr_params[..., 0])
            B = np.squeeze(mtr_params[..., 1])
            max_I = np.squeeze(mtr_params[..., 4])
            min_I = np.squeeze(mtr_params[..., 5])
            shift_I = np.squeeze(mtr_params[..., 8])
            shift = np.squeeze(mtr_params[..., 10])
            tau_I = (((max_I - min_I) * (A * I2 + B * I) /
                     (A * I2 + B * I + 1)) + min_I)
            tau = tau_I + shift_I + shift
    else:  # DEP or linear
        if p.analytical_mtrs:
            A = np.squeeze(etr_params[..., 0])
            B = np.squeeze(etr_params[..., 1])
            C = np.squeeze(etr_params[..., 2])
            D = np.squeeze(etr_params[..., 3])
            max_x = np.squeeze(etr_params[..., 4])
            min_x = np.squeeze(etr_params[..., 5])
            max_y = np.squeeze(etr_params[..., 6])
            min_y = np.squeeze(etr_params[..., 7])
            shift_x = np.squeeze(etr_params[..., 8])
            shift_y = np.squeeze(etr_params[..., 9])
            shift = np.squeeze(etr_params[..., 10])
            share = np.squeeze(etr_params[..., 11])

            tau_x = ((max_x - min_x) * (A * X2 + B * X) /
                     (A * X2 + B * X + 1) + min_x)
            tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
                     (C * Y2 + D * Y + 1) + min_y)
            etr = (((tau_x + shift_x) ** share) *
                       ((tau_y + shift_y) ** (1 - share))) + shift
            if mtr_capital:
                d_etr = ((1-share) * ((tau_y + shift_y) ** (-share)) *
                         (max_y - min_y) * ((2 * C * Y + D) /
                                            ((C * Y2 + D * Y + 1)
                                             ** 2)) *
                         ((tau_x + shift_x) ** share))
                tau = d_etr * I + etr
            else:
                d_etr = (share * ((tau_x + shift_x) ** (share - 1)) *
                         (max_x - min_x) * ((2 * A * X + B) /
                                            ((A * X2 + B * X + 1)
                                             ** 2)) *
                         ((tau_y + shift_y) ** (1 - share)))
                tau = d_etr * I + etr
        else:
            A = np.squeeze(mtr_params[..., 0])
            B = np.squeeze(mtr_params[..., 1])
            C = np.squeeze(mtr_params[..., 2])
            D = np.squeeze(mtr_params[..., 3])
            max_x = np.squeeze(mtr_params[..., 4])
            min_x = np.squeeze(mtr_params[..., 5])
            max_y = np.squeeze(mtr_params[..., 6])
            min_y = np.squeeze(mtr_params[..., 7])
            shift_x = np.squeeze(mtr_params[..., 8])
            shift_y = np.squeeze(mtr_params[..., 9])
            shift = np.squeeze(mtr_params[..., 10])
            share = np.squeeze(mtr_params[..., 11])

            tau_x = ((max_x - min_x) * (A * X2 + B * X) /
                     (A * X2 + B * X + 1) + min_x)
            tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
                     (C * Y2 + D * Y + 1) + min_y)
            tau = (((tau_x + shift_x) ** share) *
                   ((tau_y + shift_y) ** (1 - share))) + shift

    return tau


def get_biz_tax(w, Y, L, K, p):
    '''
    Finds total business income tax receipts
    Inputs:
        r           = [T,] vector, interest rate
        Y           = [T,] vector, aggregate output
        L           = [T,] vector, aggregate labor demand
        K           = [T,] vector, aggregate capital demand
    Objects in function:
        business_revenue    = [T,] vector, total revenue from business
                                           income taxes
    Returns: T_H

    '''
    business_revenue = p.tau_b * (Y - w * L) - p.tau_b * p.delta_tau * K
    return business_revenue


def total_taxes(r, w, b, n, BQ, factor, T_H, theta, j, shift, method,
                e, retire, etr_params, p):
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
        shift      = boolean, computing for periods 0--s or 1--(s+1)
                              (bool) (True for 1--(s+1))
        params = length 13 tuple, (e, lambdas, method, retire,
                                   etr_params, h_wealth, p_wealth,
                                   m_wealth, tau_payroll, theta, tau_bq,
                                   J, S)
        e           = [T,S,J] array, effective labor units
        lambdas     = [J,] vector, population weights by lifetime income group
        method      = string, 'SS' or 'TPI'
        retire      = integer, retirement age
        etr_params  = [T,S,J] array, effective tax rate function parameters
        h_wealth    = scalar, wealth tax function parameter
        p_wealth    = scalar, wealth tax function parameter
        m_wealth    = scalar, wealth tax function parameter
        tau_payroll = scalar, payroll tax rate
        theta       = [J,] vector, replacement rate values by lifetime
                                   income group
        tau_bq      = scalar, bequest tax rate
        S           = integer, number of age groups
        J           = integer, number of lifetime income groups
    Functions called:
        ETR_income
        ETR_wealth
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
    if j is not None:
        lambdas = p.lambdas[j]
    else:
        lambdas = np.transpose(p.lambdas)
        if method == 'TPI':
            r = utils.to_timepath_shape(r, p)
            w = utils.to_timepath_shape(w, p)
            T_H = utils.to_timepath_shape(T_H, p)

    I = r * b + w * e * n
    T_I = ETR_income(r, w, b, n, factor, e, etr_params, p) * I
    T_P = p.tau_payroll * w * e * n
    T_W = ETR_wealth(b, p) * b

    if method == 'SS':
        # Depending on if we are looking at b_s or b_s+1, the
        # entry for retirement will change (it shifts back one).
        # The shift boolean makes sure we start replacement rates
        # at the correct age.
        if shift is False:
            T_P[retire:] -= theta * w
        else:
            T_P[retire - 1:] -= theta * w
        T_BQ = p.tau_bq * (BQ / lambdas)
    elif method == 'TPI':
        if not shift:
            # retireTPI is different from retire, because in TPI we are
            # counting backwards with different length lists.  This will
            # always be the correct location of retirement, depending
            # on the shape of the lists.
            retireTPI = (retire - p.S)
        else:
            retireTPI = (retire - 1 - p.S)
        if len(b.shape) != 3:
            T_P[retireTPI:] -= theta[j] * w[retireTPI:]
        else:
            T_P[:, retire:, :] -= (theta.reshape(1, 1, p.J) * w)
        T_BQ = p.tau_bq * BQ / lambdas
    elif method == 'TPI_scalar':
        # The above methods won't work if scalars are used.  This option
        # is only called by the SS_TPI_firstdoughnutring function in TPI.
        T_P -= theta * w
        T_BQ = p.tau_bq * BQ / lambdas
    total_tax = T_I + T_P + T_BQ + T_W - T_H

    return total_tax
