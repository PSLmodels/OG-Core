from __future__ import print_function
'''
------------------------------------------------------------------------
Calculates steady state of OG-USA model with S age cohorts and J
ability types.

This py-file calls the following other file(s):
            tax.py
            household.py
            firm.py
            utils.py
            OUTPUT/SS/ss_vars.pkl

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/SS/ss_vars.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import scipy.optimize as opt
try:
    import cPickle as pickle
except ImportError:
    import pickle
from dask import compute, delayed
import dask.multiprocessing
from . import tax
from . import household
from . import aggregates as aggr
from . import firm
from . import utils
import os
import warnings


'''
Set minimizer tolerance
'''
MINIMIZER_TOL = 1e-13

'''
Set flag for enforcement of solution check
'''
ENFORCE_SOLUTION_CHECKS = True

'''
------------------------------------------------------------------------
    Define Functions
------------------------------------------------------------------------
'''


def euler_equation_solver(guesses, *args):
    '''
    --------------------------------------------------------------------
    Finds the euler errors for certain b and n, one ability type at a time.
    --------------------------------------------------------------------

    INPUTS:
    guesses = [2S,] vector, initial guesses for b and n
    r = scalar, real interest rate
    w = scalar, real wage rate
    T_H = scalar, lump sum transfer
    factor = scalar, scaling factor converting model units to dollars
    j = integer, ability group
    params = length 21 tuple, list of parameters
    chi_b = [J,] vector, chi^b_j, the utility weight on bequests
    chi_n = [S,] vector, chi^n_s utility weight on labor supply
    tau_bq = scalar, bequest tax rate
    rho = [S,] vector, mortality rates by age
    lambdas = [J,] vector, fraction of population with each ability type
    omega_SS = [S,] vector, stationary population weights
    e =  [S,J] array, effective labor units by age and ability type
    tax_params = length 5 tuple, (tax_func_type, analytical_mtrs,
                 etr_params, mtrx_params, mtry_params)
    tax_func_type   = string, type of tax function used
    analytical_mtrs = boolean, =True if use analytical_mtrs, =False if
                       use estimated MTRs
    etr_params      = [S,BW,#tax params] array, parameters for effective
                      tax rate function
    mtrx_params     = [S,BW,#tax params] array, parameters for marginal
                      tax rate on labor income function
    mtry_params     = [S,BW,#tax params] array, parameters for marginal
                      tax rate on capital income function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    aggr.get_BQ()
    tax.replacement_rate_vals()
    household.FOC_savings()
    household.FOC_labor()
    tax.total_taxes()
    household.get_cons()

    OBJECTS CREATED WITHIN FUNCTION:
    b_guess = [S,] vector, initial guess at household savings
    n_guess = [S,] vector, initial guess at household labor supply
    b_s = [S,] vector, wealth enter period with
    b_splus1 = [S,] vector, household savings
    BQ = scalar, aggregate bequests to lifetime income group
    theta = scalar, replacement rate for social security benenfits
    error1 = [S,] vector, errors from FOC for savings
    error2 = [S,] vector, errors from FOC for labor supply
    tax1 = [S,] vector, total income taxes paid
    cons = [S,] vector, household consumption

    RETURNS: 2Sx1 list of euler errors

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    (r, w, T_H, factor, j, p) = args

    b_guess = np.array(guesses[:p.S])
    n_guess = np.array(guesses[p.S:])
    b_s = np.array([0] + list(b_guess[:-1]))
    b_splus1 = b_guess

    BQ = aggr.get_BQ(r, b_splus1, j, p, 'SS', False)
    theta = tax.replacement_rate_vals(n_guess, w, factor, j, p)

    error1 = household.FOC_savings(r, w, b_s, b_splus1, n_guess, BQ,
                                   factor, T_H, theta, p.e[:, j], p.rho,
                                   p.retire, p.etr_params[-1, :, :],
                                   p.mtry_params[-1, :, :], j, p, 'SS')
    error2 = household.FOC_labor(r, w, b_s, b_splus1, n_guess, BQ,
                                 factor, T_H, theta, p.chi_n, p.e[:, j],
                                 p.retire, p.etr_params[-1, :, :],
                                 p.mtrx_params[-1, :, :], j, p, 'SS')

    # Put in constraints for consumption and savings.
    # According to the euler equations, they can be negative.  When
    # Chi_b is large, they will be.  This prevents that from happening.
    # I'm not sure if the constraints are needed for labor.
    # But we might as well put them in for now.
    mask1 = n_guess < 0
    mask2 = n_guess > p.ltilde
    mask3 = b_guess <= 0
    mask4 = np.isnan(n_guess)
    mask5 = np.isnan(b_guess)
    error2[mask1] = 1e14
    error2[mask2] = 1e14
    error1[mask3] = 1e14
    error1[mask5] = 1e14
    error2[mask4] = 1e14
    taxes = tax.total_taxes(r, w, b_s, n_guess, BQ, factor, T_H, theta,
                            j, False, 'SS', p.e[:, j], p.retire,
                            p.etr_params[-1, :, :], p)
    cons = household.get_cons(r, w, b_s, b_splus1, n_guess, BQ, taxes,
                              p.e[:, j], j, p)
    mask6 = cons < 0
    error1[mask6] = 1e14

    return np.hstack((error1, error2))


def inner_loop(outer_loop_vars, p, client):
    '''
    This function solves for the inner loop of
    the SS.  That is, given the guesses of the
    outer loop variables (r, w, Y, factor)
    this function solves the households'
    problems in the SS.

    Inputs:
        r          = [T,] vector, interest rate
        w          = [T,] vector, wage rate
        b          = [T,S,J] array, wealth holdings
        n          = [T,S,J] array, labor supply
        BQ         = [T,J] vector,  bequest amounts
        factor     = scalar, model income scaling factor
        Y        = [T,] vector, lump sum transfer amount(s)


    Functions called:
        euler_equation_solver()
        aggr.get_K()
        aggr.get_L()
        firm.get_Y()
        firm.get_r()
        firm.get_w()
        aggr.get_BQ()
        tax.replacement_rate_vals()
        aggr.revenue()

    Objects in function:


    Returns: euler_errors, bssmat, nssmat, new_r, new_w
             new_T_H, new_factor, new_BQ

    '''
    # unpack variables to pass to function
    if p.budget_balance:
        bssmat, nssmat, r, T_H, factor = outer_loop_vars
    else:
        bssmat, nssmat, r, Y, T_H, factor = outer_loop_vars

    euler_errors = np.zeros((2 * p.S, p.J))

    w = firm.get_w_from_r(r, p)

    lazy_values = []
    for j in range(p.J):
        guesses = np.append(bssmat[:, j], nssmat[:, j])
        euler_params = (r, w, T_H, factor, j, p)
        lazy_values.append(delayed(opt.fsolve)(euler_equation_solver,
                                               guesses * .9,
                                               args=euler_params,
                                               xtol=MINIMIZER_TOL,
                                               full_output=True))
    results = compute(*lazy_values, get=dask.multiprocessing.get,
                      num_workers=p.num_workers)

    # for j, result in results.items():
    for j, result in enumerate(results):
        [solutions, infodict, ier, message] = result
        euler_errors[:, j] = infodict['fvec']
        bssmat[:, j] = solutions[:p.S]
        nssmat[:, j] = solutions[p.S:]

    L = aggr.get_L(nssmat, p, 'SS')
    if not p.small_open:
        B = aggr.get_K(bssmat, p, 'SS', False)
        if p.budget_balance:
            K = B
        else:
            K = B - p.debt_ratio_ss * Y
    else:
        K = firm.get_K(L, r, p)
    new_Y = firm.get_Y(K, L, p)
    if p.budget_balance:
        Y = new_Y
    if not p.small_open:
        new_r = firm.get_r(Y, K, p)
        # new_w = firm.get_w_from_r(new_r, p)
    else:
        new_r = p.ss_hh_r
    new_w = firm.get_w_from_r(new_r, p)
    print('inner factor prices: ', new_r, new_w)

    b_s = np.array(list(np.zeros(p.J).reshape(1, p.J)) +
                   list(bssmat[:-1, :]))
    average_income_model = ((new_r * b_s + new_w * p.e * nssmat) *
                            p.omega_SS.reshape(p.S, 1) *
                            p.lambdas.reshape(1, p.J)).sum()
    if p.baseline:
        new_factor = p.mean_income_data / average_income_model
    else:
        new_factor = factor
    new_BQ = aggr.get_BQ(new_r, bssmat, None, p, 'SS', False)
    theta = tax.replacement_rate_vals(nssmat, new_w, new_factor, None, p)

    if p.budget_balance:
        etr_params_3D = np.tile(np.reshape(
            p.etr_params[-1, :, :], (p.S, 1, p.etr_params.shape[2])),
                                (1, p.J, 1))
        new_T_H = aggr.revenue(new_r, new_w, b_s, nssmat, new_BQ, new_Y,
                               L, K, factor, theta,
                               etr_params_3D, p, 'SS')
    elif p.baseline_spending:
        new_T_H = T_H
    else:
        new_T_H = p.alpha_T * new_Y

    return euler_errors, bssmat, nssmat, new_r, new_w, \
        new_T_H, new_Y, new_factor, new_BQ, average_income_model


def SS_solver(bmat, nmat, r, T_H, factor, Y, p, client,
              fsolve_flag=False):
    '''
    --------------------------------------------------------------------
    Solves for the steady state distribution of capital, labor, as well
    as w, r, T_H and the scaling factor, using a bisection method
    similar to TPI.
    --------------------------------------------------------------------

    INPUTS:
    b_guess_init = [S,J] array, initial guesses for savings
    n_guess_init = [S,J] array, initial guesses for labor supply
    wguess = scalar, initial guess for SS real wage rate
    rguess = scalar, initial guess for SS real interest rate
    T_Hguess = scalar, initial guess for lump sum transfer
    factorguess = scalar, initial guess for scaling factor to dollars
    chi_b = [J,] vector, chi^b_j, the utility weight on bequests
    chi_n = [S,] vector, chi^n_s utility weight on labor supply
    params = length X tuple, list of parameters
    iterative_params = length X tuple, list of parameters that determine
                       the convergence of the while loop
    tau_bq = [J,] vector, bequest tax rate
    rho = [S,] vector, mortality rates by age
    lambdas = [J,] vector, fraction of population with each ability type
    omega = [S,] vector, stationary population weights
    e =  [S,J] array, effective labor units by age and ability type


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    euler_equation_solver()
    aggr.get_K()
    aggr.get_L()
    firm.get_Y()
    firm.get_r()
    firm.get_w()
    aggr.get_BQ()
    tax.replacement_rate_vals()
    aggr.revenue()
    utils.convex_combo()
    utils.pct_diff_func()


    OBJECTS CREATED WITHIN FUNCTION:
    b_guess = [S,] vector, initial guess at household savings
    n_guess = [S,] vector, initial guess at household labor supply
    b_s = [S,] vector, wealth enter period with
    b_splus1 = [S,] vector, household savings
    b_splus2 = [S,] vector, household savings one period ahead
    BQ = scalar, aggregate bequests to lifetime income group
    theta = scalar, replacement rate for social security benenfits
    error1 = [S,] vector, errors from FOC for savings
    error2 = [S,] vector, errors from FOC for labor supply
    tax1 = [S,] vector, total income taxes paid
    cons = [S,] vector, household consumption

    OBJECTS CREATED WITHIN FUNCTION - SMALL OPEN ONLY
    Bss = scalar, aggregate household wealth in the steady state
    BIss = scalar, aggregate household net investment in the steady state

    RETURNS: solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    # Rename the inputs
    if not p.budget_balance:
        if not p.baseline_spending:
            Y = T_H / p.alpha_T
    if p.small_open:
        r = p.ss_hh_r

    dist = 10
    iteration = 0
    dist_vec = np.zeros(p.maxiter)
    maxiter_ss = p.maxiter
    nu_ss = p.nu

    if fsolve_flag:
        maxiter_ss = 1

    while (dist > p.mindist_SS) and (iteration < maxiter_ss):
        # Solve for the steady state levels of b and n, given w, r,
        # Y and factor
        if p.budget_balance:
            outer_loop_vars = (bmat, nmat, r, T_H, factor)
        else:
            outer_loop_vars = (bmat, nmat, r, Y, T_H, factor)

        (euler_errors, new_bmat, new_nmat, new_r, new_w, new_T_H, new_Y,
         new_factor, new_BQ, average_income_model) =\
            inner_loop(outer_loop_vars, p, client)

        r = utils.convex_combo(new_r, r, nu_ss)
        factor = utils.convex_combo(new_factor, factor, nu_ss)
        # bmat = utils.convex_combo(new_bmat, bmat, nu_ss)
        # nmat = utils.convex_combo(new_nmat, nmat, nu_ss)
        if p.budget_balance:
            T_H = utils.convex_combo(new_T_H, T_H, nu_ss)
            dist = np.array([utils.pct_diff_func(new_r, r)] +
                            [utils.pct_diff_func(new_T_H, T_H)] +
                            [utils.pct_diff_func(new_factor, factor)]).max()
        else:
            Y = utils.convex_combo(new_Y, Y, nu_ss)
            if Y != 0:
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                [utils.pct_diff_func(new_Y, Y)] +
                                [utils.pct_diff_func(new_factor,
                                                     factor)]).max()
            else:
                # If Y is zero (if there is no output), a percent difference
                # will throw NaN's, so we use an absoluate difference
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                [abs(new_Y - Y)] +
                                [utils.pct_diff_func(new_factor,
                                                     factor)]).max()
        dist_vec[iteration] = dist
        # Similar to TPI: if the distance between iterations increases, then
        # decrease the value of nu to prevent cycling
        if iteration > 10:
            if dist_vec[iteration] - dist_vec[iteration - 1] > 0:
                nu_ss /= 2.0
                print('New value of nu:', nu_ss)
        iteration += 1
        print("Iteration: %02d" % iteration, " Distance: ", dist)

    '''
    ------------------------------------------------------------------------
        Generate the SS values of variables, including euler errors
    ------------------------------------------------------------------------
    '''
    bssmat_s = np.append(np.zeros((1, p.J)), bmat[:-1, :], axis=0)
    bssmat_splus1 = bmat
    nssmat = nmat

    rss = r
    wss = new_w
    factor_ss = factor
    T_Hss = T_H

    Lss = aggr.get_L(nssmat, p, 'SS')
    if not p.small_open:
        Bss = aggr.get_K(bssmat_splus1, p, 'SS', False)
        if p.budget_balance:
            debt_ss = 0.0
        else:
            debt_ss = p.debt_ratio_ss * Y
        Kss = Bss - debt_ss
        Iss = aggr.get_I(bssmat_splus1, Kss, Kss, p, 'SS')
    else:
        # Compute capital (K) and wealth (B) separately
        Kss = firm.get_K(Lss, p.ss_firm_r, p)
        InvestmentPlaceholder = np.zeros(bssmat_splus1.shape)
        Iss = aggr.get_I(InvestmentPlaceholder, Kss, Kss, p, 'SS')
        Bss = aggr.get_K(bssmat_splus1, p, 'SS', False)
        BIss = aggr.get_I(bssmat_splus1, Bss, Bss, p, 'BI_SS') #+ p.delta * Bss

        # BIss_params = (0.0, g_y, omega_SS, lambdas, imm_rates,
        #                g_n_ss, 'SS')
        # BIss = aggr.get_I(bssmat_splus1, Bss, Bss, BIss_params)


        if p.budget_balance:
            debt_ss = 0.0
        else:
            debt_ss = p.debt_ratio_ss * Y

    Yss = firm.get_Y(Kss, Lss, p)
    BQss = new_BQ
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, None, p)

    etr_params_3D = np.tile(np.reshape(
        p.etr_params[-1, :, :], (p.S, 1, p.etr_params.shape[2])), (1, p.J, 1))
    revenue_ss = aggr.revenue(rss, wss, bssmat_s, nssmat, BQss, Yss,
                              Lss, Kss, factor, theta,
                              etr_params_3D, p, 'SS')
    r_gov_ss = rss
    debt_service_ss = r_gov_ss * p.debt_ratio_ss * Yss
    new_borrowing = p.debt_ratio_ss * Yss * ((1 + p.g_n_ss) *
                                             np.exp(p.g_y) - 1)
    # government spends such that it expands its debt at the same rate as GDP
    if p.budget_balance:
        Gss = 0.0
    else:
        Gss = revenue_ss + new_borrowing - (T_Hss + debt_service_ss)
        print('G components = ', new_borrowing, T_Hss, debt_service_ss)

    '''
    ------------------------------------------------------------------------
        The code below is to calulate and save model MTRs
                - only exists to help debug
    ------------------------------------------------------------------------
    '''
    # etr_params_extended = np.append(etr_params,
    #                                 np.reshape(etr_params[-1, :],
    #                                            (1, etr_params.shape[1])),
    #                                 axis=0)[1:, :]
    # etr_params_extended_3D = np.tile(np.reshape(etr_params_extended,
    #                                             (S, 1,
    #                                              etr_params_extended.shape[1])),
    #                                  (1, J, 1))
    # mtry_params_extended = np.append(mtry_params,
    #                                  np.reshape(mtry_params[-1, :],
    #                                             (1,
    #                                              mtry_params.shape[1])),
    #                                  axis=0)[1:, :]
    # mtry_params_extended_3D = np.tile(np.reshape(mtry_params_extended,
    #                                              (S, 1,
    #                                               mtry_params_extended.shape[1])),
    #                                   (1, J, 1))
    # e_extended = np.array(list(e) + list(np.zeros(J).reshape(1, J)))
    # nss_extended = np.array(list(nssmat) + list(np.zeros(J).reshape(1, J)))
    # mtry_ss_params = (e_extended[1:, :], etr_params_extended_3D,
    #                   mtry_params_extended_3D, tax_func_type,
    #                   analytical_mtrs)
    # mtry_ss = tax.MTR_income(rss, wss, bssmat_splus1,
    #                           nss_extended[1:, :], factor_ss,
    #                           mtry_ss_params)
    # mtrx_ss_params = (e, etr_params_3D, mtrx_params_3D, tax_func_type,
    #                   analytical_mtrs)
    # mtrx_ss = tax.MTR_income(rss, wss, bssmat_s, nssmat, factor_ss,
    #                         mtrx_ss_params)

    # np.savetxt("mtr_ss_capital.csv", mtry_ss, delimiter=",")
    # np.savetxt("mtr_ss_labor.csv", mtrx_ss, delimiter=",")

    # Compute total investment (not just domestic)
    Iss_total = p.delta * Kss

    # solve resource constraint
    taxss = tax.total_taxes(rss, wss, bssmat_s, nssmat, BQss, factor_ss,
                            T_Hss, theta, None, False, 'SS',
                            p.e, p.retire, etr_params_3D, p)
    cssmat = household.get_cons(rss, wss, bssmat_s, bssmat_splus1,
                                nssmat, BQss.reshape(1, p.J), taxss,
                                p.e, None, p)

    business_revenue = tax.get_biz_tax(wss, Yss, Lss, Kss, p)
    IITpayroll_revenue = revenue_ss - business_revenue

    Css = aggr.get_C(cssmat, p, 'SS')

    # solve resource constraint
    if p.small_open:
        # include term for current account
        resource_constraint = (Yss + new_borrowing - (Css + BIss + Gss)
                               + (p.ss_hh_r * Bss - (p.delta + p.ss_firm_r) *
                                  Kss - debt_service_ss))
        print('Yss= ', Yss, '\n', 'Css= ', Css, '\n', 'Bss = ', Bss,
              '\n', 'BIss = ', BIss, '\n', 'Kss = ', Kss, '\n', 'Iss = ',
              Iss, '\n', 'Lss = ', Lss, '\n', 'T_H = ', T_H, '\n',
              'Gss= ', Gss)
        print('D/Y:', debt_ss / Yss, 'T/Y:', T_Hss / Yss, 'G/Y:',
              Gss / Yss, 'Rev/Y:', revenue_ss / Yss,
              'Int payments to GDP:', (rss * debt_ss) / Yss)
        print('resource constraint: ', resource_constraint)
    else:
        resource_constraint = Yss - (Css + Iss + Gss)
        print('Yss= ', Yss, '\n', 'Gss= ', Gss, '\n', 'Css= ', Css, '\n',
              'Kss = ', Kss, '\n', 'Iss = ', Iss, '\n', 'Lss = ', Lss,
              '\n', 'Debt service = ', debt_service_ss)
        print('D/Y:', debt_ss / Yss, 'T/Y:', T_Hss / Yss, 'G/Y:',
              Gss / Yss, 'Rev/Y:', revenue_ss / Yss, 'business rev/Y: ',
              business_revenue / Yss, 'Int payments to GDP:',
              (rss * debt_ss) / Yss)
        print('Check SS budget: ', Gss - (np.exp(p.g_y) *
                                          (1 + p.g_n_ss) - 1 - rss) *
              debt_ss - revenue_ss + T_Hss)
        print('resource constraint: ', resource_constraint)

    if Gss < 0:
        print('Steady state government spending is negative to satisfy'
              + ' budget')

    if ENFORCE_SOLUTION_CHECKS and (np.absolute(resource_constraint) >
                                    p.mindist_SS):
        print('Resource Constraint Difference:', resource_constraint)
        err = "Steady state aggregate resource constraint not satisfied"
        raise RuntimeError(err)

    # check constraints
    household.constraint_checker_SS(bssmat_splus1, nssmat, cssmat, p.ltilde)

    euler_savings = euler_errors[:p.S, :]
    euler_labor_leisure = euler_errors[p.S:, :]
    print('Maximum error in labor FOC = ',
          np.absolute(euler_labor_leisure).max())
    print('Maximum error in savings FOC = ',
          np.absolute(euler_savings).max())

    '''
    ------------------------------------------------------------------------
        Return dictionary of SS results
    ------------------------------------------------------------------------
    '''
    output = {'Kss': Kss, 'Bss': Bss, 'Lss': Lss, 'Css': Css, 'Iss': Iss,
              'Iss_total': Iss_total, 'nssmat': nssmat, 'Yss': Yss,
              'Dss': debt_ss, 'wss': wss, 'rss': rss, 'theta': theta,
              'BQss': BQss, 'factor_ss': factor_ss,
              'bssmat': bssmat_splus1, 'bssmat_s': bssmat_s,
              'cssmat': cssmat, 'bssmat_splus1': bssmat_splus1,
              'T_Hss': T_Hss, 'Gss': Gss, 'revenue_ss': revenue_ss,
              'business_revenue': business_revenue,
              'IITpayroll_revenue': IITpayroll_revenue,
              'euler_savings': euler_savings,
              'euler_labor_leisure': euler_labor_leisure, 'chi_n': p.chi_n,
              'chi_b': p.chi_b}

    return output


def SS_fsolve(guesses, *args):
    '''
    Solves for the steady state distribution of capital, labor, as well as
    w, r, T_H and the scaling factor, using a root finder.
    Inputs:
        b_guess_init = guesses for b (SxJ array)
        n_guess_init = guesses for n (SxJ array)
        wguess = guess for wage rate (scalar)
        rguess = guess for rental rate (scalar)
        T_Hguess = guess for lump sum tax (scalar)
        factorguess = guess for scaling factor to dollars (scalar)
        chi_n = chi^n_s (Sx1 array)
        chi_b = chi^b_j (Jx1 array)
        params = list of parameters (list)
        iterative_params = list of parameters that determine the convergence
                           of the while loop (list)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortality rates (Sx1 array)
        lambdas = ability weights (Jx1 array)
        omega_SS = population weights (Sx1 array)
        e = ability levels (SxJ array)
    Outputs:
        solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)
    '''
    (bssmat, nssmat, p, client) = args

    # Rename the inputs
    r = guesses[0]
    T_H = guesses[1]
    factor = guesses[2]

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    if p.budget_balance:
        outer_loop_vars = (bssmat, nssmat, r, T_H, factor)
    else:
        Y = T_H / p.alpha_T
        outer_loop_vars = (bssmat, nssmat, r, Y, T_H, factor)
    (euler_errors, bssmat, nssmat, new_r, new_w, new_T_H, new_Y,
     new_factor, new_BQ, average_income_model) =\
        inner_loop(outer_loop_vars, p, client)

    error1 = new_r - r
    if p.budget_balance:
        error2 = new_T_H - T_H
    else:
        error2 = new_Y - Y
    error3 = new_factor / 1000000 - factor / 1000000

    # Check and punish violations
    if r + p.delta <= 0:
        error1 = 1e9
    # if r > 1:
    #    error1 += 1e9
    if factor <= 0:
        error3 = 1e9

    print('errors: ', error1, error2, error3)

    return [error1, error2, error3]


def SS_fsolve_reform(guesses, *args):
    '''
    Solves for the steady state distribution of capital, labor, as well
    as w, r, and T_H and the scaling factor, using a root finder. This
    solves for the reform SS and so takes the factor from the baseline
    SS as an input.
    Inputs:
        b_guess_init = guesses for b (SxJ array)
        n_guess_init = guesses for n (SxJ array)
        wguess = guess for wage rate (scalar)
        rguess = guess for rental rate (scalar)
        T_Hguess = guess for lump sum tax (scalar)
        factor = scaling factor to dollars (scalar)
        chi_n = chi^n_s (Sx1 array)
        chi_b = chi^b_j (Jx1 array)
        params = list of parameters (list)
        iterative_params = list of parameters that determine the
                           convergence of the while loop (list)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortality rates (Sx1 array)
        lambdas = ability weights (Jx1 array)
        omega_SS = population weights (Sx1 array)
        e = ability levels (SxJ array)
    Outputs:
        solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)
    '''
    (bssmat, nssmat, factor, p, client) = args
    print('Bssmat 2 in max and min = ', bssmat.max(), bssmat.min())
    print('Nssmat 2 in max and min = ', nssmat.max(), nssmat.min())

    # Rename the inputs
    r = guesses[0]
    T_H = guesses[1]

    # Solve for the steady state levels of b and n, given w, r, T_H
    if p.budget_balance:
        outer_loop_vars = (bssmat, nssmat, r, T_H, factor)
    else:
        Y = T_H / p.alpha_T
        outer_loop_vars = (bssmat, nssmat, r, Y, T_H, factor)

    (euler_errors, bssmat, nssmat, new_r, new_w, new_T_H, new_Y,
     new_factor, new_BQ, average_income_model) =\
        inner_loop(outer_loop_vars, p, client)

    error1 = new_r - r
    if p.budget_balance:
        error2 = new_T_H - T_H
    else:
        error2 = new_Y - Y

    print('errors: ', error1, error2)

    # Check and punish violations
    if r + p.delta <= 0:
        error1 = 1e9
    # if r > 1:
    #    error1 += 1e9

    return [error1, error2]


def SS_fsolve_reform_baselinespend(guesses, *args):
    '''
    Solves for the steady state distribution of capital, labor, as
    well as w, r, and Y, using a root finder. This solves for the
    reform SS when baseline_spending=True and so takes the factor and
    gov't transfers (T_H) from the baseline SS as an input.
    Inputs:
        b_guess_init = guesses for b (SxJ array)
        n_guess_init = guesses for n (SxJ array)
        wguess = guess for wage rate (scalar)
        rguess = guess for rental rate (scalar)
        T_Hguess = guess for lump sum tax (scalar)
        factor = scaling factor to dollars (scalar)
        chi_n = chi^n_s (Sx1 array)
        chi_b = chi^b_j (Jx1 array)
        params = list of parameters (list)
        iterative_params = list of parameters that determine the
                           convergence of the while loop (list)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortality rates (Sx1 array)
        lambdas = ability weights (Jx1 array)
        omega_SS = population weights (Sx1 array)
        e = ability levels (SxJ array)
    Outputs:
        solutions = steady state values of b, n, w, r, factor,
                    T_H ((2*S*J+4)x1 array)
    '''
    (bssmat, nssmat, T_Hss, factor_ss, p, client) = args

    # Rename the inputs
    r = guesses[0]
    Y = guesses[1]

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    outer_loop_vars = (bssmat, nssmat, r, Y, T_Hss, factor_ss)
    (euler_errors, bssmat, nssmat, new_r, new_w, new_T_H, new_Y,
     new_factor, new_BQ, average_income_model) =\
        inner_loop(outer_loop_vars, p, client)

    error1 = new_r - r
    error2 = new_Y - Y

    print('errors: ', error1, error2)

    # Check and punish violations
    if r + p.delta <= 0:
        error1 = 1e9
    # if r > 1:
    #    error1 += 1e9

    return [error1, error2]


def run_SS(p, client=None):
    '''
    --------------------------------------------------------------------
    Solve for SS of OG-USA.
    --------------------------------------------------------------------

    INPUTS:
    p = Specifications class with parameterization of model
    income_tax_parameters = length 5 tuple, (tax_func_type,
                            analytical_mtrs, etr_params,
                            mtrx_params, mtry_params)
    ss_parameters = length 21 tuple, (J, S, T, BW, beta, sigma, alpha,
                    gamma, epsilon, Z, delta, ltilde, nu, g_y, g_n_ss,
                    tau_payroll, retire, mean_income_data, h_wealth,
                    p_wealth, m_wealth, b_ellipse, upsilon)
    iterative_params  = [2,] vector, vector with max iterations and
                        tolerance for SS solution
    baseline = boolean, =True if run is for baseline tax policy
    calibrate_model = boolean, =True if run calibration of chi parameters
    output_dir = string, path to save output from current model run
    baseline_dir = string, path where baseline results located


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    SS_fsolve()
    SS_fsolve_reform()
    SS_solver

    OBJECTS CREATED WITHIN FUNCTION:
    chi_params = [J+S,] vector, chi_b and chi_n stacked together
    b_guess = [S,J] array, initial guess at savings
    n_guess = [S,J] array, initial guess at labor supply
    wguess = scalar, initial guess at SS real wage rate
    rguess = scalar, initial guess at SS real interest rate
    T_Hguess = scalar, initial guess at SS lump sum transfers
    factorguess = scalar, initial guess at SS factor adjustment (to
                  scale model units to dollars)

    output


    RETURNS: output

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    b_guess = np.ones((p.S, p.J)) * 0.05
    n_guess = np.ones((p.S, p.J)) * .4 * p.ltilde
    # For initial guesses of w, r, T_H, and factor, we use values that
    # are close to some steady state values.
    if p.baseline:
        rguess = 0.04  # 0.01 + delta
        T_Hguess = 0.12
        factorguess = 70000

        ss_params_baseline = (b_guess, n_guess, p, client)
        guesses = [rguess, T_Hguess, factorguess]
        [solutions_fsolve, infodict, ier, message] =\
            opt.fsolve(SS_fsolve, guesses, args=ss_params_baseline,
                       xtol=p.mindist_SS, full_output=True)
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            raise RuntimeError("Steady state equilibrium not found")
        [rss, T_Hss, factor_ss] = solutions_fsolve
        Yss = T_Hss/p.alpha_T  # may not be right - if budget_balance
        # = True, but that's ok - will be fixed in SS_solver
        fsolve_flag = True
        # Return SS values of variables
        output = SS_solver(b_guess, n_guess, rss, T_Hss, factor_ss, Yss, p,
                           client, fsolve_flag)
    else:
        baseline_ss_dir = os.path.join(p.baseline_dir, "SS/SS_vars.pkl")
        ss_solutions = pickle.load(open(baseline_ss_dir, "rb"),
                                   encoding="latin1")
        (rguess, T_Hguess, Yguess, factor) =\
            (ss_solutions['rss'], ss_solutions['T_Hss'],
             ss_solutions['Yss'], ss_solutions['factor_ss'])
        if p.baseline_spending:
            T_Hss = T_Hguess
            ss_params_reform = (b_guess, n_guess, T_Hss, factor, p, client)
            guesses = [rguess, Yguess]
            [solutions_fsolve, infodict, ier, message] =\
                opt.fsolve(SS_fsolve_reform_baselinespend, guesses,
                           args=ss_params_reform, xtol=p.mindist_SS,
                           full_output=True)
            [rss, Yss] = solutions_fsolve
        else:
            ss_params_reform = (b_guess, n_guess, factor, p, client)
            guesses = [rguess, T_Hguess]
            [solutions_fsolve, infodict, ier, message] =\
                opt.fsolve(SS_fsolve_reform, guesses,
                           args=ss_params_reform, xtol=p.mindist_SS,
                           full_output=True)
            [rss, T_Hss] = solutions_fsolve
            Yss = T_Hss/p.alpha_T  # may not be right - if
            # budget_balance = True, but that's ok - will be fixed in
            # SS_solver
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            raise RuntimeError("Steady state equilibrium not found")
        # Return SS values of variables
        fsolve_flag = True
        # Return SS values of variables
        output = SS_solver(b_guess, n_guess, rss, T_Hss, factor, Yss, p,
                           client, fsolve_flag)
        if output['Gss'] < 0.:
            warnings.warn('Warning: The combination of the tax policy '
                          + 'you specified and your target debt-to-GDP '
                          + 'ratio results in an infeasible amount of '
                          + 'government spending in order to close the '
                          + 'budget (i.e., G < 0)')
    return output
