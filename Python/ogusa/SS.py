'''
------------------------------------------------------------------------
Last updated: 12/20/2016

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
import cPickle as pickle

from . import tax
from . import household
import firm
import utils
import os


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

def create_steady_state_parameters(**sim_params):
    '''
    --------------------------------------------------------------------
    This function calls the tax function estimation routine and saves
    the resulting dictionary in pickle files corresponding to the
    baseline or reform policy.
    --------------------------------------------------------------------

    INPUTS:
    sim_params       = dictionary, dict containing variables for simulation
    analytical_mtrs  = boolean, =True if use analytical_mtrs, =False if
                       use estimated MTRs
    etr_params       = [S,BW,#tax params] array, parameters for effective tax rate function
    mtrx_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                       labor income function
    mtry_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                       capital income function
    b_ellipse        = scalar, value of b for elliptical fit of utility function
    upsilon          = scalar, value of omega for elliptical fit of utility function
    S                = integer, number of economically active periods an individual lives
    J                = integer, number of different ability groups
    T                = integer, number of time periods until steady state is reached
    BW               = integer, number of time periods in the budget window
    beta             = scalar, discount factor for model period
    sigma            = scalar, coefficient of relative risk aversion
    alpha            = scalar, capital share of income
    Z                = scalar, total factor productivity parameter in firms' production
                       function
    ltilde           = scalar, measure of time each individual is endowed with each
                       period
    nu               = scalar, contraction parameter in SS and TPI iteration process
                       representing the weight on the new distribution
    g_y              = scalar, growth rate of technology for a model period
    tau_payroll      = scalar, payroll tax rate
    alpha_T          = scalar, share of GDP remitted in transfers
    debt_ratio_ss    = scalar, steady state debt/GDP
    retire           = integer, age at which individuals eligible for retirement benefits
    mean_income_data = scalar, mean income from IRS data file used to calibrate income tax
    run_params       = ???
    output_dir       = string, directory for output files to be saved


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    income_tax_params = length 3 tuple, (analytical_mtrs, etr_params,
                        mtrx_params,mtry_params)
    wealth_tax_params = [3,] vector, contains values of three parameters
                        of wealth tax function
    ellipse_params    = [2,] vector, vector with b_ellipse and upsilon
                        paramters of elliptical utility
    parameters        = length 3 tuple, ([15,] vector of general model
                        params, wealth_tax_params, ellipse_params)
    iterative_params  = [2,] vector, vector with max iterations and tolerance
                        for SS solution

    RETURNS: (income_tax_params, wealth_tax_params, ellipse_params,
            parameters, iterative_params)

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    # Put income tax parameters in a tuple
    # Assumption here is that tax parameters of last year of budget
    # window continue forever and so will be SS values
    income_tax_params = (sim_params['analytical_mtrs'], sim_params['etr_params'][:,-1,:],
                         sim_params['mtrx_params'][:,-1,:],sim_params['mtry_params'][:,-1,:])

    # Make a vector of all one dimensional parameters, to be used in the
    # following functions
    wealth_tax_params = [sim_params['h_wealth'], sim_params['p_wealth'], sim_params['m_wealth']]
    ellipse_params = [sim_params['b_ellipse'], sim_params['upsilon']]

    if sim_params['budget_balance']:
        sim_params['debt_ratio_ss'] = 0.0

    ss_params = [sim_params['J'], sim_params['S'], sim_params['T'], sim_params['BW'],
                  sim_params['beta'], sim_params['sigma'], sim_params['alpha'],
                  sim_params['gamma'], sim_params['epsilon'],
                  sim_params['Z'], sim_params['delta'], sim_params['ltilde'],
                  sim_params['nu'], sim_params['g_y'], sim_params['g_n_ss'],
                  sim_params['tau_payroll'], sim_params['tau_bq'], sim_params['rho'], sim_params['omega_SS'],
                  sim_params['budget_balance'], sim_params['alpha_T'], sim_params['debt_ratio_ss'],
                  sim_params['tau_b'], sim_params['delta_tau'],
                  sim_params['lambdas'], sim_params['imm_rates'][-1,:], sim_params['e'], sim_params['retire'], sim_params['mean_income_data']] + \
                  wealth_tax_params + ellipse_params
    iterative_params = [sim_params['maxiter'], sim_params['mindist_SS']]
    chi_params = (sim_params['chi_b_guess'], sim_params['chi_n_guess'])
    small_open_params = [sim_params['small_open'], sim_params['ss_firm_r'], sim_params['ss_hh_r']]
    return (income_tax_params, ss_params, iterative_params, chi_params, small_open_params)


def euler_equation_solver(guesses, params):
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
    tax_params = length 4 tuple, (analytical_mtrs, etr_params, mtrx_params, mtry_params)
    analytical_mtrs = boolean, =True if use analytical_mtrs, =False if
                       use estimated MTRs
    etr_params      = [S,BW,#tax params] array, parameters for effective tax rate function
    mtrx_params     = [S,BW,#tax params] array, parameters for marginal tax rate on
                       labor income function
    mtry_params     = [S,BW,#tax params] array, parameters for marginal tax rate on
                       capital income function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    household.get_BQ()
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
    b_splus2 = [S,] vector, household savings one period ahead
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

    r, w, T_H, factor, j, J, S, beta, sigma, ltilde, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon,\
                  j, chi_b, chi_n, tau_bq, rho, lambdas, omega_SS, e,\
                  analytical_mtrs, etr_params, mtrx_params,\
                  mtry_params = params

    b_guess = np.array(guesses[:S])
    n_guess = np.array(guesses[S:])
    b_s = np.array([0] + list(b_guess[:-1]))
    b_splus1 = b_guess
    b_splus2 = np.array(list(b_guess[1:]) + [0])

    BQ_params = (omega_SS, lambdas[j], rho, g_n_ss, 'SS')
    BQ = household.get_BQ(r, b_splus1, BQ_params)
    theta_params = (e[:,j], S, retire)
    theta = tax.replacement_rate_vals(n_guess, w, factor, theta_params)

    foc_save_parms = (e[:, j], sigma, beta, g_y, chi_b[j], theta, tau_bq[j], rho, lambdas[j], J, S,
                           analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, 'SS')
    error1 = household.FOC_savings(r, w, b_s, b_splus1, b_splus2, n_guess, BQ, factor, T_H, foc_save_parms)
    foc_labor_params = (e[:, j], sigma, g_y, theta, b_ellipse, upsilon, chi_n, ltilde, tau_bq[j], lambdas[j], J, S,
                            analytical_mtrs, etr_params, mtrx_params, h_wealth, p_wealth, m_wealth, tau_payroll, retire, 'SS')
    error2 = household.FOC_labor(r, w, b_s, b_splus1, n_guess, BQ, factor, T_H, foc_labor_params)

    # Put in constraints for consumption and savings.
    # According to the euler equations, they can be negative.  When
    # Chi_b is large, they will be.  This prevents that from happening.
    # I'm not sure if the constraints are needed for labor.
    # But we might as well put them in for now.
    mask1 = n_guess < 0
    mask2 = n_guess > ltilde
    mask3 = b_guess <= 0
    mask4 = np.isnan(n_guess)
    mask5 = np.isnan(b_guess)
    error2[mask1] = 1e14
    error2[mask2] = 1e14
    error1[mask3] = 1e14
    error1[mask5] = 1e14
    error2[mask4] = 1e14

    tax1_params = (e[:, j], lambdas[j], 'SS', retire, etr_params, h_wealth, p_wealth,
                   m_wealth, tau_payroll, theta, tau_bq[j], J, S)
    tax1 = tax.total_taxes(r, w, b_s, n_guess, BQ, factor, T_H, None, False, tax1_params)
    cons_params = (e[:, j], lambdas[j], g_y)
    cons = household.get_cons(r, w, b_s, b_splus1, n_guess, BQ, tax1, cons_params)
    mask6 = cons < 0
    error1[mask6] = 1e14

    return list(error1.flatten()) + list(error2.flatten())


def inner_loop(outer_loop_vars, params, baseline, baseline_spending=False):
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
        household.get_K()
        firm.get_L()
        firm.get_Y()
        firm.get_r()
        firm.get_w()
        household.get_BQ()
        tax.replacement_rate_vals()
        tax.revenue()

    Objects in function:


    Returns: euler_errors, bssmat, nssmat, new_r, new_w
             new_T_H, new_factor, new_BQ

    '''

    # unpack variables and parameters pass to function
    ss_params, income_tax_params, chi_params, small_open_params = params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, budget_balance, \
                  alpha_T, debt_ratio_ss, tau_b, delta_tau,\
                  lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    chi_b, chi_n = chi_params

    small_open, ss_firm_r, ss_hh_r = small_open_params
    if budget_balance:
        bssmat, nssmat, r, w, T_H, factor = outer_loop_vars
    else:
        bssmat, nssmat, r, w, Y, T_H, factor = outer_loop_vars

    euler_errors = np.zeros((2*S,J))



    for j in xrange(J):
        # Solve the euler equations
        if j == 0:
            guesses = np.append(bssmat[:, j], nssmat[:, j])
        else:
            guesses = np.append(bssmat[:, j-1], nssmat[:, j-1])
        euler_params = [r, w, T_H, factor, j, J, S, beta, sigma, ltilde, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon,\
                  j, chi_b, chi_n, tau_bq, rho, lambdas, omega_SS, e,\
                  analytical_mtrs, etr_params, mtrx_params,\
                  mtry_params]

        [solutions, infodict, ier, message] = opt.fsolve(euler_equation_solver, guesses * .9,
                                   args=euler_params, xtol=MINIMIZER_TOL, full_output=True)

        euler_errors[:,j] = infodict['fvec']
      #  print 'Max Euler errors: ', np.absolute(euler_errors[:,j]).max()

        bssmat[:, j] = solutions[:S]
        nssmat[:, j] = solutions[S:]

    L_params = (e, omega_SS.reshape(S, 1), lambdas.reshape(1, J), 'SS')
    L = firm.get_L(nssmat, L_params)
    if small_open == False:
        K_params = (omega_SS.reshape(S, 1), lambdas.reshape(1, J), imm_rates, g_n_ss, 'SS')
        B = household.get_K(bssmat, K_params)
        if budget_balance:
            K = B
        else:
            K = B - debt_ratio_ss*Y
    else:
        K_params = (Z, gamma, epsilon, delta, tau_b, delta_tau)
        K = firm.get_K(L, ss_firm_r, K_params)
    # Y_params = (alpha, Z)
    Y_params = (Z, gamma, epsilon)
    new_Y = firm.get_Y(K, L, Y_params)
    #print 'inner K, L, Y: ', K, L, new_Y
    if budget_balance:
        Y = new_Y
    if small_open == False:
        r_params = (Z, gamma, epsilon, delta, tau_b, delta_tau)
        new_r = firm.get_r(Y, K, r_params)
    else:
        new_r = ss_hh_r
    w_params = (Z, gamma, epsilon)
    new_w = firm.get_w(Y, L, w_params)
    print 'inner factor prices: ', new_r, new_w

    b_s = np.array(list(np.zeros(J).reshape(1, J)) + list(bssmat[:-1, :]))
    average_income_model = ((new_r * b_s + new_w * e * nssmat) *
                            omega_SS.reshape(S, 1) *
                            lambdas.reshape(1, J)).sum()
    if baseline:
        new_factor = mean_income_data / average_income_model
    else:
        new_factor = factor

    BQ_params = (omega_SS.reshape(S, 1), lambdas.reshape(1, J), rho.reshape(S, 1), g_n_ss, 'SS')
    new_BQ = household.get_BQ(new_r, bssmat, BQ_params)
    theta_params = (e, S, retire)
    theta = tax.replacement_rate_vals(nssmat, new_w, new_factor, theta_params)

    if budget_balance:
        T_H_params = (e, lambdas.reshape(1, J), omega_SS.reshape(S, 1), 'SS', etr_params, theta, tau_bq,
                          tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J, tau_b, delta_tau)
        new_T_H = tax.revenue(new_r, new_w, b_s, nssmat, new_BQ, new_Y, L, K, factor, T_H_params)
    elif baseline_spending:
        new_T_H = T_H
    else:
        new_T_H = alpha_T*new_Y

    return euler_errors, bssmat, nssmat, new_r, new_w, \
         new_T_H, new_Y, new_factor, new_BQ, average_income_model


def SS_solver(b_guess_init, n_guess_init, rss, wss, T_Hss, factor_ss, Yss, params, baseline, fsolve_flag=False, baseline_spending=False):
    '''
    --------------------------------------------------------------------
    Solves for the steady state distribution of capital, labor, as well as
    w, r, T_H and the scaling factor, using a bisection method similar to TPI.
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
    iterative_params = length X tuple, list of parameters that determine the convergence
                       of the while loop
    tau_bq = [J,] vector, bequest tax rate
    rho = [S,] vector, mortality rates by age
    lambdas = [J,] vector, fraction of population with each ability type
    omega = [S,] vector, stationary population weights
    e =  [S,J] array, effective labor units by age and ability type


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    euler_equation_solver()
    household.get_K()
    firm.get_L()
    firm.get_Y()
    firm.get_r()
    firm.get_w()
    household.get_BQ()
    tax.replacement_rate_vals()
    tax.revenue()
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

    bssmat, nssmat, chi_params, ss_params, income_tax_params, iterative_params, small_open_params = params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, budget_balance, \
                  alpha_T, debt_ratio_ss, tau_b, delta_tau,\
                  lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    small_open, ss_firm_r, ss_hh_r = small_open_params

    # Rename the inputs
    r = rss
    w = wss
    T_H = T_Hss
    factor = factor_ss
    if budget_balance == False:
        if baseline_spending == True:
            Y = Yss
        else:
            Y = T_H / alpha_T
    if small_open == True:
        r = ss_hh_r

    dist = 10
    iteration = 0
    dist_vec = np.zeros(maxiter)

    if fsolve_flag == True:
        maxiter = 1

    while (dist > mindist_SS) and (iteration < maxiter):
        # Solve for the steady state levels of b and n, given w, r, Y and
        # factor
        if budget_balance:
            outer_loop_vars = (bssmat, nssmat, r, w, T_H, factor)
        else:
            outer_loop_vars = (bssmat, nssmat, r, w, Y, T_H, factor)
        inner_loop_params = (ss_params, income_tax_params, chi_params, small_open_params)

        euler_errors, bssmat, nssmat, new_r, new_w, \
             new_T_H, new_Y, new_factor, new_BQ, average_income_model = inner_loop(outer_loop_vars, inner_loop_params, baseline, baseline_spending)

        r = utils.convex_combo(new_r, r, nu)
        w = utils.convex_combo(new_w, w, nu)
        factor = utils.convex_combo(new_factor, factor, nu)
        if budget_balance:
            T_H = utils.convex_combo(new_T_H, T_H, nu)
            dist = np.array([utils.pct_diff_func(new_r, r)] +
                            [utils.pct_diff_func(new_w, w)] +
                            [utils.pct_diff_func(new_T_H, T_H)] +
                            [utils.pct_diff_func(new_factor, factor)]).max()
        else:
            Y = utils.convex_combo(new_Y, Y, nu)
            if Y != 0:
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                [utils.pct_diff_func(new_w, w)] +
                                [utils.pct_diff_func(new_Y, Y)] +
                                [utils.pct_diff_func(new_factor, factor)]).max()
            else:
                # If Y is zero (if there is no output), a percent difference
                # will throw NaN's, so we use an absoluate difference
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                [utils.pct_diff_func(new_w, w)] +
                                [abs(new_Y - Y)] +
                                [utils.pct_diff_func(new_factor, factor)]).max()
        dist_vec[iteration] = dist
        # Similar to TPI: if the distance between iterations increases, then
        # decrease the value of nu to prevent cycling
        if iteration > 10:
            if dist_vec[iteration] - dist_vec[iteration - 1] > 0:
                nu /= 2.0
                print 'New value of nu:', nu
        iteration += 1
        print "Iteration: %02d" % iteration, " Distance: ", dist

    '''
    ------------------------------------------------------------------------
        Generate the SS values of variables, including euler errors
    ------------------------------------------------------------------------
    '''
    bssmat_s = np.append(np.zeros((1,J)),bssmat[:-1,:],axis=0)
    bssmat_splus1 = bssmat

    rss = r
    wss = w
    factor_ss = factor
    T_Hss = T_H

    Lss_params = (e, omega_SS.reshape(S, 1), lambdas, 'SS')
    Lss = firm.get_L(nssmat, Lss_params)
    if small_open == False:
        Kss_params = (omega_SS.reshape(S, 1), lambdas, imm_rates, g_n_ss, 'SS')
        Bss = household.get_K(bssmat_splus1, Kss_params)
        if budget_balance:
            debt_ss = 0.0
        else:
            debt_ss = debt_ratio_ss*Y
        Kss = Bss - debt_ss
        Iss_params = (delta, g_y, omega_SS, lambdas, imm_rates, g_n_ss, 'SS')
        Iss = firm.get_I(bssmat_splus1, Kss, Kss, Iss_params)
    else:
        # Compute capital (K) and wealth (B) separately
        Kss_params = (Z, gamma, epsilon, delta, tau_b, delta_tau)
        Kss = firm.get_K(Lss, ss_firm_r, Kss_params)
        Iss_params = (delta, g_y, omega_SS, lambdas, imm_rates, g_n_ss, 'SS')
        InvestmentPlaceholder = np.zeros(bssmat_splus1.shape)
        Iss = firm.get_I(InvestmentPlaceholder, Kss, Kss, Iss_params)
        Bss_params = (omega_SS.reshape(S, 1), lambdas, imm_rates, g_n_ss, 'SS')
        Bss = household.get_K(bssmat_splus1, Bss_params)
        BIss_params = (0.0, g_y, omega_SS, lambdas, imm_rates, g_n_ss, 'SS')
        BIss = firm.get_I(bssmat_splus1, Bss, Bss, BIss_params)
        if budget_balance:
            debt_ss = 0.0
        else:
            debt_ss = debt_ratio_ss*Y


    # Yss_params = (alpha, Z)
    Yss_params = (Z, gamma, epsilon)
    Yss = firm.get_Y(Kss, Lss, Yss_params)

    # Verify that T_Hss = alpha_T*Yss
#    transfer_error = T_Hss - alpha_T*Yss
#    if np.absolute(transfer_error) > mindist_SS:
#        print 'Transfers exceed alpha_T percent of GDP by:', transfer_error
#        err = "Transfers do not match correct share of GDP in SS_solver"
#        raise RuntimeError(err)

    BQss = new_BQ
    theta_params = (e, S, retire)
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, theta_params)

    # Next 5 lines pulled out of inner_loop where they are used to calculate tax revenue. Now calculating G to balance gov't budget.
    b_s = np.array(list(np.zeros(J).reshape(1, J)) + list(bssmat[:-1, :]))
    lump_sum_params = (e, lambdas.reshape(1, J), omega_SS.reshape(S, 1), 'SS', etr_params, theta, tau_bq,
                      tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J, tau_b, delta_tau)
    revenue_ss = tax.revenue(new_r, new_w, b_s, nssmat, new_BQ, Yss, Lss, Kss, factor, lump_sum_params)
    r_gov_ss = rss
    debt_service_ss = r_gov_ss*debt_ratio_ss*Yss
    new_borrowing = debt_ratio_ss*Yss*((1+g_n_ss)*np.exp(g_y)-1)
    # government spends such that it expands its debt at the same rate as GDP
    if budget_balance:
        Gss = 0.0
    else:
        Gss = revenue_ss + new_borrowing - (T_Hss + debt_service_ss)

    # solve resource constraint
    etr_params_3D = np.tile(np.reshape(etr_params,(S,1,etr_params.shape[1])),(1,J,1))
    mtrx_params_3D = np.tile(np.reshape(mtrx_params,(S,1,mtrx_params.shape[1])),(1,J,1))

    '''
    ------------------------------------------------------------------------
        The code below is to calulate and save model MTRs
                - only exists to help debug
    ------------------------------------------------------------------------
    '''
    # etr_params_extended = np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:]
    # etr_params_extended_3D = np.tile(np.reshape(etr_params_extended,(S,1,etr_params_extended.shape[1])),(1,J,1))
    # mtry_params_extended = np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:]
    # mtry_params_extended_3D = np.tile(np.reshape(mtry_params_extended,(S,1,mtry_params_extended.shape[1])),(1,J,1))
    # e_extended = np.array(list(e) + list(np.zeros(J).reshape(1, J)))
    # nss_extended = np.array(list(nssmat) + list(np.zeros(J).reshape(1, J)))
    # mtry_ss_params = (e_extended[1:,:], etr_params_extended_3D, mtry_params_extended_3D, analytical_mtrs)
    # mtry_ss = tax.MTR_capital(rss, wss, bssmat_splus1, nss_extended[1:,:], factor_ss, mtry_ss_params)
    # mtrx_ss_params = (e, etr_params_3D, mtrx_params_3D, analytical_mtrs)
    # mtrx_ss = tax.MTR_labor(rss, wss, bssmat_s, nssmat, factor_ss, mtrx_ss_params)

    # np.savetxt("mtr_ss_capital.csv", mtry_ss, delimiter=",")
    # np.savetxt("mtr_ss_labor.csv", mtrx_ss, delimiter=",")

    # solve resource constraint
    taxss_params = (e, lambdas, 'SS', retire, etr_params_3D,
                    h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    taxss = tax.total_taxes(rss, wss, bssmat_s, nssmat, BQss, factor_ss, T_Hss, None, False, taxss_params)
    css_params = (e, lambdas.reshape(1, J), g_y)
    cssmat = household.get_cons(rss, wss, bssmat_s, bssmat_splus1, nssmat, BQss.reshape(
        1, J), taxss, css_params)

    biz_params = (tau_b, delta_tau)
    business_revenue = tax.get_biz_tax(wss, Yss, Lss, Kss, biz_params)

    Css_params = (omega_SS.reshape(S, 1), lambdas, 'SS')
    Css = household.get_C(cssmat, Css_params)

    if small_open == False:
        resource_constraint = Yss - (Css + Iss + Gss)
        print 'Yss= ', Yss, '\n', 'Gss= ', Gss, '\n', 'Css= ', Css, '\n', 'Kss = ', Kss, '\n', 'Iss = ', Iss, '\n', 'Lss = ', Lss, '\n', 'Debt service = ', debt_service_ss
        print 'D/Y:', debt_ss/Yss, 'T/Y:', T_Hss/Yss, 'G/Y:', Gss/Yss, 'Rev/Y:', revenue_ss/Yss, 'business rev/Y: ', business_revenue/Yss, 'Int payments to GDP:', (rss*debt_ss)/Yss
        print 'Check SS budget: ', Gss - (np.exp(g_y)*(1+g_n_ss)-1-rss)*debt_ss - revenue_ss + T_Hss
        print 'resource constraint: ', resource_constraint
    else:
        # include term for current account
        resource_constraint = Yss + new_borrowing  - (Css + BIss + Gss) + (ss_hh_r * Bss - (delta + ss_firm_r) * Kss - debt_service_ss)
        print 'Yss= ', Yss, '\n', 'Css= ', Css, '\n', 'Bss = ', Bss, '\n', 'BIss = ', BIss, '\n', 'Kss = ', Kss, '\n', 'Iss = ', Iss, '\n', 'Lss = ', Lss, '\n', 'T_H = ', T_H,'\n', 'Gss= ', Gss
        print 'D/Y:', debt_ss/Yss, 'T/Y:', T_Hss/Yss, 'G/Y:', Gss/Yss, 'Rev/Y:', revenue_ss/Yss, 'Int payments to GDP:', (rss*debt_ss)/Yss
        print 'resource constraint: ', resource_constraint

    if Gss < 0:
        print 'Steady state government spending is negative to satisfy budget'

    if ENFORCE_SOLUTION_CHECKS and np.absolute(resource_constraint) > mindist_SS:
        print 'Resource Constraint Difference:', resource_constraint
        err = "Steady state aggregate resource constraint not satisfied"
        raise RuntimeError(err)

    # check constraints
    household.constraint_checker_SS(bssmat, nssmat, cssmat, ltilde)

    euler_savings = euler_errors[:S,:]
    euler_labor_leisure = euler_errors[S:,:]

    '''
    ------------------------------------------------------------------------
        Return dictionary of SS results
    ------------------------------------------------------------------------
    '''
    output = {'Kss': Kss, 'bssmat': bssmat, 'Bss': Bss, 'Lss': Lss, 'Css':Css, 'Iss':Iss, 'nssmat': nssmat, 'Yss': Yss,
              'wss': wss, 'rss': rss, 'theta': theta, 'BQss': BQss, 'factor_ss': factor_ss,
              'bssmat_s': bssmat_s, 'cssmat': cssmat, 'bssmat_splus1': bssmat_splus1,
              'T_Hss': T_Hss, 'Gss': Gss, 'revenue_ss': revenue_ss, 'euler_savings': euler_savings,
              'euler_labor_leisure': euler_labor_leisure, 'chi_n': chi_n,
              'chi_b': chi_b}

    return output



def SS_fsolve(guesses, params):
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

    bssmat, nssmat, chi_params, ss_params, income_tax_params, iterative_params, small_open_params = params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, budget_balance,\
                  alpha_T, debt_ratio_ss, tau_b, delta_tau,\
                  lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    baseline = True

    # Rename the inputs
    r = guesses[0]
    w = guesses[1]
    T_H = guesses[2]
    factor = guesses[3]

    print 'r, w at outset: ', r, w

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    if budget_balance:
        outer_loop_vars = (bssmat, nssmat, r, w, T_H, factor)
    else:
        Y = T_H / alpha_T
        outer_loop_vars = (bssmat, nssmat, r, w, Y, T_H, factor)
    inner_loop_params = (ss_params, income_tax_params, chi_params, small_open_params)
    euler_errors, bssmat, nssmat, new_r, new_w, \
         new_T_H, new_Y, new_factor, new_BQ, average_income_model = inner_loop(outer_loop_vars, inner_loop_params, baseline)

    error1 = new_r - r
    error2 = new_w - w
    if budget_balance:
        error3 = new_T_H - T_H
    else:
        error3 = new_Y - Y
    error4 = new_factor/1000000 - factor/1000000

  #  print 'mean income in model and data: ', average_income_model, mean_income_data
  #  print 'model income with factor: ', average_income_model*factor

  #  print 'errors: ', error1, error2, error3, error4

    print 'Y: ', new_Y
  #  print 'factor: ', new_factor
  #  print 'factor prices: ', new_r, new_w


    # Check and punish violations
    if r+delta <= 0:
        error1 = 1e9
    #if r > 1:
    #    error1 += 1e9
    if w <= 0:
        error2 = 1e9
    if factor <= 0:
        error4 = 1e9

    print 'errors: ', error1, error2, error3, error4

    return [error1, error2, error3, error4]




def SS_fsolve_reform(guesses, params):
    '''
    Solves for the steady state distribution of capital, labor, as well as
    w, r, and T_H and the scaling factor, using a root finder. This solves for the
    reform SS and so takes the factor from the baseline SS as an input.
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
    bssmat, nssmat, chi_params, ss_params, income_tax_params, iterative_params, factor, small_open_params = params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, budget_balance,\
                  alpha_T, debt_ratio_ss, tau_b, delta_tau,\
                  lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params


    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    baseline = False
    # Rename the inputs
    r = guesses[0]
    w = guesses[1]
    T_H = guesses[2]

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    if budget_balance:
        outer_loop_vars = (bssmat, nssmat, r, w, T_H, factor)
    else:
        Y = T_H / alpha_T
        outer_loop_vars = (bssmat, nssmat, r, w, Y, T_H, factor)
    inner_loop_params = (ss_params, income_tax_params, chi_params, small_open_params)

    euler_errors, bssmat, nssmat, new_r, new_w, \
        new_T_H, new_Y, new_factor, new_BQ, average_income_model = inner_loop(outer_loop_vars, inner_loop_params, baseline, False)

    error1 = new_r - r
    error2 = new_w - w
    if budget_balance:
        error3 = new_T_H - T_H
    else:
        error3 = new_Y - Y


    print 'errors: ', error1, error2, error3
   # print 'factor prices: ', r, w

    # Check and punish violations
    if r+delta <= 0:
        error1 = 1e9
    #if r > 1:
    #    error1 += 1e9
    if w <= 0:
        error2 = 1e9

    return [error1, error2, error3]

def SS_fsolve_reform_baselinespend(guesses, params):
    '''
    Solves for the steady state distribution of capital, labor, as well as
    w, r, and Y, using a root finder. This solves for the
    reform SS when baseline_speding=True and so takes the factor and gov't
    transfers (T_H) from the baseline SS as an input.
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
    bssmat, nssmat, T_Hss, chi_params, ss_params, income_tax_params, iterative_params, factor, small_open_params = params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, budget_balance,\
                  alpha_T, debt_ratio_ss, tau_b, delta_tau,\
                  lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params


    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    baseline = False
    # Rename the inputs
    r = guesses[0]
    w = guesses[1]
    Y = guesses[2]

    # Solve for the steady state levels of b and n, given w, r, T_H and
    # factor
    T_H = T_Hss
    outer_loop_vars = (bssmat, nssmat, r, w, Y, T_H, factor)
    inner_loop_params = (ss_params, income_tax_params, chi_params, small_open_params)

    euler_errors, bssmat, nssmat, new_r, new_w, \
        new_T_H, new_Y, new_factor, new_BQ, average_income_model = inner_loop(outer_loop_vars, inner_loop_params, baseline, True)

    error1 = new_r - r
    error2 = new_w - w
    error3 = new_Y - Y

    print 'errors: ', error1, error2, error3
   # print 'factor prices: ', r, w

    # Check and punish violations
    if r+delta <= 0:
        error1 = 1e9
    #if r > 1:
    #    error1 += 1e9
    if w <= 0:
        error2 = 1e9

    return [error1, error2, error3]



def run_SS(income_tax_params, ss_params, iterative_params, chi_params, small_open_params, baseline=True, baseline_spending=False, baseline_dir="./OUTPUT"):
    '''
    --------------------------------------------------------------------
    Solve for SS of OG-USA.
    --------------------------------------------------------------------

    INPUTS:
    income_tax_parameters = length 4 tuple, (analytical_mtrs, etr_params, mtrx_params, mtry_params)
    ss_parameters = length 21 tuple, (J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon)
    iterative_params  = [2,] vector, vector with max iterations and tolerance
                        for SS solution
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
    factorguess = scalar, initial guess at SS factor adjustment (to scale model units to dollars)

    output


    RETURNS: output

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, budget_balance,\
                  alpha_T, debt_ratio_ss, tau_b, delta_tau,\
                  lambdas, imm_rates, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params


    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params

    chi_b, chi_n = chi_params

    maxiter, mindist_SS = iterative_params

    b_guess = np.ones((S, J)).flatten() * 0.05
    n_guess = np.ones((S, J)).flatten() * .4 * ltilde
    # For initial guesses of w, r, T_H, and factor, we use values that are close
    # to some steady state values.
    if baseline:
        rguess = 0.04#0.01 + delta
        wguess = 1.2
        T_Hguess = 0.12
        factorguess = 70000

        ss_params_baseline = [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params, small_open_params]
        guesses = [rguess, wguess, T_Hguess, factorguess]
        [solutions_fsolve, infodict, ier, message] = opt.fsolve(SS_fsolve, guesses, args=ss_params_baseline, xtol=mindist_SS, full_output=True)
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            raise RuntimeError("Steady state equilibrium not found")
        [rss, wss, T_Hss, factor_ss] = solutions_fsolve
        Yss = T_Hss/alpha_T #may not be right - if budget_balance = True, but that's ok - will be fixed in SS_solver
        fsolve_flag = True
        # Return SS values of variables
        solution_params= [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params, small_open_params]
        output = SS_solver(b_guess.reshape(S, J), n_guess.reshape(S, J), rss, wss, T_Hss, factor_ss, Yss, solution_params, baseline, fsolve_flag, baseline_spending)
        # print "solved output", wss, rss, T_Hss, factor_ss
     #   print 'analytical mtrs in SS: ', analytical_mtrs
    else:
        baseline_ss_dir = os.path.join(
            baseline_dir, "SS/SS_vars.pkl")
        ss_solutions = pickle.load(open(baseline_ss_dir, "rb"))
        [rguess, wguess, T_Hguess, Yguess, factor] = [ss_solutions['rss'], ss_solutions['wss'], ss_solutions['T_Hss'], ss_solutions['Yss'], ss_solutions['factor_ss']]
        if baseline_spending:
            T_Hss = T_Hguess
            ss_params_reform = [b_guess.reshape(S, J), n_guess.reshape(S, J), T_Hss, chi_params, ss_params, income_tax_params, iterative_params, factor, small_open_params]
            guesses = [rguess, wguess, Yguess]
            [solutions_fsolve, infodict, ier, message] = opt.fsolve(SS_fsolve_reform_baselinespend, guesses, args=ss_params_reform, xtol=mindist_SS, full_output=True)
            [rss, wss, Yss] = solutions_fsolve
        else:
            ss_params_reform = [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params, factor, small_open_params]
            guesses = [rguess, wguess, T_Hguess]
            [solutions_fsolve, infodict, ier, message] = opt.fsolve(SS_fsolve_reform, guesses, args=ss_params_reform, xtol=mindist_SS, full_output=True)
            [rss, wss, T_Hss] = solutions_fsolve
            Yss = T_Hss/alpha_T #may not be right - if budget_balance = True, but that's ok - will be fixed in SS_solver
        if ENFORCE_SOLUTION_CHECKS and not ier == 1:
            raise RuntimeError("Steady state equilibrium not found")
        # Return SS values of variables
        fsolve_flag = True
        # Return SS values of variables
        solution_params= [b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params, ss_params, income_tax_params, iterative_params, small_open_params]
        output = SS_solver(b_guess.reshape(S, J), n_guess.reshape(S, J), rss, wss, T_Hss, factor, Yss, solution_params, baseline, fsolve_flag, baseline_spending)
    return output
