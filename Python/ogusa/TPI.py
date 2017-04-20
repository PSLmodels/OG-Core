'''
------------------------------------------------------------------------
Last updated 4/9/2016

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method, where labor in inelastically supplied.

This py-file calls the following other file(s):
            tax.py
            utils.py
            household.py
            firm.py
            OUTPUT/SS/ss_vars.pkl
            OUTPUT/Saved_moments/params_given.pkl
            OUTPUT/Saved_moments/params_changed.pkl


This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/TPIinit/TPIinit_vars.pkl
            OUTPUT/TPI/TPI_vars.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.optimize as opt

import tax
import utils
import household
import firm
import fiscal
import os
import csv


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
Import steady state distribution, parameters and other objects from
steady state computation in ss_vars.pkl
------------------------------------------------------------------------
'''

def create_tpi_params(**sim_params):

    '''
    ------------------------------------------------------------------------
    Set factor and initial capital stock to SS from baseline
    ------------------------------------------------------------------------
    '''
    baseline_ss = os.path.join(sim_params['baseline_dir'], "SS/SS_vars.pkl")
    ss_baseline_vars = pickle.load(open(baseline_ss, "rb"))
    factor = ss_baseline_vars['factor_ss']
    initial_b = ss_baseline_vars['bssmat_splus1']
    initial_n = ss_baseline_vars['nssmat']
    if sim_params['baseline_spending']==True:
        baseline_tpi = os.path.join(sim_params['baseline_dir'], "TPI/TPI_vars.pkl")
        tpi_baseline_vars = pickle.load(open(baseline_tpi, "rb"))
        T_Hbaseline = tpi_baseline_vars['T_H']
        Gbaseline   = tpi_baseline_vars['G']

    theta_params = (sim_params['e'], sim_params['S'], sim_params['retire'])
    if sim_params['baseline']==True:
        SS_values = (ss_baseline_vars['Kss'], ss_baseline_vars['Bss'], ss_baseline_vars['Lss'], ss_baseline_vars['rss'],
                 ss_baseline_vars['wss'], ss_baseline_vars['BQss'], ss_baseline_vars['T_Hss'], ss_baseline_vars['revenue_ss'],
                 ss_baseline_vars['bssmat_splus1'], ss_baseline_vars['nssmat'], ss_baseline_vars['Yss'], ss_baseline_vars['Gss'])
        theta = tax.replacement_rate_vals(ss_baseline_vars['nssmat'], ss_baseline_vars['wss'], factor, theta_params)
    elif sim_params['baseline']==False:
        reform_ss = os.path.join(sim_params['input_dir'], "SS/SS_vars.pkl")
        ss_reform_vars = pickle.load(open(reform_ss, "rb"))
        SS_values = (ss_reform_vars['Kss'],ss_reform_vars['Bss'], ss_reform_vars['Lss'], ss_reform_vars['rss'],
                 ss_reform_vars['wss'], ss_reform_vars['BQss'], ss_reform_vars['T_Hss'], ss_reform_vars['revenue_ss'],
                 ss_reform_vars['bssmat_splus1'], ss_reform_vars['nssmat'], ss_reform_vars['Yss'], ss_reform_vars['Gss'])
        theta = tax.replacement_rate_vals(ss_reform_vars['nssmat'], ss_reform_vars['wss'], factor, theta_params)

    # Make a vector of all one dimensional parameters, to be used in the
    # following functions
    wealth_tax_params = [sim_params['h_wealth'], sim_params['p_wealth'], sim_params['m_wealth']]
    ellipse_params = [sim_params['b_ellipse'], sim_params['upsilon']]
    chi_params = [sim_params['chi_b_guess'], sim_params['chi_n_guess']]

    N_tilde = sim_params['omega'].sum(1) #this should just be one in each year given how we've constructed omega
    sim_params['omega'] = sim_params['omega'] / N_tilde.reshape(sim_params['T'] + sim_params['S'], 1)



    tpi_params = [sim_params['J'], sim_params['S'], sim_params['T'], sim_params['BW'],
                  sim_params['beta'], sim_params['sigma'], sim_params['alpha'],
                  sim_params['gamma'], sim_params['epsilon'],
                  sim_params['Z'], sim_params['delta'], sim_params['ltilde'],
                  sim_params['nu'], sim_params['g_y'], sim_params['g_n_vector'],
                  sim_params['tau_payroll'], sim_params['tau_bq'], sim_params['rho'], sim_params['omega'], N_tilde,
                  sim_params['lambdas'], sim_params['imm_rates'], sim_params['e'], sim_params['retire'], sim_params['mean_income_data'], factor] + \
                  wealth_tax_params + ellipse_params + chi_params + [theta]
    iterative_params = [sim_params['maxiter'], sim_params['mindist_SS'], sim_params['mindist_TPI']]
    small_open_params = [sim_params['small_open'], sim_params['tpi_firm_r'], sim_params['tpi_hh_r']]


    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params

    ## Assumption for tax functions is that policy in last year of BW is
    # extended permanently
    etr_params_TP = np.zeros((S,T+S,sim_params['etr_params'].shape[2]))
    etr_params_TP[:,:BW,:] = sim_params['etr_params']
    etr_params_TP[:,BW:,:] = np.reshape(sim_params['etr_params'][:,BW-1,:],(S,1,sim_params['etr_params'].shape[2]))

    mtrx_params_TP = np.zeros((S,T+S,sim_params['mtrx_params'].shape[2]))
    mtrx_params_TP[:,:BW,:] = sim_params['mtrx_params']
    mtrx_params_TP[:,BW:,:] = np.reshape(sim_params['mtrx_params'][:,BW-1,:],(S,1,sim_params['mtrx_params'].shape[2]))

    mtry_params_TP = np.zeros((S,T+S,sim_params['mtry_params'].shape[2]))
    mtry_params_TP[:,:BW,:] = sim_params['mtry_params']
    mtry_params_TP[:,BW:,:] = np.reshape(sim_params['mtry_params'][:,BW-1,:],(S,1,sim_params['mtry_params'].shape[2]))

    income_tax_params = (sim_params['analytical_mtrs'], etr_params_TP, mtrx_params_TP, mtry_params_TP)

    '''
    ------------------------------------------------------------------------
    Set government finance parameters
    ------------------------------------------------------------------------
    '''
    budget_balance = sim_params['budget_balance']
    ALPHA_T        = sim_params['ALPHA_T']
    ALPHA_G        = sim_params['ALPHA_G']
    tG1            = sim_params['tG1']
    tG2            = sim_params['tG2']
    rho_G          = sim_params['rho_G']
    debt_ratio_ss  = sim_params['debt_ratio_ss']
    if sim_params['baseline_spending']==False:
        fiscal_params  = (budget_balance, ALPHA_T, ALPHA_G, tG1, tG2, rho_G, debt_ratio_ss)
    else:
        fiscal_params  = (budget_balance, ALPHA_T, ALPHA_G, tG1, tG2, rho_G, debt_ratio_ss, T_Hbaseline, Gbaseline)

    initial_debt  = sim_params['initial_debt']

    '''
    ------------------------------------------------------------------------
    Set business tax parameters
    ------------------------------------------------------------------------
    '''
    tau_b = sim_params['tau_b']
    delta_tau = sim_params['delta_tau']
    biz_tax_params  = (tau_b, delta_tau)

    initial_debt  = sim_params['initial_debt']

    '''
    ------------------------------------------------------------------------
    Set other parameters and initial values
    ------------------------------------------------------------------------
    '''
    # Get an initial distribution of wealth with the initial population
    # distribution. When small_open=True, the value of K0 is used as a placeholder for first-period wealth (B0)
    omega_S_preTP = sim_params['omega_S_preTP']
    B0_params = (omega_S_preTP.reshape(S, 1), lambdas, imm_rates[0].reshape(S,1), g_n_vector[0], 'SS')
    B0 = household.get_K(initial_b, B0_params)

    b_sinit = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_b[:-1]))
    b_splus1init = initial_b

    initial_values = (B0, b_sinit, b_splus1init, factor, initial_b, initial_n, omega_S_preTP, initial_debt)

    return (income_tax_params, tpi_params, iterative_params, small_open_params, initial_values, SS_values, fiscal_params, biz_tax_params)


def firstdoughnutring(guesses, r, w, b, BQ, T_H, j, params):
    '''
    Solves the first entries of the upper triangle of the twist doughnut.  This is
    separate from the main TPI function because the the values of b and n are scalars,
    so it is easier to just have a separate function for these cases.
    Inputs:
        guesses = guess for b and n (2x1 list)
        winit = initial wage rate (scalar)
        rinit = initial rental rate (scalar)
        BQinit = initial aggregate bequest (scalar)
        T_H_init = initial lump sum tax (scalar)
        initial_b = initial distribution of capital (SxJ array)
        factor = steady state scaling factor (scalar)
        j = which ability type is being solved for (scalar)
        parameters = list of parameters (list)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rates (Jx1 array)
    Output:
        euler errors (2x1 list)
    '''

    # unpack tuples of parameters
    income_tax_params, tpi_params, initial_b = params
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params


    b_splus1 = float(guesses[0])
    n = float(guesses[1])
    b_s = float(initial_b[-2, j])

    # Euler 1 equations
    tax1_params = (e[-1, j], lambdas[j], 'TPI_scalar', retire, etr_params[-1,0,:], h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax1 = tax.total_taxes(r, w, b_s, n, BQ, factor, T_H, j, False, tax1_params)

    cons_params = (e[-1, j], lambdas[j], g_y)
    cons = household.get_cons(r, w, b_s, b_splus1, n, BQ, tax1, cons_params)

    bequest_ut = rho[-1] * np.exp(-sigma * g_y) * chi_b[j] * b_splus1 ** (-sigma)

    error1 = household.marg_ut_cons(cons, sigma) - bequest_ut

    # Euler 2 equations
    income2 = (r * b_s + w * e[-1, j] * n) * factor

    mtr_labor_params = (e[-1, j], etr_params[-1,0,:], mtrx_params[-1,0,:], analytical_mtrs)
    deriv2 = 1 - tau_payroll - tax.MTR_labor(r, w, b_s, n, factor, mtr_labor_params)

    mu_labor_params = (b_ellipse, upsilon, ltilde, chi_n[-1])
    error2 = household.marg_ut_cons(cons, sigma) * w * \
        e[-1, j] * deriv2 - household.marg_ut_labor(n, mu_labor_params)

    #### TEST THESE FUNCS BELOW TO BE SURE GET SAME OUTPUT, but should use if so ***
    # foc_save_params = (e[-1, j], sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas, J, S,
    #     analytical_mtrs, etr_params[-1,0,:], mtry_params[-1,0,:], h_wealth, p_wealth, m_wealth, tau_payroll, retire, 'TPI')
    # error3 = household.FOC_savings(r, w, b_s, b_splus1, 0., n, BQ, factor, T_H, foc_save_params)

    # foc_labor_params = (e[-1, j], sigma, g_y, theta, b_ellipse, upsilon, chi_n, ltilde, tau_bq, lambdas, J, S,
    #     analytical_mtrs, etr_params[-1,0,:], mtrx_params[-1,0,:], h_wealth, p_wealth, m_wealth, tau_payroll, retire, 'TPI')
    # error4 = household.FOC_labor(r, w, b, b_splus1, n, BQ, factor, T_H, foc_labor_params)
    # print 'check1:', error2-error4
    # print 'check2:', error1-error3

    if n <= 0 or n >= 1:
        error2 += 1e12
    if b_splus1 <= 0:
        error1 += 1e12
    if cons <= 0:
        error1 += 1e12
    return [error1] + [error2]


def twist_doughnut(guesses, r, w, BQ, T_H, j, s, t, params):
    '''
    Parameters:
        guesses = distribution of capital and labor (various length list)
        w   = wage rate ((T+S)x1 array)
        r   = rental rate ((T+S)x1 array)
        BQ = aggregate bequests ((T+S)x1 array)
        T_H = lump sum tax over time ((T+S)x1 array)
        factor = scaling factor (scalar)
        j = which ability type is being solved for (scalar)
        s = which upper triangle loop is being solved for (scalar)
        t = which diagonal is being solved for (scalar)
        params = list of parameters (list)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortalit rate (Sx1 array)
        lambdas = ability weights (Jx1 array)
        e = ability type (SxJ array)
        initial_b = capital stock distribution in period 0 (SxJ array)
        chi_b = chi^b_j (Jx1 array)
        chi_n = chi^n_s (Sx1 array)
    Output:
        Value of Euler error (various length list)
    '''

    income_tax_params, tpi_params, initial_b = params
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params

    length = len(guesses) / 2
    b_guess = np.array(guesses[:length])
    n_guess = np.array(guesses[length:])

    if length == S:
        b_s = np.array([0] + list(b_guess[:-1]))
    else:
        b_s = np.array([(initial_b[-(s + 3), j])] + list(b_guess[:-1]))

    b_splus1 = b_guess
    b_splus2 = np.array(list(b_guess[1:]) + [0])
    w_s = w[t:t + length]
    w_splus1 = w[t + 1:t + length + 1]
    r_s = r[t:t + length]
    r_splus1 = r[t + 1:t + length + 1]
    n_s = n_guess
    n_extended = np.array(list(n_guess[1:]) + [0])
    e_s = e[-length:, j]
    e_extended = np.array(list(e[-length + 1:, j]) + [0])
    BQ_s = BQ[t:t + length]
    BQ_splus1 = BQ[t + 1:t + length + 1]
    T_H_s = T_H[t:t + length]
    T_H_splus1 = T_H[t + 1:t + length + 1]


    # Savings euler equations
    tax_s_params = (e_s, lambdas[j], 'TPI', retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)

    tax_s = tax.total_taxes(r_s, w_s, b_s, n_s, BQ_s, factor, T_H_s, j, False, tax_s_params)

    etr_params_sp1 = np.append(etr_params,np.reshape(etr_params[-1,:],(1,etr_params.shape[1])),axis=0)[1:,:]
    taxsp1_params = (e_extended, lambdas[j], 'TPI', retire, etr_params_sp1, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax_splus1 = tax.total_taxes(r_splus1, w_splus1, b_splus1, n_extended, BQ_splus1, factor, T_H_splus1, j, True, taxsp1_params)


    cons_s_params = (e_s, lambdas[j], g_y)
    cons_s = household.get_cons(r_s, w_s, b_s, b_splus1, n_s,
                   BQ_s, tax_s, cons_s_params)

    cons_sp1_params = (e_extended, lambdas[j], g_y)
    cons_splus1 = household.get_cons(r_splus1, w_splus1, b_splus1, b_splus2, n_extended,
                   BQ_splus1, tax_splus1, cons_sp1_params)

    income_splus1 = (r_splus1 * b_splus1 + w_splus1 *
                     e_extended * n_extended) * factor
    savings_ut = rho[-(length):] * np.exp(-sigma * g_y) * \
        chi_b[j] * b_splus1 ** (-sigma)

    mtry_params_sp1 = np.append(mtry_params,np.reshape(mtry_params[-1,:],(1,mtry_params.shape[1])),axis=0)[1:,:]
    mtr_capital_params = (e_extended, etr_params_sp1, mtry_params_sp1, analytical_mtrs)
    deriv_savings = 1 + r_splus1 * (1 - tax.MTR_capital(r_splus1, w_splus1, b_splus1, n_extended, factor, mtr_capital_params))

    #Note equation below accounts for savings in last period because here rho=1 - so second term drops out.  Which means tax rates after last
    # period of life don't matter
    error1= household.marg_ut_cons(cons_s, sigma) - beta * (1 - rho[-(length):]) * np.exp(-sigma * g_y) * deriv_savings * household.marg_ut_cons(
        cons_splus1, sigma) - savings_ut


    # Labor leisure euler equations
    income_s = (r_s * b_s + w_s * e_s * n_s) * factor


    mtr_labor_params = (e_s, etr_params, mtrx_params, analytical_mtrs)
    deriv_laborleisure = 1 - tau_payroll - tax.MTR_labor(r_s, w_s, b_s, n_s, factor, mtr_labor_params)

    mu_labor_params = (b_ellipse, upsilon, ltilde, chi_n[-length:])
    error2 = household.marg_ut_cons(cons_s, sigma) * w_s * e[-(
        length):, j] * deriv_laborleisure - household.marg_ut_labor(n_s, mu_labor_params)
    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] += 1e12
    mask2 = n_guess > ltilde
    error2[mask2] += 1e12
    mask3 = cons_s < 0
    error2[mask3] += 1e12
    mask4 = b_guess <= 0
    error2[mask4] += 1e12
    mask5 = cons_splus1 < 0
    mask5[-1] = b_splus1[-1] < 0
    error2[mask5] += 1e12
    return list(error1.flatten()) + list(error2.flatten())


def inner_loop(guesses, outer_loop_vars, params):
    '''
    Solves inner loop of TPI.  Given path of economic aggregates and factor prices, solves
    household problem

    Inputs:
        r          = [T,] vector, interest rate
        w          = [T,] vector, wage rate
        b          = [T,S,J] array, wealth holdings
        n          = [T,S,J] array, labor supply
        BQ         = [T,J] vector,  bequest amounts
        factor     = scalar, model income scaling factor
        T_H        = [T,] vector, lump sum transfer amount(s)


    Functions called:
        firstdoughnutring()
        twist_doughnut()

    Objects in function:


    Returns: euler_errors, b_mat, n_mat

    '''
    #unpack variables and parameters pass to function
    income_tax_params, tpi_params, initial_values, ind = params
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params
    K0, b_sinit, b_splus1init, factor, initial_b, initial_n, omega_S_preTP, initial_debt = initial_values

    guesses_b, guesses_n = guesses
    r, w, K, BQ, T_H = outer_loop_vars

    # initialize arrays
    b_mat = np.zeros((T + S, S, J))
    n_mat = np.zeros((T + S, S, J))
    euler_errors = np.zeros((T, 2 * S, J))

    for j in xrange(J):
            first_doughnut_params = (income_tax_params, tpi_params, initial_b)
            b_mat[0, -1, j], n_mat[0, -1, j] = np.array(opt.fsolve(firstdoughnutring, [guesses_b[0, -1, j], guesses_n[0, -1, j]],
                                                                   args=(r[0], w[0], initial_b, BQ[0, j], T_H[0], j,
                                                                   first_doughnut_params), xtol=MINIMIZER_TOL))

            for s in xrange(S - 2):  # Upper triangle
                ind2 = np.arange(s + 2)
                b_guesses_to_use = np.diag(
                    guesses_b[:S, :, j], S - (s + 2))
                n_guesses_to_use = np.diag(guesses_n[:S, :, j], S - (s + 2))

                # initialize array of diagonal elements
                length_diag = (np.diag(np.transpose(etr_params[:,:S,0]),S-(s+2))).shape[0]
                etr_params_to_use = np.zeros((length_diag,etr_params.shape[2]))
                mtrx_params_to_use = np.zeros((length_diag,mtrx_params.shape[2]))
                mtry_params_to_use = np.zeros((length_diag,mtry_params.shape[2]))
                for i in range(etr_params.shape[2]):
                    etr_params_to_use[:,i] = np.diag(np.transpose(etr_params[:,:S,i]),S-(s+2))
                    mtrx_params_to_use[:,i] = np.diag(np.transpose(mtrx_params[:,:S,i]),S-(s+2))
                    mtry_params_to_use[:,i] = np.diag(np.transpose(mtry_params[:,:S,i]),S-(s+2))


                inc_tax_params_upper = (analytical_mtrs, etr_params_to_use, mtrx_params_to_use, mtry_params_to_use)

                TPI_solver_params = (inc_tax_params_upper, tpi_params, initial_b)
                solutions = opt.fsolve(twist_doughnut, list(
                    b_guesses_to_use) + list(n_guesses_to_use), args=(
                    r, w, BQ[:, j], T_H, j, s, 0, TPI_solver_params), xtol=MINIMIZER_TOL)

                b_vec = solutions[:len(solutions) / 2]
                b_mat[ind2, S - (s + 2) + ind2, j] = b_vec
                n_vec = solutions[len(solutions) / 2:]
                n_mat[ind2, S - (s + 2) + ind2, j] = n_vec


            for t in xrange(0, T):
                # b_guesses_to_use = .75 * \
                #     np.diag(guesses_b[t + 1:t + S + 1, :, j])
                b_guesses_to_use = .75 * \
                    np.diag(guesses_b[t:t + S, :, j])
                n_guesses_to_use = np.diag(guesses_n[t:t + S, :, j])

                # initialize array of diagonal elements
                length_diag = (np.diag(np.transpose(etr_params[:,t:t+S,0]))).shape[0]
                etr_params_to_use = np.zeros((length_diag,etr_params.shape[2]))
                mtrx_params_to_use = np.zeros((length_diag,mtrx_params.shape[2]))
                mtry_params_to_use = np.zeros((length_diag,mtry_params.shape[2]))
                for i in range(etr_params.shape[2]):
                    etr_params_to_use[:,i] = np.diag(np.transpose(etr_params[:,t:t+S,i]))
                    mtrx_params_to_use[:,i] = np.diag(np.transpose(mtrx_params[:,t:t+S,i]))
                    mtry_params_to_use[:,i] = np.diag(np.transpose(mtry_params[:,t:t+S,i]))

                inc_tax_params_TP = (analytical_mtrs, etr_params_to_use, mtrx_params_to_use, mtry_params_to_use)


                TPI_solver_params = (inc_tax_params_TP, tpi_params, None)
                [solutions, infodict, ier, message] = opt.fsolve(twist_doughnut, list(
                    b_guesses_to_use) + list(n_guesses_to_use), args=(
                    r, w, BQ[:, j], T_H, j, None, t, TPI_solver_params), xtol=MINIMIZER_TOL, full_output=True)
                euler_errors[t, :, j] = infodict['fvec']

                b_vec = solutions[:S]
                b_mat[t + ind, ind, j] = b_vec
                n_vec = solutions[S:]
                n_mat[t + ind, ind, j] = n_vec


    return euler_errors, b_mat, n_mat

def initial_GDP_level(y_guess, gamma, epsilon, Z, initial_debt, B, L):
    # Solve for first-period output given debt ratio, wealth, labor input, and production function.
    # The solution arises from a 3-unknown, 3-equation system that is simplified below.
    # (1) initial_debt = D/Y by assumption.
    # (2) Y = f(K,L)
    # (3) K = B - D, where B is household wealth
    # This could easily be modified to allow for an initial level of foreign bondholding.
    # Salim Furth, 12/28/2016

    #alpha, gamma, epsilon, Z, initial_debt, B, L = y_inputs
    # error = y_guess - Z * ((B - initial_debt*y_guess) ** alpha) * (L ** (1 - alpha))
    if epsilon == 1:
        error = y_guess - Z*((B - initial_debt*y_guess)**gamma)*(L**(1-gamma))
    elif epsilon == 0:
        error = y_guess - Z*(gamma*(B - initial_debt*y_guess) + (1-gamma)*L)
    else:
        error = y_guess - (Z * (((gamma**(1/epsilon))*((B - initial_debt*y_guess)**((epsilon-1)/epsilon))) +
          (((1-gamma)**(1/epsilon))*(L**((epsilon-1)/epsilon))))**(epsilon/(epsilon-1)))

    return error


def run_TPI(income_tax_params, tpi_params, iterative_params, small_open_params, initial_values, SS_values, fiscal_params, biz_tax_params, output_dir="./OUTPUT", baseline_spending=False):

    # unpack tuples of parameters
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    maxiter, mindist_SS, mindist_TPI = iterative_params
    J, S, T, BW, beta, sigma, alpha, gamma, epsilon, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, imm_rates, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n, theta = tpi_params
    # K0, b_sinit, b_splus1init, L0, Y0,\
    #         w0, r0, BQ0, T_H_0, factor, tax0, c0, initial_b, initial_n, omega_S_preTP = initial_values
    small_open, tpi_firm_r, tpi_hh_r = small_open_params
    B0, b_sinit, b_splus1init, factor, initial_b, initial_n, omega_S_preTP, initial_debt = initial_values
    Kss, Bss, Lss, rss, wss, BQss, T_Hss, revenue_ss, bssmat_splus1, nssmat, Yss, Gss = SS_values
    tau_b, delta_tau = biz_tax_params
    if baseline_spending==False:
        budget_balance, ALPHA_T, ALPHA_G, tG1, tG2, rho_G, debt_ratio_ss = fiscal_params
    else:
        budget_balance, ALPHA_T, ALPHA_G, tG1, tG2, rho_G, debt_ratio_ss, T_Hbaseline, Gbaseline = fiscal_params

    print 'Government spending breakpoints are tG1: ', tG1, '; and tG2:', tG2

    TPI_FIG_DIR = output_dir
    # Initialize guesses at time paths
    # Make array of initial guesses for labor supply and savings
    domain = np.linspace(0, T, T)
    domain2 = np.tile(domain.reshape(T, 1, 1), (1, S, J))
    ending_b = bssmat_splus1
    guesses_b = (-1 / (domain2 + 1)) * (ending_b - initial_b) + ending_b
    ending_b_tail = np.tile(ending_b.reshape(1, S, J), (S, 1, 1))
    guesses_b = np.append(guesses_b, ending_b_tail, axis=0)

    domain3 = np.tile(np.linspace(0, 1, T).reshape(T, 1, 1), (1, S, J))
    guesses_n = domain3 * (nssmat - initial_n) + initial_n
    ending_n_tail = np.tile(nssmat.reshape(1, S, J), (S, 1, 1))
    guesses_n = np.append(guesses_n, ending_n_tail, axis=0)
    b_mat = guesses_b#np.zeros((T + S, S, J))
    n_mat = guesses_n#np.zeros((T + S, S, J))
    ind = np.arange(S)

    L_init = np.ones((T+S,))*Lss
    B_init = np.ones((T+S,))*Bss
    L_params = (e.reshape(1, S, J), omega[:T, :].reshape(T, S, 1), lambdas.reshape(1, 1, J), 'TPI')
    L_init[:T]  = firm.get_L(n_mat[:T], L_params)
    B_params = (omega[:T-1].reshape(T-1, S, 1), lambdas.reshape(1, 1, J), imm_rates[:T-1].reshape(T-1,S,1), g_n_vector[1:T], 'TPI')
    B_init[1:T] = household.get_K(b_mat[:T-1], B_params)
    B_init[0] = B0

    if small_open == False:
        if budget_balance:
            K_init = B_init
        else:
            K_init = B_init * Kss/Bss
    else:
        K_params = (Z, gamma, epsilon, delta, tau_b, delta_tau)
        K_init = firm.get_K(L_init, tpi_firm_r, K_params)

    K = K_init
#    if np.any(K < 0):
#        print 'K_init has negative elements. Setting them positive to prevent NAN.'
#        K[:T] = np.fmax(K[:T], 0.05*B[:T])

    L = L_init
    B = B_init
    Y_params = (Z, gamma, epsilon)
    Y = firm.get_Y(K, L, Y_params)
    w_params = (Z, gamma, epsilon)
    w = firm.get_w(Y, L, w_params)
    if small_open == False:
        r_params = (Z, gamma, epsilon, delta, tau_b, delta_tau)
        r = firm.get_r(Y, K, r_params)
    else:
        r = tpi_hh_r

    BQ = np.zeros((T + S, J))
    BQ0_params = (omega_S_preTP.reshape(S, 1), lambdas, rho.reshape(S, 1), g_n_vector[0], 'SS')
    BQ0 = household.get_BQ(r[0], initial_b, BQ0_params)
    for j in xrange(J):
        BQ[:, j] = list(np.linspace(BQ0[j], BQss[j], T)) + [BQss[j]] * S
    BQ = np.array(BQ)
    if budget_balance:
        if np.abs(T_Hss) < 1e-13 :
            T_Hss2 = 0.0 # sometimes SS is very small but not zero, even if taxes are zero, this get's rid of the approximation error, which affects the perc changes below
        else:
            T_Hss2 = T_Hss
        T_H = np.ones(T + S) * T_Hss2
        REVENUE = T_H
        G = np.zeros(T + S)
    elif baseline_spending==False:
        T_H = ALPHA_T * Y
    elif baseline_spending==True:
        T_H = T_Hbaseline
        T_H_new = T_H   # Need to set T_H_new for later reference
        G   = Gbaseline
        G_0 = Gbaseline[0]

    # Initialize some inputs
    # D = np.zeros(T + S)
    D = debt_ratio_ss*Y
    omega_shift = np.append(omega_S_preTP.reshape(1,S),omega[:T-1,:],axis=0)
    BQ_params = (omega_shift.reshape(T, S, 1), lambdas.reshape(1, 1, J), rho.reshape(1, S, 1),
                     g_n_vector[:T].reshape(T, 1), 'TPI')
    tax_params = np.zeros((T,S,J,etr_params.shape[2]))
    for i in range(etr_params.shape[2]):
        tax_params[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))
    REVENUE_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas.reshape(1, 1, J), omega[:T].reshape(T, S, 1), 'TPI',
                      tax_params, theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J, tau_b, delta_tau)


    # print 'D/Y:', D[:T]/Y[:T]
    # print 'T/Y:', T_H[:T]/Y[:T]
    # print 'G/Y:', G[:T]/Y[:T]
    # print 'Int payments to GDP:', (r[:T]*D[:T])/Y[:T]
    # quit()


    TPIiter = 0
    TPIdist = 10
    PLOT_TPI = False
    report_tG1 = False

    euler_errors = np.zeros((T, 2 * S, J))
    TPIdist_vec = np.zeros(maxiter)

    print 'analytical mtrs in tpi = ', analytical_mtrs


    while (TPIiter < maxiter) and (TPIdist >= mindist_TPI):

        # Plot TPI for K for each iteration, so we can see if there is a
        # problem
        if PLOT_TPI is True:
            #K_plot = list(K) + list(np.ones(10) * Kss)
            D_plot = list(D) + list(np.ones(10) * Yss * debt_ratio_ss)
            plt.figure()
            plt.axhline(
                y=Kss, color='black', linewidth=2, label=r"Steady State $\hat{K}$", ls='--')
            plt.plot(np.arange(
                T + 10), D_plot[:T + 10], 'b', linewidth=2, label=r"TPI time path $\hat{K}_t$")
            plt.savefig(os.path.join(TPI_FIG_DIR, "TPI_D"))

        if report_tG1 is True:
            print '\tAt time tG1-1:'
            print '\t\tG = ', G[tG1-1]
            print '\t\tK = ', K[tG1-1]
            print '\t\tr = ', r[tG1-1]
            print '\t\tD = ', D[tG1-1]


        guesses = (guesses_b, guesses_n)
        outer_loop_vars = (r, w, K, BQ, T_H)
        inner_loop_params = (income_tax_params, tpi_params, initial_values, ind)

        # Solve HH problem in inner loop
        euler_errors, b_mat, n_mat = inner_loop(guesses, outer_loop_vars, inner_loop_params)

        bmat_s = np.zeros((T, S, J))
        bmat_s[0, 1:, :] = initial_b[:-1, :]
        bmat_s[1:, 1:, :] = b_mat[:T-1, :-1, :]
        bmat_splus1 = np.zeros((T, S, J))
        bmat_splus1[:, :, :] = b_mat[:T, :, :]

        #L_params = (e.reshape(1, S, J), omega[:T, :].reshape(T, S, 1), lambdas.reshape(1, 1, J), 'TPI') # defined above
        L[:T]  = firm.get_L(n_mat[:T], L_params)
        #B_params = (omega[:T-1].reshape(T-1, S, 1), lambdas.reshape(1, 1, J), imm_rates[:T-1].reshape(T-1,S,1), g_n_vector[1:T], 'TPI') # defined above
        B[1:T] = household.get_K(bmat_splus1[:T-1], B_params)
        if np.any(B) < 0:
            print 'B has negative elements. B[0:9]:', B[0:9]
            print 'B[T-2:T]:', B[T-2,T]

        if small_open == False:
            if budget_balance:
                K[:T] = B[:T]
            else:
                if baseline_spending == False:
                    Y = T_H/ALPHA_T  #SBF 3/3: This seems totally unnecessary as both these variables are defined above.

#                tax_params = np.zeros((T,S,J,etr_params.shape[2]))
#                for i in range(etr_params.shape[2]):
#                    tax_params[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))

#                REVENUE_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas.reshape(1, 1, J), omega[:T].reshape(T, S, 1), 'TPI',
#                        tax_params, theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J, tau_b, delta_tau) # define above
                REVENUE = np.array(list(tax.revenue(np.tile(r[:T].reshape(T, 1, 1),(1,S,J)), np.tile(w[:T].reshape(T, 1, 1),(1,S,J)),
                       bmat_s, n_mat[:T,:,:], BQ[:T].reshape(T, 1, J), Y[:T], L[:T], K[:T], factor, REVENUE_params)) + [revenue_ss] * S)

                D_0    = initial_debt * Y[0]
                other_dg_params = (T, r, g_n_vector, g_y)
                if baseline_spending==False:
                    G_0    = ALPHA_G[0] * Y[0]
                dg_fixed_values = (Y, REVENUE, T_H, D_0,G_0)
                Dnew, G = fiscal.D_G_path(dg_fixed_values, fiscal_params, other_dg_params, baseline_spending=baseline_spending)
                K[:T] = B[:T] - Dnew[:T]
                if np.any(K < 0):
                    print 'K has negative elements. Setting them positive to prevent NAN.'
                    K[:T] = np.fmax(K[:T], 0.05*B[:T])
        else:
            # K_params previously set to =  (Z, gamma, epsilon, delta, tau_b, delta_tau)
            K[:T] = firm.get_K(L[:T], tpi_firm_r[:T], K_params)
        Y_params = (Z, gamma, epsilon)
        Ynew = firm.get_Y(K[:T], L[:T], Y_params)
        Y = Ynew
        w_params = (Z, gamma, epsilon)
        wnew = firm.get_w(Ynew[:T], L[:T], w_params)
        if small_open == False:
            r_params = (Z, gamma, epsilon, delta, tau_b, delta_tau)
            rnew = firm.get_r(Ynew[:T], K[:T], r_params)
        else:
            rnew = r.copy()

        print 'Y and T_H: ', Y[3], T_H[3]
#        omega_shift = np.append(omega_S_preTP.reshape(1,S),omega[:T-1,:],axis=0)  # defined above
#        BQ_params = (omega_shift.reshape(T, S, 1), lambdas.reshape(1, 1, J), rho.reshape(1, S, 1),
#                     g_n_vector[:T].reshape(T, 1), 'TPI')  # defined above
        b_mat_shift = np.append(np.reshape(initial_b,(1,S,J)),b_mat[:T-1,:,:],axis=0)
        BQnew = household.get_BQ(rnew[:T].reshape(T, 1), b_mat_shift, BQ_params)

#        tax_params = np.zeros((T,S,J,etr_params.shape[2]))
#        for i in range(etr_params.shape[2]):
#            tax_params[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))

#        REVENUE_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas.reshape(1, 1, J), omega[:T].reshape(T, S, 1), 'TPI',
#                tax_params, theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J, tau_b, delta_tau) # defined above
        REVENUE = np.array(list(tax.revenue(np.tile(rnew[:T].reshape(T, 1, 1),(1,S,J)), np.tile(wnew[:T].reshape(T, 1, 1),(1,S,J)),
               bmat_s, n_mat[:T,:,:], BQnew[:T].reshape(T, 1, J), Y[:T], L[:T], K[:T], factor, REVENUE_params)) + [revenue_ss] * S)

        if budget_balance:
            T_H_new = REVENUE
        elif baseline_spending==False:
            T_H_new = ALPHA_T[:T] * Y[:T]
        # If baseline_spending==True, no need to update T_H, which remains fixed.

        if small_open==True and budget_balance==False:
            # Loop through years to calculate debt and gov't spending. This is done earlier when small_open=False.
            D_0    = initial_debt * Y[0]
            other_dg_params = (T, r, g_n_vector, g_y)
            if baseline_spending==False:
                G_0    = ALPHA_G[0] * Y[0]
            dg_fixed_values = (Y, REVENUE, T_H, D_0,G_0)
            Dnew, G = fiscal.D_G_path(dg_fixed_values, fiscal_params, other_dg_params, baseline_spending=baseline_spending)

        w[:T] = utils.convex_combo(wnew[:T], w[:T], nu)
        r[:T] = utils.convex_combo(rnew[:T], r[:T], nu)
        BQ[:T] = utils.convex_combo(BQnew[:T], BQ[:T], nu)
        # D[:T] = utils.convex_combo(Dnew[:T], D[:T], nu)
        D = Dnew
        Y[:T] = utils.convex_combo(Ynew[:T], Y[:T], nu)
        if baseline_spending==False:
            T_H[:T] = utils.convex_combo(T_H_new[:T], T_H[:T], nu)
        guesses_b = utils.convex_combo(b_mat, guesses_b, nu)
        guesses_n = utils.convex_combo(n_mat, guesses_n, nu)

        print 'r diff: ', (rnew[:T]-r[:T]).max(), (rnew[:T]-r[:T]).min()
        print 'w diff: ', (wnew[:T]-w[:T]).max(), (wnew[:T]-w[:T]).min()
        print 'BQ diff: ', (BQnew[:T]-BQ[:T]).max(), (BQnew[:T]-BQ[:T]).min()
        print 'T_H diff: ', (T_H_new[:T]-T_H[:T]).max(), (T_H_new[:T]-T_H[:T]).min()

        if baseline_spending==False:
            if T_H.all() != 0:
                TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) + list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) + list(
                    utils.pct_diff_func(wnew[:T], w[:T])) + list(utils.pct_diff_func(T_H_new[:T], T_H[:T]))).max()
            else:
                TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) + list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) + list(
                    utils.pct_diff_func(wnew[:T], w[:T])) + list(np.abs(T_H[:T]))).max()
        else:
            # TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) + list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) + list(
            #     utils.pct_diff_func(wnew[:T], w[:T])) + list(utils.pct_diff_func(Dnew[:T], D[:T]))).max()
            TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) + list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) + list(
                utils.pct_diff_func(wnew[:T], w[:T])) + list(utils.pct_diff_func(Ynew[:T], Y[:T]))).max()

        TPIdist_vec[TPIiter] = TPIdist
        # After T=10, if cycling occurs, drop the value of nu
        # wait til after T=10 or so, because sometimes there is a jump up
        # in the first couple iterations
        # if TPIiter > 10:
        #     if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter - 1] > 0:
        #         nu /= 2
        #         print 'New Value of nu:', nu
        TPIiter += 1
        print 'Iteration:', TPIiter
        print '\tDistance:', TPIdist

        # print 'D/Y:', (D[:T]/Ynew[:T]).max(), (D[:T]/Ynew[:T]).min(), np.median(D[:T]/Ynew[:T])
        # print 'T/Y:', (T_H_new[:T]/Ynew[:T]).max(), (T_H_new[:T]/Ynew[:T]).min(), np.median(T_H_new[:T]/Ynew[:T])
        # print 'G/Y:', (G[:T]/Ynew[:T]).max(), (G[:T]/Ynew[:T]).min(), np.median(G[:T]/Ynew[:T])
        # print 'Int payments to GDP:', ((r[:T]*D[:T])/Ynew[:T]).max(), ((r[:T]*D[:T])/Ynew[:T]).min(), np.median((r[:T]*D[:T])/Ynew[:T])
        #
        # print 'D/Y:', (D[:T]/Ynew[:T])
        # print 'T/Y:', (T_H_new[:T]/Ynew[:T])
        # print 'G/Y:', (G[:T]/Ynew[:T])
        #
        # print 'deficit: ', REVENUE[:T] - T_H_new[:T] - G[:T]

    # Loop through years to calculate debt and gov't spending. The re-assignment of G0 & D0 is necessary because Y0 may change in the TPI loop.
    if budget_balance == False:
        D_0    = initial_debt * Y[0]
        other_dg_params = (T, r, g_n_vector, g_y)
        if baseline_spending==False:
            G_0    = ALPHA_G[0] * Y[0]
        dg_fixed_values = (Y, REVENUE, T_H, D_0,G_0)
        D, G = fiscal.D_G_path(dg_fixed_values, fiscal_params, other_dg_params, baseline_spending=baseline_spending)

    # Solve HH problem in inner loop
    guesses = (guesses_b, guesses_n)
    outer_loop_vars = (r, w, K, BQ, T_H)
    inner_loop_params = (income_tax_params, tpi_params, initial_values, ind)
    euler_errors, b_mat, n_mat = inner_loop(guesses, outer_loop_vars, inner_loop_params)

    bmat_s = np.zeros((T, S, J))
    bmat_s[0, 1:, :] = initial_b[:-1, :]
    bmat_s[1:, 1:, :] = b_mat[:T-1, :-1, :]
    bmat_splus1 = np.zeros((T, S, J))
    bmat_splus1[:, :, :] = b_mat[:T, :, :]

    #L_params = (e.reshape(1, S, J), omega[:T, :].reshape(T, S, 1), lambdas.reshape(1, 1, J), 'TPI') # defined above
    L[:T]  = firm.get_L(n_mat[:T], L_params)
    #B_params = (omega[:T-1].reshape(T-1, S, 1), lambdas.reshape(1, 1, J), imm_rates[:T-1].reshape(T-1,S,1), g_n_vector[1:T], 'TPI') # defined above
    B[1:T] = household.get_K(bmat_splus1[:T-1], B_params)

    if small_open == False:
        K[:T] = B[:T] - D[:T]
    else:
        # K_params previously set to = (Z, gamma, epsilon, delta, tau_b, delta_tau)
        K[:T] = firm.get_K(L[:T], tpi_firm_r[:T], K_params)
    # Y_params previously set to = (Z, gamma, epsilon)
    Ynew = firm.get_Y(K[:T], L[:T], Y_params)

    # testing for change in Y
    ydiff = Ynew[:T] - Y[:T]
    ydiff_max = np.amax(np.abs(ydiff))
    print 'ydiff_max = ', ydiff_max

    w_params = (Z, gamma, epsilon)
    wnew = firm.get_w(Ynew[:T], L[:T], w_params)
    if small_open == False:
        # r_params previously set to = (Z, gamma, epsilon, delta, tau_b, delta_tau)
        rnew = firm.get_r(Ynew[:T], K[:T], r_params)
    else:
        rnew = r

    # Note: previously, Y was not reassigned to equal Ynew at this point.
    Y = Ynew[:]

#    omega_shift = np.append(omega_S_preTP.reshape(1,S),omega[:T-1,:],axis=0)
#    BQ_params = (omega_shift.reshape(T, S, 1), lambdas.reshape(1, 1, J), rho.reshape(1, S, 1),
#                 g_n_vector[:T].reshape(T, 1), 'TPI')
    b_mat_shift = np.append(np.reshape(initial_b,(1,S,J)),b_mat[:T-1,:,:],axis=0)
    BQnew = household.get_BQ(rnew[:T].reshape(T, 1), b_mat_shift, BQ_params)

#    tax_params = np.zeros((T,S,J,etr_params.shape[2]))
#    for i in range(etr_params.shape[2]):
#        tax_params[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))

#    REVENUE_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas.reshape(1, 1, J), omega[:T].reshape(T, S, 1), 'TPI',
#            tax_params, theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J, tau_b, delta_tau)
    REVENUE = np.array(list(tax.revenue(np.tile(rnew[:T].reshape(T, 1, 1),(1,S,J)), np.tile(wnew[:T].reshape(T, 1, 1),(1,S,J)),
           bmat_s, n_mat[:T,:,:], BQnew[:T].reshape(T, 1, J), Ynew[:T], L[:T], K[:T], factor, REVENUE_params)) + [revenue_ss] * S)

    etr_params_path = np.zeros((T,S,J,etr_params.shape[2]))
    for i in range(etr_params.shape[2]):
        etr_params_path[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))
    tax_path_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas, 'TPI', retire, etr_params_path, h_wealth,
                       p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S)
    tax_path = tax.total_taxes(np.tile(r[:T].reshape(T, 1, 1),(1,S,J)), np.tile(w[:T].reshape(T, 1, 1),(1,S,J)), bmat_s,
                               n_mat[:T,:,:], BQ[:T, :].reshape(T, 1, J), factor, T_H[:T].reshape(T, 1, 1), None, False, tax_path_params)

    cons_params = (e.reshape(1, S, J), lambdas.reshape(1, 1, J), g_y)
    c_path = household.get_cons(r[:T].reshape(T, 1, 1), w[:T].reshape(T, 1, 1), bmat_s, bmat_splus1, n_mat[:T,:,:],
                   BQ[:T].reshape(T, 1, J), tax_path, cons_params)
    C_params = (omega[:T].reshape(T, S, 1), lambdas, 'TPI')
    C = household.get_C(c_path, C_params)

    if budget_balance==False:
        D_0    = initial_debt * Y[0]
        other_dg_params = (T, r, g_n_vector, g_y)
        if baseline_spending==False:
            G_0    = ALPHA_G[0] * Y[0]
        dg_fixed_values = (Y, REVENUE, T_H, D_0,G_0)
        D, G = fiscal.D_G_path(dg_fixed_values, fiscal_params, other_dg_params, baseline_spending=baseline_spending)


    if small_open == False:
        I_params = (delta, g_y, omega[:T].reshape(T, S, 1), lambdas, imm_rates[:T].reshape(T, S, 1), g_n_vector[1:T+1], 'TPI')
        I = firm.get_I(bmat_splus1[:T], K[1:T+1], K[:T], I_params)
        rc_error = Y[:T] - C[:T] - I[:T] - G[:T]
    else:
        #InvestmentPlaceholder = np.zeros(bmat_splus1[:T].shape)
        #I_params = (delta, g_y, omega[:T].reshape(T, S, 1), lambdas, imm_rates[:T].reshape(T, S, 1), g_n_vector[1:T+1], 'TPI')
        I = (1+g_n_vector[:T])*np.exp(g_y)*K[1:T+1] - (1.0 - delta) * K[:T] #firm.get_I(InvestmentPlaceholder, K[1:T+1], K[:T], I_params)
        BI_params = (0.0, g_y, omega[:T].reshape(T, S, 1), lambdas, imm_rates[:T].reshape(T, S, 1), g_n_vector[1:T+1], 'TPI')
        BI = firm.get_I(bmat_splus1[:T], B[1:T+1], B[:T], BI_params)
        new_borrowing = D[1:T]*(1+g_n_vector[1:T])*np.exp(g_y) - D[:T-1]
        rc_error = Y[:T-1] + new_borrowing - (C[:T-1] + BI[:T-1] + G[:T-1] ) + (tpi_hh_r[:T-1] * B[:T-1] - (delta + tpi_firm_r[:T-1])*K[:T-1] - tpi_hh_r[:T-1]*D[:T-1])
        #print 'Y(T-1):', Y[T-1], '\n','C(T-1):', C[T-1], '\n','K(T-1):', K[T-1], '\n','B(T-1):', B[T-1], '\n','BI(T-1):', BI[T-1], '\n','I(T-1):', I[T-1]

    rce_max = np.amax(np.abs(rc_error))
    print 'Max absolute value resource constraint error:', rce_max

    print'Checking time path for violations of constraints.'
    for t in xrange(T):
        household.constraint_checker_TPI(
            b_mat[t], n_mat[t], c_path[t], t, ltilde)

    eul_savings = euler_errors[:, :S, :].max(1).max(1)
    eul_laborleisure = euler_errors[:, S:, :].max(1).max(1)

   # print 'Max Euler error, savings: ', eul_savings
   # print 'Max Euler error labor supply: ', eul_laborleisure



    '''
    ------------------------------------------------------------------------
    Save variables/values so they can be used in other modules
    ------------------------------------------------------------------------
    '''

    output = {'Y': Y, 'K': K, 'L': L, 'C': C, 'I': I, 'BQ': BQ,
              'REVENUE': REVENUE, 'T_H': T_H, 'G': G, 'D': D,
              'r': r, 'w': w, 'b_mat': b_mat, 'n_mat': n_mat,
              'c_path': c_path, 'tax_path': tax_path,
              'eul_savings': eul_savings, 'eul_laborleisure': eul_laborleisure}

    tpi_dir = os.path.join(output_dir, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
    pickle.dump(output, open(tpi_vars, "wb"))

    macro_output = {'Y': Y, 'K': K, 'L': L, 'C': C, 'I': I,
                    'BQ': BQ, 'REVENUE': REVENUE, 'T_H': T_H, 'r': r, 'w': w,
                    'tax_path': tax_path}

    growth = (1+g_n_vector)*np.exp(g_y)
    with open('TPI_output.csv', 'wb') as csvfile:
        tpiwriter = csv.writer(csvfile)
        tpiwriter.writerow(Y)
        tpiwriter.writerow(D)
        tpiwriter.writerow(REVENUE)
        tpiwriter.writerow(G)
        tpiwriter.writerow(T_H)
        tpiwriter.writerow(C)
        tpiwriter.writerow(K)
        tpiwriter.writerow(I)
        tpiwriter.writerow(r)
        if small_open == True:
            tpiwriter.writerow(B)
            tpiwriter.writerow(BI)
            tpiwriter.writerow(new_borrowing)
        tpiwriter.writerow(growth)
        tpiwriter.writerow(rc_error)
        tpiwriter.writerow(ydiff)


    if np.any(G) < 0:
        print 'Government spending is negative along transition path to satisfy budget'

    if ((TPIiter >= maxiter) or (np.absolute(TPIdist) > mindist_TPI)) and ENFORCE_SOLUTION_CHECKS :
        raise RuntimeError("Transition path equlibrium not found (TPIdist)")

    if ((np.any(np.absolute(rc_error) >= mindist_TPI))
        and ENFORCE_SOLUTION_CHECKS):
        raise RuntimeError("Transition path equlibrium not found (rc_error)")

    if ((np.any(np.absolute(eul_savings) >= mindist_TPI) or
        (np.any(np.absolute(eul_laborleisure) > mindist_TPI)))
        and ENFORCE_SOLUTION_CHECKS):
        raise RuntimeError("Transition path equlibrium not found (eulers)")

    # Non-stationary output
    # macro_ns_output = {'K_ns_path': K_ns_path, 'C_ns_path': C_ns_path, 'I_ns_path': I_ns_path,
    #           'L_ns_path': L_ns_path, 'BQ_ns_path': BQ_ns_path,
    #           'rinit': rinit, 'Y_ns_path': Y_ns_path, 'T_H_ns_path': T_H_ns_path,
    #           'w_ns_path': w_ns_path}


    return output, macro_output
