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
    #initial_b = ss_baseline_vars['bssmat_s'] + ss_baseline_vars['BQss']/lambdas
    initial_b = ss_baseline_vars['bssmat_splus1']
    initial_n = ss_baseline_vars['nssmat']

    SS_values = (ss_baseline_vars['Kss'],ss_baseline_vars['Lss'], ss_baseline_vars['rss'], 
                 ss_baseline_vars['wss'], ss_baseline_vars['BQss'], ss_baseline_vars['T_Hss'],
                 ss_baseline_vars['bssmat_splus1'], ss_baseline_vars['nssmat'])

    # Make a vector of all one dimensional parameters, to be used in the
    # following functions
    wealth_tax_params = [sim_params['h_wealth'], sim_params['p_wealth'], sim_params['m_wealth']]
    ellipse_params = [sim_params['b_ellipse'], sim_params['upsilon']]
    chi_params = [sim_params['chi_b_guess'], sim_params['chi_n_guess']]

    N_tilde = sim_params['omega'].sum(1) #this should just be one in each year given how we've constructed omega
    sim_params['omega'] = sim_params['omega'] / N_tilde.reshape(sim_params['T'] + sim_params['S'], 1)

    tpi_params = [sim_params['J'], sim_params['S'], sim_params['T'], sim_params['BW'], 
                  sim_params['beta'], sim_params['sigma'], sim_params['alpha'], 
                  sim_params['Z'], sim_params['delta'], sim_params['ltilde'], 
                  sim_params['nu'], sim_params['g_y'], sim_params['g_n_vector'], 
                  sim_params['tau_payroll'], sim_params['tau_bq'], sim_params['rho'], sim_params['omega'], N_tilde,
                  sim_params['lambdas'], sim_params['e'], sim_params['retire'], sim_params['mean_income_data'], factor] + \
                  wealth_tax_params + ellipse_params + chi_params
    iterative_params = [sim_params['maxiter'], sim_params['mindist_SS'], sim_params['mindist_TPI']]
    

    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n = tpi_params

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
    Set other parameters and initial values
    ------------------------------------------------------------------------
    '''
    # Get an initial distribution of capital with the initial population
    # distribution
    K0_params = (omega[0].reshape(S, 1), lambdas, g_n_vector[0], 'SS')
    K0 = household.get_K(initial_b, K0_params)

    b_sinit = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_b[:-1]))
    b_splus1init = initial_b
    L0_params = (e, omega[0].reshape(S, 1), lambdas, 'SS')
    L0 = firm.get_L(initial_n, L0_params)
    Y0_params = (alpha, Z)
    Y0 = firm.get_Y(K0, L0, Y0_params)
    w0 = firm.get_w(Y0, L0, alpha)
    r0_params = (alpha, delta)
    r0 = firm.get_r(Y0, K0, r0_params)

    BQ0_params = (omega[0].reshape(S, 1), lambdas, rho.reshape(S, 1), g_n_vector[0], 'SS')
    BQ0 = household.get_BQ(r0, initial_b, BQ0_params)

    theta_params = (e, J, omega[0].reshape(S, 1), lambdas)
    theta = tax.replacement_rate_vals(initial_n, w0, factor, theta_params)

    T_H_params = (e, lambdas, omega[0].reshape(S, 1), 'SS', etr_params_TP[:,0,:], 
                    theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J)
    T_H_0 = tax.get_lump_sum(r0, w0, b_sinit, initial_n, BQ0, factor, T_H_params)

    etr_params_3D = np.tile(np.reshape(etr_params_TP[:,0,:],(S,1,etr_params_TP.shape[2])),(1,J,1))
    tax0_params = (e, lambdas, 'SS', retire, etr_params_3D, h_wealth, p_wealth, m_wealth, 
                    tau_payroll, theta, tau_bq, J, S)
    tax0 = tax.total_taxes(r0, w0, b_sinit, initial_n, BQ0, factor, T_H_0, None, False, tax0_params)

    c0_params = (e, lambdas.reshape(1, J), g_y)
    c0 = household.get_cons(r0, w0, b_sinit, b_splus1init, initial_n, BQ0.reshape(
        1, J), tax0, c0_params)

    initial_values = (K0, b_sinit, b_splus1init, L0, Y0,
            w0, r0, BQ0, T_H_0, factor, tax0, c0, initial_b, initial_n)

    return (income_tax_params, tpi_params, iterative_params, initial_values, SS_values)


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
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n = tpi_params


    b_splus1 = float(guesses[0])
    n = float(guesses[1])
    b_s = float(initial_b[-2, j])
    # Euler 1 equations

    # theta_params = (e[-1, j], 1, omega[0].reshape(S, 1), lambdas[j])
    # theta = tax.replacement_rate_vals(n, w, factor, theta_params)
    theta = np.zeros((J,))

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
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n = tpi_params

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

    # theta_params = (e[-1, j], 1, omega[0].reshape(S, 1), lambdas[j])
    # theta = tax.replacement_rate_vals(n, w, factor, theta_params)
    theta = np.zeros((J,)) 

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

    error1 = household.marg_ut_cons(cons_s, sigma) - beta * (1 - rho[-(length):]) * np.exp(-sigma * g_y) * deriv_savings * household.marg_ut_cons(
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
    error2[mask5] += 1e12
    return list(error1.flatten()) + list(error2.flatten())


def inner_loop(guesses, outer_loop_vars, params):
    '''
    Solves inner loop of TPI.  Given path of economic aggregates and factor prices, solves
    househld problem 

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
    income_tax_params, tpi_params, initial_values, theta, ind = params
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n = tpi_params
    K0, b_sinit, b_splus1init, L0, Y0,\
            w0, r0, BQ0, T_H_0, factor, tax0, c0, initial_b, initial_n = initial_values

    guesses_b, guesses_n = guesses
    r, w, K, BQ, T_H = outer_loop_vars

    # initialize arrays
    b_mat = np.zeros((T + S, S, J))
    n_mat = np.zeros((T + S, S, J))
    euler_errors = np.zeros((T, 2 * S, J))

    for j in xrange(J):
            first_doughnut_params = (income_tax_params, tpi_params, initial_b)
            b_mat[1, -1, j], n_mat[0, -1, j] = np.array(opt.fsolve(firstdoughnutring, [guesses_b[1, -1, j], guesses_n[0, -1, j]],
                                                                   args=(r[1], w[1], initial_b, BQ[1, j], T_H[1], j, 
                                                                   first_doughnut_params), xtol=MINIMIZER_TOL))

            for s in xrange(S - 2):  # Upper triangle
                ind2 = np.arange(s + 2)
                b_guesses_to_use = np.diag(
                    guesses_b[1:S + 1, :, j], S - (s + 2))
                n_guesses_to_use = np.diag(guesses_n[:S, :, j], S - (s + 2))

                # initialize array of diagonal elements
                length_diag = (np.diag(np.transpose(etr_params[:S,:,0]),S-(s+2))).shape[0]
                etr_params_to_use = np.zeros((length_diag,etr_params.shape[2]))
                mtrx_params_to_use = np.zeros((length_diag,mtrx_params.shape[2]))
                mtry_params_to_use = np.zeros((length_diag,mtry_params.shape[2]))
                for i in range(etr_params.shape[2]):
                    etr_params_to_use[:,i] = np.diag(np.transpose(etr_params[:S,:,i]),S-(s+2))
                    mtrx_params_to_use[:,i] = np.diag(np.transpose(mtrx_params[:S,:,i]),S-(s+2))
                    mtry_params_to_use[:,i] = np.diag(np.transpose(mtry_params[:S,:,i]),S-(s+2))

                inc_tax_params_upper = (analytical_mtrs, etr_params_to_use, mtrx_params_to_use, mtry_params_to_use)

                TPI_solver_params = (inc_tax_params_upper, tpi_params, initial_b)
                solutions = opt.fsolve(twist_doughnut, list(
                    b_guesses_to_use) + list(n_guesses_to_use), args=(
                    r, w, BQ[:, j], T_H, j, s, 0, TPI_solver_params), xtol=MINIMIZER_TOL)

                b_vec = solutions[:len(solutions) / 2]
                b_mat[1 + ind2, S - (s + 2) + ind2, j] = b_vec
                n_vec = solutions[len(solutions) / 2:]
                n_mat[ind2, S - (s + 2) + ind2, j] = n_vec

            for t in xrange(0, T):
                b_guesses_to_use = .75 * \
                    np.diag(guesses_b[t + 1:t + S + 1, :, j])
                n_guesses_to_use = np.diag(guesses_n[t:t + S, :, j])

                # initialize array of diagonal elements
                length_diag = (np.diag(np.transpose(etr_params[:,t:t+S,i]))).shape[0]
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
                b_mat[t + 1 + ind, ind, j] = b_vec
                n_vec = solutions[S:]
                n_mat[t + ind, ind, j] = n_vec
                # inputs = list(solutions)

                # TPI_solver_params = (inc_tax_params_TP, tpi_params, None)
                # euler_errors[t, :, j] = np.abs(opt.fsolve(twist_doughnut, inputs, args=(
                #     r, w, BQ[:, j], T_H, j, None, t, TPI_solver_params), xtol=MINIMIZER_TOL))

    return euler_errors, b_mat, n_mat


def run_TPI(income_tax_params, tpi_params, iterative_params, initial_values, SS_values, output_dir="./OUTPUT"):

    # unpack tuples of parameters
    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_params
    maxiter, mindist_SS, mindist_TPI = iterative_params
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_vector, tau_payroll, tau_bq, rho, omega, N_tilde, lambdas, e, retire, mean_income_data,\
                  factor, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon, chi_b, chi_n = tpi_params
    K0, b_sinit, b_splus1init, L0, Y0,\
            w0, r0, BQ0, T_H_0, factor, tax0, c0, initial_b, initial_n = initial_values
    Kss, Lss, rss, wss, BQss, T_Hss, bssmat_splus1, nssmat = SS_values


    TPI_FIG_DIR = output_dir
    # Initialize guesses at time paths
    domain = np.linspace(0, T, T)
    K_init = (-1 / (domain + 1)) * (Kss - K0) + Kss
    K_init[-1] = Kss
    K_init = np.array(list(K_init) + list(np.ones(S) * Kss))
    L_init = np.ones(T + S) * Lss

    K = K_init
    L = L_init
    Y_params = (alpha, Z)
    Y = firm.get_Y(K, L, Y_params)
    w = firm.get_w(Y, L, alpha)
    r_params = (alpha, delta)
    r = firm.get_r(Y, K, r_params)
    BQ = np.zeros((T + S, J))
    for j in xrange(J):
        BQ[:, j] = list(np.linspace(BQ0[j], BQss[j], T)) + [BQss[j]] * S
    BQ = np.array(BQ)
    if T_Hss < 1e-13 and T_Hss > 0.0 :
        T_Hss2 = 0.0 # sometimes SS is very small but not zero, even if taxes are zero, this get's rid of the approximation error, which affects the perc changes below
    else:
        T_Hss2 = T_Hss   
    T_H = np.ones(T + S) * T_Hss2

    # Make array of initial guesses for labor supply and savings
    domain2 = np.tile(domain.reshape(T, 1, 1), (1, S, J))
    ending_b = bssmat_splus1
    guesses_b = (-1 / (domain2 + 1)) * (ending_b - initial_b) + ending_b
    ending_b_tail = np.tile(ending_b.reshape(1, S, J), (S, 1, 1))
    guesses_b = np.append(guesses_b, ending_b_tail, axis=0)

    domain3 = np.tile(np.linspace(0, 1, T).reshape(T, 1, 1), (1, S, J))
    guesses_n = domain3 * (nssmat - initial_n) + initial_n
    ending_n_tail = np.tile(nssmat.reshape(1, S, J), (S, 1, 1))
    guesses_n = np.append(guesses_n, ending_n_tail, axis=0)
    b_mat = np.zeros((T + S, S, J))
    n_mat = np.zeros((T + S, S, J))
    ind = np.arange(S)

    TPIiter = 0
    TPIdist = 10
    PLOT_TPI = False

    euler_errors = np.zeros((T, 2 * S, J))
    TPIdist_vec = np.zeros(maxiter)


    while (TPIiter < maxiter) and (TPIdist >= mindist_TPI):
        # Plot TPI for K for each iteration, so we can see if there is a
        # problem
        if PLOT_TPI is True:
            K_plot = list(K) + list(np.ones(10) * Kss)
            L_plot = list(L) + list(np.ones(10) * Lss)
            plt.figure()
            plt.axhline(
                y=Kss, color='black', linewidth=2, label=r"Steady State $\hat{K}$", ls='--')
            plt.plot(np.arange(
                T + 10), Kpath_plot[:T + 10], 'b', linewidth=2, label=r"TPI time path $\hat{K}_t$")
            plt.savefig(os.path.join(TPI_FIG_DIR, "TPI_K"))
        # Uncomment the following print statements to make sure all euler equations are converging.
        # If they don't, then you'll have negative consumption or consumption spikes.  If they don't,
        # it is the initial guesses.  You might need to scale them differently.  It is rather delicate for the first
        # few periods and high ability groups.

        # theta_params = (e[-1, j], 1, omega[0].reshape(S, 1), lambdas[j])
        # theta = tax.replacement_rate_vals(n, w, factor, theta_params)
        theta = np.zeros((J,)) 

        guesses = (guesses_b, guesses_n)
        outer_loop_vars = (r, w, K, BQ, T_H)
        inner_loop_params = (income_tax_params, tpi_params, initial_values, theta, ind)

        # Solve HH problem in inner loop
        euler_errors, b_mat, n_mat = inner_loop(guesses, outer_loop_vars, inner_loop_params)


        # if euler_errors.max() > 1e-6:
        #     print 't-loop:', euler_errors.max()
        # Force the initial distribution of capital to be as given above.
        b_mat[0, :, :] = initial_b
        K_params = (omega[:T].reshape(T, S, 1), lambdas.reshape(1, 1, J), g_n_vector[:T], 'TPI')
        K[:T] = household.get_K(b_mat[:T], K_params)
        L_params = (e.reshape(1, S, J), omega[:T, :].reshape(T, S, 1), lambdas.reshape(1, 1, J), 'TPI')
        L[:T]  = firm.get_L(n_mat[:T], L_params)

        Y_params = (alpha, Z)
        Ynew = firm.get_Y(K[:T], L[:T], Y_params)
        wnew = firm.get_w(Ynew[:T], L[:T], alpha)
        r_params = (alpha, delta)
        rnew = firm.get_r(Ynew[:T], K[:T], r_params)

        BQ_params = (omega[:T].reshape(T, S, 1), lambdas.reshape(1, 1, J), rho.reshape(1, S, 1), 
                    g_n_vector[:T].reshape(T, 1), 'TPI')
        BQnew = household.get_BQ(rnew[:T].reshape(T, 1), b_mat[:T,:,:], BQ_params)
        bmat_s = np.zeros((T, S, J))
        bmat_s[:, 1:, :] = b_mat[:T, :-1, :]
        bmat_splus1 = np.zeros((T, S, J))
        bmat_splus1[:, :, :] = b_mat[1:T + 1, :, :]

        TH_tax_params = np.zeros((T,S,J,etr_params.shape[2]))
        for i in range(etr_params.shape[2]):
            TH_tax_params[:,:,:,i] = np.tile(np.reshape(np.transpose(etr_params[:,:T,i]),(T,S,1)),(1,1,J))

        T_H_params = (np.tile(e.reshape(1, S, J),(T,1,1)), lambdas.reshape(1, 1, J), omega[:T].reshape(T, S, 1), 'TPI', 
                TH_tax_params, theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J)
        T_H_new = np.array(list(tax.get_lump_sum(np.tile(rnew[:T].reshape(T, 1, 1),(1,S,J)), np.tile(wnew[:T].reshape(T, 1, 1),(1,S,J)),
               bmat_s, n_mat[:T,:,:], BQnew[:T].reshape(T, 1, J), factor, T_H_params)) + [T_Hss] * S)

        w[:T] = utils.convex_combo(wnew[:T], w[:T], nu)
        r[:T] = utils.convex_combo(rnew[:T], r[:T], nu)
        BQ[:T] = utils.convex_combo(BQnew[:T], BQ[:T], nu)
        T_H[:T] = utils.convex_combo(T_H_new[:T], T_H[:T], nu)
        guesses_b = utils.convex_combo(b_mat, guesses_b, nu)
        guesses_n = utils.convex_combo(n_mat, guesses_n, nu)
        if T_H.all() != 0:
            TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) + list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) + list(
                utils.pct_diff_func(wnew[:T], w[:T])) + list(utils.pct_diff_func(T_H_new[:T], T_H[:T]))).max()
        else:
            TPIdist = np.array(list(utils.pct_diff_func(rnew[:T], r[:T])) + list(utils.pct_diff_func(BQnew[:T], BQ[:T]).flatten()) + list(
                utils.pct_diff_func(wnew[:T], w[:T])) + list(np.abs(T_H_new[:T], T_H[:T]))).max()
        TPIdist_vec[TPIiter] = TPIdist
        # After T=10, if cycling occurs, drop the value of nu
        # wait til after T=10 or so, because sometimes there is a jump up
        # in the first couple iterations
        # if TPIiter > 10:
        #     if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter - 1] > 0:
        #         nu /= 2
        #         print 'New Value of nu:', nu
        TPIiter += 1
        print '\tIteration:', TPIiter
        print '\t\tDistance:', TPIdist

    if ((TPIiter >= maxiter) or (np.absolute(TPIdist) > mindist_TPI)) and ENFORCE_SOLUTION_CHECKS :
        raise RuntimeError("Transition path equlibrium not found")


    Y[:T] = Ynew


    # Solve HH problem in inner loop
    guesses = (guesses_b, guesses_n)
    outer_loop_vars = (r, w, K, BQ, T_H)
    inner_loop_params = (income_tax_params, tpi_params, initial_values, theta, ind)
    euler_errors, b_mat, n_mat = inner_loop(guesses, outer_loop_vars, inner_loop_params)
    b_mat[0, :, :] = initial_b

    K_params = (omega[:T].reshape(T, S, 1), lambdas.reshape(1, 1, J), g_n_vector[:T], 'TPI')
    K[:T] = household.get_K(b_mat[:T], K_params) # this is what old code does, but it's strange - why use 
    # b_mat -- what is going on with initial period, etc.

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
    I_params = (delta, g_y, g_n_vector[:T])
    I = firm.get_I(K[1:T+1], K[:T], I_params)
    print 'Resource Constraint Difference:', Y[:T] - C[:T] - I[:T]


    print'Checking time path for violations of constaints.'
    for t in xrange(T):
        household.constraint_checker_TPI(
            b_mat[t], n_mat[t], c_path[t], t, ltilde)

    eul_savings = euler_errors[:, :S, :].max(1).max(1)
    eul_laborleisure = euler_errors[:, S:, :].max(1).max(1)

    print 'Max Euler error, savings: ', eul_savings
    print 'Max Euler error labor supply: ', eul_laborleisure

    if ((np.any(np.absolute(eul_savings) >= mindist_TPI) or
        (np.any(np.absolute(eul_laborleisure) > mindist_TPI)))
        and ENFORCE_SOLUTION_CHECKS):
        raise RuntimeError("Transition path equlibrium not found")

    '''
    ------------------------------------------------------------------------
    Save variables/values so they can be used in other modules
    ------------------------------------------------------------------------
    '''

    output = {'Y': Y, 'K': K, 'L': L, 'C': C, 'I': I, 'BQ': BQ, 
              'T_H': T_H, 'r': r, 'w': w, 'b_mat': b_mat, 'n_mat': n_mat, 
              'c_path': c_path, 'tax_path': tax_path,
              'eul_savings': eul_savings, 'eul_laborleisure': eul_laborleisure}

    tpi_dir = os.path.join(output_dir, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
    pickle.dump(output, open(tpi_vars, "wb"))
    
    macro_output = {'Y': Y, 'K': K, 'L': L, 'C': C, 'I': I,
                    'BQ': BQ, 'T_H': T_H, 'r': r, 'w': w, 
                    'tax_path': tax_path}

    # Non-stationary output
    # macro_ns_output = {'K_ns_path': K_ns_path, 'C_ns_path': C_ns_path, 'I_ns_path': I_ns_path,
    #           'L_ns_path': L_ns_path, 'BQ_ns_path': BQ_ns_path,
    #           'rinit': rinit, 'Y_ns_path': Y_ns_path, 'T_H_ns_path': T_H_ns_path,
    #           'w_ns_path': w_ns_path}


    return output, macro_output
