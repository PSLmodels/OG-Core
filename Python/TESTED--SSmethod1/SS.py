'''
------------------------------------------------------------------------
Last updated: 6/4/2015

Calculates steady state of OLG model with S age cohorts.

This py-file calls the following other file(s):
            income.py
            demographics.py
            tax_funcs.py
            OUTPUT/given_params.pkl
            OUTPUT/Saved_moments/wealth_data_moments_fit_{}.pkl
                name depends on which percentile
            OUTPUT/Saved_moments/labor_data_moments.pkl
            OUTPUT/income_demo_vars.pkl
            OUTPUT/Saved_moments/{}.pkl
                name depends on what iteration just ran
            OUTPUT/SS/d_inc_guess.pkl
                if calibrating the income tax to match the wealth tax

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/income_demo_vars.pkl
            OUTPUT/Saved_moments/{}.pkl
                name depends on what iteration is being run
            OUTPUT/Saved_moments/payroll_inputs.pkl
            OUTPUT/SSinit/ss_init.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import os
import scipy.optimize as opt
import cPickle as pickle

import tax_funcs as tax
import household_funcs as house
import firm_funcs as firm
import misc_funcs


'''
------------------------------------------------------------------------
Imported user given values
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
lambdas_init  = percent of each age cohort in each ability group
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
nu_init      = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_new
Z            = total factor productivity parameter in firms' production
               function
delta        = depreciation rate of capital for each cohort
ltilde       = measure of time each individual is endowed with each
               period
eta          = Frisch elasticity of labor supply
g_y          = growth rate of technology for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
b_ellipse    = value of b for elliptical fit of utility function
k_ellipse    = value of k for elliptical fit of utility function
slow_work    = time at which chi_n starts increasing from 1
mean_income_data  = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income = used to calibrate income tax (scalar)
b_tax_income = used to calibrate income tax (scalar)
c_tax_income = used to calibrate income tax (scalar)
d_tax_income = used to calibrate income tax (scalar)
tau_bq       = bequest tax (scalar)
tau_payroll  = payroll tax (scalar)
theta    = payback value for payroll tax (scalar)
retire       = age in which individuals retire(scalar)
h_wealth     = wealth tax parameter h
p_wealth     = wealth tax parameter p
m_wealth     = wealth tax parameter m
scal         = value to scale the initial guesses by in order to get the
               fsolve to converge
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
# Set lowest ability group's wealth to be a positive, not negative, number for the calibration
wealth_data_array[2:26, 0] = 500.0

variables = pickle.load(open("OUTPUT/Saved_moments/labor_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

variables = pickle.load(open("OUTPUT/Saved_moments/params_given.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
if os.path.isfile("OUTPUT/Saved_moments/params_changed.pkl"):
    variables = pickle.load(open("OUTPUT/Saved_moments/params_changed.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]

'''
------------------------------------------------------------------------
Generate income and demographic parameters
------------------------------------------------------------------------
e            = S x J matrix of age dependent possible working abilities
               e_s
omega        = T x S x J array of demographics
g_n          = steady state population growth rate
omega_SS     = steady state population distribution
surv_rate    = S x 1 array of survival rates
rho    = S x 1 array of mortality rates
------------------------------------------------------------------------
'''

if SS_stage == 'first_run_for_guesses':
    import income_polynomials as income
    import demographics
    # These values never change, so only run it once
    omega, g_n, omega_SS, surv_rate = demographics.get_omega(
        S, J, T, lambdas, starting_age, ending_age, E)
    e = income.get_e(S, J, starting_age, ending_age, lambdas, omega_SS)
    rho = 1-surv_rate
    rho[-1] = 1.0
    var_names = ['omega', 'g_n', 'omega_SS', 'surv_rate', 'e', 'rho']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/income_demo_vars.pkl", "w"))
else:
    variables = pickle.load(open("OUTPUT/Saved_moments/income_demo_vars.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]


'''
------------------------------------------------------------------------
    Define Functions
------------------------------------------------------------------------
'''

# Make a vector of all one dimensional parameters, to be used in the following functions
income_tax_params = [a_tax_income, b_tax_income, c_tax_income, d_tax_income]
wealth_tax_params = [h_wealth, p_wealth, m_wealth]
ellipse_params = [b_ellipse, upsilon]
parameters = [J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data] + income_tax_params + wealth_tax_params + ellipse_params

# Functions


def Steady_State_SS(guesses, chi_params, params, weights_SS, rho_vec, lambdas, theta, tau_bq, e):
    '''
    Parameters: Steady state distribution of capital guess as array
                size 2*S*J

    Returns:    Array of 2*S*J Euler equation errors
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    chi_b = np.tile(np.array(chi_params[:J]).reshape(1, J), (S, 1))
    chi_n = np.array(chi_params[J:])
    b_guess = guesses[0: S * J].reshape((S, J))
    K = house.get_K(b_guess, weights_SS)
    n_guess = guesses[S * J:-1].reshape((S, J))
    L = firm.get_L(e, n_guess, weights_SS)
    Y = firm.get_Y(K, L, params)
    w = firm.get_w(Y, L, params)
    r = firm.get_r(Y, K, params)
    BQ = (1 + r) * (b_guess * weights_SS * rho_vec.reshape(S, 1)).sum(0)
    b_s = np.array(list(np.zeros(J).reshape(1, J)) + list(b_guess[:-1, :]))
    b_splus1 = b_guess
    b_splus2 = np.array(list(b_guess[1:]) + list(np.zeros(J).reshape(1, J)))
    factor = guesses[-1]
    T_H = tax.get_lump_sum(r, b_s, w, e, n_guess, BQ, lambdas, factor, weights_SS, 'SS', params, theta, tau_bq)
    error1 = house.euler_savings_func(w, r, e, n_guess, b_s, b_splus1, b_splus2, BQ.reshape(1, J), factor, T_H, chi_b, params, theta, tau_bq, rho_vec, lambdas)
    error2 = house.euler_labor_leisure_func(w, r, e, n_guess, b_s, b_splus1, BQ.reshape(1, J), factor, T_H, chi_n, params, theta, tau_bq, lambdas)
    average_income_model = ((r * b_s + w * e * n_guess) * weights_SS).sum()
    error3 = [mean_income_data - factor * average_income_model]
    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] += 1e9
    mask2 = n_guess > ltilde
    error2[mask2] += 1e9
    if b_guess.sum() <= 0:
        error1 += 1e9
    tax1 = tax.total_taxes(r, b_s, w, e, n_guess, BQ, lambdas, factor, T_H, None, 'SS', False, params, theta, tau_bq)
    cons = house.get_cons(r, b_s, w, e, n_guess, BQ.reshape(1, J), lambdas, b_splus1, params, tax1)
    mask3 = cons < 0
    error2[mask3] += 1e9
    mask4 = b_guess[:-1] <= 0
    error1[mask4] += 1e9
    # print np.abs(np.array(list(error1.flatten()) + list(
    #     error2.flatten()) + error3)).max()
    return list(error1.flatten()) + list(
        error2.flatten()) + error3


def function_to_minimize(chi_guesses_init, params, weights_SS, rho_vec, lambdas, theta, tau_bq, e, wealth_data_array):
    '''
    Parameters:
        chi_guesses_init = guesses for chi_b
        other_guesses_init = guesses for the distribution of capital and labor
                            stock, and factor value

    Returns:
        The max absolute deviation between the actual and simulated
            wealth moments
    '''
    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    print chi_guesses_init
    Steady_State_SS_X = lambda x: Steady_State_SS(x, chi_guesses_init, params, weights_SS, rho_vec, lambdas, theta, tau_bq, e)

    variables = pickle.load(open("OUTPUT/Saved_moments/minimization_solutions.pkl", "r"))
    for key in variables:
        globals()[key+'_pre'] = variables[key]
    solutions = opt.fsolve(Steady_State_SS_X, solutions_pre, xtol=1e-13)
    b_guess = solutions[0: S * J].reshape((S, J))
    # Wealth Calibration Euler
    error5 = list(misc_funcs.check_wealth_calibration(b_guess[:-1, :], solutions[-1], wealth_data_array, params))
    print error5
    # labor calibration euler
    labor_sim = ((solutions[S*J:2*S*J]).reshape(S, J)*lambdas.reshape(1, J)).sum(axis=1)
    error6 = list(misc_funcs.perc_dif_func(labor_sim, labor_dist_data))
    # combine eulers
    output = np.array(error5 + error6)
    # Constraints
    fsolve_no_converg = np.abs(Steady_State_SS_X(solutions)).max()
    if np.isnan(fsolve_no_converg):
        fsolve_no_converg = 1e6
    if fsolve_no_converg > 1e-4:
        output += 1e9
    else:
        var_names = ['solutions']
        dictionary = {}
        for key in var_names:
            dictionary[key] = locals()[key]
        pickle.dump(dictionary, open("OUTPUT/Saved_moments/minimization_solutions.pkl", "w"))
    if (chi_guesses_init <= 0.0).any():
        output += 1e9
    weighting_mat = np.eye(2*J + S)
    scaling_val = 100.0
    value = np.dot(scaling_val * np.dot(output.reshape(1, 2*J+S), weighting_mat), scaling_val * output.reshape(2*J+S, 1))
    print value.sum()
    return value.sum()

'''
------------------------------------------------------------------------
    Run SS in various ways, depending on the stage of the simulation
------------------------------------------------------------------------
'''


chi_n_guess = np.array([47.12000874 , 22.22762421 , 14.34842241 , 10.67954008 ,  8.41097278
 ,  7.15059004 ,  6.46771332 ,  5.85495452 ,  5.46242013 ,  5.00364263
 ,  4.57322063 ,  4.53371545 ,  4.29828515 ,  4.10144524 ,  3.8617942  ,  3.57282
 ,  3.47473172 ,  3.31111347 ,  3.04137299 ,  2.92616951 ,  2.58517969
 ,  2.48761429 ,  2.21744847 ,  1.9577682  ,  1.66931057 ,  1.6878927
 ,  1.63107201 ,  1.63390543 ,  1.5901486  ,  1.58143606 ,  1.58005578
 ,  1.59073213 ,  1.60190899 ,  1.60001831 ,  1.67763741 ,  1.70451784
 ,  1.85430468 ,  1.97291208 ,  1.97017228 ,  2.25518398 ,  2.43969757
 ,  3.21870602 ,  4.18334822 ,  4.97772026 ,  6.37663164 ,  8.65075992
 ,  9.46944758 , 10.51634777 , 12.13353793 , 11.89186997 , 12.07083882
 , 13.2992811  , 14.07987878 , 14.19951571 , 14.97943562 , 16.05601334
 , 16.42979341 , 16.91576867 , 17.62775142 , 18.4885405  , 19.10609921
 , 20.03988031 , 20.86564363 , 21.73645892 , 22.6208256  , 23.37786072
 , 24.38166073 , 25.22395387 , 26.21419653 , 27.05246704 , 27.86896121
 , 28.90029708 , 29.83586775 , 30.87563699 , 31.91207845 , 33.07449767
 , 34.27919965 , 35.57195873 , 36.95045988 , 38.62308152])

if SS_stage == 'first_run_for_guesses':
    b_guess_init = np.ones((S, J)) * .01
    n_guess_init = np.ones((S, J)) * .99 * ltilde
    Kg = house.get_K(b_guess_init, omega_SS)
    Lg = firm.get_L(e, n_guess_init, omega_SS)
    Yg = firm.get_Y(Kg, Lg, parameters)
    wguess = firm.get_w(Yg, Lg, parameters)
    rguess = firm.get_r(Yg, Kg, parameters)
    avIguess = ((rguess * b_guess_init + wguess * e * n_guess_init) * omega_SS).sum()
    factor_guess = [mean_income_data / avIguess]
    guesses = list(b_guess_init.flatten()) + list(n_guess_init.flatten()) + factor_guess
    chi_guesses = np.ones(S+J)
    chi_guesses[0:J] = np.array([5, 10, 90, 250, 250, 250, 250]) + chi_b_scal
    print 'Chi_b:', chi_guesses[0:J]
    chi_guesses[J:] = chi_n_guess
    chi_guesses = list(chi_guesses)
    final_chi_params = chi_guesses
    Steady_State_SS_X2 = lambda x: Steady_State_SS(x, final_chi_params, parameters, omega_SS, rho, lambdas, theta, tau_bq, e)
    solutions = opt.fsolve(Steady_State_SS_X2, guesses, xtol=1e-13)
    print np.array(Steady_State_SS_X2(solutions)).max()
elif SS_stage == 'loop_calibration':
    variables = pickle.load(open("OUTPUT/Saved_moments/loop_calibration_solutions.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]
    guesses = list((solutions[:S*J].reshape(S, J) * scal.reshape(1, J)).flatten()) + list(
        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
    chi_guesses = np.ones(S+J)
    chi_guesses[0:J] = np.array([5, 10, 90, 250, 250, 250, 250]) + chi_b_scal
    print 'Chi_b:', chi_guesses[0:J]
    chi_guesses[J:] = chi_n_guess
    chi_guesses = list(chi_guesses)
    final_chi_params = chi_guesses
    Steady_State_SS_X2 = lambda x: Steady_State_SS(x, final_chi_params, parameters, omega_SS, rho, lambdas, theta, tau_bq, e)
    solutions = opt.fsolve(Steady_State_SS_X2, guesses, xtol=1e-13)
    print np.array(Steady_State_SS_X2(solutions)).max()
elif SS_stage == 'constrained_minimization':
    variables = pickle.load(open("OUTPUT/Saved_moments/loop_calibration_solutions.pkl", "r"))
    dictionary = {}
    for key in variables:
        globals()[key] = variables[key]
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/minimization_solutions.pkl", "w"))
    chi_guesses = final_chi_params
    function_to_minimize_X = lambda x: function_to_minimize(x, parameters, omega_SS, rho, lambdas, theta, tau_bq, e, wealth_data_array)
    bnds = tuple([(1e-6, None)] * (S + J))
    final_chi_params = opt.minimize(function_to_minimize_X, chi_guesses, method='TNC', tol=1e-7, bounds=bnds, options={'maxiter': 3}).x
    print 'The final bequest parameter values:', final_chi_params
    Steady_State_SS_X2 = lambda x: Steady_State_SS(x, final_chi_params, parameters, omega_SS, rho, lambdas, theta, tau_bq, e)
    solutions = opt.fsolve(Steady_State_SS_X2, solutions_pre, xtol=1e-13)
    print np.array(Steady_State_SS_X2(solutions)).max()
elif SS_stage == 'SS_init':
    variables = pickle.load(open("OUTPUT/Saved_moments/minimization_solutions.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]
    guesses = list((solutions[:S*J].reshape(S, J) * scal.reshape(1, J)).flatten()) + list(
        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
    chi_guesses = final_chi_params
    Steady_State_SS_X2 = lambda x: Steady_State_SS(x, final_chi_params, parameters, omega_SS, rho, lambdas, theta, tau_bq, e)
    solutions = opt.fsolve(Steady_State_SS_X2, guesses, xtol=1e-13)
    print np.array(Steady_State_SS_X2(solutions)).max()
elif SS_stage == 'SS_tax':
    variables = pickle.load(open("OUTPUT/Saved_moments/SS_init_solutions.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]
    guesses = list((solutions[:S*J].reshape(S, J) * scal.reshape(1, J)).flatten()) + list(
        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
    chi_guesses = final_chi_params
    Steady_State_SS_X2 = lambda x: Steady_State_SS(x, final_chi_params, parameters, omega_SS, rho, lambdas, theta, tau_bq, e)
    solutions = opt.fsolve(Steady_State_SS_X2, guesses, xtol=1e-13)
    print np.array(Steady_State_SS_X2(solutions)).max()

'''
------------------------------------------------------------------------
    Calculate the fits of the wealth tax
------------------------------------------------------------------------
'''

wealth_fits = misc_funcs.check_wealth_calibration(solutions[0: (S-1) * J].reshape((S-1, J)), solutions[-1], wealth_data_array, parameters)

chi_b_vals_for_fit = chi_guesses[0:J]
var_names = ['wealth_fits', 'chi_b_vals_for_fit']

dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Saved_moments/chi_b_fits.pkl", "w"))

'''
------------------------------------------------------------------------
    Save the initial values in various ways, depending on the stage of
        the simulation
------------------------------------------------------------------------
'''

if SS_stage == 'first_run_for_guesses':
    var_names = ['solutions', 'final_chi_params']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/first_run_solutions.pkl", "w"))
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/loop_calibration_solutions.pkl", "w"))
elif SS_stage == 'loop_calibration':
    var_names = ['solutions', 'final_chi_params']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/loop_calibration_solutions.pkl", "w"))
elif SS_stage == 'constrained_minimization':
    var_names = ['solutions', 'final_chi_params']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/minimization_solutions.pkl", "w"))
elif SS_stage == 'SS_init':
    var_names = ['solutions', 'final_chi_params']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/SS_init_solutions.pkl", "w"))
elif SS_stage == 'SS_tax':
    var_names = ['solutions', 'final_chi_params']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/SS_tax_solutions.pkl", "w"))


if SS_stage != 'first_run_for_guesses' and SS_stage != 'loop_calibration':
    bssmat = solutions[0:(S-1) * J].reshape(S-1, J)
    bq = solutions[(S-1)*J:S*J]
    bssmat_s = np.array(list(np.zeros(J).reshape(1, J)) + list(bssmat))
    bssmat_splus1 = np.array(list(bssmat) + list(bq.reshape(1, J)))
    Kss = house.get_K(bssmat_splus1, omega_SS)
    nssmat = solutions[S * J:-1].reshape(S, J)
    Lss = firm.get_L(e, nssmat, omega_SS)
    Yss = firm.get_Y(Kss, Lss, parameters)
    wss = firm.get_w(Yss, Lss, parameters)
    rss = firm.get_r(Yss, Kss, parameters)
    BQss = (1+rss)*(np.array(list(bssmat) + list(bq.reshape(1, J))).reshape(
        S, J) * omega_SS * rho.reshape(S, 1)).sum(0)
    b_s = np.array(list(np.zeros(J).reshape((1, J))) + list(bssmat))
    factor_ss = solutions[-1]
    T_Hss = tax.get_lump_sum(rss, b_s, wss, e, nssmat, BQss, lambdas, factor_ss, omega_SS, 'SS', parameters, theta, tau_bq)
    taxss = tax.total_taxes(rss, b_s, wss, e, nssmat, BQss, lambdas, factor_ss, T_Hss, None, 'SS', False, parameters, theta, tau_bq)
    cssmat = house.get_cons(rss, b_s, wss, e, nssmat, BQss.reshape(1, J), lambdas.reshape(1, J), bssmat_splus1, parameters, taxss)

    house.constraint_checker_SS(bssmat, nssmat, cssmat, parameters)

    '''
    ------------------------------------------------------------------------
    Generate variables for graphs
    ------------------------------------------------------------------------
    b_s        = SxJ array of bssmat in period t
    b_splus1        = SxJ array of bssmat in period t+1
    b_splus2        = SxJ array of bssmat in period t+2
    euler_savings_1      = euler errors from first euler equation
    euler_labor_leisure      = euler errors from second euler equation
    euler_savings_2      = euler errors from third euler equation
    ------------------------------------------------------------------------
    '''
    b_s = np.array(list(np.zeros(J).reshape((1, J))) + list(bssmat))
    b_splus1 = bssmat_splus1
    b_splus2 = np.array(list(bssmat_splus1[1:]) + list(np.zeros(J).reshape((1, J))))

    chi_b = np.tile(final_chi_params[:J].reshape(1, J), (S, 1))
    chi_n = np.array(final_chi_params[J:])
    euler_savings = house.euler_savings_func(wss, rss, e, nssmat, b_s, b_splus1, b_splus2, BQss.reshape(1, J), factor_ss, T_Hss, chi_b, parameters, theta, tau_bq, rho, lambdas)
    euler_labor_leisure = house.euler_labor_leisure_func(wss, rss, e, nssmat, b_s, b_splus1, BQss.reshape(1, J), factor_ss, T_Hss, chi_n, parameters, theta, tau_bq, lambdas)
'''
------------------------------------------------------------------------
    Save the values in various ways, depending on the stage of
        the simulation, to be used in TPI or graphing functions
------------------------------------------------------------------------
'''
if SS_stage == 'constrained_minimization':
    bssmat_init = bssmat_splus1
    nssmat_init = nssmat
    var_names = ['retire', 'nssmat_init', 'wss', 'factor_ss', 'e',
                 'J', 'omega_SS']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/Saved_moments/payroll_inputs.pkl", "w"))
elif SS_stage == 'SS_init':
    bssmat_init = bssmat_splus1
    nssmat_init = nssmat
    # Pickle variables for TPI initial values
    var_names = ['bssmat_init', 'nssmat_init']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/SSinit/ss_init_tpi_vars.pkl", "w"))
    # Pickle variables
    var_names = ['Kss', 'bssmat', 'Lss', 'nssmat', 'Yss', 'wss', 'rss',
                 'BQss', 'factor_ss', 'bssmat_s', 'cssmat', 'bssmat_splus1',
                 'T_Hss', 'euler_savings', 'euler_labor_leisure', 'chi_n', 'chi_b']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/SSinit/ss_init_vars.pkl", "w"))
elif SS_stage == 'SS_tax':
    # Pickle Variables
    var_names = ['Kss', 'bssmat', 'Lss', 'nssmat', 'Yss', 'wss', 'rss',
                 'chi_n', 'chi_b', 'BQss', 'factor_ss', 'bssmat_s', 'cssmat',
                 'euler_savings', 'euler_labor_leisure', 'T_Hss', 'bssmat_splus1']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/SS/ss_vars.pkl", "w"))
