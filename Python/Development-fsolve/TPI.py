'''
------------------------------------------------------------------------
Last updated 6/19/2015

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method, where labor in inelastically supplied.

This py-file calls the following other file(s):
            tax_funcs.py
            misc_funcs.py
            household_funcs.py
            firm_funcs.py
            OUTPUT/SSinit/ss_init_vars.pkl
            OUTPUT/SS/ss_vars.pkl
            OUTPUT/SSinit/ss_init_tpi.pkl
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

import tax_funcs as tax
import misc_funcs
import household_funcs as house
import firm_funcs as firm

'''
------------------------------------------------------------------------
Import steady state distribution, parameters and other objects from
steady state computation in ss_vars.pkl
------------------------------------------------------------------------
S        = number of periods an individual lives
beta     = discount factor
sigma    = coefficient of relative risk aversion
alpha    = capital share of income
nu       = contraction parameter in steady state iteration process
           representing the weight on the new distribution gamma_new
Z        = total factor productivity parameter in firms' production
           function
delta    = decreciation rate of capital
ltilde   = measure of time each individual is endowed with each period
eta      = Frisch elasticity of labor supply
e        = S x J matrix of age dependent possible working abilities e_s
J        = number of points in the support of e
Kss      = steady state aggregate capital stock: scalar
nssvec   = ((S-1) x 1) vector of the steady state level of capital
           (averaged across ability types)
bssmat   = ((S-1) x J) array of the steady state distribution of
           capital
Lss      = steady state aggregate labor: scalar
nssvec   = (S x 1) vector of the steady state level of labor
           (averaged across ability types)
nssmat   = (S x J) array of the steady state distribution of labor
Yss      = steady state aggregate output: scalar
wss      = steady state real wage: scalar
rss      = steady state real rental rate: scalar
K_agg    = Aggregate level of capital: scalar
T        = number of periods until the steady state
maxiter   = Maximum number of iterations that TPI will undergo
mindist_TPI   = Cut-off distance between iterations for TPI
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
chi_b        = discount factor of incidental bequests
get_baseline = whether this is the baseline TPI or not
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/Saved_moments/params_given.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
if get_baseline is False:
    variables = pickle.load(open("OUTPUT/Saved_moments/params_changed.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]

if get_baseline:
    variables = pickle.load(open("OUTPUT/SSinit/ss_init_vars.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]
else:
    variables = pickle.load(open("OUTPUT/SS/ss_vars.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]
    variables = pickle.load(open("OUTPUT/SSinit/ss_init_tpi_vars.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]

'''
------------------------------------------------------------------------
Set other parameters and initial values
------------------------------------------------------------------------
'''

# Make a vector of all one dimensional parameters, to be used in the following functions
income_tax_params = [a_tax_income, b_tax_income, c_tax_income, d_tax_income]
wealth_tax_params = [h_wealth, p_wealth, m_wealth]
ellipse_params = [b_ellipse, upsilon]
parameters = [J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data] + income_tax_params + wealth_tax_params + ellipse_params

N_tilde = omega.sum(1).sum(1)
omega_stationary = omega / N_tilde.reshape(T+S, 1, 1)

if get_baseline:
    initial_b = bssmat_splus1
    initial_n = nssmat
else:
    initial_b = bssmat_init
    initial_n = nssmat_init
K0 = house.get_K(initial_b, omega_stationary[0])
b_sinit = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_b[:-1]))
b_splus1init = initial_b
L0 = firm.get_L(e, initial_n, omega_stationary[0])
Y0 = firm.get_Y(K0, L0, parameters)
w0 = firm.get_w(Y0, L0, parameters)
r0 = firm.get_r(Y0, K0, parameters)
BQ0 = (1+r0)*(initial_b * omega_stationary[0] * rho.reshape(S, 1)).sum(0)
T_H_0 = tax.get_lump_sum(r0, b_sinit, w0, e, initial_n, BQ0, lambdas, factor_ss, omega_stationary[0], 'SS', parameters, theta, tau_bq)
tax0 = tax.total_taxes(r0, b_sinit, w0, e, initial_n, BQ0, lambdas, factor_ss, T_H_0, None, 'SS', False, parameters, theta, tau_bq)
c0 = house.get_cons(r0, b_sinit, w0, e, initial_n, BQ0.reshape(1, J), lambdas.reshape(1, J), b_splus1init, parameters, tax0)

'''
------------------------------------------------------------------------
Solve for equilibrium transition path by TPI
------------------------------------------------------------------------
'''


def SS_TPI_firstdoughnutring(guesses, winit, rinit, BQinit, T_H_init):
    # This function does not work.  The tax functions need to be changed.
    b2 = float(guesses[0])
    n1 = float(guesses[1])
    b1 = float(initial_b[-2, j])
    # Euler 1 equations
    tax11 = tax.total_taxes(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], factor_ss, T_H_init, j, 'TPI', False, parameters, theta, tau_bq)
    cons11 = house.get_cons(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], b2, parameters, tax11)
    bequest_ut = rho * np.exp(-sigma * g_y) * chi_b[-1, j] * b2 ** (-sigma)
    error1 = house.marg_ut_cons(cons11, parameters) - bequest_ut
    # Euler 2 equations
    tax2 = tax.total_taxes(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], factor_ss, T_H_init, j, 'TPI', True, parameters, theta, tau_bq)
    cons2 = house.get_cons(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], b2, parameters, tax2)
    income2 = (rinit * b1 + winit * e[-1, j] * n1) * factor_ss
    deriv2 = 1 - tau_payroll - tax.tau_income(rinit, b1, winit, e[
        -1, j], n1, factor_ss, parameters) - tax.tau_income_deriv(
        rinit, b1, winit, e[-1, j], n1, factor_ss, parameters) * income2
    error2 = house.marg_ut_cons(cons2, parameters) * winit * e[-1, j] * deriv2 - house.marg_ut_labor(n1, chi_n[-1], parameters)
    if n1 <= 0:
        error2 += 1e12
    return [error1] + [error2]


def Steady_state_TPI_solver(guesses, winit, rinit, BQinit, T_H_init, factor, j, s, t, params, theta, tau_bq, rho, lambdas, e, initial_b, chi_b, chi_n):
    '''
    Parameters:
        guesses = distribution of capital and labor in period t
                  ((S-1)*S*J x 1 list)
        winit   = wage rate (scalar)
        rinit   = rental rate (scalar)
        t       = time period

    Returns:
        Value of Euler error. (as an 2*S*J x 1 list)
    '''

    J, S, T, beta, sigma, alpha, Z, delta, ltilde, nu, g_y, tau_payroll, retire, mean_income_data, a_tax_income, b_tax_income, c_tax_income, d_tax_income, h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params
    length = len(guesses)/2
    b_guess = np.array(guesses[:length])
    n_guess = np.array(guesses[length:])

    if length == S:
        b_s = np.array([0] + list(b_guess[:-1]))
    else:
        b_s = np.array([(initial_b[-(s+3), j])] + list(b_guess[:-1]))
    b_splus1 = b_guess
    b_splus2 = np.array(list(b_guess[1:]) + [0])
    w_s = winit[t:t+length]
    w_splus1 = winit[t+1:t+length+1]
    r_s = rinit[t:t+length]
    r_splus1 = rinit[t+1:t+length+1]
    n_s = n_guess
    n_extended = np.array(list(n_guess[1:]) + [0])
    e_s = e[-length:, j]
    e_extended = np.array(list(e[-length+1:, j]) + [0])
    BQ_s = BQinit[t:t+length]
    BQ_splus1 = BQinit[t+1:t+length+1]
    T_H_s = T_H_init[t:t+length]
    T_H_splus1 = T_H_init[t+1:t+length+1]
    # Savings euler equations
    tax_s = tax.total_taxes(r_s, b_s, w_s, e_s, n_s, BQ_s, lambdas[j], factor, T_H_s, j, 'TPI', False, params, theta, tau_bq)
    tax_splus1 = tax.total_taxes(r_splus1, b_splus1, w_splus1, e_extended, n_extended, BQ_splus1, lambdas[j], factor, T_H_splus1, j, 'TPI', True, params, theta, tau_bq)
    cons_s = house.get_cons(r_s, b_s, w_s, e_s, n_s, BQ_s, lambdas[j], b_splus1, params, tax_s)
    cons_splus1 = house.get_cons(r_splus1, b_splus1, w_splus1, e_extended, n_extended, BQ_splus1, lambdas[j], b_splus2, params, tax_splus1)
    income_splus1 = (r_splus1 * b_splus1 + w_splus1 * e_extended * n_extended) * factor
    savings_ut = rho[-(length):] * np.exp(-sigma * g_y) * chi_b[-(length):, j] * b_splus1 ** (-sigma)
    deriv_savings = 1 + r_splus1 * (1 - tax.tau_income(
        r_splus1, b_splus1, w_splus1, e_extended, n_extended, factor, params) - tax.tau_income_deriv(
        r_splus1, b_splus1, w_splus1, e_extended, n_extended, factor, params) * income_splus1) - tax.tau_w_prime(
        b_splus1, params)*b_splus1 - tax.tau_wealth(b_splus1, params)
    error1 = house.marg_ut_cons(cons_s, params) - beta * (1-rho[-(length):]) * np.exp(-sigma * g_y) * deriv_savings * house.marg_ut_cons(
        cons_splus1, params) - savings_ut
    # Labor leisure euler equations
    income_s = (r_s * b_s + w_s * e_s * n_s) * factor
    deriv_laborleisure = 1 - tau_payroll - tax.tau_income(r_s, b_s, w_s, e_s, n_s, factor, params) - tax.tau_income_deriv(
        r_s, b_s, w_s, e_s, n_s, factor, params) * income_s
    error2 = house.marg_ut_cons(cons_s, params) * w_s * e[-(length):, j] * deriv_laborleisure - house.marg_ut_labor(n_s, chi_n[-length:], params)
    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] += 1e12
    mask2 = n_guess > ltilde
    error2[mask2] += 1e12
    mask3 = cons_s < 0
    error2[mask3] += 1e12
    mask4 = b_guess <= 0
    error2[mask4] += 1e12
    return list(error1.flatten()) + list(error2.flatten())

# Initialize Time paths
domain = np.linspace(0, T, T)
Kinit = (-1/(domain + 1)) * (Kss-K0) + Kss
Kinit[-1] = Kss
Kinit = np.array(list(Kinit) + list(np.ones(S)*Kss))
Linit = np.ones(T+S) * Lss
Yinit = firm.get_Y(Kinit, Linit, parameters)
winit = firm.get_w(Yinit, Linit, parameters)
rinit = firm.get_r(Yinit, Kinit, parameters)
BQinit = np.zeros((T+S, J))
for j in xrange(J):
    BQinit[:, j] = list(np.linspace(BQ0[j], BQss[j], T)) + [BQss[j]]*S
BQinit = np.array(BQinit)
T_H_init = np.ones(T+S) * T_Hss

# Make array of initial guesses
domain2 = np.tile(domain.reshape(T, 1, 1), (1, S, J))
ending_b = bssmat_splus1
guesses_b = (-1/(domain2 + 1)) * (ending_b-initial_b) + ending_b
ending_b_tail = np.tile(ending_b.reshape(1, S, J), (S, 1, 1))
guesses_b = np.append(guesses_b, ending_b_tail, axis=0)

domain3 = np.tile(np.linspace(0, 1, T).reshape(T, 1, 1), (1, S, J))
guesses_n = domain3 * (nssmat - initial_n) + initial_n
ending_n_tail = np.tile(nssmat.reshape(1, S, J), (S, 1, 1))
guesses_n = np.append(guesses_n, ending_n_tail, axis=0)

TPIiter = 0
TPIdist = 10

euler_errors = np.zeros((T, 2*S, J))
TPIdist_vec = np.zeros(maxiter)

while (TPIiter < maxiter) and (TPIdist >= mindist_TPI):
    b_mat = np.zeros((T+S, S, J))
    n_mat = np.zeros((T+S, S, J))
    Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)
    Lpath_TPI = list(Linit) + list(np.ones(10)*Lss)
    # Plot TPI for K for each iteration, so we can see if there is a problem
    plt.figure()
    plt.axhline(
        y=Kss, color='black', linewidth=2, label=r"Steady State $\hat{K}$", ls='--')
    plt.plot(np.arange(
        T+10), Kpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path $\hat{K}_t$")
    plt.savefig("OUTPUT/TPI_K")
    for j in xrange(J):
        # b_mat[1, -1, j], n_mat[0, -1, j] = np.array(opt.fsolve(SS_TPI_firstdoughnutring, [guesses_b[1, -2, j], guesses_n[0, -2, j]],
        #     args=(winit[1], rinit[1], BQinit[1, j], T_H_init[1])))
        for s in xrange(S-2):  # Upper triangle
            b_guesses_to_use = np.diag(guesses_b[1:S+1, :, j], S-(s+2))
            n_guesses_to_use = np.diag(guesses_n[:S, :, j], S-(s+2))
            solutions = opt.fsolve(Steady_state_TPI_solver, list(
                b_guesses_to_use) + list(n_guesses_to_use), args=(
                winit, rinit, BQinit[:, j], T_H_init, factor_ss, j, s, 0, parameters, theta, tau_bq, rho, lambdas, e, initial_b, chi_b, chi_n), xtol=1e-13)
            b_vec = solutions[:len(solutions)/2]
            b_mat[1:S+1, :, j] += np.diag(b_vec, S-(s+2))
            n_vec = solutions[len(solutions)/2:]
            n_mat[:S, :, j] += np.diag(n_vec, S-(s+2))

        for t in xrange(0, T):
            b_guesses_to_use = np.diag(guesses_b[t+1:t+S+1, :, j])
            n_guesses_to_use = np.diag(guesses_n[t:t+S, :, j])
            solutions = opt.fsolve(Steady_state_TPI_solver, list(
                b_guesses_to_use) + list(n_guesses_to_use), args=(
                winit, rinit, BQinit[:, j], T_H_init, factor_ss, j, None, t, parameters, theta, tau_bq, rho, lambdas, e, None, chi_b, chi_n), xtol=1e-13)
            b_vec = solutions[:S]
            b_mat[t+1:t+S+1, :, j] += np.diag(b_vec)
            n_vec = solutions[S:]
            n_mat[t:t+S, :, j] += np.diag(n_vec)
            inputs = list(solutions)
            euler_errors[t, :, j] = np.abs(Steady_state_TPI_solver(
                inputs, winit, rinit, BQinit[:, j], T_H_init, factor_ss, j, None, t, parameters, theta, tau_bq, rho, lambdas, e, None, chi_b, chi_n))
    
    b_mat[0, :, :] = initial_b
    b_mat[1, -1, :]= b_mat[1, -2, :]
    n_mat[0, -1, :] = n_mat[0, -2, :]
    Kinit = (omega_stationary[:T, :, :] * b_mat[:T, :, :]).sum(2).sum(1)
    Linit = (omega_stationary[:T, :, :] * e.reshape(
        1, S, J) * n_mat[:T, :, :]).sum(2).sum(1)

    Ynew = firm.get_Y(Kinit, Linit, parameters)
    wnew = firm.get_w(Ynew, Linit, parameters)
    rnew = firm.get_r(Ynew, Kinit, parameters)

    BQnew = (1+rnew.reshape(T, 1))*(b_mat[:T] * omega_stationary[:T] * rho.reshape(1, S, 1)).sum(1)
    bmat_s = np.zeros((T, S, J))
    bmat_s[:, 1:, :] = b_mat[:T, :-1, :]
    T_H_new = np.array(list(tax.get_lump_sum(rnew.reshape(T, 1, 1), bmat_s, wnew.reshape(
        T, 1, 1), e.reshape(1, S, J), n_mat[:T], BQnew.reshape(T, 1, J), lambdas.reshape(
        1, 1, J), factor_ss, omega_stationary[:T], 'TPI', parameters, theta, tau_bq)) + [T_Hss]*S)

    winit[:T] = misc_funcs.convex_combo(wnew, winit[:T], parameters)
    rinit[:T] = misc_funcs.convex_combo(rnew, rinit[:T], parameters)
    BQinit[:T] = misc_funcs.convex_combo(BQnew, BQinit[:T], parameters)
    T_H_init[:T] = misc_funcs.convex_combo(T_H_new[:T], T_H_init[:T], parameters)
    guesses_b = misc_funcs.convex_combo(b_mat, guesses_b, parameters)
    guesses_n = misc_funcs.convex_combo(n_mat, guesses_n, parameters)

    TPIdist = np.array(list(misc_funcs.perc_dif_func(rnew, rinit[:T]))+list(misc_funcs.perc_dif_func(BQnew, BQinit[:T]).flatten())+list(
        misc_funcs.perc_dif_func(wnew, winit[:T]))+list(misc_funcs.perc_dif_func(T_H_new, T_H_init))).max()
    TPIdist_vec[TPIiter] = TPIdist
    # After T=10, if cycling occurs, drop the value of nu
    # wait til after T=10 or so, because sometimes there is a jump up
    # in the first couple iterations
    if TPIiter > 10:
        if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter-1] > 0:
            nu /= 2
            print 'New Value of nu:', nu
    TPIiter += 1
    print '\tIteration:', TPIiter
    print '\t\tDistance:', TPIdist

print 'Computing final solutions'

b_mat = np.zeros((T+S, S, J))
n_mat = np.zeros((T+S, S, J))

for j in xrange(J):
    for s in xrange(S-2):  # Upper triangle
        b_guesses_to_use = np.diag(guesses_b[1:S+1, :, j], S-(s+2))
        n_guesses_to_use = np.diag(guesses_n[:S, :, j], S-(s+2))
        solutions = opt.fsolve(Steady_state_TPI_solver, list(
            b_guesses_to_use) + list(n_guesses_to_use), args=(
            winit, rinit, BQinit[:, j], T_H_init, factor_ss, j, s, 0, parameters, theta, tau_bq, rho, lambdas, e, initial_b, chi_b, chi_n), xtol=1e-13)
        b_vec = solutions[:len(solutions)/2]
        b_mat[1:S+1, :, j] += np.diag(b_vec, S-(s+2))
        n_vec = solutions[len(solutions)/2:]
        n_mat[:S, :, j] += np.diag(n_vec, S-(s+2))

    for t in xrange(0, T):
        b_guesses_to_use = np.diag(guesses_b[t+1:t+S+1, :, j])
        n_guesses_to_use = np.diag(guesses_n[t:t+S, :, j])
        solutions = opt.fsolve(Steady_state_TPI_solver, list(
            b_guesses_to_use) + list(n_guesses_to_use), args=(
            winit, rinit, BQinit[:, j], T_H_init, factor_ss, j, None, t, parameters, theta, tau_bq, rho, lambdas, e, None, chi_b, chi_n), xtol=1e-13)
        b_vec = solutions[:S]
        b_mat[t+1:t+S+1, :, j] += np.diag(b_vec)
        n_vec = solutions[S:]
        n_mat[t:t+S, :, j] += np.diag(n_vec)
        inputs = list(solutions)
        euler_errors[t, :, j] = np.abs(Steady_state_TPI_solver(
            inputs, winit, rinit, BQinit[:, j], T_H_init, factor_ss, j, None, t, parameters, theta, tau_bq, rho, lambdas, e, None, chi_b, chi_n))
    # b_mat[1, -1, j], n_mat[0, -1, j] = np.array(opt.fsolve(SS_TPI_firstdoughnutring, [b_mat[1, -2, j], n_mat[0, -2, j]],
    #     args=(winit[1], rinit[1], BQinit[1, j], T_H_init[1])))

b_mat[0, :, :] = initial_b
b_mat[1, -1, :]= b_mat[1, -2, :]
n_mat[0, -1, :] = n_mat[0, -2, :]

Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)
Lpath_TPI = list(Linit) + list(np.ones(10)*Lss)
BQpath_TPI = np.array(list(BQinit) + list(np.ones((10, J))*BQss))


b_s = np.zeros((T, S, J))
b_s[:, 1:, :] = b_mat[:T, :-1, :]
b_splus1 = np.zeros((T, S, J))
b_splus1[:, :, :] = b_mat[1:T+1, :, :]

tax_path = tax.total_taxes(rinit[:T].reshape(T, 1, 1), b_s, winit[:T].reshape(T, 1, 1), e.reshape(
    1, S, J), n_mat[:T], BQinit[:T, :].reshape(T, 1, J), lambdas, factor_ss, T_H_init[:T].reshape(T, 1, 1), None, 'TPI', False, parameters, theta, tau_bq)
cinit = house.get_cons(rinit[:T].reshape(T, 1, 1), b_s, winit[:T].reshape(T, 1, 1), e.reshape(1, S, J), n_mat[:T], BQinit[:T].reshape(T, 1, J), lambdas.reshape(1, 1, J), b_splus1, parameters, tax_path)

print'Checking time path for violations of constaints.'
for t in xrange(T):
    house.constraint_checker_TPI(b_mat[t, :-1, :], n_mat[
        t], cinit[t], t, parameters, N_tilde)

'''
------------------------------------------------------------------------
Generate values for TPI graphs
------------------------------------------------------------------------
'''
eul_savings = euler_errors[:, :S, :].max(1).max(1)
eul_laborleisure = euler_errors[:, S:, :].max(1).max(1)

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

var_names = ['Kpath_TPI', 'b_mat', 'cinit',
             'eul_savings', 'eul_laborleisure', 'Lpath_TPI', 'BQpath_TPI',
             'n_mat', 'rinit', 'winit', 'Yinit', 'T_H_init', 'tax_path']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]

if get_baseline:
    pickle.dump(dictionary, open("OUTPUT/TPIinit/TPIinit_vars.pkl", "w"))
else:
    pickle.dump(dictionary, open("OUTPUT/TPI/TPI_vars.pkl", "w"))
