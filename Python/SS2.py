'''
------------------------------------------------------------------------
Last updated: 5/21/2015

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
import time
import os
import scipy.optimize as opt
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import income
import demographics
import tax_funcs as tax


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

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments_fit_25.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
top25 = highest_wealth_data_new
top25[2:26] = 500.0

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments_fit_50.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
top50 = highest_wealth_data_new

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments_fit_70.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
top70 = highest_wealth_data_new

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments_fit_80.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
top80 = highest_wealth_data_new

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments_fit_90.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
top90 = highest_wealth_data_new

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments_fit_99.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
top99 = highest_wealth_data_new

variables = pickle.load(open("OUTPUT/Saved_moments/wealth_data_moments_fit_100.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
top100 = highest_wealth_data_new



variables = pickle.load(open("OUTPUT/Saved_moments/labor_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
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

# These values never change, so only run it once
omega, g_n, omega_SS, surv_rate = demographics.get_omega(
        S, J, T, lambdas, starting_age, ending_age, E)
e = income.get_e(S, J, starting_age, ending_age, lambdas, omega_SS)
rho = 1-surv_rate
var_names = ['omega', 'g_n', 'omega_SS', 'surv_rate', 'e', 'rho']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/income_demo_vars.pkl", "w"))


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


rho[-1] = 1


'''
------------------------------------------------------------------------
    Define Functions
------------------------------------------------------------------------
'''

# Functions and Definitions


def get_Y(K_now, L_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = Z * (K_now ** alpha) * ((L_now) ** (1 - alpha))
    print('Z')
    print(Z)
    return Y_now


def get_w(Y_now, L_now):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    w_now = (1 - alpha) * Y_now / L_now
    return w_now


def get_r(Y_now, K_now):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_L(e, n):
    '''
    Parameters: e, n

    Returns:    Aggregate labor
    '''
    L_now = np.sum(e * omega_SS * n)
    return L_now


def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''
    output = c**(-sigma)
    return output


def MUl(n, chi_n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    deriv = b_ellipse * (1/ltilde) * ((1 - (n / ltilde) ** upsilon) ** (
        (1/upsilon)-1)) * (n / ltilde) ** (upsilon - 1)
    output = chi_n.reshape(S, 1) * deriv
    return output


def MUb(chi_b, bequest):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b[-1, :].reshape(1, J) * (bequest ** (-sigma))
    return output


def get_cons(r, b1, w, e, n, BQ, lambdas, b2, g_y, net_tax):
    '''
    Parameters: rental rate, capital stock (t-1), wage, e, labor stock,
                bequests, lambdas, capital stock (t), growth rate y, taxes

    Returns:    Consumption
    '''
    cons = (1 + r)*b1 + w*e*n + BQ / lambdas - b2*np.exp(g_y) - net_tax
    return cons


def Euler1(w, r, e, n_guess, b1, b2, b3, BQ, factor, T_H, chi_b):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b1       = distribution of capital in period t ((S-1) x J array)
        b2       = distribution of capital in period t+1 ((S-1) x J array)
        b3       = distribution of capital in period t+2 ((S-1) x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        T_H  = lump sum transfer from the government to the households
        xi       = coefficient of relative risk aversion
        chi_b    = discount factor of savings

    Returns:
        Value of Euler error.
    '''
    BQ_euler = BQ.reshape(1, J)
    tax1 = tax.total_taxes_SS(r, b1, w, e[:-1, :], n_guess[:-1, :], BQ_euler, lambdas, factor, T_H)
    tax2 = tax.total_taxes_SS2(r, b2, w, e[1:, :], n_guess[1:, :], BQ_euler, lambdas, factor, T_H)
    cons1 = get_cons(r, b1, w, e[:-1, :], n_guess[:-1, :], BQ_euler, lambdas, b2, g_y, tax1)
    cons2 = get_cons(r, b2, w, e[1:, :], n_guess[1:, :], BQ_euler, lambdas, b3, g_y, tax2)
    income = (r * b2 + w * e[1:, :] * n_guess[1:, :]) * factor
    deriv = (
        1 + r*(1-tax.tau_income(r, b1, w, e[1:, :], n_guess[1:, :], factor)-tax.tau_income_deriv(
            r, b1, w, e[1:, :], n_guess[1:, :], factor)*income)-tax.tau_w_prime(b2)*b2-tax.tau_wealth(b2))
    bequest_ut = rho[:-1].reshape(S-1, 1) * np.exp(-sigma * g_y) * chi_b[:-1].reshape(S-1, J) * b2 ** (-sigma)
    euler = MUc(cons1) - beta * (1-rho[:-1].reshape(S-1, 1)) * deriv * MUc(
        cons2) * np.exp(-sigma * g_y) - bequest_ut
    return euler


def Euler2(w, r, e, n_guess, b1_2, b2_2, BQ, factor, T_H, chi_n):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b1_2     = distribution of capital in period t (S x J array)
        b2_2     = distribution of capital in period t+1 (S x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        T_H  = lump sum transfer from the government to the households

    Returns:
        Value of Euler error.
    '''
    BQ = BQ.reshape(1, J)
    tax1 = tax.total_taxes_SS(r, b1_2, w, e, n_guess, BQ, lambdas, factor, T_H)
    cons = get_cons(r, b1_2, w, e, n_guess, BQ, lambdas, b2_2, g_y, tax1)
    income = (r * b1_2 + w * e * n_guess) * factor
    deriv = 1 - tau_payroll - tax.tau_income(r, b1_2, w, e, n_guess, factor) - tax.tau_income_deriv(
        r, b1_2, w, e, n_guess, factor) * income
    euler = MUc(cons) * w * deriv * e - MUl(n_guess, chi_n)
    return euler


def Euler3(w, r, e, n_guess, b_guess, BQ, factor, chi_b, T_H):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b_guess  = distribution of capital in period t (S-1 x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        chi_b    = discount factor of savings
        T_H  = lump sum transfer from the government to the households

    Returns:
        Value of Euler error.
    '''
    BQ = BQ.reshape(1, J)
    tax1 = tax.total_taxes_eul3_SS(r, b_guess[-2, :], w, e[-1, :], n_guess[-1, :], BQ, lambdas, factor, T_H)
    cons = get_cons(r, b_guess[-2, :], w, e[-1, :], n_guess[-1, :], BQ, lambdas, b_guess[-1, :], g_y, tax1)
    euler = MUc(cons) - np.exp(-sigma * g_y) * MUb(
        chi_b, b_guess[-1, :])
    return euler


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data and
        simulated data
    '''
    frac = (simul - data)/data
    output = np.abs(frac)
    return output


def Steady_State(guesses, params):
    '''
    Parameters: Steady state distribution of capital guess as array
                size 2*S*J

    Returns:    Array of 2*S*J Euler equation errors
    '''
    chi_b = np.tile(np.array(params[:J]).reshape(1, J), (S, 1))
    chi_n = np.array(params[J:])
    b_guess = guesses[0: S * J].reshape((S, J))
    B = (b_guess * omega_SS * rho.reshape(S, 1)).sum(0)
    K = (omega_SS * b_guess).sum()
    n_guess = guesses[S * J:-1].reshape((S, J))
    L = get_L(e, n_guess)
    Y = get_Y(K, L)
    w = get_w(Y, L)
    r = get_r(Y, K)
    BQ = (1 + r) * B
    b1 = np.array(list(np.zeros(J).reshape(1, J)) + list(b_guess[:-2, :]))
    b2 = b_guess[:-1, :]
    b3 = b_guess[1:, :]
    b1_2 = np.array(list(np.zeros(J).reshape(1, J)) + list(b_guess[:-1, :]))
    b2_2 = b_guess
    factor = guesses[-1]
    T_H = tax.tax_lump(r, b1_2, w, e, n_guess, BQ, lambdas, factor, omega_SS)
    error1 = Euler1(w, r, e, n_guess, b1, b2, b3, BQ, factor, T_H, chi_b)
    error2 = Euler2(w, r, e, n_guess, b1_2, b2_2, BQ, factor, T_H, chi_n)
    error3 = Euler3(w, r, e, n_guess, b_guess, BQ, factor, chi_b, T_H)
    average_income_model = ((r * b1_2 + w * e * n_guess) * omega_SS).sum()
    error4 = [mean_income_data - factor * average_income_model]
    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] += 1e9
    mask2 = n_guess > ltilde
    error2[mask2] += 1e9
    if b_guess.sum() <= 0:
        error1 += 1e9
    tax1 = tax.total_taxes_SS(r, b1_2, w, e, n_guess, BQ, lambdas, factor, T_H)
    cons = get_cons(r, b1_2, w, e, n_guess, BQ.reshape(1, J), lambdas, b2_2, g_y, tax1)
    mask3 = cons < 0
    error2[mask3] += 1e9
    mask4 = b_guess[:-1] <= 0
    error1[mask4] += 1e9
    print np.abs(np.array(list(error1.flatten()) + list(
         error2.flatten()) + list(error3.flatten()) + error4)).max()
    return list(error1.flatten()) + list(
        error2.flatten()) + list(error3.flatten()) + error4


def constraint_checker(bssmat, nssmat, cssmat):
    '''
    Parameters:
        bssmat = steady state distribution of capital ((S-1)xJ array)
        nssmat = steady state distribution of labor (SxJ array)
        wss    = steady state wage rate (scalar)
        rss    = steady state rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        cssmat = steady state distribution of consumption (SxJ array)
        BQ     = bequests

    Created Variables:
        flag1 = False if all borrowing constraints are met, true
               otherwise.
        flag2 = False if all labor constraints are met, true otherwise

    Returns:
        # Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    # print 'Checking constraints on capital, labor, and consumption.'
    flag1 = False
    if bssmat.sum() <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to zero.'
        flag1 = True
    if flag1 is False:
        print '\tThere were no violations of the borrowing constraints.'
    flag2 = False
    if (nssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (nssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Consumption volates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'


def func_to_min(chi_b_guesses_init, other_guesses_init):
    '''
    Parameters:
        chi_b_guesses_init = guesses for chi_b
        other_guesses_init = guesses for the distribution of capital and labor
                            stock, and factor value

    Returns:
        The max absolute deviation between the actual and simulated
            wealth moments
    '''
    print chi_b_guesses_init
    Steady_State_X = lambda x: Steady_State(x, chi_b_guesses_init)

    variables = pickle.load(open("OUTPUT/Saved_moments/minimization_solutions.pkl", "r"))
    for key in variables:
        globals()[key+'_pre'] = variables[key]
    solutions = opt.fsolve(Steady_State_X, solutions_pre, xtol=1e-13)
    b_guess = solutions[0: S * J].reshape((S, J))
    b2 = b_guess[:-1, :]
    factor = solutions[-1]
    # Wealth Calibration Euler
    p25_sim = b2[:, 0] * factor
    p50_sim = b2[:, 1] * factor
    p70_sim = b2[:, 2] * factor
    p80_sim = b2[:, 3] * factor
    p90_sim = b2[:, 4] * factor
    p99_sim = b2[:, 5] * factor
    p100_sim = b2[:, 6] * factor
    b_perc_diff_25 = [perc_dif_func(np.mean(p25_sim[:24]), np.mean(top25[2:26]))] + [perc_dif_func(np.mean(p25_sim[24:45]), np.mean(top25[26:47]))]
    b_perc_diff_50 = [perc_dif_func(np.mean(p50_sim[:24]), np.mean(top50[2:26]))] + [perc_dif_func(np.mean(p50_sim[24:45]), np.mean(top50[26:47]))]
    b_perc_diff_70 = [perc_dif_func(np.mean(p70_sim[:24]), np.mean(top70[2:26]))] + [perc_dif_func(np.mean(p70_sim[24:45]), np.mean(top70[26:47]))]
    b_perc_diff_80 = [perc_dif_func(np.mean(p80_sim[:24]), np.mean(top80[2:26]))] + [perc_dif_func(np.mean(p80_sim[24:45]), np.mean(top80[26:47]))]
    b_perc_diff_90 = [perc_dif_func(np.mean(p90_sim[:24]), np.mean(top90[2:26]))] + [perc_dif_func(np.mean(p90_sim[24:45]), np.mean(top90[26:47]))]
    b_perc_diff_99 = [perc_dif_func(np.mean(p99_sim[:24]), np.mean(top99[2:26]))] + [perc_dif_func(np.mean(p99_sim[24:45]), np.mean(top99[26:47]))]
    b_perc_diff_100 = [perc_dif_func(np.mean(p100_sim[:24]), np.mean(top100[2:26]))] + [perc_dif_func(np.mean(p100_sim[24:45]), np.mean(top100[26:47]))]
    error5 = b_perc_diff_25 + b_perc_diff_50 + b_perc_diff_70 + b_perc_diff_80 + b_perc_diff_90 + b_perc_diff_99 + b_perc_diff_100
    print error5
    # labor calibration euler
    labor_sim = ((solutions[S*J:2*S*J]).reshape(S, J)*lambdas.reshape(1, J)).sum(axis=1)
    error6 = list(perc_dif_func(labor_sim, labor_dist_data))
    # combine eulers
    output = np.array(error5 + error6)
    fsolve_no_converg = np.abs(Steady_State_X(solutions)).max()
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
    if (chi_b_guesses_init <= 0.0).any():
        output += 1e9
    weighting_mat = np.eye(2*J + S)
    # weighting_mat[10:10] = 10.0
    # weighting_mat[11:11] = 10.0
    # weighting_mat[12:12] = 10.0
    # weighting_mat[13:13] = 10.0
    scaling_val = 100.0
    value = np.dot(scaling_val * np.dot(output.reshape(1, 2*J+S), weighting_mat), scaling_val * output.reshape(2*J+S, 1))
    print value.sum()
    return value.sum()

'''
------------------------------------------------------------------------
    Run SS in various ways, depending on the stage of the simulation
------------------------------------------------------------------------
'''

bnds = tuple([(1e-6, None)] * (S + J))

b_guess_init = np.ones((S, J)) * .01
n_guess_init = np.ones((S, J)) * .99 * ltilde
Kg = (omega_SS * b_guess_init).sum()
Lg = get_L(e, n_guess_init)
Yg = get_Y(Kg, Lg)
wguess = get_w(Yg, Lg)
rguess = get_r(Yg, Kg)
avIguess = ((rguess * b_guess_init + wguess * e * n_guess_init) * omega_SS).sum()
factor_guess = [mean_income_data / avIguess]
guesses = list(b_guess_init.flatten()) + list(n_guess_init.flatten()) + factor_guess

chi_b_guesses = np.ones(S+J)
chi_b_guesses[0:J] = np.array([5, 10, 90, 250, 250, 250, 250]) + chi_b_scal
print 'Chi_b:', chi_b_guesses[0:J]
chi_b_guesses[J:] = chi_n_guess
chi_b_guesses = list(chi_b_guesses)
final_chi_b_params = chi_b_guesses
Steady_State_X2 = lambda x: Steady_State(x, final_chi_b_params)
solutions = opt.fsolve(Steady_State_X2, guesses, xtol=1e-13)
print np.array(Steady_State_X2(solutions)).max()
print('solutions')
print(solutions)
#elif
#SS_stage == 'loop_calibration':
#    variables = pickle.load(open("OUTPUT/Saved_moments/loop_calibration_solutions.pkl", "r"))
#    for key in variables:
#        globals()[key] = variables[key]
#    guesses = list((solutions[:S*J].reshape(S, J) * scal.reshape(1, J)).flatten()) + list(
#        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
#    chi_b_guesses = np.ones(S+J)
#    chi_b_guesses[0:J] = np.array([5, 10, 90, 250, 250, 250, 250]) + chi_b_scal
#    print 'Chi_b:', chi_b_guesses[0:J]
#    chi_b_guesses[J:] = chi_n_guess
#    chi_b_guesses = list(chi_b_guesses)
#    final_chi_b_params = chi_b_guesses
#    Steady_State_X2 = lambda x: Steady_State(x, final_chi_b_params)
#    solutions = opt.fsolve(Steady_State_X2, guesses, xtol=1e-13)
#    print np.array(Steady_State_X2(solutions)).max()
#elif SS_stage == 'constrained_minimization':
#    variables = pickle.load(open("OUTPUT/Saved_moments/loop_calibration_solutions.pkl", "r"))
#    dictionary = {}
#    for key in variables:
#        globals()[key] = variables[key]
#        dictionary[key] = globals()[key]
#    pickle.dump(dictionary, open("OUTPUT/Saved_moments/minimization_solutions.pkl", "w"))
#    guesses = list((solutions[:S*J].reshape(S, J) * scal.reshape(1, J)).flatten()) + list(
#        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
#    chi_b_guesses = final_chi_b_params
#    func_to_min_X = lambda x: func_to_min(x, guesses)
#    final_chi_b_params = opt.minimize(func_to_min_X, chi_b_guesses, method='TNC', tol=1e-7, bounds=bnds).x
#    print 'The final bequest parameter values:', final_chi_b_params
#    Steady_State_X2 = lambda x: Steady_State(x, final_chi_b_params)
#    solutions = opt.fsolve(Steady_State_X2, solutions_pre, xtol=1e-13)
#    print np.array(Steady_State_X2(solutions)).max()
#elif SS_stage == 'SS_init':
#    variables = pickle.load(open("OUTPUT/Saved_moments/minimization_solutions.pkl", "r"))
#    for key in variables:
#        globals()[key] = variables[key]
#    guesses = list((solutions[:S*J].reshape(S, J) * scal.reshape(1, J)).flatten()) + list(
#        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
#    chi_b_guesses = final_chi_b_params
#    Steady_State_X2 = lambda x: Steady_State(x, final_chi_b_params)
#    solutions = opt.fsolve(Steady_State_X2, guesses, xtol=1e-13)
#    print np.array(Steady_State_X2(solutions)).max()
#elif SS_stage == 'SS_tax':
#    variables = pickle.load(open("OUTPUT/Saved_moments/SS_init_solutions.pkl", "r"))
#    for key in variables:
#        globals()[key] = variables[key]
#    guesses = list((solutions[:S*J].reshape(S, J) * scal.reshape(1, J)).flatten()) + list(
#        solutions[S*J:-1].reshape(S, J).flatten()) + [solutions[-1]]
#    chi_b_guesses = final_chi_b_params
#    Steady_State_X2 = lambda x: Steady_State(x, final_chi_b_params)
#    solutions = opt.fsolve(Steady_State_X2, guesses, xtol=1e-13)
#    print np.array(Steady_State_X2(solutions)).max()


'''
------------------------------------------------------------------------
    Save the values in various ways, depending on the stage of
        the simulation, to be used in TPI or graphing functions
------------------------------------------------------------------------
'''



var_names = ['solutions', 'final_chi_b_params']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]

bssmat = solutions[0:(S-1) * J].reshape(S-1, J)
BQ = solutions[(S-1)*J:S*J]
Bss = (np.array(list(bssmat) + list(BQ.reshape(1, J))).reshape(
        S, J) * omega_SS * rho.reshape(S, 1)).sum(0)
bssmat2 = np.array(list(np.zeros(J).reshape(1, J)) + list(bssmat))
bssmat3 = np.array(list(bssmat) + list(BQ.reshape(1, J)))
Kss = (omega_SS[:-1, :] * bssmat).sum() + (omega_SS[-1, :]*BQ).sum()
nssmat = solutions[S * J:-1].reshape(S, J)
Lss = get_L(e, nssmat)
Yss = get_Y(Kss, Lss)
wss = get_w(Yss, Lss)
rss = get_r(Yss, Kss)
b1_2 = np.array(list(np.zeros(J).reshape((1, J))) + list(bssmat))
factor_ss = solutions[-1]
BQ = Bss * (1+rss)
T_Hss = tax.tax_lump(rss, bssmat2, wss, e, nssmat, BQ, lambdas, factor_ss, omega_SS)
taxss = tax.total_taxes_SS(rss, bssmat2, wss, e, nssmat, BQ, lambdas, factor_ss, T_Hss)
cssmat = get_cons(rss, bssmat2, wss, e, nssmat, BQ.reshape(1, J), lambdas.reshape(1, J), bssmat3, g_y, taxss)

bssmat_init = np.array(list(bssmat) + list(BQ.reshape(1, J)))
nssmat_init = nssmat

var_names = ['bssmat_init', 'nssmat_init']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
#pickle.dump(dictionary, open("OUTPUT/SSinit/ss_init_tpi.pkl", "w"))
#
#var_names = ['S', 'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'e', 'E',
#                 'J', 'Kss', 'bssmat', 'Lss', 'nssmat',
#                 'Yss', 'wss', 'rss', 'omega', 'chi_n', 'chi_b', 'ltilde', 'T',
#                 'g_n', 'g_y', 'omega_SS', 'TPImaxiter', 'TPImindist', 'BQ',
#                 'rho', 'Bss', 'lambdas',
#                 'b_ellipse', 'k_ellipse', 'upsilon',
#                 'factor_ss',  'a_tax_income', 'b_tax_income',
#                 'c_tax_income', 'd_tax_income', 'tau_payroll',
#                 'tau_bq', 'theta', 'retire',
#                 'mean_income_data', 'bssmat2', 'cssmat',
#                 'starting_age', 'bssmat3',
#                 'ending_age', 'T_Hss', 'euler1', 'euler2', 'euler3',
#                 'h_wealth', 'p_wealth', 'm_wealth']
#dictionary = {}
#for key in var_names:
#    dictionary[key] = globals()[key]

print('interest rate')
print(rss)
print('wage rate')
print(wss)