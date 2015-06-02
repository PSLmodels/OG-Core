'''
------------------------------------------------------------------------
Last updated 5/21/2015

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method, where labor in inelastically supplied.

This py-file calls the following other file(s):
            tax_funcs.py
            OUTPUT/Saved_moments/tpi_var.pkl
            OUTPUT/SSinit/ss_init.pkl
            OUTPUT/SS/ss_vars.pkl
            OUTPUT/SSinit/ss_init_tpi.pkl


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
import time
import pickle
import scipy.optimize as opt

import tax_funcs as tax

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
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
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
TPI_initial_run = whether this is the baseline TPI or not
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/Saved_moments/tpi_var.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

if TPI_initial_run:
    variables = pickle.load(open("OUTPUT/SSinit/ss_init.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]
else:
    variables = pickle.load(open("OUTPUT/SS/ss_vars.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]
    variables = pickle.load(open("OUTPUT/SSinit/ss_init_tpi.pkl", "r"))
    for key in variables:
        globals()[key] = variables[key]

'''
------------------------------------------------------------------------
Set other parameters, objects, and functions
------------------------------------------------------------------------
initial_b       = (S-1)xJ array of the initial distribution of capital
                  for TPI
K0              = initial aggregate capital stock
b1_2init        = (S-1)xJ array of the initial distribution of capital
                  for TPI (period t+1)
b2_2init        = (S-1)xJ array of the initial distribution of capital
                  for TPI (period t+2)
initial_n       = SxJ arry of the initial distribution of labor for TPI
L0              = initial aggregate labor supply (scalar)
Y0              = initial aggregate output (scalar)
w0              = initial wage (scalar)
r0              = intitial rental rate (scalar)
c0              = SxJ arry of the initial distribution of consumption
------------------------------------------------------------------------
'''

N_tilde = omega.sum(1).sum(1)
omega_stationary = omega / N_tilde.reshape(T+S, 1, 1)


def constraint_checker_SS(b_dist, n_dist, c_dist):
    '''
    Parameters:
        b_dist = distribution of capital ((S-1)xJ array)
        n_dist = distribution of labor (SxJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        c_dist = distribution of consumption (SxJ array)

    Created Variables:
        flag1 = False if all borrowing constraints are met, true
               otherwise.
        flag2 = False if all labor constraints are met, true otherwise

    Returns:
        Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    print 'Checking constraints on the initial distributions of' \
        ' capital, labor, and consumption for TPI.'
    flag1 = False
    if b_dist.sum() / N_tilde[-1] <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to zero.'
        flag1 = True
    if flag1 is False:
        print '\tThere were no violations of the borrowing constraints.'
    flag2 = False
    if (n_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (n_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (c_dist < 0).any():
        print '\tWARNING: Consumption violates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'


def constraint_checker_TPI(b_dist, n_dist, c_dist, t):
    '''
    Parameters:
        b_dist = distribution of capital ((S-1)xJ array)
        n_dist = distribution of labor (SxJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        c_dist = distribution of consumption (SxJ array)

    Returns:
        Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    if b_dist.sum() / N_tilde[t] <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to ' \
            'zero in period %.f.' % t
    if (n_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints ' \
            'in period %.f.' % t
    if (n_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint in '\
            'period %.f.' % t
    if (c_dist < 0).any():
        print '\tWARNING: Consumption violates nonnegativity constraints in ' \
            'period %.f.' % t



def get_Y(K_now, L_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = Z * (K_now ** alpha) * (L_now ** (1 - alpha))
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


def get_L(e, n, omega):
    '''
    Parameters: e, n

    Returns:    Aggregate labor
    '''
    L_now = np.sum(e * omega * n)
    return L_now


def get_K(b, omega):
    '''
    Parameters: b, omega

    Returns:    Aggregate labor
    '''
    L_now = np.sum(b * omega)
    return L_now


def marg_ut_cons(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''

    return c**(-sigma)


def marg_ut_labor(n, chi_n1):
    '''
    Parameters: Labor
    This function allows for different
    sizes of chi_n

    Returns:    Marginal Utility of Labor
    '''
    deriv = b_ellipse * (1/ltilde) * ((1 - (n / ltilde) ** upsilon) ** (
        (1/upsilon)-1)) * (n / ltilde) ** (upsilon - 1)
    output = chi_n1 * deriv
    return output


def marg_ut_sav(bequest, chi_b):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b[-1] * (bequest ** (-sigma))
    return output


def get_cons(r, b1, w, e, n, bq, lambdas, b2, g_y, net_tax):
    '''
    Parameters: rental rate, capital stock (t-1), wage, e, labor stock,
                bequests, lambdas, capital stock (t), growth rate y, taxes

    Returns:    Consumption
    '''
    cons = (1 + r)*b1 + w*e*n + bq / lambdas - b2*np.exp(g_y) - net_tax
    return cons


def convex_combo(var1, var2, scalar):
    combo = scalar * var1 + (1-scalar)*var2
    return combo

'''
------------------------------------------------------------------------
    Set initial values
------------------------------------------------------------------------
'''

if TPI_initial_run:
    initial_b = np.array(list(bssmat) + list(BQ.reshape(1, J)))
    initial_n = nssmat
else:
    initial_b = bssmat_init
    initial_n = nssmat_init
K0 = get_K(initial_b, omega_stationary[0])
b1_2init = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_b[:-1]))
b2_2init = initial_b
L0 = get_L(e, initial_n, omega_stationary[1])
Y0 = get_Y(K0, L0)
w0 = get_w(Y0, L0)
r0 = get_r(Y0, K0)
B0 = (initial_b * omega_stationary[0] * rho.reshape(S, 1)).sum(0)
T_H_0 = tax.tax_lump(r0, b1_2init, w0, e, initial_n, (1+r0)*B0, lambdas, factor_ss, omega_stationary[0])
tax0 = tax.total_taxes_SS(r0, b1_2init, w0, e, initial_n, (1+r0)*B0, lambdas, factor_ss, T_H_0)
c0 = get_cons(r0, b1_2init, w0, e, initial_n, (1+r0)*B0.reshape(1, J), lambdas.reshape(1, J), b2_2init, g_y, tax0)
constraint_checker_SS(initial_b[:-1], initial_n, c0)

'''
------------------------------------------------------------------------
Solve for equilibrium transition path by TPI
------------------------------------------------------------------------
'''


def SS_TPI_firstdoughnutring(guesses, winit, rinit, BQinit, T_H_init):
    b2 = float(guesses[0])
    n1 = float(guesses[1])
    b1 = float(initial_b[-2, j])
    # Euler 1 equations
    tax11 = tax.total_taxes_eul3_TPI(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], factor_ss, T_H_init, j)
    cons11 = get_cons(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], b2, g_y, tax11)
    bequest_ut = rho * np.exp(-sigma * g_y) * chi_b[-1, j] * b2 ** (-sigma)
    error1 = marg_ut_cons(cons11) - bequest_ut
    # Euler 2 equations
    tax2 = tax.total_taxes_eul3_TPI(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], factor_ss, T_H_init, j)
    cons2 = get_cons(rinit, b1, winit, e[-1, j], n1, BQinit, lambdas[j], b2, g_y, tax2)
    income2 = (rinit * b1 + winit * e[-1, j] * n1) * factor_ss
    deriv2 = 1 - tau_payroll - tax.tau_income(rinit, b1, winit, e[
        -1, j], n1, factor_ss) - tax.tau_income_deriv(
        rinit, b1, winit, e[-1, j], n1, factor_ss) * income2
    error2 = marg_ut_cons(cons2) * winit * e[-1, j] * deriv2 - marg_ut_labor(n1, chi_n[-1])
    if n1 <= 0:
        error2 += 1e12
    return [error1] + [error2]


def Steady_state_TPI_solver(guesses, winit, rinit, BQinit, T_H_init, t):
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
    length = len(guesses)/2
    b_guess = np.array(guesses[:length])
    n_guess = np.array(guesses[length:])

    if length == S:
        b1 = np.array([0] + list(b_guess[:-2]))
    else:
        b1 = np.array([(initial_b[-(s+2), j])] + list(b_guess[:-2]))
    b2 = b_guess[:-1]
    b3 = b_guess[1:]
    w1 = winit[t:t+length-1]
    w2 = winit[t+1:t+length]
    r1 = rinit[t:t+length-1]
    r2 = rinit[t+1:t+length]
    n1 = n_guess[:-1]
    n2 = n_guess[1:]
    e1 = e[-length:-1, j]
    e2 = e[-length+1:, j]
    B1 = BQinit[t:t+length-1]
    B2 = BQinit[t+1:t+length]
    T_H1 = T_H_init[t:t+length-1]
    T_H2 = T_H_init[t+1:t+length]
    # Euler 1 equations
    tax11 = tax.total_taxes_TPI1(r1, b1, w1, e1, n1, B1, lambdas[j], factor_ss, T_H1, j)
    tax12 = tax.total_taxes_TPI1_2(r2, b2, w2, e2, n2, B2, lambdas[j], factor_ss, T_H2, j)
    cons11 = get_cons(r1, b1, w1, e1, n1, B1, lambdas[j], b2, g_y, tax11)
    cons12 = get_cons(r2, b2, w2, e2, n2, B2, lambdas[j], b3, g_y, tax12)
    income1 = (r2 * b2 + w2 * e2 * n2) * factor_ss
    bequest_ut = rho[-(length):-1] * np.exp(-sigma * g_y) * chi_b[-(length):-1, j] * b2 ** (-sigma)
    deriv1 = 1 + r2 * (1 - tax.tau_income(
        r2, b2, w2, e2, n2, factor_ss) - tax.tau_income_deriv(
        r2, b2, w2, e2, n2, factor_ss) * income1) - tax.tau_w_prime(
        b2)*b2 - tax.tau_wealth(b2)
    error1 = marg_ut_cons(cons11) - beta * (1-rho[-(length):-1]) * np.exp(-sigma * g_y) * deriv1 * marg_ut_cons(
        cons12) - bequest_ut
    # Euler 2 equations
    if length == S:
        b1_2 = np.array([0] + list(b_guess[:-1]))
    else:
        b1_2 = np.array([(initial_b[-(s+2), j])] + list(b_guess[:-1]))

    b2_2 = b_guess
    w = winit[t:t+length]
    r = rinit[t:t+length]
    B = BQinit[t:t+length]
    T_H = T_H_init[t:t+length]
    tax2 = tax.total_taxes_TPI2(r, b1_2, w, e[-(length):, j], n_guess, B, lambdas[j], factor_ss, T_H, j)
    cons2 = get_cons(r, b1_2, w, e[-(length):, j], n_guess, B, lambdas[j], b2_2, g_y, tax2)
    income2 = (r * b1_2 + w * e[-(length):, j] * n_guess) * factor_ss
    deriv2 = 1 - tau_payroll - tax.tau_income(r, b1_2, w, e[
        -(length):, j], n_guess, factor_ss) - tax.tau_income_deriv(
        r, b1_2, w, e[-(length):, j], n_guess, factor_ss) * income2
    error2 = marg_ut_cons(cons2) * w * e[-(length):, j] * deriv2 - marg_ut_labor(n_guess, chi_n[-length:])
    # Euler 3 equations
    tax3 = tax.total_taxes_eul3_TPI(r[-1], b_guess[-2], w[-1], e[-1, j], n_guess[-1], B[-1], lambdas[j], factor_ss, T_H[-1], j)
    cons3 = get_cons(r[-1], b_guess[-2], w[-1], e[-1, j], n_guess[-1], B[-1], lambdas[j], b_guess[-1], g_y, tax3)
    error3 = marg_ut_cons(cons3) - np.exp(
        -sigma * g_y) * marg_ut_sav(b_guess[-1], chi_b[:, j])
    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] += 1e12
    mask2 = n_guess > ltilde
    error2[mask2] += 1e12
    mask3 = cons2 < 0
    error2[mask3] += 1e12
    mask4 = b_guess <= 0
    error2[mask4] += 1e12
    return list(error1.flatten()) + list(
        error2.flatten()) + list(error3.flatten())


domain = np.linspace(0, T, T)
Kinit = (-1/(domain + 1)) * (Kss-K0) + Kss
Kinit[-1] = Kss
Kinit = np.array(list(Kinit) + list(np.ones(S)*Kss))
Linit = np.ones(T+S) * Lss
Yinit = get_Y(Kinit, Linit)
winit = get_w(Yinit, Linit)
rinit = get_r(Yinit, Kinit)
BQinit = np.zeros((T+S, J))
for j in xrange(J):
    BQinit[:, j] = list(np.linspace((1+r0)*B0[j], BQ[j], T)) + [BQ[j]]*S
BQinit = np.array(BQinit)
T_H_init = np.ones(T+S) * T_Hss

# Make array of initial guesses
domain2 = np.tile(domain.reshape(T, 1, 1), (1, S, J))
ending_b = np.array(list(bssmat) + list(BQ.reshape(1, J)))
guesses_b = (-1/(domain2 + 1)) * (ending_b-initial_b) + ending_b
ending_b_tail = np.tile(ending_b.reshape(1, S, J), (S, 1, 1))
guesses_b = np.append(guesses_b, ending_b_tail, axis=0)

domain3 = np.tile(np.linspace(0, 1, T).reshape(T, 1, 1), (1, S, J))
guesses_n = domain3 * (nssmat - initial_n) + initial_n
ending_n_tail = np.tile(nssmat.reshape(1, S, J), (S, 1, 1))
guesses_n = np.append(guesses_n, ending_n_tail, axis=0)

TPIiter = 0
TPIdist = 10
print 'Starting time path iteration.'

euler_errors = np.zeros((T, 2*S, J))
TPIdist_vec = np.zeros(TPImaxiter)
nu_current = nu

while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
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
        for s in xrange(S-2):  # Upper triangle
            b_guesses_to_use = np.diag(guesses_b[1:S+1, :, j], S-(s+2))
            n_guesses_to_use = np.diag(guesses_n[:S, :, j], S-(s+2))
            solutions = opt.fsolve(Steady_state_TPI_solver, list(
                b_guesses_to_use) + list(n_guesses_to_use), args=(
                winit, rinit, BQinit[:, j], T_H_init, 0), xtol=1e-13)
            b_vec = solutions[:len(solutions)/2]
            b_mat[1:S+1, :, j] += np.diag(b_vec, S-(s+2))
            n_vec = solutions[len(solutions)/2:]
            n_mat[:S, :, j] += np.diag(n_vec, S-(s+2))

        for t in xrange(0, T):
            b_guesses_to_use = np.diag(guesses_b[t+1:t+S+1, :, j])
            n_guesses_to_use = np.diag(guesses_n[t:t+S, :, j])
            solutions = opt.fsolve(Steady_state_TPI_solver, list(
                b_guesses_to_use) + list(n_guesses_to_use), args=(
                winit, rinit, BQinit[:, j], T_H_init, t), xtol=1e-13)
            b_vec = solutions[:S]
            b_mat[t+1:t+S+1, :, j] += np.diag(b_vec)
            n_vec = solutions[S:]
            n_mat[t:t+S, :, j] += np.diag(n_vec)
            inputs = list(solutions)
            euler_errors[t, :, j] = np.abs(Steady_state_TPI_solver(
                inputs, winit, rinit, BQinit[:, j], T_H_init, t))
        # b_mat[1, -1, j], n_mat[0, -1, j] = np.array(opt.fsolve(SS_TPI_firstdoughnutring, [b_mat[1, -2, j], n_mat[0, -2, j]],
        #     args=(winit[1], rinit[1], BQinit[1, j], T_H_init[1])))
    
    b_mat[0, :, :] = initial_b
    b_mat[1, -1, :]= b_mat[1, -2, :]
    n_mat[0, -1, :] = n_mat[0, -2, :]
    Knew = get_K(b_mat[:T], omega_stationary[:T])
    Lnew = get_L(e.reshape(1, S, J), n_mat[:T], omega_stationary[1:T+1])
    Bnew = (b_mat[:T, :, :] * omega_stationary[:T, :, :] * rho.reshape(1, S, 1)).sum(1)
    Kinit = convex_combo(Knew, Kinit[:T], nu)
    Linit = convex_combo(Lnew, Linit[:T], nu)
    BQinit[:T] = convex_combo(Bnew, BQinit[:T], nu)
    guesses_b = convex_combo(b_mat, guesses_b, nu)
    guesses_n = convex_combo(n_mat, guesses_n, nu)
    TPIdist = np.array(list(
        np.abs(Knew - Kinit)) + list(np.abs(Bnew - BQinit[
            :T]).flatten()) + list(np.abs(Lnew - Linit))).max()
    TPIdist_vec[TPIiter] = TPIdist
    # After T=10, if cycling occurs, drop the value of nu
    # wait til after T=10 or so, because sometimes there is a jump up
    # in the first couple iterations
    if TPIiter > 10:
        if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter-1] > 0:
            nu_current /= 2
            print 'New Value of nu:', nu_current
    TPIiter += 1
    print '\tIteration:', TPIiter
    print '\t\tDistance:', TPIdist
    if (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
        bmat2 = np.zeros((T, S, J))
        bmat2[:, 1:, :] = b_mat[:T, :-1, :]
        T_H_init = np.array(list(tax.tax_lumpTPI(rinit[:T].reshape(T, 1, 1), bmat2, winit[:T].reshape(
            T, 1, 1), e.reshape(1, S, J), n_mat[:T], BQinit[:T].reshape(T, 1, J), lambdas.reshape(
            1, 1, J), factor_ss, omega_stationary[:T])) + [T_Hss]*S)
        Yinit = get_Y(Kinit, Linit)
        winit = np.array(list(get_w(Yinit, Linit)) + list(np.ones(S)*wss))
        rinit = np.array(list(get_r(Yinit, Kinit)) + list(np.ones(S)*rss))
    if TPIdist < TPImindist:
        BQinit[:T] = Bnew
        Kinit[:T] = Knew
        Linit[:T] = Lnew
    

Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)
Lpath_TPI = list(Linit) + list(np.ones(10)*Lss)
BQpath_TPI = np.array(list(BQinit) + list(np.ones((10, J))*Bss))

print 'TPI is finished.'



b1 = np.zeros((T, S, J))
b1[:, 1:, :] = b_mat[:T, :-1, :]
b2 = np.zeros((T, S, J))
b2[:, :, :] = b_mat[:T, :, :]
if TPI_initial_run:
    taxinit = tax.total_taxes_path(rinit[:T], b1, winit[:T], e.reshape(
        1, S, J), n_mat[:T], BQinit[:T, :].reshape(T, 1, J), lambdas, factor_ss, T_H_init[:T])
    cinit = get_cons(rinit[:T].reshape(T, 1, 1), b1, winit[:T].reshape(T, 1, 1), e.reshape(1, S, J), n_mat[:T], BQinit[:T].reshape(T, 1, J), lambdas.reshape(1, 1, J), b2, g_y, taxinit)
else:
    taxinit2 = tax.total_taxes_path(rinit[:T], b1, winit[:T], e.reshape(
        1, S, J), n_mat[:T], BQinit[:T, :].reshape(T, 1, J), lambdas, factor_ss, T_H_init[:T])
    cinit = get_cons(rinit[:T].reshape(T, 1, 1), b1, winit[:T].reshape(T, 1, 1), e.reshape(1, S, J), n_mat[:T], BQinit[:T].reshape(T, 1, J), lambdas.reshape(1, 1, J), b2, g_y, taxinit2)
print'Checking time path for violations of constaints.'
for t in xrange(T):
    constraint_checker_TPI(b_mat[t, :-1, :], n_mat[
        t], cinit[t], t)
print '\tFinished.'

'''
------------------------------------------------------------------------
Generate values for TPI graphs
------------------------------------------------------------------------
eul1   = results of euler 1
eul2   = results of euler 2
eul3   = results of euler 3
------------------------------------------------------------------------
'''
eul1 = euler_errors[:, :S-1, :].max(1).max(1)
eul2 = euler_errors[:, S-1:, :].max(1).max(1)
eul3 = euler_errors[:, S-1, :].max(1)

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

print 'Saving TPI variable values.'

if TPI_initial_run:
    var_names = ['Kpath_TPI', 'TPIiter', 'TPIdist', 'T', 'b_mat',
                 'eul1', 'eul2', 'eul3', 'Lpath_TPI', 'BQpath_TPI',
                 'n_mat', 'rinit', 'winit', 'Yinit', 'T_H_init', 'taxinit',
                 'cinit']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/TPIinit/TPIinit_vars.pkl", "w"))
else:
    var_names = ['Kpath_TPI', 'TPIiter', 'TPIdist', 'T', 'b_mat',
                 'eul1', 'eul2', 'eul3', 'Lpath_TPI', 'BQpath_TPI',
                 'n_mat', 'rinit', 'winit', 'Yinit', 'T_H_init', 'taxinit2',
                 'cinit']
    dictionary = {}
    for key in var_names:
        dictionary[key] = globals()[key]
    pickle.dump(dictionary, open("OUTPUT/TPI/TPI_vars.pkl", "w"))
print '\tFinished.'
