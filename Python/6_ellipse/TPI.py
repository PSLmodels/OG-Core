'''
------------------------------------------------------------------------
Last updated 9/19/2014

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method, where labor in inelastically supplied.

This py-file calls the following other file(s):
            ss_vars.pkl

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/TPI_K.png
            OUTPUT/TPI_N.png
            OUTPUT/euler_errors_TPI_2D.png
            OUTPUT/TPI_vars.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
import scipy.optimize as opt

'''
------------------------------------------------------------------------
Import steady state distribution, parameters and other objects from
steady state computation in ss_vars.pkl
------------------------------------------------------------------------
S        = number of periods an individual lives
beta     = discount factor (0.96 per year)
sigma    = coefficient of relative risk aversion
alpha    = capital share of income
nu       = contraction parameter in steady state iteration process
           representing the weight on the new distribution gamma_new
A        = total factor productivity parameter in firms' production
           function
bqtilde  = minimum bequest value
delta    = decreciation rate of capital
ltilde   = measure of time each individual is endowed with each period
ctilde   = minimum value of consumption
chi_n    = discount factor of labor
chi_b    = discount factor of incidental bequests
eta      = Frisch elasticity of labor supply
e        = S x J matrix of age dependent possible working abilities e_s
J        = number of points in the support of e
Kss      = steady state aggregate capital stock: scalar
Kssvec   = ((S-1) x 1) vector of the steady state level of capital
           (averaged across ability types)
Kssmat   = ((S-1) x J) array of the steady state distribution of
           capital
Lss      = steady state aggregate labor: scalar
Lssvec   = (S x 1) vector of the steady state level of labor
           (averaged across ability types)
Lssmat   = (S x J) array of the steady state distribution of labor
Yss      = steady state aggregate output: scalar
wss      = steady state real wage: scalar
rss      = steady state real rental rate: scalar
K_agg    = Aggregate level of capital: scalar
runtime  = total time (in seconds) that the steady state solver took to
            run
hours    = total hours that the steady state solver took to run
minutes  = total minutes (minus the total hours) that the steady state
            solver took to run
seconds  = total seconds (minus the total hours and minutes) that the
            steady state solver took to run
T        = number of periods until the steady state
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/ss_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

start_time = time.time()  # Start timer

'''
------------------------------------------------------------------------
Set other parameters, objects, and functions
------------------------------------------------------------------------
initial_K       = (S-1)xJ array of the initial distribution of capital
                  for TPI
K0              = initial aggregate capital stock
K1_2init        = (S-1)xJ array of the initial distribution of capital
                  for TPI (period t+1)
K2_2init        = (S-1)xJ array of the initial distribution of capital
                  for TPI (period t+2)
initial_L       = SxJ arry of the initial distribution of labor for TPI
L0              = initial aggregate labor supply (scalar)
Y0              = initial aggregate output (scalar)
w0              = initial wage (scalar)
r0              = intitial rental rate (scalar)
c0              = SxJ arry of the initial distribution of consumption
------------------------------------------------------------------------
'''

N_tilde = omega.sum(1).sum(1)
omega_stationary = omega / N_tilde.reshape(T+S, 1, 1)


def constraint_checker1(k_dist, l_dist, w, r, e, c_dist, BQ):
    '''
    Parameters:
        k_dist = distribution of capital ((S-1)xJ array)
        l_dist = distribution of labor (SxJ array)
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
    if k_dist.sum() / N_tilde[-1] <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to zero.'
        flag1 = True
    if borrowing_constraints(k_dist, w, r, e, l_dist, BQ) is True:
        print '\tWARNING: Borrowing constraints have been violated.'
        flag1 = True
    if flag1 is False:
        print '\tThere were no violations of the borrowing constraints.'
    flag2 = False
    if (l_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (l_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (c_dist < 0).any():
        print '\tWARNING: Consumption volates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'


def constraint_checker2(k_dist, l_dist, w, r, e, c_dist, t):
    '''
    Parameters:
        k_dist = distribution of capital ((S-1)xJ array)
        l_dist = distribution of labor (SxJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        c_dist = distribution of consumption (SxJ array)

    Returns:
        Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    if k_dist.sum() / N_tilde[t] <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to ' \
            'zero in period %.f.' % t
    if (l_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints ' \
            'in period %.f.' % t
    if (l_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint in '\
            'period %.f.' % t
    if (c_dist < 0).any():
        print '\tWARNING: Consumption volates nonnegativity constraints in ' \
            'period %.f.' % t


def borrowing_constraints(K_dist, w, r, e, n, BQ):
    '''
    Parameters:
        K_dist = Distribution of capital ((S-1)xJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        n      = distribution of labor (SxJ array)

    Returns:
        False value if all the borrowing constraints are met, True
            if there are violations.
    '''
    b_min = np.zeros((S-1, J))
    b_min[-1, :] = (ctilde + bqtilde - w * e[S-1, :] * ltilde - BQ.reshape(1, J) / bin_weights) / (1 + r)
    for i in xrange(S-2):
        b_min[-(i+2), :] = (ctilde + np.exp(g_y) * b_min[-(i+1), :] - w * e[
            -(i+2), :] * ltilde - BQ.reshape(1, J) / bin_weights) / (1 + r)
    difference = K_dist - b_min
    if (difference < 0).any():
        return True
    else:
        return False


def get_Y(K_now, L_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = A * (K_now ** alpha) * (L_now ** (1 - alpha))
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
    L_now = np.sum(e * omega_stationary[0, :, :] * n)
    return L_now


def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''

    return c**(-sigma)


def MUl(n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    deriv = -b_ellipse * ((1 - (n / ltilde) ** omega_ellipse) ** ((1/omega_ellipse)-1)) * (n / ltilde) ** (omega_ellipse - 1)
    output = chi_n.reshape(S, 1) * deriv
    return output


def MUb(bq):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b * (bq ** (-sigma))
    return output

initial_K = np.array(list(Kssmat) + list(BQ.reshape(1, J)))
K0 = (omega_stationary[0] * initial_K[:, :]).sum()
K1_2init = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_K[:-1]))
K2_2init = initial_K
initial_L = Lssmat
L0 = get_L(e, initial_L)
Y0 = get_Y(K0, L0)
w0 = get_w(Y0, L0)
r0 = get_r(Y0, K0)
B0 = (initial_K * omega_stationary[0] * mort_rate.reshape(S,1)).sum(0)
c0 = (1 + r0) * K1_2init + w0 * e * initial_L - K2_2init * np.exp(g_y) + (1 + rss) * Bss.reshape(1, J)/bin_weights.reshape(1,J)
constraint_checker1(initial_K[:-1], initial_L, w0, r0, e, c0, B0)
print 'K0 divided by Kss =', K0/Kss

'''
------------------------------------------------------------------------
Solve for equilibrium transition path by TPI
------------------------------------------------------------------------
Kinit        = 1 x T+S vector, initial time path of aggregate capital
               stock
Linit        = 1 x T+S vector, initial time path of aggregate labor
               demand. This is just equal to a 1 x T+S vector of Lss
               because labor is supplied inelastically
Yinit        = 1 x T+S vector, initial time path of aggregate output
winit        = 1 x T+S vector, initial time path of real wage
rinit        = 1 x T+S vector, initial time path of real interest rate
Binit        = T+S x J array, time paths for incidental bequests
TPIiter      = Iterations of TPI
TPIdist      = Current distance between iterations of TPI
K_mat        = (T+S)x(S-1)xJ array of distribution of capital across
               time, age, and ability
Knew         = 1 x T vector, new time path of aggregate capital stock
L_mat        = (T+S)xSxJ array of distribution of labor across
               time, age, and ability
Lnew         = 1 x T vector, new time path of aggregate labor supply
Kpath_TPI    = 1 x T vector, final time path of aggregate capital stock
Lpath_TPI    = 1 x T vector, final time path of aggregate labor supply
elapsed_time = elapsed time of TPI
hours        = Hours needed to find the steady state
minutes      = Minutes needed to find the steady state, less the number
               of hours
seconds      = Seconds needed to find the steady state, less the number
               of hours and minutes
euler_errors = TxSxJ array of euler errors
------------------------------------------------------------------------
'''


def MUl2(n, chi_n1):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    deriv = -b_ellipse * ((1 - (n / ltilde) ** omega_ellipse) ** ((1/omega_ellipse)-1)) * (n / ltilde) ** (omega_ellipse - 1)
    output = chi_n1 * deriv
    return output


def Euler_Error(guesses, winit, rinit, Binit, t):
    '''
    Parameters:
        guesses = distribution of capital and labor in period t
                  ((S-1)*S*J x 1 list)
        winit   = wage rate (scalar)
        rinit   = rental rate (scalar)
        t       = time period

    Returns:
        Value of Euler error. (as an (S-1)*S*J x 1 list)
    '''
    length = len(guesses)/2
    K_guess = np.array(guesses[:length])
    L_guess = np.array(guesses[length:])

    if length == S:
        K1 = np.array([0] + list(K_guess[:-2]))
    else:
        K1 = np.array([(initial_K[-(s+2), j])] + list(K_guess[:-2]))
    K2 = K_guess[:-1]
    K3 = K_guess[1:]
    w1 = winit[t:t+length-1]
    w2 = winit[t+1:t+length]
    r1 = rinit[t:t+length-1]
    r2 = rinit[t+1:t+length]
    l1 = L_guess[:-1]
    l2 = L_guess[1:]
    e1 = e[-length:-1, j]
    e2 = e[-length+1:, j]
    B1 = Binit[t:t+length-1]
    B2 = Binit[t+1:t+length]

    error1 = MUc((1 + r1)*K1 + w1 * e1 * l1 + (1 + r1)*B1/bin_weights[j] - np.exp(g_y) * K2) \
        - beta * surv_rate[-(length):-1] * np.exp(
            -sigma * g_y) * (1 + r2)*MUc(
            (1 + r2)*K2 + w2*e2*l2 + (1 + r2)*B2/bin_weights[j] - np.exp(g_y) * K3)

    if length == S:
        K1_2 = np.array([0] + list(K_guess[:-1]))
    else:
        K1_2 = np.array([(initial_K[-(s+2), j])] + list(K_guess[:-1]))

    K2_2 = K_guess
    w = winit[t:t+length]
    r = rinit[t:t+length]
    B = Binit[t:t+length]
    if length == S:
        error2 = MUc((1 + r)*K1_2 + w * e[
            -(length):, j] * L_guess + (1 + r)*B/bin_weights[j] - np.exp(g_y) * K2_2) * w * e[
            -(length):, j] + MUl2(L_guess, chi_n)
    else:
        error2 = MUc((1 + r)*K1_2 + w * e[
            -(length):, j] * L_guess + (1 + r)*B/bin_weights[j] - np.exp(g_y) * K2_2) * w * e[
            -(length):, j] + MUl2(L_guess, chi_n[-length:])

    error3 = MUc((1 + r[-1])*K_guess[-2] + w[-1] * e[-1, j] * L_guess[-1] + (1 + r[-1])*B[-1]/bin_weights[j] - K_guess[-1] * 
        np.exp(g_y)) - np.exp(-sigma * g_y) * MUb(K_guess[-1])

    # Check and punish constraint violations
    # mask1 = L_guess < 0
    # error2[mask1] += 1e9
    # mask2 = L_guess > ltilde
    # error2[mask2] += 1e9
    # cons = (1 + r) * K1_2 + w * e[
    #     -(length):, j] * L_guess + (1+r)*B/bin_weights[j] - K2_2 * np.exp(g_y)
    # mask3 = cons < 0
    # error2[mask3] += 1e9
    # bin1 = bin_weights[j]
    # b_min = np.zeros(length-1)
    # b_min[-1] = (ctilde + bqtilde - w1[-1] * e1[-1] * ltilde - B1[-1] / bin1) / (1 + r1[-1])
    # for i in xrange(length - 2):
    #     b_min[-(i+2)] = (ctilde + np.exp(
    #         g_y) * b_min[-(i+1)] - w1[-(i+2)] * e1[
    #         -(i+2)] * ltilde - B1[-(i+2)] / bin1) / (1 + r1[-(i+2)])
    # difference = K_guess[:-1] - b_min
    # mask4 = difference < 0
    # error1[mask4] += 1e9
    return list(error1.flatten()) + list(error2.flatten()) + list(error3.flatten())


domain = np.linspace(0, T, T)
Kinit = (-1/(domain + 1)) * (Kss-K0) + Kss
Kinit[-1] = Kss
Kinit = np.array(list(Kinit) + list(np.ones(S)*Kss))
Linit = np.ones(T + S) * Lss
Yinit = A*(Kinit**alpha) * (Linit**(1-alpha))
winit = (1-alpha) * Yinit / Linit
rinit = (alpha * Yinit / Kinit) - delta
Binit = np.zeros((T+S, J))
for j in xrange(J):
    Binit[:, j] = list(np.linspace(B0[j], Bss[j], T)) + [Bss[j]]*S
Binit = np.array(Binit)

TPIiter = 0
TPIdist = 10
print 'Starting time path iteration.'

euler_errors = np.zeros((T, 2*S, J))
TPIdist_vec = np.zeros(TPImaxiter)
nu_current = nu

while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
    # to be deleted starts
    Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)
    Lpath_TPI = list(Linit) + list(np.ones(10)*Lss)
    plt.figure()
    plt.axhline(
        y=Kss, color='black', linewidth=2, label=r"Steady State $\hat{K}$", ls='--')
    plt.plot(np.arange(
        T+10), Kpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path $\hat{K}_t$")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Per-capita Capital $\hat{K}$")
    plt.savefig("OUTPUT/TPI_K")

    plt.figure()
    plt.axhline(
        y=Lss, color='black', linewidth=2, label=r"Steady State $\hat{L}$", ls='--')
    plt.plot(np.arange(
        T+10), Lpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path $\hat{L}_t$")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Per-capita Effective Labor Supply $\hat{L}$")
    plt.savefig("OUTPUT/TPI_L")
    # to be deleted ends
    K_mat = np.zeros((T+S, S, J))
    L_mat = np.zeros((T+S, S, J))
    for j in xrange(J):
        for s in xrange(S-2):  # Upper triangle
            solutions = opt.fsolve(Euler_Error, list(
                initial_K[-(s+2):, j]) + list(initial_L[-(s+2):, j]), args=(
                winit, rinit, Binit[:, j], 0), xtol=1e-10)
            K_vec = solutions[:len(solutions)/2]
            K_mat[1:S+1, :, j] += np.diag(K_vec, S-(s+2))
            L_vec = solutions[len(solutions)/2:]
            L_mat[:S, :, j] += np.diag(L_vec, S-(s+2))

        for t in xrange(0, T):
            solutions = opt.fsolve(Euler_Error, list(
                initial_K[:, j]) + list(initial_L[:, j]), args=(
                winit, rinit, Binit[:, j], t), xtol=1e-10)
            K_vec = solutions[:S]
            K_mat[t+1:t+S+1, :, j] += np.diag(K_vec)
            L_vec = solutions[S:]
            L_mat[t:t+S, :, j] += np.diag(L_vec)
            inputs = list(solutions)
            euler_errors[t, :, j] = np.abs(Euler_Error(inputs, winit, rinit, Binit[:, j], t))

    K_mat[0, :, :] = initial_K
    L_mat[0, -1, :] = initial_L[-1, :]
    Knew = (omega_stationary[:T, :, :] * K_mat[:T, :, :]).sum(2).sum(1)
    Lnew = (omega_stationary[1:T+1, :, :] * e.reshape(
        1, S, J) * L_mat[:T, :, :]).sum(2).sum(1)
    Bnew = (K_mat[:T, :, :] * omega_stationary[:T, :, :] * mort_rate.reshape(1, S, 1)).sum(1)
    TPIiter += 1
    Kinit = nu*Knew + (1-nu)*Kinit[:T]
    Linit = nu*Lnew + (1-nu)*Linit[:T]
    Binit[:T] = nu*Bnew + (1-nu)*Binit[:T]
    TPIdist = np.array(list(
        np.abs(Knew - Kinit)) + list(np.abs(Bnew - Binit[:T]).flatten()) + list(np.abs(Lnew - Linit))).max()
    TPIdist_vec[TPIiter] = TPIdist
    # After T=7, if cycling occurs, drop the value of nu
    # wait til after T=7 or so, because sometimes there is a jump up in the first couple iterations
    if TPIiter > 7:
        if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter-1] > 0:
            nu_current /= 2
            print 'New Value of nu:', nu_current
    print '\tIteration:', TPIiter
    print '\t\tDistance:', TPIdist
    if (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
        Yinit = A*(Kinit**alpha) * (Linit**(1-alpha))
        winit = np.array(
            list((1-alpha) * Yinit / Linit) + list(np.ones(S)*wss))
        rinit = np.array(list((alpha * Yinit / Kinit) - delta) + list(
            np.ones(S)*rss))


Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)
Lpath_TPI = list(Linit) + list(np.ones(10)*Lss)

print 'TPI is finished.'


def borrowing_constraints2(K_dist, w, r, e, B):
    '''
    Parameters:
        K_dist = Distribution of capital ((S-1)xJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        n      = distribution of labor (SxJ array)

    Returns:
        False value if all the borrowing constraints are met, True
            if there are violations.
    '''
    b_min = np.zeros((T, S-1, J))
    for t in xrange(T):
        b_min[t, -1, :] = (
            ctilde + bqtilde - w[S-1+t] * e[S-1, :] * ltilde - B[S-1+t] / bin_weights) / (1 + r[S-1+t])
        for i in xrange(S-2):
            b_min[t, -(i+2), :] = (
                ctilde + np.exp(g_y) * b_min[t, -(i+1), :] - w[S+t-(i+2)] * e[
                    -(i+2), :] * ltilde - B[S+t-(i+2)] / bin_weights) / (1 + r[S+t-(i+2)])
    difference = K_mat[:T, :-1, :] - b_min
    for t in xrange(T):
        if (difference[t, :, :] < 0).any():
            print 'There has been a borrowing constraint violation in period %.f.' % t

K1 = np.zeros((T, S, J))
K1[:, 1:, :] = K_mat[:T, :-1, :]
K2 = np.zeros((T, S, J))
K2[:, :, :] = K_mat[:T, :, :]
cinit = (1 + rinit[:T].reshape(T, 1, 1)) * K1 + winit[:T].reshape(
    T, 1, 1) * e.reshape(1, S, J) * L_mat[:T] - np.exp(g_y) * K2 + (1 + rinit[:T].reshape(T, 1, 1)) * Binit[:T].reshape(T, 1, J) / bin_weights.reshape(1, 1, J)
print'Checking time path for violations of constaints.'
for t in xrange(T):
    constraint_checker2(K_mat[t, :-1, :], L_mat[t], winit[t], rinit[t], e, cinit[t], t)
borrowing_constraints2(K_mat, winit, rinit, e, Binit)
print '\tFinished.'

elapsed_time = time.time() - start_time
hours = elapsed_time / 3600
minutes = (elapsed_time / 60) % 60
seconds = elapsed_time % 60
print 'TPI took %.0f hours, %.0f minutes, and %.0f seconds.' % (
    abs(hours - .5), abs(minutes - .5), seconds)

'''
------------------------------------------------------------------------
Plot Timepath for K and N
------------------------------------------------------------------------
'''

print 'Generating TPI graphs.'

plt.figure()
plt.axhline(
    y=Kss, color='black', linewidth=2, label=r"Steady State $\hat{K}$", ls='--')
plt.plot(np.arange(
    T+10), Kpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path $\hat{K}_t$")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Per-capita Capital $\hat{K}$")
# plt.title(r"Time Path of Capital Stock K$_t$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI_K")

plt.figure()
plt.axhline(
    y=Lss, color='black', linewidth=2, label=r"Steady State $\hat{L}$", ls='--')
plt.plot(np.arange(
    T+10), Lpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path $\hat{L}_t$")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Per-capita Effective Labor Supply $\hat{L}$")
# plt.title(r"Time Path of Labor Supply L$_t$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI_L")

'''
------------------------------------------------------------------------
Compute Plot Euler Errors
------------------------------------------------------------------------
eul1   = results of euler 1
eul2   = results of euler 2
eul3   = results of euler 3
domain     = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''

eul1 = euler_errors[:, :S-1, :].max(1).max(1)
eul2 = euler_errors[:, S-1:, :].max(1).max(1)
eul3 = euler_errors[:, S-1, :].max(1)


domain = np.linspace(1, T, T)
plt.figure()
plt.plot(domain, eul1, label='Euler1')
plt.plot(domain, eul2, label='Euler2')
plt.plot(domain, eul3, label='Euler3')
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.legend(loc=0)
plt.title('Maximum Euler Error for each period across S and J')
plt.savefig('OUTPUT/euler_errors_TPI_2D')

print '\tFinished.'

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

print 'Saving TPI variable values.'
var_names = ['Kpath_TPI', 'TPIiter', 'TPIdist', 'elapsed_time',
             'hours', 'minutes', 'seconds', 'T', 'K_mat']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/TPI_vars.pkl", "w"))
print '\tFinished.'
