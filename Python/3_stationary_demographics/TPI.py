'''
------------------------------------------------------------------------
Last updated 7/29/2014

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
delta    = decreciation rate of capital
ltilde   = measure of time each individual is endowed with each period
ctilde   = minimum value of consumption
chi      = discount factor
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


def get_N(children, omega):
    N = children.sum(1).sum(1) + omega.sum(1).sum(1)
    return N

N = get_N(children, omega)
omega_stationary = omega / N.reshape(T, 1, 1)


def constraint_checker1(k_dist, l_dist, w, r, e, c_dist):
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
    if k_dist.sum() / N[-1] <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to zero.'
        flag1 = True
    if borrowing_constraints(k_dist, w, r, e, l_dist) is True:
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
        print '\tWARNING: Conusmption volates nonnegativity constraints.'
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
    if k_dist.sum() / N[t] <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to ' \
            'zero in period %.f.' % t
    if (l_dist < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints ' \
            'in period %.f.' % t
    if (l_dist > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint in '\
            'period %.f.' % t
    if (c_dist < 0).any():
        print '\tWARNING: Conusmption volates nonnegativity constraints in ' \
            'period %.f.' % t


def borrowing_constraints(K_dist, w, r, e, n):
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
    b_min[-1, :] = (ctilde - w * e[S-1, :] * ltilde) / (1 + r)
    for i in xrange(S-2):
        b_min[-(i+2), :] = (ctilde + np.exp(g_y) * b_min[-(i+1), :] - w * e[
            -(i+2), :] * ltilde) / (1 + r)
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


def MUl_2(n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    output = - chi * ((ltilde-n) ** (-eta))
    return output


initial_K = Kssmat * .95
K0 = (omega_stationary[0, 1:, :] * initial_K).sum()
K1_2init = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_K))
K2_2init = np.array(list(initial_K) + list(np.zeros(J).reshape(1, J)))
initial_L = Lssmat
L0 = get_L(e, initial_L)
Y0 = get_Y(K0, L0)
w0 = get_w(Y0, L0)
r0 = get_r(Y0, K0)
c0 = (1 + r0) * K1_2init + w0 * e * initial_L - K2_2init * np.exp(g_y)
constraint_checker1(initial_K, initial_L, w0, r0, e, c0)
print 'K0 divided by Kss =', K0/Kss

'''
------------------------------------------------------------------------
Solve for equilibrium transition path by TPI
------------------------------------------------------------------------
Kinit        = 1 x T vector, initial time path of aggregate capital
               stock
Linit        = 1 x T vector, initial time path of aggregate labor
               demand. This is just equal to a 1 x T vector of Lss
               because labor is supplied inelastically
Yinit        = 1 x T vector, initial time path of aggregate output
winit        = 1 x T vector, initial time path of real wage
rinit        = 1 x T vector, initial time path of real interest rate
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
------------------------------------------------------------------------
'''


def Euler1(w1, r1, w2, r2, e, n, K1, K2, K3):
    '''
    Parameters:
        w  = wage rate (scalar)
        r  = rental rate (scalar)
        e  = distribution of abilities (SxJ array)
        n  = distribution of labor (Sx1 vector)
        K1 = distribution of capital in period t ((S-1) x J array)
        K2 = distribution of capital in period t+1 ((S-1) x J array)
        K3 = distribution of capital in period t+2 ((S-1) x J array)

    Returns:
        Value of Euler error.
    '''
    euler = MUc(
        (1 + r1)*K1 + w1 * e[:-1, :] * n[:-1, :] - np.exp(g_y) * K2) - beta * np.exp(-sigma * g_y) * (1 + r2)*MUc(
        (1 + r2)*K2 + w2 * e[1:, :] * n[1:, :] - np.exp(g_y) * K3)
    return euler


def Euler2(w, r, e, L_guess, K1_2, K2_2):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K1_2     = distribution of capital in period t (S x J array)
        K2_2     = distribution of capital in period t+1 (S x J array)

    Returns:
        Value of Euler error.
    '''
    euler = MUc((1 + r)*K1_2 + w * e * L_guess - np.exp(g_y) * K2_2) * w * e + MUl_2(L_guess)
    return euler


def Euler_Error(guesses, winit, rinit, t):
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
    K_guess = guesses[:length]
    L_guess = guesses[length:]

    if length == S-1:
        K1 = np.array([0] + list(K_guess[:-1]))
    else:
        K1 = np.array([(initial_K[-(s+2), j])] + list(K_guess[:-1]))
    K2 = K_guess
    K3 = np.array(list(K_guess[1:]) + [0])
    w1 = winit[t:t+length]
    w2 = winit[t+1:t+1+length]
    r1 = rinit[t:t+length]
    r2 = rinit[t+1:t+1+length]
    l1 = L_guess[:-1]
    l2 = L_guess[1:]
    e1 = e[-(length+1):-1, j]
    e2 = e[-length:, j]
    error1 = MUc((1 + r1)*K1 + w1 * e1 * l1 - np.exp(g_y) * K2) \
        - beta * np.exp(-sigma * g_y) * (1 + r2)*MUc((1 + r2)*K2 + w2*e2*l2 - np.exp(g_y) * K3)

    if length == S-1:
        K1_2 = np.array([0] + list(K_guess))
    else:
        K1_2 = np.array([(initial_K[-(s+2), j])] + list(K_guess))

    K2_2 = np.array(list(K_guess) + [0])
    w = winit[t:t+length+1]
    r = rinit[t:t+length+1]
    error2 = MUc((1 + r)*K1_2 + w * e[
        -(length+1):, j] * L_guess - np.exp(g_y) * K2_2) * w * e[
        -(length+1):, j] + MUl_2(L_guess)
    # Check and punish constraing violations
    mask1 = L_guess < 0
    error2[mask1] += 1e9
    mask2 = L_guess > ltilde
    error2[mask2] += 1e9
    if K_guess.sum() <= 0:
        error1 += 1e9
    cons = (1 + r) * K1_2 + w * e[-(length+1):, j] * L_guess - K2_2 * np.exp(g_y)
    mask3 = cons < 0
    error2[mask3] += 1e9

    b_min = np.zeros(length)
    b_min[-1] = (ctilde - w1[-1] * e1[-1] * ltilde) / (1 + r1[-1])
    for i in xrange(length - 1):
        b_min[-(i+2)] = (ctilde + np.exp(g_y) * b_min[-(i+1)] - w1[-(i+2)] * e1[
            -(i+2)] * ltilde) / (1 + r1[-(i+2)])
    difference = K_guess - b_min
    mask4 = difference < 0
    error1[mask4] += 1e9

    return list(error1.flatten()) + list(error2.flatten())


domain = np.linspace(0, T, T)
Kinit = (-1/(domain + 1)) * (Kss-K0) + Kss
Kinit[-1] = Kss
Kinit = np.array(list(Kinit) + list(np.ones(S)*Kss))
Linit = (1/(domain+1)) * (L0-Lss) + Lss
Linit[-1] = Lss
Linit = np.array(list(Linit) + list(np.ones(S)*Lss))
Yinit = A*(Kinit**alpha) * (Linit**(1-alpha))
winit = (1-alpha) * Yinit / Linit
rinit = (alpha * Yinit / Kinit) - delta

TPIiter = 0
TPIdist = 10
print 'Starting time path iteration.'

while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
    K_mat = np.zeros((T+S, S-1, J))
    L_mat = np.zeros((T+S, S, J))
    for j in xrange(J):
        for s in xrange(S-2):  # Upper triangle
            solutions = opt.fsolve(Euler_Error, list(
                initial_K[-(s+1):, j]) + list(initial_L[-(s+2):, j]), args=(
                winit, rinit, 0))
            K_vec = solutions[:len(solutions)/2]
            K_mat[1:S, :, j] += np.diag(K_vec, S-(s+2))
            L_vec = solutions[len(solutions)/2:]
            L_mat[:S, :, j] += np.diag(L_vec, S-(s+2))

        for t in xrange(0, T):
            solutions = opt.fsolve(Euler_Error, list(
                initial_K[:, j]) + list(initial_L[:, j]), args=(
                winit, rinit, t))
            K_vec = solutions[:S-1]
            K_mat[t+1:t+S, :, j] += np.diag(K_vec)
            L_vec = solutions[S-1:]
            L_mat[t:t+S, :, j] += np.diag(L_vec)

    K_mat[0, :, :] = initial_K
    L_mat[0, -1, :] = initial_L[-1, :]
    Knew = (omega_stationary[:, 1:,:] * K_mat[:T, :, :]).sum(2).sum(1)
    Lnew = (omega_stationary[:, :, :] * e.reshape(1, S, J) * L_mat[:T, :, :]).sum(2).sum(1)
    TPIiter += 1
    Kinit = nu*Knew + (1-nu)*Kinit[:T]
    Linit = nu*Lnew + (1-nu)*Linit[:T]
    TPIdist = np.array(list(np.abs(Knew - Kinit)) + list(np.abs(Lnew - Linit))).max()
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


def borrowing_constraints2(K_dist, w, r, e):
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
        b_min[t, -1, :] = (ctilde - w[S-1+t] * e[S-1, :] * ltilde) / (1 + r[S-1+t])
        for i in xrange(S-2):
            b_min[t, -(i+2), :] = (ctilde + np.exp(g_y) * b_min[t, -(i+1), :] - w[S+t-(i+2)] * e[
                -(i+2), :] * ltilde) / (1 + r[S+t-(i+2)])
    difference = K_mat[:T, :, :] - b_min
    for t in xrange(T):
        if (difference[t, :, :] < 0).any():
            print 'There has been a borrowing constraint violation in period %.f.' % t

K1 = np.zeros((T, S, J))
K1[:, 1:, :] = K_mat[:T, :, :] / (np.exp(g_y * np.arange(T).reshape(T, 1, 1)))
K2 = np.zeros((T, S, J))
K2[:, :-1, :] = K_mat[:T, :, :] / (np.exp(g_y * np.arange(T).reshape(T, 1, 1)))
cinit = (1 + rinit[:T].reshape(T, 1, 1)) * K1 + winit[:T].reshape(
    T, 1, 1) * e.reshape(1, S, J) * L_mat[:T] - np.exp(g_y) * K2
print'Checking time path for violations of constaints.'
for t in xrange(T):
    constraint_checker2(K_mat[t], L_mat[t], winit[t], rinit[t], e, cinit[t], t)
borrowing_constraints2(K_mat, winit, rinit, e)
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
k1         = Tx(S-1)xJ array of Kssmat in period t-1
k2         = copy of K_mat through period T-1
k3         = Tx(S-1)xJ array of Kssmat in period t+1
k1_2       = TxSxJ array of Kssmat in period t
k2_2       = TxSxJ array of Kssmat in period t+1
euler_mat1 = Tx(S-1)xJ arry of euler errors across time, age, and
              ability level for first Euler equation
euler_mat2 = TxSxJ arry of euler errors across time, age, and
              ability level for second Euler equation
domain     = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''
k1 = np.zeros((T, S-1, J))
k1[:, 1:, :] = K_mat[:T, :-1, :]  / (np.exp(g_y * np.arange(T).reshape(T, 1, 1)))
k2 = K_mat[:T, :, :] / (np.exp(g_y * np.arange(T).reshape(T, 1, 1)))
k3 = np.zeros((T, S-1, J))
k3[:, :-1, :] = K_mat[:T, 1:, :] / (np.exp(g_y * np.arange(T).reshape(T, 1, 1)))
k1_2 = np.zeros((T, S, J))
k1_2[:, 1:, :] = K_mat[:T, :, :] / (np.exp(g_y * np.arange(T).reshape(T, 1, 1)))
k2_2 = np.zeros((T, S, J))
k2_2[:, :-1, :] = K_mat[:T, :, :] / (np.exp(g_y * np.arange(T).reshape(T, 1, 1)))
euler_mat1 = np.zeros((T, S-1, J))
euler_mat2 = np.zeros((T, S, J))

for t in xrange(T):
    euler_mat1[t, :, :] = Euler1(
        winit[t], rinit[t], winit[t+1], rinit[t+1], e, L_mat[t], k1[t], k2[
            t], k3[t])
    euler_mat2[t] = Euler2(winit[t], rinit[t], e, L_mat[t], k1_2[t], k2_2[t])

domain = np.linspace(1, T, T)
plt.figure()
plt.plot(domain, np.abs(euler_mat1).max(1).max(1), label='Euler1')
plt.plot(domain, np.abs(euler_mat2).max(1).max(1), label='Euler2')
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
