'''
------------------------------------------------------------------------
Last updated 7/18/2014
Python version of Evans/Philips 2014 paper

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method, where labor in inelastically supplied.

This py-file calls the following other file(s):
            ss_vars.pkl

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/euler_errors_TPI_2D.png
            OUTPUT/euler_errors_TPI_3D.png
            OUTPUT/TPI_vars.pkl
            OUTPUT/TPI.png
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
rho      = contraction parameter in steady state iteration process
           representing the weight on the new distribution gamma_new
A        = total factor productivity parameter in firms' production
           function
delta    = decreciation rate of capital
n        = 1 x S vector of inelastic labor supply for each age s
e        = S x J matrix of age dependent possible working abilities e_s
J        = number of points in the support of e
Kss      = steady state aggregate capital stock: scalar
Nss      = steady state aggregate labor: scalar
Yss      = steady state aggregate output: scalar
wss      = steady state real wage: scalar
rss      = steady state real rental rate: scalar
K_agg    = Aggregate level of capital: scalar
cssmat   = SxJ array of consumption across age and ability groups
ctilde   = minimum value of consumption
runtime  = total time (in seconds) that the steady state solver took to
            run
hours    = total hours that the steady state solver took to run
minutes  = total minutes (minus the total hours) that the steady state
            solver took to run
seconds  = total seconds (minus the total hours and minutes) that the
            steady state solver took to run
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/ss_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

start_time = time.time()  # Start timer

'''
------------------------------------------------------------------------
Set other parameters and objects
------------------------------------------------------------------------
T       = number of periods until the steady state
initial = (S-1)xJ array of the initial distribution of capital for TPI
K0      = initial aggregate capital stock
problem = boolean, true if 'initial' does not fulfill the borrowing
          constraints, true otherwise.
------------------------------------------------------------------------
'''


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
    b_min[-1, :] = (ctilde - w * e[S-1, :] * n[S-1, :]) / (1 + r)
    for i in xrange(S-2):
        b_min[-(i+2), :] = (ctilde + b_min[-(i+1), :] - w * e[-(i+2), :] * n[-(i+2), :]) / (1 + r)
    difference = K_dist - b_min
    if (difference < 0).any():
        return True
    else:
        return False


def get_Y(K_now, N_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = A * (K_now ** alpha) * (N_now ** (1 - alpha))
    return Y_now


def get_w(Y_now, N_now):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    w_now = (1 - alpha) * Y_now / N_now
    return w_now


def get_r(Y_now, K_now):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_N(e, n):
    '''
    Parameters: e, n

    Returns:    Aggregate labor
    '''
    N_now = np.mean(e * n)
    return N_now


def MUc(c):

    """
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    """

    return c**(-sigma)


def MUn(n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    output = - chi * ((ltilde-n) ** (-eta))
    return output


def get_N_init(e, N_guess, K1_2, K2_2):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        N_guess  = distribution of labor (SxJ array)
        K1_2     = distribution of capital in period t (S x J array)
        K2_2     = distribution of capital in period t+1 (S x J array)

    Returns:
        Value of Euler error.
    '''
    N_guess = N_guess.reshape(S, J)
    K = K2_2[:-1, :].mean()
    N = get_N(e, N_guess)
    Y = get_Y(K, N)
    w = get_w(Y, N)
    r = get_r(Y, K)
    euler = MUc((1 + r)*K1_2 + w * e * N_guess - K2_2) * w * e + MUn(N_guess)
    euler = euler.flatten()
    return euler


T = 70
# r = (np.random.rand(S-1,J) + .5) * .2
initial_K = .9*Kssmat
K0 = initial_K.mean()

K1_2init = np.array(list(np.zeros(J).reshape(1, J)) + list(initial_K))
K2_2init = np.array(list(initial_K) + list(np.zeros(J).reshape(1, J)))
initial_N_guess = .9*Nssmat.flatten()
get_N_init_zero = lambda x: get_N_init(e, x, K1_2init, K2_2init)
initial_N = opt.fsolve(get_N_init_zero, initial_N_guess).reshape(S, J)
N0 = get_N(e, initial_N)

problem = borrowing_constraints(initial_K, wss, rss, e, Nssmat)
if problem is True:
    print 'The initial distribution does not fulfill the' \
        ' borrowing constraints.'
else:
    print 'The initial distribution fulfills the borrowing constraints.'

'''
------------------------------------------------------------------------
Solve for equilibrium transition path by TPI
------------------------------------------------------------------------
Kinit        = 1 x T vector, initial time path of aggregate capital
               stock
Ninit        = 1 x T vector, initial time path of aggregate labor
               demand. This is just equal to a 1 x T vector of Nss
               because labor is supplied inelastically
Yinit        = 1 x T vector, initial time path of aggregate output
winit        = 1 x T vector, initial time path of real wage
rinit        = 1 x T vector, initial time path of real interest rate
TPIiter      = Iterations of TPI
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPIdist      = Current distance between iterations of TPI
TPImindist   = Cut-off distance between iterations for TPI
K_mat        = (T+S)x(S-1)xJ array of distribution of capital across
               time, age, and ability
Knew         = 1 x T vector, new time path of aggregate capital stock
Kpath_TPI    = 1 x T vector, final time path of aggregate capital stock
flag         = boolean, false if no borrowing constraints are violated
               and aggregate capital is positve, true otherwise
problem      = boolean, true if borrowing constraints are violated,
               false otherwise
problem2     = boolean, true if aggregate capital is negative, false
               otherwise
elapsed_time = elapsed time of TPI
hours        = Hours needed to find the steady state
minutes      = Minutes needed to find the steady state, less the number
               of hours
seconds      = Seconds needed to find the steady state, less the number
               of hours and minutes
------------------------------------------------------------------------
'''


def Euler_justcapital(w1, r1, w2, r2, e, n, K1, K2, K3):
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
    euler = MUc((1 + r1)*K1 + w1 * e[:-1, :] * n[:-1, :] - K2) - beta * (1 + r2)*MUc(
        (1 + r2)*K2 + w2 * e[1:, :] * n[1:, :] - K3)
    return euler


def Euler_Error(guesses, winit, rinit, t):
    '''
    Parameters:
        K_guess = distribution of capital in period t ((S-1) x J array)
        w  = wage rate (scalar)
        r  = rental rate (scalar)
        t  = time period

    Returns:
        Value of Euler error.
    '''
    length = len(guesses)/2
    K_guess = guesses[:length]
    N_guess = guesses[length:]

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
    n1 = N_guess[:-1]
    n2 = N_guess[1:]
    e1 = e[-(length+1):-1, j]
    e2 = e[-length:, j]
    error1 = MUc((1 + r1)*K1 + w1 * e1 * n1 - K2) \
        - beta * (1 + r2)*MUc((1 + r2)*K2 + w2*e2*n2 - K3)

    if length == S-1:
        K1_2 = np.array([0] + list(K_guess))
    else:
        K1_2 = np.array([(initial_K[-(s+2), j])] + list(K_guess))
        
    K2_2 = np.array(list(K_guess) + [0])
    w = winit[t:t+length+1]
    r = rinit[t:t+length+1]
    error2 = MUc((1 + r)*K1_2 + w * e[-(length+1):,j] * N_guess - K2_2) * w * e[-(length+1):,j] + MUn(N_guess)
    mask = N_guess < 0
    error2[mask] += 1e9
    return list(error1.flatten()) + list(error2.flatten())
 

def check_agg_K(K_matrix):
    if (K_matrix.sum() <= 0).any():
        return True
    else:
        return False

Kinit = np.array(list(np.linspace(K0, Kss, T)) + list(np.ones(S)*Kss))
Ninit = np.array(list(np.linspace(N0, Nss, T)) + list(np.ones(S)*Nss))
Yinit = A*(Kinit**alpha) * (Ninit**(1-alpha))
winit = (1-alpha) * Yinit / Ninit
rinit = (alpha * Yinit / Kinit) - delta


TPIiter = 0
TPImaxiter = 100
TPIdist = 10
TPImindist = 3 * 1e-6
print 'TPI has started.\n'

while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
    K_mat = np.zeros((T+S, S-1, J))
    N_mat = np.zeros((T+S, S, J))
    for j in xrange(J):
        for s in xrange(S-2):  # Upper triangle
            solutions = opt.fsolve(Euler_Error, list(initial_K[-(s+1):, j]) + list(initial_N[-(s+2):, j]), args=(winit, rinit, 0))
            K_vec = solutions[:len(solutions)/2]
            K_mat[1:S, :, j] += np.diag(K_vec, S-(s+2))
            N_vec = solutions[len(solutions)/2:]
            N_mat[:S, :, j] += np.diag(N_vec, S-(s+2))

        for t in xrange(0, T):
            solutions = opt.fsolve(Euler_Error, list(initial_K[:, j]) + list(initial_N[:, j]), args=(winit, rinit, t))
            K_vec = solutions[:S-1]
            K_mat[t+1:t+S, :, j] += np.diag(K_vec)
            N_vec = solutions[S-1:]
            N_mat[t:t+S, :, j] += np.diag(N_vec)

    K_mat[0, :, :] = initial_K
    # K_mat[T-1, :, :] = Kssmat.reshape(S-1, J)
    N_mat[0, -1, :] = initial_N[-1,:]
    # N_mat[T-1, :, :] = Nssmat.reshape(S, J)
    Knew = K_mat[:T, :, :].mean(2).mean(1)
    Nnew = (e.reshape(1,S,J) * N_mat[:T,:,:]).mean(2).mean(1)
    TPIiter += 1
    Kinit = rho*Knew + (1-rho)*Kinit[:T]
    Ninit = rho*Nnew + (1-rho)*Ninit[:T]
    TPIdist = (np.abs(Knew - Kinit)).max() + (np.abs(Nnew - Ninit)).max()
    print 'Iteration:', TPIiter
    print '\tDistance:', TPIdist
    if (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
        Yinit = A*(Kinit**alpha) * (Ninit**(1-alpha))
        winit = np.array(list((1-alpha) * Yinit / Ninit) + list(np.ones(S)*wss))
        rinit = np.array(list((alpha * Yinit / Kinit) - delta) + list(
            np.ones(S)*rss))


Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)
Npath_TPI = list(Ninit) + list(np.ones(10)*Nss)

print '\nTPI is finished.'
flag = False
for t in xrange(T):
    problem = borrowing_constraints(K_mat[t], winit[t], rinit[t], e, Nssmat)
    if problem is True:
        print 'There is a violation in the borrowing constraints' \
            ' in period %.f.' % t
        flag = True
    problem2 = check_agg_K(K_mat[t])
    if problem2 is True:
        print 'WARNING: Aggregate capital stock is less than or' \
            ' equal to zero in period %.f.' % t
        flag = True
if flag is False:
    print 'There were no violations of the borrowing constraints' \
        ' in any period.'

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

plt.figure(13)
plt.axhline(
    y=Kss, color='black', linewidth=2, label="Steady State K", ls='--')
plt.plot(np.arange(
    T+10), Kpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path K$_t$")
plt.xlabel("Time t")
plt.ylabel("Aggregate Capital K")
plt.title(r"Time Path of Capital Stock K$_t$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI_K")

plt.figure(14)
plt.axhline(
    y=Nss, color='black', linewidth=2, label="Steady State N", ls='--')
plt.plot(np.arange(
    T+10), Npath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path N$_t$")
plt.xlabel("Time t")
plt.ylabel("Aggregate Labor N")
plt.title(r"Time Path of Labor Supply N$_t$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI_N")


'''
------------------------------------------------------------------------
Compute Plot Euler Errors
------------------------------------------------------------------------
k1          = Tx(S-1)xJ array of Kssmat in period t-1
k2          = copy of K_mat through period T-1
k3          = Tx(S-1)xJ array of Kssmat in period t+1
euler_mat   = Tx(S-1)xJ arry of euler errors across time, age, and
              ability level
domain      = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''
k1 = np.zeros((T, S-1, J))
k1[:, 1:, :] = K_mat[:T, :-1, :]
k2 = K_mat[:T, :, :]
k3 = np.zeros((T, S-1, J))
k3[:, :-1, :] = K_mat[:T, 1:, :]
euler_mat = np.zeros((T, S-1, J))

for t in xrange(T):
    euler_mat[t, :, :] = Euler_justcapital(
        winit[t], rinit[t], winit[t+1], rinit[t+1], e, Nssmat, k1[t, :, :], k2[
            t, :, :], k3[t, :, :])

domain = np.linspace(1, T, T)
plt.figure(15)
plt.plot(domain, np.abs(euler_mat).max(1).max(1))
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.title('Maximum Euler Error for each period across S and J')
plt.savefig('OUTPUT/euler_errors_TPI_2D')

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

var_names = ['Kpath_TPI', 'TPIiter', 'TPIdist', 'elapsed_time',
             'hours', 'minutes', 'seconds', 'T', 'K_mat']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/TPI_vars.pkl", "w"))
