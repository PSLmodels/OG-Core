'''
------------------------------------------------------------------------
Last updated 7/15/2014
Python version of Evans/Philips 2014 paper

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method

This py-file calls the following other file(s):
            ss_vars.pkl

This py-file creates the following other file(s):
            TPI_vars.pkl
            TPI.png
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
rho      = contraction parameter in steady state iteration process
           representing the weight on the new distribution gamma_new
A        = total factor productivity parameter in firms' production
           function
delta    = decreciation rate of capital
n        = 1 x S vector of inelastic labor supply for each age s
e        = S x J matrix of age dependent possible working abilities e_s
f        = S x J x J matrix of age dependent discrete probability mass
           function for e as Markov proccess: f(e_s)
J        = number of points in the support of e
Kss      = steady state aggregate capital stock: scalar
Nss      = steady state aggregate labor: scalar
Yss      = steady state aggregate output: scalar
wss      = steady state real wage: scalar
rss      = steady state real rental rate: scalar
runtime   = total time (in seconds) that the steady state solver took to
            run
hours     = total hours that the steady state solver took to run
minutes   = total minutes (minus the total hours) that the steady state
            solver took to run
seconds   = total seconds (minus the total hours and minutes) that the
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
K0      = initial aggregate capital stock
------------------------------------------------------------------------
'''

T = 70
initial = .9 * Kssmat.reshape(S-1, J)
K0 = initial.mean()

'''
------------------------------------------------------------------------
Solve for equilibrium transition path by TPI
------------------------------------------------------------------------
Kinit   = 1 x T vector, initial time path of aggregate capital stock
Ninit   = 1 x T vector, initial time path of aggregate labor demand.
          This is just equal to a 1 x T vector of Nss because labor is
          supplied inelastically
Yinit   = 1 x T vector, initial time path of aggregate output
winit   = 1 x T vector, initial time path of real wage
rinit   = 1 x T vector, initial time path of real interest rate
------------------------------------------------------------------------
'''


def MUc(c):

    """
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    """

    return c**(-sigma)

def MUn(n):
    return - chi * ((ltilde-n)**(-eta))


def Euler_justcapital(w1, r1, w2, r2, e, n, K1, K2, K3):
    euler = MUc((1 + r1)*K1 + w1 * e[:-1, :] * n[:-1].reshape(
        S-1, 1) - K2) - beta * (1 + r2)*MUc(
        (1 + r2)*K2 + w2 * e[1:, :] * n[1:].reshape(S-1, 1) - K3)
    return euler


def Euler_Error(guesses, winit, rinit, t):
    length = len(guesses)/2
    K_guess = guesses[:length]
    N_guess = guesses[length:]

    if length == S-1:
        K1 = np.array([0] + list(K_guess[:-1]))
    else:
        K1 = np.array([(initial[-(s+2), j])] + list(K_guess[:-1]))
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
        K1_2 = np.array([(initial[-(s+2), j])] + list(K_guess))
    K2_2 = np.array(list(K_guess) + [0])
    w = winit[t:t+length+1]
    r = rinit[t:t+length+1]
    error2 = MUc((1 + r)*K1_2 + w * e[-(length+1):,j] * N_guess - K2_2) * w * e[-(length+1):,j] + MUn(N_guess)
    mask = N_guess < 0
    error2[mask] += 1e3
    return list(error1.flatten()) + list(error2.flatten())


def check_agg_K(K_matrix):
    if (K_matrix.sum() <= 0).any():
        print 'WARNING: Aggregate capital stock is less than or' \
            ' equal to zero.'


Kinit = np.array(list(np.linspace(K0, Kss, T)) + list(np.ones(S)*Kss))
Ninit = np.ones(T+S) * Nss
Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
winit = (1-alpha) * (Yinit/Ninit)
rinit = alpha * (Yinit/Kinit) - delta

TPIiter = 0
TPImaxiter = 100
TPIdist = 10
TPImindist = 3 * 1e-6

while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
    K_mat = np.zeros((T+S, S-1, J))
    N_mat = np.zeros((T+S, S, J))
    for j in xrange(J):
        for s in xrange(S-2):  # Upper triangle
            solutions = opt.fsolve(Euler_Error, list(.9*Kssmat.reshape(S-1, J)[-(s+1):, j]) + list(.9*Nssmat.reshape(S, J)[-(s+2):, j]), args=(winit, rinit, 0))
            K_vec = solutions[:len(solutions)/2]
            K_mat[1:S, :, j] += np.diag(K_vec, S-(s+2))
            N_vec = solutions[len(solutions)/2:]
            N_mat[:S, :, j] += np.diag(N_vec, S-(s+2))

        for t in xrange(1, T-1):
            solutions = opt.fsolve(Euler_Error, list(.9*Kssmat.reshape(S-1, J)[:, j]) + list(.9*Nssmat.reshape(S, J)[:, j]), args=(winit, rinit, t))
            K_vec = solutions[:S-1]
            K_mat[t:t+S-1, :, j] += np.diag(K_vec)
            N_vec = solutions[S-1:]
            N_mat[t:t+S, :, j] += np.diag(N_vec)

    K_mat[0, :, :] = initial
    K_mat[T-1, :, :] = Kssmat.reshape(S-1, J)
    Knew = K_mat[:T, :, :].mean(2).mean(1)
    TPIiter += 1
    Kinit = rho*Knew + (1-rho)*Kinit[:T]
    TPIdist = (np.abs(Knew - Kinit[:T])).max()
    print 'Iteration:', TPIiter
    print '\tDistance:', TPIdist
    Ninit = N_mat[:T,:,:].mean(2).mean(1)
    Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
    winit = np.array(list((1-alpha) * (Yinit/Ninit)) + list(np.ones(S)*wss))
    rinit = np.array(list(alpha * (Yinit/Kinit) - delta) + list(
        np.ones(S)*rss))


Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)

print 'TPI is finished.'
check_agg_K(K_mat)
# print "The time path is", np.array(Kpath_TPI)

elapsed_time = time.time() - start_time
hours = elapsed_time / 3600
minutes = (elapsed_time / 60) % 60
seconds = elapsed_time % 60
print 'TPI took %.0f hours, %.0f minutes, and %.0f seconds.' % (
    abs(hours - .5), abs(minutes - .5), seconds)

'''
------------------------------------------------------------------------
Plot Timepath
------------------------------------------------------------------------
'''

plt.figure(7)
plt.axhline(y=Kss, color='black', linewidth=2, label="Steady State K", ls='--')
plt.plot(
    np.arange(T+10), Kpath_TPI[:T+10], 'b', linewidth=2, label=r"TPI time path K$_t$")
plt.xlabel("Time t")
plt.ylabel("Aggregate Capital K")
plt.title(r"Time Path of Capital Stock K$_t$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI")

'''
------------------------------------------------------------------------
Compute Plot Euler Errors
------------------------------------------------------------------------
'''
k1 = np.zeros((T, S-1, J))
k1[:, 1:, :] = K_mat[:T, :-1, :]
k2 = K_mat[:T, :, :]
k3 = np.zeros((T, S-1, J))
k3[:, :-1, :] = K_mat[:T, 1:, :]
euler_mat = np.zeros((T, S-1, J))

for t in xrange(T):
    euler_mat[t, :, :] = Euler_justcapital(winit[t], rinit[t], winit[t+1], rinit[t+1], e, n, k1[t, :, :], k2[t, :, :], k3[t, :, :])

domain = np.linspace(1, T, T)
plt.figure(6)
plt.plot(domain, np.abs(euler_mat).max(1).max(1))
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.title('Maximum Euler Error for each period across S and J')
plt.savefig('OUTPUT/euler_errors_TPI')


'''
------------------------------------------------------------------------
Pickle variables
------------------------------------------------------------------------
'''

var_names = ['Kpath_TPI', 'TPIiter', 'TPIdist', 'elapsed_time',
             'hours', 'minutes', 'seconds', 'T', 'K_mat']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/TPI_vars.pkl", "w"))
