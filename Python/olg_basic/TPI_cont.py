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

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

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

variables = pickle.load(open("ss_vars.pkl", "r"))
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

T = 60
K0 = .9*Kss
initial = 0.9*Kssvec[1:]

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


def Euler_Error(K_guess, winit, rinit, t):
    length = len(K_guess)
    K_guess = K_guess.reshape((length, 1))

    if length==S-1:
        K1 = np.array(list(np.zeros(1).reshape((1,1))) + list(K_guess[:-1,:]))
    else:
        K1 = np.array(list((.9*Kssmat.reshape(S-1,J)[-(s+2),j]).reshape(1,1)) + list(K_guess[:-1,:]))
    K2 = K_guess
    K3 = np.array(list(K_guess[1:,:]) + list(np.zeros(1).reshape((1,1))))

    w1 = winit[t:t+length].reshape(length,1)
    w2 = winit[t+1:t+1+length].reshape(length,1)

    r1 = rinit[t:t+length].reshape(length,1)
    r2 = rinit[t+1:t+1+length].reshape(length,1)

    n1 = n[-(length+1):-1].reshape(length,1)
    n2 = n[-length:].reshape(length,1)

    e1 = e[-(length+1):-1,j].reshape((length,1))
    e2 = e[-length:,j].reshape((length,1))

    error = MUc((1 + r1)*K1 + w1 * e1 * n1 - K2) \
    - beta * (1 + r2)*MUc((1 + r2)*K2 + w2*e2*n2 - K3)

    return error.flatten()


def Euler_Error2(K_guess, winit, rinit, e, n):

    K_guess = K_guess.reshape(T+S-1,S-1)

    K1 = np.zeros((T+S-1,S-1))
    K1[0,1:] = .9*Kssmat.reshape(S-1,J)[:-1,i]
    K1[1:,1:] = K_guess[:-1,:-1]

    K2 = K_guess

    K3 = np.zeros((T+S-1,S-1))
    K3[-1,:-1] = Kssmat.reshape(S-1,J)[1:,i]
    K3[:-1,:-1] = K_guess[1:,1:]

    n1 = n[:-1].reshape(1,S-1)
    n2 = n[1:].reshape(1,S-1)

    w1 = winit[:-1].reshape(T+S-1,1)
    w2 = winit[1:].reshape(T+S-1,1)

    r1 = rinit[:-1].reshape(T+S-1,1)
    r2 = rinit[1:].reshape(T+S-1,1)

    e1 = e[:-1,i].reshape(1,S-1)
    e2 = e[1:,i].reshape(1,S-1)

    error = MUc((1 + r1)*K1 + w1 * e1 * n1 - K2) \
        - beta * (1 + r2)*MUc((1 + r2)*K2 + w2*e2*n2 - K3)

    return error.flatten()

zero_func = lambda x: Euler_Error2(x, winit, rinit, e, n)

Kinit = np.array(list(np.linspace(K0, Kss, T)) + list(np.ones(S)*Kss))
Ninit = np.ones(T+S) * Nss
Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
winit = (1-alpha) * (Yinit/Ninit)
rinit = alpha * (Yinit/Kinit) - delta

TPIiter = 0
TPImaxiter = 1
TPIdist = 10
TPImindist = 3.0*10**(-6)
K_mat = np.tile(.9*Kssmat.reshape(1,S-1,J), (T+S,1,1))
while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
    # K_mat = np.ones((T+S, S-1, J)) * 2.0
    
    for j in xrange(J):
        for s in xrange(S-3): # Upper triangle
            K_vec = opt.fsolve(Euler_Error, .9*Kssmat.reshape(S-1,J)[-(s+2):,j], args=(winit, rinit, 0))
            # K_vec = (K_vec.reshape(s+2,J)).mean(1)
            K_mat[1:S,:,j] += np.diag(K_vec, S-(s+3))

        for t in xrange(1,T):
            K_vec = opt.fsolve(Euler_Error, np.diag(K_mat[t:t+S-1, :, j]), args=(winit, rinit, t))
            # K_vec = (K_vec.reshape(S-1, J)).mean(1)
            K_mat[t:t+S-1, :, j] += np.diag(K_vec)
    
    # for i in xrange(J):
    #     K_mat[1:, :, i] = (opt.newton_krylov(zero_func, K_mat[1:, :, i])).reshape(T+S-1,S-1)
    #     print i

    K_mat[0, :, :] = .9*Kssmat.reshape(S-1, J)
    Knew = K_mat[:T, :, :].mean(1).mean(1)
    TPIiter += 1
    TPIdist = (np.abs(Knew - Kinit[:T])).max()
    print 'Iteration:', TPIiter
    print '\tDistance:', TPIdist
    Kinit = rho*Knew + (1-rho)*Kinit[:T]
    Ninit = np.ones(T) * Nss
    Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
    winit = np.array(list((1-alpha) * (Yinit/Ninit)) + list(np.ones(S)*wss))
    rinit = np.array(list(alpha * (Yinit/Kinit) - delta) + list(np.ones(S)*rss))

Kpath_TPI = list(Kinit) + list(np.ones(10)*Kss)

elapsed_time = time.time() - start_time

print "The time path is", Kpath_TPI

hours = elapsed_time / 3600
minutes = (elapsed_time / 60) % 60
seconds = elapsed_time % 60

print 'TPI took %.0f hours, %.0f minutes, and %.0f seconds.' % (
    abs(hours - .5), abs(minutes - .5), seconds)

plt.figure(5)
plt.plot(
    np.arange(T+10), Kpath_TPI[:T+10], 'b', linewidth=2, label="Capital Path")
plt.xlabel("Time")
plt.ylabel("Aggregate Capital")
plt.title("Time Path of Capital Stock")
plt.axhline(y=Kss, color='black', linewidth=2, label="Steady State", ls='--')
plt.legend(loc=0)
plt.savefig("TPI")

var_names = ['Kpath_TPI', 'TPIiter', 'TPIdist', 'elapsed_time',
             'hours', 'minutes', 'seconds', 'T']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("TPI_vars.pkl", "w"))
