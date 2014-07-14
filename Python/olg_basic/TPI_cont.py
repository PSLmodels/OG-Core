'''
------------------------------------------------------------------------
Last updated 7/9/2014
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

T = 30
K0 = .8*Kss

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

def Euler_Error(K_guess, winit, rinit, i):

    K_guess = K_guess.reshape((S-1, J))
    
    K1 = np.array(list(np.zeros(J).reshape((1,J))) + list(K_guess[:-1,:]))
    K2 = K_guess
    K3 = np.array(list(K_guess[1:,:]) + list(np.zeros(J).reshape((1,J))))

    w1 = winit[i:i+S-1].reshape(S-1,1)
    w2 = winit[i+1:i+S].reshape(S-1,1)

    r1 = rinit[i:i+S-1].reshape(S-1,1)
    r2 = rinit[i+1:i+S].reshape(S-1,1)

    n1 = n[:-1].reshape((S-1,1))
    n2 = n[1:].reshape((S-1,1))

    error = MUc((1 + r1)*K1 + w1 * e[:-1,:] * n1 - K2) \
    - beta * (1 + r2)*MUc((1 + r2)*K2 + w2*(e[1:,:].dot(f[1:,:,:])[:,0,:])*n2 - K3)

    return error.flatten()

Kinit = np.array(list(np.linspace(K0, Kss, T)) + list(np.ones(S-2)*Kss))
Ninit = np.ones(T+S-2) * Nss
Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
winit = (1-alpha) * (Yinit/Ninit)
rinit = alpha * (Yinit/Kinit) - delta

TPIiter = 0
TPImaxiter = 500
TPIdist = 10
TPImindist = 3.0*10**(-6)

while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
    K_mat = np.zeros((T+S-1, S-1))
    for i in xrange(T):
        K_vec = opt.fsolve(Euler_Error, np.ones((S-1, J)), args=(winit, rinit, i))
        K_vec = (K_vec.reshape(S-1, J)).mean(1)
        K_mat[i:i+S-1, :] += np.diag(K_vec)
        print i
    Knew = K_mat[:T,:].sum(1)
    TPIiter += 1
    TPIdist = (np.abs(Knew - Kinit[:T])).max()
    print 'Iteration:', TPIiter
    print '\tDistance:', TPIdist
    Kinit = rho_TPI*Knew + (1-rho_TPI)*Kinit
    Ninit = np.ones(T+S-2) * Nss
    Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
    winit = (1-alpha) * (Yinit/Ninit)
    rinit = alpha * (Yinit/Kinit) - delta

Kpath_TPI = Kinit

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
