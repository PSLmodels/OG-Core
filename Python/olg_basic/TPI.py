'''
------------------------------------------------------------------------
Last updated 6/25/2014
Python version of Evans/Philips 2014 paper

This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method

This py-file calls the following other file(s):
            ss_vars.pkl
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

'''
------------------------------------------------------------------------
Import steady state distribution, parameters and other objects from
steady state computation in OLGabilss.m
------------------------------------------------------------------------
S        = number of periods an individual lives
beta     = discount factor (0.96 per year)
sigma    = coefficient of relative risk aversion
alpha    = capital share of income
rho      = contraction parameter in steady state iteration process
           representing the weight on the new distribution gamma_new
A        = total factor productivity parameter in firms' production
           function
n        = 1 x S vector of inelastic labor supply for each age s
e        = S x J matrix of age dependent possible working abilities e_s
f        = S x J matrix of age dependent discrete probability mass
           function for e: f(e_s)
J        = number of points in the support of e
bmin     = minimum value of b (usually is bmin = 0)
bmax     = maximum value of b
bsize    = scalar, number of discrete points in the support of b
b        = 1 x bsize vector of possible values for initial wealth b and
           savings b'
gamma_ss = (S-1) x J x bsize array of steady state distribution of
           wealth
Kss      = steady state aggregate capital stock: scalar
Nss      = steady state aggregate labor: scalar
Yss      = steady state aggregate output: scalar
wss      = steady state real wage: scalar
rss      = steady state real rental rate: scalar
phiind_ss = S x J x bsize steady-state policy function indicies for
            b' = phi(s,e,b). The last row phiind(S,e,b) is ones and
            corresponds to b_{S+1}. The first row phi(1,e,b) corresponds
            to b_2.
phi_ss    = S x J x bsize steady-state policy function values for
            b' = phi(s,e,b). The last row phi(S,e,b) is zeros and
            corresponds to b_{S+1}. The first row corresponds to b_2
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
gamma0  = (S-1) x J x bsize array, initial distribution of wealth
K0      = initial aggregate capital stock as a function of the initial
          distribution of wealth
rho_TPI = contraction parameter in TPI process representing weight on
          new time path of aggregate capital stock
------------------------------------------------------------------------
'''

T = 60
I0 = np.identity(bsize)
I0row = I0[bsize/2 - S/3, :]
gamma0 = np.tile(I0row.reshape(1, 1, bsize), (
    S-1, J, 1)) * np.tile(f[:S-1, :].reshape(S-1, J, 1), (1, 1, bsize)) / (S-1)
K0 = (float(S-1)/S) * (gamma0 * np.tile(b.reshape(1, 1, bsize), (
    S-1, J, 1))).sum()
rho_TPI = 0.2

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
gammat  = (S-1) x J x bsize x T array time path of the distribution of
          capital
phiindt = S x J x bsize x T array time path of policy function indicies
          for b' = phi(s,e,b,t). The last row phiindt(S,e,b,t) is ones
          and corresponds to b_{S+1}=0. The first row phi(1,e,b,t)
          corresponds to b_2.
phit    = S x J x bsize x T array time path of policy function values
          for b' = phi(s,e,b,t). The last row phi(S,e,b,t) is zeros and
          corresponds to b_{S+1}=0. The first row corresponds to b_2.
p1aind  = index of period-1 age
Vinit   = bsize x J matrix of values of the state in the next period:
          V(b',e')
sind    = index of age from period 1
tind    = index of time period
------------------------------------------------------------------------
c          = bsize x J x bsize matrix of values for consumption in the
             current period: c(b,e,b')
cposind    = bsize x J x bsize array of = 1 if c > 0 and = 0 if c <= 0
cpos       = bsize x J x bsize c array with c <= 0 values replaces with
             positive values very close to zero
bposind    = bsize x J x bsize array of = 1 if c >= 0 and = 0 if c < 0.
             This matrix is important because it allows for b'=0 to be
             the optimal choice when income equals zero
uc         = utility of consumption. The utility of c<0 is set to -10^8
EVprime    = the expected value of the value function in the next period
EVprimenew = the expected value of the value function in the next period
             reshaped to be added to the utility of current consumption
Vnewarray  = new value function in terms of b, e, and the unoptimized
             possible values of b'
Vnew       = the new value function when b' is chosen optimally
bprimeind  = the index of the optimal
------------------------------------------------------------------------
'''

Kinit = np.array(list(np.linspace(K0, Kss, T)) + list(np.ones(S-2)*Kss))
Ninit = np.ones(T+S-2) * Nss
Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
winit = (1-alpha) * (Yinit/Ninit)
rinit = alpha * (Yinit/Kinit)

TPIiter = 0
TPImaxiter = 500
TPIdist = 10
TPImindist = 3.0*10**(-6)

while (TPIiter < TPImaxiter) and (TPIdist >= TPImindist):
    gammat = np.zeros((S-1, J, bsize, T))
    gammat[:, :, :, 0] = gamma0
    phiindt = np.zeros((S, J, bsize, T+S-2))
    phit = np.zeros((S, J, bsize, T+S-2))

    # Solve for the lifetime savings policy rules of each type of
    # person at t = 1.
    for p1aind in xrange(S):
        Vinit = np.zeros((bsize, J))
        for sind in xrange(p1aind+1):
            if sind < S-1:
                c = (1+rinit.flatten()[p1aind-sind]) * np.tile(b.reshape(
                    bsize, 1, 1), (1, J, bsize)) + (winit.flatten()[
                    p1aind-sind] * n[S-sind-1]) * np.tile(
                    e[S-sind-1, :].reshape((1, 7, 1)), (
                        bsize, 1, bsize)) - np.tile(
                    b.reshape((1, 1, bsize)), (bsize, J, 1))
            elif sind == S-1:
                c = (winit.flatten()[p1aind-sind] * n[S-sind-1]) * np.tile(
                    e[S-sind-1, :].reshape((1, 7, 1)),  (
                        bsize, 1, bsize)) - np.tile(b.reshape(
                    (1, 1, bsize)),  (bsize, J, 1))
            cposind = c > 0
            cnonposind = np.ones((bsize, J, bsize)) - cposind
            cpos = c*cposind + (10**(-8))*cnonposind
            uc = ((cpos**(1-sigma) - np.ones((bsize, J, bsize))) / (
                1 - sigma)) * cposind - (10**(8))*cnonposind
            if sind == 0:
                EVprime = Vinit.sum(axis=1)
            else:
                EVprime = (Vinit * np.tile(f[S-sind, :], (bsize, 1))).sum(
                    axis=1)
            EVprimenew = np.tile(EVprime.reshape(1, 1, bsize), (bsize, J, 1))
            Vnewarray = uc + beta * (EVprimenew * cposind)
            Vnew = Vnewarray.max(axis=2)
            bprimeind = Vnewarray.argmax(axis=2)
            phiindt[S-sind-1, :, :, p1aind-sind] = bprimeind.T.reshape(
                (1, J, bsize))
            phit[S-1-sind, :, :, p1aind-sind] = b[bprimeind].T.reshape(
                (1, J, bsize))
            Vinit = Vnew

    # Solve for the lifetime savings policy rules of each age-1 person
    # from t = 2 to t = T-1
    for tind in xrange(1, T-1):
        Vinit = np.zeros((bsize, J))
        for sind in xrange(S):
            if sind < S-1:
                c = (1+rinit.flatten()[tind+S-sind-2]) * np.tile(b.reshape(
                    bsize, 1, 1), (1, J, bsize)) + (winit.flatten()[
                    tind+S-sind-2] * n[S-sind-1]) * np.tile(
                    e[S-sind-1, :].reshape((1, 7, 1)), (
                        bsize, 1, bsize)) - np.tile(
                    b.reshape((1, 1, bsize)), (bsize, J, 1))
            elif sind == S-1:
                c = (winit.flatten()[tind] * n[0]) * np.tile(
                    e[0, :].reshape((1, 7, 1)),  (
                        bsize, 1, bsize)) - np.tile(b.reshape(
                    (1, 1, bsize)),  (bsize, J, 1))
            cposind = c > 0
            cnonposind = np.ones((bsize, J, bsize)) - cposind
            cpos = c*cposind + (10**(-8))*cnonposind
            uc = ((cpos**(1-sigma) - np.ones((bsize, J, bsize))) / (
                1 - sigma)) * cposind - (10**(8))*cnonposind
            if sind == 0:
                EVprime = Vinit.sum(axis=1)
            else:
                EVprime = (Vinit * np.tile(f[S-sind, :], (bsize, 1))).sum(
                    axis=1)
            EVprimenew = np.tile(EVprime.reshape(1, 1, bsize), (bsize, J, 1))
            Vnewarray = uc + beta * (EVprimenew * cposind)
            Vnew = Vnewarray.max(axis=2)
            bprimeind = Vnewarray.argmax(axis=2)
            phiindt[S-sind-1, :, :, tind+S-sind-2] = bprimeind.T.reshape(
                (1, J, bsize))
            phit[S-1-sind, :, :, tind+S-sind-2] = b[bprimeind].T.reshape(
                (1, J, bsize))
            Vinit = Vnew

    # Generate new time path for distribution of capital gammat and of
    # the aggregate capital stock K
    Knew = np.array([[K0] + list(np.zeros(T-1)) + list(Kss*np.ones(S-2))])
    for tind in xrange(1, T):
        for sind in xrange(S-1):
            for eind in xrange(J):
                for bind in xrange(bsize):
                    if sind == 0:
                        gammat[sind, eind, bind, tind] = (1/float(S-1)) * \
                            f[sind+1, eind] * ((phiindt[
                                sind, :, 0, tind-1] == bind) * f[
                                sind, :]).sum()
                    else:
                        gammat[sind, eind, bind, tind] = f[sind+1, eind] * \
                            ((phiindt[sind, :, :, tind-1] == bind) * gammat[
                                sind-1, :, :, tind-1]).sum()
        Knew[0, tind] = (float(S-1)/S*(gammat[:, :, :, tind] * np.tile(
            b.reshape(1, 1, bsize),  (S-1, J, 1))).sum())

    TPIiter += 1
    TPIdist = ((Knew - Kinit).abs()).max()
    print TPIiter
    print TPIdist
    Kinit = rho_TPI*Knew + (1-rho_TPI)*Kinit
    Yinit = A*((Kinit**alpha) * (Ninit**(1-alpha)))
    winit = (1-alpha) * (Yinit/Kinit)
    rinit = alpha * (Yinit/Kinit)

Kpath_TPI = Kinit.flatten()
gammat_TPI = gammat

elapsed_time = time.time() - start_time

print "TPI took", elapsed_time
print "The time path is", Kpath_TPI
print "Iterations:", TPIiter
print "Distance:", TPIdist

# plt.plot(np.arange(T+10), Kssvec)
plt.plot(np.arange(T+10), Kpath_TPI[:T+10])
plt.axhline(y=Kss)
plt.savefig("TPI")

var_names = ['S', 'beta', 'sigma', 'alpha', 'rho', 'A', 'delta', 'n', 'e',
             'f', 'J', 'bmin', 'bmax', 'bsize', 'b', 'gamma_ss', 'Kss',
             'Nss', 'Yss', 'wss', 'rss', 'phiind_ss', 'phi_ss', 'runtime',
             'hours', 'minutes', 'seconds', 'ssiter', 'ssdist', 'bsavg',
             'gxbar', 'Kpath_TPI', 'TPIiter', 'TPIdist', 'elapsed_time',
             'gammat_TPI']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("TPI_vars.pkl", "w"))
