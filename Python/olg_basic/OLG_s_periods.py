'''
Python version of the Evan/Phillips 2014 paper
Last updated: 6/25/2014
Python version of the Evan/Phillips 2014 paper
Calculates steady state, and then usies timepath iteration and
alternate model forecast method to solve
'''

#Packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math

'''
Setting up the Model
'''

'''
Definitions:

S     = number of periods an individual lives
beta  = discount factor
sigma = coefficient of relative risk aversion
alpha = capital share of income
rho   = contraction parameter in steady state iteration process
        representing the weight on the new distribution gamma_new
A     = total factor productivity parameter in firms' production function
delta = depreciation rate of capital
n     = 1 x S vector of inelastic labor supply for each age s
e     = S x J matrix of age dependent possible working abilities e_s
f     = S x J matrix of age dependent discrete probability mass function
        for e: f(e_s)
J     = number of points in the support of e
bmin  = minimum value of b
bmax  = maximum value of b
bsize = number of discrete points in the support of b
b     = 1 x bsize vector of possible values for initial wealth b
        and savings b'

'''

S = 60
beta = .96
sigma = 3.0
alpha = .35
rho = .2
A = 1.0
delta = 0.0
n = np.ones(S)
n[0:6] = np.array([.87, .89, .91, .93, .96, .98])
n[40:] = np.array(
    [.95, .89, .84, .79, .73, .68, .63, .57, .52, .47,
     .4, .33, .26, .19, .12, .11, .11, .10, .10, .09])
e = np.array([.1, .5, .8, 1.0, 1.2, 1.5, 1.9]).T
e = np.tile(e, (S, 1))
f = np.array([.04, .09, .2, .34, .2, .09, .04]).T
f = np.tile(f, (S, 1))
J = e.shape[1]
bsize = 350
bmin = 0
bmax = 15
b = np.linspace(bmin, bmax, bsize)

'''
Finding the Steady State
'''

'''
Definitions:

ssiter     = index value that tracks the iteration number
ssmaxiter  = maximum number of iterations
ssdist     = sup norm distance measure of the elements of the wealth
             distribution gamma_init and gamma_new
ssmindist  = minimum value of the sup norm distance measure velow which
             the process has converged to the steady state
gamma_init = initial guess for steady state capital distribution:
             (S-1) x J x bsize array
gamma_new  = (S-1) x J x bsize array, new iteration distribution of
             wealth computed from the old distribution gamma_init and the
             policy rule bprime
Kss        = steady state aggregate capital stock
Nss        = steady state aggregate labor
Yss        = steady state aggregate output
wss        = steady state real wage
rss        = steady state real rental rate
phi_ind    = S x J x bsize policy function indicies for b' = phi(s,e,b).
             The last row phiind(S,e,b) is ones and corresponds to
             b_{S+1}. The first row phi(1,e,b) corresponds to b_2.
phi        = S x J x bsize policy function values for b' = phi(s,e,b).
             The last row phi(S,e,b) is zeros and corresponds to b_{S+1}.
             The first row corresponds to b_2.
Vinit      = bsize x J matrix of values of the state in the next period
             V(b',e')
sind       = year index variable in for-loop
eind       = productivity shock index variable in for-loop
bind       = wealth level index variable in for-loop
c          = bsize x J x bsize matrix of values for consumption in the
             current period: c(b,e,b')
cposind    = bsize x J x bsize array of = 1 if c > 0 and = 0 if c <= 0
cpos       = bsize x J x bsize c array with c <= 0 values replaces with
             positive values very close to zero
uc         = utility of consumption. The utility of c<0 is set to -10^8
EVprime    = the expected value of the value function in the next period
EVprimenew = the expected value of the value function in the next period
             reshaped to be added to the utility of current consumption
Vnewarray  = new value function in terms of b, e, and the unoptimized
             possible values of b'
Vnew       = the new value function when b' is chosen optimally
bprimeind  = the index of the optimal
gamma_ss   = (S-1) x J x bsize array of steady state distribution of
             wealth
phi_ind_ss  = S x J x bsize steady-state policy function indicies for
             b' = phi(s,e,b)
phi_ss     = S x J x bsize steady-state policy function values for
             b' = phi(s,e,b)

'''

# Functions and Definitions


def get_Y(K_now, N_now):
    # Equation 2.11
    Y_now = A * (K_now ** alpha) * (N_now ** (1 - alpha))
    return Y_now


def get_w(Y_now, N_now):
    # Equation 2.13
    w_now = (1 - alpha) * Y_now / N_now
    return w_now


def get_r(Y_now, K_now):
    # Equation 2.14
    r_now = alpha * Y_now / K_now
    return r_now


def get_N(f, e, n):
    # Equation 2.15
    N_now = np.sum(f * e * n.reshape(S, 1))
    N_now /= S
    return N_now


def get_K(e, f, b, gamma):
    # Equation 2.16
    # where gamma is the distribution of capital between s, e, and b
    K_now = np.sum(gamma[:, :, :] * b)
    K_now *= float(S - 1) / S
    return K_now

ssiter = 0
ssmaxiter = 700
ssdist = 10
ssmindist = 1e-9
# Generate gamma_init
gamma_init = np.copy(f).reshape(S, J, 1)
gamma_init = np.tile(gamma_init, (1, 1, bsize))
gamma_init = gamma_init[1:, :, :]
gamma_init /= bsize * (S - 1)

while (ssiter < ssmaxiter) & (ssdist >= ssmindist):
    Kss = get_K(e, f, b, gamma_init)
    Nss = get_N(f, e, n)
    Yss = get_Y(Kss, Nss)
    wss = get_w(Yss, Nss)
    rss = get_r(Yss, Kss)
    phi_ind = np.zeros((S, J, bsize))
    phi = np.zeros((S, J, bsize))
    Vinit = np.zeros((bsize, J))
    for sind in xrange(1, S):
        if sind < S:
            c = (1 + rss - delta) * np.tile(b.T, (1, J, bsize)) + (
                wss * n[S-sind+1]) * np.tile(
                e[S-sind+1, :], (bsize, 1, bsize)) - np.tile(
                b.reshape(1, 1, bsize), (bsize, J, 1))
        else:
            c = (wss * n[S-sind+1]) * np.tile(
                e[S-sind+1, :], (bsize, 1, bsize)) - np.tile(
                b.reshape(1, 1, bsize), (bsize, J, 1))
        cposind = c > 0
        cnonposind = np.ones((bsize, J, bsize)) - cposind
        cpos = c * cposind + (1e-8) * cnonposind
        uc = ((np.power(cpos, (1 - sigma)) - np.ones((bsize, J, bsize))) / (
            1 - sigma)) * cposind - (1e-8) * cnonposind
        if sind == 1:
            EVprime = np.sum(Vinit, 2)
        else:
            EVprime = np.sum(Vinit * np.tile(f[S-sind+2, :], (
                bsize, 1)), axis=1)
        EVprimenew = np.tile(EVprime.reshape(1, 1, bsize), (bsize, J, 1))
        Vnewarray = uc + beta * (EVprimenew * cposind)
        Vnew, bprimeind = Vnewarray.max(2), Vnewarray.argmax(2)
        phi_ind[S-sind+1, :, :] = bprimeind.T.reshape(1, J, bsize)
        phi[S-sind+1, :, :] = b[bprimeind].T.reshape(1, J, bsize)
        Vinit = Vnew
    gamma_new = np.zeros(S-1, J, bsize)
    for sind in (1, S-1):
        for eind in (1, J):
            for bind in (1, bsize):
                if sind == 1:
                    gamma_new[sind, eind, bind] = (
                        1 / float(S - 1)) * f[sind+1, eind] * np.sum(
                        (phi_ind[sind, :, 1] == bind) * f[sind, :])
                else:
                    gamma_new[sind, eind, bind] = f[sind+1, eind] * np.sum(
                        (phi_ind[sind, :, :] == bind) * gamma_init[sind-1, :, :])
    ssiter += 1
    ssdist = np.max(abs(gamma_new - gamma_init))
    gamma_init = rho * gamma_new + (1 - rho) * gamma_init

gamma_ss = gamma_init
phi_ind_ss = phi_ind
phi_ss = phi




