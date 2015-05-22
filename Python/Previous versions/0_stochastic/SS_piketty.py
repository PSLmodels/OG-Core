'''
------------------------------------------------------------------------
Last updated: 7/10/2014
Python version of the Evan/Phillips 2014 paper
Calculates steady state of OLG model with S age cohorts

This py-file calls the following other file(s):
            income.py

This py-file creates the following other file(s):
            ss_vars.pkl
            distribution_of_capital.png
            euler_errors.png
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import income
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle

'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S     = number of periods an individual lives
J     = number of different ability groups
bsize = number of discrete points in the support of b
beta  = discount factor
sigma = coefficient of relative risk aversion
alpha = capital share of income
rho   = contraction parameter in steady state iteration process
        representing the weight on the new distribution gamma_new
A     = total factor productivity parameter in firms' production
        function
delta = depreciation rate of capital
n     = 1 x S vector of inelastic labor supply for each age s
e     = S x J matrix of age dependent possible working abilities e_s
f     = S x J x J matrix of age dependent discrete probability mass
        function for e as Markov proccess: f(e_s)
bmin  = minimum value of b
bmax  = maximum value of b
b     = 1 x bsize vector of possible values for initial wealth b
        and savings b'
------------------------------------------------------------------------
'''

starttime = time.time()
S = 60
J = 7
bsize = 20
nsize = 10
beta = .96 ** (60 / S)
sigma = 3
alpha = .35
rho = .50
A = 1
delta = 0.0 ** (60 / S)
eta = 0.5
xi = 3
e = income.get_e(S, J)
f = income.get_f_markov(S, J)
bmin = 0
bmax = 20
b = np.linspace(bmin, bmax, bsize)
nmin = 0.001
nmax = 1.0
n = np.linspace(nmin, nmax, nsize)

'''
------------------------------------------------------------------------
Finding the Steady State
------------------------------------------------------------------------
ssiter     = index value that tracks the iteration number
ssmaxiter  = maximum number of iterations
ssdist     = sup norm distance measure of the elements of the wealth
             distribution gamma_init and gamma_new
ssmindist  = minimum value of the sup norm distance measure velow which
             the process has converged to the steady state
gamma_init = initial guess for steady state capital distribution:
             (S-1) x J x bsize array
gamma_new  = (S-1) x J x bsize array, new iteration distribution of
             wealth computed from the old distribution gamma_init and
             the policy rule bprime
Kss        = steady state aggregate capital stock
Nss        = steady state aggregate labor
Yss        = steady state aggregate output
wss        = steady state real wage
rss        = steady state real rental rate
phiind     = S x J x bsize policy function indicies for b' = phi(s,e,b).
             The last row phiind(S,e,b) is ones and corresponds to
             b_{S+1}. The first row phi(1,e,b) corresponds to b_2.
phi        = S x J x bsize policy function values for b' = phi(s,e,b).
             The last row phi(S,e,b) is zeros and corresponds to
             b_{S+1}. The first row corresponds to b_2.
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
phiind_ss  = S x J x bsize steady-state policy function indicies for
             b' = phi(s,e,b)
phi_ss     = S x J x bsize steady-state policy function values for
             b' = phi(s,e,b)
------------------------------------------------------------------------
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


def get_N(gamma, e, n):
    # Equation 2.15
    N_now = np.sum(gamma * e.reshape(S, J, 1) * n.reshape(1, 1, nsize)) 
    return N_now


def get_K(b, gamma):
    # Equation 2.16
    # where gamma is the distribution of capital between s, e, and b
    K_now = np.sum(gamma[:, :, :] * b)
    K_now *= float(S - 1) / S
    return K_now

ssiter = 0
# Takes 590 iterations when S=60, J=7, and bsize=350
ssmaxiter = 200
ssdist = 10
ssmindist = 3e-6
# Generate gamma_init
gamma_init_b = np.copy(f[:, :, 0]).reshape(S, J, 1)
gamma_init_b = np.tile(gamma_init_b, (1, 1, bsize))
gamma_init_b = gamma_init_b[1:, :, :]
gamma_init_b /= bsize * (S - 1)

# gamma_init_n = np.copy(f[:, :, 0]).reshape(S, J, 1)
# gamma_init_n = np.tile(gamma_init_n, (1, 1, bsize))
# gamma_init_n = gamma_init_n[:, :, :]
# gamma_init_n /= bsize * (S)
gamma_init_n = np.ones(S*J*nsize).reshape((S,J,nsize))/(S*J*nsize)

while (ssiter < ssmaxiter) & (ssdist >= ssmindist):
    Kss = get_K(b, gamma_init_b)
    Nss = get_N(gamma_init_n, e, n)
    Yss = get_Y(Kss, Nss)
    wss = get_w(Yss, Nss)
    rss = get_r(Yss, Kss)
    phiind_b = np.zeros((S, J, bsize))
    phi_b = np.zeros((S, J, bsize))
    phiind_n = np.zeros((S, J, bsize))
    phi_n = np.zeros((S, J, bsize))
    Vinit = np.zeros((bsize, J))
    for sind in xrange(S):
        if sind < S - 1:
            c = (1 + rss - delta) * np.tile(b.reshape(
                bsize, 1, 1, 1), (1, J, bsize, nsize)) + \
                wss * np.tile(n.reshape(1,1,1,nsize), (bsize, J, bsize, 1)) * np.tile(
                e[S-sind-1, :].reshape(1, J, 1, 1), (bsize, 1, bsize, nsize)) - np.tile(
                b.reshape(1, 1, bsize, 1), (bsize, J, 1, nsize))
        else:
            c = wss * np.tile(n.reshape(1,1,1,nsize), (bsize, J, bsize, 1)) * np.tile(
                e[S-sind-1, :].reshape(1, J, 1, 1), (bsize, 1, bsize, nsize)) - np.tile(
                b.reshape(1, 1, bsize, 1), (bsize, J, 1, nsize))
        cposind = c > 0
        cnonposind = np.ones((bsize, J, bsize, nsize)) - cposind
        cpos = c * cposind + (1e-8) * cnonposind
        uc = ((((cpos ** (1 - sigma)) - np.ones((bsize, J, bsize, nsize))) / (
            1 - sigma)) - eta*(((np.tile(n.reshape(1,1,1,nsize),(bsize,J,bsize,1)))**(1-xi) - np.ones((bsize,J,bsize,nsize))) / (1 - xi)))* cposind - (10 ** 8) * cnonposind
        EVprime = Vinit.dot(f[S-sind-1, :, :]).mean(1)
        EVprimenew = np.tile(EVprime.reshape(1, 1, bsize, 1), (bsize, J, 1, nsize))
        Vnewarray = uc + beta * (EVprimenew * cposind)
        Vnew_n, nprimeind = Vnewarray.max(3), Vnewarray.argmax(3)
        Vnew, bprimeind = Vnew_n.max(2), Vnew_n.argmax(2)
        i, j = np.meshgrid(np.arange(bsize), np.arange(J))
        nprimeind = nprimeind[i,j,bprimeind[i,j]].T
        phiind_b[S-sind-1, :, :] = bprimeind.T.reshape(1, J, bsize)
        phi_b[S-sind-1, :, :] = b[bprimeind].T.reshape(1, J, bsize)
        phiind_n[S-sind-1, :, :] = nprimeind.T.reshape(1, J, bsize)
        phi_n[S-sind-1, :, :] = n[nprimeind].T.reshape(1, J, bsize)
        Vinit = Vnew
    gamma_new_b = np.zeros((S-1, J, bsize))
    gamma_new_b[0, :, :] = (
        1 / float(S - 1)) * f[1, :, :].mean(1).reshape(J, 1).dot(
        ((phiind_b[0, :, 0].reshape(1, J, 1) == np.arange(
            bsize)) * f[0, :, :].mean(1).reshape(J, 1)).sum(axis=1))
    for sind in xrange(1, S-1):
        for bind in xrange(bsize):
            gamma_new_b[sind, :, bind] = f[sind+1, :, :].mean(1) * np.sum((
                phiind_b[sind, :, :] == bind) * gamma_init_b[sind-1, :, :])
    gamma_new_n = np.zeros((S, J, nsize))
    gamma_new_n[0, :, :] = (
        1 / float(S - 1)) * f[1, :, :].mean(1).reshape(J, 1).dot(
        ((phiind_n[0, :, 0].reshape(1, J, 1) == np.arange(
            nsize)) * f[0, :, :].mean(1).reshape(J, 1)).sum(axis=1))
    for sind in xrange(1, S):
        for nind in xrange(nsize):
            gamma_new_n[sind, :, nind] = f[sind, :, :].mean(1) * np.sum((
                phiind_n[sind, :, :] == nind).reshape(J,bsize,1) * gamma_init_n[sind-1, :, :].reshape(J,1,nsize))/bsize
    ssiter += 1
    ssdist = np.max(abs(gamma_new_b - gamma_init_b)) + np.max(abs(gamma_new_n - gamma_init_n))
    gamma_init_b = rho * gamma_new_b + (1 - rho) * gamma_init_b
    gamma_init_n = rho * gamma_new_n + (1 - rho) * gamma_init_n
    print ssiter
    print ssdist
gamma_ss_b = gamma_init_b
phiind_ss_b = phiind_b
phi_ss_b = phi_b
gamma_ss_n = gamma_init_n

runtime = time.time() - starttime
hours = runtime / 3600
minutes = (runtime / 60) % 60
seconds = runtime % 60
print 'Finding the steady state took %.0f hours, %.0f minutes, and %.0f \
seconds.' % (abs(hours - .5), abs(minutes - .5), seconds)

'''
------------------------------------------------------------------------
 Generate graph of the steady-state distribution of wealth
------------------------------------------------------------------------
domain     = 1 x S vector of each age cohort
bsavg      = 1 x S vector of the average wealth b of each age cohort
wealthwgts = (S-1) x J x bsize array of distribution weights times the
             value of wealth
------------------------------------------------------------------------
'''

domain = np.linspace(0, S, S)
bsavg = np.zeros(S)
wealthwgts = ((S-1) * gamma_ss_b) * np.tile(b.reshape(1, 1, bsize), (S-1, J, 1))
bsavg[1:] = wealthwgts[:, :, :].sum(axis=2).sum(axis=1)

plt.figure(1)
plt.plot(domain, bsavg, color='b', linewidth=2, label='Average capital stock')
plt.axhline(y=Kss, color='r', label='Steady State')
plt.title('Steady-state Distribution of Capital')
plt.legend(loc=0)
# plt.show()
plt.savefig("distribution_of_capital")

'''
------------------------------------------------------------------------
 Generate graph of the steady-state labor supply
------------------------------------------------------------------------
navg      = A vector of the average labor supply n of each age cohort
------------------------------------------------------------------------
'''

navg = (gamma_ss_n * np.tile(n.reshape(1,1,nsize),(S,J,1))).sum(2).sum(1)
plt.figure(4)
plt.plot(domain, navg, color='b', linewidth=2, label='Average labor supply')
plt.legend(loc=0)
# plt.show()
plt.savefig("labor_supply")

'''
------------------------------------------------------------------------
Generate steady state multiplier values
------------------------------------------------------------------------
'''

ssbsavg = (S-1) * (gamma_ss_b * np.tile(b.reshape(1, 1, bsize), (
    S-1, J, 1))).sum(axis=2).sum(axis=1)
esavg = e.mean(1)
cssvec = (1 + rss - delta) * np.array([0]+list(ssbsavg[:S-2])) + wss * esavg[
    :S-1] * navg[:S-1] - ssbsavg
cp1ssvec = (1 + rss - delta) * ssbsavg + wss * esavg[1:] * navg[1:] - \
    np.array(list(ssbsavg[1:])+[0])
gxbar = (cssvec**(-sigma)) / (
    (beta * (1 + rss - delta)) * cp1ssvec ** (-sigma))

'''
------------------------------------------------------------------------
Generate graph of Euler Errors
------------------------------------------------------------------------
'''

plt.figure(2)
plt.plot(np.arange(0, S-1), gxbar, linewidth=2)
plt.title('Euler errors: S = {}'.format(S))
plt.ylim((0, 2))
# plt.legend(loc=0)
# plt.show()
plt.savefig("euler_errors")

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

var_names = ['S', 'beta', 'sigma', 'alpha', 'rho', 'A', 'delta', 'n', 'e',
             'f', 'J', 'bmin', 'bmax', 'bsize', 'b', 'gamma_ss_b', 'Kss',
             'Nss', 'Yss', 'wss', 'rss', 'phiind_ss_b', 'phi_ss_b', 'runtime',
             'hours', 'minutes', 'seconds', 'ssiter', 'ssdist', 'bsavg',
             'gxbar']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("ss_vars.pkl", "w"))
