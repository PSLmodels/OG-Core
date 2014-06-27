'''
Python version of the Evan/Phillips 2014 paper
Last updated: 6/25/2014
Python version of the Evan/Phillips 2014 paper
Calculates steady state, and then usies timepath iteration and
alternate model forecast method to solve
'''

#Packages
import numpy as np
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
f     = S x J matrix of age dependent discrete probability mass function
        for e: f(e_s)
J     = number of points in the support of e
bmin  = minimum value of b
bmax  = maximum value of b
bsize = number of discrete points in the support of b
b     = 1 x bsize vector of possible values for initial wealth b
        and savings b'
------------------------------------------------------------------------
'''
starttime = time.time()

S = 6
beta = .96 ** (60 / S)
sigma = 3.0
alpha = .35
rho = .2
A = 1.0
delta = 0.0 ** (60 / S)
n = np.ones(60)
n[0:6] = np.array([.87, .89, .91, .93, .96, .98])
n[40:] = np.array(
    [.95, .89, .84, .79, .73, .68, .63, .57, .52, .47,
     .4, .33, .26, .19, .12, .11, .11, .10, .10, .09])
#if S is not 60:
n = n[60 % S:: 60 / S]
e = np.array([.1, .5, .8, 1.0, 1.2, 1.5, 1.9]).T
e = np.tile(e, (S, 1))
f = np.array([.04, .09, .2, .34, .2, .09, .04]).T
f = np.tile(f, (S, 1))
J = e.shape[1]
bsize = 35
bmin = 0
bmax = 15
b = np.linspace(bmin, bmax, bsize)


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
    phiind = np.zeros((S, J, bsize))
    phi = np.zeros((S, J, bsize))
    Vinit = np.zeros((bsize, J))
    for sind in xrange(S):
        if sind < S - 1:
            c = (1 + rss - delta) * np.tile(b.reshape(
                bsize, 1, 1), (1, J, bsize)) + (
                wss * n[S-sind-1]) * np.tile(
                e[S-sind-1, :].reshape(1, J, 1), (bsize, 1, bsize)) - np.tile(
                b.reshape(1, 1, bsize), (bsize, J, 1))
        else:
            c = (wss * n[S-sind-1]) * np.tile(
                e[S-sind-1, :].reshape(1, J, 1), (bsize, 1, bsize)) - np.tile(
                b.reshape(1, 1, bsize), (bsize, J, 1))
        cposind = c > 0
        cnonposind = np.ones((bsize, J, bsize)) - cposind
        cpos = c * cposind + (1e-8) * cnonposind
        uc = ((np.power(cpos, (1 - sigma)) - np.ones((bsize, J, bsize))) / (
            1 - sigma)) * cposind - (1e-8) * cnonposind
        if sind == 1:
            EVprime = np.sum(Vinit, 1)
        else:
            EVprime = np.sum(Vinit * np.tile(f[S-sind-1, :], (
                bsize, 1)), axis=1)
        EVprimenew = np.tile(EVprime.reshape(1, 1, bsize), (bsize, J, 1))
        Vnewarray = uc + beta * (EVprimenew * cposind)
        Vnew, bprimeind = Vnewarray.max(2), Vnewarray.argmax(2)
        phiind[S-sind-1, :, :] = bprimeind.T.reshape(1, J, bsize)
        phi[S-sind-1, :, :] = b[bprimeind].T.reshape(1, J, bsize)
        Vinit = Vnew
    gamma_new = np.zeros((S-1, J, bsize))
    for sind in xrange(S-1):
        for eind in xrange(J):
            for bind in xrange(bsize):
                if sind == 0:
                    gamma_new[sind, eind, bind] = (
                        1 / float(S - 1)) * f[sind+1, eind] * np.sum(
                        (phiind[sind, :, 0] == bind) * f[sind, :])
                else:
                    gamma_new[sind, eind, bind] = f[sind+1, eind] * np.sum(
                        (phiind[sind, :, :] == bind) * gamma_init[
                            sind-1, :, :])
    ssiter += 1
    ssdist = np.max(abs(gamma_new - gamma_init))
    gamma_init = rho * gamma_new + (1 - rho) * gamma_init

gamma_ss = gamma_init
phiind_ss = phiind
phi_ss = phi

runtime = time.time() - starttime
inhours = runtime / float(60 * 60)
hours = np.round(inhours)
minutes = abs(inhours - hours) * 60


'''
------------------------------------------------------------------------
 Generate graph of the steady-state distribution of wealth
------------------------------------------------------------------------
domain     = 1 x S vector of each age cohort
bsavg      = 1 x S vector of the average wealth b of each age cohort
b(i)        = 1 x S vector of the (i)th percentile of wealth holdings of
             each age cohort
wealthwgts = (S-1) x J x bsize array of distribution weights times the
             value of wealth
bpct       = (S-1) x 1 x bsize array of percent of population with each
             wealth level for each age cohort
bpctl      = (S-1) x 1 x bsize array of percentile of each wealth level
             for each age cohort
pctl_init  = (S-1) x 1 vector of zeros for initial percentile
------------------------------------------------------------------------
'''

domain = np.linspace(0, S, S)
bsavg = np.zeros((1, S))
b90 = np.zeros((1, S))
b75 = np.zeros((1, S))
b50 = np.zeros((1, S))
b25 = np.zeros((1, S))
b10 = np.zeros((1, S))

wealthwgts = ((S-1) * gamma_ss) * np.tile(b.reshape(1, 1, bsize), (S-1, J, 1))
bpct = (S-1) * np.sum(gamma_init, axis=1).reshape(S-1, 1, bsize)
bpctl = np.zeros((S-1, 1, bsize))
pctl_init = np.zeros((S-1, 1))
for bind in xrange(bsize):
    if bind == 0:
        bpctl[:, 0, bind] = bpct[:, 0, bind] + pctl_init.flatten()
    else:
        bpctl[:, 0, bind] = bpct[:, 0, bind] + bpctl[:, 0, bind-1]

for sind in xrange(1, S):
    bsavg[0, sind] = np.sum(wealthwgts[sind-1, :, :])
    b90diffsq = np.power(
        (bpctl[sind-1, 0, :] - .90 * np.ones((1, 1, bsize))), 2)
    b75diffsq = np.power(
        (bpctl[sind-1, 0, :] - .75 * np.ones((1, 1, bsize))), 2)
    b50diffsq = np.power(
        (bpctl[sind-1, 0, :] - .50 * np.ones((1, 1, bsize))), 2)
    b25diffsq = np.power(
        (bpctl[sind-1, 0, :] - .25 * np.ones((1, 1, bsize))), 2)
    b10diffsq = np.power(
        (bpctl[sind-1, 0, :] - .10 * np.ones((1, 1, bsize))), 2)
    b90minind = b90diffsq.argmin()
    b75minind = b75diffsq.argmin()
    b50minind = b50diffsq.argmin()
    b25minind = b25diffsq.argmin()
    b10minind = b10diffsq.argmin()
    b90[0, sind] = b[b90minind]
    b75[0, sind] = b[b75minind]
    b50[0, sind] = b[b50minind]
    b25[0, sind] = b[b25minind]
    b10[0, sind] = b[b10minind]
bsavg = bsavg.flatten()
b90 = b90.flatten()
b75 = b75.flatten()
b50 = b50.flatten()
b25 = b25.flatten()
b10 = b10.flatten()

plt.figure(1)
plt.plot(domain, bsavg, color='b', label='Average capital stock')
plt.axhline(y=Kss, color='r', label='Steady State')
plt.title('Steady-state Distribution of Capital')
plt.legend(loc=0)
# plt.show()
plt.savefig("distribution_of_capital")

'''
------------------------------------------------------------------------
Generate steady state multiplier values
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
Generate steady state multiplier values
------------------------------------------------------------------------
'''

ssbsavg = (S-1) * (gamma_ss * np.tile(b.reshape(1, 1, bsize), (
    S-1, J, 1))).sum(axis=2).sum(axis=1)
esavg = (e*f).sum(axis=1)
nsavg = n.T
cssvec = (1+rss) * np.array([0]+list(ssbsavg[:S-2])) + wss * \
    esavg[:S-1] * nsavg[:S-1] - ssbsavg
cp1ssvec = (1+rss) * ssbsavg + wss*esavg[1:] * nsavg[1:] - \
    np.array(list(ssbsavg[1:])+[0])
gxbar = (cssvec**(-sigma)) / ((beta*(1+rss)) * cp1ssvec**(-sigma))

b0array = np.array(list(np.zeros((1, J, bsize))) + list(np.tile(
    b.reshape(1, 1, bsize), (S-2, J, 1))))
e0array = np.tile(e[:S-1, :].reshape(S-1, J, 1), (1, 1, bsize))
n0array = np.tile(n[:S-1].reshape(S-1, 1, 1), (1, J, bsize))
b1array = phi_ss[:S-1, :, :]
c0array = ((1 + rss) * b0array) + (wss * e0array * n0array) - b1array
mu0array = np.power(c0array, (-sigma))
Eb1array = np.tile(b1array.reshape(S-1, J, bsize, 1), (1, 1, 1, J))
Een1start = e[1:S, :] * np.tile(n[1:S].reshape((S-1), 1), (1, J))
Een1array = np.tile(Een1start.reshape(S-1, 1, 1, J), (1, J, bsize, 1))
b2b1ind = np.tile(phiind_ss.reshape(S, J, bsize, 1), (1, 1, 1, J))
b2s1ind = np.tile(np.arange(1, S).reshape(S-1, 1, 1, 1), (1, J, bsize, J))
b2e1ind = np.tile(np.arange(J).reshape(1, 1, 1, J), (S-1, J, bsize, 1))
Eb2array = np.zeros((S-1, J, bsize, J))
for sind in xrange(S-1):
    for e1ind in xrange(J):
        for bind in xrange(bsize):
            for e2ind in xrange(J):
                Eb2array[sind, e1ind, bind, e2ind] = phi_ss[
                    b2s1ind[sind, e1ind, bind, e2ind], b2e1ind[
                        sind, e1ind, bind, e2ind], b2b1ind[
                        sind, e1ind, bind, e2ind]]
f1array = np.tile(f[1:S, :].reshape(S-1, 1, 1, J), (1, J, bsize, 1))
Ec1array = ((1 + rss) * Eb1array) + (wss * Een1array) - Eb2array
Emu1array = (np.power(Ec1array, (-sigma)) * f1array).sum(3)
lamdif = (mu0array - (beta * (1 + rss)) * Emu1array) / beta
lam1pos = phiind_ss[:S-1, :, :] == 1
lamdifpos = lamdif > 0
lambda1 = lamdif * lam1pos * lamdifpos
lam2pos = phiind[:S-1, :, :] == bsize
lamdifneg = lamdif < 0
lambda2 = -lamdif * lam2pos * lamdifneg
lambda1sbar = ((S-1) * lambda1 * gamma_ss).sum(2).sum(1)
lambda2sbar = ((S-1) * lambda2 * gamma_ss).sum(2).sum(1)
lamgxbar = (np.power(cssvec, (
    -sigma)) - beta * lambda1sbar + beta * lambda2sbar) / (
    (beta * (1 + rss)) * np.power(cssvec, (-sigma)))

'''
------------------------------------------------------------------------
Generate graph of Euler Errors
------------------------------------------------------------------------
'''

plt.figure(2)
plt.plot(np.arange(1, S), gxbar)
plt.title('Euler errors: S = {}'.format(S))
# plt.legend(loc=0)
# plt.show()
plt.savefig("euler_errors")

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

var_names = ['S', 'beta', 'sigma', 'alpha', 'rho', 'A', 'delta', 'n', 'e',
             'f', 'J', 'bmin', 'bmax', 'bsize', 'b', 'gamma_ss', 'Kss',
             'Nss', 'Yss', 'wss', 'rss', 'phiind_ss', 'phi_ss']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("ss_vars.pkl", "w"))
