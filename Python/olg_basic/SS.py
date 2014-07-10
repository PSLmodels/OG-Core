'''
------------------------------------------------------------------------
Last updated: 7/1/2014
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

#Packages
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
J     = number of points in the support of e_s
bmin  = minimum value of b
bmax  = maximum value of b
bsize = number of discrete points in the support of b
b     = 1 x bsize vector of possible values for initial wealth b
        and savings b'
------------------------------------------------------------------------
'''
starttime = time.time()

S = 60
J = 7
bsize = 100
beta = .96 ** (60 / S)
sigma = 3
alpha = .35
rho = .20
A = 1
delta = 0.0 ** (60 / S)

if S >= 12:
    n = np.ones(S)
    n[0:S/10+1] = np.linspace(0.865, 1, (S/10)+1)
    n[- 2*((S/12)+1)+1 - ((S/6)+2): - 2*((S/12)+1)+1] = np.linspace(
        1, 0.465, (S/6)+2)
    n[- 2*((S/12)+1): -(S/12+1)+1] = np.linspace(0.465, .116, (S/12)+2)
    n[-(S/12+1):] = np.linspace(0.116, .093, (S/12)+1)
# The above method doesn't work well with very small S, so we have this 
# alternative
else: 
    n = np.ones(60)
    n[0:6] = np.array([.865, .8875, .91, .9325, .955, .9775])
    n[40:] = np.array(
        [.9465, .893, .8395, .786, .7325, .679, .6255, .572, .5185,
         .465, .3952, .3254, .2556, .1858, .116, .1114, .1068, .1022,
         .0976, .093])
    n = n[60 % S:: 60 / S]

e = income.get_e(S, J)
f = income.get_f_markov(S, J)

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
    N_now = np.sum(f * e.reshape(S,J,1) * n.reshape(S, 1, 1)) / J
    N_now /= S
    return N_now


def get_K(b, gamma):
    # Equation 2.16
    # where gamma is the distribution of capital between s, e, and b
    K_now = np.sum(gamma[:, :, :] * b)
    K_now *= float(S - 1) / S
    return K_now

ssiter = 0
#Takes 590 iterations when S=60, J=7, and bsize=350
ssmaxiter = 700
ssdist = 10
ssmindist = 1e-9
# Generate gamma_init
gamma_init = np.copy(f[:,:,0]).reshape(S,J,1)
gamma_init = np.tile(gamma_init, (1, 1, bsize))
gamma_init = gamma_init[1:, :, :]
gamma_init /= bsize * (S - 1)

while (ssiter < ssmaxiter) & (ssdist >= ssmindist):
    Kss = get_K(b, gamma_init)
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
        uc = (((cpos ** (1 - sigma)) - np.ones((bsize, J, bsize))) / (
            1 - sigma)) * cposind - (10 ** 8) * cnonposind
        EVprime = Vinit.dot(f[S-sind-1, :, :]).mean(1)
        EVprimenew = np.tile(EVprime.reshape(1, 1, bsize), (bsize, J, 1))
        Vnewarray = uc + beta * (EVprimenew * cposind)
        Vnew, bprimeind = Vnewarray.max(2), Vnewarray.argmax(2)
        phiind[S-sind-1, :, :] = bprimeind.T.reshape(1, J, bsize)
        phi[S-sind-1, :, :] = b[bprimeind].T.reshape(1, J, bsize)
        Vinit = Vnew
    gamma_new = np.zeros((S-1, J, bsize))
    gamma_new[0, :, :] = (
        1 / float(S - 1)) * f[1, :, :].mean(1).reshape(J, 1).dot(
        ((phiind[0, :, 0].reshape(1, J, 1) == np.arange(
            bsize)) * f[0, :, :].mean(1).reshape(J, 1)).sum(axis=1))
    for sind in xrange(1, S-1):
        for bind in xrange(bsize):
            gamma_new[sind, :, bind] = f[sind+1, :, :].mean(1) * np.sum((
                phiind[sind, :, :] == bind) * gamma_init[sind-1, :, :])
    # gamma_new[1:,:,:] = f[2:,:].reshape((J,S-2)).dot(((phiind[
        #1:-1,:,:]==np.arange(bsize)) * gamma_init[:-1,:,:]).sum(axis=1))
    ssiter += 1
    ssdist = np.max(abs(gamma_new - gamma_init))
    gamma_init = rho * gamma_new + (1 - rho) * gamma_init
    print ssiter
    print ssdist

gamma_ss = gamma_init
phiind_ss = phiind
phi_ss = phi

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

wealthwgts = ((S-1) * gamma_ss) * np.tile(b.reshape(1, 1, bsize), (S-1, J, 1))

bsavg[1:] = wealthwgts[:,:,:].sum(axis=2).sum(axis=1)

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

ssbsavg = (S-1) * (gamma_ss * np.tile(b.reshape(1, 1, bsize), (
    S-1, J, 1))).sum(axis=2).sum(axis=1)
esavg = e.mean(1)
nsavg = n.T
cssvec = (1 + rss - delta) * np.array([0]+list(ssbsavg[:S-2])) + wss * esavg[
    :S-1] * nsavg[:S-1] - ssbsavg
cp1ssvec = (1 + rss - delta) * ssbsavg + wss * esavg[1:] * nsavg[1:] - \
    np.array(list(ssbsavg[1:])+[0])
gxbar = (cssvec**(-sigma)) / ((beta * (1 + rss - delta)) * cp1ssvec ** (-sigma))

'''
------------------------------------------------------------------------
Generate graph of Euler Errors
------------------------------------------------------------------------
'''

plt.figure(2)
plt.plot(np.arange(0, S-1), gxbar)
plt.title('Euler errors: S = {}'.format(S))
plt.ylim((0,2))
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
             'Nss', 'Yss', 'wss', 'rss', 'phiind_ss', 'phi_ss', 'runtime',
             'hours', 'minutes', 'seconds', 'ssiter', 'ssdist', 'bsavg',
             'gxbar']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("ss_vars.pkl", "w"))
