'''
------------------------------------------------------------------------
Last updated: 7/14/2014
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
import scipy.optimize as opt
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
------------------------------------------------------------------------
'''

starttime = time.time()
S = 60
J = 7
beta = .96 ** (60.0 / S)
sigma = 3
alpha = .35
rho = .50
A = 1
delta = 1 - (0.95 ** (60.0 / S))
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
f = income.get_f_noswitch(S, J)

'''
------------------------------------------------------------------------
Finding the Steady State
------------------------------------------------------------------------
Kss        = steady state aggregate capital stock
Nss        = steady state aggregate labor
Yss        = steady state aggregate output
wss        = steady state real wage
rss        = steady state real rental rate
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
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_N(f, e, n):
    # Equation 2.15
    N_now = np.sum(f * e.reshape(S, J, 1) * n.reshape(S, 1, 1)) / J
    N_now /= S
    return N_now


def MUc(c):

    """
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    """

    return c**(-sigma)


def MUl(l):

    """
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    """

    return - eta * l**(-xi)


def Steady_State(guesses):

    """
    Parameters: Steady state distribution of capital guess as array
                size S-1

    Returns:    Array of S-1 Euler equation errors
    """
    K_guess = guesses[0:(S-1) * J] 
    K = K_guess.mean()
    N = get_N(f, e, n)
    Y = get_Y(K, N)
    w = get_w(Y, N)
    r = get_r(Y, K)
    K_guess = K_guess.reshape((S-1, J))

    K1 = np.array(list(np.zeros(J).reshape((1, J))) + list(K_guess[:-1, :]))
    K2 = K_guess
    K3 = np.array(list(K_guess[1:, :]) + list(np.zeros(J).reshape((1, J))))

    error = MUc((1 + r)*K1 + w * e[:-1, :] * (
        n[:-1]).reshape((S-1, 1)) - K2) - beta * (1 + r)*MUc(
        (1 + r)*K2 + w * (e[1:, :].reshape(S-1, J, 1) * f[
            1:, :, :]).sum(axis=2) * n[1:].reshape((S-1, 1)) - K3)

    return error.flatten()

K_guess = np.ones((S-1, J)) * .05
N_guess = np.ones((S, J)) * .1
guesses = [K_guess, N_guess]
Kssmat = opt.fsolve(Steady_State, K_guess, xtol=1e-5)
Kssvec = Kssmat.reshape((S-1, J)).mean(1)
Kssvec = np.array([0]+list(Kssvec))
Kss = Kssvec.mean()
print Kss
Nss = get_N(f, e, n)
print Nss
Yss = get_Y(Kss, Nss)
print Yss
wss = get_w(Yss, Nss)
print wss
rss = get_r(Yss, Kss)
print rss

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
------------------------------------------------------------------------
'''

domain = np.linspace(0, S, S)


plt.figure(1)
plt.plot(domain, Kssvec, color='b', linewidth=2, label='Average capital stock')
plt.axhline(y=Kss, color='r', label='Steady State capital stock')
plt.title('Steady-state Distribution of Capital')
plt.legend(loc=0)
# plt.show()
plt.savefig("distribution_of_capital")


'''
------------------------------------------------------------------------
Generate graph of Consumption
------------------------------------------------------------------------
'''

newK = np.array(list(Kssvec) + [0])
esavg = e.mean(1)
nsavg = n
cssvec = (1 + rss) * newK[:-1] + wss * esavg * nsavg - newK[1:]


plt.figure(2)
# plt.plot(domain, bsavg, label='Average capital stock')
plt.plot(domain, cssvec, label='Consumption')
# plt.plot(domain, n * wss * e.mean(axis=1), label='Income')
plt.title('Consumption: S = {}'.format(S))
# plt.legend(loc=0)
# plt.show()
plt.savefig("consumption")

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

var_names = ['S', 'beta', 'sigma', 'alpha', 'rho', 'A', 'delta', 'n', 'e',
             'f', 'J', 'Kss', 'Kssvec', 'Nss', 'Yss', 'wss', 'rss', 'runtime',
             'hours', 'minutes', 'seconds']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("ss_vars.pkl", "w"))
