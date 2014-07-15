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
from scipy.optimize import newton_krylov
from scipy.optimize import anderson
import pickle

'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S     = number of periods an individual lives
J     = number of different ability groups
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
sigma = 3.0
alpha = .35
rho = .50
A = 1.0
delta = 1 - (0.95 ** (60.0 / S))
xi = 3.0
eta = .5
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
    output = c**(-sigma)
    return output


def MUl(l):

    """
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    """
    output = - eta * (l ** (-xi))
    return output


def Steady_State(guesses):

    """
    Parameters: Steady state distribution of capital guess as array
                size S-1

    Returns:    Array of S-1 Euler equation errors
    """
    K_guess = guesses[0: (S-1) * J]
    K_guess = K_guess.reshape((S-1, J))
    K = K_guess.mean()
    
    # Labor Leisure only:
    # N_guess = guesses[(S-1) * J:]
    # N_guess = N_guess.reshape((S, J))
    # N = (f * (e*N_guess).reshape(S, J, 1)).mean()

    N = get_N(f, e, n)
    Y = get_Y(K, N)
    w = get_w(Y, N)
    r = get_r(Y, K)

    K1 = np.array(list(np.zeros(J).reshape((1, J))) + list(K_guess[:-1, :]))
    K2 = K_guess
    K3 = np.array(list(K_guess[1:, :]) + list(np.zeros(J).reshape((1, J))))

    K1_2 = np.array(list(np.zeros(J).reshape((1, J))) + list(K_guess))
    K2_2 = np.array(list(K_guess) + list(np.zeros(J).reshape((1, J))))

    error_onlycapital = MUc((1 + r)*K1 + w * e[:-1, :] * n[:-1].reshape(S-1, 1) - K2) - beta * (1 + r)*MUc(
        (1 + r)*K2 + w * (e[1:, :].reshape(S-1, J, 1) * f[
            1:, :, :]).sum(axis=2) * n[1:].reshape(S-1, 1) - K3)

    # Labor Leisure only:
    # error1 = MUc((1 + r)*K1 + w * e[:-1, :] * N_guess[:-1, :] - K2) - beta * (1 + r)*MUc(
    #     (1 + r)*K2 + w * (e[1:, :].reshape(S-1, J, 1) * f[
    #         1:, :, :]).sum(axis=2) * N_guess[1:, :] - K3)
    # error2 = MUc((1 + r)*K1_2 + w * e * N_guess - K2_2) * w * e + MUl(N_guess)

    # if (((1 + r)*K1_2 + w * e * N_guess - K2_2) <= 0).any():
    #     error1 += 10000
    # if (N_guess < 0.0).any() or (N_guess>3.0).any():
    #     error2 += 10000

    # Labor Leisure only:
    # error1, error2 = error1.flatten(), error2.flatten()
    # return np.array(list(error1) + list(error2))

    return error_onlycapital.flatten()

K_guess = np.ones((S-1, J)) / ((S-1) * J)
N_guess = np.ones((S, J)) * .5
guesses = np.array(list(K_guess.flatten()) + list(N_guess.flatten()))
solutions = opt.fsolve(Steady_State, K_guess, xtol=1e-9)

# Solvers for large matrices
# solutions = newton_krylov(Steady_State, guesses, method='gmres')
# solutions = anderson(Steady_State, guesses)

Kssmat = solutions[0:(S-1) * J]
Kssvec = Kssmat.reshape((S-1, J)).mean(1)
Kssvec = np.array([0]+list(Kssvec))
Kss = Kssvec.mean()
print "Kss:", Kss

# Labor Leisure only
# Nssmat = solutions[(S-1) * J:]
# Nssvec = Nssmat.reshape((S, J)).mean(1)
# Nss = Nssvec.mean()

Nss = get_N(f, e, n)
print "Nss:", Nss
Yss = get_Y(Kss, Nss)
print "Yss:", Yss
wss = get_w(Yss, Nss)
print "wss:", wss
rss = get_r(Yss, Kss)
print "rss:", rss

runtime = time.time() - starttime
hours = runtime / 3600
minutes = (runtime / 60) % 60
seconds = runtime % 60
print 'Finding the steady state took %.0f hours, %.0f minutes, and %.0f \
seconds.' % (abs(hours - .5), abs(minutes - .5), seconds)

if Kss > 20 or Kss < .1:
    print 'And it was wrong.'

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
             'f', 'J', 'Kss', 'Kssvec', 'Kssmat', 'Nss', 'Yss', 'wss', 'rss',
             'runtime', 'hours', 'minutes', 'seconds', 'eta', 'xi']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("ss_vars.pkl", "w"))
