'''
------------------------------------------------------------------------
Last updated: 7/15/2014
Calculates steady state of OLG model with S age cohorts

This py-file calls the following other file(s):
            income.py

This py-file creates the following other file(s):
            ss_vars.pkl
            distribution_of_capital.png
            consumption.png
            euler_errors.png
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import income
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
xi    = ...
eta   = ...
n     = 1 x S vector of inelastic labor supply for each age s
e     = S x J matrix of age dependent possible working abilities e_s
f     = S x J x J matrix of age dependent discrete probability mass
        function for e as Markov proccess: f(e_s)
------------------------------------------------------------------------
'''

starttime = time.time()
S = 60
J = 5
beta = .96 ** (60.0 / S)
sigma = 3.0
alpha = .35
rho = .20
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

K_guess_init = (S-1 x J) array for the initial guess of the distribution
               of capital
N_guess_init = (S x J) array for the initial guess of the distribution
               of labor
solutions    = ((S * (S-1) * J * J) x 1) array of solutions of the
               steady state distributions of capital and labor
Kssmat       = ((S-1) x J) array of the steady state distribution of
               capital
Kssvec       = ((S-1) x 1) vector of the steady state level of capital
               (averaged across ability types)
Kss          = steady state aggregate capital stock
Nssmat       = (S x J) array of the steady state distribution of labor
Nssvec       = (S x 1) vector of the steady state level of labor
               (averaged across ability types)
Nss          = steady state aggregate labor
Yss          = steady state aggregate output
wss          = steady state real wage
rss          = steady state real rental rate
runtime      = Time needed to find the steady state (seconds)
hours        = Hours needed to find the steady state
minutes      = Minutes needed to find the steady state, less the number
               of hours
seconds      = Seconds needed to find the steady state, less the number
               of hours and minutes
------------------------------------------------------------------------
'''

# Functions and Definitions


def get_Y(K_now, N_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = A * (K_now ** alpha) * (N_now ** (1 - alpha))
    return Y_now


def get_w(Y_now, N_now):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    w_now = (1 - alpha) * Y_now / N_now
    return w_now


def get_r(Y_now, K_now):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_N(f, e, n):
    '''
    Parameters: f, e, n

    Returns:    Aggregate labor
    '''
    N_now = np.sum(f * e.reshape(S, J, 1) * n.reshape(S, 1, 1)) / J
    N_now /= S
    return N_now


def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''
    output = c**(-sigma)
    return output


def MUl(l):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    output = - eta * (l ** (-xi))
    return output


def Euler_justcapital(w, r, f, e, n, K1, K2, K3):
    euler = MUc((1 + r)*K1 + w * e[:-1, :] * n[:-1].reshape(
        S-1, 1) - K2) - beta * (1 + r)*MUc(
        (1 + r)*K2 + w * (e[1:, :].reshape(S-1, J, 1) * f[
            1:, :, :]).sum(axis=2) * n[1:].reshape(S-1, 1) - K3)
    return euler


def Euler1(w, r, f, e, N_guess, K1, K2, K3):
    euler = MUc((1 + r)*K1 + w * e[:-1, :] * N_guess[:-1, :] - K2) - beta * (
        1 + r)*MUc((1 + r)*K2 + w * (e[1:, :].reshape(S-1, J, 1) * f[
            1:, :, :]).sum(axis=2) * N_guess[1:, :] - K3)
    return euler


def Euler2(w, r, e, N_guess, K1_2, K2_2):
    euler = MUc((1 + r)*K1_2 + w * e * N_guess - K2_2) * w * e + MUl(N_guess)
    return euler


def Steady_State(guesses):
    '''
    Parameters: Steady state distribution of capital guess as array
                size S-1

    Returns:    Array of S-1 Euler equation errors
    '''
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

    error_onlycapital = Euler_justcapital(w, r, f, e, n, K1, K2, K3)

    # Labor Leisure only:
    # error1 = Euler1(w, r, f, e, N_guess, K1, K2, K3)
    # error2 = Euler2(w, r, e, N_guess, K1_2, K2_2)

    # if (((1 + r)*K1_2 + w * e * N_guess - K2_2) <= 0).any():
    #     error1 += 10000
    # if (N_guess < 0.0).any() or (N_guess>3.0).any():
    #     error2 += 10000

    # Labor Leisure only:
    # error1, error2 = error1.flatten(), error2.flatten()
    # return np.array(list(error1) + list(error2))

    return error_onlycapital.flatten()

K_guess_init = np.ones((S-1, J)) / ((S-1) * J)
N_guess_init = np.ones((S, J)) / (S * J)
guesses = np.array(list(K_guess_init.flatten()) + list(N_guess_init.flatten()))

solutions = opt.fsolve(Steady_State, K_guess_init, xtol=1e-9)

# Labor Leisure Only:
# solutions = opt.fsolve(Steady_State, guesses, xtol=1e-9)

# Solvers for large matrices
# solutions = newton_krylov(Steady_State, guesses, method='gmres', verbose=1)
# solutions = anderson(Steady_State, guesses, verbose=1)

Kssmat = solutions[0:(S-1) * J].reshape(S-1, J)
Kssvec = Kssmat.mean(1)
Kss = Kssvec.mean()
Kssvec = np.array([0]+list(Kssvec))

# Labor Leisure only
# Nssmat = solutions[(S-1) * J:].reshape(S, J)
# Nssvec = Nssmat.mean(1)
# Nss = Nssvec.mean()

Nss = get_N(f, e, n)
Yss = get_Y(Kss, Nss)
wss = get_w(Yss, Nss)
rss = get_r(Yss, Kss)

print "Kss:", Kss
print "Nss:", Nss
print "Yss:", Yss
print "wss:", wss
print "rss:", rss

runtime = time.time() - starttime
hours = runtime / 3600
minutes = (runtime / 60) % 60
seconds = runtime % 60
print 'Finding the steady state took %.0f hours, %.0f minutes, and %.0f \
seconds.' % (abs(hours - .5), abs(minutes - .5), seconds)


'''
------------------------------------------------------------------------
 Generate graphs of the steady-state distribution of wealth
------------------------------------------------------------------------
domain     = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''

domain = np.linspace(1, S, S)

# 2D Graph
plt.figure(1)
plt.plot(domain, Kssvec, color='b', linewidth=2, label='Average capital stock')
plt.axhline(y=Kss, color='r', label='Steady State capital stock')
plt.title('Steady-state Distribution of Capital')
plt.legend(loc=0)
plt.savefig("capital_dist_2D")

# 3D Graph
my_cmap = matplotlib.cm.get_cmap('summer')
Kssmat2 = np.array(list(Kssmat) + list(np.zeros(J).reshape(1, J)))
Sgrid = np.linspace(1, S, S)
Jgrid = np.linspace(1, J, J)
X, Y = np.meshgrid(Sgrid, Jgrid)
fig = plt.figure(2)
ax1 = fig.gca(projection='3d')
ax1.set_xlabel('S')
ax1.set_ylabel('J')
ax1.set_zlabel('K')
ax1.set_title('Distribution of Capital Stock')
ax1.plot_surface(X, Y, Kssmat2.T, rstride=1, cstride=1, cmap=my_cmap)

plt.savefig('capital_dist_3D')

'''
------------------------------------------------------------------------
Generate graph of Consumption
------------------------------------------------------------------------
'''
Kssmat3 = np.array(list(np.zeros(J).reshape(1, J)) + list(Kssmat))
cssmat = (1 + rss) * Kssmat3 + wss * e * n.reshape(S, 1) - Kssmat2

# 2D Graph
plt.figure(3)
# plt.plot(domain, bsavg, label='Average capital stock')
plt.plot(domain, cssmat.mean(1), label='Consumption')
# plt.plot(domain, n * wss * e.mean(axis=1), label='Income')
plt.title('Consumption: S = {}'.format(S))
# plt.legend(loc=0)
plt.savefig("consumption_2D")

# 3D Graph
fig2 = plt.figure(4)
ax2 = fig2.gca(projection='3d')
ax2.plot_surface(X, Y, cssmat.T, rstride=1, cstride=1, cmap=my_cmap)
ax2.set_xlabel('S')
ax2.set_ylabel('J')
ax2.set_zlabel('C')
ax2.set_title('Distribution of Consumption')
plt.savefig('consumption_3D')


'''
------------------------------------------------------------------------
Check Euler Equations
------------------------------------------------------------------------

'''
k1 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat[:-1, :]))
k2 = Kssmat
k3 = np.array(list(Kssmat[1:, :]) + list(np.zeros(J).reshape((1, J))))

euler_justcapital = Euler_justcapital(wss, rss, f, e, n, k1, k2, k3)

# labor Leisure Only:
# k1_2 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat))
# k2_2 = np.array(list(Kssmat) + list(np.zeros(J).reshape((1, J))))
# euler1 = Euler1(wss, rss, f, e, Nssmat, k1, k2, k3)
# euler2 = Euler2(wss, rss, e, Nssmat, k1_2, k2_2)

plt.figure(5)
plt.plot(domain[1:], np.abs(euler_justcapital).max(1))

# Labor Leisure Only:
# plt.plot(domain[1:], np.abs(euler1).max(1), label='Capital')
# plt.plot(domain, np.abs(euler2).max(1), label='Labor')
# plt.legend(loc=0)

plt.title('Euler Errors')
plt.savefig('euler_errors')

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
