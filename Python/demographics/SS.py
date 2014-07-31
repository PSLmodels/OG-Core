'''
------------------------------------------------------------------------
Last updated: 7/29/2014

Calculates steady state of OLG model with S age cohorts

This py-file calls the following other file(s):
            income.py

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/ss_vars.pkl
            OUTPUT/ability_3D.png
            OUTPUT/capital_dist_2D.png
            OUTPUT/capital_dist_3D.png
            OUTPUT/consumption_2D.png
            OUTPUT/consumption_3D.png
            OUTPUT/euler_errors_SS_2D.png
            OUTPUT/euler_errors_euler1_SS_3D.png
            OUTPUT/euler_errors_euler2_SS_3D.png
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
import pickle
import demographics

'''
------------------------------------------------------------------------
Setting up the Model
------------------------------------------------------------------------
S      = number of periods an individual lives
J      = number of different ability groups
beta   = discount factor
sigma  = coefficient of relative risk aversion
alpha  = capital share of income
rho    = contraction parameter in steady state iteration process
         representing the weight on the new distribution gamma_new
A      = total factor productivity parameter in firms' production
         function
delta  = depreciation rate of capital
ctilde = minimum value amount of consumption
ltilde = measure of time each individual is endowed with each period
chi    = discount factor
eta    = Frisch elasticity of labor supply
e      = S x J matrix of age dependent possible working abilities e_s
T      = number of periods until the steady state
------------------------------------------------------------------------
'''

# Parameters
S = 10
J = 7
T = 70
beta = .96 ** (60.0 / S)
sigma = 3.0
alpha = .35
rho = .20
A = 1.0
delta = 1 - (0.95 ** (60.0 / S))
ctilde = .01
ltilde = 1.0
chi = 1.0
eta = 2.5
e = income.get_e(S, J)
omega = demographics.get_omega(S, J, T)

print 'The following are the parameter values of the simulation:'
print '\tS:\t\t\t', S
print '\tJ:\t\t\t', J
print '\tT:\t\t\t', T
print '\tBeta:\t\t', beta
print '\tSigma:\t\t', sigma
print '\tAlpha:\t\t', alpha
print '\tRho:\t\t', rho
print '\tA:\t\t\t', A
print '\tDelta:\t\t', delta
print '\tl-tilde:\t', ltilde
print '\tChi:\t\t', chi
print '\tEta:\t\t', eta

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
Kssmat2      = SxJ array of capital (zeros appended at the end of
               Kssmat2)
Kssmat3      = SxJ array of capital (zeros appended at the beginning of
               Kssmat)
Kssvec       = ((S-1) x 1) vector of the steady state level of capital
               (averaged across ability types)
Kss          = steady state aggregate capital stock
K_agg        = Aggregate level of capital
Nssmat       = (S x J) array of the steady state distribution of labor
Nssvec       = (S x 1) vector of the steady state level of labor
               (averaged across ability types)
Nss          = steady state aggregate labor
Yss          = steady state aggregate output
wss          = steady state real wage
rss          = steady state real rental rate
cssmat       = SxJ array of consumption across age and ability groups
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


def get_N(e, n):
    '''
    Parameters: e, n

    Returns:    Aggregate labor
    '''
    N_now = np.sum(e * omega[-1, :, :] * n)
    return N_now


def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''
    output = c**(-sigma)
    return output


def MUn(n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    output = - chi * ((ltilde-n) ** (-eta))
    return output


def Euler1(w, r, e, N_guess, K1, K2, K3):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        N_guess  = distribution of labor (SxJ array)
        K1       = distribution of capital in period t ((S-1) x J array)
        K2       = distribution of capital in period t+1 ((S-1) x J array)
        K3       = distribution of capital in period t+2 ((S-1) x J array)

    Returns:
        Value of Euler error.
    '''
    euler = MUc((1 + r)*K1 + w * e[:-1, :] * N_guess[:-1, :] - K2) - beta * (
        1 + r)*MUc((1 + r)*K2 + w * e[1:, :] * N_guess[1:, :] - K3)
    return euler


def Euler2(w, r, e, N_guess, K1_2, K2_2):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        N_guess  = distribution of labor (SxJ array)
        K1_2     = distribution of capital in period t (S x J array)
        K2_2     = distribution of capital in period t+1 (S x J array)

    Returns:
        Value of Euler error.
    '''
    euler = MUc((1 + r)*K1_2 + w * e * N_guess - K2_2) * w * e + MUn(N_guess)
    return euler


def Steady_State(guesses):
    '''
    Parameters: Steady state distribution of capital guess as array
                size S-1

    Returns:    Array of S-1 Euler equation errors
    '''
    K_guess = guesses[0: (S-1) * J].reshape((S-1, J))
    K = (omega[-1, 1:, :] * K_guess).sum()
    N_guess = guesses[(S-1) * J:].reshape((S, J))
    N = get_N(e, N_guess)
    Y = get_Y(K, N)
    w = get_w(Y, N)
    r = get_r(Y, K)
    K1 = np.array(list(np.zeros(J).reshape(1, J)) + list(K_guess[:-1, :]))
    K2 = K_guess
    K3 = np.array(list(K_guess[1:, :]) + list(np.zeros(J).reshape(1, J)))
    K1_2 = np.array(list(np.zeros(J).reshape(1, J)) + list(K_guess))
    K2_2 = np.array(list(K_guess) + list(np.zeros(J).reshape(1, J)))
    error1 = Euler1(w, r, e, N_guess, K1, K2, K3)
    error2 = Euler2(w, r, e, N_guess, K1_2, K2_2)
    # Check and punish constraing violations
    mask1 = N_guess < 0
    error2[mask1] += 1e9
    mask2 = N_guess > ltilde
    error2[mask2] += 1e9
    if K_guess.sum() <= 0:
        error1 += 1e9
    cons = (1 + r) * K1_2 + w * e * N_guess - K2_2
    mask3 = cons < 0
    error2[mask3] += 1e9
    return list(error1.flatten()) + list(error2.flatten())


def borrowing_constraints(K_dist, w, r, e, n):
    '''
    Parameters:
        K_dist = Distribution of capital ((S-1)xJ array)
        w      = wage rate (scalar)
        r      = rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        n      = distribution of labor (SxJ array)

    Returns:
        False value if all the borrowing constraints are met, True
            if there are violations.
    '''
    b_min = np.zeros((S-1, J))
    b_min[-1, :] = (ctilde - w * e[S-1, :] * n[S-1, :]) / (1 + r)
    for i in xrange(S-2):
        b_min[-(i+2), :] = (ctilde + b_min[-(i+1), :] - w * e[
            -(i+2), :] * n[-(i+2), :]) / (1 + r)
    difference = K_dist - b_min
    if (difference < 0).any():
        return True
    else:
        return False


def constraint_checker(Kssmat, Nssmat, wss, rss, e, cssmat):
    '''
    Parameters:
        Kssmat = steady state distribution of capital ((S-1)xJ array)
        Nssmat = steady state distribution of labor (SxJ array)
        wss    = steady state wage rate (scalar)
        rss    = steady state rental rate (scalar)
        e      = distribution of abilities (SxJ array)
        cssmat = steady state distribution of consumption (SxJ array)

    Created Variables:
        flag1 = False if all borrowing constraints are met, true
               otherwise.
        flag2 = False if all labor constraints are met, true otherwise

    Returns:
        Prints warnings for violations of capital, labor, and
            consumption constraints.
    '''
    print 'Checking constraints on capital, labor, and consumption.'
    flag1 = False
    if Kssmat.sum() <= 0:
        print '\tWARNING: Aggregate capital is less than or equal to zero.'
        flag1 = True
    if borrowing_constraints(Kssmat, wss, rss, e, Nssmat) is True:
        print '\tWARNING: Borrowing constraints have been violated.'
        flag1 = True
    if flag1 is False:
        print '\tThere were no violations of the borrowing constraints.'
    flag2 = False
    if (Nssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (Nssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Conusmption volates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'

starttime = time.time()

K_guess_init = np.ones((S-1, J)) * .05
N_guess_init = np.ones((S, J)) * .95
guesses = list(K_guess_init.flatten()) + list(N_guess_init.flatten())

print 'Solving for steady state level distribution of capital and labor.'
solutions = opt.fsolve(Steady_State, guesses, xtol=1e-9, col_deriv=1)
print '\tFinished.'

runtime = time.time() - starttime
hours = runtime / 3600
minutes = (runtime / 60) % 60
seconds = runtime % 60
print 'Finding the steady state took %.0f hours, %.0f minutes, and %.0f \
seconds.' % (abs(hours - .5), abs(minutes - .5), seconds)

Kssmat = solutions[0:(S-1) * J].reshape(S-1, J)
Kssmat2 = np.array(list(np.zeros(J).reshape(1, J)) + list(Kssmat))
Kssmat3 = np.array(list(Kssmat) + list(np.zeros(J).reshape(1, J)))

Kssvec = Kssmat.sum(1)
Kss = (omega[-1, 1:, :] * Kssmat).sum()
Kssavg = Kssvec.mean()
K_agg = Kssmat.sum()
Kssvec = np.array([0]+list(Kssvec))
Nssmat = solutions[(S-1) * J:].reshape(S, J)
Nssvec = Nssmat.sum(1)
Nss = get_N(e, Nssmat)
Nssavg = Nssvec.mean()
Yss = get_Y(Kss, Nss)
wss = get_w(Yss, Nss)
rss = get_r(Yss, Kss)

cssmat = (1 + rss) * Kssmat2 + wss * e * Nssmat - Kssmat3


constraint_checker(Kssmat, Nssmat, wss, rss, e, cssmat)

print 'The steady state values for:'
print "\tCapital:\t\t", Kss
print "\tLabor:\t\t\t", Nss
print "\tOutput:\t\t\t", Yss
print "\tWage:\t\t\t", wss
print "\tRental Rate:\t", rss

'''
------------------------------------------------------------------------
 Generate graphs of the steady-state distribution of wealth
------------------------------------------------------------------------
domain     = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''

print 'Generating steady state graphs.'
domain = np.linspace(1, S, S)

# 2D Graph
plt.figure(1)
plt.plot(domain, Kssvec, color='b', linewidth=2, label='Average capital stock')
plt.axhline(y=Kssavg, color='r', label='Steady state capital stock')
plt.title('Steady-state Distribution of Capital')
plt.legend(loc=0)
plt.xlabel(r'Age Cohorts $S$')
plt.ylabel('Capital')
plt.savefig("OUTPUT/capital_dist_2D")

# 3D Graph
cmap1 = matplotlib.cm.get_cmap('summer')
Sgrid = np.linspace(1, S, S)
Jgrid = np.linspace(1, J, J)
X, Y = np.meshgrid(Sgrid, Jgrid)
fig2 = plt.figure(2)
ax2 = fig2.gca(projection='3d')
ax2.set_xlabel(r'age-$s$')
ax2.set_ylabel(r'ability-$j$')
ax2.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
# ax2.set_title(r'Steady State Distribution of Capital Stock $K$')
ax2.plot_surface(X, Y, Kssmat2.T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('OUTPUT/capital_dist_3D')

'''
------------------------------------------------------------------------
 Generate graphs of the steady-state distribution of labor
------------------------------------------------------------------------
'''

# 2D Graph
plt.figure(3)
plt.plot(domain, Nssvec, color='b', linewidth=2, label='Average Labor Supply')
plt.axhline(y=Nssavg, color='r', label='Steady state labor supply')
plt.title('Steady-state Distribution of Labor')
plt.legend(loc=0)
plt.xlabel(r'Age Cohorts $S$')
plt.ylabel('Labor')
plt.savefig("OUTPUT/labor_dist_2D")

# 3D Graph
fig4 = plt.figure(4)
ax4 = fig4.gca(projection='3d')
ax4.set_xlabel(r'age-$s$')
ax4.set_ylabel(r'ability-$j$')
ax4.set_zlabel(r'individual labor supply $\bar{n}_{j,s}$')
# ax4.set_title(r'Steady State Distribution of Labor Supply $K$')
ax4.plot_surface(X, Y, Nssmat.T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('OUTPUT/labor_dist_3D')

'''
------------------------------------------------------------------------
Generate graph of Consumption
------------------------------------------------------------------------
'''

# 2D Graph
plt.figure(5)
plt.plot(domain, cssmat.mean(1), label='Consumption')
plt.title('Consumption across cohorts: S = {}'.format(S))
# plt.legend(loc=0)
plt.xlabel('Age cohorts')
plt.ylabel('Consumption')
plt.savefig("OUTPUT/consumption_2D")

# 3D Graph
cmap2 = matplotlib.cm.get_cmap('jet')
fig6 = plt.figure(6)
ax6 = fig6.gca(projection='3d')
ax6.plot_surface(X, Y, cssmat.T, rstride=1, cstride=1, cmap=cmap2)
ax6.set_xlabel(r'age-$s$')
ax6.set_ylabel(r'ability-$j$')
ax6.set_zlabel('Consumption')
ax6.set_title('Steady State Distribution of Consumption')
plt.savefig('OUTPUT/consumption_3D')

'''
------------------------------------------------------------------------
Graph of Distribution of Income
------------------------------------------------------------------------
'''

# 3D Graph
cmap2 = matplotlib.cm.get_cmap('summer')
fig7 = plt.figure(7)
ax7 = fig7.gca(projection='3d')
ax7.plot_surface(X, Y, e.T, rstride=1, cstride=2, cmap=cmap2)
ax7.set_xlabel(r'age-$s$')
ax7.set_ylabel(r'ability-$j$')
ax7.set_zlabel(r'Income Level $e_j(s)$')
# ax7.set_title('Income Levels')
plt.savefig('OUTPUT/ability_3D')

'''
------------------------------------------------------------------------
Check Euler Equations
------------------------------------------------------------------------
k1          = (S-1)xJ array of Kssmat in period t-1
k2          = copy of Kssmat
k3          = (S-1)xJ array of Kssmat in period t+1
k1_2        = SxJ array of Kssmat in period t
k2_2        = SxJ array of Kssmat in period t+1
euler1      = euler errors from first euler equation
euler2      = euler errors from second euler equation
------------------------------------------------------------------------
'''

k1 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat[:-1, :]))
k2 = Kssmat
k3 = np.array(list(Kssmat[1:, :]) + list(np.zeros(J).reshape((1, J))))
k1_2 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat))
k2_2 = np.array(list(Kssmat) + list(np.zeros(J).reshape((1, J))))

euler1 = Euler1(wss, rss, e, Nssmat, k1, k2, k3)
euler2 = Euler2(wss, rss, e, Nssmat, k1_2, k2_2)

# 2D Graph
plt.figure(8)
plt.plot(domain[1:], np.abs(euler1).max(1), label='Euler1')
plt.plot(domain, np.abs(euler2).max(1), label='Euler2')
plt.legend(loc=0)
plt.title('Euler Errors')
plt.savefig('OUTPUT/euler_errors_SS_2D')

# 3D Graph
X2, Y2 = np.meshgrid(Sgrid[1:], Jgrid)

fig9 = plt.figure(9)
ax9 = fig9.gca(projection='3d')
ax9.plot_surface(X2, Y2, euler1.T, rstride=1, cstride=2, cmap=cmap2)
ax9.set_xlabel(r'Age Cohorts $S$')
ax9.set_ylabel(r'Ability Types $J$')
ax9.set_zlabel('Error Level')
ax9.set_title('Euler Errors')
plt.savefig('OUTPUT/euler_errors_euler1_SS_3D')

fig10 = plt.figure(10)
ax10 = fig10.gca(projection='3d')
ax10.plot_surface(X, Y, euler2.T, rstride=1, cstride=2, cmap=cmap2)
ax10.set_xlabel(r'Age Cohorts $S$')
ax10.set_ylabel(r'Ability Types $J$')
ax10.set_zlabel('Error Level')
ax10.set_title('Euler Errors')
plt.savefig('OUTPUT/euler_errors_euler2_SS_3D')

print '\tFinished.'

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

print 'Saving steady state variable values.'
var_names = ['S', 'beta', 'sigma', 'alpha', 'rho', 'A', 'delta', 'e',
             'J', 'Kss', 'Kssvec', 'Kssmat', 'Nss', 'Nssvec', 'Nssmat',
             'Yss', 'wss', 'rss', 'runtime', 'hours', 'minutes', 'omega',
             'seconds', 'eta', 'chi', 'K_agg', 'ltilde', 'ctilde', 'T']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/ss_vars.pkl", "w"))
print '\tFinished.'
