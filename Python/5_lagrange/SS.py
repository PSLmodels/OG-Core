'''
------------------------------------------------------------------------
Last updated: 9/19/2014

Calculates steady state of OLG model with S age cohorts

This py-file calls the following other file(s):
            income.py
            demographics.py
            OUTPUT/given_params.pkl

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/ss_vars.pkl
            OUTPUT/capital_dist.png
            OUTPUT/consumption.png
            OUTPUT/labor_dist.png
            OUTPUT/lambdamultiplier.png
            OUTPUT/chi_n.png
            OUTPUT/intentional_bequests.png
            OUTPUT/euler_errors_euler1_SS.png
            OUTPUT/euler_errors_euler2_SS.png
            OUTPUT/euler_errors_euler3_SS.png (if J != 1, otherwise, it
                is printed to the screen)
            OUTPUT/euler_errors_euler4_SS.png
            OUTPUT/euler_errors1and2and4_SS.png
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.optimize as opt
import pickle
import income
import demographics


'''
------------------------------------------------------------------------
Imported user given values
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
bin_weights  = percent of each age cohort in each ability group
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
nu_init      = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_new
A            = total factor productivity parameter in firms' production
               function
delta        = depreciation rate of capital for each cohort
ctilde       = minimum value amount of consumption
bqtilde      = minimum bequest value
ltilde       = measure of time each individual is endowed with each
               period
chi_n        = discount factor of labor that changes with S (Sx1 array)
chi_b        = discount factor of incidental bequests
eta          = Frisch elasticity of labor supply
g_y          = growth rate of technology for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

'''
------------------------------------------------------------------------
Generate income and demographic parameters
------------------------------------------------------------------------
e            = S x J matrix of age dependent possible working abilities
               e_s
omega        = T x S x J array of demographics
g_n          = steady state population growth rate
omega_SS     = steady state population distribution
children     = T x starting_age x J array of children demographics
surv_rate    = S x 1 array of survival rates
mort_rate    = S x 1 array of mortality rates
slow_work    = time at which chi_n starts increasing from 1
------------------------------------------------------------------------
'''

print 'Generating income distribution.'
e = income.get_e(S, J, starting_age, ending_age, bin_weights)
print '\tFinished.'
print 'Generating demographics.'
omega, g_n, omega_SS, children, surv_rate = demographics.get_omega(
    S, J, T, bin_weights, starting_age, ending_age, E)
mort_rate = 1-surv_rate
slow_work = np.round(7.0 * S / 16.0)
chi_n_multiplier = 14

# increase expentially
chi_n[slow_work:] = (mort_rate[slow_work:] + 1 - mort_rate[slow_work])**chi_n_multiplier
# develop an initial guess for the lambda multipliers
lambdy_scalar = np.ones(S) * 1e-6
lambdy_scalar[slow_work:] = np.ones(S-slow_work) * chi_n_multiplier * 5

# Force individuals to die in the last period
surv_rate[-1] = 0.0
mort_rate[-1] = 1

print '\tFinished.'

print 'The following are the parameter values of the simulation:'
print '\tS:\t\t\t\t', S
print '\tJ:\t\t\t\t', J
print '\tT:\t\t\t\t', T
print '\tStarting Age:\t', starting_age
print '\tbeta:\t\t\t', beta
print '\tsigma:\t\t\t', sigma
print '\talpha:\t\t\t', alpha
print '\tnu:\t\t\t\t', nu_init
print '\tA:\t\t\t\t', A
print '\tdelta:\t\t\t', delta
print '\tl-tilde:\t\t', ltilde
print '\tchi_n:\t\t\tSee graph'
print '\tchi_b:\t\t\t', chi_b
print '\teta:\t\t\t', eta
print '\tg_n:\t\t\t', g_n
print '\tg_y:\t\t\t', g_y

'''
------------------------------------------------------------------------
Finding the Steady State
------------------------------------------------------------------------
K_guess_init = (S-1 x J) array for the initial guess of the distribution
               of capital
L_guess_init = (S x J) array for the initial guess of the distribution
               of labor
lambdy       = (S x J) array of lambda multipliers
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
Lssmat       = (S x J) array of the steady state distribution of labor
Lssvec       = (S x 1) vector of the steady state level of labor
               (averaged across ability types)
Lss          = steady state aggregate labor
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


def get_Y(K_now, L_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = A * (K_now ** alpha) * ((L_now) ** (1 - alpha))
    return Y_now


def get_w(Y_now, L_now):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    w_now = (1 - alpha) * Y_now / L_now
    return w_now


def get_r(Y_now, K_now):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    r_now = (alpha * Y_now / K_now) - delta
    return r_now


def get_L(e, n):
    '''
    Parameters: e, n

    Returns:    Aggregate labor
    '''
    L_now = np.sum(e * omega_SS * n)
    return L_now


def MUc(c):
    '''
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    '''
    output = c**(-sigma)
    return output


def MUl(n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    output = - chi_n.reshape(S, 1) * ((ltilde-n) ** (-eta))
    return output


def MUb(bq):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b * (bq ** (-sigma))
    return output


def Euler1(w, r, e, L_guess, K1, K2, K3, B):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K1       = distribution of capital in period t ((S-1) x J array)
        K2       = distribution of capital in period t+1 ((S-1) x J array)
        K3       = distribution of capital in period t+2 ((S-1) x J array)
        B        = distribution of incidental bequests (1 x J array)

    Returns:
        Value of Euler error.
    '''
    euler = MUc((1 + r)*K1 + w * e[:-1, :] * L_guess[:-1, :] + B.reshape(1, J) / bin_weights - K2 * np.exp(
        g_y)) - beta * surv_rate[:-1].reshape(S-1, 1) * (
        1 + r)*MUc((1 + r)*K2 + w * e[1:, :] * L_guess[1:, :] + B.reshape(1, J) / bin_weights - K3 * np.exp(
            g_y)) * np.exp(-sigma * g_y)
    return euler


def Euler2(w, r, e, L_guess, K1_2, K2_2, B, lambdy):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K1_2     = distribution of capital in period t (S x J array)
        K2_2     = distribution of capital in period t+1 (S x J array)
        lambdy   = distribution of lambda multipliers (S x J array)
        B        = distribution of incidental bequests (1 x J array)

    Returns:
        Value of Euler error.
    '''
    euler = MUc((1 + r)*K1_2 + w * e * L_guess + B.reshape(1, J) / bin_weights - K2_2 * 
        np.exp(g_y)) * w * e + MUl(L_guess) + lambdy
    return euler


def Euler3(w, r, e, L_guess, K_guess, B):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        L_guess  = distribution of labor (SxJ array)
        K_guess  = distribution of capital in period t (S-1 x J array)
        B        = distribution of incidental bequests (1 x J array)

    Returns:
        Value of Euler error.
    '''
    euler = MUc((1 + r)*K_guess[-2, :] + w * e[-1, :] * L_guess[-1, :] + B.reshape(1, J) / bin_weights - K_guess[-1, :] * 
        np.exp(g_y)) - np.exp(-sigma * g_y) * MUb(K_guess[-1, :])
    return euler


def Steady_State(guesses):
    '''
    Parameters: Steady state distribution of capital guess as array
                size S-1

    Returns:    Array of S-1 Euler equation errors
    '''
    K_guess = guesses[0: S * J].reshape((S, J))
    B = (K_guess * omega_SS * mort_rate.reshape(S, 1)).sum(0)
    K = (omega_SS * K_guess).sum()
    L_guess = guesses[S * J:2*S*J].reshape((S, J))
    lambdy = guesses[2*S*J:].reshape((S, J))
    L_guess = L_guess ** 2
    lambdy = lambdy ** 2
    L = get_L(e, L_guess)
    Y = get_Y(K, L)
    w = get_w(Y, L)
    r = get_r(Y, K)
    BQ = (1 + r) * B
    K1 = np.array(list(np.zeros(J).reshape(1, J)) + list(K_guess[:-2, :]))
    K2 = K_guess[:-1, :]
    K3 = K_guess[1:, :]
    K1_2 = np.array(list(np.zeros(J).reshape(1, J)) + list(K_guess[:-1, :]))
    K2_2 = K_guess
    error1 = Euler1(w, r, e, L_guess, K1, K2, K3, BQ)
    error2 = Euler2(w, r, e, L_guess, K1_2, K2_2, BQ, lambdy)
    error3 = Euler3(w, r, e, L_guess, K_guess, BQ)
    error4 = lambdy * L_guess
    # Check and punish constraing violations
    mask1 = L_guess < 0
    error2[mask1] += 1e9
    mask2 = L_guess > ltilde
    error2[mask2] += 1e9
    if K_guess.sum() <= 0:
        error1 += 1e9
    cons = (1 + r) * K1_2 + w * e * L_guess + BQ.reshape(1, J) / bin_weights - K2_2 * np.exp(g_y)
    mask3 = cons < 0
    error2[mask3] += 1e9
    return list(error1.flatten()) + list(error2.flatten()) + list(error3.flatten()) + list(error4.flatten())


def borrowing_constraints(K_dist, w, r, e, n, BQ):
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
    b_min[-1, :] = (ctilde + bqtilde - w * e[S-1, :] * ltilde - BQ.reshape(1, J) / bin_weights) / (1 + r)
    for i in xrange(S-2):
        b_min[-(i+2), :] = (ctilde + np.exp(g_y) * b_min[-(i+1), :] - w * e[
            -(i+2), :] * ltilde - BQ.reshape(1, J) / bin_weights) / (1 + r)
    difference = K_dist - b_min
    if (difference < 0).any():
        return True
    else:
        return False


def constraint_checker(Kssmat, Lssmat, wss, rss, e, cssmat, BQ):
    '''
    Parameters:
        Kssmat = steady state distribution of capital ((S-1)xJ array)
        Lssmat = steady state distribution of labor (SxJ array)
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
    if borrowing_constraints(Kssmat, wss, rss, e, Lssmat, BQ) is True:
        print '\tWARNING: Borrowing constraints have been violated.'
        flag1 = True
    if flag1 is False:
        print '\tThere were no violations of the borrowing constraints.'
    flag2 = False
    if (Lssmat < 0).any():
        print '\tWARNING: Labor supply violates nonnegativity constraints.'
        flag2 = True
    if (Lssmat > ltilde).any():
        print '\tWARNING: Labor suppy violates the ltilde constraint.'
    if flag2 is False:
        print '\tThere were no violations of the constraints on labor supply.'
    if (cssmat < 0).any():
        print '\tWARNING: Conusmption volates nonnegativity constraints.'
    else:
        print '\tThere were no violations of the constraints on consumption.'

starttime = time.time()

K_guess_init = np.ones((S, J)) * .05
L_guess_init = np.ones((S, J)) * .95
lambdy = np.ones((S, J)) * lambdy_scalar.reshape(S, 1)
guesses = list(K_guess_init.flatten()) + list(L_guess_init.flatten()) + list(lambdy.flatten())

print 'Solving for steady state level distribution of capital and labor.'
solutions = opt.fsolve(Steady_State, guesses, xtol=1e-11)
print '\tFinished.'
print np.abs(np.array(Steady_State(solutions))).max()

runtime = time.time() - starttime
hours = runtime / 3600
minutes = (runtime / 60) % 60
seconds = runtime % 60
print 'Finding the steady state took %.0f hours, %.0f minutes, and %.0f \
seconds.' % (abs(hours - .5), abs(minutes - .5), seconds)

Kssmat = solutions[0:(S-1) * J].reshape(S-1, J)
BQ = solutions[(S-1)*J:S*J]
Bss = (np.array(list(Kssmat) + list(BQ.reshape(1, J))).reshape(S, J) * omega_SS * mort_rate.reshape(S, 1)).sum(0)
Kssmat2 = np.array(list(np.zeros(J).reshape(1, J)) + list(Kssmat))
Kssmat3 = np.array(list(Kssmat) + list(BQ.reshape(1, J)))

Kssvec = Kssmat.sum(1)
Kss = (omega_SS[:-1, :] * Kssmat).sum() + (omega_SS[-1,:]*BQ).sum()
Kssavg = Kssvec.mean()
Kssvec = np.array([0]+list(Kssvec))
Lssmat = solutions[S * J:2*S*J].reshape(S, J) ** 2
Lssvec = Lssmat.sum(1)
Lss = get_L(e, Lssmat)
Lssavg = Lssvec.mean()
Yss = get_Y(Kss, Lss)
wss = get_w(Yss, Lss)
rss = get_r(Yss, Kss)

lambdy = solutions[2*S*J:].reshape(S, J) ** 2

cssmat = (1 + rss) * Kssmat2 + wss * e * Lssmat + (1 + rss) * Bss.reshape(1, J)/bin_weights.reshape(1,J) - np.exp(g_y) * Kssmat3


constraint_checker(Kssmat, Lssmat, wss, rss, e, cssmat, BQ)

print 'The steady state values for:'
print "\tCapital:\t\t", Kss
print "\tLabor:\t\t\t", Lss
print "\tOutput:\t\t\t", Yss
print "\tWage:\t\t\t", wss
print "\tRental Rate:\t", rss
print "\tBequest:\t\t", Bss.sum()

'''
------------------------------------------------------------------------
Graph of Chi_n
------------------------------------------------------------------------
'''

plt.figure()
plt.plot(np.arange(S)+21, chi_n)
plt.xlabel(r'Age cohort - $s$')
plt.ylabel(r'$\chi _n$')
plt.savefig('OUTPUT/chi_n')

'''
------------------------------------------------------------------------
 Generate graphs of the steady-state distribution of wealth
------------------------------------------------------------------------
domain     = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''

print 'Generating steady state graphs.'
domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += bin_weights[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)

if J == 1:
    # 2D Graph
    plt.figure()
    plt.plot(domain, Kssvec, color='b', linewidth=2, label='Average capital stock')
    plt.axhline(y=Kssavg, color='r', label='Steady state capital stock')
    plt.title('Steady-state Distribution of Capital')
    plt.legend(loc=0)
    plt.xlabel(r'Age Cohorts $S$')
    plt.ylabel('Capital')
    plt.savefig("OUTPUT/capital_dist")
else:
    # 3D Graph
    # Jgrid = np.linspace(1, J, J)
    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
    # ax5.set_title(r'Steady State Distribution of Capital Stock $K$')
    ax5.plot_surface(X, Y, Kssmat2.T, rstride=1, cstride=1, cmap=cmap2)
    plt.savefig('OUTPUT/capital_dist')

'''
------------------------------------------------------------------------
 Generate graph of the steady-state distribution of intentional bequests
------------------------------------------------------------------------
'''
if J == 1:
    print '\tIntentional bequests:', BQ
else:
    plt.figure()
    plt.plot(np.arange(J)+1, BQ)
    plt.xlabel(r'ability-$j$')
    plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
    plt.savefig('OUTPUT/intentional_bequests')

'''
------------------------------------------------------------------------
 Generate graphs of the steady-state distribution of labor
------------------------------------------------------------------------
'''
if J == 1:
    # 2D Graph
    plt.figure()
    plt.plot(domain, Lssvec, color='b', linewidth=2, label='Average Labor Supply')
    plt.axhline(y=Lssavg, color='r', label='Steady state labor supply')
    plt.title('Steady-state Distribution of Labor')
    plt.legend(loc=0)
    plt.xlabel(r'Age Cohorts $S$')
    plt.ylabel('Labor')
    plt.savefig("OUTPUT/labor_dist")
else:
    # 3D Graph
    fig4 = plt.figure()
    ax4 = fig4.gca(projection='3d')
    ax4.set_xlabel(r'age-$s$')
    ax4.set_ylabel(r'ability-$j$')
    ax4.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
    # ax4.set_title(r'Steady State Distribution of Labor Supply $K$')
    ax4.plot_surface(X, Y, (Lssmat).T, rstride=1, cstride=1, cmap=cmap1)
    plt.savefig('OUTPUT/labor_dist')

'''
------------------------------------------------------------------------
Generate graph of Consumption
------------------------------------------------------------------------
'''
if J == 1:
    # 2D Graph
    plt.figure()
    plt.plot(domain, cssmat.mean(1), label='Consumption')
    plt.title('Consumption across cohorts: S = {}'.format(S))
    # plt.legend(loc=0)
    plt.xlabel('Age cohorts')
    plt.ylabel('Consumption')
    plt.savefig("OUTPUT/consumption")
else:
    # 3D Graph
    fig9 = plt.figure()
    ax9 = fig9.gca(projection='3d')
    ax9.plot_surface(X, Y, cssmat.T, rstride=1, cstride=1, cmap=cmap2)
    ax9.set_xlabel(r'age-$s$')
    ax9.set_ylabel(r'ability-$j$')
    ax9.set_zlabel('Consumption')
    ax9.set_title('Steady State Distribution of Consumption')
    plt.savefig('OUTPUT/consumption')

if J>1:
    fig99 = plt.figure()
    ax99 = fig99.gca(projection='3d')
    ax99.plot_surface(X, Y, lambdy.T, rstride=1, cstride=1, cmap=cmap2)
    ax99.set_xlabel(r'age-$s$')
    ax99.set_ylabel(r'ability-$j$')
    ax99.set_zlabel('Lambda')
    ax99.set_title('Steady State Distribution of Multipliers')
    plt.savefig('OUTPUT/lamdamultiplier')
else:
    plt.figure()
    plt.plot(domain, lambdy)
    plt.savefig('OUTPUT/lamdamultiplier')

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
k3 = np.array(list(Kssmat[1:, :]) + list(BQ.reshape(1, J)))
k1_2 = np.array(list(np.zeros(J).reshape((1, J))) + list(Kssmat))
k2_2 = np.array(list(Kssmat) + list(BQ.reshape(1, J)))
B = Bss * (1+rss)

K_eul3 = np.zeros((S, J))
K_eul3[:S-1, :] = Kssmat
K_eul3[-1, :] = BQ

euler1 = Euler1(wss, rss, e, Lssmat, k1, k2, k3, B)
euler2 = Euler2(wss, rss, e, Lssmat, k1_2, k2_2, B, lambdy)
euler3 = Euler3(wss, rss, e, Lssmat, K_eul3, B)
euler4 = lambdy * Lssmat


# 3D Graph
X2, Y2 = np.meshgrid(domain[1:], Jgrid)

if J == 1:
    # 2D Graph
    plt.figure()
    plt.plot(domain[1:], np.abs(euler1), label='Euler1')
    plt.plot(domain, np.abs(euler2), label='Euler2')
    plt.plot(domain, np.abs(euler4), label='Euler4')
    plt.legend(loc=0)
    plt.title('Euler Errors')
    plt.xlabel(r'age cohort-$s$')
    plt.savefig('OUTPUT/euler_errors1and2and4_SS')
    print '\tEuler3=', euler3
else:
    fig16 = plt.figure()
    ax16 = fig16.gca(projection='3d')
    ax16.plot_surface(X2, Y2, euler1.T, rstride=1, cstride=2, cmap=cmap2)
    ax16.set_xlabel(r'Age Cohorts $S$')
    ax16.set_ylabel(r'Ability Types $J$')
    ax16.set_zlabel('Error Level')
    ax16.set_title('Euler Errors')
    plt.savefig('OUTPUT/euler_errors_euler1_SS')

    fig17 = plt.figure()
    ax17 = fig17.gca(projection='3d')
    ax17.plot_surface(X, Y, euler2.T, rstride=1, cstride=2, cmap=cmap2)
    ax17.set_xlabel(r'Age Cohorts $S$')
    ax17.set_ylabel(r'Ability Types $J$')
    ax17.set_zlabel('Error Level')
    ax17.set_title('Euler Errors')
    plt.savefig('OUTPUT/euler_errors_euler2_SS')

    fig18 = plt.figure()
    ax18 = fig18.gca(projection='3d')
    ax18.plot_surface(X, Y, euler4.T, rstride=1, cstride=2, cmap=cmap2)
    ax18.set_xlabel(r'Age Cohorts $S$')
    ax18.set_ylabel(r'Ability Types $J$')
    ax18.set_zlabel('Error Level')
    ax18.set_title('Euler Errors')
    plt.savefig('OUTPUT/euler_errors_euler4_SS')

print '\tFinished.'

'''
------------------------------------------------------------------------
Save variables/values so they can be used in other modules
------------------------------------------------------------------------
'''

print 'Saving steady state variable values.'
var_names = ['S', 'beta', 'sigma', 'alpha', 'nu_init', 'A', 'delta', 'e', 'E',
             'J', 'Kss', 'Kssvec', 'Kssmat', 'Lss', 'Lssvec', 'Lssmat',
             'Yss', 'wss', 'rss', 'runtime', 'hours', 'minutes', 'omega',
             'seconds', 'eta', 'chi_n', 'chi_b', 'ltilde', 'ctilde', 'T',
             'g_n', 'g_y', 'omega_SS', 'TPImaxiter', 'TPImindist', 'BQ',
             'children', 'surv_rate', 'mort_rate', 'Bss', 'bin_weights',
             'bqtilde', 'lambdy', 'slow_work']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/ss_vars.pkl", "w"))
print '\tFinished.'
