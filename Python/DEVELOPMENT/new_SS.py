# Import packages
import numpy as np
import time
import os
import scipy.optimize as opt
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import income
import demographics
import tax_funcs as tax


#Parameters from other files
variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
print S
print J
print T
print starting_age
print ending_age
print E

omega, g_n, omega_SS, surv_rate = demographics.get_omega(
        S, J, T, lambdas, starting_age, ending_age, E)
e = income.get_e(S, J, starting_age, ending_age, lambdas, omega_SS)
rho = 1-surv_rate

chi_n = np.array([47.12000874 , 22.22762421 , 14.34842241 , 10.67954008 ,  8.41097278
 ,  7.15059004 ,  6.46771332 ,  5.85495452 ,  5.46242013 ,  5.00364263
 ,  4.57322063 ,  4.53371545 ,  4.29828515 ,  4.10144524 ,  3.8617942  ,  3.57282
 ,  3.47473172 ,  3.31111347 ,  3.04137299 ,  2.92616951 ,  2.58517969
 ,  2.48761429 ,  2.21744847 ,  1.9577682  ,  1.66931057 ,  1.6878927
 ,  1.63107201 ,  1.63390543 ,  1.5901486  ,  1.58143606 ,  1.58005578
 ,  1.59073213 ,  1.60190899 ,  1.60001831 ,  1.67763741 ,  1.70451784
 ,  1.85430468 ,  1.97291208 ,  1.97017228 ,  2.25518398 ,  2.43969757
 ,  3.21870602 ,  4.18334822 ,  4.97772026 ,  6.37663164 ,  8.65075992
 ,  9.46944758 , 10.51634777 , 12.13353793 , 11.89186997 , 12.07083882
 , 13.2992811  , 14.07987878 , 14.19951571 , 14.97943562 , 16.05601334
 , 16.42979341 , 16.91576867 , 17.62775142 , 18.4885405  , 19.10609921
 , 20.03988031 , 20.86564363 , 21.73645892 , 22.6208256  , 23.37786072
 , 24.38166073 , 25.22395387 , 26.21419653 , 27.05246704 , 27.86896121
 , 28.90029708 , 29.83586775 , 30.87563699 , 31.91207845 , 33.07449767
 , 34.27919965 , 35.57195873 , 36.95045988 , 38.62308152])

rho[-1] = 1


# Define functions
def get_Y(K_now, L_now):
    Y_now = Z * (K_now ** alpha) * ((L_now) ** (1 - alpha))
    return Y_now

def get_L(e, n):
    L_now = np.sum(e * omega_SS * n)
    return L_now

def get_w(Y_now, L_now):
    w_now = (1 - alpha) * Y_now / L_now
    return w_now

def get_r(Y_now, K_now):
    r_now = (alpha * Y_now / K_now) - delta
    return r_now

def MUc(c):
    output = c**(-sigma)
    return output

def MUl(n, chi_n):
    deriv = b_ellipse * (1/ltilde) * ((1 - (n / ltilde) ** upsilon) ** (
        (1/upsilon)-1)) * (n / ltilde) ** (upsilon - 1)
    output = chi_n * deriv
    return output

def MUb(chi_b, bequest):
    output = chi_b * (bequest ** (-sigma))
    return output

def get_cons(r, b1, w, e, n, BQ, lambdas, b2, g_y, net_tax):
    cons = (1 + r)*b1 + w*e*n + BQ / lambdas - b2*np.exp(g_y) - net_tax
    return cons


# Euler Equations
def Solver(guesses, r, w, BQ, T_H, e, chi_b, j):
    b = np.array(guesses[:len(guesses)/2])
    n = np.array(guesses[len(guesses)/2:])

    error1 = Euler1(w, r, e, n, np.append([0],b[:-2]), b[:-1], b[1:], BQ, factor, T_H, chi_b, j)

    error2 = Euler2(w, r, e, n, np.append([0],b[:-1]), b, BQ, factor, T_H, chi_n, j)

    error3 = Euler3(w, r, e, n, b, BQ, factor, chi_b, T_H, j)

    return list(error1.flatten()) + list(error2.flatten()) + list(error3.flatten())

def Euler1(w, r, e, n_guess, b1, b2, b3, BQ, factor, T_H, chi_b, j):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b1       = distribution of capital in period t ((S-1) x J array)
        b2       = distribution of capital in period t+1 ((S-1) x J array)
        b3       = distribution of capital in period t+2 ((S-1) x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        T_H  = lump sum transfer from the government to the households
        xi       = coefficient of relative risk aversion
        chi_b    = discount factor of savings

    Returns:
        Value of Euler error.
    '''
    tax1 = tax.total_taxes_SS(r, b1, w, e[:-1], n_guess[:-1], BQ, lambdas[j], factor, T_H, j)
    tax2 = tax.total_taxes_SS2(r, b2, w, e[1:], n_guess[1:], BQ, lambdas[j], factor, T_H, j)
    cons1 = get_cons(r, b1, w, e[:-1], n_guess[:-1], BQ, lambdas[j], b2, g_y, tax1)
    cons2 = get_cons(r, b2, w, e[1:], n_guess[1:], BQ, lambdas[j], b3, g_y, tax2)
    income = (r * b2 + w * e[1:] * n_guess[1:]) * factor
    deriv = (
        1 + r*(1-tax.tau_income(r, b1, w, e[1:], n_guess[1:], factor)-tax.tau_income_deriv(
            r, b1, w, e[1:], n_guess[1:], factor)*income)-tax.tau_w_prime(b2)*b2-tax.tau_wealth(b2))
    bequest_ut = rho[:-1] * np.exp(-sigma * g_y) * chi_b * b2 ** (-sigma)
    euler = MUc(cons1) - beta * (1-rho[:-1]) * deriv * MUc(
        cons2) * np.exp(-sigma * g_y) - bequest_ut
    return euler


def Euler2(w, r, e, n_guess, b1_2, b2_2, BQ, factor, T_H, chi_n, j):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b1_2     = distribution of capital in period t (S x J array)
        b2_2     = distribution of capital in period t+1 (S x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        T_H  = lump sum transfer from the government to the households

    Returns:
        Value of Euler error.
    '''
    tax1 = tax.total_taxes_SS(r, b1_2, w, e, n_guess, BQ, lambdas[j], factor, T_H, j)
    cons = get_cons(r, b1_2, w, e, n_guess, BQ, lambdas[j], b2_2, g_y, tax1)
    income = (r * b1_2 + w * e * n_guess) * factor
    deriv = 1 - tau_payroll - tax.tau_income(r, b1_2, w, e, n_guess, factor) - tax.tau_income_deriv(
        r, b1_2, w, e, n_guess, factor) * income
    euler = MUc(cons) * w * deriv * e - MUl(n_guess, chi_n)
    return euler

def Euler3(w, r, e, n_guess, b_guess, BQ, factor, chi_b, T_H, j):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = distribution of abilities (SxJ array)
        n_guess  = distribution of labor (SxJ array)
        b_guess  = distribution of capital in period t (S-1 x J array)
        B        = distribution of incidental bequests (1 x J array)
        factor   = scaling value to make average income match data
        chi_b    = discount factor of savings
        T_H  = lump sum transfer from the government to the households

    Returns:
        Value of Euler error.
    '''
    tax1 = tax.total_taxes_eul3_SS(r, b_guess[-2], w, e[-1], n_guess[-1], BQ, lambdas[j], factor, T_H, j)
    cons = get_cons(r, b_guess[-2], w, e[-1], n_guess[-1], BQ, lambdas[j], b_guess[-1], g_y, tax1)
    euler = MUc(cons) - np.exp(-sigma * g_y) * MUb(chi_b, b_guess[-1])
    return euler


# Solve for the Steady State
chi_b = np.ones(J)
w = 1.2
r = .06
T_H = 0
BQ = np.ones(J) * .01
factor = 100000
temp_xi = .5
max_it = 100
dist = 10
dist_tol = .0001
it = 0
bssmat = np.ones((S,J)) * .01
nssmat = np.ones((S,J)) * .5

while (dist > dist_tol) and (it < max_it):
    

    for j in xrange(J):
        # Solve the Euler equations
        guesses = np.append(bssmat[:,j], nssmat[:,j])
        solutions = opt.fsolve(Solver, guesses*.5, args=(r, w, BQ[j], T_H, e[:,j], chi_b[j], j))
        bssmat[:,j] = solutions[:S]
        nssmat[:,j] = solutions[S:]
    
    K = (omega_SS * bssmat).sum()
    L = get_L(e, nssmat)
    Y = get_Y(K, L)
    new_r = get_r(Y, K)
    new_w = get_w(Y, L)
    b1_2 = np.array(list(np.zeros(J).reshape(1, J)) + list(bssmat[:-1, :]))
    average_income_model = ((new_r * b1_2 + new_w * e * nssmat) * omega_SS).sum()
    new_factor = mean_income_data / average_income_model 
    new_BQ = (1+new_r)*(bssmat * omega_SS * rho.reshape(S, 1)).sum(0)
    new_T_H = tax.tax_lump(new_r, b1_2, new_w, e, nssmat, new_BQ, lambdas, factor, omega_SS)

    r = temp_xi*new_r + (1-temp_xi)*r
    w = temp_xi*new_w + (1-temp_xi)*w
    factor = temp_xi*new_factor + (1-temp_xi)*factor 
    BQ = temp_xi*new_BQ + (1-temp_xi)*BQ 
    T_H = temp_xi*new_T_H + (1-temp_xi)*T_H 
    
    dist = (abs(r-new_r) + abs(w-new_w))
    it += 1
    print "Iteration: ", it
    print "Distance: ", dist


# Prepare results for saving and graphing 
rss = new_r 
wss = new_w
Lss = get_L(e,nssmat)
Kss = (omega_SS*bssmat).sum()
Yss = get_Y(Kss, Lss)
factor_ss = factor 

domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += lambdas[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)
X2, Y2 = np.meshgrid(domain[1:], Jgrid)

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'age-$s$')
ax5.set_ylabel(r'ability type-$j$')
ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax5.plot_surface(X, Y, bssmat.T, rstride=1, cstride=1, cmap=cmap2)
plt.show()

# Save the results
bssmat_init = np.array(list(bssmat) + list(BQ.reshape(1, J)))
nssmat_init = nssmat

var_names = ['bssmat_init', 'nssmat_init']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/SSinit/ss_init_tpi.pkl", "w"))

var_names = ['S', 'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'e', 'E',
             'J', 'Kss', 'bssmat', 'Lss', 'nssmat',
             'Yss', 'wss', 'rss', 'omega', 'chi_n', 'chi_b', 'ltilde', 'T',
             'g_n', 'g_y', 'omega_SS', 'TPImaxiter', 'TPImindist', 'BQ',
             'rho', 'lambdas',
             'b_ellipse', 'k_ellipse', 'upsilon',
             'factor_ss',  'a_tax_income', 'b_tax_income',
             'c_tax_income', 'd_tax_income', 'tau_payroll',
             'tau_bq', 'theta', 'retire',
             'mean_income_data', 'bssmat2', 'cssmat',
             'starting_age', 'bssmat3',
             'ending_age', 'T_Hss', 'euler1', 'euler2', 'euler3',
             'h_wealth', 'p_wealth', 'm_wealth']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/SSinit/ss_init.pkl", "w"))



