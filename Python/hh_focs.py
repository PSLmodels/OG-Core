'''
------------------------------------------------------------------------
Last updated: 5/29/2015

Gives functions for the first order necessary conditions of the household's
optimization problem.

This py-file calls the following other file(s):
            tax_funcs.py
            OUTPUT/given_params.pkl

------------------------------------------------------------------------
'''

# Packages
import numpy as np
import pickle
import os

import tax_funcs as tax

'''
------------------------------------------------------------------------
Imported user given values
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
lambdas_init  = percent of each age cohort in each ability group
starting_age = age of first members of cohort
ending age   = age of the last members of cohort
E            = number of cohorts before S=1
beta         = discount factor for each age cohort
sigma        = coefficient of relative risk aversion
alpha        = capital share of income
nu_init      = contraction parameter in steady state iteration process
               representing the weight on the new distribution gamma_new
Z            = total factor productivity parameter in firms' production
               function
delta        = depreciation rate of capital for each cohort
ltilde       = measure of time each individual is endowed with each
               period
eta          = Frisch elasticity of labor supply
g_y          = growth rate of technology for one cohort
TPImaxiter   = Maximum number of iterations that TPI will undergo
TPImindist   = Cut-off distance between iterations for TPI
b_ellipse    = value of b for elliptical fit of utility function
k_ellipse    = value of k for elliptical fit of utility function
slow_work    = time at which chi_n starts increasing from 1
mean_income_data  = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income = used to calibrate income tax (scalar)
b_tax_income = used to calibrate income tax (scalar)
c_tax_income = used to calibrate income tax (scalar)
d_tax_income = used to calibrate income tax (scalar)
tau_bq       = bequest tax (scalar)
tau_payroll  = payroll tax (scalar)
theta    = payback value for payroll tax (scalar)
retire       = age in which individuals retire(scalar)
h_wealth     = wealth tax parameter h
p_wealth     = wealth tax parameter p
m_wealth     = wealth tax parameter m
scal         = value to scale the initial guesses by in order to get the
               fsolve to converge
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

'''
------------------------------------------------------------------------
    Define Functions
------------------------------------------------------------------------
'''

# Functions and Definitions


def get_Y(K_now, L_now):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y_now = Z * (K_now ** alpha) * ((L_now) ** (1 - alpha))
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


def get_L(e, n, omega_SS):
    '''
    Parameters: e, n, omega_SS

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


def MUl(n, chi_n):
    '''
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    '''
    deriv = b_ellipse * (1/ltilde) * ((1 - (n / ltilde) ** upsilon) ** (
        (1/upsilon)-1)) * (n / ltilde) ** (upsilon - 1)
    output = chi_n.reshape(S, 1) * deriv
    return output


def MUb(chi_b, bequest):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b[-1, :].reshape(1, J) * (bequest ** (-sigma))
    return output


def get_cons(r, b1, w, e, n, BQ, lambdas, b2, g_y, net_tax):
    '''
    Parameters: rental rate, capital stock (t-1), wage, e, labor stock,
                bequests, lambdas, capital stock (t), growth rate y, taxes

    Returns:    Consumption
    '''
    cons = (1 + r)*b1 + w*e*n + BQ / lambdas - b2*np.exp(g_y) - net_tax
    return cons


def Euler1(w, r, e, n_guess, b1, b2, b3, BQ, factor, T_H, chi_b, rho):
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
    BQ_euler = BQ.reshape(1, J)
    tax1 = tax.total_taxes_SS(r, b1, w, e[:-1, :], n_guess[:-1, :], BQ_euler, lambdas, factor, T_H)
    tax2 = tax.total_taxes_SS2(r, b2, w, e[1:, :], n_guess[1:, :], BQ_euler, lambdas, factor, T_H)
    cons1 = get_cons(r, b1, w, e[:-1, :], n_guess[:-1, :], BQ_euler, lambdas, b2, g_y, tax1)
    cons2 = get_cons(r, b2, w, e[1:, :], n_guess[1:, :], BQ_euler, lambdas, b3, g_y, tax2)
    income = (r * b2 + w * e[1:, :] * n_guess[1:, :]) * factor
    deriv = (
        1 + r*(1-tax.tau_income(r, b1, w, e[1:, :], n_guess[1:, :], factor)-tax.tau_income_deriv(
            r, b1, w, e[1:, :], n_guess[1:, :], factor)*income)-tax.tau_w_prime(b2)*b2-tax.tau_wealth(b2))
    bequest_ut = rho[:-1].reshape(S-1, 1) * np.exp(-sigma * g_y) * chi_b[:-1].reshape(S-1, J) * b2 ** (-sigma)
    euler = MUc(cons1) - beta * (1-rho[:-1].reshape(S-1, 1)) * deriv * MUc(
        cons2) * np.exp(-sigma * g_y) - bequest_ut
    return euler


def Euler2(w, r, e, n_guess, b1_2, b2_2, BQ, factor, T_H, chi_n):
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
    BQ = BQ.reshape(1, J)
    tax1 = tax.total_taxes_SS(r, b1_2, w, e, n_guess, BQ, lambdas, factor, T_H)
    cons = get_cons(r, b1_2, w, e, n_guess, BQ, lambdas, b2_2, g_y, tax1)
    income = (r * b1_2 + w * e * n_guess) * factor
    deriv = 1 - tau_payroll - tax.tau_income(r, b1_2, w, e, n_guess, factor) - tax.tau_income_deriv(
        r, b1_2, w, e, n_guess, factor) * income
    euler = MUc(cons) * w * deriv * e - MUl(n_guess, chi_n)
    return euler


def Euler3(w, r, e, n_guess, b_guess, BQ, factor, chi_b, T_H):
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
    BQ = BQ.reshape(1, J)
    tax1 = tax.total_taxes_eul3_SS(r, b_guess[-2, :], w, e[-1, :], n_guess[-1, :], BQ, lambdas, factor, T_H)
    cons = get_cons(r, b_guess[-2, :], w, e[-1, :], n_guess[-1, :], BQ, lambdas, b_guess[-1, :], g_y, tax1)
    euler = MUc(cons) - np.exp(-sigma * g_y) * MUb(
        chi_b, b_guess[-1, :])
    return euler


