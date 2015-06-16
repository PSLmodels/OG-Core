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

import tax_funcs_jmd as tax
reload(tax)

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
    output = chi_n* deriv
    return output


def MUb(chi_b, bequest):
    '''
    Parameters: Intentional bequests

    Returns:    Marginal Utility of Bequest
    '''
    output = chi_b * (bequest ** (-sigma))
    return output


def budget(r, b, w, e, n, c, bq, lambdas, factor, b1, T_H, g_y, s, j):
    '''
    Parameters: rental rate, capital stock (t-1), wage, e, labor stock,
                bequests, lambdas, capital stock (t), growth rate y, taxes

    Returns:    Consumption
    '''
    net_tax = tax.total_taxes_SS(r, b, w, e, n, bq, lambdas, factor, T_H, s, j)
    budget_err = (1+r)*b1 + w*e*n + bq / lambdas - net_tax - c - np.exp(g_y)*b
    return budget_err


def foc_last(b_last, chi_b):
    ''' 
    Parameters:
        xi       = coefficient of relative risk aversion
        chi_b    = discount factor of savings

    Variables input:
        b_last   = b_{j,E+S+1}
        
    Returns:
        Last period consumption = c_{j,E+S}
    '''
    c = ((np.exp(-sigma * g_y)*chi_b)**(-1/sigma)) * b_last
    return c 

def foc_b(w, r, e, c, n, b, bq, factor, T_H, chi_b, rho, j):
    '''
    Parameters:
        w        = wage rate (scalar)
        r        = rental rate (scalar)
        e        = ability/effective labor units
        b        = wealth 
        c        = consumption
        bq       = bequests
        factor   = scaling value to make average income match data
        T_H  = lump sum transfer from the government to the households
        xi       = coefficient of relative risk aversion
        chi_b    = discount factor of savings

    Returns:
        Value of Euler error.
    '''
    lambdas_j = lambdas[j]
    income = (r * b + w * e * n) * factor 
    deriv = (
        1 + r*(1-tax.tau_income(r, b, w, e, n, factor)-tax.tau_income_deriv(
            r, b, w, e, n, factor)*income)-tax.tau_w_prime(b)*b-tax.tau_wealth(b))
    cm1  = ((np.exp(-sigma * g_y)*(rho*chi_b*(b**(-sigma)) + beta*(1-rho)*(c**(-sigma))*((1+r)-deriv))))**(-1/sigma)
    return cm1


def foc_l(n, w, r, e, b, c, bq, factor, T_H, chi_n):
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
    income = (r * b + w * e * n) * factor
    deriv = 1 - tau_payroll - tax.tau_income(r, b, w, e, n, factor) - tax.tau_income_deriv(
        r, b, w, e, n, factor) * income
    foc_l_err = MUc(c) * w * deriv * e - MUl(n, chi_n)
    return foc_l_err



