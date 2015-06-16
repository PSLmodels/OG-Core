'''
------------------------------------------------------------------------
Last updated: 6/12/2015

This is a file to try each function in the larger code to make
sure it's working as intended.

------------------------------------------------------------------------
'''

# Packages
import numpy as np
import os
import scipy.optimize as opt
import pickle

import hh_focs_jmd
import tax_funcs_jmd as tax
reload(hh_focs_jmd)


sigma = 3.0
g_y = 0.03
chi_b = 2.0
b_last = 0.1
w = 0.4 
r = 0.04
n = .4
b = 0.1
e = 20.0 
bq = 0.005
factor = 200000.0
T_H = 0.001
rho = 0.01 
j = 4.0
beta = 0.95
chi_n = 0.5
tau_payroll = 0.15
b_ellipse = 0.67
ltilde = 1.0
upsilon = 1.3499
lambdas = 0.2
b1 = b 
s = 40.0

c = ((np.exp(-sigma * g_y)*chi_b)**(-1/sigma)) * b_last

print(c)


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
    lambdas_j = 0.2
    income = (r * b + w * e * n) * factor 
    deriv = (
        1 + r*(1-tax.tau_income(r, b, w, e, n, factor)-tax.tau_income_deriv(
            r, b, w, e, n, factor)*income)-tax.tau_w_prime(b)*b-tax.tau_wealth(b))
    cm1  = ((np.exp(-sigma * g_y)*(rho*chi_b*(b**(-sigma)) + beta*(1-rho)*(c**(-sigma))*((1+r)-deriv))))**(-1/sigma)
    return cm1

cm1 = foc_b(w, r, e, c, n, b, bq, factor, T_H, chi_b, rho, j)
print('cm1')
print(cm1)


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
    euler = MUc(c) * w * deriv * e - MUl(n, chi_n)
    return euler

euler = foc_l(n, w, r, e, b, c, bq, factor, T_H, chi_n)
print(euler)

def budget(r, b, w, e, n, c, bq, lambdas, factor, b1, T_H, g_y, s, j):
    '''
    Parameters: rental rate, capital stock (t-1), wage, e, labor stock,
                bequests, lambdas, capital stock (t), growth rate y, taxes

    Returns:    Consumption
    '''
    net_tax = tax.total_taxes_SS(r, b, w, e, n, bq, lambdas, factor, T_H, s, j)
    budget_err = (1+r)*b1 + w*e*n + bq / lambdas - net_tax - c - (np.exp(g_y)*b)
    return budget_err

budget_err = budget(r, b, w, e, n, c, bq, lambdas, factor, b1, T_H, g_y, s, j)
print(budget_err)

def hh_equations(p): 
             n_guess, b_guess = p 
             foc_l_err = foc_l(n_guess, w, r, e, b_guess, c, bq, factor, T_H, chi_n) 
             bc_err = budget(r, b_guess, w, e, n_guess, c, bq, lambdas, factor, b1, T_H, g_y, s, j) 
             print(foc_l_err)
             print(bc_err)
             print('n and b guess')
             print(n_guess)
             print(b_guess)
             return (foc_l_err,bc_err)
n_guess = n
b_guess = b  
print('starting fsolve')   
hh_equations_x2 = lambda x: hh_equations(x)
solutions = opt.fsolve(hh_equations_x2,(n_guess,b_guess), xtol=1e-13)
print(solutions)