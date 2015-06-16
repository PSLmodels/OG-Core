'''
------------------------------------------------------------------------
Last updated: 6/5/2015

Gives functions for the first order necessary conditions of the firms's
optimization problem.

This py-file calls the following other file(s):
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


def get_Y(Z, K, L):
    '''
    Parameters: Aggregate capital, Aggregate labor

    Returns:    Aggregate output
    '''
    Y = Z * (K ** alpha) * ((L) ** (1 - alpha))
    return Y


def foc_l(Y, L):
    '''
    Parameters: Aggregate output, Aggregate labor

    Returns:    Returns to labor
    '''
    w = (1 - alpha) * Y / L
    return w


def foc_k(Y, K):
    '''
    Parameters: Aggregate output, Aggregate capital

    Returns:    Returns to capital
    '''
    r = (alpha * Y / K) - delta
    return r


