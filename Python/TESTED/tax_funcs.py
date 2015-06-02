'''
------------------------------------------------------------------------
Last updated 5/21/2015

Functions for taxes in SS and TPI.

This py-file calls the following other file(s):
            OUTPUT/given_params.pkl
            OUTPUT/SS/d_inc_guess.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import pickle
import os


'''
------------------------------------------------------------------------
Imported user given values
------------------------------------------------------------------------
S            = number of periods an individual lives
J            = number of different ability groups
T            = number of time periods until steady state is reached
lambdas  = percent of each age cohort in each ability group
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
mean_income  = mean income from IRS data file used to calibrate income tax
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
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

if os.path.isfile("OUTPUT/SS/d_inc_guess.pkl"):
    d_tax_income = pickle.load(open("OUTPUT/SS/d_inc_guess.pkl", "r"))

'''
------------------------------------------------------------------------
Tax functions
------------------------------------------------------------------------
    The first 4 functions are the wealth and income tax functions,
        with their derivative functions.  The remaining functions
        are used to get the total amount of taxes.  There are many
        different versions because different euler equations (either
        in SS.py or TPI.py) have different shapes of input arrays.
        TPI tax functions, for example, process arrays of 3 dimensions,
        SS ones with 2 dimensions, and both SS and TPI have a tax
        function that just uses scalars.  Ideally, these will all be
        consolidated to one function, with different methods.
------------------------------------------------------------------------
'''


def tau_wealth(b):
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w = p * h * b / (h*b + m)
    return tau_w


def tau_w_prime(b):
    h = h_wealth
    m = m_wealth
    p = p_wealth
    tau_w_prime = h * m * p / (b*h + m) ** 2
    return tau_w_prime


def tau_income(r, b, w, e, n, factor):
    '''
    Gives income tax value at a
    certain income level
    '''
    a = a_tax_income
    b = b_tax_income
    c = c_tax_income
    d = d_tax_income
    I = r * b + w * e * n
    I *= factor
    num = a * (I ** 2) + b * I
    denom = a * (I ** 2) + b * I + c
    tau = d * num / denom
    return tau


def tau_income_deriv(r, b, w, e, n, factor):
    '''
    Gives derivative of income tax value at a
    certain income level
    '''
    a = a_tax_income
    b = b_tax_income
    c = c_tax_income
    d = d_tax_income
    I = r * b + w * e * n
    I *= factor
    denom = a * (I ** 2) + b * I + c
    num = (2 * a * I + b)
    tau = d * c * num / (denom ** 2)
    return tau


def total_taxes_SS(r, b, w, e, n, BQ, lambdas, factor, T_H):
    '''
    Gives the total amount of taxes in the steady state
    '''
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor) * I
    T_P = tau_payroll * w * e * n
    T_P[retire:] -= theta * w
    T_BQ = tau_bq * BQ / lambdas
    T_W = tau_wealth(b) * b
    tot = T_I + T_P + T_BQ + T_W - T_H
    return tot


def total_taxes_SS2(r, b, w, e, n, BQ, lambdas, factor, T_H):
    '''
    Gives the total amount of taxes in the steady state
    '''
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor) * I
    T_P = tau_payroll * w * e * n
    T_P[retire-1:] -= theta * w
    T_BQ = tau_bq * BQ / lambdas
    T_W = tau_wealth(b) * b
    tot = T_I + T_P + T_BQ + T_W - T_H
    return tot


def tax_lump(r, b, w, e, n, BQ, lambdas, factor, omega_SS):
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor) * I
    T_P = tau_payroll * w * e * n
    T_P[retire:] -= theta * w
    T_BQ = tau_bq * BQ / lambdas
    T_W = tau_wealth(b) * b
    T_H = (omega_SS * (T_I + T_P + T_BQ + T_W)).sum()
    return T_H


def tax_lumpTPI(r, b, w, e, n, BQ, lambdas, factor, omega_stationary):
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor) * I
    T_P = tau_payroll * w * e * n
    T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w
    T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
    T_W = tau_wealth(b) * b
    T_H = (omega_stationary * (T_I + T_P + T_BQ + T_W)).sum(1).sum(1)
    return T_H


def total_taxes_TPI1(r, b, w, e, n, BQ, lambdas, factor, T_H, j):
    '''
    Gives the total amount of taxes in TPI
    '''
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor) * I
    T_P = tau_payroll * w * e * n
    retireTPI = (retire - S)
    T_P[retireTPI:] -= theta[j] * w[retireTPI:]
    T_BQ = tau_bq[j] * BQ / lambdas
    T_W = tau_wealth(b) * b
    tot = T_I + T_P + T_BQ + T_W - T_H
    return tot


def total_taxes_TPI1_2(r, b, w, e, n, BQ, lambdas, factor, T_H, j):
    '''
    Gives the total amount of taxes in TPI
    '''
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor) * I
    T_P = tau_payroll * w * e * n
    retireTPI = (retire-1 - S)
    T_P[retireTPI:] -= theta[j] * w[retireTPI:]
    T_BQ = tau_bq[j] * BQ / lambdas
    T_W = tau_wealth(b) * b
    tot = T_I + T_P + T_BQ + T_W - T_H
    return tot


def total_taxes_path(r, b, w, e, n, BQ, lambdas, factor, T_H):
    '''
    Gives the total amount of taxes for an entire
    timepath.
    '''
    r = r.reshape(T, 1, 1)
    w = w.reshape(T, 1, 1)
    I = r * b + w * e * n
    T_I = tau_income(r, b, w, e, n, factor) * I
    T_P = tau_payroll * w * e * n
    T_P[:, retire:, :] -= theta.reshape(1, 1, J) * w
    T_BQ = tau_bq.reshape(1, 1, J) * BQ / lambdas
    T_W = tau_wealth(b) * b
    T_H = T_H.reshape(T, 1, 1)
    tot = T_I + T_P + T_BQ + T_W - T_H
    return tot
