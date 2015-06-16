'''
------------------------------------------------------------------------
Last updated: 6/5/2015

Calculates steady state of OLG model with S age cohorts.

This py-file calls the following other file(s):
            income.py
            demographics.py
            tax_funcs.py
            hh_foc_jmd.py
            OUTPUT/given_params.pkl
            OUTPUT/Saved_moments/wealth_data_moments_fit_{}.pkl
                name depends on which percentile
            OUTPUT/Saved_moments/labor_data_moments.pkl
            OUTPUT/income_demo_vars.pkl
            OUTPUT/Saved_moments/{}.pkl
                name depends on what iteration just ran
            OUTPUT/SS/d_inc_guess.pkl
                if calibrating the income tax to match the wealth tax

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/income_demo_vars.pkl
            OUTPUT/Saved_moments/{}.pkl
                name depends on what iteration is being run
            OUTPUT/Saved_moments/payroll_inputs.pkl
            OUTPUT/SSinit/ss_init.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import time
import os
import scipy.optimize as opt
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import income
import demographics
import hh_focs_jmd
import tax_funcs_jmd as tax
import HH_prob_jmd
import firm_focs
reload(hh_focs_jmd)
reload(HH_prob_jmd)
reload(firm_focs)


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
    globals()[key] = variables[key] # want to not make these globals, but just in array of parameters


'''
------------------------------------------------------------------------
Generate income and demographic parameters
------------------------------------------------------------------------
e            = S x J matrix of age dependent possible working abilities
               e_s
omega        = T x S x J array of demographics
g_n          = steady state population growth rate
omega_SS     = steady state population distribution
surv_rate    = S x 1 array of survival rates
rho    = S x 1 array of mortality rates
------------------------------------------------------------------------
'''


# These values never change, so only run it once
omega, g_n, omega_SS, surv_rate = demographics.get_omega(
    S, J, T, lambdas, starting_age, ending_age, E)
e = income.get_e(S, J, starting_age, ending_age, lambdas, omega_SS)
rho = 1-surv_rate
var_names = ['omega', 'g_n', 'omega_SS', 'surv_rate', 'e', 'rho']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key] # we should make these so not global variables

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

#Z = 1 


def Steady_State(guesses, params):
    '''
    Parameters: Steady state interest rate, wage rate, distribution of bequests
                government transfers  

    Returns:    Array of 2*S*J Euler equation errors
    '''
    
    
    bq =guesses[0:J]
    r = guesses[J]
    w = guesses[J+1]
    T_H = guesses[J+1]
    factor= guesses[-1]
    
    b_end_guess = bq/(lambdas*(1/rho.sum(0)))  # initial guess at end of life assets
    
    
    b_end=np.zeros((J,1))
    
    # loop over j - solving HH problem separately for each lifetime income group
    for j in xrange(0, J):
        HH_solve_X2 = lambda x: HH_prob_jmd.HH_solve(x, r, w, bq, T_H, final_chi_b_params, params, factor, lambdas, g_y, rho, e, j, S, J)
        b_end[j] = opt.fsolve(HH_solve_X2, b_end_guess[j], xtol=1e-13)
    
    ## Now doing to find the SS solution given the correct b_end (note, what I'd 
    # like to do, but don't know how, is to pass some additional arguments 
    # back from the fsolve call above
    
    # initialize arrays
    c = np.zeros((S+1, J))
    b = np.zeros((S+1, J)) 
    n = np.zeros((S+1, J))
    
    chi_b = np.tile(np.array(params[:J]).reshape(1, J), (S, 1))
    chi_n = np.array(params[J:])
        
    for j in xrange(0, J):
        b[S,j] = b_end[j] # 
    
        # use FOC from last period to get capital consumption in last period
        c[S-1,j] = hh_focs_jmd.foc_last(b[S,j],chi_b[S-1,j])
    
        for i in xrange(1, S):
            # use FOC for labor and budget constraint to jointly determine current 
            # period labor supply and assets enter period with
            s = S-i
            n_guess = 0.3 # it'd be nice to have this guess vary by age.. 
            b_guess = 0.01 # it'd be nice to use some value to help inform this guess
            def hh_equations(p): 
                n_guess, b_guess = p 
                foc_l_err = hh_focs_jmd.foc_l(n_guess, w, r, e[i,j], b_guess, c[S-i,0], bq[j], factor, T_H, chi_n[S-i]) 
                bc_err = hh_focs_jmd.budget(r, b_guess, w, e[i,j], n_guess, c[S-i,0], bq[j], lambdas[j], factor, b[S-i+1,0], T_H, g_y, s, j) 
                return (foc_l_err,bc_err)
     
            n[S-i,j], b[S-i,j] = opt.fsolve(hh_equations,(n_guess,b_guess), xtol=1e-13)
        
            # use FOC for savings/bequests to determine consumption one period prior
            c[S-i-1,j] = hh_focs_jmd.foc_b(w, r, e[i,j], c[S-i,0], n[S-i,0], b[S-i,0], bq[j], factor, T_H, chi_b[S-i,j], rho[S-i-1], j)
    
    
    ## SOME issues with the size of arrays - I did them with 81 years, since we need
    # savings die with - but omega_SS only 80 years - do we need to do something to reconcile??
    b = b[:-1,:]
    c = c[:-1,:]
    n = n[:-1,:]

             
    # get aggregate capital and labor from the distribution of individual values
    K = (omega_SS * b).sum()
    L = hh_focs_jmd.get_L(e, n, omega_SS)
    
    # find aggregate output given the production tech
    Y = firm_focs.get_Y(Z, K,L)
    
    # find the interest rate and wage rate that is consisent with the firm's
    # problem given K and L
    r_firm = firm_focs.foc_k(K,L)
    w_firm = firm_focs.foc_l(K,L)
    
    # firm the bequests from the model
    B = (b_guess * omega_SS * rho.reshape(S, 1)).sum(0)
    BQ_model = (1 + r) * B
    
    # find lump sum transfers from the model
    b1_2 = np.array(list(np.zeros(J).reshape(1, J)) + list(b[:-1, :]))
    T_H_model = tax.tax_lump(r, b1_2, w, e, n, BQ_model, lambdas, factor, omega_SS)
    
    # find errors
    error_r = r - r_firm
    error_w = w - w_firm
    error_bq = bq  - BQ_model
    error_th = T_H - T_H_model
    b1_2 = np.array(list(np.zeros(J).reshape(1, J)) + list(b[:-1, :]))
    average_income_model = ((r * b1_2 + w * e * n) * omega_SS).sum()
    error_factor = mean_income_data - factor * average_income_model

    # Check and punish constraint violations
    ## But should actually use an 'if' and exception so funcion not valuate if guesses are bad
    if r<0:
        error_r += 1e9
    if w<0:
        error_w += 1e9
    error_bq[np.where(bq<0)]= 1e9
    if T_H < 0:
        error_th += 1e9
    
    # print error
    print('SS loop error')
    print np.abs(np.array(list(error_bq.flatten()) + [error_r] + [error_w] + [error_th] + [error_factor])).max()
        
    # return error      
    return list(error_bq.flatten()) + [error_r] + [error_w] + [error_th] + [error_factor]
    #return [list(error_bq.flatten()), error_r, error_w, error_th]

'''
------------------------------------------------------------------------
    Run SS
------------------------------------------------------------------------
'''

bnds = tuple([(1e-6, None)] * (S + J))

# make initial guesses at w, r, BQ, T_H
r_guess_init = [0.04] # ideally this guess should probably be a function of the discount rate
w_guess_init = [0.1]
bq_guess_init = np.ones((1, J)) * 0.001
T_H_guess_init = [0.05]
factor_guess_init = [mean_income_data]
guesses = list(bq_guess_init.flatten()) + r_guess_init + w_guess_init + T_H_guess_init + factor_guess_init

# leaving in this format for chi_b now, but chi_b should be loaded with given params
chi_b = np.ones(S+J)
chi_b[0:J] = np.array([5, 10, 90, 250, 250, 250, 250]) + chi_b_scal
print 'Chi_b:', chi_b[0:J]
chi_b[J:] = chi_n
chi_b = list(chi_b)
final_chi_b_params = chi_b

Steady_State_X2 = lambda x: Steady_State(x, chi_b)
solutions = opt.fsolve(Steady_State_X2, guesses, xtol=1e-13)
print np.array(Steady_State_X2(solutions)).max()
print(solutions)

print('SS capital stock and interest rate')
#print(K)
#print(r)