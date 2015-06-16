'''
------------------------------------------------------------------------
Last updated: 6/5/2015

Calculates steady state of OLG model with S age cohorts.

This py-file calls the following other file(s):
            tax_funcs_jmd.py
            hh_foc_jmd.py

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            
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


def HH_solve(b_last, r, w, bq, T_H, final_chi_b_params,params, factor, lambdas, g_y, rho, e, j, S, J):
    
    '''
    Parameters: Steady state interest rate, wage rate, distribution of bequests
                government transfers, parameters and exogenous variables, 
                ability type (j - will loop over this).

    Returns:    1) Error between backwards inducted initial capital stock and 
                   its exogenous value (0)
                2) Values for asset holdings, labor supply, and consumption from
                   FOCS
    '''
    # initialize arrays
    c = np.zeros((S+1, 1))
    b = np.zeros((S+1, 1)) 
    n = np.zeros((S+1, 1))
    
    ## But should actually use an 'if' and exception so if guess b_last <0 then don't continue =- st eitwth larg eor   
    b[S,0] = b_last # note the timing here = b(s) is assets chosen to exit age s with 
    
    
    chi_b = np.tile(np.array(params[:J]).reshape(1, J), (S, 1))
    chi_n = np.array(params[J:])

    # use FOC from last period to get capital consumption in last period
    c[S-1,0] = hh_focs_jmd.foc_last(b_last,chi_b[S-1,j])
    
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
             print(foc_l_err)
             print(bc_err)
             return (foc_l_err,bc_err)
     
        n[S-i,0], b[S-i,0] = opt.fsolve(hh_equations,(n_guess,b_guess), xtol=1e-13)
        
        print(n[S-i,0])
        print(b[S-i,0])
        
        # use FOC for savings/bequests to determine consumption one period prior
        c[S-i-1,0] = hh_focs_jmd.foc_b(w, r, e[i,j], c[S-i,0], n[S-i,0], b[S-i,0], bq[j], factor, T_H, chi_b[S-i,j], rho[S-i-1], j)
        
        print('b as we go')
        print(S-i)
        print(b[S-i,0])
       
    
    error_b = b[0,0] - 0 
    
    print('initial b')
    print(b[0,0])
    #print('hh_solve error')
    #print(error_b)
    
    return error_b      

