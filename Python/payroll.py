'''
------------------------------------------------------------------------
Last updated 1/29/2015

Gives the reimbursement rates for the payroll tax.

This py-file calls the following other file(s):
            OUTPUT/Nothing/payroll_inputs.pkl
            data/e_vec_data/cwhs_earn_rate_age_profile.csv
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import pickle
import numpy as np

variables = pickle.load(open("OUTPUT/Nothing/payroll_inputs.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]


def vals():
    A = ((wss * factor_ss * e * Lssmat_init)*omega_SS).sum(0) / 12.0
    P = np.zeros(J)
    rep_rate = np.zeros(J)
    # Bins from data for each level of replacement
    for j in xrange(J):
        if A[j] < 749.0:
            P[j] = .9 * A[j]
        elif A[j] < 4517.0:
            P[j] = 674.1+.32*(A[j] - 749.0)
        else:
            P[j] = 1879.86 + .15*(A[j] - 4517.0)
    rep_rate = P / A
    theta = rep_rate * (e * Lssmat_init).mean(0)
    # Set the maximum replacment rate to be $30,000
    maxpayment = 30000.0/(factor_ss * wss)
    theta[theta > maxpayment] = maxpayment
    return theta
