'''
------------------------------------------------------------------------
Last updated 5/21/2015

Gives the reimbursement rates for the payroll tax.

This py-file calls the following other file(s):
            OUTPUT/Saved_moments/payroll_inputs.pkl
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import pickle
import numpy as np

'''
------------------------------------------------------------------------
    Import data need to compute replacement rates, outputed from SS.py
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/Saved_moments/payroll_inputs.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

'''
------------------------------------------------------------------------
    Compute replacement rates
------------------------------------------------------------------------
'''


def vals():
    AIME = ((wss * factor_ss * e * nssmat_init)*omega_SS).sum(0) / 12.0
    PIA = np.zeros(J)
    # Bins from data for each level of replacement
    for j in xrange(J):
        if AIME[j] < 749.0:
            PIA[j] = .9 * AIME[j]
        elif AIME[j] < 4517.0:
            PIA[j] = 674.1+.32*(AIME[j] - 749.0)
        else:
            PIA[j] = 1879.86 + .15*(AIME[j] - 4517.0)
    theta = PIA * (e * nssmat_init).mean(0) / AIME
    # Set the maximum replacment rate to be $30,000
    maxpayment = 30000.0/(factor_ss * wss)
    theta[theta > maxpayment] = maxpayment
    print theta
    return theta
