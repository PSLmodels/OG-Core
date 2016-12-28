'''
------------------------------------------------------------------------
Created 12/28/2016

Fiscal policy functions for unbalanced budgeting. In particular, some
functions require time-path calculation.

------------------------------------------------------------------------
'''

# Packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''



def D_G_path(Y, REVENUE, T_H, D0, G0, params):
    '''
    Calculate the time paths of debt and government spending
    '''
    
    tG1, tG2, T, S, r = params
    D = np.zeros(T+S)
    G = np.zeros(T+S)
    D[0] = D0
    G[0] = G0
    t = 1
    while t < T+S:
        D[t] = (1+r) * D[t-1] + G[t-1] + T_H[t-1]
# hmm, what about debt service
    return 