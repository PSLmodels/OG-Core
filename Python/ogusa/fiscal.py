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



def D_G_path(dg_fixed_values, fiscal_params, other_dg_params, baseline_spending=False):
    '''
    Calculate the time paths of debt and government spending
    '''
    if baseline_spending==False:
        budget_balance, ALPHA_T, ALPHA_G, tG1, tG2, rho_G, debt_ratio_ss = fiscal_params
    else:
        budget_balance, ALPHA_T, ALPHA_G, tG1, tG2, rho_G, debt_ratio_ss, T_Hbaseline, Gbaseline = fiscal_params
    T, r_gov, g_n_vector, g_y = other_dg_params
    Y, REVENUE, T_H, D0, G0 = dg_fixed_values

    D = np.zeros(T+1)
    D[0] = D0
    if baseline_spending==False:
        G = ALPHA_G * Y[:T]
        G[0] = G0
    else:
        G = Gbaseline
    growth = (1+g_n_vector)*np.exp(g_y)

    t = 1
    while t < T-1:
        D[t] = (1/growth[t]) * ((1+r_gov[t-1])*D[t-1] + G[t-1] + T_H[t-1] - REVENUE[t-1])
        #debt_service = r_gov[t]*D[t]
        if (t >= tG1) and (t < tG2):
            G[t] = growth[t+1] * (rho_G*debt_ratio_ss*Y[t] + (1-rho_G)*D[t]) - (1+r_gov[t])*D[t] + REVENUE[t] - T_H[t]
        elif t >= tG2:
            G[t] = growth[t+1] * (debt_ratio_ss*Y[t]) - (1+r_gov[t])*D[t] + REVENUE[t] - T_H[t]
        t += 1

    # in final period, growth rate has stabilized, so we can replace growth[t+1] with growth[t]
    t = T-1
    D[t] = (1/growth[t]) * ((1+r_gov[t-1])*D[t-1] + G[t-1] + T_H[t-1] - REVENUE[t-1])
    #debt_service = r_gov[t]*D[t]
    G[t] = growth[t] * (debt_ratio_ss*Y[t]) - (1+r_gov[t])*D[t] + REVENUE[t] - T_H[t]
    D[t+1] = (1/growth[t+1]) * ((1+r_gov[t])*D[t] + G[t] + T_H[t] - REVENUE[t])
    D_ratio_max = np.amax(D[:T] / Y[:T])
    print 'Maximum debt ratio: ', D_ratio_max

    return D, G


