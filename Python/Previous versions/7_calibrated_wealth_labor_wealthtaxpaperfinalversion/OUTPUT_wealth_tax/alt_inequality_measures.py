'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

'''
baseline
'''

variables = pickle.load(open("SSinit/ss_init.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]


def the_inequalizer(dist, weights):
    flattened_dist = dist.flatten()
    flattened_weights = weights.flatten()
    idx = np.argsort(flattened_dist)
    sort_dist = flattened_dist[idx]
    sort_weights = flattened_weights[idx]
    cum_weights = np.zeros(S*J)
    cum_weights[0] = sort_weights[0]
    for i in xrange(1, S*J):
        cum_weights[i] = sort_weights[i] + cum_weights[i-1]
    # variance
    print np.var(np.log(dist * weights))
    # 90/10 ratio
    loc_90th = np.argmin(np.abs(cum_weights-.9))
    loc_10th = np.argmin(np.abs(cum_weights-.1))
    print sort_dist[loc_90th]/sort_dist[loc_10th]
    # 10% ratio
    print (sort_dist[loc_90th:]*sort_weights[loc_90th:]).sum() / (sort_dist * sort_weights).sum()
    # 1% ratio
    loc_99th = np.argmin(np.abs(cum_weights-.99))
    print (sort_dist[loc_99th:]*sort_weights[loc_99th:]).sum() / (sort_dist * sort_weights).sum()


kssmatinit = Kssmat
Kssmat2_init = Kssmat2
BQ_init = BQ
Lssmat_init = Lssmat
cssmat_init = cssmat

savings = np.zeros((S, J))
savings[:-1, :] = Kssmat2_init[1:, :]
savings[-1, :] = BQ_init

beq_ut = chi_b.reshape(S, J) * (mort_rate.reshape(S, 1)) * (savings**(1-sigma) -1)/(1-sigma)
utility = ((cssmat_init ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(S, 1) * (b_ellipse * (1-(Lssmat_init/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
utility_init = utility.sum(0)

income_init = rss * Kssmat2 + wss * e * Lssmat + rss * Bss / bin_weights -Tss

Css = (cssmat * omega_SS).sum()
Kssmat3 = np.array(list(Kssmat) + list(BQ.reshape(1, J)))
yss = cssmat + delta * Kssmat3

print 'baseline:'

the_inequalizer(Kssmat3, omega_SS)


'''
with tax experiments
'''

variables = pickle.load(open("SS/ss_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

savings = np.zeros((S, J))
savings[:-1, :] = Kssmat2[1:, :]
savings[-1, :] = BQ
beq_ut = chi_b.reshape(S, J) * (mort_rate.reshape(S, 1)) * (savings**(1-sigma)-1)/(1-sigma)
utility = ((cssmat ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(S, 1) * (b_ellipse * (1-(Lssmat/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
utility = utility.sum(0)

income = rss * Kssmat2 + wss * e * Lssmat + (rss) * Bss / bin_weights -Tss

Css = (cssmat * omega_SS).sum()


Kssmat3 = np.array(list(Kssmat) + list(BQ.reshape(1, J)))
yss = cssmat + delta * Kssmat3

print 'With tax:'
the_inequalizer(Kssmat3, omega_SS)
