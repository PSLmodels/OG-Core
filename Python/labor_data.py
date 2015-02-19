'''
------------------------------------------------------------------------
Last updated 2/18/2015

Returns the labor hours for all ages of a certain percentile.

This py-file calls the following other file(s):
            jason_savings_data/scf2007to2013_wealth_age.csv

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Nothing/wealth_data_moments.pkl
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
from scipy import stats
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S = 60
J = 99

data = pd.read_table(
    "data\labor\cps_hours_by_age_hourspct.txt", header=0)

piv = data.pivot(index='age', columns='hours_pct', values='mean_hrs')
lab_mat_basic = np.array(piv)
lab_mat_basic /= np.nanmax(lab_mat_basic)

piv2 = data.pivot(index='age', columns='hours_pct', values='num_obs')
weights = np.array(piv2)
weights /= np.nansum(weights, axis=1).reshape(S, 1)
weighted = np.nansum((lab_mat_basic * weights), axis=1)

slope = (weighted[56] - weighted[49])/(56-49)
intercept = weighted[56] - slope * 56
extension = slope * (np.linspace(56, 80, 23)) + intercept
to_dot = slope * (np.linspace(45, 56, 11)) + intercept

labor_dist_data = np.zeros(80)
labor_dist_data[:57] = weighted[:57]
labor_dist_data[57:] = extension


domain = np.linspace(20, 80, S)
Jgrid = np.linspace(1, 100, J)
X, Y = np.meshgrid(domain, Jgrid)
cmap2 = matplotlib.cm.get_cmap('summer')


plt.plot(domain, weighted)
plt.plot(np.linspace(76, 100, 23), extension, color='red')
plt.plot(np.linspace(65, 76, 11), to_dot, linestyle='--', color='red')
plt.axvline(x=76, color='green', linestyle='--')
plt.savefig('OUTPUT/Demographics/labor_dist_data_withfit.png')

fig10 = plt.figure()
ax10 = fig10.gca(projection='3d')
ax10.plot_surface(X, Y, lab_mat_basic.T, rstride=1, cstride=2, cmap=cmap2)
ax10.set_xlabel(r'age-$s$')
ax10.set_ylabel(r'ability type -$j$')
ax10.set_zlabel(r'labor $e_j(s)$')
plt.savefig('OUTPUT/Demographics/data_labor_dist')

var_names = ['labor_dist_data']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/labor_data_moments.pkl", "w"))