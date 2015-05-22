'''
------------------------------------------------------------------------
Last updated 5/21/2015

Returns the wealth for all ages of a certain percentile.

This py-file calls the following other file(s):
            data/wealth/scf2007to2013_wealth_age_all_percentiles.csv

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Demographics/distribution_of_wealth_data.png
            OUTPUT/Demographics/distribution_of_wealth_data_log.png
            Temporarily:
            OUTPUT/Nothing/wealth_data_moments_fit_25.pkl
            OUTPUT/Nothing/wealth_data_moments_fit_50.pkl
            OUTPUT/Nothing/wealth_data_moments_fit_70.pkl
            OUTPUT/Nothing/wealth_data_moments_fit_80.pkl
            OUTPUT/Nothing/wealth_data_moments_fit_90.pkl
            OUTPUT/Nothing/wealth_data_moments_fit_99.pkl
            OUTPUT/Nothing/wealth_data_moments_fit_100.pkl
            Eventually:
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

'''
------------------------------------------------------------------------
    Import Data
------------------------------------------------------------------------
'''

data = pd.read_table("data/wealth/scf2007to2013_wealth_age_all_percentiles.csv", sep=',', header=0)

'''
------------------------------------------------------------------------
    Graph Data
------------------------------------------------------------------------
'''

to_graph = np.array(data)[:, 1:-1]

domain = np.linspace(18, 95, 78)
Jgrid = np.linspace(1, 99, 99)
X, Y = np.meshgrid(domain, Jgrid)
cmap2 = matplotlib.cm.get_cmap('summer')
fig10 = plt.figure()
ax10 = fig10.gca(projection='3d')
ax10.plot_surface(X, Y, (to_graph).T, rstride=1, cstride=2, cmap=cmap2)
ax10.set_xlabel(r'age-$s$')
ax10.set_ylabel(r'percentile')
ax10.set_zlabel(r'wealth')
plt.savefig('OUTPUT/Demographics/distribution_of_wealth_data')

fig10 = plt.figure()
ax10 = fig10.gca(projection='3d')
ax10.plot_surface(X, Y, np.log(to_graph).T, rstride=1, cstride=2, cmap=cmap2)
ax10.set_xlabel(r'age-$s$')
ax10.set_ylabel(r'percentile')
ax10.set_zlabel(r'log of wealth')
plt.savefig('OUTPUT/Demographics/distribution_of_wealth_data_log')

'''
------------------------------------------------------------------------
    Get wealth moments of a desired percentile
------------------------------------------------------------------------
'''


def get_highest_wealth_data(bin_weights):
    # To do: make this function fully generalized
    # The purpose of this function is to return an array of the desired
    # wealth moments for each percentile group
    last_ability_size = bin_weights[-1]
    percentile = 100 - int(last_ability_size * 100)
    highest_wealth_data = np.array(data['p{}_wealth'.format(percentile)])
    var_names = ['highest_wealth_data']
    dictionary = {}
    for key in var_names:
        dictionary[key] = locals()[key]
    pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments.pkl", "w"))


'''
------------------------------------------------------------------------
    Pickle the wealth moments for the 25, 50, 70, 80, 90, 99, and 100th
        percentiles by hand, since the previous function is not yet fully
        generalized.  Each pickle is an Sx1 vector.
------------------------------------------------------------------------
'''

perc_array = np.array([25, 50, 70, 80, 90, 99, 100])

ar25 = np.ones((78, 24))
for i in xrange(1, 25):
    ar25[:, i-1] = np.array(data['p{}_wealth'.format(i)])
highest_wealth_data_new = np.mean(ar25, axis=1)
var_names = ['highest_wealth_data_new']
dictionary = {}
for key in var_names:
    dictionary[key] = locals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments_fit_25.pkl", "w"))

ar25 = np.ones((78, 25))
for i in xrange(25, 50):
    ar25[:, i-25] = np.array(data['p{}_wealth'.format(i)])
highest_wealth_data_new = np.mean(ar25, axis=1)
var_names = ['highest_wealth_data_new']
dictionary = {}
for key in var_names:
    dictionary[key] = locals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments_fit_50.pkl", "w"))

ar20 = np.ones((78, 20))
for i in xrange(50, 70):
    ar20[:, i-50] = np.array(data['p{}_wealth'.format(i)])
highest_wealth_data_new = np.mean(ar20, axis=1)
var_names = ['highest_wealth_data_new']
dictionary = {}
for key in var_names:
    dictionary[key] = locals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments_fit_70.pkl", "w"))

ar10 = np.ones((78, 10))
for i in xrange(70, 80):
    ar10[:, i-70] = np.array(data['p{}_wealth'.format(i)])
highest_wealth_data_new = np.mean(ar10, axis=1)
var_names = ['highest_wealth_data_new']
dictionary = {}
for key in var_names:
    dictionary[key] = locals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments_fit_80.pkl", "w"))

ar10 = np.ones((78, 10))
for i in xrange(80, 90):
    ar10[:, i-80] = np.array(data['p{}_wealth'.format(i)])
highest_wealth_data_new = np.mean(ar10, axis=1)
var_names = ['highest_wealth_data_new']
dictionary = {}
for key in var_names:
    dictionary[key] = locals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments_fit_90.pkl", "w"))

ar09 = np.ones((78, 9))
for i in xrange(90, 99):
    ar09[:, i-90] = np.array(data['p{}_wealth'.format(i)])
highest_wealth_data_new = np.mean(ar09, axis=1)
var_names = ['highest_wealth_data_new']
dictionary = {}
for key in var_names:
    dictionary[key] = locals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments_fit_99.pkl", "w"))

highest_wealth_data_new = np.array(data['p99_wealth'.format(i)])
var_names = ['highest_wealth_data_new']
dictionary = {}
for key in var_names:
    dictionary[key] = locals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments_fit_100.pkl", "w"))