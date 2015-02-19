'''
------------------------------------------------------------------------
Last updated 1/29/2015

Returns the wealth for all ages of a certain percentile.

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

data = pd.read_table(
    "data/wealth/scf2007to2013_wealth_age.csv", sep=',', header=0)
del data['num_obs']
# rearrange columns so the median values are after the 10th percentile values
cols = ['age', 'mean_wealth', 'sd_wealth', 'p10_wealth', 'median_wealth', 'p90_wealth', 'p95_wealth', 'p96_wealth', 'p98_wealth', 'p99_wealth']
data = data[cols]

p98 = np.array(data['p98_wealth'])
p99 = np.array(data['p99_wealth'])

var_names = ['p98', 'p99']
dictionary = {}
for key in var_names:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("OUTPUT/Nothing/wealth_data_moments.pkl", "w"))
