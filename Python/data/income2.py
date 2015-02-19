import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


J = 7
bin_weights = np.array([.25, .25, .2, .1, .1, .09, .01])
cum_bins = bin_weights.copy()
for j in xrange(1, J):
    cum_bins[j] += cum_bins[j-1]
cum_bins = np.array([0] + list(cum_bins)) * 100

data = pd.read_csv('cwhs_earn_rate_age_profile.csv')

data = data[data['age']<=71]
data['age'] -= 21
data['q_earn'] -= 1
data = data.set_index('age')
del data['obs_earn']

e = np.zeros((80, J))

for i in xrange(50):
    for j in xrange(J):
        e[i, j] = data.ix[i:i+1].mean_earn_rate.values[cum_bins[j]:cum_bins[j+1]].mean()
for j in xrange(J):
    e[50:, j] = np.linspace(e[49,j], e[49,j]/2.0, 30)

e /= (e * bin_weights.reshape(1, J)).sum()/80.0

X, Y = np.meshgrid(np.arange(80)+21, cum_bins[1:])
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.set_xlabel(r'age-$s$')
ax1.set_ylabel(r'ability-$j$')
ax1.plot_surface(X, Y, e.T, rstride=1, cstride=10)
plt.savefig('stuff')

