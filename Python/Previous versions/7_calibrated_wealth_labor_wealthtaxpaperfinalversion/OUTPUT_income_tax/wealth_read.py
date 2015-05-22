import pickle
import numpy as np

variables = pickle.load(open("SSinit/ss_init.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

variables = pickle.load(open("Nothing/wealth_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]


# print bin_weights
# print (Kssmat2*omega_SS).sum(0)/bin_weights

# print factor_ss

# print chi_b

# for j in xrange(7):
#     print 'j=', j+1
#     print factor_ss*(Kssmat2*omega_SS)[1:21, j].sum()/omega_SS[1:21, j].sum()
#     print factor_ss*(Kssmat2*omega_SS)[21:46, j].sum()/omega_SS[21:46, j].sum()
#     print factor_ss*(Kssmat2*omega_SS)[46:, j].sum()/omega_SS[46:, j].sum()

# variables = pickle.load(open("Nothing/chi_b_fits.pkl", "r"))
# for key in variables:
#     globals()[key] = variables[key]

# print chi_fits_new

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

variables = pickle.load(open("Nothing/wealth_data_moments_fit_25.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
domain = np.linspace(20, 95, 76)
plt.figure()
plt.plot(domain, highest_wealth_data_new[2:]/1000000, label='Data')
plt.plot(domain, factor_ss * Kssmat[:76, 0]/1000000, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_25')


variables = pickle.load(open("Nothing/wealth_data_moments_fit_50.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
plt.figure()
plt.plot(domain, highest_wealth_data_new[2:]/1000000, label='Data')
plt.plot(domain, factor_ss * Kssmat[:76, 1]/1000000, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_50')


variables = pickle.load(open("Nothing/wealth_data_moments_fit_70.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
plt.figure()
plt.plot(domain, highest_wealth_data_new[2:]/1000000, label='Data')
plt.plot(domain, factor_ss * Kssmat[:76, 2]/1000000, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_70')

variables = pickle.load(open("Nothing/wealth_data_moments_fit_80.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
plt.figure()
plt.plot(domain, highest_wealth_data_new[2:]/1000000, label='Data')
plt.plot(domain, factor_ss * Kssmat[:76, 3]/1000000, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_80')

variables = pickle.load(open("Nothing/wealth_data_moments_fit_90.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
plt.figure()
plt.plot(domain, highest_wealth_data_new[2:]/1000000, label='Data')
plt.plot(domain, factor_ss * Kssmat[:76, 4]/1000000, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_90')

variables = pickle.load(open("Nothing/wealth_data_moments_fit_99.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
plt.figure()
plt.plot(domain, highest_wealth_data_new[2:]/1000000, label='Data')
plt.plot(domain, factor_ss * Kssmat[:76, 5]/1000000, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_99')

variables = pickle.load(open("Nothing/wealth_data_moments_fit_100.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
plt.figure()
plt.plot(domain, highest_wealth_data_new[2:]/1000000, label='Data')
plt.plot(domain, factor_ss * Kssmat[:76, 6]/1000000, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_100')

domain = np.linspace(20, 100, 80)
plt.figure()
plt.plot(domain, factor_ss * cssmat[:, 0], label='25%')
plt.plot(domain, factor_ss * cssmat[:, 1], label='50%')
plt.plot(domain, factor_ss * cssmat[:, 2], label='70%')
plt.plot(domain, factor_ss * cssmat[:, 3], label='80%')
plt.plot(domain, factor_ss * cssmat[:, 4], label='90%')
plt.plot(domain, factor_ss * cssmat[:, 5], label='99%')
plt.plot(domain, factor_ss * cssmat[:, 6], label='100%')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual consumption, in dollars')
plt.legend(loc=0)
plt.savefig('Nothing/css_fit')

income_ss = rss * Kssmat2 + wss * e * Lssmat

plt.figure()
plt.plot(domain, factor_ss * income_ss[:, 0], label='25%')
plt.plot(domain, factor_ss * income_ss[:, 1], label='50%')
plt.plot(domain, factor_ss * income_ss[:, 2], label='70%')
plt.plot(domain, factor_ss * income_ss[:, 3], label='80%')
plt.plot(domain, factor_ss * income_ss[:, 4], label='90%')
plt.plot(domain, factor_ss * income_ss[:, 5], label='99%')
plt.plot(domain, factor_ss * income_ss[:, 6], label='100%')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual income, in dollars')
plt.legend(loc=0)
plt.savefig('Nothing/income_fit')
