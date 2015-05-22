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

print chi_b[0]

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

variables = pickle.load(open("Nothing/wealth_data_moments_fit_25.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
domain = np.linspace(20, 95, 76)

pct_25_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Nothing/wealth_data_moments_fit_50.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_50_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Nothing/wealth_data_moments_fit_70.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_70_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Nothing/wealth_data_moments_fit_80.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_80_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Nothing/wealth_data_moments_fit_90.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_90_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Nothing/wealth_data_moments_fit_99.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_99_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Nothing/wealth_data_moments_fit_100.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_100_data = highest_wealth_data_new[2:]/1000000

pct_25_model = factor_ss * Kssmat[:76, 0]/1000000
pct_50_model = factor_ss * Kssmat[:76, 1]/1000000
pct_70_model = factor_ss * Kssmat[:76, 2]/1000000
pct_80_model = factor_ss * Kssmat[:76, 3]/1000000
pct_90_model = factor_ss * Kssmat[:76, 4]/1000000
pct_99_model = factor_ss * Kssmat[:76, 5]/1000000
pct_100_model = factor_ss * Kssmat[:76, 6]/1000000

# Wealth fit graphs

plt.figure()
plt.plot(domain, pct_25_data, label='Data')
plt.plot(domain, pct_25_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_25')

plt.figure()
plt.plot(domain, pct_50_data, label='Data')
plt.plot(domain, pct_50_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_50')

plt.figure()
plt.plot(domain, pct_70_data, label='Data')
plt.plot(domain, pct_70_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_70')

plt.figure()
plt.plot(domain, pct_80_data, label='Data')
plt.plot(domain, pct_80_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_80')

plt.figure()
plt.plot(domain, pct_90_data, label='Data')
plt.plot(domain, pct_90_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_90')

plt.figure()
plt.plot(domain, pct_99_data, label='Data')
plt.plot(domain, pct_99_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_99')

plt.figure()
plt.plot(domain, pct_100_data, label='Data')
plt.plot(domain, pct_100_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('Nothing/wealth_fit_graph_100')

# all 7 together

# To Do: 
# vert spacing
# horiz spacing
# get rid of age label and/or x tick marks except bottom row?
# get rid of dollar label and/or y tick marks except left column?


f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(9, 9))

ax1.plot(domain, pct_100_data, color='black', label='Data')
ax1.plot(domain, pct_100_model, color='black', label='Model', linestyle='--')
# ax1.set_xlabel(r'age-$s$')
# ax1.set_ylabel(r'$b_s$, in millions of dollars')
# ax1.set_ylim([0, 6])
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width, box.height])
ax1.legend(loc='center right', bbox_to_anchor=(2.15, .5), ncol=2)
ax1.set_title(r'$100^{th}$ Percentile')

ax2.axis('off')

ax3.plot(domain, pct_99_data, color='black', label='Data')
ax3.plot(domain, pct_99_model, color='black', label='Model', linestyle='--')
# ax3.set_xlabel(r'age-$s$')
# ax3.set_ylabel(r'$b_s$, in millions of dollars')
# ax3.set_ylim([0, 6])
ax3.set_title(r'$90-99^{th}$ Percentile')

ax4.plot(domain, pct_90_data, color='black', label='Data')
ax4.plot(domain, pct_90_model, color='black', label='Model', linestyle='--')
# ax4.set_xlabel(r'age-$s$')
# ax4.set_ylabel(r'$b_s$, in millions of dollars')
ax4.set_ylim([0, 6])
ax4.set_title(r'$80-89^{th}$ Percentile')

ax5.plot(domain, pct_80_data, color='black', label='Data')
ax5.plot(domain, pct_80_model, color='black', label='Model', linestyle='--')
# ax5.set_xlabel(r'age-$s$')
# ax5.set_ylabel(r'$b_s$, in millions of dollars')
# ax5.set_ylim([0, 6])
ax5.set_title(r'$70-79^{th}$ Percentile')

ax6.plot(domain, pct_70_data, color='black', label='Data')
ax6.plot(domain, pct_70_model, color='black', label='Model', linestyle='--')
# ax6.set_xlabel(r'age-$s$')
# ax6.set_ylabel(r'$b_s$, in millions of dollars')
ax6.set_ylim([0, 1])
ax6.set_title(r'$50-69^{th}$ Percentile')

ax7.plot(domain, pct_50_data, color='black', label='Data')
ax7.plot(domain, pct_50_model, color='black', label='Model', linestyle='--')
ax7.set_xlabel(r'age-$s$')
ax7.set_ylabel(r'$b_s$, in millions of dollars')
# ax7.set_ylim([0, 6])
ax7.set_title(r'$25-49^{th}$ Percentile')

ax8.plot(domain, pct_25_data, color='black', label='Data')
ax8.plot(domain, pct_25_model, color='black', label='Model', linestyle='--')
# ax8.set_xlabel(r'age-$s$')
# ax8.set_ylabel(r'$b_s$, in millions of dollars')
ax8.set_ylim([-.05, .25])
ax8.set_title(r'$0-24^{th}$ Percentile')


plt.savefig('Nothing/wealth_fits_all.png')

# Other Fit Graphs

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

# print np.mean(pct_100_model[:24] * 1000000)
# print np.mean(pct_100_data[2:26] * 1000000)

# print np.mean(pct_100_model[24:45] * 1000000)
# print np.mean(pct_100_data[26:47] * 1000000)

print (np.mean(pct_90_model[:24] * 1000000) - np.mean(pct_90_data[2:26] * 1000000)) / np.mean(pct_90_data[2:26] * 1000000)

print (np.mean(pct_90_model[24:45] * 1000000) - np.mean(pct_90_data[26:47] * 1000000)) / np.mean(pct_90_data[26:47] * 1000000)