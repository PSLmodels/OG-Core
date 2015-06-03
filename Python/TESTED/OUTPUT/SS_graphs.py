'''
------------------------------------------------------------------------
Last updated 5/21/2015

Creates graphs for steady state values.

This py-file calls the following other file(s):
            SSinit/ss_init.pkl
            SS/ss_vars.pkl
------------------------------------------------------------------------
'''

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
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


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

'''
------------------------------------------------------------------------
    Create variables for SS baseline graphs
------------------------------------------------------------------------
'''

variables = pickle.load(open("SSinit/ss_init_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("Saved_moments/params_given.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("Saved_moments/income_demo_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

bssmatinit = bssmat
bssmat_s_init = bssmat_s
BQss_init = BQss
nssmat_init = nssmat
cssmat_init = cssmat

factor_ss_init = factor_ss

savings = np.copy(bssmat_splus1)

beq_ut = chi_b.reshape(S, J) * (rho.reshape(S, 1)) * (savings**(1-sigma) -1)/(1-sigma)
utility = ((cssmat_init ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(S, 1) * (b_ellipse * (1-(nssmat_init/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
utility_init = utility.sum(0)


Css = (cssmat * omega_SS).sum()
income_init = cssmat + delta * bssmat_splus1
print (income_init*omega_SS).sum()
# print Css + delta * Kss
# print Kss
# print Lss
# print Css
# print (utility_init * omega_SS).sum()
# the_inequalizer(income_init, omega_SS)


'''
------------------------------------------------------------------------
    SS baseline graphs
------------------------------------------------------------------------
'''

domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += lambdas[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)
X2, Y2 = np.meshgrid(domain[1:], Jgrid)

plt.figure()
plt.plot(np.arange(J)+1, utility_init)
plt.savefig('SSinit/lifetime_utility')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'age-$s$')
ax5.set_ylabel(r'ability type-$j$')
ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax5.plot_surface(X, Y, bssmat_s.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('SSinit/capital_dist')
# plt.show()

fig112 = plt.figure()
ax = plt.subplot(111)
ax.plot(domain, bssmat_s[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, bssmat_s[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, bssmat_s[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, bssmat_s[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, bssmat_s[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, bssmat_s[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, bssmat_s[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'individual savings $\bar{b}_{j,s}$')
plt.savefig('SSinit/capital_dist_2D')

fig53 = plt.figure()
ax53 = fig53.gca(projection='3d')
ax53.set_xlabel(r'age-$s$')
ax53.set_ylabel(r'ability type-$j$')
ax53.set_zlabel(r'log individual savings $log(\bar{b}_{j,s})$')
ax53.plot_surface(X, Y, np.log(bssmat_s).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SSinit/capital_dist_log')

plt.figure()
plt.plot(np.arange(J)+1, BQss)
plt.xlabel(r'ability-$j$')
plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
plt.savefig('SSinit/intentional_bequests')

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
ax4.set_xlabel(r'age-$s$')
ax4.set_ylabel(r'ability-$j$')
ax4.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
ax4.plot_surface(X, Y, (nssmat).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SSinit/labor_dist')

# Plot 2d comparison of labor distribution to data
# First import the labor data
variables = pickle.load(open("Saved_moments/labor_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

plt.figure()
plt.plot(np.arange(80)+20, (nssmat * lambdas).sum(1), label='Model', color='black', linestyle='--')
plt.plot(np.arange(80)+20, labor_dist_data, label='Data', color='black', linestyle='-')
plt.legend()
plt.ylabel(r'individual labor supply $\bar{l}_{s}$')
plt.xlabel(r'age-$s$')
plt.savefig('SSinit/labor_dist_comparison')

fig113 = plt.figure()
ax = plt.subplot(111)
ax.plot(domain, nssmat[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, nssmat[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, nssmat[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, nssmat[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, nssmat[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, nssmat[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, nssmat[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'individual labor supply $\bar{l}_{j,s}$')
plt.savefig('SSinit/labor_dist_2D')

fig9 = plt.figure()
ax9 = fig9.gca(projection='3d')
ax9.plot_surface(X, Y, cssmat.T, rstride=1, cstride=1, cmap=cmap2)
ax9.set_xlabel(r'age-$s$')
ax9.set_ylabel(r'ability-$j$')
ax9.set_zlabel('Consumption')
# ax9.set_title('Steady State Distribution of Consumption')
plt.savefig('SSinit/consumption')

fig114 = plt.figure()
ax = plt.subplot(111)
ax.plot(domain, cssmat[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, cssmat[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, cssmat[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, cssmat[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, cssmat[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, cssmat[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, cssmat[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'individual consumption $\bar{c}_{j,s}$')
plt.savefig('SSinit/consumption_2D')

fig93 = plt.figure()
ax93 = fig93.gca(projection='3d')
ax93.plot_surface(X, Y, np.log(cssmat).T, rstride=1, cstride=1, cmap=cmap2)
ax93.set_xlabel(r'age-$s$')
ax93.set_ylabel(r'ability type-$j$')
ax93.set_zlabel('log consumption')
# ax93.set_title('Steady State Distribution of Consumption')
plt.savefig('SSinit/consumption_log')

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_xlabel(r'age-$s$')
ax2.set_ylabel(r'ability-$j$')
ax2.set_zlabel(r'individual income $\bar{y}_{j,s}$')
ax2.plot_surface(X, Y, (income_init).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SSinit/income')

plt.figure()
plt.plot(domain, chi_n)
plt.xlabel(r'Age cohort - $s$')
plt.ylabel(r'$\chi _n$')
plt.savefig('SSinit/chi_n')

fig16 = plt.figure()
ax16 = fig16.gca(projection='3d')
ax16.plot_surface(X, Y, euler_savings.T, rstride=1, cstride=2, cmap=cmap2)
ax16.set_xlabel(r'Age Cohorts $S$')
ax16.set_ylabel(r'Ability Types $J$')
ax16.set_zlabel('Error Level')
ax16.set_title('Euler Errors')
plt.savefig('SSinit/euler_errors_savings_SS')
fig17 = plt.figure()
ax17 = fig17.gca(projection='3d')
ax17.plot_surface(X, Y, euler_labor_leisure.T, rstride=1, cstride=2, cmap=cmap2)
ax17.set_xlabel(r'Age Cohorts $S$')
ax17.set_ylabel(r'Ability Types $J$')
ax17.set_zlabel('Error Level')
ax17.set_title('Euler Errors')
plt.savefig('SSinit/euler_errors_laborleisure_SS')


'''
------------------------------------------------------------------------
    Create variables for graphs for SS with tax experiments
------------------------------------------------------------------------
'''

variables = pickle.load(open("SS/ss_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("Saved_moments/params_changed.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

# If you want to see the average capital stock levels to calibrate the
# wealth tax, uncomment the following:
# print (bssmat2*omega_SS).sum(0)/lambdas
# print factor_ss

savings = np.copy(bssmat_splus1)
beq_ut = chi_b.reshape(S, J) * (rho.reshape(S, 1)) * (savings**(1-sigma)-1)/(1-sigma)
utility = ((cssmat ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(S, 1) * (b_ellipse * (1-(nssmat/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
utility = utility.sum(0)


Css = (cssmat * omega_SS).sum()
income = cssmat + delta * bssmat_splus1
print (income*omega_SS).sum()
# print Css + delta * Kss
# print Kss
# print Lss
# print Css
# print (utility * omega_SS).sum()
# the_inequalizer(yss, omega_SS)

'''
------------------------------------------------------------------------
    Graphs for SS with tax experiments
------------------------------------------------------------------------
'''


plt.figure()
plt.plot(np.arange(J)+1, utility)
plt.savefig('SS/lifetime_utility')

fig15 = plt.figure()
ax15 = fig15.gca(projection='3d')
ax15.set_xlabel(r'age-$s$')
ax15.set_ylabel(r'ability-$j$')
ax15.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax15.plot_surface(X, Y, bssmat_s.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('SS/capital_dist')

plt.figure()
plt.plot(np.arange(J)+1, BQss)
plt.xlabel(r'ability-$j$')
plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
plt.savefig('SS/intentional_bequests')

fig14 = plt.figure()
ax14 = fig14.gca(projection='3d')
ax14.set_xlabel(r'age-$s$')
ax14.set_ylabel(r'ability-$j$')
ax14.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
ax14.plot_surface(X, Y, (nssmat).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SS/labor_dist')

fig19 = plt.figure()
ax19 = fig19.gca(projection='3d')
ax19.plot_surface(X, Y, cssmat.T, rstride=1, cstride=1, cmap=cmap2)
ax19.set_xlabel(r'age-$s$')
ax19.set_ylabel(r'ability-$j$')
ax19.set_zlabel('Consumption')
ax19.set_title('Steady State Distribution of Consumption')
plt.savefig('SS/consumption')

fig12 = plt.figure()
ax12 = fig12.gca(projection='3d')
ax12.set_xlabel(r'age-$s$')
ax12.set_ylabel(r'ability-$j$')
ax12.set_zlabel(r'individual income $\bar{y}_{j,s}$')
ax12.plot_surface(X, Y, (income).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SS/income')

plt.figure()
plt.plot(domain, chi_n)
plt.xlabel(r'Age cohort - $s$')
plt.ylabel(r'$\chi _n$')
plt.savefig('SS/chi_n')

fig116 = plt.figure()
ax116 = fig116.gca(projection='3d')
ax116.plot_surface(X2, Y2, euler1.T, rstride=1, cstride=2, cmap=cmap2)
ax116.set_xlabel(r'Age Cohorts $S$')
ax116.set_ylabel(r'Ability Types $J$')
ax116.set_zlabel('Error Level')
ax116.set_title('Euler Errors')
plt.savefig('SS/euler_errors_euler1_SS')
fig117 = plt.figure()
ax117 = fig117.gca(projection='3d')
ax117.plot_surface(X, Y, euler2.T, rstride=1, cstride=2, cmap=cmap2)
ax117.set_xlabel(r'Age Cohorts $S$')
ax117.set_ylabel(r'Ability Types $J$')
ax117.set_zlabel('Error Level')
ax117.set_title('Euler Errors')
plt.savefig('SS/euler_errors_euler2_SS')

'''
------------------------------------------------------------------------
    Graphs comparing tax experments to the basline
------------------------------------------------------------------------
'''

bssmat_percdif = (bssmat - bssmatinit)/ bssmatinit
BQss_percdif = (BQss - BQss_init)/ BQss_init
nssmat_percdif = (nssmat - nssmat_init)/ nssmat_init 
cssmat_percdif = (cssmat - cssmat_init)/ cssmat_init
utility_dif = (utility - utility_init) / np.abs(utility_init)
income_dif = (income - income_init) / income_init


plt.figure()
plt.plot(np.arange(J)+1, utility_dif)
plt.savefig('SS/lifetime_utility_percdif')

fig25 = plt.figure()
ax25 = fig25.gca(projection='3d')
ax25.set_xlabel(r'age-$s$')
ax25.set_ylabel(r'ability-$j$')
ax25.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax25.plot_surface(X2, Y2, bssmat_percdif.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('SS/capital_dist_percdif')

plt.figure()
plt.plot(np.arange(J)+1, BQss_percdif)
plt.xlabel(r'ability-$j$')
plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
plt.savefig('SS/intentional_bequests_percdif')

fig24 = plt.figure()
ax24 = fig24.gca(projection='3d')
ax24.set_xlabel(r'age-$s$')
ax24.set_ylabel(r'ability-$j$')
ax24.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
ax24.plot_surface(X, Y, (nssmat_percdif).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SS/labor_dist_percdif')

fig29 = plt.figure()
ax29 = fig29.gca(projection='3d')
ax29.plot_surface(X, Y, cssmat_percdif.T, rstride=1, cstride=1, cmap=cmap2)
ax29.set_xlabel(r'age-$s$')
ax29.set_ylabel(r'ability-$j$')
ax29.set_zlabel('Consumption')
ax29.set_title('Steady State Distribution of Consumption')
plt.savefig('SS/consumption_percdif')

fig22 = plt.figure()
ax22 = fig22.gca(projection='3d')
ax22.set_xlabel(r'age-$s$')
ax22.set_ylabel(r'ability-$j$')
ax22.set_zlabel(r'individual income $\bar{y}_{j,s}$')
ax22.plot_surface(X, Y, (income_dif).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SS/income_percdif')


domain2 = np.linspace(starting_age, ending_age, S-1)


fig999 = plt.figure()
ax = plt.subplot(311)
ax.plot(domain2, bssmat_percdif[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain2, bssmat_percdif[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain2, bssmat_percdif[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain2, bssmat_percdif[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain2, bssmat_percdif[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain2, bssmat_percdif[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain2, bssmat_percdif[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * .4, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'% change in $\bar{b}_{j,s}$')
ax.set_title('Wealth Tax')

ax = plt.subplot(312)
ax.plot(domain, cssmat_percdif[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, cssmat_percdif[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, cssmat_percdif[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, cssmat_percdif[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, cssmat_percdif[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, cssmat_percdif[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, cssmat_percdif[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * .4, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
# ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'% change in $\bar{c}_{j,s}$')

ax = plt.subplot(313)
ax.plot(domain, nssmat_percdif[:, 0], label='0 - 24%', linestyle='-', color='black')
ax.plot(domain, nssmat_percdif[:, 1], label='25 - 49%', linestyle='--', color='black')
ax.plot(domain, nssmat_percdif[:, 2], label='50 - 69%', linestyle='-.', color='black')
ax.plot(domain, nssmat_percdif[:, 3], label='70 - 79%', linestyle=':', color='black')
ax.plot(domain, nssmat_percdif[:, 4], label='80 - 89%', marker='x', color='black')
ax.plot(domain, nssmat_percdif[:, 5], label='90 - 99%', marker='v', color='black')
ax.plot(domain, nssmat_percdif[:, 6], label='99 - 100%', marker='1', color='black')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * .4, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel(r'age-$s$')
ax.set_ylabel(r'% change in $\bar{l}_{j,s}$')


plt.savefig('SS/combograph')


'''
------------------------------------------------------------------------
    Import wealth moments
------------------------------------------------------------------------
'''

variables = pickle.load(open("Saved_moments/wealth_data_moments_fit_25.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
domain = np.linspace(20, 95, 76)

pct_25_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Saved_moments/wealth_data_moments_fit_50.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_50_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Saved_moments/wealth_data_moments_fit_70.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_70_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Saved_moments/wealth_data_moments_fit_80.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_80_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Saved_moments/wealth_data_moments_fit_90.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_90_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Saved_moments/wealth_data_moments_fit_99.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_99_data = highest_wealth_data_new[2:]/1000000

variables = pickle.load(open("Saved_moments/wealth_data_moments_fit_100.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

pct_100_data = highest_wealth_data_new[2:]/1000000


pct_25_model = factor_ss_init * bssmatinit[:76, 0]/1000000
pct_50_model = factor_ss_init * bssmatinit[:76, 1]/1000000
pct_70_model = factor_ss_init * bssmatinit[:76, 2]/1000000
pct_80_model = factor_ss_init * bssmatinit[:76, 3]/1000000
pct_90_model = factor_ss_init * bssmatinit[:76, 4]/1000000
pct_99_model = factor_ss_init * bssmatinit[:76, 5]/1000000
pct_100_model = factor_ss_init * bssmatinit[:76, 6]/1000000

'''
------------------------------------------------------------------------
    Plot graphs of the wealth fit
------------------------------------------------------------------------
'''

plt.figure()
plt.plot(domain, pct_25_data, label='Data')
plt.plot(domain, pct_25_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('SSinit/wealth_fit_graph_25')

plt.figure()
plt.plot(domain, pct_50_data, label='Data')
plt.plot(domain, pct_50_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('SSinit/wealth_fit_graph_50')

plt.figure()
plt.plot(domain, pct_70_data, label='Data')
plt.plot(domain, pct_70_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('SSinit/wealth_fit_graph_70')

plt.figure()
plt.plot(domain, pct_80_data, label='Data')
plt.plot(domain, pct_80_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('SSinit/wealth_fit_graph_80')

plt.figure()
plt.plot(domain, pct_90_data, label='Data')
plt.plot(domain, pct_90_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('SSinit/wealth_fit_graph_90')

plt.figure()
plt.plot(domain, pct_99_data, label='Data')
plt.plot(domain, pct_99_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('SSinit/wealth_fit_graph_99')

plt.figure()
plt.plot(domain, pct_100_data, label='Data')
plt.plot(domain, pct_100_model, label='Model', linestyle='--')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual savings, in millions of dollars')
plt.legend(loc=0)
plt.savefig('SSinit/wealth_fit_graph_100')

# all 7 together

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


plt.savefig('SSinit/wealth_fits_all.png')

'''
------------------------------------------------------------------------
    Plot graphs of baseline SS consumption and income, in dollars
------------------------------------------------------------------------
'''

domain = np.linspace(20, 100, 80)
plt.figure()
plt.plot(domain, factor_ss_init * cssmat_init[:, 0], label='25%')
plt.plot(domain, factor_ss_init * cssmat_init[:, 1], label='50%')
plt.plot(domain, factor_ss_init * cssmat_init[:, 2], label='70%')
plt.plot(domain, factor_ss_init * cssmat_init[:, 3], label='80%')
plt.plot(domain, factor_ss_init * cssmat_init[:, 4], label='90%')
plt.plot(domain, factor_ss_init * cssmat_init[:, 5], label='99%')
plt.plot(domain, factor_ss_init * cssmat_init[:, 6], label='100%')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual consumption, in dollars')
plt.legend(loc=0)
plt.savefig('SSinit/css_dollars')

plt.figure()
plt.plot(domain, factor_ss_init * income_init[:, 0], label='25%')
plt.plot(domain, factor_ss_init * income_init[:, 1], label='50%')
plt.plot(domain, factor_ss_init * income_init[:, 2], label='70%')
plt.plot(domain, factor_ss_init * income_init[:, 3], label='80%')
plt.plot(domain, factor_ss_init * income_init[:, 4], label='90%')
plt.plot(domain, factor_ss_init * income_init[:, 5], label='99%')
plt.plot(domain, factor_ss_init * income_init[:, 6], label='100%')
plt.xlabel(r'age-$s$')
plt.ylabel(r'Individual income, in dollars')
plt.legend(loc=0)
plt.savefig('SSinit/income_dollars')

'''
------------------------------------------------------------------------
    Print dollar levels of wealth, model vs data, and percent differences
------------------------------------------------------------------------
'''

# change percentile, as needed

# For age 20-44:
# print np.mean(pct_100_model[:24] * 1000000)
# print np.mean(pct_100_data[2:26] * 1000000)

# For age 45-65:
# print np.mean(pct_100_model[24:45] * 1000000)
# print np.mean(pct_100_data[26:47] * 1000000)

# Percent differences
# print (np.mean(pct_100_model[:24] * 1000000) - np.mean(pct_100_data[2:26] * 1000000)) / np.mean(pct_100_data[2:26] * 1000000)
# print (np.mean(pct_100_model[24:45] * 1000000) - np.mean(pct_100_data[26:47] * 1000000)) / np.mean(pct_100_data[26:47] * 1000000)
