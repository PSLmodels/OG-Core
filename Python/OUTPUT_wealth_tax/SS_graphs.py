'''
------------------------------------------------------------------------
Last updated 12/1/2014

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

variables = pickle.load(open("SSinit/ss_init.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
kssmatinit = Kssmat
Kssmat2_init = Kssmat2
BQ_init = BQ
Lssmat_init = Lssmat
cssmat_init = cssmat

beq_ut = chi_b.reshape(S, 1) * (Kssmat2_init**(1-sigma) -1)/(1-sigma)
beq_ut[0, :] = 0
utility = ((cssmat_init ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(S, 1) * (b_ellipse * (1-(Lssmat_init/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility[-1] += chi_b[-1] * (BQ_init**(1-sigma)-1) / (1-sigma)
utility *= mort_rate.reshape(S, 1)
utility += beq_ut * (1- mort_rate.reshape(S, 1))
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(S, 1)
cum_morts = np.zeros(S)
for i in xrange(S):
    cum_morts[i] = np.prod(1-mort_rate[:i])
utility *= cum_morts.reshape(S, 1)
utility_init = utility.sum(0)

income_init = rss * Kssmat2 + wss * e * Lssmat + rss * BQ / bin_weights -Tss


'''
SS init graphs
'''

domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += bin_weights[j]
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
ax5.plot_surface(X, Y, Kssmat2.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('SSinit/capital_dist')

fig53 = plt.figure()
ax53 = fig53.gca(projection='3d')
ax53.set_xlabel(r'age-$s$')
ax53.set_ylabel(r'ability type-$j$')
ax53.set_zlabel(r'log individual savings $log(\bar{b}_{j,s})$')
ax53.plot_surface(X, Y, np.log(Kssmat2).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SSinit/capital_dist_log')

plt.figure()
plt.plot(np.arange(J)+1, BQ)
plt.xlabel(r'ability-$j$')
plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
plt.savefig('SSinit/intentional_bequests')

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
ax4.set_xlabel(r'age-$s$')
ax4.set_ylabel(r'ability-$j$')
ax4.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
ax4.plot_surface(X, Y, (Lssmat).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('SSinit/labor_dist')
# plt.show()

fig9 = plt.figure()
ax9 = fig9.gca(projection='3d')
ax9.plot_surface(X, Y, cssmat.T, rstride=1, cstride=1, cmap=cmap2)
ax9.set_xlabel(r'age-$s$')
ax9.set_ylabel(r'ability-$j$')
ax9.set_zlabel('Consumption')
# ax9.set_title('Steady State Distribution of Consumption')
plt.savefig('SSinit/consumption')

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
ax16.plot_surface(X2, Y2, euler1.T, rstride=1, cstride=2, cmap=cmap2)
ax16.set_xlabel(r'Age Cohorts $S$')
ax16.set_ylabel(r'Ability Types $J$')
ax16.set_zlabel('Error Level')
ax16.set_title('Euler Errors')
plt.savefig('SSinit/euler_errors_euler1_SS')
fig17 = plt.figure()
ax17 = fig17.gca(projection='3d')
ax17.plot_surface(X, Y, euler2.T, rstride=1, cstride=2, cmap=cmap2)
ax17.set_xlabel(r'Age Cohorts $S$')
ax17.set_ylabel(r'Ability Types $J$')
ax17.set_zlabel('Error Level')
ax17.set_title('Euler Errors')
plt.savefig('SSinit/euler_errors_euler2_SS')


'''
SS graphs
'''

variables = pickle.load(open("SS/ss_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

beq_ut = chi_b.reshape(S, 1) * (Kssmat2**(1-sigma)-1)/(1-sigma)
beq_ut[0, :] = 0
utility = ((cssmat ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(S, 1) * (b_ellipse * (1-(Lssmat/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility[-1] += chi_b[-1] * (BQ**(1-sigma)-1) / (1-sigma)
utility *= mort_rate.reshape(S, 1)
utility += beq_ut * (1- mort_rate.reshape(S, 1))
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(S, 1)
cum_morts = np.zeros(S)
for i in xrange(S):
    cum_morts[i] = np.prod(1-mort_rate[:i])
utility *= cum_morts.reshape(S, 1)
utility = utility.sum(0)

income = rss * Kssmat2 + wss * e * Lssmat + rss * BQ / bin_weights -Tss


plt.figure()
plt.plot(np.arange(J)+1, utility)
plt.savefig('SS/lifetime_utility')

fig15 = plt.figure()
ax15 = fig15.gca(projection='3d')
ax15.set_xlabel(r'age-$s$')
ax15.set_ylabel(r'ability-$j$')
ax15.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax15.plot_surface(X, Y, Kssmat2.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('SS/capital_dist')

plt.figure()
plt.plot(np.arange(J)+1, BQ)
plt.xlabel(r'ability-$j$')
plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
plt.savefig('SS/intentional_bequests')

fig14 = plt.figure()
ax14 = fig14.gca(projection='3d')
ax14.set_xlabel(r'age-$s$')
ax14.set_ylabel(r'ability-$j$')
ax14.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
ax14.plot_surface(X, Y, (Lssmat).T, rstride=1, cstride=1, cmap=cmap1)
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
Combos of both graphs
'''

Kssmat_percdif = (Kssmat - kssmatinit)/ kssmatinit
BQ_percdif = (BQ - BQ_init)/ BQ_init
Lssmat_percdif = (Lssmat - Lssmat_init)/ Lssmat_init 
cssmat_percdif = (cssmat - cssmat_init)/ cssmat_init
utility_dif = utility - utility_init
income_dif = (income - income_init) / income_init


plt.figure()
plt.plot(np.arange(J)+1, utility_dif)
plt.savefig('SS/lifetime_utility_dif')

fig25 = plt.figure()
ax25 = fig25.gca(projection='3d')
ax25.set_xlabel(r'age-$s$')
ax25.set_ylabel(r'ability-$j$')
ax25.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax25.plot_surface(X2, Y2, Kssmat_percdif.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('SS/capital_dist_dif')

plt.figure()
plt.plot(np.arange(J)+1, BQ_percdif)
plt.xlabel(r'ability-$j$')
plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
plt.savefig('SS/intentional_bequests_percdif')

fig24 = plt.figure()
ax24 = fig24.gca(projection='3d')
ax24.set_xlabel(r'age-$s$')
ax24.set_ylabel(r'ability-$j$')
ax24.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
ax24.plot_surface(X, Y, (Lssmat_percdif).T, rstride=1, cstride=1, cmap=cmap1)
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
