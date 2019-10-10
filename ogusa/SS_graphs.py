'''
------------------------------------------------------------------------
Last updated 4/8/2016

Creates graphs for steady state output.

This py-file calls the following other file(s):
            firm.py
            household.py
            SSinit/ss_init_vars.pkl
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
import cPickle as pickle
import os

import firm
import household

import parameters
parameters.DATASET = 'REAL'


'''
------------------------------------------------------------------------
    Create variables for SS baseline graphs
------------------------------------------------------------------------
'''


SS_FIG_DIR = "OUTPUT"
COMPARISON_DIR = "OUTPUT"

ss_init = os.path.join(SS_FIG_DIR, "SSinit/ss_init_vars.pkl")
with open(ss_init, "rb") as f:
    variables = pickle.load(f)
for key in variables:
    globals()[key] = variables[key]

#globals().update(ogusa.parameters.get_parameters_from_file())
globals().update(parameters.get_parameters())
param_names = ['S', 'J', 'T', 'BW', 'lambdas', 'starting_age', 'ending_age',
             'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
             'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
             'b_ellipse', 'k_ellipse', 'upsilon',
             'chi_b_guess', 'chi_n_guess','etr_params','mtrx_params',
             'mtry_params','tau_payroll', 'tau_bq',
             'retire', 'mean_income_data', 'g_n_vector',
             'h_wealth', 'p_wealth', 'm_wealth',
             'omega', 'g_n_ss', 'omega_SS', 'surv_rate', 'e', 'rho']

variables = {}
for key in param_names:
    variables[key] = globals()[key]
for key in variables:
    globals()[key] = variables[key]



bssmatinit = bssmat
bssmat_s_init = bssmat_s
BQss_init = BQss
nssmat_init = nssmat
cssmat_init = cssmat

factor_ss_init = factor_ss

savings = np.copy(bssmat_splus1)

beq_ut = chi_b.reshape(S, J) * (rho.reshape(S, 1)) * \
    (savings**(1 - sigma) - 1) / (1 - sigma)
utility = ((cssmat_init ** (1 - sigma) - 1) / (1 - sigma)) + chi_n.reshape(S, 1) * \
    (b_ellipse * (1 - (nssmat_init / ltilde)**upsilon) ** (1 / upsilon) + k_ellipse)
utility += beq_ut
utility_init = utility.sum(0)

T_Hss_init = T_Hss
Kss_init = Kss
Lss_init = Lss

Css_init = household.get_C(cssmat, omega_SS.reshape(S, 1), lambdas, 'SS')
iss_init = firm.get_I(bssmat_splus1, bssmat_splus1, delta, g_y, g_n_ss)
income_init = cssmat + iss_init
# print((income_init*omega_SS).sum())
# print(Css + delta * Kss)
# print(Kss)
# print(Lss)
# print(Css_init)
# print()(utility_init * omega_SS).sum())
the_inequalizer(income_init, omega_SS, lambdas, S, J)


'''
------------------------------------------------------------------------
    SS baseline graphs
------------------------------------------------------------------------
'''

domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in range(J):
    Jgrid[j:] += lambdas[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)
X2, Y2 = np.meshgrid(domain[1:], Jgrid)

plt.figure()
plt.plot(np.arange(J) + 1, utility_init)
lt_utility = os.path.join(SS_FIG_DIR, "SSinit/lifetime_utility")
plt.savefig(lt_utility)

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'age-$s$')
ax5.set_ylabel(r'ability type-$j$')
ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax5.plot_surface(X, Y, bssmat_s.T, rstride=1, cstride=1, cmap=cmap2)
capital_dist = os.path.join(SS_FIG_DIR, "SSinit/capital_dist")
plt.savefig(capital_dist)
# plt.show()


fig53 = plt.figure()
ax53 = fig53.gca(projection='3d')
ax53.set_xlabel(r'age-$s$')
ax53.set_ylabel(r'ability type-$j$')
ax53.set_zlabel(r'log individual savings $log(\bar{b}_{j,s})$')
ax53.plot_surface(X2, Y2, np.log(
    bssmat_s[1:]).T, rstride=1, cstride=1, cmap=cmap1)
capital_dist_log = os.path.join(SS_FIG_DIR, "SSinit/capital_dist_log")
plt.savefig(capital_dist_log)

plt.figure()
plt.plot(np.arange(J) + 1, BQss)
plt.xlabel(r'ability-$j$')
plt.ylabel(r'bequests $\overline{bq}_{j,E+S+1}$')
intentional_bequests = os.path.join(SS_FIG_DIR, "SSinit/intentional_bequests")
plt.savefig(intentional_bequests)


fig9 = plt.figure()
ax9 = fig9.gca(projection='3d')
ax9.plot_surface(X, Y, cssmat.T, rstride=1, cstride=1, cmap=cmap2)
ax9.set_xlabel(r'age-$s$')
ax9.set_ylabel(r'ability-$j$')
ax9.set_zlabel('Consumption')
# ax9.set_title('Steady State Distribution of Consumption')
consumption = os.path.join(SS_FIG_DIR, "SSinit/consumption")
plt.savefig(consumption)


fig93 = plt.figure()
ax93 = fig93.gca(projection='3d')
ax93.plot_surface(X, Y, np.log(cssmat).T, rstride=1, cstride=1, cmap=cmap2)
ax93.set_xlabel(r'age-$s$')
ax93.set_ylabel(r'ability type-$j$')
ax93.set_zlabel('log consumption')
# ax93.set_title('Steady State Distribution of Consumption')
consumption_log = os.path.join(SS_FIG_DIR, "SSinit/consumption_log")
plt.savefig(consumption_log)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_xlabel(r'age-$s$')
ax2.set_ylabel(r'ability-$j$')
ax2.set_zlabel(r'individual income $\bar{y}_{j,s}$')
ax2.plot_surface(X, Y, (income_init).T, rstride=1, cstride=1, cmap=cmap1)
income = os.path.join(SS_FIG_DIR, "SSinit/income")
plt.savefig(income)


'''
------------------------------------------------------------------------
    Import wealth moments
------------------------------------------------------------------------
'''
domain = np.linspace(20, 95, 76)

wealth_data_tograph = wealth_data_array[2:] / 1000000
wealth_model_tograph = factor_ss_init * bssmatinit[:76] / 1000000


'''
------------------------------------------------------------------------
    Plot graphs of the wealth fit
------------------------------------------------------------------------
'''

whichpercentile = [25, 50, 70, 80, 90, 99, 100]

for j in range(J):
    plt.figure()
    plt.plot(domain, wealth_data_tograph[:, j], label='Data')
    plt.plot(domain, wealth_model_tograph[:, j], label='Model', linestyle='--')
    plt.xlabel(r'age-$s$')
    plt.ylabel(r'Individual savings, in millions of dollars')
    plt.legend(loc=0)
    fig_j = os.path.join(
        SS_FIG_DIR, "SSinit/wealth_fit_graph_{}".format(whichpercentile[j]))
    plt.savefig(fig_j)

# all 7 together

f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
    4, 2, sharex=True, sharey='row', figsize=(9, 9))

ax1.plot(domain, wealth_data_tograph[:, 6], color='black', label='Data')
ax1.plot(domain, wealth_model_tograph[
         :, 6], color='black', label='Model', linestyle='--')
# ax1.set_xlabel(r'age-$s$')
# ax1.set_ylabel(r'$b_s$, in millions of dollars')
# ax1.set_ylim([0, 6])
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width, box.height])
ax1.legend(loc='center right', bbox_to_anchor=(2.15, .5), ncol=2)
ax1.set_title(r'$100^{th}$ Percentile')

ax2.axis('off')

ax3.plot(domain, wealth_data_tograph[:, 5], color='black', label='Data')
ax3.plot(domain, wealth_model_tograph[
         :, 5], color='black', label='Model', linestyle='--')
# ax3.set_xlabel(r'age-$s$')
# ax3.set_ylabel(r'$b_s$, in millions of dollars')
# ax3.set_ylim([0, 6])
ax3.set_title(r'$90-99^{th}$ Percentile')

ax4.plot(domain, wealth_data_tograph[:, 4], color='black', label='Data')
ax4.plot(domain, wealth_model_tograph[
         :, 4], color='black', label='Model', linestyle='--')
# ax4.set_xlabel(r'age-$s$')
# ax4.set_ylabel(r'$b_s$, in millions of dollars')
ax4.set_ylim([0, 6])
ax4.set_title(r'$80-89^{th}$ Percentile')

ax5.plot(domain, wealth_data_tograph[:, 3], color='black', label='Data')
ax5.plot(domain, wealth_model_tograph[
         :, 3], color='black', label='Model', linestyle='--')
# ax5.set_xlabel(r'age-$s$')
# ax5.set_ylabel(r'$b_s$, in millions of dollars')
# ax5.set_ylim([0, 6])
ax5.set_title(r'$70-79^{th}$ Percentile')

ax6.plot(domain, wealth_data_tograph[:, 2], color='black', label='Data')
ax6.plot(domain, wealth_model_tograph[
         :, 2], color='black', label='Model', linestyle='--')
# ax6.set_xlabel(r'age-$s$')
# ax6.set_ylabel(r'$b_s$, in millions of dollars')
ax6.set_ylim([0, 1])
ax6.set_title(r'$50-69^{th}$ Percentile')

ax7.plot(domain, wealth_data_tograph[:, 1], color='black', label='Data')
ax7.plot(domain, wealth_model_tograph[
         :, 1], color='black', label='Model', linestyle='--')
ax7.set_xlabel(r'age-$s$')
ax7.set_ylabel(r'$b_s$, in millions of dollars')
# ax7.set_ylim([0, 6])
ax7.set_title(r'$25-49^{th}$ Percentile')

ax8.plot(domain, wealth_data_tograph[:, 0], color='black', label='Data')
ax8.plot(domain, wealth_model_tograph[
         :, 0], color='black', label='Model', linestyle='--')
# ax8.set_xlabel(r'age-$s$')
# ax8.set_ylabel(r'$b_s$, in millions of dollars')
ax8.set_ylim([-.05, .25])
ax8.set_title(r'$0-24^{th}$ Percentile')


wealth_fits_all_png = os.path.join(SS_FIG_DIR, "SSinit/wealth_fits_all_png")
plt.savefig(wealth_fits_all_png)

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
css_dollars = os.path.join(SS_FIG_DIR, "SSinit/css_dollars")
plt.savefig(css_dollars)

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
income_dollars = os.path.join(SS_FIG_DIR, "SSinit/income_dollars")
plt.savefig(income_dollars)

'''
------------------------------------------------------------------------
    Print dollar levels of wealth, model vs data, and percent differences
------------------------------------------------------------------------
'''

# change percentile, as needed
# for j in range(J):
#     print('j=', j)
#     # For age 20-44:
#     print(np.mean(wealth_data_tograph[:24, j]))
#     print(np.mean(wealth_model_tograph[2:26, j]))

#     # For age 45-65:
#     print(np.mean(wealth_data_tograph[24:45, j]))
#     print(np.mean(wealth_model_tograph[26:47, j]))

#     # Percent differences
#     print((np.mean(wealth_model_tograph[:24, j]) -
#           np.mean(wealth_data_tograph[2:26, j])) /
#           np.mean(wealth_data_tograph[2:26, j]))
#     print((np.mean(wealth_model_tograph[24:45, j]) -
#           np.mean(wealth_data_tograph[26:47, j])) /
#           np.mean(wealth_data_tograph[26:47, j]))
