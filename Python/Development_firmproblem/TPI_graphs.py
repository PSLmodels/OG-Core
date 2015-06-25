'''
------------------------------------------------------------------------
Last updated 6/19/2015

Creates graphs for TPI values.

This py-file calls the following other file(s):
            SSinit/ss_init_vars.pkl
            TPIinit/TPIinit_vars.pkl
            SS/ss_vars.pkl
            TPI/TPI_vars.pkl
            OUTPUT/Saved_moments/params_given.pkl
            OUTPUT/Saved_moments/params_changed.pkl          
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

'''
------------------------------------------------------------------------
    Create variables for baseline TPI graphs
------------------------------------------------------------------------
'''

variables = pickle.load(open("OUTPUT/SSinit/ss_init_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("OUTPUT/TPIinit/TPIinit_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("OUTPUT/Saved_moments/params_given.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

N_tilde = omega.sum(1).sum(1)
omega_stationary_init = omega / N_tilde.reshape(T+S, 1, 1)
omega_stationary_init = omega_stationary_init[:T]

Kpath_TPIbase = Kpath_TPI
Lpath_TPIbase = Lpath_TPI
w_base = winit
r_base = rinit
Y_base = Yinit
BQpath_TPIbase = BQpath_TPI
eul_savings_init = eul_savings
eul_laborleisure_init = eul_laborleisure
b_mat_init = b_mat
n_mat_init = n_mat
T_H_initbase = T_H_init


b1 = np.zeros((T, S, J))
b1[:, 1:, :] = b_mat_init[:T, :-1, :]
b2 = np.zeros((T, S, J))
b2[:, :, :] = b_mat_init[:T, :, :]
cinitbase = cinit

y_mat_init = cinitbase + b_mat_init[1:T+1] - (1-delta)*b_mat_init[:T]

# Lifetime Utility Graphs:
c_ut_init = np.zeros((S, S, J))
for s in xrange(S-1):
    c_ut_init[:, s+1, :] = cinitbase[s+1:s+1+S, s+1, :]
c_ut_init[:, 0, :] = cinitbase[:S, 0, :]
L_ut_init = np.zeros((S, S, J))
for s in xrange(S-1):
    L_ut_init[:, s+1, :] = n_mat_init[s+1:s+1+S, s+1, :]
L_ut_init[:, 0, :] = n_mat_init[:S, 0, :]
B_ut_init = BQpath_TPIbase[S:T]
b_ut_init = np.zeros((S, S, J))
for s in xrange(S):
    b_ut_init[:, s, :] = b_mat_init[s:s+S, s, :]

beq_ut = chi_b.reshape(1, S, J) * (rho.reshape(1, S, 1)) * (b_ut_init[:S]**(1-sigma)-1)/(1-sigma)
utility = ((c_ut_init ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_ut_init/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
cum_morts = np.cumprod(1-rho)
utility *= cum_morts.reshape(1, S, 1)
utility_lifetime_init = utility.sum(1)

# Period Utility Graphs
beq_ut_period = chi_b.reshape(1, S, J) * (rho.reshape(1, S, 1)) * (b_mat_init[:S]**(1-sigma)-1)/(1-sigma)
utility_period = ((cinitbase[:S] ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(n_mat_init[:S]/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility_period += beq_ut_period
utility_period *= beta_string.reshape(1, S, 1)
utility_period *= cum_morts.reshape(1, S, 1)
utility_period_init = utility_period.sum(1)

'''
------------------------------------------------------------------------
    Create variables for tax experiment TPI graphs 
------------------------------------------------------------------------
'''


variables = pickle.load(open("OUTPUT/SS/ss_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("OUTPUT/TPI/TPI_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("OUTPUT/Saved_moments/params_changed.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

N_tilde = omega.sum(1).sum(1)
omega_stationary = omega / N_tilde.reshape(T+S, 1, 1)
omega_stationary = omega_stationary[:T]

b1 = np.zeros((T, S, J))
b1[:, 1:, :] = b_mat[:T, :-1, :]
b2 = np.zeros((T, S, J))
b2[:, :, :] = b_mat[:T, :, :]


y_mat = cinit + b_mat[1:T+1] - (1-delta)*b_mat[:T]

# Lifetime Utility
c_ut = np.zeros((S, S, J))
for s in xrange(S-1):
    c_ut[:, s+1, :] = cinit[s+1:s+1+S, s+1, :]
c_ut[:, 0, :] = cinit[:S, 0, :]
L_ut = np.zeros((S, S, J))
for s in xrange(S-1):
    L_ut[:, s+1, :] = n_mat[s+1:s+1+S, s+1, :]
L_ut[:, 0, :] = n_mat[:S, 0, :]
B_ut = BQpath_TPI[S:T]
b_ut = np.zeros((S, S, J))
for s in xrange(S):
    b_ut[:, s, :] = b_mat[s:s+S, s, :]

beq_ut = chi_b.reshape(1, S, J) * (rho.reshape(1, S, 1)) * (b_ut[:S]**(1-sigma)-1)/(1-sigma)
utility = ((c_ut ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_ut/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
utility *= cum_morts.reshape(1, S, 1)
utility_lifetime = utility.sum(1)

# Period Utility
beq_ut_period = chi_b.reshape(1, S, J) * (rho.reshape(1, S, 1)) * (b_mat[:S]**(1-sigma)-1)/(1-sigma)
utility_period = ((cinit[:S] ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(n_mat[:S]/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility_period += beq_ut_period
utility_period *= beta_string.reshape(1, S, 1)
utility_period *= cum_morts.reshape(1, S, 1)
utility_period = utility_period.sum(1)


'''
------------------------------------------------------------------------
Plot Timepath for K, N, w, r, Y, U
------------------------------------------------------------------------
'''

plt.figure()
plt.plot(np.arange(T), Kpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), Kpath_TPI[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Capital Stock $\hat{K}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/TPI_K")

plt.figure()
plt.plot(np.arange(T), Kpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Capital Stock $\hat{K}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/TPI_K")

plt.figure()
plt.plot(np.arange(T), Lpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), Lpath_TPI[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Labor Supply $\hat{L}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/TPI_L")

plt.figure()
plt.plot(np.arange(T), Lpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Labor Supply $\hat{L}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/TPI_L")

plt.figure()
plt.plot(np.arange(T), (y_mat_init*omega_stationary).sum(1).sum(1)[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), (y_mat*omega_stationary).sum(1).sum(1)[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Output $\hat{Y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/TPI_Y")

plt.figure()
plt.plot(np.arange(T), (y_mat_init*omega_stationary).sum(1).sum(1)[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Output $\hat{Y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/TPI_Y")

plt.figure()
plt.plot(np.arange(T), w_base[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), winit[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Wage $\hat{w}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/TPI_w")

plt.figure()
plt.plot(np.arange(T), w_base[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Wage $\hat{w}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/TPI_w")

plt.figure()
plt.plot(np.arange(T), r_base[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), rinit[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Rental Rate $\hat{r}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/TPI_r")

plt.figure()
plt.plot(np.arange(T), r_base[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Rental Rate $\hat{r}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/TPI_r")

X3, Y3 = np.meshgrid(np.arange(S), np.arange(J)+1)
cmap2 = matplotlib.cm.get_cmap('winter')
fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, ((utility_lifetime - utility_lifetime_init)/np.abs(utility_lifetime_init)).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('OUTPUT/TPI/utility_lifetime_percdif')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, (utility_lifetime_init).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('OUTPUT/TPIinit/utility_lifetime')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, utility_lifetime.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('OUTPUT/TPI/utility_lifetime')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, ((utility_period - utility_period_init)/np.abs(utility_period_init)).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('OUTPUT/TPI/utility_period_percdif')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, (utility_period_init).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('OUTPUT/TPIinit/utility_period')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, utility_period.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('OUTPUT/TPI/utility_period')


'''
------------------------------------------------------------------------
Plot Timepath for B
------------------------------------------------------------------------
'''

for i in xrange(J):
    plt.figure()
    plt.plot(np.arange(
        T), BQpath_TPIbase[:T, i], linewidth=2, color='b', label="Base TPI time path for group j={}".format(i+1) )
    plt.plot(np.arange(
        T), BQpath_TPI[:T, i], linewidth=2, linestyle='--', color='g', label="TPI time path for group j={}".format(i+1) )
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Aggregate $\hat{BQ_{j,t}}$")
    plt.legend(loc=0)
    plt.savefig("OUTPUT/TPI/TPI_B_j{}".format(i+1))

'''
------------------------------------------------------------------------
Compute Plot Euler Errors
------------------------------------------------------------------------
domain     = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''

domain = np.linspace(1, T, T)
plt.figure()
plt.plot(domain, eul_savings_init, label='Euler1')
plt.plot(domain, eul_laborleisure_init, label='Euler2')
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.legend(loc=0)
plt.title('Maximum Euler Error for each period across S and J')
plt.savefig('OUTPUT/TPIinit/euler_errors_TPI')

domain = np.linspace(1, T, T)
plt.figure()
plt.plot(domain, eul_savings, label='Euler1')
plt.plot(domain, eul_laborleisure, label='Euler2')
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.legend(loc=0)
plt.title('Maximum Euler Error for each period across S and J')
plt.savefig('OUTPUT/TPI/euler_errors_TPI')

'''
------------------------------------------------------------------------
    Gini functions
------------------------------------------------------------------------
'''


def gini_cols(path, omega):
    # Inequality across ability type
    path2 = np.copy(path)
    mask = path2 < 0
    path2[mask] = 0
    collapseS = path2.sum(1)
    total = collapseS.sum(1)
    collapseS /= total.reshape(T, 1)
    omega1 = omega.sum(1).reshape(T, J)
    idx = np.argsort(collapseS-omega1, axis=1)
    collapseS2 = collapseS[:, idx][np.eye(T,T, dtype=bool)]
    omega_sorted = omega1[:, idx][np.eye(T, T, dtype=bool)]
    cum_levels = np.cumsum(collapseS2, axis=1)
    cum_omega = np.cumsum(omega_sorted, axis=1)
    # plt.figure()
    # plt.plot(cum_omega[10], cum_levels[10])
    # plt.plot(cum_omega[10], cum_omega[10])
    # plt.savefig('gini_test.png')
    cum_levels[:, 1:] = cum_levels[:, :-1]
    cum_levels[:, 0] = 0
    G = 2 * (.5-(cum_levels * omega_sorted).sum(1))
    print G[-1]
    return G


def gini_colj(path, omega):
    # Inequality across age
    path2 = np.copy(path)
    mask = path2 < 0
    path2[mask] = 0
    collapseJ = path2.sum(2)
    total = collapseJ.sum(1)
    collapseJ /= total.reshape(T, 1)
    omega1 = omega.sum(2).reshape(T, S)
    idx = np.argsort(collapseJ-omega1, axis=1)
    collapseJ2 = collapseJ[:, idx][np.eye(T,T, dtype=bool)]
    omega_sorted = omega1[:, idx][np.eye(T, T, dtype=bool)]
    cum_levels = np.cumsum(collapseJ2, axis=1)
    cum_omega = np.cumsum(omega_sorted, axis=1)
    cum_levels[:, 1:] = cum_levels[:, :-1]
    cum_levels[:, 0] = 0
    G = 2 * (.5-(cum_levels * omega_sorted).sum(1))
    print G[-1]
    return G


def gini_nocol(path, omega):
    # Inequality across age and ability
    pathx = np.copy(path)
    mask = pathx < 0
    pathx[mask] = 0
    pathx = pathx.reshape(T, S*J)
    total = pathx.sum(1)
    pathx /= total.reshape(T, 1)
    omega = omega.reshape(T, S*J)
    idx = np.argsort(pathx-omega, axis=1)
    path2 = pathx[:, idx][np.eye(T,T, dtype=bool)]
    omega_sorted = omega[:, idx][np.eye(T,T, dtype=bool)]
    cum_levels = np.cumsum(path2, axis=1)
    cum_omega = np.cumsum(omega_sorted, axis=1)
    cum_levels[:, 1:] = cum_levels[:, :-1]
    cum_levels[:, 0] = 0
    G = 2 * (.5-(cum_levels * omega_sorted).sum(1))
    print G[-1]
    return G

'''
------------------------------------------------------------------------
    GINI plots comparing the tax experiment to the baseline
------------------------------------------------------------------------
'''


plt.figure()
plt.plot(np.arange(T), gini_cols(b_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(b_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_b_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(n_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(n_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_l_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(y_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_y_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(cinit[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_c_cols")


plt.figure()
plt.plot(np.arange(T), gini_colj(b_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(b_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_b_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(n_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(n_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_l_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(y_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_y_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(cinit[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_c_colj")

plt.figure()
plt.plot(np.arange(T), gini_nocol(b_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(b_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_b_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(n_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(n_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_l_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(y_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_y_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(cinit[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPI/gini_c_nocol")

'''
------------------------------------------------------------------------
    Baseline TPI graphs
------------------------------------------------------------------------
'''

plt.figure()
plt.plot(np.arange(T), gini_cols(b_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_b_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(n_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_l_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_y_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_c_cols")

plt.figure()
plt.plot(np.arange(T), gini_colj(b_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_b_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(n_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_l_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_y_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_c_colj")

plt.figure()
plt.plot(np.arange(T), gini_nocol(b_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_b_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(n_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_l_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_y_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("OUTPUT/TPIinit/gini_c_nocol")


'''
------------------------------------------------------------------------
    GIF graphs
------------------------------------------------------------------------
'''

domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += lambdas[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)


print 'Starting movies'
# top zlim is for the income tax, bottom zlim is for the wealth tax


# for t in xrange(60):

#     fig5 = plt.figure()
#     ax5 = fig5.gca(projection='3d')
#     ax5.set_xlabel(r'age-$s$')
#     ax5.set_ylabel(r'ability-$j$')
#     ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
#     # ax5.set_zlim([-.30, .05])
#     ax5.set_zlim([-.30, .20])
#     ax5.set_title('T = {}'.format(t))
#     ax5.plot_surface(X, Y, ((b_mat[t] - b_mat_init[t])/b_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
#     name = "%03d" % t
#     plt.savefig('OUTPUT/TPI/movies/b_dif/b_dif_T{}'.format(name))

#     fig5 = plt.figure()
#     ax5 = fig5.gca(projection='3d')
#     ax5.set_xlabel(r'age-$s$')
#     ax5.set_ylabel(r'ability-$j$')
#     ax5.set_zlabel(r'individual labor supply $l_{j,s}$')
#     # ax5.set_zlim([-.15, .15])
#     ax5.set_zlim([-.5, .2])
#     ax5.set_title('T = {}'.format(t))
#     ax5.plot_surface(X, Y, ((n_mat[t] - n_mat_init[t])/n_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
#     name = "%03d" % t
#     plt.savefig('OUTPUT/TPI/movies/l_dif/l_dif_T{}'.format(name))

#     fig5 = plt.figure()
#     ax5 = fig5.gca(projection='3d')
#     ax5.set_xlabel(r'age-$s$')
#     ax5.set_ylabel(r'ability-$j$')
#     ax5.set_zlabel(r'Consumption $c_{j,s}$')
#     # ax5.set_zlim([-.20, .15])
#     ax5.set_zlim([-.30, .30])
#     ax5.set_title('T = {}'.format(t))
#     ax5.plot_surface(X, Y, ((cinit[t] - cinitbase[t])/cinitbase[t]).T, rstride=1, cstride=1, cmap=cmap2)
#     name = "%03d" % t
#     plt.savefig('OUTPUT/TPI/movies/c_dif/c_dif_T{}'.format(name))

#     fig5 = plt.figure()
#     ax5 = fig5.gca(projection='3d')
#     ax5.set_xlabel(r'age-$s$')
#     ax5.set_ylabel(r'ability-$j$')
#     ax5.set_zlabel(r'Income $y_{j,s}$')
#     # ax5.set_zlim([-.2, .15])
#     ax5.set_zlim([-.3, .3])
#     ax5.set_title('T = {}'.format(t))
#     ax5.plot_surface(X, Y, ((y_mat[t] - y_mat_init[t])/y_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
#     name = "%03d" % t
#     plt.savefig('OUTPUT/TPI/movies/y_dif/y_dif_T{}'.format(name))

#     fig5 = plt.figure()
#     ax5 = fig5.gca(projection='3d')
#     ax5.set_xlabel(r'age-$s$')
#     ax5.set_ylabel(r'ability-$j$')
#     ax5.set_zlabel(r'Consumption $c_{j,s}$')
#     # ax5.set_zlim([-2.5, 1.1])
#     # ax5.set_zlim([0, 2])
#     ax5.set_title('T = {}'.format(t))
#     ax5.plot_surface(X, Y, cinitbase[t].T, rstride=1, cstride=1, cmap=cmap2)
#     name = "%03d" % t
#     plt.savefig('OUTPUT/TPI/movies/cons_base/c_base_T{}'.format(name))

