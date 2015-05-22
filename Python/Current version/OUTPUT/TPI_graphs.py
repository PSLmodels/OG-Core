'''
------------------------------------------------------------------------
Last updated 5/21/2015

Creates graphs for TPI values.

This py-file calls the following other file(s):
            SSinit/ss_init.pkl
            TPIinit/TPIinit_vars.pkl
            SS/ss_vars.pkl
            TPI/TPI_vars.pkl          
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
    Create variables for baseline TPI graphs
------------------------------------------------------------------------
'''

variables = pickle.load(open("SSinit/ss_init.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("TPIinit/TPIinit_vars.pkl", "r"))
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
Bpath_TPIbase = Bpath_TPI
eul1_init = eul1
eul2_init = eul2
eul3_init = eul3
K_mat_init = K_mat
L_mat_init = L_mat
Tinitbase = Tinit


K1 = np.zeros((T, S, J))
K1[:, 1:, :] = K_mat_init[:T, :-1, :]
K2 = np.zeros((T, S, J))
K2[:, :, :] = K_mat_init[:T, :, :]
cinitbase = cinit

Y_mat_init = cinitbase + K_mat_init[1:T+1] - (1-delta)*K_mat_init[:T]

# Lifetime Utility Graphs:
c_ut_init = np.zeros((S, S, J))
for s in xrange(S-1):
    c_ut_init[:, s+1, :] = cinitbase[s+1:s+1+S, s+1, :]
c_ut_init[:, 0, :] = cinitbase[:S, 0, :]
L_ut_init = np.zeros((S, S, J))
for s in xrange(S-1):
    L_ut_init[:, s+1, :] = L_mat_init[s+1:s+1+S, s+1, :]
L_ut_init[:, 0, :] = L_mat_init[:S, 0, :]
B_ut_init = Bpath_TPIbase[S:T]
K_ut_init = np.zeros((S, S, J))
for s in xrange(S):
    K_ut_init[:, s, :] = K_mat_init[s:s+S, s, :]

beq_ut = chi_b.reshape(1, S, J) * (mort_rate.reshape(1, S, 1)) * (K_ut_init[:S]**(1-sigma)-1)/(1-sigma)
utility = ((c_ut_init ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_ut_init/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
cum_morts = np.zeros(S)
for i in xrange(S):
    cum_morts[i] = np.prod(1-mort_rate[:i])
utility *= cum_morts.reshape(1, S, 1)
utility_lifetime_init = utility.sum(1)

# Period Utility Graphs
beq_ut_period = chi_b.reshape(1, S, J) * (mort_rate.reshape(1, S, 1)) * (K_mat_init[:S]**(1-sigma)-1)/(1-sigma)
utility_period = ((cinitbase[:S] ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_mat_init[:S]/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility_period += beq_ut_period
utility_period *= beta_string.reshape(1, S, 1)
utility_period *= cum_morts.reshape(1, S, 1)
utility_period_init = utility_period.sum(1)

'''
------------------------------------------------------------------------
    Create variables for tax experiment TPI graphs 
------------------------------------------------------------------------
'''


variables = pickle.load(open("SS/ss_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]
variables = pickle.load(open("TPI/TPI_vars.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

N_tilde = omega.sum(1).sum(1)
omega_stationary = omega / N_tilde.reshape(T+S, 1, 1)
omega_stationary = omega_stationary[:T]

K1 = np.zeros((T, S, J))
K1[:, 1:, :] = K_mat[:T, :-1, :]
K2 = np.zeros((T, S, J))
K2[:, :, :] = K_mat[:T, :, :]


Y_mat = cinit + K_mat[1:T+1] - (1-delta)*K_mat[:T]

# Lifetime Utility
c_ut = np.zeros((S, S, J))
for s in xrange(S-1):
    c_ut[:, s+1, :] = cinit[s+1:s+1+S, s+1, :]
c_ut[:, 0, :] = cinit[:S, 0, :]
L_ut = np.zeros((S, S, J))
for s in xrange(S-1):
    L_ut[:, s+1, :] = L_mat[s+1:s+1+S, s+1, :]
L_ut[:, 0, :] = L_mat[:S, 0, :]
B_ut = Bpath_TPI[S:T]
K_ut = np.zeros((S, S, J))
for s in xrange(S):
    K_ut[:, s, :] = K_mat[s:s+S, s, :]

beq_ut = chi_b.reshape(1, S, J) * (mort_rate.reshape(1, S, 1)) * (K_ut[:S]**(1-sigma)-1)/(1-sigma)
utility = ((c_ut ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_ut/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility += beq_ut 
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
cum_morts = np.zeros(S)
for i in xrange(S):
    cum_morts[i] = np.prod(1-mort_rate[:i])
utility *= cum_morts.reshape(1, S, 1)
utility_lifetime = utility.sum(1)

# Period Utility
beq_ut_period = chi_b.reshape(1, S, J) * (mort_rate.reshape(1, S, 1)) * (K_mat[:S]**(1-sigma)-1)/(1-sigma)
utility_period = ((cinit[:S] ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_mat[:S]/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
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
plt.savefig("TPI/TPI_K")

plt.figure()
plt.plot(np.arange(T), Kpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Capital Stock $\hat{K}$")
plt.legend(loc=0)
plt.savefig("TPIinit/TPI_K")

plt.figure()
plt.plot(np.arange(T), Lpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), Lpath_TPI[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Labor Supply $\hat{L}$")
plt.legend(loc=0)
plt.savefig("TPI/TPI_L")

plt.figure()
plt.plot(np.arange(T), Lpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Labor Supply $\hat{L}$")
plt.legend(loc=0)
plt.savefig("TPIinit/TPI_L")

plt.figure()
plt.plot(np.arange(T), (Y_mat_init*omega_stationary).sum(1).sum(1)[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), (Y_mat*omega_stationary).sum(1).sum(1)[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Output $\hat{Y}$")
plt.legend(loc=0)
plt.savefig("TPI/TPI_Y")

plt.figure()
plt.plot(np.arange(T), (Y_mat_init*omega_stationary).sum(1).sum(1)[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Output $\hat{Y}$")
plt.legend(loc=0)
plt.savefig("TPIinit/TPI_Y")

plt.figure()
plt.plot(np.arange(T), w_base[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), winit[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Wage $\hat{w}$")
plt.legend(loc=0)
plt.savefig("TPI/TPI_w")

plt.figure()
plt.plot(np.arange(T), w_base[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Wage $\hat{w}$")
plt.legend(loc=0)
plt.savefig("TPIinit/TPI_w")

plt.figure()
plt.plot(np.arange(T), r_base[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), rinit[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Rental Rate $\hat{r}$")
plt.legend(loc=0)
plt.savefig("TPI/TPI_r")

plt.figure()
plt.plot(np.arange(T), r_base[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Rental Rate $\hat{r}$")
plt.legend(loc=0)
plt.savefig("TPIinit/TPI_r")

X3, Y3 = np.meshgrid(np.arange(S), np.arange(J)+1)
cmap2 = matplotlib.cm.get_cmap('winter')
fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, ((utility_lifetime - utility_lifetime_init)/np.abs(utility_lifetime_init)).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPI/utility_lifetime_percdif')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, (utility_lifetime_init).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPIinit/utility_lifetime')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, utility_lifetime.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPI/utility_lifetime')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, ((utility_period - utility_period_init)/np.abs(utility_period_init)).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPI/utility_period_percdif')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, (utility_period_init).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPIinit/utility_period')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, utility_period.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPI/utility_period')


'''
------------------------------------------------------------------------
Plot Timepath for B
------------------------------------------------------------------------
'''

for i in xrange(J):
    plt.figure()
    plt.plot(np.arange(
        T), Bpath_TPIbase[:T, i], linewidth=2, color='b', label="Base TPI time path for group j={}".format(i+1) )
    plt.plot(np.arange(
        T), Bpath_TPI[:T, i], linewidth=2, linestyle='--', color='g', label="TPI time path for group j={}".format(i+1) )
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Aggregate $\hat{BQ_{j,t}}$")
    plt.legend(loc=0)
    plt.savefig("TPI/TPI_B_j{}".format(i+1))

'''
------------------------------------------------------------------------
Compute Plot Euler Errors
------------------------------------------------------------------------
domain     = 1 x S vector of each age cohort
------------------------------------------------------------------------
'''

domain = np.linspace(1, T, T)
plt.figure()
plt.plot(domain, eul1_init, label='Euler1')
plt.plot(domain, eul2_init, label='Euler2')
plt.plot(domain, eul3_init, label='Euler3')
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.legend(loc=0)
plt.title('Maximum Euler Error for each period across S and J')
plt.savefig('TPIinit/euler_errors_TPI')

domain = np.linspace(1, T, T)
plt.figure()
plt.plot(domain, eul1, label='Euler1')
plt.plot(domain, eul2, label='Euler2')
plt.plot(domain, eul3, label='Euler3')
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.legend(loc=0)
plt.title('Maximum Euler Error for each period across S and J')
plt.savefig('TPI/euler_errors_TPI')

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
    cum_omega = np.zeros((T, J))
    cum_levels = np.zeros((T, J))
    cum_levels[:, 0] = collapseS2[:, 0]
    cum_omega[:, 0] = omega_sorted[:, 0]
    for j in xrange(1, J):
        cum_levels[:, j] = collapseS2[:, j] + cum_levels[:, j-1]
        cum_omega[:, j] = omega_sorted[:, j] + cum_omega[:, j-1]
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
    cum_levels = np.zeros((T, S))
    cum_omega = np.zeros((T, S))
    cum_levels[:, 0] = collapseJ2[:, 0]
    cum_omega[:, 0] = omega_sorted[:, 0]
    for s in xrange(1, S):
        cum_levels[:, s] = collapseJ2[:, s] + cum_levels[:, s-1]
        cum_omega[:, s] = omega_sorted[:, s] + cum_omega[:, s-1]
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
    cum_levels = np.zeros((T, S*J))
    cum_omega = np.zeros((T, S*J))
    cum_omega[:, 0] = omega_sorted[:, 0]
    cum_levels[:, 0] = path2[:, 0]
    for i in xrange(1, S*J):
        cum_levels[:, i] = path2[:, i] + cum_levels[:, i-1]
        cum_omega[:, i] = omega_sorted[:, i] + cum_omega[:, i-1]
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
plt.plot(np.arange(T), gini_cols(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(K_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_b_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(L_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(L_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_l_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(Y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(Y_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_y_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(cinit[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_c_cols")


plt.figure()
plt.plot(np.arange(T), gini_colj(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(K_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_b_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(L_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(L_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_l_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(Y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(Y_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_y_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(cinit[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_c_colj")

plt.figure()
plt.plot(np.arange(T), gini_nocol(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(K_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_b_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(L_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(L_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_l_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(Y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(Y_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_y_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(cinit[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_c_nocol")

'''
------------------------------------------------------------------------
    Baseline TPI graphs
------------------------------------------------------------------------
'''

plt.figure()
plt.plot(np.arange(T), gini_cols(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_b_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(L_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_l_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(Y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_y_cols")

plt.figure()
plt.plot(np.arange(T), gini_cols(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_c_cols")

plt.figure()
plt.plot(np.arange(T), gini_colj(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_b_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(L_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_l_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(Y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_y_colj")

plt.figure()
plt.plot(np.arange(T), gini_colj(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_c_colj")

plt.figure()
plt.plot(np.arange(T), gini_nocol(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_b_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(L_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_l_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(Y_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_y_nocol")

plt.figure()
plt.plot(np.arange(T), gini_nocol(cinitbase[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_c_nocol")



'''
------------------------------------------------------------------------
    Pickle gini timepaths, to be used by gini_grapher.py
    (change names to be _income or _wealth, depending on whether
        this is the income and wealth tax experiment)
------------------------------------------------------------------------
'''

wealth_baseline = gini_nocol(K_mat_init[:T], omega_stationary_init)
wealth_wealth = gini_nocol(K_mat[:T], omega_stationary)
income_baseline = gini_nocol(Y_mat_init[:T], omega_stationary_init)
income_wealth = gini_nocol(Y_mat[:T], omega_stationary)
cons_baseline = gini_nocol(cinitbase[:T], omega_stationary_init)
cons_wealth = gini_nocol(cinit[:T], omega_stationary)
lab_baseline = gini_nocol(L_mat_init[:T], omega_stationary_init)
lab_wealth = gini_nocol(L_mat[:T], omega_stationary)

vars_to_pickle = ['wealth_baseline', 'wealth_wealth',
                  'income_baseline', 'income_wealth',
                  'cons_baseline', 'cons_wealth',
                  'lab_baseline', 'lab_wealth',
                  'T', 'S', 'J']
dictionary = {}
for key in vars_to_pickle:
    dictionary[key] = globals()[key]
pickle.dump(dictionary, open("TPI/gini_vectors.pkl", "w"))

'''
------------------------------------------------------------------------
    GIF graphs
------------------------------------------------------------------------
'''

domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += bin_weights[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)


print 'Starting movies'
# top zlim is for the income tax, bottom zlim is for the wealth tax


for t in xrange(60):

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
    # ax5.set_zlim([-.30, .05])
    ax5.set_zlim([-.30, .20])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((K_mat[t] - K_mat_init[t])/K_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/b_dif/b_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'individual labor supply $l_{j,s}$')
    # ax5.set_zlim([-.15, .15])
    ax5.set_zlim([-.5, .2])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((L_mat[t] - L_mat_init[t])/L_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/l_dif/l_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'Consumption $c_{j,s}$')
    # ax5.set_zlim([-.20, .15])
    ax5.set_zlim([-.30, .30])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((cinit[t] - cinitbase[t])/cinitbase[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/c_dif/c_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'Income $y_{j,s}$')
    # ax5.set_zlim([-.2, .15])
    ax5.set_zlim([-.3, .3])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((Y_mat[t] - Y_mat_init[t])/Y_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/y_dif/y_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'Consumption $c_{j,s}$')
    # ax5.set_zlim([-2.5, 1.1])
    # ax5.set_zlim([0, 2])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, cinitbase[t].T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/cons_base/c_base_T{}'.format(name))

