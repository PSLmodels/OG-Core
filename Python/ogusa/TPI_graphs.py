'''
------------------------------------------------------------------------
Last updated 6/19/2015

Creates graphs for TPI values.

This py-file calls the following other file(s):
            firm.py
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

import firm

'''
------------------------------------------------------------------------
    Create variables for baseline TPI graphs
------------------------------------------------------------------------
'''

TPI_FIG_DIR = "OUTPUT"
VAR_DIR = "OUTPUT"
ss_init = os.path.join(VAR_DIR, "SSinit/ss_init_vars.pkl")
variables = pickle.load(open(ss_init, "rb"))
for key in variables:
    globals()[key] = variables[key]
tpi_init = os.path.join(VAR_DIR, "TPIinit/TPIinit_vars.pkl")
variables = pickle.load(open(tpi_init, "rb"))
for key in variables:
    globals()[key] = variables[key]
params_given = os.path.join(VAR_DIR, "Saved_moments/params_given.pkl")
variables = pickle.load(open(params_given, "rb"))
for key in variables:
    globals()[key] = variables[key]

N_tilde = omega.sum(1)
omega_stationary_init = omega / N_tilde.reshape(T + S, 1)
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
c_path_init = c_path

inv_mat_init = firm.get_I(
    b_mat_init[1:T + 1], b_mat_init[:T], delta, g_y, g_n_vector[:T].reshape(T, 1, 1))
y_mat_init = c_path_init + inv_mat_init

# Lifetime Utility Graphs:
c_ut_init = np.zeros((S, S, J))
for s in xrange(S - 1):
    c_ut_init[:, s + 1, :] = c_path_init[s + 1:s + 1 + S, s + 1, :]
c_ut_init[:, 0, :] = c_path_init[:S, 0, :]
L_ut_init = np.zeros((S, S, J))
for s in xrange(S - 1):
    L_ut_init[:, s + 1, :] = n_mat_init[s + 1:s + 1 + S, s + 1, :]
L_ut_init[:, 0, :] = n_mat_init[:S, 0, :]
B_ut_init = BQpath_TPIbase[S:T]
b_ut_init = np.zeros((S, S, J))
for s in xrange(S):
    b_ut_init[:, s, :] = b_mat_init[s:s + S, s, :]

beq_ut = chi_b.reshape(1, S, J) * (rho.reshape(1, S, 1)) * \
    (b_ut_init[:S]**(1 - sigma) - 1) / (1 - sigma)
utility = ((c_ut_init ** (1 - sigma) - 1) / (1 - sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1 - (L_ut_init / ltilde)**upsilon) ** (1 / upsilon) + k_ellipse)
utility += beq_ut
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
cum_morts = np.cumprod(1 - rho)
utility *= cum_morts.reshape(1, S, 1)
utility_lifetime_init = utility.sum(1)

# Period Utility Graphs
beq_ut_period = chi_b.reshape(
    1, S, J) * (rho.reshape(1, S, 1)) * (b_mat_init[:S]**(1 - sigma) - 1) / (1 - sigma)
utility_period = ((c_path_init[:S] ** (1 - sigma) - 1) / (1 - sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1 - (n_mat_init[:S] / ltilde)**upsilon) ** (1 / upsilon) + k_ellipse)
utility_period += beq_ut_period
utility_period *= beta_string.reshape(1, S, 1)
utility_period *= cum_morts.reshape(1, S, 1)
utility_period_init = utility_period.sum(1)

'''
------------------------------------------------------------------------
    Create variables for tax experiment TPI graphs
------------------------------------------------------------------------
'''


ss_vars = os.path.join(VAR_DIR, "SS/ss_vars.pkl")
variables = pickle.load(open(ss_vars, "rb"))
for key in variables:
    globals()[key] = variables[key]
tpi_vars = os.path.join(VAR_DIR, "TPI/TPI_vars.pkl")
variables = pickle.load(open(tpi_vars, "rb"))
for key in variables:
    globals()[key] = variables[key]
params_changed = os.path.join(VAR_DIR, "Saved_moments/params_changed.pkl")
variables = pickle.load(open(params_changed, "rb"))
for key in variables:
    globals()[key] = variables[key]

N_tilde = omega.sum(1)
omega_stationary = omega / N_tilde.reshape(T + S, 1)
omega_stationary = omega_stationary[:T]

b1 = np.zeros((T, S, J))
b1[:, 1:, :] = b_mat[:T, :-1, :]
b2 = np.zeros((T, S, J))
b2[:, :, :] = b_mat[:T, :, :]

inv_mat = firm.get_I(b_mat[1:T + 1], b_mat[:T], delta,
                     g_y, g_n_vector[:T].reshape(T, 1, 1))
y_mat = c_path + inv_mat

# Lifetime Utility
c_ut = np.zeros((S, S, J))
for s in xrange(S - 1):
    c_ut[:, s + 1, :] = c_path[s + 1:s + 1 + S, s + 1, :]
c_ut[:, 0, :] = c_path[:S, 0, :]
L_ut = np.zeros((S, S, J))
for s in xrange(S - 1):
    L_ut[:, s + 1, :] = n_mat[s + 1:s + 1 + S, s + 1, :]
L_ut[:, 0, :] = n_mat[:S, 0, :]
B_ut = BQpath_TPI[S:T]
b_ut = np.zeros((S, S, J))
for s in xrange(S):
    b_ut[:, s, :] = b_mat[s:s + S, s, :]

beq_ut = chi_b.reshape(1, S, J) * (rho.reshape(1, S, 1)) * \
    (b_ut[:S]**(1 - sigma) - 1) / (1 - sigma)
utility = ((c_ut ** (1 - sigma) - 1) / (1 - sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1 - (L_ut / ltilde)**upsilon) ** (1 / upsilon) + k_ellipse)
utility += beq_ut
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
utility *= cum_morts.reshape(1, S, 1)
utility_lifetime = utility.sum(1)

# Period Utility
beq_ut_period = chi_b.reshape(
    1, S, J) * (rho.reshape(1, S, 1)) * (b_mat[:S]**(1 - sigma) - 1) / (1 - sigma)
utility_period = ((c_path[:S] ** (1 - sigma) - 1) / (1 - sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1 - (n_mat[:S] / ltilde)**upsilon) ** (1 / upsilon) + k_ellipse)
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
TPI_K = os.path.join(TPI_FIG_DIR, "TPI/TPI_K")
plt.savefig(TPI_K)

plt.figure()
plt.plot(np.arange(T), Kpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Capital Stock $\hat{K}$")
plt.legend(loc=0)
TPI_K = os.path.join(TPI_FIG_DIR, "TPIinit/TPI_K")
plt.savefig(TPI_K)

plt.figure()
plt.plot(np.arange(T), Lpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), Lpath_TPI[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Labor Supply $\hat{L}$")
plt.legend(loc=0)
TPI_L = os.path.join(TPI_FIG_DIR, "TPI/TPI_L")
plt.savefig(TPI_L)

plt.figure()
plt.plot(np.arange(T), Lpath_TPIbase[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Labor Supply $\hat{L}$")
plt.legend(loc=0)
TPI_L = os.path.join(TPI_FIG_DIR, "TPIinit/TPI_L")
plt.savefig(TPI_L)

plt.figure()
plt.plot(np.arange(T), (y_mat_init * omega_stationary.reshape(T, S, 1)
                        * lambdas).sum(1).sum(1)[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), (y_mat * omega_stationary.reshape(T, S, 1) *
                        lambdas).sum(1).sum(1)[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Output $\hat{Y}$")
plt.legend(loc=0)
TPI_Y = os.path.join(TPI_FIG_DIR, "TPI/TPI_Y")
plt.savefig(TPI_Y)

plt.figure()
plt.plot(np.arange(T), (y_mat_init * omega_stationary.reshape(T, S, 1)
                        * lambdas).sum(1).sum(1)[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Output $\hat{Y}$")
plt.legend(loc=0)
TPI_Y = os.path.join(TPI_FIG_DIR, "TPIinit/TPI_Y")
plt.savefig(TPI_Y)

plt.figure()
plt.plot(np.arange(T), w_base[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), winit[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Wage $\hat{w}$")
plt.legend(loc=0)
TPI_w = os.path.join(TPI_FIG_DIR, "TPI/TPI_w")
plt.savefig(TPI_w)

plt.figure()
plt.plot(np.arange(T), w_base[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Wage $\hat{w}$")
plt.legend(loc=0)
TPI_w = os.path.join(TPI_FIG_DIR, "TPIinit/TPI_w")
plt.savefig(TPI_w)

plt.figure()
plt.plot(np.arange(T), r_base[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), rinit[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Rental Rate $\hat{r}$")
plt.legend(loc=0)
TPI_r = os.path.join(TPI_FIG_DIR, "TPI/TPI_r")
plt.savefig(TPI_r)

plt.figure()
plt.plot(np.arange(T), r_base[:T], 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Rental Rate $\hat{r}$")
plt.legend(loc=0)
TPI_r = os.path.join(TPI_FIG_DIR, "TPIinit/TPI_r")
plt.savefig(TPI_r)

X3, Y3 = np.meshgrid(np.arange(S), np.arange(J) + 1)
cmap2 = matplotlib.cm.get_cmap('winter')
fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, ((utility_lifetime - utility_lifetime_init) /
                          np.abs(utility_lifetime_init)).T, rstride=1, cstride=1, cmap=cmap2)
utility_lifetime_percdif = os.path.join(
    TPI_FIG_DIR, "TPI/utility_lifetime_percdif")
plt.savefig(utility_lifetime_percdif)

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, (utility_lifetime_init).T,
                 rstride=1, cstride=1, cmap=cmap2)
utility_lifetime = os.path.join(TPI_FIG_DIR, "TPIinit/utility_lifetime")
plt.savefig(utility_lifetime)

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, utility_lifetime.T, rstride=1, cstride=1, cmap=cmap2)
utility_lifetime = os.path.join(TPI_FIG_DIR, "TPI/utility_lifetime")
plt.savefig(utility_lifetime)

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, ((utility_period - utility_period_init) /
                          np.abs(utility_period_init)).T, rstride=1, cstride=1, cmap=cmap2)
utility_period_percdif = os.path.join(
    TPI_FIG_DIR, "TPI/utility_period_percdif")
plt.savefig(utility_period_percdif)

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, (utility_period_init).T,
                 rstride=1, cstride=1, cmap=cmap2)
utility_period = os.path.join(TPI_FIG_DIR, "TPIinit/utility_period")
plt.savefig(utility_period)

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, utility_period.T, rstride=1, cstride=1, cmap=cmap2)
utility_period = os.path.join(TPI_FIG_DIR, "TPI/utility_period")
plt.savefig(utility_period)


'''
------------------------------------------------------------------------
Plot Timepath for B
------------------------------------------------------------------------
'''

for i in xrange(J):
    plt.figure()
    plt.plot(np.arange(
        T), BQpath_TPIbase[:T, i], linewidth=2, color='b', label="Base TPI time path for group j={}".format(i + 1))
    plt.plot(np.arange(
        T), BQpath_TPI[:T, i], linewidth=2, linestyle='--', color='g', label="TPI time path for group j={}".format(i + 1))
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Aggregate $\hat{BQ_{j,t}}$")
    plt.legend(loc=0)
    fig_i = os.path.join(TPI_FIG_DIR, "TPI/TPI_B_j{}".format(i + 1))
    plt.savefig(fig_i)

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
euler_errors_TPI = os.path.join(TPI_FIG_DIR, "TPIinit/euler_errors_TPI")
plt.savefig(euler_errors_TPI)

domain = np.linspace(1, T, T)
plt.figure()
plt.plot(domain, eul_savings, label='Euler1')
plt.plot(domain, eul_laborleisure, label='Euler2')
plt.ylabel('Error Value')
plt.xlabel(r'Time $t$')
plt.legend(loc=0)
plt.title('Maximum Euler Error for each period across S and J')
euler_errors_TPI = os.path.join(TPI_FIG_DIR, "TPI/euler_errors_TPI")
plt.savefig(euler_errors_TPI)

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
    idx = np.argsort(collapseS - omega1, axis=1)
    collapseS2 = collapseS[:, idx][np.eye(T, T, dtype=bool)]
    omega_sorted = omega1[:, idx][np.eye(T, T, dtype=bool)]
    cum_levels = np.cumsum(collapseS2, axis=1)
    cum_omega = np.cumsum(omega_sorted, axis=1)
    # plt.figure()
    # plt.plot(cum_omega[10], cum_levels[10])
    # plt.plot(cum_omega[10], cum_omega[10])
    # plt.savefig('gini_test.png')
    cum_levels[:, 1:] = cum_levels[:, :-1]
    cum_levels[:, 0] = 0
    G = 2 * (.5 - (cum_levels * omega_sorted).sum(1))
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
    idx = np.argsort(collapseJ - omega1, axis=1)
    collapseJ2 = collapseJ[:, idx][np.eye(T, T, dtype=bool)]
    omega_sorted = omega1[:, idx][np.eye(T, T, dtype=bool)]
    cum_levels = np.cumsum(collapseJ2, axis=1)
    cum_omega = np.cumsum(omega_sorted, axis=1)
    cum_levels[:, 1:] = cum_levels[:, :-1]
    cum_levels[:, 0] = 0
    G = 2 * (.5 - (cum_levels * omega_sorted).sum(1))
    print G[-1]
    return G


def gini_nocol(path, omega):
    # Inequality across age and ability
    pathx = np.copy(path)
    mask = pathx < 0
    pathx[mask] = 0
    pathx = pathx.reshape(T, S * J)
    total = pathx.sum(1)
    pathx /= total.reshape(T, 1)
    omega = omega.reshape(T, S * J)
    idx = np.argsort(pathx - omega, axis=1)
    path2 = pathx[:, idx][np.eye(T, T, dtype=bool)]
    omega_sorted = omega[:, idx][np.eye(T, T, dtype=bool)]
    cum_levels = np.cumsum(path2, axis=1)
    cum_omega = np.cumsum(omega_sorted, axis=1)
    cum_levels[:, 1:] = cum_levels[:, :-1]
    cum_levels[:, 0] = 0
    G = 2 * (.5 - (cum_levels * omega_sorted).sum(1))
    print G[-1]
    return G

'''
------------------------------------------------------------------------
    GINI plots comparing the tax experiment to the baseline
------------------------------------------------------------------------
'''

omega_stationary_init_gini = np.tile(
    omega_stationary_init.reshape(T, S, 1), (1, 1, J)) * lambdas
omega_stationary_gini = np.tile(
    omega_stationary.reshape(T, S, 1), (1, 1, J)) * lambdas


plt.figure()
plt.plot(np.arange(T), gini_cols(b_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(
    b_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
gini_b_cols = os.path.join(TPI_FIG_DIR, "TPI/gini_b_cols")
plt.savefig(gini_b_cols)

plt.figure()
plt.plot(np.arange(T), gini_cols(n_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(
    n_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
gini_l_cols = os.path.join(TPI_FIG_DIR, "TPI/gini_l_cols")
plt.savefig(gini_l_cols)

plt.figure()
plt.plot(np.arange(T), gini_cols(y_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(
    y_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
gini_y_cols = os.path.join(TPI_FIG_DIR, "TPI/gini_y_cols")
plt.savefig(gini_y_cols)

plt.figure()
plt.plot(np.arange(T), gini_cols(c_path_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(
    c_path[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
gini_c_cols = os.path.join(TPI_FIG_DIR, "TPI/gini_c_cols")
plt.savefig(gini_c_cols)


plt.figure()
plt.plot(np.arange(T), gini_colj(b_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(
    b_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
gini_b_colj = os.path.join(TPI_FIG_DIR, "TPI/gini_b_colj")
plt.savefig(gini_b_colj)

plt.figure()
plt.plot(np.arange(T), gini_colj(n_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(
    n_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
gini_l_colj = os.path.join(TPI_FIG_DIR, "TPI/gini_l_colj")
plt.savefig(gini_l_colj)

plt.figure()
plt.plot(np.arange(T), gini_colj(y_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(
    y_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
gini_y_colj = os.path.join(TPI_FIG_DIR, "TPI/gini_y_colj")
plt.savefig(gini_y_colj)

plt.figure()
plt.plot(np.arange(T), gini_colj(c_path_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_colj(
    c_path[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
gini_c_colj = os.path.join(TPI_FIG_DIR, "TPI/gini_c_colj")
plt.savefig(gini_c_colj)

plt.figure()
plt.plot(np.arange(T), gini_nocol(b_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(
    b_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
gini_b_nocol = os.path.join(TPI_FIG_DIR, "TPI/gini_b_nocol")
plt.savefig(gini_b_nocol)

plt.figure()
plt.plot(np.arange(T), gini_nocol(n_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(
    n_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
gini_l_nocol = os.path.join(TPI_FIG_DIR, "TPI/gini_l_nocol")
plt.savefig(gini_l_nocol)

plt.figure()
plt.plot(np.arange(T), gini_nocol(y_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(
    y_mat[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
gini_y_nocol = os.path.join(TPI_FIG_DIR, "TPI/gini_y_nocol")
plt.savefig(gini_y_nocol)

plt.figure()
plt.plot(np.arange(T), gini_nocol(c_path_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_nocol(
    c_path[:T], omega_stationary_gini), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
gini_c_nocol = os.path.join(TPI_FIG_DIR, "TPI/gini_c_nocol")
plt.savefig(gini_c_nocol)

'''
------------------------------------------------------------------------
    Baseline TPI graphs
------------------------------------------------------------------------
'''

plt.figure()
plt.plot(np.arange(T), gini_cols(b_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
gini_b_cols = os.path.join(TPI_FIG_DIR, "TPIinit/gini_b_cols")
plt.savefig(gini_b_cols)

plt.figure()
plt.plot(np.arange(T), gini_cols(n_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
gini_l_cols = os.path.join(TPI_FIG_DIR, "TPIinit/gini_l_cols")
plt.savefig(gini_l_cols)

plt.figure()
plt.plot(np.arange(T), gini_cols(y_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
gini_y_cols = os.path.join(TPI_FIG_DIR, "TPIinit/gini_y_cols")
plt.savefig(gini_y_cols)

plt.figure()
plt.plot(np.arange(T), gini_cols(c_path_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
gini_c_cols = os.path.join(TPI_FIG_DIR, "TPIinit/gini_c_cols")
plt.savefig(gini_c_cols)

plt.figure()
plt.plot(np.arange(T), gini_colj(b_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
gini_b_colj = os.path.join(TPI_FIG_DIR, "TPIinit/gini_b_colj")
plt.savefig(gini_b_colj)

plt.figure()
plt.plot(np.arange(T), gini_colj(n_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
gini_l_colj = os.path.join(TPI_FIG_DIR, "TPIinit/gini_l_colj")
plt.savefig(gini_l_colj)

plt.figure()
plt.plot(np.arange(T), gini_colj(y_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
gini_y_colj = os.path.join(TPI_FIG_DIR, "TPIinit/gini_y_colj")
plt.savefig(gini_y_colj)

plt.figure()
plt.plot(np.arange(T), gini_colj(c_path_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
gini_c_colj = os.path.join(TPI_FIG_DIR, "TPIinit/gini_c_colj")
plt.savefig(gini_c_colj)

plt.figure()
plt.plot(np.arange(T), gini_nocol(b_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
gini_b_nocol = os.path.join(TPI_FIG_DIR, "TPIinit/gini_b_nocol")
plt.savefig(gini_b_nocol)

plt.figure()
plt.plot(np.arange(T), gini_nocol(n_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{l}$")
plt.legend(loc=0)
gini_l_nocol = os.path.join(TPI_FIG_DIR, "TPIinit/gini_l_nocol")
plt.savefig(gini_l_nocol)

plt.figure()
plt.plot(np.arange(T), gini_nocol(y_mat_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{y}$")
plt.legend(loc=0)
gini_y_nocol = os.path.join(TPI_FIG_DIR, "TPIinit/gini_y_nocol")
plt.savefig(gini_y_nocol)

plt.figure()
plt.plot(np.arange(T), gini_nocol(c_path_init[
         :T], omega_stationary_init_gini), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{c}$")
plt.legend(loc=0)
gini_c_nocol = os.path.join(TPI_FIG_DIR, "TPIinit/gini_c_nocol")
plt.savefig(gini_c_nocol)


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
#     ax5.plot_surface(X, Y, ((c_path[t] - c_path_init[t])/c_path_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
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
#     ax5.plot_surface(X, Y, c_path_init[t].T, rstride=1, cstride=1, cmap=cmap2)
#     name = "%03d" % t
#     plt.savefig('OUTPUT/TPI/movies/cons_base/c_base_T{}'.format(name))
