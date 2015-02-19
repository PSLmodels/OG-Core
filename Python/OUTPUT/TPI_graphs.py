'''
------------------------------------------------------------------------
Last updated 12/1/2014

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
Y_mat_init = A * (K_mat_init.sum(1) ** alpha) * (L_mat_init.sum(1) ** (1-alpha))
Tinitbase = Tinit


K1 = np.zeros((T, S, J))
K1[:, 1:, :] = K_mat_init[:T, :-1, :]
K2 = np.zeros((T, S, J))
K2[:, :, :] = K_mat_init[:T, :, :]
cinitbase = cinit
Y_mat_init = r_base[:T].reshape(T, 1, 1) * K1[:T].reshape(T, S, J) + w_base[:T].reshape(
    T, 1, 1) * e.reshape(1, S, J) * L_mat_init[:T].reshape(T, S, J) + r_base[:T].reshape(
    T, 1, 1) * Bpath_TPIbase[:T].reshape(T, 1, J) / bin_weights.reshape(1, 1, J) - taxinit.reshape(T, S, J)

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
for s in xrange(S-1):
    K_ut_init[:, s+1, :] = K_mat_init[s+1:s+1+S, s+1, :]
K_ut_init[:, 0, :] = 0

beq_ut = chi_b.reshape(1, S, 1) * (K_ut_init[:S]**(1-sigma)-1)/(1-sigma)
beq_ut[:, 0, :] = 0
utility = ((c_ut_init ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_ut_init/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility[:, -1, :] += chi_b.reshape(S, 1) * (B_ut_init**(1-sigma)-1) / (1-sigma)
utility *= mort_rate.reshape(1, S, 1)
utility += beq_ut * (1- mort_rate.reshape(1, S, 1))
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
cum_morts = np.zeros(S)
for i in xrange(S):
    cum_morts[i] = np.prod(1-mort_rate[:i])
utility *= cum_morts.reshape(1, S, 1)
utility_init = utility.sum(1)


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
Y_mat = rinit[:T].reshape(T, 1, 1) * K1[:T].reshape(T, S, J) + winit[:T].reshape(T, 1, 1) * e.reshape(
    1, S, J) * L_mat[:T].reshape(T, S, J) + rinit[:T].reshape(T, 1, 1) * Bpath_TPI[:T].reshape(
    T, 1, J) / bin_weights.reshape(1, 1, J) - taxinit2.reshape(T, S, J)

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
for s in xrange(S-1):
    K_ut[:, s+1, :] = K_mat[s+1:s+1+S, s+1, :]
K_ut[:, 0, :] = 0

beq_ut = chi_b.reshape(1, S, 1) * (K_ut[:S]**(1-sigma)-1)/(1-sigma)
beq_ut[:, 0, :] = 0
utility = ((c_ut ** (1-sigma) - 1)/(1- sigma)) + chi_n.reshape(1, S, 1) * (
    b_ellipse * (1-(L_ut/ltilde)**upsilon) ** (1/upsilon) + k_ellipse)
utility[:, -1, :] += chi_b.reshape(S, 1) * (B_ut**(1-sigma)-1) / (1-sigma)
utility *= mort_rate.reshape(1, S, 1)
utility += beq_ut * (1- mort_rate.reshape(1, S, 1))
beta_string = np.ones(S) * beta
for i in xrange(S):
    beta_string[i] = beta_string[i] ** i
utility *= beta_string.reshape(1, S, 1)
cum_morts = np.zeros(S)
for i in xrange(S):
    cum_morts[i] = np.prod(1-mort_rate[:i])
utility *= cum_morts.reshape(1, S, 1)
utility = utility.sum(1)


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
plt.plot(np.arange(T), Y_base[:T], 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), Yinit[:T], 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Aggregate Output $\hat{Y}$")
plt.legend(loc=0)
plt.savefig("TPI/TPI_Y")

plt.figure()
plt.plot(np.arange(T), Y_base[:T], 'b', linewidth=2, label='Baseline')
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
ax5.plot_surface(X3, Y3, ((utility - utility_init)).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPI/utility_dif')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, (utility_init).T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPIinit/utility')

fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'time-$t$')
ax5.set_ylabel(r'ability-$j$')
ax5.set_zlabel(r'Utility $\bar{u}_{j,t}$')
ax5.plot_surface(X3, Y3, utility.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('TPI/utility')

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
GINI
'''


def gini_cols(path, omega):
    mask = path < 0
    path[mask] = 0
    collapseS = path.sum(1)
    omega_stationary1 = omega_stationary.sum(1)
    omega_stationary_sorted = omega_stationary1.reshape(T, J)
    idx = np.argsort(collapseS, axis=1)
    collapseS = collapseS[:, idx][np.eye(T,T, dtype=bool)]
    omega_stationary_sorted = omega_stationary_sorted[:, idx][np.eye(T, T, dtype=bool)]
    Y = collapseS.sum(1)
    Y_M = np.tile(collapseS.reshape(T, 1, J), (1, J, 1))
    OME_mat = np.tril(np.tile(omega_stationary_sorted.reshape(T, J, 1), (1, 1, J)), -1)
    G = .5 - (OME_mat * Y_M).sum(1).sum(1) / Y
    return G


def gini_colj(path, omega):
    mask = path < 0
    path[mask] = 0
    collapseS = path.sum(2)
    omega_stationary1 = omega_stationary.sum(2)
    omega_stationary_sorted = omega_stationary1.reshape(T, S)
    idx = np.argsort(collapseS, axis=1)
    collapseS = collapseS[:, idx][np.eye(T,T, dtype=bool)]
    omega_stationary_sorted = omega_stationary_sorted[:, idx][np.eye(T, T, dtype=bool)]
    Y = collapseS.sum(1)
    Y_M = np.tile(collapseS.reshape(T, 1, S), (1, S, 1))
    OME_mat = np.tril(np.tile(omega_stationary_sorted.reshape(T, S, 1), (1, 1, S)), -1)
    G = .5 - (OME_mat * Y_M).sum(1).sum(1) / Y
    return G


def gini_nocol(path, omega):
    mask = path < 0
    path[mask] = 0
    collapseS = path.reshape(T, S*J)
    omega_stationary_sorted = omega_stationary.reshape(T, S*J)
    idx = np.argsort(collapseS, axis=1)
    collapseS = collapseS[:, idx][np.eye(T,T, dtype=bool)]
    omega_stationary_sorted = omega_stationary_sorted[:, idx][np.eye(T,T, dtype=bool)]
    Y = collapseS.sum(1)
    Y_M = np.tile(collapseS.reshape(T, 1, S*J), (1, S*J, 1))
    OME_mat = np.tril(np.tile(omega_stationary_sorted.reshape(T, S*J, 1), (1, 1, S*J)), -1)
    G = .5 - (OME_mat * Y_M).sum(1).sum(1) / Y
    return G

'''
gini graphs showing both lines
'''

plt.figure()
plt.plot(np.arange(T), gini_cols(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.plot(np.arange(T), gini_cols(K_mat[:T], omega_stationary), 'g--', linewidth=2, label="Tax")
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPI/gini_b_cols")

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
gini graphs showing one line
'''

plt.figure()
plt.plot(np.arange(T), gini_cols(K_mat_init[:T], omega_stationary_init), 'b', linewidth=2, label='Baseline')
plt.xlabel(r"Time $t$")
plt.ylabel(r"Gini for $\hat{b}$")
plt.legend(loc=0)
plt.savefig("TPIinit/gini_b_cols")

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

# Wealth, income, consumption

# '''
# SS init graphs
# '''

domain = np.linspace(starting_age, ending_age, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += bin_weights[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)

# glance = [10, 15, 20, 25]
# for i in glance:
#     fig5 = plt.figure()
#     ax5 = fig5.gca(projection='3d')
#     ax5.set_xlabel(r'age-$s$')
#     ax5.set_ylabel(r'ability-$j$')
#     ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
#     ax5.plot_surface(X, Y, (K_mat[i]-K_mat_init[i]).T, rstride=1, cstride=1, cmap=cmap2)
#     plt.savefig('TPI/wealth_T{}_absolutedif'.format(i))


#     fig4 = plt.figure()
#     ax4 = fig4.gca(projection='3d')
#     ax4.set_xlabel(r'age-$s$')
#     ax4.set_ylabel(r'ability-$j$')
#     ax4.set_zlabel(r'individual consumption $\bar{c}_{j,s}$')
#     ax4.plot_surface(X, Y, ((Y_mat[i]-Y_mat_init[i])/Y_mat_init[i]).T, rstride=1, cstride=1, cmap=cmap1)
#     plt.savefig('TPI/income_T{}_percdif'.format(i))

#     fig9 = plt.figure()
#     ax9 = fig9.gca(projection='3d')
#     ax9.plot_surface(X, Y, ((cinit[i]-cinitbase[i])/cinitbase[i]).T, rstride=1, cstride=1, cmap=cmap2)
#     ax9.set_xlabel(r'age-$s$')
#     ax9.set_ylabel(r'ability-$j$')
#     ax9.set_zlabel('Consumption')
#     ax9.set_title('Steady State Distribution of Consumption')
#     plt.savefig('TPI/cons_T{}_percdif'.format(i))

'''
Gen graphs for movies
'''
print 'Starting movies'

for t in xrange(60):

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
    ax5.set_zlim([-.25, .15])
    # ax5.set_zlim([-.2, .05])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((K_mat[t] - K_mat_init[t])/K_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/b_dif/b_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'individual labor supply $l_{j,s}$')
    ax5.set_zlim([-.35, .25])
    # ax5.set_zlim([-.05, .3])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((L_mat[t] - L_mat_init[t])/L_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/l_dif/l_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'Consumption $c_{j,s}$')
    ax5.set_zlim([-.35, .15])
    # ax5.set_zlim([-.45, .15])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((cinit[t] - cinitbase[t])/cinitbase[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/c_dif/c_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'Income $y_{j,s}$')
    ax5.set_zlim([-.25, .15])
    # ax5.set_zlim([-.25, 0.05])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, ((Y_mat[t] - Y_mat_init[t])/Y_mat_init[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/y_dif/y_dif_T{}'.format(name))

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability-$j$')
    ax5.set_zlabel(r'Log Consumption $log(c_{j,s})$')
    ax5.set_zlim([-2.5, 1.1])
    # ax5.set_zlim([-.2, .05])
    ax5.set_title('T = {}'.format(t))
    ax5.plot_surface(X, Y, np.log(cinitbase[t]).T, rstride=1, cstride=1, cmap=cmap2)
    name = "%03d" % t
    plt.savefig('TPI/movies/cons_base/c_base_T{}'.format(name))

