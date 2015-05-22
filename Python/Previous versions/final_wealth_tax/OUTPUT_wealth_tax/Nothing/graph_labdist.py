import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S = 80
J = 7
bin_weights = np.array([.25, .25, .2, .1, .1, .09, .01])


variables = pickle.load(open("SS_init_solutions.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

variables = pickle.load(open("labor_data_moments.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

Kssmat = solutions[:S*J].reshape(S, J)
Lssmat = solutions[S*J:2*S*J].reshape(S, J)

domain = np.linspace(20, 100, S)
Jgrid = np.zeros(J)
for j in xrange(J):
    Jgrid[j:] += bin_weights[j]
cmap1 = matplotlib.cm.get_cmap('summer')
cmap2 = matplotlib.cm.get_cmap('jet')
X, Y = np.meshgrid(domain, Jgrid)
X2, Y2 = np.meshgrid(domain[1:], Jgrid)


fig5 = plt.figure()
ax5 = fig5.gca(projection='3d')
ax5.set_xlabel(r'age-$s$')
ax5.set_ylabel(r'ability type-$j$')
ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
ax5.plot_surface(X, Y, Kssmat.T, rstride=1, cstride=1, cmap=cmap2)
plt.savefig('capital_dist')

fig4 = plt.figure()
ax4 = fig4.gca(projection='3d')
ax4.set_xlabel(r'age-$s$')
ax4.set_ylabel(r'ability-$j$')
ax4.set_zlabel(r'individual labor supply $\bar{l}_{j,s}$')
ax4.plot_surface(X, Y, (Lssmat).T, rstride=1, cstride=1, cmap=cmap1)
plt.savefig('labor_dist')

onedlab = (Lssmat * bin_weights).sum(1)
fig6 = plt.figure()
plt.plot(np.arange(80)+20, onedlab, label='Model', color='black', linestyle='--')
plt.plot(np.arange(80)+20, labor_dist_data, label='Data', color='black', linestyle='-')
plt.legend()
plt.ylabel(r'individual labor supply $\bar{l}_{s}$')
plt.xlabel(r'age-$s$')
plt.savefig('labor_dist_comparison')