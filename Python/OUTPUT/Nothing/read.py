import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

S = 80
J = 7
bin_weights = np.array([.25, .25, .2, .1, .1, .08, .02])


variables = pickle.load(open("eight_two.pkl", "r"))
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