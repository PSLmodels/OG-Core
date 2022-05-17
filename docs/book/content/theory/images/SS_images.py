"""
------------------------------------------------------------------------
This script creates the plots for the steady-state chapter of the OG-Core
documentation
------------------------------------------------------------------------
"""
# Import libraries, packages, and modules
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

"""
------------------------------------------------------------------------
Set current folder path. Set path for pickle. Load pickle
------------------------------------------------------------------------
"""
cur_path = os.path.split(os.path.abspath(__file__))[0]
ss_path = (
    "/Users/rwe/Documents/Economics/OSPC/OG-Core/run_examples/"
    + "OUTPUT_BASELINE_CLOSED/SS/SS_vars.pkl"
)
tpi_path = (
    "/Users/rwe/Documents/Economics/OSPC/OG-Core/run_examples/"
    + "OUTPUT_BASELINE_CLOSED/TPI/TPI_vars.pkl"
)
ss_vars = pickle.load(open(ss_path, "rb"))
cssmat = ss_vars["cssmat"]
nssmat = ss_vars["nssmat"]
bssmat = ss_vars["bssmat"]
tpi_vars = pickle.load(open(tpi_path, "rb"))

"""
------------------------------------------------------------------------
Plot the steady-state distribution of household consumption
------------------------------------------------------------------------
"""
E = 20
S = 80
J = 7
jgrid = np.array([0.125, 0.375, 0.6, 0.75, 0.85, 0.94, 0.995])
sgrid = np.linspace(E + 1, E + S, S)
smat, jmat = np.meshgrid(sgrid, jgrid)
cmap_c = cm.get_cmap("summer")
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_xlabel(r"age-$s$")
ax.set_ylabel(r"lifetime income group-$j$")
ax.set_zlabel(r"consumption $c_{j,s}$")
strideval = max(int(1), int(round(S / 10)))
ax.plot_surface(
    smat, jmat, cssmat.T, rstride=strideval, cstride=strideval, cmap=cmap_c
)
output_path = os.path.join(cur_path, "HHcons_SS")
plt.savefig(output_path)
# plt.show()
plt.close()

"""
------------------------------------------------------------------------
Plot the steady-state distribution of household labor supply
------------------------------------------------------------------------
"""
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_xlabel(r"age-$s$")
ax.set_ylabel(r"lifetime income group-$j$")
ax.set_zlabel(r"labor supply $n_{j,s}$")
strideval = max(int(1), int(round(S / 10)))
ax.plot_surface(
    smat, jmat, nssmat.T, rstride=strideval, cstride=strideval, cmap=cmap_c
)
output_path = os.path.join(cur_path, "HHlab_SS")
plt.savefig(output_path)
# plt.show()
plt.close()

"""
------------------------------------------------------------------------
Plot the steady-state distribution of household savings
------------------------------------------------------------------------
"""
bssmat_s = np.vstack((np.zeros(J), bssmat))
sgrid_s = np.linspace(E + 1, E + S + 1, S + 1)
smat_s, jmat = np.meshgrid(sgrid_s, jgrid)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_xlabel(r"age-$s$")
ax.set_ylabel(r"lifetime income group-$j$")
ax.set_zlabel(r"savings $b_{j,s}$")
strideval = max(int(1), int(round(S / 10)))
ax.plot_surface(
    smat_s, jmat, bssmat_s.T, rstride=strideval, cstride=strideval, cmap=cmap_c
)
output_path = os.path.join(cur_path, "HHsav_SS")
plt.savefig(output_path)
# plt.show()
plt.close()

"""
------------------------------------------------------------------------
Print steady-state table values
------------------------------------------------------------------------
"""
Kss = ss_vars["Kss"]
Iss = ss_vars["Iss"]
BQss = ss_vars["BQss"].sum()
Css = ss_vars["Css"]
eulers_lab = ss_vars["euler_labor_leisure"]
maxabs_eul_lab = np.absolute(eulers_lab).max()
eulers_sav = ss_vars["euler_savings"]
maxabs_eul_sav = np.absolute(eulers_sav).max()
Bss = ss_vars["Bss"]
Gss = ss_vars["Gss"]
T_Hss = ss_vars["T_Hss"]
Yss = ss_vars["Yss"]
rss = ss_vars["rss"]
wss = ss_vars["wss"]
revenue_ss = ss_vars["revenue_ss"]
Lss = ss_vars["Lss"]
factor_ss = ss_vars["factor_ss"]
Dpath = tpi_vars["D"]
Dss = Dpath[-5]
res_constr_err = Yss - Css - Iss - Gss

print("rss", rss, "wss", wss)
print("Yss", Yss, "Css", Css)
print("Iss", Iss, "Kss", Kss)
print("Lss", Lss, "Bss", Bss)
print("BQss", BQss, "factor", factor_ss)
print("Rev", revenue_ss, "TR", T_Hss)
print("Gss", Gss, "Dss", Dss)
print("max lab err", maxabs_eul_lab, "max sav err", maxabs_eul_sav)
print("Res const err", res_constr_err, "ss_time", "Need this value")
