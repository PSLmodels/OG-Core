############################################################
# Bivarite Tax Function Plots                              #
# Author: Kerk Phillips                                    #
# Last Update: 7/30/12                                     #
############################################################

# Import Statements
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from KLP_TaxFuncs import TaxFuncCES


# Declare Function Parameters
A = 1.0E-10
B = 0.0001
D = 1.0E-10
E = 0.0001
minx = 0.0
miny = -.05
maxx = .35
maxy = .3
alf = .5
eta = 2.0
par = np.array([A, B, D, E, minx, miny, maxx, maxy, alf, eta])

# Grid Size
xsize = 8
ysize = 8
scale = 20000

# Initialize Matrix of Inputs and Tax Rates
Xmat = np.zeros((xsize,ysize))
Ymat = np.zeros((xsize,ysize))
tauCD = np.zeros((xsize,ysize))
tauCES = np.zeros((xsize,ysize))

# Calculate Taxes
for x in range(0, xsize):
    for y in range(0, ysize):
        Xmat[x, y] = x*scale
        Ymat[x, y] = y*scale
        # tauCD[x, y] = TaxFuncCD(x, y, par)
        tauCES[x, y] = TaxFuncCES(x*scale, y*scale, par)

# CES plot
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_wireframe(Xmat, Ymat, tauCES)
plt.title('CES')

plt.show()

print(tauCES)