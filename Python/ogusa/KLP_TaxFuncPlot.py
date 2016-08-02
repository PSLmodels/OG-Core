############################################################
# Bivarite Tax Function Plots                              #
# Author: Kerk Phillips                                    #
# Last Update: 7/30/12                                     #
############################################################

# Import Statements
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# def TaxFuncCD(x, y, par):

#     # unpack par
#     A = par[0]
#     B = par[1]
#     C = par[2]
#     D = par[3]
#     E = par[4]
#     F = par[5]
#     minx = par[6]
#     miny = par[7]
#     maxx = par[8]
#     maxy = par[9]
#     alf = par[10]

#     # Calculate x and y Tax Functions
#     taux = ((A*x**2 + B*x) / (A*x**2 + B*x + C))*(maxx - minx) + minx
#     tauy = ((D*y**2 + E*y) / (D*y**2 + E*y + F))*(maxy - miny) + miny

#     # Combine These via Cobb-Douglas aggregator
#     tau = (taux - minx)**alf * (tauy - miny)**(1-alf)
#     adj = alf*minx + (1-alf)*miny
#     tau = tau + adj
#     return tau


def TaxFuncCES(x, y, par):

    # unpack par
    A = par[0]
    B = par[1]
    C = par[2]
    D = par[3]
    E = par[4]
    F = par[5]
    minx = par[6]
    miny = par[7]
    maxx = par[8]
    maxy = par[9]
    alf = par[10]
    eta = par[11]

    # Calculate x and y Tax Functions
    taux = ((A*x**2 + B*x) / (A*x**2 + B*x + C))*(maxx - minx) + minx
    tauy = ((D*y**2 + E*y) / (D*y**2 + E*y + F))*(maxy - miny) + miny

    # Combine These via Cobb-Douglas aggregator
    tau = (alf*(taux - minx)**eta + (1-alf)*(tauy - miny)**eta)**(1/eta)
    adj = alf*minx + (1-alf)*miny
    tau = tau + adj
    return tau


# Declare Function Parameters
A = 1.0
B = 1.0
C = 1.0
D = 1.0
E = 1.0
F = 1.0
minx = -.1
miny = -.1
maxx = .5
maxy = .25
alf = .5
eta = 2.0
par = np.array([A, B, C, D, E, F, minx, miny, maxx, maxy, alf, eta])

# Grid Size
xsize = 10
ysize = 10

# Initialize Matrix of Inputs and Tax Rates
Xmat = np.zeros((xsize,ysize))
Ymat = np.zeros((xsize,ysize))
tauCD = np.zeros((xsize,ysize))
tauCES = np.zeros((xsize,ysize))

# Calculate Taxes
for x in range(0, xsize):
    for y in range(0, ysize):
        Xmat[x, y] = x
        Ymat[x, y] = y
        # tauCD[x, y] = TaxFuncCD(x, y, par)
        tauCES[x, y] = TaxFuncCES(x, y, par)

# Plot Surface Plot of Tax Rates
# # Twice as wide as it is tall.
# fig = plt.figure(figsize=plt.figaspect(0.5))

# # CD plot
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# surf = ax.plot_wireframe(Xmat, Ymat, tauCD)
# plt.title('Cobb-Douglas')

# CES plot
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_wireframe(Xmat, Ymat, tauCES)
plt.title('CES')

plt.show()