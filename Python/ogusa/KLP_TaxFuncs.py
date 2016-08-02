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