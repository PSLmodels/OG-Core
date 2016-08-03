############################################################
# Bivarite Tax Function Plots                              #
# Author: Kerk Phillips                                    #
# Last Update: 7/30/12                                     #
############################################################

# Import Statements
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def TaxFuncCES(x, y, par):

    # unpack par
    A = par[0]
    B = par[1]
    D = par[2]
    E = par[3]
    minx = par[4]
    miny = par[5]
    maxx = par[6]
    maxy = par[7]
    alf = par[8]
    eta = par[9]

    # Calculate x and y Tax Functions
    taux = ((A*x**2 + B*x) / (A*x**2 + B*x + 1))*(maxx - minx) + minx
    tauy = ((D*y**2 + E*y) / (D*y**2 + E*y + 1))*(maxy - miny) + miny

    # Combine These via Cobb-Douglas aggregator
    tau = (alf*(taux - minx)**eta + (1-alf)*(tauy - miny)**eta)**(1/eta)
    adj = alf*minx + (1-alf)*miny
    tau = tau + adj

    return tau