

# Packages
import numpy as np
import time
import scipy.optimize as opt
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import income
import demographics
import tax_funcs as tax
import os


variables = pickle.load(open("OUTPUT/given_params.pkl", "r"))
for key in variables:
    globals()[key] = variables[key]

if os.path.isfile("OUTPUT/SS/d_inc_guess.pkl"):
    d_tax_income = pickle.load(open("OUTPUT/SS/d_inc_guess.pkl", "r"))


omega, g_n, omega_SS, children, surv_rate = demographics.get_omega(
    S, J, T, bin_weights, starting_age, ending_age, E)
e = income.get_e(S, J, starting_age, ending_age, bin_weights, omega_SS)