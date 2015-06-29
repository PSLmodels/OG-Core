import numpy as np
from demographics import get_omega
from income import get_e

# Parameters
S = 80
J = 7
T = int(2 * S)
lambdas = np.array([.25, .25, .2, .1, .1, .09, .01])
starting_age = 20
ending_age = 100
E = int(starting_age * (S / float(ending_age-starting_age)))
beta_annual = .96
beta = beta_annual ** (float(ending_age-starting_age) / S)
sigma = 3.0
alpha = .35
Z = 1.0
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (float(ending_age-starting_age) / S))
ltilde = 1.0
g_y_annual = 0.03
g_y = (1 + g_y_annual)**(float(ending_age-starting_age)/S) - 1
# TPI parameters
maxiter = 250
mindist_SS = 1e-9
mindist_TPI = 1e-6
nu = .40
# Ellipse parameters
b_ellipse = 25.6594
k_ellipse = -26.4902
upsilon = 3.0542
# Tax parameters:
mean_income_data = 84377.0
a_tax_income = 3.03452713268985e-06
b_tax_income = .222
c_tax_income = 133261.0
d_tax_income = .219
retire = np.round(9.0 * S / 16.0) - 1
# Wealth tax params
# These won't be used for the wealth tax, h and m just need
# need to be nonzero to avoid errors
h_wealth = 0.1
m_wealth = 1.0
p_wealth = 0.0
# Tax parameters that are zeroed out for SS
# Initial taxes below
tau_bq = np.zeros(J)
tau_payroll = 0.15
# Flag to prevent graphing from occuring in demographic, income, wealth, and labor files
flag_graphs = False
# Generate Income and Demographic parameters
omega, g_n, omega_SS, surv_rate = get_omega(
    S, J, T, lambdas, starting_age, ending_age, E, flag_graphs)
e = get_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)
rho = 1-surv_rate
rho[-1] = 1.0
