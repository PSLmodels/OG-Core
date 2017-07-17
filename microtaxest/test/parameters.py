'''
------------------------------------------------------------------------
This file generates all parameters for both the full model and the
scaled down version for Travis CI testing.

This py-file calls the following other file(s):
            income.py
            demographics.py
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
Import Packages
------------------------------------------------------------------------
'''
import numpy as np
import txfunc as txfn
from demographics import get_omega
from income import get_e

'''
------------------------------------------------------------------------
Parameters
------------------------------------------------------------------------
Model Parameters:
------------------------------------------------------------------------
S            = number of periods an individual lives (scalar)
J            = number of different ability groups (scalar)
T            = number of time periods until steady state is reached (scalar)
lambdas      = percentiles for ability groups (Jx1 array)
starting_age = age of first members of cohort (scalar)
ending age   = age of the last members of cohort (scalar)
E            = number of cohorts before S=1 (scalar)
beta_annual  = discount factor for one year (scalar)
beta         = discount factor for each age cohort (scalar)
sigma        = coefficient of relative risk aversion (scalar)
alpha        = capital share of income (scalar)
Z            = total factor productivity parameter in firms' production
               function (scalar)
delta_annual = depreciation rate of capital for one year (scalar)
delta        = depreciation rate of capital for each cohort (scalar)
ltilde       = measure of time each individual is endowed with each
               period (scalar)
g_y_annual   = annual growth rate of technology (scalar)
g_y          = growth rate of technology for one cohort (scalar)
b_ellipse    = value of b for elliptical fit of utility function (scalar)
k_ellipse    = value of k for elliptical fit of utility function (scalar)
upsilon      = value of omega for elliptical fit of utility function (scalar)
------------------------------------------------------------------------
Tax Parameters:
------------------------------------------------------------------------
mean_income_data  = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income = used to calibrate income tax (scalar)
b_tax_income = used to calibrate income tax (scalar)
c_tax_income = used to calibrate income tax (scalar)
d_tax_income = used to calibrate income tax (scalar)
h_wealth     = wealth tax parameter h (scalar)
m_wealth     = wealth tax parameter m (scalar)
p_wealth     = wealth tax parameter p (scalar)
tau_bq       = bequest tax (Jx1 array)
tau_payroll  = payroll tax (scalar)
retire       = age at which individuals retire (scalar)
------------------------------------------------------------------------
Simulation Parameters:
------------------------------------------------------------------------
MINIMIZER_TOL= Tolerance level for the minimizer in the calibration of chi's (scalar)
MINIMIZER_OPTIONS = dictionary for options to put into the minimizer, usually
                    to set a max iteration (dict)
PLOT_TPI     = Plot the path of K as TPI iterates (for debugging purposes) (bool)
maxiter      = Maximum number of iterations that SS and TPI will undergo (scalar)
mindist_SS   = Cut-off distance between iterations for SS (scalar)
mindist_TPI  = Cut-off distance between iterations for TPI (scalar)
nu           = contraction parameter in SS and TPI iteration process
               representing the weight on the new distribution (scalar)
flag_graphs  = Flag to prevent graphing from occuring in demographic, income,
               wealth, and labor files (True=graph) (bool)
chi_b_guess  = Chi^b_j initial guess for model (Jx1 array)
               (if no calibration occurs, these are the values that will be used for chi^b_j)
chi_n_guess  = Chi^n_s initial guess for model (Sx1 array)
               (if no calibration occurs, these are the values that will be used for chi^n_s)
------------------------------------------------------------------------
Demographics and Ability variables:
------------------------------------------------------------------------
omega        =  Time path of of population size for each age across T ((T+S)xS array)
g_n_ss       = steady state population growth rate (scalar)
omega_SS     = stationarized steady state population distribution (Sx1 array)
surv_rate    = survival rates (Sx1 array)
rho          = mortality rates (Sx1 array)
g_n_vector   = population size for each T ((T+S)x1 array)
e            = age dependent possible working abilities (SxJ array)
------------------------------------------------------------------------
'''

def get_full_parameters():
    '''
    --------------------------------------------------------------------
    Set exogenous model parameters including parameters determined
    outside the model from other processes
    --------------------------------------------------------------------

    --------------------------------------------------------------------
    '''
    # Model Parameters
    S = int(80)
    J = int(7)
    T = int(3 * S)
    lambdas = np.array([.25, .25, .2, .1, .1, .09, .01])
    starting_age = int(21)
    ending_age = int(100)
    E = int(round(float(S) * (float(starting_age) - 1) /
        (float(ending_age - starting_age) + 1)))
    beta_annual = .96
    beta = beta_annual ** (float(ending_age - starting_age + 1) / S)
    sigma = 3.0
    alpha = .35
    Z = 1.0
    delta_annual = .05
    delta = 1 - ((1 - delta_annual) **
            (float(ending_age - starting_age + 1) / S))
    ltilde = 1.0
    g_y_annual = 0.03
    g_y = (1 + g_y_annual) ** (float(ending_age - starting_age + 1) /
          S) - 1

    # Elliptical disutility of labor parameters
    b_ellipse = 25.6594
    k_ellipse = -26.4902
    upsilon = 3.0542

    # Set parameters to generate effective tax rate parameters
    beg_yr = int(2015)
    end_yr = int(2024)
    tpers = int(end_yr - beg_yr + 1)
    numparams = int(10)
    desc_data = False
    graph_data = False
    graph_est = False
    dmtrgr_est = False
    params_txfn = (starting_age, ending_age, beg_yr, end_yr, tpers,
        numparams, desc_data, graph_data, graph_est, dmtrgr_est)
    dict_tfparams = txfn.get_TaxFunParams(params_txfn)

    # Simulation Parameters
    MINIMIZER_TOL = 1e-14
    MINIMIZER_OPTIONS = None
    PLOT_TPI = True
    maxiter = 250
    mindist_SS = 1e-9
    mindist_TPI = 1e-6
    nu = .4
    flag_graphs = False
    #   Calibration parameters
    # These guesses are close to the calibrated values
    chi_b_guess = np.array([2, 10, 90, 350, 1700, 22000, 120000])
    chi_n_guess = np.array([47.12000874, 22.22762421, 14.34842241, 10.67954008, 8.41097278, 7.15059004, 6.46771332, 5.85495452, 5.46242013, 5.00364263, 4.57322063, 4.53371545, 4.29828515, 4.10144524, 3.8617942, 3.57282, 3.47473172, 3.31111347, 3.04137299, 2.92616951, 2.58517969, 2.48761429, 2.21744847, 1.9577682, 1.66931057, 1.6878927, 1.63107201, 1.63390543, 1.5901486, 1.58143606, 1.58005578, 1.59073213, 1.60190899, 1.60001831, 1.67763741, 1.70451784, 1.85430468, 1.97291208, 1.97017228,
                            2.25518398, 2.43969757, 3.21870602, 4.18334822, 4.97772026, 6.37663164, 8.65075992, 9.46944758, 10.51634777, 12.13353793, 11.89186997, 12.07083882, 13.2992811, 14.07987878, 14.19951571, 14.97943562, 16.05601334, 16.42979341, 16.91576867, 17.62775142, 18.4885405, 19.10609921, 20.03988031, 20.86564363, 21.73645892, 22.6208256, 23.37786072, 24.38166073, 25.22395387, 26.21419653, 27.05246704, 27.86896121, 28.90029708, 29.83586775, 30.87563699, 31.91207845, 33.07449767, 34.27919965, 35.57195873, 36.95045988, 38.62308152])
    # Generate Income and Demographic parameters
    omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_omega(
        S, T, starting_age, ending_age, E, flag_graphs)
    e = get_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)
    allvars = dict(locals())
    return allvars
