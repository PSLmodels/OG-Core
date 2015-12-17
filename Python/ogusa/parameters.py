'''
------------------------------------------------------------------------
Last updated 7/21/2015

This file generates demographic and ability variables.

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
import os
import json
import numpy as np
from demographics import get_omega
from income import get_e
import pickle

DATASET = 'REAL'
PARAMS_FILE = os.path.join(os.path.dirname(__file__), 'default_full_parameters.json')

TAX_ESTIMATE_PATH = os.environ.get("TAX_ESTIMATE_PATH", ".")


def get_parameters_from_file():
    with open(PARAMS_FILE,'r') as f:
        j = json.load(f)
        for key in j:
            if isinstance(j[key], list):
                j[key] = np.array(j[key])
        return j

def get_parameters(baseline=False):
    if DATASET == 'REAL':
        return get_full_parameters(baseline=baseline)
    elif DATASET == 'SMALL':
        return get_reduced_parameters(baseline=baseline)
    else:
        raise ValueError("Unknown value {0}".format(DATASET))


'''
------------------------------------------------------------------------
Parameters
------------------------------------------------------------------------
Model Parameters:
------------------------------------------------------------------------
S            = number of periods an individual lives (scalar)
J            = number of different ability groups (scalar)
T            = number of time periods until steady state is reached (scalar)
BW           = number of time periods in the budget window (scalar)
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
mean_income_data = mean income from IRS data file used to calibrate income tax
               (scalar)
a_tax_income     = used to calibrate income tax (SxBW array)
b_tax_income     = used to calibrate income tax (SxBW array)
c_tax_income     = used to calibrate income tax (SxBW array)
d_tax_income     = used to calibrate income tax (SxBW array)
e_tax_income     = used to calibrate income tax (SxBW array)
f_tax_income     = used to calibrate income tax (SxBW array)
min_x_tax_income = used to calibrate income tax (SxBW array)
max_x_tax_income = used to calibrate income tax (SxBW array)
min_y_tax_income = used to calibrate income tax (SxBW array)
max_y_tax_income = used to calibrate income tax (SxBW array)
h_wealth         = wealth tax parameter h (scalar)
m_wealth         = wealth tax parameter m (scalar)
p_wealth         = wealth tax parameter p (scalar)
tau_bq           = bequest tax (Jx1 array)
tau_payroll      = payroll tax (scalar)
retire           = age at which individuals retire (scalar)
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


def get_reduced_parameters(baseline):
    # Model Parameters
    starting_age = 40
    ending_age = 50
    S = int(ending_age-starting_age)
    J = int(2)
    T = int(2 * S)
    BW = int(10) 
    lambdas = np.array([.50, .50])
    E = int(starting_age * (S / float(ending_age - starting_age)))
    beta_annual = .96
    beta = beta_annual ** (float(ending_age - starting_age) / S)
    sigma = 3.0
    alpha = .35
    Z = 1.0
    delta_annual = .05
    delta = 1 - ((1 - delta_annual) ** (float(ending_age - starting_age) / S))
    ltilde = 1.0
    g_y_annual = 0.03
    g_y = (1 + g_y_annual)**(float(ending_age - starting_age) / S) - 1
    #   Ellipse parameters
    b_ellipse = 25.6594
    k_ellipse = -26.4902
    upsilon = 3.0542

    # Tax parameters:
    #   Income Tax Parameters
    ####  will call tax function estimation function here...
    ### do output such that each parameters is in a separate SxBW array
    if baseline:
        estimate_file = os.path.join(TAX_ESTIMATE_PATH,
                                     "TxFuncEst_baseline_w_mtrs2.pkl")
    else:
        estimate_file = os.path.join(TAX_ESTIMATE_PATH,
                                     "TxFuncEst_policy.pkl")

    dict_params = pickle.load( open( estimate_file, "rb" ) )

    # print 'etr mins: ', dict_params['tfunc_etr_params_S'].min(axis=(0,1))
    # print 'etr maxes: ', dict_params['tfunc_etr_params_S'].max(axis=(0,1))
    # print 'mtrx mins: ', dict_params['tfunc_mtrx_params_S'].min(axis=(0,1))
    # print 'mtrx maxes: ', dict_params['tfunc_mtrx_params_S'].max(axis=(0,1))
    # print 'mtry mins: ', dict_params['tfunc_mtry_params_S'].min(axis=(0,1))
    # print 'mtry maxes: ', dict_params['tfunc_mtry_params_S'].max(axis=(0,1))
    # quit()

    # print 'age 61 ', dict_params['tfunc_etr_params_S'][42,9,:]
    # print 'age 62 ', dict_params['tfunc_etr_params_S'][43,9,:]
    # print 'age 63 ', dict_params['tfunc_etr_params_S'][44,9,:]
    # print 'age 64 ', dict_params['tfunc_etr_params_S'][45,9,:]
    # print 'age 65 ', dict_params['tfunc_etr_params_S'][46,9,:]
    # print 'age 66 ', dict_params['tfunc_etr_params_S'][47,9,:]
    # print 'age 67 ', dict_params['tfunc_etr_params_S'][48,9,:]
    # quit()

    mean_income_data = dict_params['tfunc_avginc'][0]
    a_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,0]
    b_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,1]
    c_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,2]
    d_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,3]
    e_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,4]
    f_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,5]
    max_x_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,6]
    min_x_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,7]
    max_y_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,8]
    min_y_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,9]

    a_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,0]
    b_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,1]
    c_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,2]
    d_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,3]
    e_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,4]
    f_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,5]
    max_x_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,6]
    min_x_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,7]
    max_y_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,8]
    min_y_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,9]

    a_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,0]
    b_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,1]
    c_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,2]
    d_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,3]
    e_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,4]
    f_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,5]
    max_x_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,6]
    min_x_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,7]
    max_y_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,8]
    min_y_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,9]


    # # zero out income taxes:
    # max_x_etr_income = np.ones((S,BW))*0.0
    # min_x_etr_income = np.ones((S,BW))*0.0
    # max_y_etr_income = np.ones((S,BW))*0.0
    # min_y_etr_income = np.ones((S,BW))*0.0

    # max_x_mtrx_income = np.ones((S,BW))*0.0
    # min_x_mtrx_income = np.ones((S,BW))*0.0
    # max_y_mtrx_income = np.ones((S,BW))*0.0
    # min_y_mtrx_income = np.ones((S,BW))*0.0

    # max_x_mtry_income = np.ones((S,BW))*0.0
    # min_x_mtry_income = np.ones((S,BW))*0.0
    # max_y_mtry_income = np.ones((S,BW))*0.0
    # min_y_mtry_income = np.ones((S,BW))*0.0


    #   Wealth tax params
    #       These are non-calibrated values, h and m just need
    #       need to be nonzero to avoid errors. When p_wealth
    #       is zero, there is no wealth tax.
    h_wealth = 0.1
    m_wealth = 1.0
    p_wealth = 0.0
    #   Bequest and Payroll Taxes
    tau_bq = np.zeros(J)
    tau_payroll = 0.15
    retire = np.round(9.0 * S / 16.0) - 1

    # Simulation Parameters
    MINIMIZER_TOL = 1e-3
    MINIMIZER_OPTIONS = {'maxiter': 1}
    PLOT_TPI = False
    maxiter = 10
    mindist_SS = 1e-3
    mindist_TPI = 1e-6
    nu = .4
    flag_graphs = False
    #   Calibration parameters
    # These guesses are close to the calibrated values
    chi_b_guess = np.array([1, 100000])
    chi_n_guess = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    # Generate Income and Demographic parameters
    omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_omega(
        S, T, starting_age, ending_age, E, flag_graphs)
    e = np.array([[0.25, 1.25]] * 10)
    allvars = dict(locals())
    return allvars


def get_full_parameters(baseline):
    # Model Parameters
    S = int(80)
    J = int(7)
    T = int(2 * S)
    BW = int(10) 
    lambdas = np.array([.25, .25, .2, .1, .1, .09, .01])
    starting_age = 20
    ending_age = 100
    E = int(starting_age * (S / float(ending_age - starting_age)))
    beta_annual = .96
    beta = beta_annual ** (float(ending_age - starting_age) / S)
    sigma = 3.0
    alpha = .35
    Z = 1.0
    delta_annual = .05
    delta = 1 - ((1 - delta_annual) ** (float(ending_age - starting_age) / S))
    ltilde = 1.0
    g_y_annual = 0.03
    g_y = (1 + g_y_annual)**(float(ending_age - starting_age) / S) - 1
    #   Ellipse parameters
    b_ellipse = 25.6594
    k_ellipse = -26.4902
    upsilon = 3.0542

    # Tax parameters:
    #   Income Tax Parameters
    #  will call tax function estimation function here...
    # do output such that each parameters is in a separate SxBW array
    # read in estimated parameters
    if baseline:
        dict_params = pickle.load( open( "TxFuncEst_baseline_w_mtrs2.pkl", "rb" ) )
    else:
        dict_params = pickle.load( open( "TxFuncEst_policy.pkl", "rb" ) )


    # print 'etr mins: ', dict_params['tfunc_etr_params_S'].min(axis=(0,1))
    # print 'etr maxes: ', dict_params['tfunc_etr_params_S'].max(axis=(0,1))
    # print 'mtrx mins: ', dict_params['tfunc_mtrx_params_S'].min(axis=(0,1))
    # print 'mtrx maxes: ', dict_params['tfunc_mtrx_params_S'].max(axis=(0,1))
    # print 'mtry mins: ', dict_params['tfunc_mtry_params_S'].min(axis=(0,1))
    # print 'mtry maxes: ', dict_params['tfunc_mtry_params_S'].max(axis=(0,1))
    # quit()

    # print 'age 61 ', dict_params['tfunc_etr_params_S'][42,9,:]
    # print 'age 62 ', dict_params['tfunc_etr_params_S'][43,9,:]
    # print 'age 63 ', dict_params['tfunc_etr_params_S'][44,9,:]
    # print 'age 64 ', dict_params['tfunc_etr_params_S'][45,9,:]
    # print 'age 65 ', dict_params['tfunc_etr_params_S'][46,9,:]
    # print 'age 66 ', dict_params['tfunc_etr_params_S'][47,9,:]
    # print 'age 67 ', dict_params['tfunc_etr_params_S'][48,9,:]
    # quit()

    mean_income_data = dict_params['tfunc_avginc'][0]
    a_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,0]
    b_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,1]
    c_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,2]
    d_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,3]
    e_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,4]
    f_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,5]
    max_x_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,6]
    min_x_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,7]
    max_y_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,8]
    min_y_etr_income = dict_params['tfunc_etr_params_S'][:S,:BW,9]

    a_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,0]
    b_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,1]
    c_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,2]
    d_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,3]
    e_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,4]
    f_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,5]
    max_x_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,6]
    min_x_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,7]
    max_y_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,8]
    min_y_mtrx_income = dict_params['tfunc_mtrx_params_S'][:S,:BW,9]

    a_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,0]
    b_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,1]
    c_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,2]
    d_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,3]
    e_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,4]
    f_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,5]
    max_x_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,6]
    min_x_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,7]
    max_y_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,8]
    min_y_mtry_income = dict_params['tfunc_mtry_params_S'][:S,:BW,9]


    # # zero out income taxes:
    # max_x_etr_income = np.ones((S,BW))*0.0
    # min_x_etr_income = np.ones((S,BW))*0.0
    # max_y_etr_income = np.ones((S,BW))*0.0
    # min_y_etr_income = np.ones((S,BW))*0.0

    # max_x_mtrx_income = np.ones((S,BW))*0.0
    # min_x_mtrx_income = np.ones((S,BW))*0.0
    # max_y_mtrx_income = np.ones((S,BW))*0.0
    # min_y_mtrx_income = np.ones((S,BW))*0.0

    # max_x_mtry_income = np.ones((S,BW))*0.0
    # min_x_mtry_income = np.ones((S,BW))*0.0
    # max_y_mtry_income = np.ones((S,BW))*0.0
    # min_y_mtry_income = np.ones((S,BW))*0.0


    #   Wealth tax params
    #       These are non-calibrated values, h and m just need
    #       need to be nonzero to avoid errors. When p_wealth
    #       is zero, there is no wealth tax.
    h_wealth = 0.1
    m_wealth = 1.0
    p_wealth = 0.0
    #   Bequest and Payroll Taxes
    tau_bq = np.zeros(J)
    tau_payroll = 0.0 #0.15 # were are inluding payroll taxes in tax functions for now
    retire = np.round(9.0 * S / 16.0) - 1

    # Simulation Parameters
    MINIMIZER_TOL = 1e-14
    MINIMIZER_OPTIONS = None
    PLOT_TPI = False
    maxiter = 250
    mindist_SS = 1e-9
    mindist_TPI = 2e-5 #1e-6
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
