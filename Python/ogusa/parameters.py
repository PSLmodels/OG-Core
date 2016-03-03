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
from demog import get_pop_objs
from income import get_e
import pickle
import txfunc
import elliptical_u_est


DATASET = 'REAL'
PARAMS_FILE = os.path.join(os.path.dirname(__file__), 'default_full_parameters.json')
PARAMS_FILE_METADATA_NAME = 'parameters_metadata.json'
PARAMS_FILE_METADATA_PATH = os.path.join(os.path.dirname(__file__), PARAMS_FILE_METADATA_NAME)

TAX_ESTIMATE_PATH = os.environ.get("TAX_ESTIMATE_PATH", ".")

USER_MODIFIABLE_PARAMS = ['g_y_annual', 'frisch']

def read_parameter_metadata():
    if os.path.exists(PARAMS_FILE_METADATA_PATH):
        with open(PARAMS_FILE_METADATA_PATH) as pfile:
            params_dict = json.load(pfile)
    else:
        from pkg_resources import resource_stream, Requirement
        path_in_egg = os.path.join('ogusa', PARAMS_FILE_METADATA_NAME)
        buf = resource_stream(Requirement.parse('ogusa'), path_in_egg)
        as_bytes = buf.read()
        as_string = as_bytes.decode("utf-8")
        params_dict = json.loads(as_string)

    return params_dict

def read_tax_func_estimate(pickle_path, pickle_file):
    if os.path.exists(pickle_path):
        with open(pickle_path) as pfile:
            dict_params = pickle.load(pfile)
    else:
        from pkg_resources import resource_stream, Requirement
        path_in_egg = pickle_file
        buf = resource_stream(Requirement.parse('ogusa'), path_in_egg)
        as_bytes = buf.read()
        as_string = as_bytes.decode("utf-8")
        dict_params = pickle.loads(as_string)

    return dict_params


def get_parameters_from_file():
    with open(PARAMS_FILE,'r') as f:
        j = json.load(f)
        for key in j:
            if isinstance(j[key], list):
                j[key] = np.array(j[key])
        return j

def get_parameters(baseline=False, guid='', user_modifiable=False, metadata=False):
    if DATASET == 'REAL':
        return get_full_parameters(baseline=baseline, guid=guid,
                                   user_modifiable=user_modifiable,
                                   metadata=metadata)

    elif DATASET == 'SMALL':
        return get_reduced_parameters(baseline=baseline, guid=guid,
                                      user_modifiable=user_modifiable,
                                      metadata=metadata)
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


def get_reduced_parameters(baseline, guid, user_modifiable, metadata):
    # Model Parameters
    starting_age = 40
    ending_age = 50
    S = int(ending_age-starting_age)
    J = int(2)
    T = int(2 * S)
    BW = int(10) 
    lambdas = np.array([.50, .50])
    E = int(starting_age * (S / float(ending_age - starting_age)))
    beta_annual = .96 # Carroll (JME, 2009)
    beta = beta_annual ** (float(ending_age - starting_age) / S)
    sigma = 1.5 # value from Attanasio, Banks, Meghir and Weber (JEBS, 1999)
    alpha = .35 # many use 0.33, but many find that capitals share is increasing (e.g. Elsby, Hobijn, and Sahin (BPEA, 2013))
    Z = 1.0
    delta_annual = .05 # approximately the value from Kehoe calibration exercise: http://www.econ.umn.edu/~tkehoe/classes/calibration-04.pdf
    delta = 1 - ((1 - delta_annual) ** (float(ending_age - starting_age) / S))
    ltilde = 1.0
    g_y_annual = 0.03
    g_y = (1 + g_y_annual)**(float(ending_age - starting_age) / S) - 1
    #   Ellipse parameters
    frisch = 0.4 # Frisch elasticity consistent with Altonji (JPE, 1996) and Peterman (Econ Inquiry, 2016)
    b_ellipse, upsilon = elliptical_u_est.estimation(frisch,ltilde)
    k_ellipse = 0 # this parameter is just a level shifter in utlitiy - irrelevant for analysis

    # Tax parameters:
    #   Income Tax Parameters
    ####  will call tax function estimation function here...
    ### do output such that each parameters is in a separate SxBW array

    if baseline:
        baseline_pckl = "TxFuncEst_baseline{}.pkl".format(guid)
        estimate_file = os.path.join(TAX_ESTIMATE_PATH,
                                     baseline_pckl)
        print 'using baseline1 tax parameters'
        dict_params = read_tax_func_estimate(estimate_file, baseline_pckl)
    else:
        policy_pckl = "TxFuncEst_policy{}.pkl".format(guid)
        estimate_file = os.path.join(TAX_ESTIMATE_PATH,
                                     policy_pckl)
        print 'using policy1 tax parameters'
        dict_params = read_tax_func_estimate(estimate_file, policy_pckl)


    mean_income_data = dict_params['tfunc_avginc'][0]

    etr_params = dict_params['tfunc_etr_params_S'][:S,:BW,:]
    mtrx_params = dict_params['tfunc_mtrx_params_S'][:S,:BW,:]
    mtry_params = dict_params['tfunc_mtry_params_S'][:S,:BW,:]

    # To zero out income taxes, uncomment the following 3 lines:
    # etr_params[:,:,6:] = 0.0
    # mtrx_params[:,:,6:] = 0.0
    # mtry_params[:,:,6:] = 0.0

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
    mindist_TPI = 1e-3 #1e-6
    nu = .4
    flag_graphs = False
    #   Calibration parameters
    # These guesses are close to the calibrated values
    chi_b_guess = np.array([1, 100000])
    chi_n_guess = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    # Generate Income and Demographic parameters
    #omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_omega(
    #    S, T, starting_age, ending_age, E, flag_graphs)
    omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_pop_objs(
        E, S, T, 0, 100, 2015, flag_graphs)
    e = np.array([[0.25, 1.25]] * 10)
    allvars = dict(locals())

    if user_modifiable:
        allvars = {k:allvars[k] for k in USER_MODIFIABLE_PARAMS}

    if metadata:
        params_meta = read_parameter_metadata()
        for k,v in allvars.iteritems():
            params_meta[k]["value"] = v
        allvars = params_meta

    return allvars


def get_full_parameters(baseline, guid, user_modifiable, metadata):
    # Model Parameters
    S = int(80)
    J = int(7)
    T = int(2 * S)
    BW = int(10) 
    lambdas = np.array([.25, .25, .2, .1, .1, .09, .01])
    starting_age = 20
    ending_age = 100
    E = int(starting_age * (S / float(ending_age - starting_age)))
    beta_annual = .96 # Carroll (JME, 2009)
    beta = beta_annual ** (float(ending_age - starting_age) / S)
    sigma = 1.5 # value from Attanasio, Banks, Meghir and Weber (JEBS, 1999)
    alpha = .35 # many use 0.33, but many find that capitals share is increasing (e.g. Elsby, Hobijn, and Sahin (BPEA, 2013))
    Z = 1.0
    delta_annual = .05 # approximately the value from Kehoe calibration exercise: http://www.econ.umn.edu/~tkehoe/classes/calibration-04.pdf
    delta = 1 - ((1 - delta_annual) ** (float(ending_age - starting_age) / S))
    ltilde = 1.0
    g_y_annual = 0.03
    g_y = (1 + g_y_annual)**(float(ending_age - starting_age) / S) - 1
    #   Ellipse parameters
    frisch = 0.4 # Frisch elasticity consistent with Altonji (JPE, 1996) and Peterman (Econ Inquiry, 2016)
    b_ellipse, upsilon = elliptical_u_est.estimation(frisch,ltilde)
    k_ellipse = 0 # this parameter is just a level shifter in utlitiy - irrelevant for analysis

    # Tax parameters:
    #   Income Tax Parameters
    #  will call tax function estimation function here...
    # do output such that each parameters is in a separate SxBW array
    # read in estimated parameters
    print 'baselines is:', baseline
    if baseline:
        baseline_pckl = "TxFuncEst_baseline{}.pkl".format(guid)
        estimate_file = os.path.join(TAX_ESTIMATE_PATH,
                                     baseline_pckl)
        print 'using baseline2 tax parameters'
        dict_params = read_tax_func_estimate(estimate_file, baseline_pckl)

    else:
        policy_pckl = "TxFuncEst_policy{}.pkl".format(guid)
        estimate_file = os.path.join(TAX_ESTIMATE_PATH,
                                     policy_pckl)
        print 'using policy2 tax parameters'
        dict_params = read_tax_func_estimate(estimate_file, policy_pckl)


    mean_income_data = dict_params['tfunc_avginc'][0]

    etr_params = dict_params['tfunc_etr_params_S'][:S,:BW,:]
    mtrx_params = dict_params['tfunc_mtrx_params_S'][:S,:BW,:]
    mtry_params = dict_params['tfunc_mtry_params_S'][:S,:BW,:]

    # To zero out income taxes, uncomment the following 3 lines:
    # etr_params[:,:,6:] = 0.0
    # mtrx_params[:,:,6:] = 0.0
    # mtry_params[:,:,6:] = 0.0


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
    mindist_TPI = 2e-5 
    nu = .4
    flag_graphs = False
    #   Calibration parameters
    # These guesses are close to the calibrated values
    #chi_b_guess = np.array([0.7, 0.7, 1.0, 1.2, 1.2, 1.2, 1.4])
    #chi_b_guess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    chi_b_guess = np.array([5, 10, 90, 250, 250, 250, 250])
    chi_n_guess = np.array([38.12000874, 33.22762421, 25.34842241, 26.67954008, 24.41097278, 
                            23.15059004, 22.46771332, 21.85495452, 21.46242013, 22.00364263, 
                            21.57322063, 21.53371545, 21.29828515, 21.10144524, 20.8617942, 
                            20.57282, 20.47473172, 20.31111347, 19.04137299, 18.92616951, 
                            20.58517969, 20.48761429, 20.21744847, 19.9577682, 19.66931057, 
                            19.6878927, 19.63107201, 19.63390543, 19.5901486, 19.58143606, 
                            19.58005578, 19.59073213, 19.60190899, 19.60001831, 21.67763741, 
                            21.70451784, 21.85430468, 21.97291208, 21.97017228, 22.25518398, 
                            22.43969757, 23.21870602, 24.18334822, 24.97772026, 26.37663164, 
                            29.65075992, 30.46944758, 31.51634777, 33.13353793, 32.89186997, 
                            38.07083882, 39.2992811, 40.07987878, 35.19951571, 35.97943562, 
                            37.05601334, 37.42979341, 37.91576867, 38.62775142, 39.4885405, 
                            37.10609921, 40.03988031, 40.86564363, 41.73645892, 42.6208256, 
                            43.37786072, 45.38166073, 46.22395387, 50.21419653, 51.05246704, 
                            53.86896121, 53.90029708, 61.83586775, 64.87563699, 66.91207845, 
                            68.07449767, 71.27919965, 73.57195873, 74.95045988, 76.62308152])


   # Generate Income and Demographic parameters
    omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_omega(
        S, T, starting_age, ending_age, E, flag_graphs)
    # omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_pop_objs(
    #     E, S, T, 0, 100, 2015, flag_graphs)
    # print 'Differences:'
    # print 'omega diffs: ', (np.absolute(omega-omega2)).max()
    # print 'g_n', g_n_ss, g_n_ss2
    # print 'omega SS diffs: ', (np.absolute(omega_SS-omega_SS2)).max()
    # print 'surv diffs: ', (np.absolute(surv_rate- surv_rate2)).max()
    # print 'mort diffs: ', (np.absolute(rho- rho2)).max()
    # print 'g_n_TP diffs: ', (np.absolute(g_n_vector- g_n_vector2)).max()
    # quit() 
    #print 'omega_SS shape: ', omega_SS.shape
    g_n_ss = 0.0   
    surv_rate1 = np.ones((S,))# prob start at age S
    surv_rate1[1:] = np.cumprod(surv_rate[:-1], dtype=float)
    omega_SS = np.ones(S)*surv_rate1# number of each age alive at any time
    omega_SS = omega_SS/omega_SS.sum()
    
    omega = np.tile(np.reshape(omega_SS,(1,S)),(T+S,1))
    g_n_vector = np.tile(g_n_ss,(T+S,))


    e = get_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)

    allvars = dict(locals())

    if user_modifiable:
        allvars = {k:allvars[k] for k in USER_MODIFIABLE_PARAMS}

    if metadata:
        params_meta = read_parameter_metadata()
        for k,v in allvars.iteritems():
            params_meta[k]["value"] = v
        allvars = params_meta

    return allvars
