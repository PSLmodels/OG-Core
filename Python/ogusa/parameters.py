'''
------------------------------------------------------------------------
Last updated 4/8/2015

This file sets parameters for the model run.

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
import matplotlib.pyplot as plt


'''
------------------------------------------------------------------------
Set paths, define user modifiable parameters
------------------------------------------------------------------------
'''
DATASET = 'REAL'
PARAMS_FILE = os.path.join(os.path.dirname(__file__), 'default_full_parameters.json')
PARAMS_FILE_METADATA_NAME = 'parameters_metadata.json'
PARAMS_FILE_METADATA_PATH = os.path.join(os.path.dirname(__file__), PARAMS_FILE_METADATA_NAME)
TAX_ESTIMATE_PATH = os.environ.get("TAX_ESTIMATE_PATH", ".")
USER_MODIFIABLE_PARAMS = ['g_y_annual', 'frisch']


def read_parameter_metadata():
    '''
    --------------------------------------------------------------------
    This function reads in parameter metadata
    --------------------------------------------------------------------

    INPUTS: None

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    /PARAMS_FILE_METADATA_PATH/ = json file with metadata

    OBJECTS CREATED WITHIN FUNCTION:
    params_dict = dictionary of metadata

    RETURNS: params_dict
    --------------------------------------------------------------------
    '''
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
    '''
    --------------------------------------------------------------------
    This function reads in tax function parameters
    --------------------------------------------------------------------

    INPUTS:
    pickle_path = string, path to pickle with tax function parameter estimates
    pickle_file = string, name of pickle file with tax function parmaeter estimates

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    /picklepath/ = pickle file with dictionary of tax function estimated parameters

    OBJECTS CREATED WITHIN FUNCTION:
    dict_params = dictionary, contains numpy arrays of tax function estimates

    RETURNS: dict_params
    --------------------------------------------------------------------
    '''
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
    '''
    --------------------------------------------------------------------
    This function loads the json file with model parameters
    --------------------------------------------------------------------

    INPUTS: None

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    /PARAMS_FILE/ = json file with model parameters

    OBJECTS CREATED WITHIN FUNCTION:

    RETURNS: j
    --------------------------------------------------------------------
    '''
    with open(PARAMS_FILE,'r') as f:
        j = json.load(f)
        for key in j:
            if isinstance(j[key], list):
                j[key] = np.array(j[key])
        return j


def get_parameters(baseline=False, guid='', user_modifiable=False, metadata=False):
    '''
    --------------------------------------------------------------------
    This function returns the model parameters.
    --------------------------------------------------------------------

    INPUTS:
    baseline        = boolean, =True if run is of baseline policy
    guid            = string, id for model run
    user_modifiable = boolean, =True if allow user modifiable parameters
    metadata        = boolean, =True if use metadata file for parameter
                       values (rather than what is entered in parameters below)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    get_full_parameters()
    get_reduced_parameters()

    OBJECTS CREATED WITHIN FUNCTION:

    RETURNS: dictionary with model parameters
    --------------------------------------------------------------------
    '''
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
S            = integer, number of economically active periods an individual lives
J            = integer, number of different ability groups
T            = integer, number of time periods until steady state is reached
BW           = integer, number of time periods in the budget window
lambdas      = [J,] vector, percentiles for ability groups
starting_age = integer, age agents enter population
ending age   = integer, maximum age agents can live until
E            = integer, age agents become economically active
beta_annual  = scalar, discount factor as an annual rate
beta         = scalar, discount factor for model period
sigma        = scalar, coefficient of relative risk aversion
alpha        = scalar, capital share of income
Z            = scalar, total factor productivity parameter in firms' production
               function
delta_annual = scalar, depreciation rate as an annual rate
delta        = scalar, depreciation rate for model period
ltilde       = scalar, measure of time each individual is endowed with each
               period
g_y_annual   = scalar, annual growth rate of technology
g_y          = scalar, growth rate of technology for a model period
frisch       = scalar, Frisch elasticity that is used to fit ellipitcal utility
               to constant Frisch elasticity function
b_ellipse    = scalar, value of b for elliptical fit of utility function
k_ellipse    = scalar, value of k for elliptical fit of utility function
upsilon      = scalar, value of omega for elliptical fit of utility function
------------------------------------------------------------------------
Tax Parameters:
------------------------------------------------------------------------
mean_income_data = scalar, mean income from IRS data file used to calibrate income tax
etr_params       = [S,BW,#tax params] array, parameters for effective tax rate function
mtrx_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                    labor income function
mtry_params      = [S,BW,#tax params] array, parameters for marginal tax rate on
                    capital income function
h_wealth         = scalar, wealth tax parameter h (scalar)
m_wealth         = scalar, wealth tax parameter m (scalar)
p_wealth         = scalar, wealth tax parameter p (scalar)
tau_bq           = [J,] vector, bequest tax
tau_payroll      = scalar, payroll tax rate
retire           = integer, age at which individuals eligible for retirement benefits
------------------------------------------------------------------------
Simulation Parameters:
------------------------------------------------------------------------
MINIMIZER_TOL = scalar, tolerance level for the minimizer in the calibration of chi parameters
MINIMIZER_OPTIONS = dictionary, dictionary for options to put into the minimizer, usually
                    to set a max iteration
PLOT_TPI     = boolean, =Ture if plot the path of K as TPI iterates (for debugging purposes)
maxiter      = integer, maximum number of iterations that SS and TPI solution methods will undergo
mindist_SS   = scalar, tolerance for SS solution
mindist_TPI  = scalar, tolerance for TPI solution
nu           = scalar, contraction parameter in SS and TPI iteration process
               representing the weight on the new distribution
flag_graphs  = boolean, =True if produce graphs in demographic, income,
               wealth, and labor files (True=graph)
chi_b_guess  = [J,] vector, initial guess of \chi^{b}_{j} parameters
               (if no calibration occurs, these are the values that will be used for \chi^{b}_{j})
chi_n_guess  = [S,] vector, initial guess of \chi^{n}_{s} parameters
               (if no calibration occurs, these are the values that will be used for \chi^{n}_{s})
------------------------------------------------------------------------
Demographics and Ability variables:
------------------------------------------------------------------------
omega        = [T+S,S] array, time path of stationary distribution of economically active population by age
g_n_ss       = scalar, steady state population growth rate
omega_SS     = [S,] vector, stationary steady state population distribution
surv_rate    = [S,] vector, survival rates by age
rho          = [S,] vector, mortality rates by age
g_n_vector   = [T+S,] vector, growth rate in economically active pop for each period in transition path
e            = [S,J] array, normalized effective labor units by age and ability type
------------------------------------------------------------------------
'''


def get_reduced_parameters(baseline, guid, user_modifiable, metadata):
    '''
    --------------------------------------------------------------------
    This function sets the parameters for the reduced model, which is
    simplified to run more quickly for testing.
    --------------------------------------------------------------------

    INPUTS:
    baseline        = boolean, =True if baseline tax policy, =False if reform
    guid            = string, id for reform run
    user_modifiable = boolean, =True if allow user modifiable parameters
    metadata        = boolean, =True if use metadata file for parameter
                       values (rather than what is entered in parameters below)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    read_tax_func_estimate()
    elliptical_u_est.estimation()
    read_parameter_metadata()

    OBJECTS CREATED WITHIN FUNCTION:
    See parameters defined above
    allvars = dictionary, dictionary with all parameters defined in this function

    RETURNS: allvars

    OUTPUT: None
    --------------------------------------------------------------------
    '''
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
    omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_omega(
       S, T, starting_age, ending_age, E, flag_graphs)
    # omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_pop_objs(
    #     E, S, T, 0, 100, 2015, flag_graphs)
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
    '''
    --------------------------------------------------------------------
    This function sets the parameters for the full model.
    --------------------------------------------------------------------

    INPUTS:
    baseline        = boolean, =True if baseline tax policy, =False if reform
    guid            = string, id for reform run
    user_modifiable = boolean, =True if allow user modifiable parameters
    metadata        = boolean, =True if use metadata file for parameter
                       values (rather than what is entered in parameters below)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    read_tax_func_estimate()
    elliptical_u_est.estimation()
    read_parameter_metadata()

    OBJECTS CREATED WITHIN FUNCTION:
    See parameters defined above
    allvars = dictionary, dictionary with all parameters defined in this function

    RETURNS: allvars

    OUTPUT: None
    --------------------------------------------------------------------
    '''
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

    # etr_params = dict_params['tfunc_etr_params_S'][:S,:BW,:]
    # mtrx_params = dict_params['tfunc_mtrx_params_S'][:S,:BW,:]
    # mtry_params = dict_params['tfunc_mtry_params_S'][:S,:BW,:]

    # set etrs and mtrs to constant rates over income/age
    etr_params = np.zeros((S,BW,10))
    mtrx_params = np.zeros((S,BW,10))
    mtry_params = np.zeros((S,BW,10))
    etr_params[:,:,7] = dict_params['tfunc_avg_etr']
    mtrx_params[:,:,7] = dict_params['tfunc_avg_mtrx']
    mtry_params[:,:,7] = dict_params['tfunc_avg_mtry']
    etr_params[:,:,9] = dict_params['tfunc_avg_etr']
    mtrx_params[:,:,9] = dict_params['tfunc_avg_mtrx']
    mtry_params[:,:,9] = dict_params['tfunc_avg_mtry']
    etr_params[:,:,5] = 1.0
    mtrx_params[:,:,5] = 1.0
    mtry_params[:,:,5] = 1.0


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
    chi_b_guess = np.ones((J,)) * 80.0
    #chi_b_guess = np.array([0.7, 0.7, 1.0, 1.2, 1.2, 1.2, 1.4])
    #chi_b_guess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 10.0])
    #chi_b_guess = np.array([5, 10, 90, 250, 250, 250, 250])
    #chi_b_guess = np.array([2, 10, 90, 350, 1700, 22000, 120000])
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
    omega_hat, g_n_ss, omega_SS, surv_rate, rho, g_n_vector = get_omega(
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


    e_hetero = get_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)
    e = np.tile(((e_hetero*lambdas).sum(axis=1)).reshape(S,1),(1,J))
    e /= (e * omega_SS.reshape(S, 1)* lambdas.reshape(1, J)).sum()


    allvars = dict(locals())

    if user_modifiable:
        allvars = {k:allvars[k] for k in USER_MODIFIABLE_PARAMS}

    if metadata:
        params_meta = read_parameter_metadata()
        for k,v in allvars.iteritems():
            params_meta[k]["value"] = v
        allvars = params_meta

    return allvars
