'''
------------------------------------------------------------------------
This file sets parameters for the OG-USA model run.

This module calls the following other module(s):
    demographics.py
    income.py
    txfunc.py
    elliptical_u_est.py

This module defines the following function(s):
    read_parameter_metadata()
    read_tax_func_estimate()
    get_parameters_from_file()
    get_parameters()
    get_reduced_parameters()
    get_full_parameters()
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
import scipy.interpolate as si
import demographics as dem
import income as inc
import pickle
import txfunc
import elliptical_u_est as ellip
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
        print 'pickle path exists'
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
imm_rates    = [J,T+S] array, immigration rates by age and year
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
Small Open Economy Parameters:
------------------------------------------------------------------------

ss_firm_r   = scalar, world interest rate available to firms in the steady state
ss_hh_r     = scalar, world interest rate available to households in the steady state
tpi_firm_r  = [T+S,] vector, world interest rate (firm). Must be ss_firm_r in last period.
tpi_hh_r    = [T+S,] vector, world interest rate (household). Must be ss_firm_r in last period.

------------------------------------------------------------------------
Fiscal imbalance Parameters:
------------------------------------------------------------------------

alpha_T          = scalar, share of GDP that goes to transfers.
alpha_G          = scalar, share of GDP that goes to gov't spending in early years.
tG1             = scalar < t_G2, period at which change government spending rule from alpha_G*Y to glide toward SS debt ratio
tG2             = scalar < T, period at which change gov't spending rule with final discrete jump to achieve SS debt ratio
debt_ratio_ss    = scalar, steady state debt/GDP.

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
chi_n_guess_80 = (80,) vector, initial guess of chi_{n,s} parameters for
                 80 one-year-period ages from 21 to 100
chi_n_guess    = (S,) vector, interpolated initial guess of chi^{n,s}
                 parameters (if no calibration occurs, these are the
                 values that will be used
age_midp_80    = (80,) vector, midpoints of age bins for 80 one-year-
                 period ages from 21 to 100 for interpolation
chi_n_interp   = function, interpolation function for chi_n_guess
newstep        = scalar > 1, duration in years of each life period
age_midp_S     = (S,) vector, midpoints of age bins for S one-year-
                 period ages from 21 to 100 for interpolation
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
    ellip.estimation()
    read_parameter_metadata()

    OBJECTS CREATED WITHIN FUNCTION:
    See parameters defined above
    allvars = dictionary, dictionary with all parameters defined in this function

    RETURNS: allvars

    OUTPUT: None
    --------------------------------------------------------------------
    '''
    # Model Parameters
    start_year = 2016
    starting_age = 40
    ending_age = 50
    S = int(ending_age-starting_age)
    lambdas = np.array([.50, .50])
    J = lambdas.shape[0]
    T = int(2 * S)
    BW = int(10)

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
    b_ellipse, upsilon = ellip.estimation(frisch,ltilde)
    k_ellipse = 0 # this parameter is just a level shifter in utlitiy - irrelevant for analysis
    
    # Small Open Economy parameters. Currently these are placeholders. Can introduce a 
    # borrow/lend spread and a time path from t=0 to t=T-1. However, from periods T through 
    # T+S, the steady state rate should hold.
    ss_firm_r   = 0.04
    ss_hh_r     = 0.04
    tpi_firm_r  = np.ones(T+S)*ss_firm_r
    tpi_hh_r    = np.ones(T+S)*ss_hh_r

    # Fiscal imbalance parameters. These allow government deficits, debt, and savings.
    alpha_T            = 0.12  # share of GDP that goes to transfers each period.
    alpha_G            = 0.05  # share of GDP of government spending for periods t<t_G1
    tG1                = int(T/4)  # change government spending rule from alpha_G*Y to glide toward SS debt ratio
    tG2                = int(T*0.75)  # change gov't spending rule with final discrete jump to achieve SS debt ratio
    rho_G              = 0.1  # 0 < rho_G < 1 is transition speed for periods [tG1, tG2-1]. Lower rho_G => slower convergence.
    debt_ratio_ss      = 0.4  # assumed steady-state debt/GDP ratio. Savings would be a negative number.
    initial_debt       = 0.2  # first-period debt/GDP ratio. Savings would be a negative number.
    
    if tG1 > tG2:
        print 'The first government spending rule change date, (', tG1, ') is after the second one (', tG2, ').'
        err = "Gov't spending rule dates are inconsistent"
        raise RuntimeError(err)
    if tG2 > T:
        print 'The second government spending rule change date, (', tG2, ') is after time T (', T, ').'
        err = "Gov't spending rule dates are inconsistent"
        raise RuntimeError(err)

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
    (omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector, imm_rates,
        omega_S_preTP) = dem.get_pop_objs(E, S, T, 1, 100, start_year,
        flag_graphs)
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
    ellip.estimation()
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
    lambdas = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
    J = lambdas.shape[0]
    T = int(4 * S)
    BW = int(10)

    start_year = 2016
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
    b_ellipse, upsilon = ellip.estimation(frisch,ltilde)
    k_ellipse = 0 # this parameter is just a level shifter in utlitiy - irrelevant for analysis
    
    # Small Open Economy parameters. Currently these are placeholders. Can introduce a 
    # borrow/lend spread and a time path from t=0 to t=T-1. However, from periods T through 
    # T+S, the steady state rate should hold.
    ss_firm_r_annual   =  0.04
    ss_hh_r_annual     =  0.04
    ss_firm_r          = (1 + ss_firm_r_annual) ** (float(ending_age - starting_age) / S) - 1
    ss_hh_r            = (1 + ss_hh_r_annual)   ** (float(ending_age - starting_age) / S) - 1
    tpi_firm_r         = np.ones(T+S)*ss_firm_r
    tpi_hh_r           = np.ones(T+S)*ss_hh_r

    # Fiscal imbalance parameters. These allow government deficits, debt, and savings.
    alpha_T            = 0.15  # share of GDP that goes to transfers each period.
    alpha_G            = 0.05  # share of GDP of government spending for periods t<tG1
    tG1                = int(T/4)  # change government spending rule from alpha_G*Y to glide toward SS debt ratio
    tG2                = int(T*0.75)  # change gov't spending rule with final discrete jump to achieve SS debt ratio
    rho_G              = 0.1  # 0 < rho_G < 1 is transition speed for periods [tG1, tG2-1]. Lower rho_G => slower convergence.
    debt_ratio_ss      = 0.6  # assumed steady-state debt/GDP ratio. Savings would be a negative number.
    initial_debt       = 0.6  # first-period debt/GDP ratio. Savings would be a negative number.
    
    if tG1 > tG2:
        print 'The first government spending rule change date, (', tG1, ') is after the second one (', tG2, ').'
        err = "Gov't spending rule dates are inconsistent"
        raise RuntimeError(err)
    if tG2 > T:
        print 'The second government spending rule change date, (', tG2, ') is after time T (', T, ').'
        err = "Gov't spending rule dates are inconsistent"
        raise RuntimeError(err)

    

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
        print 'using baseline tax parameters'
        dict_params = read_tax_func_estimate(estimate_file, baseline_pckl)

    else:
        policy_pckl = "TxFuncEst_policy{}.pkl".format(guid)
        estimate_file = os.path.join(TAX_ESTIMATE_PATH,
                                     policy_pckl)
        print 'using policy tax parameters'
        dict_params = read_tax_func_estimate(estimate_file, policy_pckl)


    mean_income_data = dict_params['tfunc_avginc'][0]

    etr_params = dict_params['tfunc_etr_params_S'][:S,:BW,:]
    mtrx_params = dict_params['tfunc_mtrx_params_S'][:S,:BW,:]
    mtry_params = dict_params['tfunc_mtry_params_S'][:S,:BW,:]

    # # Make all ETRs equal the average
    # etr_params = np.zeros(etr_params.shape)
    # etr_params[:, :, 10] = dict_params['tfunc_avg_etr'] # set shift to average rate

    # # Make all MTRx equal the average
    # mtrx_params = np.zeros(mtrx_params.shape)
    # mtrx_params[:, :, 10] = dict_params['tfunc_avg_mtrx'] # set shift to average rate

    # # Make all MTRy equal the average
    # mtry_params = np.zeros(mtry_params.shape)
    # mtry_params[:, :, 10] = dict_params['tfunc_avg_mtry'] # set shift to average rate

    # # Make MTRx depend only on labor income
    # mtrx_params[:, :, 11] = 1.0 # set share parameter to 1

    # # Make MTRy depend only on capital income
    # mtry_params[:, :, 11] = 0.0 # set share parameter to 0

    # # set all MTRx parameters equal to the 43-yr-old values from 2016
    # mtrx_params = np.tile(mtrx_params[11, 0, :], (S, 10, 1))

    #
    #
    #
    #
    # etr_params[:,:,:] = dict_params['tfunc_etr_params_S'][20,0,:]
    # mtrx_params[:,:,:] = dict_params['tfunc_mtrx_params_S'][20,0,:]
    # mtry_params[:,:,:] = dict_params['tfunc_mtry_params_S'][20,0,:]
    #
    #
    # mtrx_params[:,:,1] = 0.
    # mtrx_params[:,:,2] = 0.
    # mtrx_params[:,:,4] = 0.
    # mtrx_params[:,:,8] = 0.
    # mtrx_params[:,:,9] = 0.
    #
    # mtry_params[:,:,0] = 0.
    # mtry_params[:,:,2] = 0.
    # mtry_params[:,:,3] = 0.
    # mtry_params[:,:,6] = 0.
    # mtry_params[:,:,7] = 0.

    # # unocmmenting the block below ensures no tax rates are negative
    # etr_params[:,:,7] = 0.
    # mtrx_params[:,:,7] = 0.
    # mtry_params[:,:,7] = 0.
    # etr_params[:,:,9] = 0.
    # mtrx_params[:,:,9] = 0.
    # mtry_params[:,:,9] = 0.

    # print 'tax diffs: ', np.absolute(etr_params-etr_params.mean(axis=(0,1))).max()
    # quit()

    # set etrs and mtrs to constant rates over income/age by uncommenting following code block
    # etr_params = np.zeros((S,BW,10))
    # mtrx_params = np.zeros((S,BW,10))
    # mtry_params = np.zeros((S,BW,10))

    # etr_params = np.zeros((S,BW,10))
    # mtrx_params = np.zeros((S,BW,10))
    # mtry_params = np.zeros((S,BW,10))
    # for i in xrange(S):
    #     etr_params[:,:,7] = 0.005*i
    #     mtrx_params[:,:,7] = 0.005*i
    #     mtry_params[:,:,7] = 0.005*i
    #     etr_params[:,:,9] = 0.005*i
    #     mtrx_params[:,:,9] = 0.005*i
    #     mtry_params[:,:,9] = 0.005*i
    # etr_params[:,:,5] = 1.0
    # mtrx_params[:,:,5] = 1.0
    # mtry_params[:,:,5] = 1.0


    # make etrs and mtrs constant over time, uncomment following code block
    # etr_params[:,:,7] = dict_params['tfunc_avg_etr'][0]
    # mtrx_params[:,:,7] = dict_params['tfunc_avg_mtrx'][0]
    # mtry_params[:,:,7] = dict_params['tfunc_avg_mtry'][0]
    # etr_params[:,:,9] = dict_params['tfunc_avg_etr'][0]
    # mtrx_params[:,:,9] = dict_params['tfunc_avg_mtrx'][0]
    # mtry_params[:,:,9] = dict_params['tfunc_avg_mtry'][0]

    # To zero out income taxes, uncomment the following 3 lines:
    # etr_params[:,:,6:] = 0.0
    # mtrx_params[:,:,6:] = 0.0
    # mtry_params[:,:,6:] = 0.0


    # #plot some tax functions
    # cap_income = np.array([0.0,1000., 10000., 100000., 1000000.]) # fix capital income
    # N = 1000 # number of points in income grids
    # labinc_sup = np.linspace(5, 1000000, N)
    # capinc_sup = np.linspace(5, 1000000, N)

    # A = mtrx_params[0,0,0]
    # B = mtrx_params[0,0,1]
    # C = mtrx_params[0,0,2]
    # D = mtrx_params[0,0,3]
    # E = mtrx_params[0,0,4]
    # F = mtrx_params[0,0,5]
    # max_x = mtrx_params[0,0,6]
    # min_x = mtrx_params[0,0,7]
    # max_y = mtrx_params[0,0,8]
    # min_y = mtrx_params[0,0,9]

    # analytical_mtrs=True
    # marginal_rates = np.zeros((5,N))

    # for i in xrange((5)):
    #     y = cap_income[i] # fix capital income
    #     x = labinc_sup # labor income varies
    #     I = x+y

    #     if analytical_mtrs:
    #         num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
    #         denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F
    #         Lambda = num/denom

    #         d_num = (2*A*x + C*y + D)*F
    #         d_denom = ((A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F)**2
    #         d_Lambda = d_num/d_denom

    #         marginal_rates[i,:] =  (max_x-min_x)*Lambda + (x*(max_x-min_x) + y*(max_y-min_y))*d_Lambda + min_x

    #     else:

    #         phi = x/I
    #         Phi = phi*(max_x-min_x) + (1-phi)*(max_y-min_y)
    #         K = phi*min_x + (1-phi)*min_y

    #         num = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y)
    #         denom = (A*(x**2)) + (B*(y**2)) + (C*x*y) + (D*x) + (E*y) + F

    #         marginal_rates[i,:]  =  (Phi*(num/denom)) + K


    # plt.plot(labinc_sup, marginal_rates[0,:], label='cap inc=0')
    # plt.plot(labinc_sup, marginal_rates[1,:], label='cap inc=1,000')
    # plt.plot(labinc_sup, marginal_rates[2,:], label='cap inc=10,000')
    # plt.plot(labinc_sup, marginal_rates[3,:], label='cap inc=100,000')
    # plt.plot(labinc_sup, marginal_rates[4,:], label='cap inc=1,000,000')
    # plt.legend(loc='center right')
    # plt.title('MTRx by labor income')
    # plt.xlabel(r'Labor Income')
    # plt.ylabel(r'MTR')
    # plt.show()
    # quit()



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
    retire = np.int(np.round(9.0 * S / 16.0) - 1)

    # Simulation Parameters
    MINIMIZER_TOL = 1e-14
    MINIMIZER_OPTIONS = None
    PLOT_TPI = False
    maxiter = 250
    mindist_SS = 1e-9
    mindist_TPI = 1e-9 #2e-5 
    nu = .4
    flag_graphs = False
    #   Calibration parameters
    # These guesses are close to the calibrated values
    chi_b_guess = np.ones((J,)) * 80.0
    #chi_b_guess = np.array([0.7, 0.7, 1.0, 1.2, 1.2, 1.2, 1.4])
    #chi_b_guess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 10.0])
    #chi_b_guess = np.array([5, 10, 90, 250, 250, 250, 250])
    #chi_b_guess = np.array([2, 10, 90, 350, 1700, 22000, 120000])
    chi_n_guess_80 = np.array(
        [38.12000874, 33.22762421, 25.3484224, 26.67954008, 24.41097278,
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
        68.07449767, 71.27919965, 73.57195873, 74.95045988, 76.6230815])

    # Generate Income and Demographic parameters
    (omega, g_n_ss, omega_SS, surv_rate, rho, g_n_vector, imm_rates,
        omega_S_preTP) = dem.get_pop_objs(E, S, T, 1, 100, start_year,
        flag_graphs)
    # Interpolate chi_n_guesses and create omega_SS_80 if necessary
    if S == 80:
        chi_n_guess = chi_n_guess_80.copy()
        omega_SS_80 = omega_SS.copy()
    elif S < 80:
        age_midp_80 = np.linspace(20.5, 99.5, 80)
        chi_n_interp = si.interp1d(age_midp_80, chi_n_guess_80,
                       kind='cubic')
        newstep = 80.0 / S
        age_midp_S = np.linspace(20 + 0.5 * newstep,
                     100 - 0.5 * newstep, S)
        chi_n_guess = chi_n_interp(age_midp_S)
        (_, _, omega_SS_80, _, _, _, _,_) = dem.get_pop_objs(20, 80,
            320, 1, 100, start_year, False)





    ## To shut off demographics, uncomment the following 9 lines of code
    # g_n_ss = 0.0
    # surv_rate1 = np.ones((S,))# prob start at age S
    # surv_rate1[1:] = np.cumprod(surv_rate[:-1], dtype=float)
    # omega_SS = np.ones(S)*surv_rate1# number of each age alive at any time
    # omega_SS = omega_SS/omega_SS.sum()
    # imm_rates = np.zeros((T+S,S))
    # omega = np.tile(np.reshape(omega_SS,(1,S)),(T+S,1))
    # omega_S_preTP = omega_SS
    # g_n_vector = np.tile(g_n_ss,(T+S,))

    e = inc.get_e_interp(S, omega_SS, omega_SS_80, lambdas, plot=False)
    # e_hetero = get_e(S, J, starting_age, ending_age, lambdas, omega_SS, flag_graphs)
    # e = np.tile(((e_hetero*lambdas).sum(axis=1)).reshape(S,1),(1,J))
    # e /= (e * omega_SS.reshape(S, 1)* lambdas.reshape(1, J)).sum()


    allvars = dict(locals())

    if user_modifiable:
        allvars = {k:allvars[k] for k in USER_MODIFIABLE_PARAMS}

    if metadata:
        params_meta = read_parameter_metadata()
        for k,v in allvars.iteritems():
            params_meta[k]["value"] = v
        allvars = params_meta

    return allvars
