import os
import sys
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_PATH, "../../"))
import pytest
import tempfile
import cPickle as pickle
import numpy as np
import os

'''
------------------------------------------------------------------------
Setup
------------------------------------------------------------------------
get_baseline = Flag to run baseline or tax experiments (bool)
calibrate_model = Flag to run calibration of chi values or not (bool)
------------------------------------------------------------------------
'''

TEST_OUTPUT = "./TESTOUT"

newfiles = [os.path.join(TEST_OUTPUT, "SSinit/ss_init_tpi_vars.pkl"),
            os.path.join(TEST_OUTPUT, "SSinit/ss_init_vars.pkl"),
            os.path.join(TEST_OUTPUT, "Saved_moments/SS_init_solutions.pkl"),
            os.path.join(TEST_OUTPUT, "Saved_moments/labor_data_moments.pkl"),
            os.path.join(TEST_OUTPUT, "Saved_moments/wealth_data_moments.pkl"),
            os.path.join(TEST_OUTPUT, "TPIinit/TPIinit_vars.pkl")]

oldfiles = [os.path.join(CUR_PATH, "../../../Python/OUTPUT/SSinit/ss_init_tpi_vars.pkl"),
            os.path.join(
                CUR_PATH, "../../../Python/OUTPUT/SSinit/ss_init_vars.pkl"),
            os.path.join(
                CUR_PATH, "../../../Python/OUTPUT/Saved_moments/SS_init_solutions.pkl"),
            os.path.join(
                CUR_PATH, "../../../Python/OUTPUT/Saved_moments/labor_data_moments.pkl"),
            os.path.join(
                CUR_PATH, "../../../Python/OUTPUT/Saved_moments/wealth_data_moments.pkl"),
            os.path.join(CUR_PATH, "../../../Python/OUTPUT/TPIinit/TPIinit_vars.pkl")]


@pytest.mark.full_run
def test_sstpi():
    import tempfile
    import pickle
    import numpy as np
    import numpy as np
    import cPickle as pickle
    import os

    import ogusa
    ogusa.parameters.DATASET = 'REAL'

    from ogusa.utils import comp_array
    from ogusa.utils import comp_scalar
    from ogusa.utils import dict_compare
    from ogusa.utils import pickle_file_compare

    import ogusa.SS
    import ogusa.TPI
    from ogusa import parameters, wealth, labor, demographics, income, SS, TPI

    globals().update(ogusa.parameters.get_parameters())

    # Generate Wealth data moments
    output_dir = TEST_OUTPUT
    input_dir = "./OUTPUT"
    wealth.get_wealth_data(lambdas, J, flag_graphs, output_dir)

    # Generate labor data moments
    labor.labor_data_moments(flag_graphs, output_dir=output_dir)

    get_baseline = True
    calibrate_model = False

    # List of parameter names that will not be changing (unless we decide to
    # change them for a tax experiment)
    param_names = ['S', 'J', 'T', 'lambdas', 'starting_age', 'ending_age',
                   'beta', 'sigma', 'alpha', 'nu', 'Z', 'delta', 'E',
                   'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
                   'b_ellipse', 'k_ellipse', 'upsilon',
                   'a_tax_income', 'chi_b_guess', 'chi_n_guess',
                   'b_tax_income', 'c_tax_income', 'd_tax_income',
                   'tau_payroll', 'tau_bq', 'calibrate_model',
                   'retire', 'mean_income_data', 'g_n_vector',
                   'h_wealth', 'p_wealth', 'm_wealth', 'get_baseline',
                   'omega', 'g_n_ss', 'omega_SS', 'surv_rate', 'e', 'rho']

    '''
    ------------------------------------------------------------------------
        Run SS with minimization to fit chi_b and chi_n
    ------------------------------------------------------------------------
    '''

    # This is the simulation before getting the replacement rate values
    sim_params = {}
    for key in param_names:
        try:
            sim_params[key] = locals()[key]
        except KeyError:
            sim_params[key] = globals()[key]

    sim_params['output_dir'] = output_dir
    sim_params['input_dir'] = input_dir
    income_tax_params, wealth_tax_params, ellipse_params, ss_parameters, \
        iterative_params = SS.create_steady_state_parameters(**sim_params)

    ss_outputs = SS.run_steady_state(ss_parameters, iterative_params,
                                     get_baseline, calibrate_model,
                                     output_dir=output_dir)

    '''
    ------------------------------------------------------------------------
        Run the baseline TPI simulation
    ------------------------------------------------------------------------
    '''

    ss_outputs['get_baseline'] = get_baseline
    income_tax_params, wealth_tax_params, ellipse_params, parameters, N_tilde, omega_stationary, K0, b_sinit, \
        b_splus1init, L0, Y0, w0, r0, BQ0, T_H_0, tax0, c0, initial_b, initial_n = TPI.create_tpi_params(
            **sim_params)
    ss_outputs['output_dir'] = output_dir
    ss_outputs['income_tax_params'] = income_tax_params
    ss_outputs['wealth_tax_params'] = wealth_tax_params
    ss_outputs['ellipse_params'] = ellipse_params
    ss_outputs['parameters'] = parameters
    ss_outputs['N_tilde'] = N_tilde
    ss_outputs['omega_stationary'] = omega_stationary
    ss_outputs['K0'] = K0
    ss_outputs['b_sinit'] = b_sinit
    ss_outputs['b_splus1init'] = b_splus1init
    ss_outputs['L0'] = L0
    ss_outputs['Y0'] = Y0
    ss_outputs['r0'] = r0
    ss_outputs['BQ0'] = BQ0
    ss_outputs['T_H_0'] = T_H_0
    ss_outputs['tax0'] = tax0
    ss_outputs['c0'] = c0
    ss_outputs['initial_b'] = initial_b
    ss_outputs['initial_n'] = initial_n
    ss_outputs['tau_bq'] = tau_bq
    ss_outputs['g_n_vector'] = g_n_vector
    TPI.run_time_path_iteration(**ss_outputs)

    # Platform specific exceptions:
    if sys.platform == "darwin":
        exceptions = {'tax_path': 0.08,
                      'c_path': 0.02,
                      'b_mat': 0.0017,
                      'solutions': 0.005}
    else:
        exceptions = {}

    # compare results to test data
    for old, new in zip(oldfiles, newfiles):
        print "trying a pair"
        print old, new
        assert pickle_file_compare(
            old, new, exceptions=exceptions, relative=True)
        print "next pair"
