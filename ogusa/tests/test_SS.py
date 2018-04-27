from __future__ import print_function

import pytest
import json
import pickle
import numpy as np
import os
import multiprocessing
from multiprocessing import Process
from dask.distributed import Client
from ogusa import SS, utils

# Define parameters to use for multiprocessing
# client = Client(processes=False)
# # num_workers = int(os.cpu_count())  # not in os on Python 2.7?
# num_workers = multiprocessing.cpu_count()

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_SS_fsolve():
    # Test SS.SS_fsolve function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_inputs.pkl'))
    guesses, params = input_tuple
    params = params + (None, 1)
    (bssmat, nssmat, chi_params, ss_params, income_tax_params,
     iterative_params, small_open_params, client, num_workers) = params
    income_tax_params = ('DEP',) + income_tax_params
    params = (bssmat, nssmat, chi_params, ss_params, income_tax_params,
              iterative_params, small_open_params, client, num_workers)
    test_list = SS.SS_fsolve(guesses, params)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_outputs.pkl'))
    print('outputs = ', np.absolute(np.array(test_list) -
                                    np.array(expected_list)).max())
    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_fsolve_reform():
    # Test SS.SS_fsolve_reform function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_inputs.pkl'))
    guesses, params = input_tuple
    params = params + (None, 1)
    (bssmat, nssmat, chi_params, ss_params, income_tax_params,
     iterative_params, factor, small_open_params, client,
     num_workers) = params
    income_tax_params = ('DEP',) + income_tax_params
    params = (bssmat, nssmat, chi_params, ss_params, income_tax_params,
              iterative_params, factor, small_open_params, client,
              num_workers)
    test_list = SS.SS_fsolve_reform(guesses, params)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_fsolve_reform_baselinespend():
    # Test SS.SS_fsolve_reform_baselinespend function.  Provide inputs
    # to function and ensure that output returned matches what it has
    # been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_baselinespend_inputs.pkl'))
    guesses, params = input_tuple
    params = params + (None, 1)
    (bssmat, nssmat, T_Hss, chi_params, ss_params, income_tax_params,
     iterative_params, factor, small_open_params, client,
     num_workers) = params
    income_tax_params = ('DEP',) + income_tax_params
    params = (bssmat, nssmat, T_Hss, chi_params, ss_params,
              income_tax_params, iterative_params, factor,
              small_open_params, client, num_workers)
    test_list = SS.SS_fsolve_reform_baselinespend(guesses, params)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_baselinespend_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_solver():
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_solver_inputs.pkl'))
    (b_guess_init, n_guess_init, rss, T_Hss, factor_ss, Yss, params,
     baseline, fsolve_flag, baseline_spending) = input_tuple
    (bssmat, nssmat, chi_params, ss_params, income_tax_params,
     iterative_params, small_open_params) = params
    income_tax_params = ('DEP',) + income_tax_params
    params = (bssmat, nssmat, chi_params, ss_params, income_tax_params,
              iterative_params, small_open_params)
    test_dict = SS.SS_solver(
        b_guess_init, n_guess_init, rss, T_Hss, factor_ss, Yss, params,
        baseline, fsolve_flag, baseline_spending)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_solver_outputs.pkl'))

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v))


def test_inner_loop():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/inner_loop_inputs.pkl'))
    (outer_loop_vars, params, baseline, baseline_spending) = input_tuple
    ss_params, income_tax_params, chi_params, small_open_params = params
    income_tax_params = ('DEP',) + income_tax_params
    params = (ss_params, income_tax_params, chi_params,
              small_open_params)
    test_tuple = SS.inner_loop(
         outer_loop_vars, params, baseline, baseline_spending)

    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/inner_loop_outputs.pkl'))

    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_tuple[i], v))


def test_euler_equation_solver():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/euler_eqn_solver_inputs.pkl'))
    (guesses, params) = input_tuple
    (r, w, T_H, factor, j, J, S, beta, sigma, ltilde, g_y, g_n_ss,
     tau_payroll, retire, mean_income_data, h_wealth, p_wealth,
     m_wealth, b_ellipse, upsilon, j, chi_b, chi_n, tau_bq, rho, lambdas,
     omega_SS, e, analytical_mtrs, etr_params, mtrx_params,
     mtry_params) = params
    tax_func_type = 'DEP'
    params = (r, w, T_H, factor, j, J, S, beta, sigma, ltilde, g_y,
              g_n_ss, tau_payroll, retire, mean_income_data, h_wealth,
              p_wealth, m_wealth, b_ellipse, upsilon, j, chi_b, chi_n,
              tau_bq, rho, lambdas, omega_SS, e, tax_func_type,
              analytical_mtrs, etr_params, mtrx_params, mtry_params)
    test_list = SS.euler_equation_solver(guesses, params)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/euler_eqn_solver_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_create_steady_state_parameters():
    # Test that SS parameters creates same objects with same inputs.
    input_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/create_params_inputs.pkl'))
    input_dict['tax_func_type'] = 'DEP'
    test_tuple = SS.create_steady_state_parameters(**input_dict)

    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/create_params_outputs.pkl'))
    (income_tax_params, ss_params, iterative_params, chi_params,
     small_open_params) = expected_tuple
    income_tax_params = ('DEP', ) + income_tax_params
    expected_tuple = (income_tax_params, ss_params, iterative_params,
                      chi_params, small_open_params)

    for i, v in enumerate(expected_tuple):
        for i2, v2 in enumerate(v):
            try:
                assert(all(test_tuple[i][i2] == v2))
            except ValueError:
                assert((test_tuple[i][i2] == v2).all())
            except TypeError:
                assert(test_tuple[i][i2] == v2)


@pytest.mark.parametrize('input_path,expected_path',
                         [('run_SS_open_unbal_inputs.pkl',
                           'run_SS_open_unbal_outputs.pkl'),
                          ('run_SS_closed_balanced_inputs.pkl',
                           'run_SS_closed_balanced_outputs.pkl')],
                         ids=['Open, Unbalanced', 'Closed Balanced'])
def test_run_SS(input_path, expected_path):
    # Test SS.run_SS function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', input_path))
    (income_tax_params, ss_params, iterative_params, chi_params,
     small_open_params, baseline, baseline_spending, baseline_dir) =\
        input_tuple
    income_tax_params = ('DEP',) + income_tax_params
    test_dict = SS.run_SS(
        income_tax_params, ss_params, iterative_params, chi_params,
        small_open_params, baseline, baseline_spending, baseline_dir)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', expected_path))

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v))
