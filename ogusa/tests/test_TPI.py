import pytest
import json
import pickle
import numpy as np
import os
from ogusa import TPI, utils
from ogusa.pb_api import Specifications, reform_warnings_errors

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


# def test_create_tpi_params():
#     # Test that TPI parameters creates same objects with same inputs.
#     with open(os.path.join(CUR_PATH,
#                            'test_io_data/create_tpi_params_inputs.pkl'),
#               'rb') as f:
#         input_dict = pickle.load(f, encoding='latin1')
#     test_tuple = TPI.create_tpi_params(**input_dict)
#
#     with open(os.path.join(CUR_PATH,
#                            'test_io_data/create_tpi_params_outputs.pkl'),
#               'rb') as f:
#         expected_tuple = pickle.load(f, encoding='latin1')
#
#     for i, v in enumerate(expected_tuple):
#         for i2, v2 in enumerate(v):
#             try:
#                 assert(all(test_tuple[i][i2]==v2))
#             except ValueError:
#                 assert((test_tuple[i][i2]==v2).all())
#             except TypeError:
#                 assert(test_tuple[i][i2]==v2)


def test_firstdoughnutring():
    # Test TPI.firstdoughnutring function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/firstdoughnutring_inputs.pkl'))
    guesses, r, w, b, BQ, T_H, j, params = input_tuple
    income_tax_params, tpi_params, initial_b = params
    tpi_params = tpi_params + [True]
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, p.tau_b, p.delta_tau,
     p.tau_payroll, p.tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, p.retire, p.mean_income_data, factor, p.h_wealth,
     p.p_wealth, p.m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params
    p.tax_func_type = 'DEP'
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params, (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params, (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params, (1, 0, 2))
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.num_workers = 1
    # test_list = TPI.firstdoughnutring(guesses, r, w, b, BQ, T_H, j, params)
    test_list = TPI.firstdoughnutring(guesses, r, w, BQ, T_H, theta,
                                      factor, j, initial_b, p)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/firstdoughnutring_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


# def test_twist_doughnut():
#     # Test TPI.twist_doughnut function.  Provide inputs to function and
#     # ensure that output returned matches what it has been before.
#     input_tuple = utils.safe_read_pickle(
#         os.path.join(CUR_PATH, 'test_io_data/twist_doughnut_inputs.pkl'))
#     guesses, r, w, BQ, T_H, j, s, t, params = input_tuple
#     income_tax_params, tpi_params, initial_b = params
#     tpi_params = tpi_params + [True]
#     income_tax_params = ('DEP',) + income_tax_params
#     params = (income_tax_params, tpi_params, initial_b)
#     test_list = TPI.twist_doughnut(guesses, r, w, BQ, T_H, j, s, t, params)
#
#     expected_list = utils.safe_read_pickle(
#         os.path.join(CUR_PATH, 'test_io_data/twist_doughnut_outputs.pkl'))
#
#     assert(np.allclose(np.array(test_list), np.array(expected_list)))
#
#
# @pytest.mark.full_run
# def test_inner_loop():
#     # Test TPI.inner_loop function.  Provide inputs to function and
#     # ensure that output returned matches what it has been before.
#     input_tuple = utils.safe_read_pickle(
#         os.path.join(CUR_PATH, 'test_io_data/tpi_inner_loop_inputs.pkl'))
#     guesses, outer_loop_vars, params, j = input_tuple
#     income_tax_params, tpi_params, initial_values, ind = params
#     initial_values = initial_values #+ (0.0,)
#     tpi_params = tpi_params #+ [True]
#     income_tax_params = ('DEP',) + income_tax_params
#     params = (income_tax_params, tpi_params, initial_values, ind)
#     guesses = (guesses[0], guesses[1])
#     test_tuple = TPI.inner_loop(guesses, outer_loop_vars, params, j)
#
#     expected_tuple = utils.safe_read_pickle(
#         os.path.join(CUR_PATH, 'test_io_data/tpi_inner_loop_outputs.pkl'))
#
#     for i, v in enumerate(expected_tuple):
#         assert(np.allclose(test_tuple[i], v))
#
#
# @pytest.mark.full_run
# def test_run_TPI():
#     # Test TPI.run_TPI function.  Provide inputs to function and
#     # ensure that output returned matches what it has been before.
#     input_tuple = utils.safe_read_pickle(
#         os.path.join(CUR_PATH, 'test_io_data/run_TPI_inputs.pkl'))
#     (income_tax_params, tpi_params, iterative_params, small_open_params,
#      initial_values, SS_values, fiscal_params, biz_tax_params,
#      output_dir, baseline_spending) = input_tuple
#     tpi_params = tpi_params + [True]
#     initial_values = initial_values + (0.0,)
#     income_tax_params = ('DEP',) + income_tax_params
#     test_dict = TPI.run_TPI(
#         income_tax_params, tpi_params, iterative_params,
#         small_open_params, initial_values, SS_values, fiscal_params,
#         biz_tax_params, output_dir, baseline_spending)
#
#     expected_dict = utils.safe_read_pickle(
#         os.path.join(CUR_PATH, 'test_io_data/run_TPI_outputs.pkl'))
#
#     for k, v in expected_dict.items():
#         assert(np.allclose(test_dict[k], v))
