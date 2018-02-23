import pytest
import json
import pickle
import numpy as np
import os
from ogusa import SS

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


# def test_SS_fsolve():
#     # Test SS.SS_fsolve function.  Provide inputs to function and
#     # ensure that output returned matches what it has been before.
#     input_tuple = tuple(json.load(open(os.path.join(
#         CUR_PATH, 'SS_fsolve_inputs.json'))))
#     guesses, params = input_tuple
#     params = tuple(params)
#     bssmat = np.array(params[0])
#     nssmat = np.array(params[1])
#
#     (bssmat, nssmat, chi_params, ss_params, income_tax_params,
#      iterative_params, small_open_params) = params
#     print 'Guesses = ', type(params[0]), type(params[1])
#     test_list = SS.SS_fsolve(guesses, params)
#
#     expected_list = json.load(open(os.path.join(
#         CUR_PATH, 'SS_fsolve_outputs.json')))
#
#     assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_fsolve():
    # Test SS.SS_fsolve function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_fsolve_inputs.pkl'), 'rb'))
    guesses, params = input_tuple
    test_list = SS.SS_fsolve(guesses, params)

    expected_list = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_fsolve_outputs.pkl'), 'rb'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_fsolve_reform():
    # Test SS.SS_fsolve_reform function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_fsolve_reform_inputs.pkl'), 'rb'))
    guesses, params = input_tuple
    test_list = SS.SS_fsolve_reform(guesses, params)

    expected_list = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_fsolve_reform_outputs.pkl'), 'rb'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_fsolve_reform_baselinespend():
    # Test SS.SS_fsolve_reform_baselinespend function.  Provide inputs
    # to function and ensure that output returned matches what it has
    # been before.
    input_tuple = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_fsolve_reform_baselinespend_inputs.pkl'), 'rb'))
    guesses, params = input_tuple
    test_list = SS.SS_fsolve_reform_baselinespend(guesses, params)

    expected_list = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_fsolve_reform_baselinespend_outputs.pkl'), 'rb'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_solver():
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_solver_inputs.pkl'), 'rb'))
    (b_guess_init, n_guess_init, rss, T_Hss, factor_ss, Yss, params,
     baseline, fsolve_flag, baseline_spending) = input_tuple
    test_dict = SS.SS_solver(
        b_guess_init, n_guess_init, rss, T_Hss, factor_ss, Yss, params,
        baseline, fsolve_flag, baseline_spending)

    expected_dict = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/SS_solver_outputs.pkl'), 'rb'))

    for k, v in expected_dict.iteritems():
        assert(np.allclose(test_dict[k], v))


def test_inner_loop():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/inner_loop_inputs.pkl'), 'rb'))
    (outer_loop_vars, params, baseline, baseline_spending) = input_tuple
    test_dict = SS.inner_loop(
         outer_loop_vars, params, baseline, baseline_spending)

    expected_tuple = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/inner_loop_outputs.pkl'), 'rb'))

    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_dict[i], v))


def test_euler_equation_solver():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/euler_eqn_solver_inputs.pkl'), 'rb'))
    (guesses, params) = input_tuple
    test_list = SS.euler_equation_solver(guesses, params)

    expected_list = pickle.load(open(os.path.join(
        CUR_PATH, 'test_io_data/euler_eqn_solver_outputs.pkl'), 'rb'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))
