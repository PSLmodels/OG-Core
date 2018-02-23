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
        CUR_PATH, 'SS_fsolve_inputs.pkl'), 'rb'))
    guesses, params = input_tuple
    test_list = SS.SS_fsolve(guesses, params)

    expected_list = pickle.load(open(os.path.join(
        CUR_PATH, 'SS_fsolve_outputs.pkl'), 'rb'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))
