import pytest
import json
import pickle
import numpy as np
import os
from ogusa import TPI

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


# def test_create_tpi_params():
#     # Test that TPI parameters creates same objects with same inputs.
#     with open(os.path.join(CUR_PATH,
#                            'test_io_data/create_tpi_params_inputs.pkl'),
#               'rb') as f:
#         input_dict = pickle.load(f)
#     test_tuple = TPI.create_tpi_params(**input_dict)
#
#     with open(os.path.join(CUR_PATH,
#                            'test_io_data/create_tpi_params_outputs.pkl'),
#               'rb') as f:
#         expected_tuple = pickle.load(f)
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
    with open(os.path.join(CUR_PATH,
                           'test_io_data/firstdoughnutring_inputs.pkl'),
              'rb') as f:
        input_tuple = pickle.load(f)
    guesses, r, w, b, BQ, T_H, j, params = input_tuple
    test_list = TPI.firstdoughnutring(guesses, r, w, b, BQ, T_H, j, params)

    with open(os.path.join(CUR_PATH,
                           'test_io_data/firstdoughnutring_outputs.pkl'),
              'rb') as f:
        expected_list = pickle.load(f)

    assert(np.allclose(np.array(test_list), np.array(expected_list)))
#
#
# def test_twist_doughnut():
#
#
# def test_inner_loop():
#
#
# def test_run_TPI():
