from ogusa import txfunc

import pytest
import numpy as np
import pickle
import os

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize('tax_func_type,expected_mat',
                         [('DEP', 0.032749763), ('DS', 1.475619188)],
                         ids=['DEP', 'GS'])
def test_wsumsq(tax_func_type, expected):
    A = 0.01
    B = 0.02
    C = 0.1
    D = 1.1
    max_x = 0.55
    min_x = 0.17
    max_y = 0.46
    min_y = 0.04
    shift_x = 0.1
    shift_y = 0.2
    shift = 0.04
    share = 0.8
    phi0 = 0.396
    phi1 = 0.7
    phi2 = 0.9
    X = np.array([32.0, 44.0, 1.6, 0.4])
    Y = np.array([32.0, 55.0, 0.9, 0.03])
    txrates = np.array([0.6, 0.5, 0.3, 0.25])
    wgts = np.array([0.1, 0.25, 0.55, 0.1])
    if tax_func_type == 'DEP':
        params = A, B, C, D, max_x, max_y, share
        args = (min_x, min_y, shift), X, Y, txrates, wgts, tax_func_type
    elif tax_func_type == 'GS':
        params = phi0, phi1, phi2
        args = None, X, Y, txrates, wgts, tax_func_type
    test_val = txfunc.wsumsq(params, *args)

    assert(np.allclose(test_val, expected))


@pytest.mark.parametrize('se_mult,expected_mat',
                         [(2,
                           np.array([[False, False], [False, False],
                                     [False, True]])),
                          (8,
                           np.array([[False, False], [False, False],
                                     [False, False]]))],
                         ids=['2', '8'])
def test_find_outliers(se_mult, expected_mat):
    # Test the find outliers function
    sse_mat = np.array([[21.0, 22.0], [20.0, 32.0], [20.0, 100.0]])
    age_vec = np.array([40, 41])
    start_year = 2018
    varstr = 'MTRy'
    test_mat = txfunc.find_outliers(sse_mat, age_vec, se_mult,
                                    start_year, varstr, False)

    assert(np.allclose(test_mat, expected_mat))


def test_replace_outliers():
    """
    4 cases:
        s is an outlier and is 0
        s is an outlier and is in the interior (s > 0 and s < S)
        s is not an outlier but the first s - 1 ages were (s = 1 in our case)
        s is an outlier and is the max age
    """
    S = 10
    BW = 2
    numparams = 3
    random_state = np.random.RandomState(10)
    param_arr = random_state.rand(S * BW * numparams).reshape(S, BW, numparams)
    sse_big_mat = ~ np.ones((S, BW), dtype=bool)
    sse_big_mat[0, 0] = True
    sse_big_mat[1, 0] = True
    sse_big_mat[S-4, 0] = True
    sse_big_mat[S-5, 0] = True
    sse_big_mat[S - 2, 0] = True
    sse_big_mat[S - 1, 0] = True

    act = txfunc.replace_outliers(param_arr, sse_big_mat)

    exp = [[[0.00394827, 0.51219226, 0.81262096],
            [0.74880388, 0.49850701, 0.22479665]],

           [[0.00394827, 0.51219226, 0.81262096],
            [0.08833981, 0.68535982, 0.95339335]],

           [[0.00394827, 0.51219226, 0.81262096],
            [0.61252607, 0.72175532, 0.29187607]],

           [[0.91777412, 0.71457578, 0.54254437],
            [0.14217005, 0.37334076, 0.67413362]],

           [[0.44183317, 0.43401399, 0.61776698],
            [0.51313824, 0.65039718, 0.60103895]],

           [[0.3608713, 0.57495943, 0.5290622],
            [0.31923609, 0.09045935, 0.30070006]],

           [[0.27990942, 0.71590487, 0.44035742],
            [0.62628715, 0.54758616, 0.819287]],

           [[0.19894754, 0.8568503, 0.35165264],
            [0.75464769, 0.29596171, 0.88393648]],

           [[0.19894754, 0.8568503, 0.35165264],
            [0.09346037, 0.82110566, 0.15115202]],

           [[0.19894754, 0.8568503, 0.35165264],
            [0.45630455, 0.82612284, 0.25137413]]]

    assert np.allclose(act, exp)


# def test_txfunc_est():


# def test_tax_func_loop():


# def test_get_tax_rates():


# def test_tax_func_estimate():


# def test_get_tax_func_estimate():
# saves a pickle file
