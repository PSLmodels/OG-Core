import pytest
import numpy as np
from ogusa import elliptical_u_est as ee


def test_sumsq():
    '''
    Test of the sumsq() funcion
    '''
    expected_val = 47.1926846
    theta = 1.5
    l_tilde = 0.8
    b = 1.4
    k = 1.1
    upsilon = 2.5
    n_grid = np.array([0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.44, 0.55])
    objs = (theta, l_tilde, n_grid)
    params = b, k, upsilon
    test_val = ee.sumsq(params, *objs)

    assert np.allclose(expected_val, test_val)
