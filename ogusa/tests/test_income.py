'''
Tests of income.py module
'''

import pytest
import numpy as np
from ogusa import income


def test_artctan_func():
    '''
    Test of arctan_func()
    '''
    expected_vals = np.array([0.14677821, 0.083305594, 0.057901228])
    xvals = np.array([1, 2, 3])
    a = 1.3
    b = 2.2
    c = 0.5
    test_vals = income.arctan_func(xvals, a, b, c)

    assert np.allclose(test_vals, expected_vals)


def test_artctan_deriv_func():
    '''
    Test of arctan_deriv_func()
    '''
    expected_vals = np.array([-0.109814991, -0.036400091, -0.017707961])
    xvals = np.array([1, 2, 3])
    a = 1.3
    b = 2.2
    c = 0.5
    test_vals = income.arctan_deriv_func(xvals, a, b, c)

    assert np.allclose(test_vals, expected_vals)
