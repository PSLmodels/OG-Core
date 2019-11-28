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


def test_arc_error():
    '''
    Test of arc_error()
    '''
    expected_vals = np.array([30.19765553, -1.40779391, 14.19212336])
    a = 1.3
    b = 2.2
    c = 0.5
    abc_vals = (a, b, c)
    first_point = 30.2
    coef1 = 0.05995294
    coef2 = -0.00004086
    coef3 = -0.00000521
    abil_deprec = 0.47
    params = (first_point, coef1, coef2, coef3, abil_deprec)
    test_vals = income.arc_error(abc_vals, params)

    assert np.allclose(test_vals, expected_vals)


def test_arctan_fit():
    '''
    Test arctan_fit() function
    '''
    expected_vals = np.array(
        [30.19999399, 30.19998699, 30.19997918, 30.19997039,
         30.19996043, 30.19994904, 30.1999359, 30.19992057,
         30.19990246, 30.19988072, 30.19985415, 30.19982094,
         30.19977824, 30.19972131, 30.1996416, 30.19952204, 30.19932277,
         30.19892423, 30.19772859, 14.19399974])
    a = 1.3
    b = 2.2
    c = 0.5
    init_guesses = (a, b, c)
    first_point = 30.2
    coef1 = 0.05995294
    coef2 = -0.00004086
    coef3 = -0.00000521
    abil_deprec = 0.47
    test_vals = income.arctan_fit(
        first_point, coef1, coef2, coef3, abil_deprec, init_guesses)

    assert np.allclose(test_vals, expected_vals)
