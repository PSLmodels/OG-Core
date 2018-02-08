import pytest
import numpy as np
from ogusa import household


test_data = [(0.1, 1, 10),
             (0.2, 2.5, 55.90169944),
             (np.array([0.5, 6.2, 1.5]), 3.2,
              np.array([9.18958684, 0.002913041, 0.273217159]))]


@pytest.mark.parametrize('c,sigma,expected', test_data,
                         ids=['Scalar 0', 'Scalar 1', 'Vector'])
def test_marg_ut_cons(c, sigma, expected):
    # Test marginal utility of consumption calculation
    test_value = household.marg_ut_cons(c, sigma)

    assert np.allclose(test_value, expected)


# Note that params tuple in order: b_ellipse, upsilon, ltilde, chi_n
test_data = [(0.87, (0.527, 1.497, 1.0, 3.3), 2.825570309),
             (0.0, (0.527, 1.497, 1.0, 3.3), 0),
             (0.99999, (0.527, 1.497, 1.0, 3.3), 69.52423604),
             (0.00001, (0.527, 1.497, 1.0, 3.3), 0.005692782),
             (0.8, (0.527, 0.9, 1.0, 3.3), 1.471592068),
             (0.8, (0.527, 0.9, 2.3, 3.3), 0.795937549),
             (0.8, (2.6, 1.497, 1.0, 3.3), 11.66354267),
             (np.array([[0.8, 0.9, 0.3], [0.5, 0.2, 0.99]]),
              (0.527, 1.497, 1.0, 3.3),
              np.array([[2.364110379, 3.126796062, 1.014935377],
                        [1.4248841, 0.806333875, 6.987729463]]))]


@pytest.mark.parametrize('n,params,expected', test_data,
                         ids=['1', '2', '3', '4', '5', '6', '7', '8'])
def test_marg_ut_labor(n, params, expected):
    # Test marginal utility of labor calculation
    test_value = household.marg_ut_labor(n, params)

    assert np.allclose(test_value, expected)


# def test_get_cons():
#
#     assert np.allclose()
#
#
# def test_FOC_savings():
#
#     assert np.allclose()
#
#
# def test_FOC_labor():
#
#     assert np.allclose()
#
#
# def test_constraint_checker_SS():
#
#     assert np.allclose()
#
#
# def test_constraint_checker_TPI():
#
#     assert np.allclose()
