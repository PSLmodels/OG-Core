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


# def test_marg_ut_labor():
#
#     assert np.allclose()
#
#
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
