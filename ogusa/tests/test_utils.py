import pytest
from ogusa import utils
from ogusa.utils import Inequality
import numpy as np


def test_rate_conversion():
    '''
    Test of utils.rate_conversion
    '''
    expected_rate = 0.3
    annual_rate = 0.3
    start_age = 20
    end_age = 80
    s = 60
    test_rate = utils.rate_conversion(annual_rate, start_age, end_age, s)
    assert(np.allclose(expected_rate, test_rate))


# Parameter values to use for inequality tests
J = 2
S = 10
dist = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 1, 1, 2, 3, 4, 5, 6, 7, 20]])
pop_weights = np.ones(S) / S
ability_weights = np.array([0.5, 0.5])


def test_create_ineq_object():
    '''
    Test that Inequality class and be created
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    assert ineq


def test_gini():
    '''
    Test of Gini coefficient calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    gini = ineq.gini()
    assert np.allclose(gini, 0.41190476190476133)


def test_var_of_logs():
    '''
    Test of variance of logs calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    var_log = ineq.var_of_logs()
    assert np.allclose(var_log, 0.7187890928713506)


def test_ratio_pct1_pct2():
    '''
    Test of percentile ratio calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    ratio = ineq.ratio_pct1_pct2(0.9, 0.1)
    assert np.allclose(ratio, 9)


def test_top_share():
    '''
    Test of top share calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    top_share = ineq.top_share(0.05)
    assert np.allclose(top_share, 0.285714286)
