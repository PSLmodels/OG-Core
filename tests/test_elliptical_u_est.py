import pytest
import numpy as np
from ogcore import elliptical_u_est as ee


def test_CFE_u():
    """
    Test of the CFE_u() function
    """
    expected_val = np.array(
        [
            0.070710678,
            0.03444595,
            0.123526471,
            0.194855716,
            0.0125,
            0.002209709,
            0.070710678,
            0.089736002,
            0.156762344,
        ]
    )
    theta = 1.5
    l_tilde = 0.8
    n_grid = np.array([0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.44, 0.55])
    test_val = ee.CFE_u(theta, l_tilde, n_grid)

    assert np.allclose(expected_val, test_val)


def test_CFE_mu():
    """
    Test of the CFE_mu() function
    """
    expected_val = np.array(
        [
            0.441941738,
            0.287049579,
            0.617632356,
            0.811898816,
            0.15625,
            0.055242717,
            0.441941738,
            0.509863646,
            0.712556107,
        ]
    )
    theta = 1.5
    l_tilde = 0.8
    n_grid = np.array([0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.44, 0.55])
    test_val = ee.CFE_mu(theta, l_tilde, n_grid)

    assert np.allclose(expected_val, test_val)


def test_elliptical_u():
    """
    Test of the elliptical_u() function
    """
    expected_val = np.array(
        [
            2.395194818,
            2.450469186,
            2.307717784,
            2.171834687,
            2.482333146,
            2.496901266,
            2.395194818,
            2.364726426,
            2.24740401,
        ]
    )
    b = 1.4
    k = 1.1
    upsilon = 2.5
    l_tilde = 0.8
    n_grid = np.array([0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.44, 0.55])
    test_val = ee.elliptical_u(b, k, upsilon, l_tilde, n_grid)

    assert np.allclose(expected_val, test_val)


def test_elliptical_mu():
    """
    Test of the elliptical_mu() function
    """
    expected_val = np.array(
        [
            0.695316381,
            0.424179802,
            1.079199998,
            1.696798556,
            0.222956959,
            0.07759729,
            0.695316381,
            0.831341091,
            1.344511778,
        ]
    )
    b = 1.4
    upsilon = 2.5
    l_tilde = 0.8
    n_grid = np.array([0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.44, 0.55])
    test_val = ee.elliptical_mu(b, upsilon, l_tilde, n_grid)

    assert np.allclose(expected_val, test_val)


def test_sumsq():
    """
    Test of the sumsq() funcion
    """
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


def test_sumsq_MU():
    """
    Test of the sumsq_MU() funcion
    """
    expected_val = 1.650959619
    theta = 1.5
    l_tilde = 0.8
    b = 1.4
    upsilon = 2.5
    n_grid = np.array([0.4, 0.3, 0.5, 0.6, 0.2, 0.1, 0.4, 0.44, 0.55])
    objs = (theta, l_tilde, n_grid)
    params = b, upsilon
    test_val = ee.sumsq_MU(params, *objs)

    assert np.allclose(expected_val, test_val)


def test_estimation():
    """
    Test of estimation() function
    """
    expected_b = 0.7048132709249104
    expected_upsilon = 1.4465752174288222
    frisch = 1.5
    l_tilde = 1.0
    test_b, test_upsilon = ee.estimation(frisch, l_tilde)

    assert np.allclose(expected_b, test_b)
    assert np.allclose(expected_upsilon, test_upsilon)
