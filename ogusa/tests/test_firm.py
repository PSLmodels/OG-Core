import pytest
from ogusa import firm
import numpy as np
from ogusa.pb_api import Specifications


def test_get_r():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    p = Specifications()
    new_param_values = {
        'Z': 0.5,
        'gamma': 0.5,
        'delta_annual': 0.25,
        'tau_b': 0.5,
        'delta_tau_annual': 0.35,
        'epsilon': 0.0
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)
    # assign values for Y and K variables
    Y = np.array([2.0, 2.0])
    K = np.array([1.0, 1.0])

    r = firm.get_r(Y, K, p)
    assert (np.allclose(r, np.array([0.0, 0.0])))

    new_param_values = {
        'epsilon': 0.5,
        'delta_annual': 0.5
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    r = firm.get_r(Y, K, p)
    assert (np.allclose(r, np.array([0.675, 0.675])))


def test_get_w():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    p = Specifications()
    new_param_values = {
        'Z': 0.5,
        'gamma': 0.5,
        'epsilon': 0.0
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    Y = np.array([2.0, 2.0])
    L = np.array([1.0, 1.0])

    w = firm.get_w(Y, L, p)
    assert (np.allclose(w, np.array([0.5, 0.5])))

    new_param_values = {
        'epsilon': 0.5
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    w = firm.get_w(Y, L, p)
    assert (np.allclose(w, np.array([2.0, 2.0])))


def test_get_Y():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    p = Specifications()
    new_param_values = {
        'Z': 2.0,
        'gamma': 0.5,
        'epsilon': 1.0
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    L = np.array([4.0, 4.0])
    K = np.array([9.0, 9.0])

    Y = firm.get_Y(K, L, p)
    assert (np.allclose(Y, np.array([12.0, 12.0])))

    new_param_values = {
        'Z': 2.0,
        'gamma': 0.5,
        'epsilon': 0.0
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    Y = firm.get_Y(K, L, p)
    assert (np.allclose(Y, np.array([13.0, 13.0])))

    new_param_values = {
        'Z': 2.0,
        'gamma': 0.5,
        'epsilon': 0.5
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    L = np.array([1/12.0, 1/12.0])
    K = np.array([1/4.0, 1/4.0])

    Y = firm.get_Y(K, L, p)
    assert (np.allclose(Y, np.array([0.5, 0.5])))


def test_get_K():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    p = Specifications()
    new_param_values = {
        'gamma': 0.5,
        'tau_b': 0.75,
        'delta_annual': 0.15,
        'delta_tau_annual': 0.03,
        'Z': 2.0,
        'epsilon': 1.0
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    L = np.array([2.0, 2.0])
    r = np.array([1.0, 1.0])

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([0.09832793, 0.09832793])))

    new_param_values = {
        'epsilon': 0.0
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([0.0, 0.0])))

    new_param_values = {
        'epsilon': 0.5,
        'Z': 4.0,
        'tau_b': 0.0,
        'delta_tau_annual': 2.0,
        'delta_annual': 3.0
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([2.0, 2.0])))

    new_param_values = {
        'tau_b': 0.5
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([1.26598632, 1.26598632])))
