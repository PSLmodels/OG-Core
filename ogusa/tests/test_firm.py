import pytest
from ogusa import firm
import numpy as np
from ogusa.parameters import Specifications


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
    p.update_specifications(new_param_values)

    L = np.array([4.0, 4.0])
    K = np.array([9.0, 9.0])

    Y = firm.get_Y(K, L, p)
    assert (np.allclose(Y, np.array([12.0, 12.0])))

    new_param_values = {
        'Z': 2.0,
        'gamma': 0.5,
        'epsilon': 0.2
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

    Y = firm.get_Y(K, L, p)
    assert (np.allclose(Y, np.array([18.84610765, 18.84610765]),
                        atol=1e-6))

    new_param_values = {
        'Z': 2.0,
        'gamma': 0.5,
        'epsilon': 0.2
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

    L = np.array([1 / 12.0, 1 / 12.0])
    K = np.array([1 / 4.0, 1 / 4.0])

    Y = firm.get_Y(K, L, p)
    assert (np.allclose(Y, np.array([0.39518826, 0.39518826]),
                        atol=1e-6))


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
        'epsilon': 0.2
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)
    # assign values for Y and K variables
    Y = np.array([2.0, 2.0])
    K = np.array([1.0, 1.0])

    r = firm.get_r(Y, K, p)
    assert (np.allclose(r, np.array([7.925, 7.925])))

    new_param_values = {
        'epsilon': 0.5,
        'delta_annual': 0.5
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

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
        'epsilon': 0.2
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

    Y = np.array([2.0, 2.0])
    L = np.array([1.0, 1.0])

    w = firm.get_w(Y, L, p)
    assert (np.allclose(w, np.array([16., 16.]), atol=1e-6))

    new_param_values = {
        'epsilon': 0.5
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

    w = firm.get_w(Y, L, p)
    assert (np.allclose(w, np.array([2.0, 2.0])))


# def test_get_KLrat_from_r():
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     p = Specifications()
#     new_param_values = {
#         'Z': 0.5,
#         'gamma': 0.4,
#         'epsilon': 0.8,
#         'delta_annual': 0.05,
#         'delta_tau_annual': 0.35
#     }
#     # update parameters instance with new values for test
#     p.update_specifications(new_param_values)

#     r = np.array([0.04, 0.55])
#     KLratio = firm.get_KLrat_from_r(r, p)
#     assert (np.allclose(KLratio, ?))


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
    p.update_specifications(new_param_values)

    L = np.array([2.0, 2.0])
    r = np.array([1.0, 1.0])

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([0.09832793, 0.09832793])))

    new_param_values = {
        'epsilon': 0.2
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([0.91363771, 0.91363771]),
                        atol=1e-6))

    new_param_values = {
        'epsilon': 0.5,
        'Z': 4.0,
        'tau_b': 0.0,
        'delta_tau_annual': 0.5,
        'delta_annual': 0.05
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([5.80720058, 5.80720058]),
                        atol=1e-6))

    new_param_values = {
        'tau_b': 0.5
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values)

    K = firm.get_K(L, r, p)
    assert (np.allclose(K, np.array([4.32455532, 4.32455532]),
                        atol=1e-6))
