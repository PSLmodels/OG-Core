import pytest
from ogusa import firm
import numpy as np


def test_get_r():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    Z = 0.5
    gamma = 0.5
    delta = 0.25
    tau_b = 0.5
    delta_tau = 0.35

    Y = np.array([2.0, 2.0])
    K = np.array([1.0, 1.0])

    epsilon = 0.0
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.array([0.0, 0.0])))

    epsilon = 0.5
    delta = 0.5
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.array([0.675, 0.675])))


def test_get_w():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    Z = 0.5
    gamma = 0.5
    Y = np.array([2.0, 2.0])
    L = np.array([1.0, 1.0])

    epsilon = 0.0
    w = firm.get_w(Y, L, (Z, gamma, epsilon))
    assert (np.allclose(w, np.array([0.5, 0.5])))

    epsilon = 0.5
    w = firm.get_w(Y, L, (Z, gamma, epsilon))
    assert (np.allclose(w, np.array([2.0, 2.0])))


def test_get_Y():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    Z = 2.0
    gamma = 0.5
    L = np.array([4.0, 4.0])
    K = np.array([9.0, 9.0])

    epsilon = 1
    Y = firm.get_Y(K, L, (Z, gamma, epsilon))
    assert (np.allclose(Y, np.array([12.0, 12.0])))

    epsilon = 0
    Y = firm.get_Y(K, L, (Z, gamma, epsilon))
    assert (np.allclose(Y, np.array([13.0, 13.0])))

    epsilon = 0.5
    Z = 2
    L = np.array([1/12.0, 1/12.0])
    K = np.array([1/4.0, 1/4.0])

    Y = firm.get_Y(K, L, (Z, gamma, epsilon))
    assert (np.allclose(Y, np.array([0.5, 0.5])))


def test_get_K():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    L = np.array([2.0, 2.0])
    r = np.array([1.0, 1.0])
    gamma = 0.5
    tau_b = 0.75
    delta = 0.15
    delta_tau = 0.2 * delta
    Z = 2.0

    epsilon = 1
    K = firm.get_K(L, r, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(K, np.array([0.09832793, 0.09832793])))

    epsilon = 0
    K = firm.get_K(L, r, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(K, np.array([0.0, 0.0])))

    epsilon = 0.5
    Z = 4.0
    tau_b = 0.0
    delta_tau = 2.0
    delta = 3.0
    K = firm.get_K(L, r, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(K, np.array([2.0, 2.0])))

    tau_b = 0.5
    K = firm.get_K(L, r, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(K, np.array([1.26598632, 1.26598632])))
