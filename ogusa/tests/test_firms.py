import pytest
from ogusa import firm
import numpy as np


def test_get_r():
    # numbers to make calculation easy
    # example of observed parameters:
    # Z     gamma   epsilon  delta  tau_b  delta_tau
    # 1.0   0.35     1.0    0.0975  0.2     0.0975
    Z = 0.5
    gamma = 0.5
    epsilon = 0.5
    delta = 0.25
    tau_b = 0.5
    delta_tau = 1.0

    Y = np.array([2.0, 2.0])
    K = np.array([1.0, 1.0])

    # test if epsilon == .... works
    # test general accuracy and vector operations
    epsilon = 0.0
    r = firm.get_r(1.0, 1.0, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    # r_test = (1-0.5) * 0.5 - 0.25 = 0
    assert (np.allclose(r, 0.0))
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.zeros(2)))

    epsilon = 0.5
    delta = 0.5
    r = firm.get_r(2.0, 1.0, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, 1))
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.ones(2)))
