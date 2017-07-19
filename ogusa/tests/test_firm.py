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
    delta = 0.25
    tau_b = 0.5
    delta_tau = 1.0

    Y = np.array([2.0, 2.0])
    K = np.array([1.0, 1.0])

    # test if epsilon == .... works
    # test general accuracy and vector operations
    epsilon = 0.0
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.array([0.0, 0.0])))

    epsilon = 0.5
    delta = 0.5
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.array([1.0, 1.0])))


def test_get_w():
    # numbers to make calculation easy
    # example of observed parameters:
    # Z     gamma   epsilon
    # 1.0   0.35     1.0
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
    # numbers to make calculation easy
    # example of observed parameters:
    # Z     gamma   epsilon
    # 1.0   0.35     1.0
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


def test_get_L():
    T = 150
    s, j = 40, 2
    n = 10 * np.random.rand(s, j)
    e = 10 * np.random.rand(s, j)
    omega = np.array([np.random.rand(s)]).T
    lam = np.array([np.random.rand(j)])

    L_test = np.ones(s * j).reshape(s, j)
    for i in range(s):
        for k in range(j):
            L_test[i, k] *= omega[i, 0] * lam[0, k] * n[i, k] * e[i, k]

    method = 'SS'
    print(L_test.sum(), L_test.sum(axis=0))
    L = firm.get_L(n, (e, omega, lam, method))
    assert (np.allclose(L, L_test.sum()))
    # print("LSHAPE1", L.shape, L.ndim)


    method = 'TPI'
    L_test_tile = np.tile(L_test, (T, 1, 1))
    n_tile = np.tile(n, (T, 1, 1))
    e_tile = np.tile(e, (T, 1, 1))

    L = firm.get_L(n_tile, (e_tile, omega, lam, method))
    assert (np.allclose(L, L_test_tile.sum(1).sum(1)))
    # print("LSHAPE2", L.shape, L.ndim)
