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
    delta_tau = 1.0

    Y = np.array([2.0, 2.0])
    K = np.array([1.0, 1.0])

    epsilon = 0.0
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.array([0.0, 0.0])))

    epsilon = 0.5
    delta = 0.5
    r = firm.get_r(Y, K, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(r, np.array([1.0, 1.0])))


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


def test_get_L():
    """
        Simulate data similar to observed and carry out get_L
        in simplest way possible.
    """
    T = 160
    s, j = 40, 2
    omega = np.random.rand(s).reshape(s, 1)
    lambdas = np.random.rand(j).reshape(1, j)
    n = np.random.rand(T * s * j).reshape(T, s, j)
    e = np.tile(np.random.rand(s, j), (T, 1, 1))

    L_loop = np.ones(T * s * j).reshape(T, s, j)
    for t in range(T):
        for i in range(s):
            for k in range(j):
                L_loop[t, i, k] *= (omega[i, 0] * lambdas[0, k] *
                                    n[t, i, k] * e[t, i, k])

    L_matrix = e * omega * lambdas * n
    assert (np.allclose(L_loop, L_matrix))

    method = 'SS'
    L = firm.get_L(n[0], (e[0], omega, lambdas, method))
    assert (np.allclose(L, L_loop[0].sum()))

    method = 'TPI'
    L = firm.get_L(n, (e, omega, lambdas, method))
    assert (np.allclose(L, L_loop.sum(1).sum(1)))

def test_get_I():
    """
        Simulate data similar to observed and carry out get_I
        in simplest way possible.
    """
    T = 160
    s, j = 40, 2

    b_splus1 = 10 * np.random.rand(T * s * j).reshape(T, s, j)
    K_p1 = 0.9 + np.random.rand(T)
    K = 0.9 + np.random.rand(T)

    delta = np.random.rand()
    g_y = np.random.rand()

    def shifted_arr():
        arr_t = []
        arr_shift_t = []
        for t in range(T):
            arr = np.random.rand(s).reshape(s, 1)
            arr_shift = np.append(arr[1:], [0.0])

            arr_t.append(arr)
            arr_shift_t.append(arr_shift)

        return (np.array(arr_t).reshape(T, s, 1),
                np.array(arr_shift_t).reshape(T, s, 1))

    lambdas = np.random.rand(2)
    imm_rates, imm_shift = shifted_arr()
    imm_rates = imm_rates - 0.5
    imm_shift = imm_shift - 0.5
    omega, omega_shift = shifted_arr()

    g_n = np.random.rand(T)

    res_loop = np.ones(T * s * j).reshape(T, s, j)
    for t in range(T):
        for i in range(s):
            for k in range(j):
                res_loop[t, i, k] *= (omega_shift[t, i, 0] * imm_shift[t, i, 0] *
                                      lambdas[k] * b_splus1[t, i, k])

    res_matrix = (b_splus1 * (imm_shift * omega_shift) * lambdas)

    assert (np.allclose(res_loop, res_matrix))

    aggI_SS_test = ((1 + g_n[0]) * np.exp(g_y) *
                    (K_p1[0] - res_matrix[0].sum() / (1 + g_n[0])) -
                    (1.0 - delta) * K[0])
    aggI_SS = firm.get_I(b_splus1[0], K_p1[0], K[0],
                         (delta, g_y, omega[0], lambdas, imm_rates[0], g_n[0], 'SS'))
    assert (np.allclose(aggI_SS, aggI_SS_test))

    aggI_TPI_test = ((1 + g_n) * np.exp(g_y) *
                     (K_p1 - res_matrix.sum(1).sum(1) / (1 + g_n)) -
                     (1.0 - delta) * K)
    aggI_TPI = firm.get_I(b_splus1, K_p1, K,
                          (delta, g_y, omega, lambdas, imm_rates, g_n, 'TPI'))
    assert (np.allclose(aggI_TPI, aggI_TPI_test))


def test_get_K():
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    L = np.array([2.0, 2.0])
    r = np.array([1.0, 1.0])
    gamma = 0.5
    tau_b = 0.75
    delta = 4.0
    delta_tau = 4.0/3.0
    Z = 2.0

    epsilon = 1
    K = firm.get_K(L, r, (Z, gamma, epsilon, delta, tau_b, delta_tau))
    assert (np.allclose(K, np.array([2./(16**2), 2./(16**2)])))

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
