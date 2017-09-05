import pytest
import numpy as np
from ogusa import aggregates as aggr


def test_get_L():
    """
        Simulate data similar to observed and carry out get_L
        in simplest way possible.
    """
    T = 160
    S, J = 40, 2

    # normalize across S and J axes
    omega = np.random.rand(S).reshape(S, 1)
    omega = omega/omega.sum()
    lambdas = np.random.rand(J).reshape(1, J)
    lambdas = lambdas/lambdas.sum()
    assert np.allclose(lambdas.sum(), 1.0)
    assert np.allclose(omega.sum(), 1.0)

    n = np.random.rand(T * S * J).reshape(T, S, J)
    e = np.tile(np.random.rand(S, J), (T, 1, 1))

    # test matrix multiplication in 3 dimensions works as expected
    L_loop = np.ones(T * S * J).reshape(T, S, J)
    for t in range(T):
        for i in range(S):
            for k in range(J):
                L_loop[t, i, k] *= (omega[i, 0] * lambdas[0, k] *
                                    n[t, i, k] * e[t, i, k])

    L_matrix = e * omega * lambdas * n
    assert (np.allclose(L_loop, L_matrix))

    # test SS
    method = 'SS'
    L = aggr.get_L(n[0], (e[0], omega, lambdas, method))
    assert (np.allclose(L, L_loop[0].sum()))

    # test TPI
    method = 'TPI'
    L = aggr.get_L(n, (e, omega, lambdas, method))
    assert (np.allclose(L, L_loop.sum(1).sum(1)))


def test_get_I():
    """
        Simulate data similar to observed and carry out get_I
        in simplest way possible.
    """
    T = 160
    S, J = 40, 2

    b_splus1 = 10 * np.random.rand(T * S * J).reshape(T, S, J)
    K_p1 = 0.9 + np.random.rand(T)
    K = 0.9 + np.random.rand(T)

    delta = np.random.rand()
    g_y = np.random.rand()

    # make sure array shifting works as expected
    def shifted_arr(normalize=False):
        arr_t = []
        arr_shift_t = []
        for t in range(T):
            arr = np.random.rand(S).reshape(S, 1)
            if normalize:
                arr = arr/arr.sum()
            arr_shift = np.append(arr[1:], [0.0])

            arr_t.append(arr)
            arr_shift_t.append(arr_shift)

        return (np.array(arr_t).reshape(T, S, 1),
                np.array(arr_shift_t).reshape(T, S, 1))

    imm_rates, imm_shift = shifted_arr()
    imm_rates = imm_rates - 0.5
    imm_shift = imm_shift - 0.5
    # normalize across S and J axes
    lambdas = np.random.rand(2)
    lambdas = lambdas/lambdas.sum()
    omega, omega_shift = shifted_arr(normalize=True)
    assert np.allclose(lambdas.sum(), 1.0)
    assert np.allclose(omega.sum(), T)

    g_n = np.random.rand(T)

    res_loop = np.ones(T * S * J).reshape(T, S, J)
    for t in range(T):
        for i in range(S):
            for k in range(J):
                res_loop[t, i, k] *= (omega_shift[t, i, 0] * imm_shift[t, i, 0]
                                      * lambdas[k] * b_splus1[t, i, k])

    res_matrix = (b_splus1 * (imm_shift * omega_shift) * lambdas)

    assert (np.allclose(res_loop, res_matrix))

    # test SS
    aggI_SS_test = ((1 + g_n[0]) * np.exp(g_y) *
                    (K_p1[0] - res_matrix[0].sum() / (1 + g_n[0])) -
                    (1.0 - delta) * K[0])
    aggI_SS = aggr.get_I(b_splus1[0], K_p1[0], K[0],
                         (delta, g_y, omega[0], lambdas, imm_rates[0], g_n[0],
                          'SS'))
    assert (np.allclose(aggI_SS, aggI_SS_test))

    # test TPI
    aggI_TPI_test = ((1 + g_n) * np.exp(g_y) *
                     (K_p1 - res_matrix.sum(1).sum(1) / (1 + g_n)) -
                     (1.0 - delta) * K)
    aggI_TPI = aggr.get_I(b_splus1, K_p1, K,
                          (delta, g_y, omega, lambdas, imm_rates, g_n, 'TPI'))
    assert (np.allclose(aggI_TPI, aggI_TPI_test))


def test_get_K():
    """
    Simulate data similar to observed
    """
    T = 160
    S, J = 40, 2

    b = -0.1 + (7 * np.random.rand(T * S * J).reshape(T, S, J))
    # normalize across S and J axes
    omega = 0.5 * np.random.rand(T * S).reshape(T, S, 1)
    omega = omega/omega.sum(axis=1).reshape(T, 1, 1)
    lambdas = 0.4 + (0.2 * np.random.rand(J).reshape(1, 1, J))
    lambdas = lambdas/lambdas.sum()
    assert np.allclose(lambdas.sum(), 1.0)
    assert np.allclose(omega.sum(), T)

    g_n = 0.1 * np.random.rand(T)
    imm_rates = -0.1 + np.random.rand(T * S * 1).reshape(T, S, 1)

    omega_extended = np.append(omega[:, 1:, :], np.zeros((T, 1, 1)), axis=1)
    imm_extended = np.append(imm_rates[:, 1:, :], np.zeros((T, 1, 1)), axis=1)

    K_test = ((b * omega * lambdas) +
              (b * (omega_extended * imm_extended) * lambdas))
    K = aggr.get_K(b[0], (omega[0], lambdas[0], imm_rates[0], g_n[0], "SS"))

    assert np.allclose(K_test[0].sum()/(1.0 + g_n[0]), K)

    K = aggr.get_K(b, (omega, lambdas, imm_rates, g_n, "TPI"))
    assert np.allclose(K_test.sum(1).sum(1)/(1.0 + g_n), K)


def test_get_BQ():
    """
    Simulate data similar to observed
    """
    T = 160
    S, J = 40, 2

    r = 0.5 + 0.5 * np.random.rand(T).reshape(T, 1)
    b_splus1 = 0.06 + 7 * np.random.rand(T, S, J)
    # normalize across S and J axes
    omega = 0.5 * np.random.rand(T * S).reshape(T, S, 1)
    omega = omega/omega.sum(axis=1).reshape(T, 1, 1)
    lambdas = 0.4 + 0.2 * np.random.rand(J).reshape(1, 1, J)
    lambdas = lambdas/lambdas.sum()
    assert np.allclose(lambdas.sum(), 1.0)
    assert np.allclose(omega.sum(), T)

    rho = np.random.rand(S).reshape(1, S, 1)
    g_n = 0.1 * np.random.rand(T).reshape(T, 1)

    BQ_presum = b_splus1 * omega * rho * lambdas
    factor = (1.0 + r) / (1.0 + g_n)

    # test SS
    BQ = aggr.get_BQ(r[0], b_splus1[0],
                     (omega[0], lambdas[0], rho[0], g_n[0], "SS"))
    assert np.allclose(BQ_presum[0].sum(0) * factor[0], BQ)

    # test TPI
    BQ = aggr.get_BQ(r, b_splus1,
                     (omega, lambdas, rho, g_n, "TPI"))
    assert np.allclose(BQ_presum.sum(1) * factor, BQ)


def test_get_C():
    """
    Simulate data similar to observed
    """
    T = 160
    S, J = 40, 2

    c = 0.1 + 0.5 * np.random.rand(T * S * J).reshape(T, S, J)
    # normalize across S and J axes
    omega = 0.5 * np.random.rand(T * S).reshape(T, S, 1)
    omega = omega/omega.sum(axis=1).reshape(T, 1, 1)
    lambdas = np.random.rand(J)
    lambdas = lambdas/lambdas.sum()
    assert np.allclose(lambdas.sum(), 1.0)
    assert np.allclose(omega.sum(), T)


    aggC_presum = c * omega * lambdas

    # test SS
    aggC = aggr.get_C(c[0], (omega[0], lambdas, "SS"))
    assert np.allclose(aggC_presum[0].sum(), aggC)
    # test TPI
    aggC = aggr.get_C(c, (omega, lambdas, "TPI"))
    assert np.allclose(aggC_presum.sum(1).sum(1), aggC)


def test_revenue():
    """
    Simulate data similar to observed and compare current results with saved
    results
    """
    T = 30
    S, J = 20, 2
    dim4 = 12
    random_state = np.random.RandomState(10)
    r = 0.067 + (0.086 - 0.067) * random_state.rand(T * S * J).reshape(T, S, J)
    w = 0.866 + (0.927 - 0.866) * random_state.rand(T * S * J).reshape(T, S, J)
    b = 6.94 * random_state.rand(T * S * J).reshape(T, S, J)
    n = 0.191 + (0.503 - 0.191) * random_state.rand(T * S * J).reshape(T, S, J)
    BQ = (0.032 + (0.055 - 0.032) *
          random_state.rand(T * S * J).reshape(T, S, J))
    Y = 0.561 + (0.602 - 0.561) * random_state.rand(T).reshape(T)
    L = 0.416 + (0.423 - 0.416) * random_state.rand(T).reshape(T)
    K = 0.957 + (1.163 - 0.957) * random_state.rand(T).reshape(T)
    factor = 140000.0
    e = 0.263 + (2.024 - 0.263) * random_state.rand(T * S * J).reshape(T, S, J)
    # normalize across S and J axes
    lambdas = 0.4 + (0.6 - 0.4) * random_state.rand(1 * 1 * J).reshape(1, 1, J)
    lambdas = lambdas/lambdas.sum()
    omega = 0.039 * random_state.rand(T * S * 1).reshape(T, S, 1)
    omega = omega/omega.sum(axis=1).reshape(T, 1, 1)
    assert np.allclose(lambdas.sum(), 1.0)
    assert np.allclose(omega.sum(), T)

    etr_params = (0.22 *
                  random_state.rand(T * S * J * dim4).reshape(T, S, J, dim4))
    theta = 0.101 + (0.156 - 0.101) * random_state.rand(J)
    tau_bq = random_state.rand(J)
    tau_payroll = 0.5
    h_wealth = 0.1
    p_wealth = 0.2
    m_wealth = 1.0
    retire = 21
    tau_b = 0.2
    delta_tau = 0.0975

    # SS cases
    # case where I.ndim == 2 and etr_params.ndim == 2
    method = "SS"
    params = (e[0], lambdas[0], omega[0], method, etr_params[0, :S, 0, :dim4],
              theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire,
              T, S, J, tau_b, delta_tau)
    res = aggr.revenue(r[0, 0, 0], w[0, 0, 0], b[0], n[0], BQ[0], Y[0], L[0],
                       K[0], factor, params)
    assert(np.allclose(res,  0.62811573441331581))

    # case where I.ndim == 2 and etr_params.ndim == 1
    method = "SS"
    params = (e[0], lambdas[0], omega[0], method, etr_params[0, 0, 0, :dim4],
              theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth,
              retire, T, S, J, tau_b, delta_tau)
    res = aggr.revenue(r[0, 0, 0], w[0, 0, 0], b[0], n[0], BQ[0], Y[0], L[0],
                       K[0], factor, params)
    assert(np.allclose(res,  0.72406672579590448))

    # TPI cases
    # case where I.ndim == 3 and etr_params.ndim == 3
    method = "TPI"
    params = (e, lambdas, omega, method, etr_params[0, :, :, :], theta, tau_bq,
              tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J,
              tau_b, delta_tau)
    res = aggr.revenue(r, w, b, n, BQ, Y, L, K, factor, params)
    test = [0.62360144, 0.74817083, 0.71287424, 0.68285447, 0.64298028,
            0.69488446, 0.70770547, 0.66313781, 0.7175277, 0.64296948,
            0.67107476, 0.69960495, 0.63951371, 0.73104403, 0.68674457,
            0.66307339, 0.66636669, 0.64870362, 0.75359951, 0.68470411,
            0.50771554, 0.71878888, 0.6983747, 0.62996017, 0.67288954,
            0.69745476, 0.64180526, 0.6668633, 0.72454797, 0.71758819]
    assert(np.allclose(res, test))

    # case where I.ndim == 3 and etr_params.ndim == 4
    method = "TPI"
    test = [0.62360144, 0.7705223, 0.71433003, 0.69590516, 0.64187822,
            0.69069099, 0.68437605, 0.66896378, 0.69317402, 0.67131389,
            0.66756797, 0.69466778, 0.64910748, 0.74363875, 0.6986025,
            0.64086681, 0.67091728, 0.65072774, 0.74296341, 0.69073292,
            0.48942517, 0.73170343, 0.69319158, 0.64553276, 0.67911291,
            0.72327757, 0.63002155, 0.68856491, 0.71801762, 0.69659916]
    params = (e, lambdas, omega, method, etr_params, theta, tau_bq,
              tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S, J,
              tau_b, delta_tau)
    res = aggr.revenue(r, w, b, n, BQ, Y, L, K, factor, params)
    assert(np.allclose(res, test))
