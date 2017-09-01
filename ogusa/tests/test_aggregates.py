import pytest
import numpy as np
import pandas as pd
import os

from ogusa import aggregates as aggr

CURR_PATH = os.path.abspath(os.path.dirname(__file__))


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

    # test matrix multiplication in 3 dimensions works as expected
    L_loop = np.ones(T * s * j).reshape(T, s, j)
    for t in range(T):
        for i in range(s):
            for k in range(j):
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
    s, j = 40, 2

    b_splus1 = 10 * np.random.rand(T * s * j).reshape(T, s, j)
    K_p1 = 0.9 + np.random.rand(T)
    K = 0.9 + np.random.rand(T)

    delta = np.random.rand()
    g_y = np.random.rand()

    # make sure array shifting works as expected
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
    s, j = 40, 2

    b = -0.1 + (7 * np.random.rand(T * s * j).reshape(T, s, j))
    omega = 0.5 * np.random.rand(T * s).reshape(T, s, 1)
    lambdas = 0.4 + (0.2 * np.random.rand(j).reshape(1, 1, j))
    g_n = 0.1 * np.random.rand(T)
    imm_rates = -0.1 + np.random.rand(T * s * 1).reshape(T, s, 1)

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
    s, j = 40, 2

    r = 0.5 + 0.5 * np.random.rand(T).reshape(T, 1)
    b_splus1 = 0.06 + 7 * np.random.rand(T, s, j)
    omega = 0.5 * np.random.rand(T, s, 1)
    lambdas = 0.4 + 0.2 * np.random.rand(j).reshape(1,1,j)
    rho = np.random.rand(s).reshape(1,s,1)
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
    s, j = 40, 2

    c = 0.1 + 0.5 * np.random.rand(T * s * j).reshape(T, s, j)
    omega = 0.5 * np.random.rand(T * s).reshape(T, s, 1)
    lambdas = np.random.rand(j)

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
    T = 160
    s, j = 40, 2
    dim4 = 12
    # need to reset seed everyt time np.rand.random function is called
    random_state = np.random.RandomState(10)
    r = 0.067 + (0.086 - 0.067) * random_state.rand(T * s * j).reshape(T, s, j)
    w = 0.866 + (0.927 - 0.866) * random_state.rand(T * s * j).reshape(T, s, j)
    b = 0.0 + (6.94 - 0.0) * random_state.rand(T * s * j).reshape(T, s, j)
    n = 0.191 + (0.503 - 0.191) * random_state.rand(T * s * j).reshape(T, s, j)
    BQ = 0.032 + (0.055 - 0.032) * random_state.rand(T * s * j).reshape(T, s, j)
    Y = 0.561 + (0.602 - 0.561) * random_state.rand(T).reshape(T)
    L = 0.416 + (0.423 - 0.416) * random_state.rand(T).reshape(T)
    K = 0.957 + (1.163 - 0.957) * random_state.rand(T).reshape(T)
    factor = 140000.0
    e = 0.263 + (2.024 - 0.263) * random_state.rand(T * s * j).reshape(T, s, j)
    lambdas = 0.4 + (0.6 - 0.4) * random_state.rand(1 * 1 * j).reshape(1, 1, j)
    omega = 0.0 + (0.039 - 0.0) * random_state.rand(T * s * 1).reshape(T, s, 1)
    etr_params = (0.0 + (0.22 - 0.0) *
                  random_state.rand(T * s * j * dim4).reshape(T, s, j, dim4))
    theta = 0.101 + (0.156 - 0.101) * random_state.rand(j)
    tau_bq = random_state.rand(j)
    tau_payroll = 0.5
    h_wealth = 0.1
    p_wealth = 0.2
    m_wealth = 1.0
    retire = 21
    tau_b = 0.2
    delta_tau = 0.0975

    results_path = os.path.join(CURR_PATH, 'test_revenue_result.csv')

    # SS cases
    # case where I.ndim == 2 and etr_params.ndim == 2
    method = "SS"
    params = (e[0], lambdas[0], omega[0], method, etr_params[0, :s, 0, :dim4],
              theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire,
              T, s, j, tau_b, delta_tau)
    res = aggr.revenue(r[0, 0, 0], w[0, 0, 0], b[0], n[0], BQ[0], Y[0], L[0],
                       K[0], factor, params)
    # assert(np.allclose(res,  0.48262471425641323))

    # case where I.ndim == 3 and etr_params.ndim == 1
    method = "SS"
    params = (e[0], lambdas[0], omega[0], method, etr_params[0, 0, 0, :dim4],
              theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth,
              retire, T, s, j, tau_b, delta_tau)
    res = aggr.revenue(r[0, 0, 0], w[0, 0, 0], b[0], n[0], BQ[0], Y[0], L[0],
                       K[0], factor, params)
    assert(np.allclose(res,  0.44215216429920967))

    df = pd.read_csv(results_path)

    # TPI cases
    # case where I.ndim == 3 and etr_params.ndim == 3
    method = "TPI"
    params = (e, lambdas, omega, method, etr_params[0, :, :, :], theta, tau_bq,
              tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, s, j,
              tau_b, delta_tau)
    res0 = aggr.revenue(r, w, b, n, BQ, Y, L, K, factor, params)
    assert(np.allclose(res0, df.revenue_result_0))

    # case where I.ndim == 3 and etr_params.ndim == 4
    method = "TPI"
    params = (e, lambdas, omega, method, etr_params, theta, tau_bq,
              tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, s, j,
              tau_b, delta_tau)
    res1 = aggr.revenue(r, w, b, n, BQ, Y, L, K, factor, params)
    assert(np.allclose(df.revenue_result_1, res1))

    # if we update the simulated data, uncomment this part to create new test
    # result data
    # df = pd.DataFrame(np.array([res0, res1]).T, columns=['revenue_result_0',
    #                                                      'revenue_result_1'])
    # df.to_csv(results_path, print_index=False)
