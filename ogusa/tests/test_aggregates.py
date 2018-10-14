import pytest
import numpy as np
from ogusa import aggregates as aggr
from ogusa.pb_api import Specifications


def test_get_L():
    """
        Simulate data similar to observed and carry out get_L
        in simplest way possible.
    """
    p = Specifications()
    new_param_values = {
        'T': 160,
        'S': 40,
        'J': 2,
        'lambdas': [0.6, 0.4]
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    # check that distributional parameters sum to one
    assert np.allclose(p.lambdas.sum(), 1.0)
    assert np.allclose(p.omega_SS.sum(), 1.0)

    n = np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)

    # test matrix multiplication in 3 dimensions works as expected
    L_loop = np.ones(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
    for t in range(p.T):
        for i in range(p.S):
            for k in range(p.J):
                L_loop[t, i, k] *= (p.omega[t, i] * p.lambdas[k] *
                                    n[t, i, k] * p.e[i, k])

    L_matrix = p.e * np.transpose(p.omega_SS * p.lambdas) * n[-1, :, :]
    assert (np.allclose(L_loop[-1, :, :], L_matrix))

    # test SS
    method = 'SS'
    L = aggr.get_L(n[-1, :, :], p, method)
    assert (np.allclose(L, L_loop[-1, :, :].sum()))

    # test TPI
    method = 'TPI'
    L = aggr.get_L(n, p, method)
    assert (np.allclose(L, L_loop.sum(1).sum(1)))


def test_get_I():
    """
        Simulate data similar to observed and carry out get_I
        in simplest way possible.
    """
    p = Specifications()
    new_param_values = {
        'T': 160,
        'S': 40,
        'J': 2,
        'lambdas': [0.6, 0.4],
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    # assign some values to variables
    b_splus1 = 10 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
    K_p1 = 0.9 + np.random.rand(p.T)
    K = 0.9 + np.random.rand(p.T)

    # check sum of distributional parameters
    assert np.allclose(p.lambdas.sum(), 1.0)
    assert np.allclose(p.omega[:p.T, :].sum(), p.T)

    res_loop = np.ones(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
    for t in range(p.T):
        for i in range(p.S):
            for k in range(p.J):
                res_loop[t, i, k] *= (p.omega[t, i] * p.imm_rates[t, i]
                                      * p.lambdas[k] * b_splus1[t, i, k])

    res_matrix = ((b_splus1 * np.squeeze(p.lambdas)) *
                  np.tile(np.reshape((p.imm_rates[:p.T, :] *
                                      p.omega[:p.T, :]), (p.T, p.S, 1)),
                          (1, 1, p.J)))
    # check that matrix operation and loop return same values
    assert (np.allclose(res_loop, res_matrix))

    # test SS
    omega_extended = np.append(p.omega_SS[1:], [0.0])
    imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
    part2 = (((b_splus1[-1, :, :] *
               np.transpose((omega_extended * imm_extended) *
               p.lambdas)).sum()) / (1 + p.g_n_ss))
    aggI_SS_test = ((1 + p.g_n_ss) * np.exp(p.g_y) *
                    (K_p1[-1] - part2) -
                    (1.0 - p.delta) * K[-1])
    aggI_SS = aggr.get_I(b_splus1[-1, :, :], K_p1[-1], K[-1], p, 'SS')

    assert (np.allclose(aggI_SS, aggI_SS_test))

    omega_shift = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
                            axis=1)
    imm_shift = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
                          axis=1)
    part2 = ((((b_splus1 * np.squeeze(p.lambdas)) *
              np.tile(np.reshape(imm_shift * omega_shift,
                                 (p.T, p.S, 1)),
                      (1, 1, p.J))).sum(1).sum(1)) /
             (1 + np.squeeze(np.hstack((p.g_n[1:p.T], p.g_n_ss)))))
    aggI_TPI_test = ((1 + np.squeeze(np.hstack((p.g_n[1:p.T], p.g_n_ss))))
                     * np.exp(p.g_y) * (K_p1 - part2) - (1.0 - p.delta)
                     * K)
    aggI_TPI = aggr.get_I(b_splus1, K_p1, K, p, 'TPI')
    assert (np.allclose(aggI_TPI, aggI_TPI_test))


def test_get_K():
    """
    Simulate data similar to observed
    """
    p = Specifications()
    new_param_values = {
        'T': 160,
        'S': 40,
        'J': 2,
        'lambdas': [0.6, 0.4],
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    b = -0.1 + (7 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))

    # check that distributional parameters sum to one
    assert np.allclose(p.lambdas.sum(), 1.0)
    assert np.allclose(p.omega[:p.T, :].sum(), p.T)

    omega_extended = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
                               axis=1)
    imm_extended = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
                             axis=1)

    K_test = ((b * np.squeeze(p.lambdas) *
               np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                       (1, 1, p.J))) +
              (b * np.squeeze(p.lambdas) *
               np.tile(np.reshape(omega_extended *
                                  imm_extended, (p.T, p.S, 1)),
                       (1, 1, p.J))))
    K = aggr.get_K(b[-1, :, :], p, "SS", False)

    assert np.allclose(K_test[-1, :, :].sum() / (1.0 + p.g_n_ss), K)

    K = aggr.get_K(b, p, "TPI", False)
    assert np.allclose(K_test.sum(1).sum(1) /
                       (1.0 + np.hstack((p.g_n[1:p.T], p.g_n_ss))), K)


def test_get_BQ():
    """
    Simulate data similar to observed
    """
    p = Specifications()
    new_param_values = {
        'T': 160,
        'S': 40,
        'J': 2,
        'lambdas': [0.6, 0.4],
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    # set values for some variables
    r = 0.5 + 0.5 * np.random.rand(p.T)
    b_splus1 = 0.06 + 7 * np.random.rand(p.T, p.S, p.J)

    # check that distributional parameters sum to one
    assert np.allclose(p.lambdas.sum(), 1.0)
    assert np.allclose(p.omega[:p.T, :].sum(), p.T)

    pop = np.append(p.omega_S_preTP.reshape(1, p.S),
                    p.omega[:p.T - 1, :], axis=0)
    BQ_presum = ((b_splus1 * np.squeeze(p.lambdas)) *
                 np.tile(np.reshape(p.rho * pop, (p.T, p.S, 1)),
                         (1, 1, p.J)))
    growth_adj = (1.0 + r) / (1.0 + p.g_n[:p.T])

    # test SS
    BQ = aggr.get_BQ(r[-1], b_splus1[-1, :, :], None, p, "SS", False)
    assert np.allclose(BQ_presum[-1, :, :].sum(0) * growth_adj[-1], BQ)

    # test SS for specific j
    BQ = aggr.get_BQ(r[-1], b_splus1[-1, :, 1], 1, p, "SS", False)
    assert np.allclose(BQ_presum[-1, :, 1].sum(0) * growth_adj[-1], BQ)

    # test TPI
    BQ = aggr.get_BQ(r, b_splus1, None, p, "TPI", False)
    assert np.allclose(BQ_presum.sum(1) *
                       np.tile(np.reshape(growth_adj, (p.T, 1)),
                               (1, p.J)), BQ)

    # test TPI for specific j
    BQ = aggr.get_BQ(r, b_splus1[:, :, 1], 1, p, "TPI", False)
    assert np.allclose(BQ_presum[:, :, 1].sum(1) * growth_adj, BQ)


def test_get_C():
    """
    Simulate data similar to observed
    """
    p = Specifications()
    new_param_values = {
        'T': 160,
        'S': 40,
        'J': 2,
        'lambdas': [0.6, 0.4],
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)

    # make up some consumption values for testing
    c = 0.1 + 0.5 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
    aggC_presum = ((c * np.squeeze(p.lambdas)) *
                   np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                           (1, 1, p.J)))

    # check that distributional parameters sum to one
    assert np.allclose(p.lambdas.sum(), 1.0)
    assert np.allclose(p.omega[:p.T, :].sum(), p.T)

    # test SS
    aggC = aggr.get_C(c[-1], p, "SS")
    assert np.allclose(aggC_presum[-1, :, :].sum(), aggC)
    # test TPI
    aggC = aggr.get_C(c, p, "TPI")
    assert np.allclose(aggC_presum.sum(1).sum(1), aggC)


def test_revenue():
    """
    Simulate data similar to observed and compare current results with saved
    results
    """
    p = Specifications()
    dim4 = 12

    new_param_values = {
        'T': 30,
        'S': 20,
        'J': 2,
        'lambdas': [0.6, 0.4],
        'tau_bq': 0.17,
        'tau_payroll': 0.5,
        'h_wealth': 0.1,
        'p_wealth': 0.2,
        'm_wealth': 1.0,
        'tau_b': 0.2,
        'delta_tau_annual': float(1 - ((1 - 0.0975) **
                                       (20 / (p.ending_age -
                                              p.starting_age))))
    }
    p.update_specifications(new_param_values, raise_errors=False)

    # Assign values to variables for tests
    random_state = np.random.RandomState(10)
    r = 0.067 + (0.086 - 0.067) * random_state.rand(p.T)
    w = 0.866 + (0.927 - 0.866) * random_state.rand(p.T)
    b = 6.94 * random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
    n = (0.191 + (0.503 - 0.191) *
         random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
    BQ = (0.032 + (0.055 - 0.032) *
          random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
    Y = 0.561 + (0.602 - 0.561) * random_state.rand(p.T).reshape(p.T)
    L = 0.416 + (0.423 - 0.416) * random_state.rand(p.T).reshape(p.T)
    K = 0.957 + (1.163 - 0.957) * random_state.rand(p.T).reshape(p.T)
    factor = 140000.0

    # update parameters instance with new values for test
    p.e = (0.263 + (2.024 - 0.263) *
           random_state.rand(p.S * p.J).reshape(p.S, p.J))
    p.omega = 0.039 * random_state.rand(p.T * p.S * 1).reshape(p.T, p.S)
    p.omega = p.omega/p.omega.sum(axis=1).reshape(p.T, 1)
    p.omega_SS = p.omega[-1, :]

    etr_params = (0.22 *
                  random_state.rand(p.T * p.S * dim4).reshape(p.T, p.S,
                                                              dim4))
    etr_params = np.tile(np.reshape(etr_params, (p.T, p.S, 1, dim4)),
                         (1, 1, p.J, 1))
    theta = 0.101 + (0.156 - 0.101) * random_state.rand(p.J)

    # check that distributional parameters sum to one
    assert np.allclose(p.lambdas.sum(), 1.0)
    assert np.allclose(p.omega[:p.T, :].sum(), p.T)

    # SS case
    method = "SS"
    res = aggr.revenue(r[0], w[0], b[0, :, :], n[0, :, :], BQ[0, :, :],
                       Y[0], L[0], K[0], factor, theta,
                       etr_params[-1, :, :, :], p, method)
    assert(np.allclose(res,  0.5562489534339288))

    # TPI case
    method = "TPI"
    res = aggr.revenue(r, w, b, n, BQ, Y, L, K, factor, theta,
                       etr_params, p, method)
    test = [0.52178543, 0.49977116, 0.52015768, 0.52693363, 0.59695398,
            0.61360011, 0.54679056, 0.54096669, 0.56301133, 0.5729165,
            0.52734917, 0.51432562, 0.50060814, 0.5633982,  0.51509517,
            0.60189683, 0.56766507, 0.56439768, 0.68919173, 0.57765917,
            0.60292137, 0.56621788, 0.51913478, 0.48952262, 0.52142782,
            0.5735005, 0.51166718, 0.57939994, 0.52585236, 0.53767652]
    assert(np.allclose(res, test))
