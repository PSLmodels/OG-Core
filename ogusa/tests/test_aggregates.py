import pytest
import numpy as np
from ogusa import aggregates as aggr
from ogusa.pb_api import Specifications, reform_warnings_errors

# def test_get_L():
#     """
#         Simulate data similar to observed and carry out get_L
#         in simplest way possible.
#     """
#     p = Specifications()
#     new_param_values = {
#         'T': 160,
#         'S': 40,
#         'J': 2,
#         'lambdas': [0.6, 0.4]
#     }
#     # update parameters instance with new values for test
#     p.update_specifications(new_param_values, raise_errors=False)
#
#     # check that distributional parameters sum to one
#     assert np.allclose(p.lambdas.sum(), 1.0)
#     assert np.allclose(p.omega_SS.sum(), 1.0)
#
#     n = np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
#     # e = np.tile(np.random.rand(S, J), (T, 1, 1))
#
#     # test matrix multiplication in 3 dimensions works as expected
#     L_loop = np.ones(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
#     for t in range(p.T):
#         for i in range(p.S):
#             for k in range(p.J):
#                 L_loop[t, i, k] *= (p.omega[t, i] * p.lambdas[k] *
#                                     n[t, i, k] * p.e[i, k])
#
#     L_matrix = p.e * np.transpose(p.omega_SS * p.lambdas) * n[-1, :, :]
#     assert (np.allclose(L_loop[-1, :, :], L_matrix))
#
#     # test SS
#     method = 'SS'
#     # L = aggr.get_L(n[0], (e[0], omega, lambdas, method))
#     L = aggr.get_L(n[-1, :, :], p, method)
#     assert (np.allclose(L, L_loop[-1, :, :].sum()))
#
#     # test TPI
#     method = 'TPI'
#     # L = aggr.get_L(n, (e, omega, lambdas, method))
#     L = aggr.get_L(n, p, method)
#     assert (np.allclose(L, L_loop.sum(1).sum(1)))
#
#
# def test_get_I():
#     """
#         Simulate data similar to observed and carry out get_I
#         in simplest way possible.
#     """
#     p = Specifications()
#     new_param_values = {
#         'T': 160,
#         'S': 40,
#         'J': 2,
#         'lambdas': [0.6, 0.4],
#     }
#     # update parameters instance with new values for test
#     p.update_specifications(new_param_values, raise_errors=False)
#
#     # assign some values to variables
#     b_splus1 = 10 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
#     K_p1 = 0.9 + np.random.rand(p.T)
#     K = 0.9 + np.random.rand(p.T)
#
#     # check sum of distributional parameters
#     assert np.allclose(p.lambdas.sum(), 1.0)
#     assert np.allclose(p.omega[:p.T, :].sum(), p.T)
#
#     res_loop = np.ones(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
#     for t in range(p.T):
#         for i in range(p.S):
#             for k in range(p.J):
#                 res_loop[t, i, k] *= (p.omega[t, i] * p.imm_rates[t, i]
#                                       * p.lambdas[k] * b_splus1[t, i, k])
#
#     res_matrix = ((b_splus1 * np.squeeze(p.lambdas)) *
#                   np.tile(np.reshape((p.imm_rates[:p.T, :] *
#                                       p.omega[:p.T, :]), (p.T, p.S, 1)),
#                           (1, 1, p.J)))
#     # check that matrix operation and loop return same values
#     assert (np.allclose(res_loop, res_matrix))
#
#     # test SS
#     omega_extended = np.append(p.omega_SS[1:], [0.0])
#     imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
#     part2 = (((b_splus1[-1, :, :] *
#                np.transpose((omega_extended * imm_extended) *
#                p.lambdas)).sum()) / (1 + p.g_n_ss))
#     aggI_SS_test = ((1 + p.g_n_ss) * np.exp(p.g_y) *
#                     (K_p1[-1] - part2) -
#                     (1.0 - p.delta) * K[-1])
#     aggI_SS = aggr.get_I(b_splus1[-1, :, :], K_p1[-1], K[-1], p, 'SS')
#
#     assert (np.allclose(aggI_SS, aggI_SS_test))
#
#     omega_shift = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
#                             axis=1)
#     imm_shift = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
#                           axis=1)
#     part2 = ((((b_splus1 * np.squeeze(p.lambdas)) *
#               np.tile(np.reshape(imm_shift * omega_shift,
#                                  (p.T, p.S, 1)),
#                       (1, 1, p.J))).sum(1).sum(1)) /
#              (1 + np.squeeze(p.g_n[:p.T])))
#     aggI_TPI_test = ((1 + np.squeeze(p.g_n[:p.T])) * np.exp(p.g_y) *
#                      (K_p1 - part2) - (1.0 - p.delta) * K)
#     aggI_TPI = aggr.get_I(b_splus1, K_p1, K, p, 'TPI')
#     assert (np.allclose(aggI_TPI, aggI_TPI_test))
#
#
# def test_get_K():
#     """
#     Simulate data similar to observed
#     """
#     p = Specifications()
#     new_param_values = {
#         'T': 160,
#         'S': 40,
#         'J': 2,
#         'lambdas': [0.6, 0.4],
#     }
#     # update parameters instance with new values for test
#     p.update_specifications(new_param_values, raise_errors=False)
#
#     b = -0.1 + (7 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
#
#     # check that distributional parameters sum to one
#     assert np.allclose(p.lambdas.sum(), 1.0)
#     assert np.allclose(p.omega[:p.T, :].sum(), p.T)
#
#     omega_extended = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
#                                axis=1)
#     imm_extended = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
#                              axis=1)
#
#     K_test = ((b * np.squeeze(p.lambdas) *
#                np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
#                        (1, 1, p.J))) +
#               (b * np.squeeze(p.lambdas) *
#                np.tile(np.reshape(omega_extended *
#                                   imm_extended, (p.T, p.S, 1)),
#                        (1, 1, p.J))))
#     K = aggr.get_K(b[-1, :, :], p, "SS")
#
#     assert np.allclose(K_test[-1, :, :].sum() / (1.0 + p.g_n_ss), K)
#
#     K = aggr.get_K(b, p, "TPI")
#     assert np.allclose(K_test.sum(1).sum(1) /
#                        (1.0 + np.squeeze(p.g_n[:p.T])), K)
#
#
# def test_get_BQ():
#     """
#     Simulate data similar to observed
#     """
#     p = Specifications()
#     new_param_values = {
#         'T': 160,
#         'S': 40,
#         'J': 2,
#         'lambdas': [0.6, 0.4],
#     }
#     # update parameters instance with new values for test
#     p.update_specifications(new_param_values, raise_errors=False)
#
#     # set values for some variables
#     r = 0.5 + 0.5 * np.random.rand(p.T)
#     b_splus1 = 0.06 + 7 * np.random.rand(p.T, p.S, p.J)
#
#     # check that distributional parameters sum to one
#     assert np.allclose(p.lambdas.sum(), 1.0)
#     assert np.allclose(p.omega[:p.T, :].sum(), p.T)
#
#     BQ_presum = ((b_splus1 * np.squeeze(p.lambdas)) *
#                  np.tile(np.reshape(p.rho * p.omega[:p.T, :],
#                                     (p.T, p.S, 1)), (1, 1, p.J)))
#     growth_adj = (1.0 + r) / (1.0 + p.g_n[:p.T])
#
#     # test SS
#     BQ = aggr.get_BQ(r[-1], b_splus1[-1, :, :], None, p, "SS")
#     assert np.allclose(BQ_presum[-1, :, :].sum(0) * growth_adj[-1], BQ)
#
#     # test SS for specific j
#     BQ = aggr.get_BQ(r[-1], b_splus1[-1, :, 1], 1, p, "SS")
#     assert np.allclose(BQ_presum[-1, :, 1].sum(0) * growth_adj[-1], BQ)
#
#     # test TPI
#     BQ = aggr.get_BQ(r, b_splus1, None, p, "TPI")
#     assert np.allclose(BQ_presum.sum(1) *
#                        np.tile(np.reshape(growth_adj, (p.T, 1)),
#                                (1, p.J)), BQ)
#
#     # test TPI for specific j
#     BQ = aggr.get_BQ(r, b_splus1[:, :, 1], 1, p, "TPI")
#     assert np.allclose(BQ_presum[:, :, 1].sum(1) * growth_adj, BQ)
#
#
# def test_get_C():
#     """
#     Simulate data similar to observed
#     """
#     p = Specifications()
#     new_param_values = {
#         'T': 160,
#         'S': 40,
#         'J': 2,
#         'lambdas': [0.6, 0.4],
#     }
#     # update parameters instance with new values for test
#     p.update_specifications(new_param_values, raise_errors=False)
#
#     # make up some consumption values for testing
#     c = 0.1 + 0.5 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
#     aggC_presum = ((c * np.squeeze(p.lambdas)) *
#                    np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
#                            (1, 1, p.J)))
#
#     # check that distributional parameters sum to one
#     assert np.allclose(p.lambdas.sum(), 1.0)
#     assert np.allclose(p.omega[:p.T, :].sum(), p.T)
#
#     # test SS
#     aggC = aggr.get_C(c[-1], p, "SS")
#     assert np.allclose(aggC_presum[-1, :, :].sum(), aggC)
#     # test TPI
#     aggC = aggr.get_C(c, p, "TPI")
#     assert np.allclose(aggC_presum.sum(1).sum(1), aggC)
#

def test_revenue():
    """
    Simulate data similar to observed and compare current results with saved
    results
    """
    p = Specifications()
    dim4 = 12
    random_state = np.random.RandomState(10)
    print('the value of retirement as a default is: ', p.retire)
    new_param_values = {
        'T': 30,
        'S': 20,
        'J': 2,
        'lambdas': [0.6, 0.4],
        'tau_bq': random_state.rand(),
        'tau_payroll': 0.5,
        'h_wealth': 0.1,
        'p_wealth': 0.2,
        'm_wealth': 1.0,
        'tau_b': 0.2,
        'delta_tau_annual': float(1 - ((1 - 0.0975) ** (20 / (p.ending_age - p.starting_age))))
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)
    p.e = 0.263 + (2.024 - 0.263) * random_state.rand(p.S * p.J).reshape(p.S, p.J)
    # test_etr_params = (0.22 * random_state.rand(p.T * p.S * dim4).reshape(p.T, p.S, dim4))

    p.etr_params = (0.22 *
                    random_state.rand(p.T * p.S * dim4).reshape(p.T, p.S,
                                                                dim4))

    p.retire = 21  # do this here because doesn't work with update_specifications because retirement not in the default parameters json

    # Assign values to variables for tests
    r = 0.067 + (0.086 - 0.067) * random_state.rand(p.T)
    w = 0.866 + (0.927 - 0.866) * random_state.rand(p.T)
    b = 6.94 * random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
    n = 0.191 + (0.503 - 0.191) * random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
    BQ = (0.032 + (0.055 - 0.032) *
          random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
    Y = 0.561 + (0.602 - 0.561) * random_state.rand(p.T).reshape(p.T)
    L = 0.416 + (0.423 - 0.416) * random_state.rand(p.T).reshape(p.T)
    K = 0.957 + (1.163 - 0.957) * random_state.rand(p.T).reshape(p.T)
    theta = 0.101 + (0.156 - 0.101) * random_state.rand(p.J)
    factor = 140000.0

    # check that distributional parameters sum to one
    assert np.allclose(p.lambdas.sum(), 1.0)
    assert np.allclose(p.omega[:p.T, :].sum(), p.T)

    # SS cases
    # case where I.ndim == 2 and etr_params.ndim == 2
    method = "SS"
    ## NEED to check calcs by hand on THIS
    ## Also, need to think about how will handle p.etr_params in time path... extrapolate for all T in pb_api?  Probably... and change order there so it's (T, S, params)

    # params = (e[0], lambdas[0], omega[0], method,
    #           etr_params[0, :S, 0, :dim4], 'DEP', theta, tau_bq,
    #           tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S,
    #           J, tau_b, delta_tau)
    # res = aggr.revenue(r[0, 0, 0], w[0, 0, 0], b[0], n[0], BQ[0], Y[0], L[0],
    #                    K[0], factor, params)

    res = aggr.revenue(r[0], w[0], b[0, :, :], n[0, :, :], BQ[0, :, :], Y[0], L[0],
                       K[0], factor, theta, p, method)
    print('Result 1 = ', res)
    # assert(np.allclose(res,  0.5975982884926914))

    # case where I.ndim == 2 and etr_params.ndim == 1
    method = "SS"
    # params = (p.e[0], lambdas[0], omega[0], method,
    #           etr_params[0, 0, 0, :dim4], 'DEP', theta, tau_bq,
    #           tau_payroll, h_wealth, p_wealth, m_wealth, retire, T, S,
    #           J, tau_b, delta_tau)
    # res = aggr.revenue(r[0, 0, 0], w[0, 0, 0], b[0], n[0], BQ[0], Y[0], L[0],
    #                    K[0], factor, params)
    res = aggr.revenue(r[0], w[0], b[0, :, :], n[0, :, :], BQ[0, :, :], Y[0], L[0],
                       K[0], factor, theta, p, method)
    print('Result 2 = ', res)
    # assert(np.allclose(res,  0.72406672579590448))

    # TPI cases
    # case where I.ndim == 3 and etr_params.ndim == 3
    method = "TPI"
    # params = (e, lambdas, omega, method, etr_params[0, :, :, :], 'DEP',
    #           theta, tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth,
    #           retire, T, S, J, tau_b, delta_tau)
    # res = aggr.revenue(r, w, b, n, BQ, Y, L, K, factor, params)
    res = aggr.revenue(r, w, b, n, BQ, Y, L,
                       K, factor, theta, p, method)
    test = [0.62360144, 0.74817083, 0.71287424, 0.68285447, 0.64298028,
            0.69488446, 0.70770547, 0.66313781, 0.7175277, 0.64296948,
            0.67107476, 0.69960495, 0.63951371, 0.73104403, 0.68674457,
            0.66307339, 0.66636669, 0.64870362, 0.75359951, 0.68470411,
            0.50771554, 0.71878888, 0.6983747, 0.62996017, 0.67288954,
            0.69745476, 0.64180526, 0.6668633, 0.72454797, 0.71758819]
    print('Result 3 = ', res)
    # assert(np.allclose(res, test))

    # case where I.ndim == 3 and etr_params.ndim == 4
    method = "TPI"
    test = [0.62360144, 0.7705223, 0.71433003, 0.69590516, 0.64187822,
            0.69069099, 0.68437605, 0.66896378, 0.69317402, 0.67131389,
            0.66756797, 0.69466778, 0.64910748, 0.74363875, 0.6986025,
            0.64086681, 0.67091728, 0.65072774, 0.74296341, 0.69073292,
            0.48942517, 0.73170343, 0.69319158, 0.64553276, 0.67911291,
            0.72327757, 0.63002155, 0.68856491, 0.71801762, 0.69659916]
    # params = (e, lambdas, omega, method, etr_params, 'DEP', theta,
    #           tau_bq, tau_payroll, h_wealth, p_wealth, m_wealth, retire,
    #           T, S, J, tau_b, delta_tau)
    # res = aggr.revenue(r, w, b, n, BQ, Y, L, K, factor, params)
    res = aggr.revenue(r, w, b, n, BQ, Y, L,
                       K, factor, theta, p, method)
    print('Result 4 = ', res)
    assert(np.allclose(res, test))
