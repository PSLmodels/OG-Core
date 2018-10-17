import numpy as np
import pytest
from ogusa import tax
from ogusa.pb_api import Specifications


def test_replacement_rate_vals():
    # Test replacement rate function, making sure to trigger all three
    # cases of AIME
    p = Specifications()
    new_param_values = {
        'S': 4,
        'J': 1,
    }
    p.update_specifications(new_param_values, raise_errors=False)

    # Use just a column of e
    p.e = np.transpose(np.array([[0.1, 0.3, 0.5, 0.2], [0.1, 0.3, 0.5, 0.2]]))
    p.retire = 3
    nssmat = np.array([0.5, 0.5, 0.5, 0.5])
    wss = 0.5
    factor_ss = 100000
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, 0, p)
    assert np.allclose(theta, np.array([0.042012]))

    # e has two dimensions
    nssmat = np.array([[0.4, 0.4], [0.4, 0.4], [0.4, 0.4], [0.4, 0.4]])
    p.e = np.array([[0.4, 0.3], [0.5, 0.4], [.6, .4], [.4, .3]])
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, None, p)
    assert np.allclose(theta, np.array([0.042012, 0.03842772]))

    # hit AIME case2
    nssmat = np.array([[0.3, .35], [0.3, .35], [0.3, .35], [0.3, .35]])
    factor_ss = 10000
    p.e = np.array([[0.35, 0.3], [0.55, 0.4], [.65, .4], [.45, .3]])
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, None, p)
    assert np.allclose(theta, np.array([0.1145304, 0.0969304]))

    # hit AIME case1
    factor_ss = 1000
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, None, p)
    assert np.allclose(theta, np.array([0.1755, 0.126]))


def test_ETR_wealth():
    # Test wealth tax computation
    p = Specifications()
    new_param_values = {
        'h_wealth': 2,
        'p_wealth': 3,
        'm_wealth': 4
    }
    p.update_specifications(new_param_values, raise_errors=False)
    b = np.array([0.1, 0.5, 0.9])
    tau_w_prime = tax.ETR_wealth(b, p)

    assert np.allclose(tau_w_prime, np.array([0.14285714, 0.6, 0.93103448]))


def test_MTR_wealth():
    # Test marginal tax rate on wealth
    p = Specifications()
    new_param_values = {
        'h_wealth': 3,
        'p_wealth': 4,
        'm_wealth': 5
    }
    p.update_specifications(new_param_values, raise_errors=False)
    b = np.array([0.2, 0.6, 0.8])
    tau_w_prime = tax.MTR_wealth(b, p)

    assert np.allclose(tau_w_prime, np.array([1.91326531, 1.29757785,
                                              1.09569028]))


p1 = Specifications()
p1.S = 2
p1.J = 1
p1.e = np.array([0.5, 0.45])
p1.tax_func_type = 'DEP'
etr_params1 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15, 0.16,
     -0.15, 0.83], [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15,
                    0.15, 0.16, -0.15, 0.83]]), (1, p1.S, 12))

p2 = Specifications()
p2.S = 2
p2.J = 1
p2.e = np.array([0.5, 0.45])
p2.tax_func_type = 'GS'
etr_params2 = np.reshape(np.array([
    [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), (1, p2.S, 12))

p3 = Specifications()
p3.S = 2
p3.J = 1
p3.e = np.array([0.5, 0.45])
p3.tax_func_type = 'DEP_totalinc'
etr_params3 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15, 0.16,
     -0.15, 0.83], [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                    -0.15, 0.15, 0.16, -0.15, 0.83]]), (1, p3.S, 12))

p4 = Specifications()
p4.S = 3
p4.J = 1
p4.e = np.array([0.5, 0.45, 0.3])
p4.tax_func_type = 'DEP'
etr_params4 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15, 0.16,
     -0.15, 0.83],
    [0.002, 0.001, 0.002, 0.04, 0.8, -0.14, 0.8, -0.15, 0.15, 0.16,
     -0.15, 0.83],
    [0.011, 0.001, 0.003, 0.06, 0.8, -0.14, 0.8, -0.15, 0.15, 0.16,
     -0.15, 0.83]]), (1, p4.S, 12))


@pytest.mark.parametrize('b,n,etr_params,params,expected',
                         [(np.array([0.4, 0.4]), np.array([0.5, 0.4]),
                           etr_params1, p1,
                           np.array([0.80167091, 0.80167011])),
                          (np.array([0.4, 0.4]), np.array([0.5, 0.4]),
                           etr_params2, p2,
                           np.array([0.395985449, 0.395980186])),
                          (np.array([0.4, 0.4]), np.array([0.5, 0.4]),
                           etr_params3, p3,
                           np.array([0.799999059, 0.799998254])),
                          (np.reshape(np.array([0.4, 0.3, 0.5]), (3)),
                           np.reshape(np.array([0.8, 0.4, 0.7]), (3)),
                           etr_params4, p4,
                           np.array([0.80167144, 0.80163711, 0.8016793]))
                          ],
                         ids=['DEP', 'GS', 'DEP_totalinc',
                              'DEP, >1 dim'])
def test_ETR_income(b, n, etr_params, params, expected):
    # Test income tax function
    r = 0.04
    w = 1.2
    factor = 100000
    # test_ETR_income = tax.ETR_income(r, w, b, n, factor,
    #                                  (e, etr_params, tax_func_type))
    test_ETR_income = tax.ETR_income(r, w, b, n, factor, params.e,
                                     etr_params, params)
    assert np.allclose(test_ETR_income, expected)


p1 = Specifications()
p1.e = np.array([0.5, 0.45, 0.3])
p1.S = 3
p1.J = 1
p1.tax_func_type = 'DEP'
p1.analytical_mtrs = True
etr_params1 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8,
     -0.14, 0.8, -0.15, 0.15, 0.16, -0.15,
     0.83],
    [0.002, 0.001, 0.002, 0.04, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83],
    [0.011, 0.001, 0.003, 0.06, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83]]), (1, p1.S, 12))
mtrx_params1 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.68,
     -0.17, 0.8, -0.42, 0.18, 0.43, -0.42,
     0.96],
    [0.001, 0.002, 0.003, 0.0015, 0.65,
     -0.17, 0.8, -0.42, 0.18, 0.33, -0.12,
     0.90],
    [0.001, 0.002, 0.003, 0.0015, 0.56,
     -0.17, 0.8, -0.42, 0.18, 0.38, -0.22,
     0.65]]), (1, p1.S, 12))

p2 = Specifications()
p2.e = np.array([0.5, 0.45, 0.3])
p2.S = 3
p2.J = 1
p2.tax_func_type = 'DEP'
p2.analytical_mtrs = True
etr_params2 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8,
     -0.14, 0.8, -0.15, 0.15, 0.16, -0.15,
     0.83],
    [0.002, 0.001, 0.002, 0.04, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83],
    [0.011, 0.001, 0.003, 0.06, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83]]), (1, p2.S, 12))
mtry_params2 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.68,
     -0.17, 0.8, -0.42, 0.18, 0.43, -0.42,
     0.96],
    [0.001, 0.002, 0.003, 0.0015, 0.65,
     -0.17, 0.8, -0.42, 0.18, 0.33, -0.12,
     0.90],
    [0.001, 0.002, 0.003, 0.0015, 0.56,
     -0.17, 0.8, -0.42, 0.18, 0.38, -0.22,
     0.65]]), (1, p2.S, 12))

p3 = Specifications()
p3.e = np.array([0.5, 0.45, 0.3])
p3.S = 3
p3.J = 1
p3.tax_func_type = 'DEP'
p3.analytical_mtrs = False
etr_params3 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8,
     -0.14, 0.8, -0.15, 0.15, 0.16, -0.15,
     0.83],
    [0.002, 0.001, 0.002, 0.04, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83],
    [0.011, 0.001, 0.003, 0.06, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83]]), (1, p3.S, 12))
mtrx_params3 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.68,
     -0.17, 0.8, -0.42, 0.18, 0.43, -0.42,
     0.96],
    [0.001, 0.002, 0.003, 0.0015, 0.65,
     -0.17, 0.8, -0.42, 0.18, 0.33, -0.12,
     0.90],
    [0.001, 0.002, 0.003, 0.0015, 0.56,
     -0.17, 0.8, -0.42, 0.18, 0.38, -0.22,
     0.65]]), (1, p3.S, 12))

p4 = Specifications()
p4.e = np.array([0.5, 0.45, 0.3])
p4.S = 3
p4.J = 1
p4.tax_func_type = 'GS'
p4.analytical_mtrs = False
etr_params4 = np.reshape(np.array([
    [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0,
     0],
    [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0,
     0],
    [0.6, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0, 0,
     0]]), (1, p4.S, 12))
mtrx_params4 = np.reshape(np.array([
    [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0,
     0],
    [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0,
     0],
    [0.6, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0, 0,
     0]]), (1, p4.S, 12))

p5 = Specifications()
p5.e = np.array([0.5, 0.45, 0.3])
p5.S = 3
p5.J = 1
p5.tax_func_type = 'DEP_totalinc'
p5.analytical_mtrs = True
etr_params5 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8,
     -0.14, 0.8, -0.15, 0.15, 0.16, -0.15,
     0.83],
    [0.002, 0.001, 0.002, 0.04, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83],
    [0.011, 0.001, 0.003, 0.06, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83]]), (1, p5.S, 12))
mtrx_params5 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.68,
     -0.17, 0.8, -0.42, 0.18, 0.43, -0.42,
     0.96],
    [0.001, 0.002, 0.003, 0.0015, 0.65,
     -0.17, 0.8, -0.42, 0.18, 0.33, -0.12,
     0.90],
    [0.001, 0.002, 0.003, 0.0015, 0.56,
     -0.17, 0.8, -0.42, 0.18, 0.38, -0.22,
     0.65]]), (1, p5.S, 12))

p6 = Specifications()
p6.e = np.array([0.5, 0.45, 0.3])
p6.S = 3
p6.J = 1
p6.tax_func_type = 'DEP_totalinc'
p6.analytical_mtrs = False
etr_params6 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8,
     -0.14, 0.8, -0.15, 0.15, 0.16, -0.15,
     0.83],
    [0.002, 0.001, 0.002, 0.04, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83],
    [0.011, 0.001, 0.003, 0.06, 0.8, -0.14,
     0.8, -0.15, 0.15, 0.16, -0.15, 0.83]]), (1, p6.S, 12))
mtrx_params6 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.68,
     -0.17, 0.8, -0.42, 0.18, 0.43, -0.42,
     0.96],
    [0.001, 0.002, 0.003, 0.0015, 0.65,
     -0.17, 0.8, -0.42, 0.18, 0.33, -0.12,
     0.90],
    [0.001, 0.002, 0.003, 0.0015, 0.56,
     -0.17, 0.8, -0.42, 0.18, 0.38, -0.22,
     0.65]]), (1, p6.S, 12))


@pytest.mark.parametrize('etr_params,mtr_params,params,mtr_capital,expected',
                         [(etr_params1, mtrx_params1, p1, False,
                           np.array([0.801675428, 0.801647645,
                                     0.801681744])),
                          (etr_params2, mtry_params2, p2, True,
                           np.array([0.8027427, 0.80335305,
                                     0.80197745])),
                          (etr_params3, mtrx_params3, p3, False,
                           np.array([0.45239409, 0.73598958,
                                     0.65126073])),
                          (etr_params4, mtrx_params4, p4, False,
                           np.array([0.395999995, 0.395999983,
                                     0.599999478])),
                          (etr_params5, mtrx_params5, p5, False,
                           np.array([0.800001028, 0.800002432,
                                     0.800000311])),
                          (etr_params6, mtrx_params6, p6, False,
                           np.array([0.439999714, 0.709998696,
                                     0.519999185]))],
                         ids=['DEP, analytical mtr, labor income',
                              'DEP, analytical mtr, capital income',
                              'DEP, not analytical mtr', 'GS',
                              'DEP_totalinc, analytical mtr',
                              'DEP_totalinc, not analytical mtr'])
def test_MTR_income(etr_params, mtr_params, params, mtr_capital, expected):
    # Test the MTR on income function
    r = 0.04
    w = 1.2
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    factor = 110000

    test_mtr = tax.MTR_income(r, w, b, n, factor, mtr_capital,
                              params.e, etr_params, mtr_params, params)
    assert np.allclose(test_mtr, expected)


def test_get_biz_tax():
    # Test function for business tax receipts
    w = np.array([1.2, 1.1])
    Y = np.array([3.0, 7.0])
    L = np.array([2.0, 3.0])
    K = np.array([5.0, 6.0])
    p = Specifications()
    p.tau_b = 0.20
    p.delta_tau = 0.06
    biz_tax = tax.get_biz_tax(w, Y, L, K, p)
    assert np.allclose(biz_tax, np.array([0.06, 0.668]))


def test_total_taxes():
    # Test function that computes total net taxes for the household
    p = Specifications()
    p.tax_func_type = 'DEP'
    p.J = 1
    p.S = 3
    p.lambdas = np.array([1.0])
    p.e = np.array([0.5, 0.45, 0.3]).reshape(3, 1)
    etr_params = np.reshape(np.array([
        [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
         0.16, -0.15, 0.83],
        [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
         0.16, -0.15, 0.83],
        [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
         0.16, -0.15, 0.83]]), (1, p.S, 12))

    p.h_wealth = 1
    p.p_wealth = 2
    p.m_wealth = 3
    p.tau_payroll = 0.15
    p.tau_bq = np.array([0.1])
    p.retire = 2
    r = 0.04
    w = 1.2
    factor = 105000
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    BQ = np.array([0.3])
    T_H = np.array([0.12])
    theta = np.array([0.225])
    j = 0
    shift = True

    # method = ss
    method = 'SS'
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, theta, j,
                                  shift, method, p.e[:, j], p.retire,
                                  etr_params[-1, :, :], p)
    assert np.allclose(total_taxes, np.array([0.47374766, -0.09027663,
                                              0.03871394]))

    # method = TPI_scalar
    method = 'TPI_scalar'
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, theta, j,
                                  shift, method, p.e[:, j], p.retire,
                                  etr_params, p)
    assert np.allclose(total_taxes, np.array([0.20374766, -0.09027663,
                                              0.03871394]))

    # method = TPI
    method = 'TPI'
    shift = True
    p.T = 3
    r = np.array([0.04, 0.045, 0.04])
    w = np.array([1.2, 1.3, 1.1])
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    BQ = np.array([0.3, 0.4, 0.45])
    T_H = np.array([0.12, 0.1, 0.11])
    etr_params = np.array([[0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                           -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83]])
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, theta, j,
                                  shift, method, p.e[:, j], p.retire,
                                  etr_params, p)
    assert np.allclose(total_taxes,
                       np.array([0.47374766, -0.06444251, 0.06622862]))

    # shift = False
    method = 'TPI'
    shift = False
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, theta, j,
                                  shift, method, p.e[:, j], p.retire,
                                  etr_params, p)
    assert np.allclose(total_taxes,
                       np.array([0.47374766, 0.22805749, 0.06622862]))

    # b.shape =3
    b = np.array([[[0.2, 0.3], [0.3, 0.35], [0.4, 0.35]],
                  [[0.4, 0.3], [0.4, 0.35], [0.5, 0.35]],
                  [[0.6, 0.4], [0.3, 0.4], [0.4, 0.22]]])
    n = np.array([[[0.6, 0.5], [0.5, 0.55], [0.7, 0.8]],
                  [[0.4, 0.43], [0.5, 0.66], [0.7, 0.7]],
                  [[0.46, 0.44], [0.63, 0.64], [0.74, 0.72]]])
    p.e = np.array([[0.3, 0.2], [0.5, 0.4], [0.45, 0.3]])
    BQ = np.array([[0.3, 0.35], [0.25, 0.3], [0.4, 0.45]])
    T_H = np.array([0.12, 0.1, 0.11])
    p.J = 2
    p.S = 3
    p.T = 3
    etr_params = np.tile(np.reshape(np.array([
        [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
         0.16, -0.15, 0.83],
        [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
         0.16, -0.15, 0.83],
        [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
         0.16, -0.15, 0.83]]), (1, p.S, 12)), (p.T, 1, 1))
    etr_params = np.tile(np.reshape(etr_params, (p.T, p.S, 1, 12)),
                         (1, 1, p.J, 1))
    p.lambdas = np.array([0.65, 0.35])
    theta = np.array([0.225, 0.3])
    p.tau_bq = np.array([0.1])
    p.retire = 2
    j = None
    shift = True
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, theta, j,
                                  shift, method, p.e, p.retire,
                                  etr_params, p)
    print('Total taxes result = ', total_taxes.shape)
    print('Expected = ',  np.array([[[0.16311573,  0.1583638],
                                     [0.26812436, 0.30131202],
                                    [0.13821536, 0.00700922]],
                                    [[0.2031629, 0.1717635],
                                    [0.3563044, 0.39808896],
                                    [0.2196475, -0.01586141]],
                                    [[0.29985939, 0.18906559],
                                    [0.32237654, 0.35064977],
                                    [0.15958077, -0.0482051]]]).shape)
    assert np.allclose(total_taxes,
                       np.array([[[0.16311573,  0.1583638],
                                  [0.26812436, 0.30131202],
                                  [0.13821536, 0.00700922]],
                                 [[0.2031629, 0.1717635],
                                  [0.3563044, 0.39808896],
                                  [0.2196475, -0.01586141]],
                                 [[0.29985939, 0.18906559],
                                  [0.32237654, 0.35064977],
                                  [0.15958077, -0.0482051]]]))
