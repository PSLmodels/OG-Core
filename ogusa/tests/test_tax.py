import numpy as np
import pytest
from ogusa import tax


def test_replacement_rate_vals():
    # Test replacement rate function, making sure to trigger all three
    # cases of AIME
    nssmat = np.array([0.5, 0.5, 0.5, 0.5])
    wss = 0.5
    factor_ss = 100000
    retire = 3
    S = 4
    e = np.array([0.1, 0.3, 0.5, 0.2])
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, (e, S, retire))
    assert np.allclose(theta, np.array([0.042012]))

    # e has two dimensions
    nssmat = np.array([[0.4, 0.4], [0.4, 0.4], [0.4, 0.4], [0.4, 0.4]])
    e = np.array([[0.4, 0.3], [0.5, 0.4], [.6, .4], [.4, .3]])
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, (e, S, retire))
    assert np.allclose(theta, np.array([0.042012, 0.03842772]))

    # hit AIME case2
    nssmat = np.array([[0.3, .35], [0.3, .35], [0.3, .35], [0.3, .35]])
    # wss = 5
    e = np.array([[0.35, 0.3], [0.55, 0.4], [.65, .4], [.45, .3]])
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, (e, S, retire))
    assert np.allclose(theta, np.array([0.1145304, 0.0969304]))

    # hit AIME case1
    factor_ss = 1000
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, (e, S, retire))
    assert np.allclose(theta, np.array([0.1755, 0.126]))


def test_tau_wealth():
    # Test wealth tax computation
    b = np.array([0.1, 0.5, 0.9])
    h_wealth = 2
    p_wealth = 3
    m_wealth = 4
    tau_w_prime = tax.tau_wealth(b, (h_wealth, p_wealth, m_wealth))

    assert np.allclose(tau_w_prime, np.array([0.6 / 4.2, 0.5, 5.4 / 5.8]))


def test_tau_w_prime():
    # Test marginal tax rate on wealth
    b = np.array([0.2, 0.6, 0.8])
    h_wealth = 3
    p_wealth = 4
    m_wealth = 5
    tau_w_prime = tax.tau_w_prime(b, (h_wealth, p_wealth, m_wealth))

    assert np.allclose(tau_w_prime, np.array([1.91326531, 1.29757785,
                                           1.09569028]))


def test_tau_income():
    # Test income tax function
    r = 0.04
    w = 1.2
    b = np.array([0.4, 0.4])
    n = np.array([0.5, 0.4])
    factor = 100000
    e = np.array([0.5, 0.45])
    etr_params = np.array([0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                           -0.15, 0.15, 0.16, -0.15, 0.83])
    test_tau_income = tax.tau_income(r, w, b, n, factor, (e, etr_params))
    assert np.allclose(test_tau_income, np.array([0.80167091, 0.80167011]))

    # Test etr_params having dimension greater than 1
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    e = np.array([0.5, 0.45, 0.3])
    etr_params = np.array([[0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                           -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.002, 0.001, 0.002, 0.04, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.011, 0.001, 0.003, 0.06, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83]])
    test_tau_income = tax.tau_income(r, w, b, n, factor, (e, etr_params))
    assert np.allclose(test_tau_income, np.array([0.80167144,
                                                  0.80163711,
                                                  0.8016793]))


def test_MTR_capital():
    # Test the MTR on capital income function
    r = 0.04
    w = 1.2
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    e = np.array([0.5, 0.45, 0.3])
    factor = 110000
    etr_params = np.array([[0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                           -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.002, 0.001, 0.002, 0.04, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.011, 0.001, 0.003, 0.06, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83]])
    mtry_params = np.array([[0.001, 0.002, 0.003, 0.0015, 0.68, -0.17, 0.8,
                           -0.42, 0.18, 0.43, -0.42, 0.96],
                           [0.001, 0.002, 0.003, 0.0015, 0.65, -0.17, 0.8,
                            -0.42, 0.18, 0.33, -0.12, 0.90],
                           [0.001, 0.002, 0.003, 0.0015, 0.56, -0.17, 0.8,
                            -0.42, 0.18, 0.38, -0.22, 0.65]])

    # Test with analytical MTR = True
    analytical_mtrs = True
    mtr_capital = tax.MTR_capital(r, w, b, n, factor, (e, etr_params,
                                                       mtry_params,
                                                       analytical_mtrs))
    assert np.allclose(mtr_capital, np.array([0.80167516, 0.80164698,
                                                   0.80168175]))

    # Test with analytical MTR = False
    analytical_mtrs = False
    mtr_capital = tax.MTR_capital(r, w, b, n, factor, (e, etr_params,
                                                       mtry_params,
                                                       analytical_mtrs))
    assert np.allclose(mtr_capital, np.array([0.45239409, 0.73598958,
                                                   0.65126073]))


def test_MTR_labor():
    # Test the MTR on labor income function
    r = 0.04
    w = 1.2
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    e = np.array([0.5, 0.45, 0.3])
    factor = 110000
    etr_params = np.array([[0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                           -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.002, 0.001, 0.002, 0.04, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.011, 0.001, 0.003, 0.06, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83]])
    mtrx_params = np.array([[0.001, 0.002, 0.003, 0.0015, 0.68, -0.17, 0.8,
                           -0.42, 0.18, 0.43, -0.42, 0.96],
                           [0.001, 0.002, 0.003, 0.0015, 0.65, -0.17, 0.8,
                            -0.42, 0.18, 0.33, -0.12, 0.90],
                           [0.001, 0.002, 0.003, 0.0015, 0.56, -0.17, 0.8,
                            -0.42, 0.18, 0.38, -0.22, 0.65]])

    # Test with analytical MTR = True
    analytical_mtrs = True
    mtr_labor = tax.MTR_labor(r, w, b, n, factor, (e, etr_params,
                                                   mtrx_params,
                                                   analytical_mtrs))
    assert np.allclose(mtr_labor, np.array([0.80167516, 0.80164698,
                                            0.80168175]))

    # Test with analytical MTR = False
    analytical_mtrs = False
    mtr_labor = tax.MTR_labor(r, w, b, n, factor, (e, etr_params,
                                                   mtrx_params,
                                                   analytical_mtrs))
    assert np.allclose(mtr_labor, np.array([0.45239409, 0.73598958,
                                            0.65126073]))


def test_get_biz_tax():
    # Test function for business tax receipts
    w = np.array([1.2, 1.1])
    Y = np.array([3.0, 7.0])
    L = np.array([2.0, 3.0])
    K = np.array([5.0, 6.0])
    tau_b = 0.20
    delta_tau = 0.06
    biz_tax = tax.get_biz_tax(w, Y, L, K, (tau_b, delta_tau))
    assert np.allclose(biz_tax, np.array([0.06, 0.668]))


def test_total_taxes():
    # Test function that computes total net taxes for the household
    r = 0.04
    w = 1.2
    factor = 105000
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    e = np.array([0.5, 0.45, 0.3])
    BQ = np.array([0.3])
    T_H = np.array([0.12])
    lambdas = np.array([1.0])
    etr_params = np.array([0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                           -0.15, 0.15, 0.16, -0.15, 0.83])
    h_wealth = 1
    p_wealth = 2
    m_wealth = 3
    tau_payroll = 0.15
    theta = np.array([0.225])
    tau_bq = np.array([0.1])
    J = 1
    S = 3
    retire = 2
    j = 0
    shift = True

    # method = ss
    method = 'SS'
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift,
                                  (e, lambdas, method, retire,
                                   etr_params, h_wealth, p_wealth,
                                   m_wealth, tau_payroll, theta, tau_bq,
                                   J, S))
    assert np.allclose(total_taxes, np.array([0.47374766, -0.09027663,
                                              0.03871394]))

    # method = TPI_scalar
    method = 'TPI_scalar'
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift,
                                  (e, lambdas, method, retire,
                                   etr_params, h_wealth, p_wealth,
                                   m_wealth, tau_payroll, theta, tau_bq,
                                   J, S))
    assert np.allclose(total_taxes, np.array([0.20374766, -0.09027663,
                                              0.03871394]))

    # method = TPI
    method = 'TPI'
    r = np.array([0.04, 0.045, 0.04])
    w = np.array([1.2, 1.3, 1.1])
    b = np.array([[0.4, 0.3, 0.5], [0.4, 0.35, 0.52], [0.4, 0.45, 0.55]])
    n = np.array([[0.8, 0.4, 0.7], [0.75, 0.35, 0.6], [0.66, 0.44, 0.77]])
    BQ = np.array([0.3, 0.35, 0.25])
    T_H = np.array([0.12, 0.1, 0.11])
    etr_params = np.array([[0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                           -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83],
                           [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8,
                            -0.15, 0.15, 0.16, -0.15, 0.83]])
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift,
                                  (e, lambdas, method, retire,
                                   etr_params, h_wealth, p_wealth,
                                   m_wealth, tau_payroll, theta, tau_bq,
                                   J, S))
    assert np.allclose(total_taxes, np.array([1.59285714, 1.59285714, 1.59285714, 1.59285714]))


    # shift = 0
    method = 'TPI'
    shift = False
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift, (
    e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S))
    assert (np.allclose(total_taxes, np.array([[1.59285714, 1.59285714, 1.59285714, 1.59285714],[1.59285714, 1.59285714, 1.59285714, 1.59285714],[1.59285714, 1.59285714, 1.59285714, 1.59285714]])))


    # b.shape =3
    r = np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    w = np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    b = np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    n = np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    BQ = np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    factor = 1
    T_H = np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    j = 0
    shift = 0
    e= np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    lambdas = np.array([[0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5, 0.5]])
    retire = 2
    etr_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    h_wealth = 1
    p_wealth = 2
    m_wealth = 3
    tau_payroll = 4
    theta = np.array([0.225])
    tau_bq = np.array([1])
    J = 1
    S = 1
    method="TPI"

    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift, (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S))
    assert (np.allclose(total_taxes, np.array([[ 1.59285714, 1.59285714, 1.59285714],
                                               [ 1.59285714, 1.59285714, 1.59285714],
                                               [ 1.59285714, 1.59285714, 1.59285714]])))
