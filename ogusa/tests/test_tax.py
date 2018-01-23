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
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, (e, S,
                                                               retire))
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
    assert np.allclose(theta, np.array([0.1755, 0.126 ]))


def test_tau_wealth():

	b = np.array([0.5, 0.5, 0.5])
	h_wealth = 2
	p_wealth = 3
	m_wealth = 4
	tau_w_prime = tax.tau_wealth(b, (h_wealth, p_wealth, m_wealth))

	assert (np.allclose(tau_w_prime, np.array([ 0.6, 0.6, 0.6])))



def test_tau_w_prime():

	b = np.array([0.5, 0.5, 0.5])
	h_wealth = 2
	p_wealth = 3
	m_wealth = 4
	tau_w_prime = tax.tau_w_prime(b, (h_wealth, p_wealth, m_wealth))

	assert (np.allclose(tau_w_prime, np.array([ 0.96, 0.96, 0.96])))


def test_tau_income():

	r = np.array([0.5])
	w = np.array([0.5])
	b = np.array([0.5])
	n = np.array([0.5])
	factor = 1
	e = np.array([0.5])
	etr_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	test_tau_income= tax.tau_income(r, w, b, n, factor, (e, etr_params))
	assert (np.allclose(test_tau_income, np.array([1.5])))

    # etr_params has dimensions from 1 to 4
	etr_params= np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
						  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
	test_tau_income = tax.tau_income(r, w, b, n, factor, (e, etr_params))
	assert (np.allclose(test_tau_income, np.array([1.5, 1.5])))

	etr_params = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
						   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
						   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
	test_tau_income = tax.tau_income(r, w, b, n, factor, (e, etr_params))
	assert (np.allclose(test_tau_income, np.array([1.5, 1.5, 1.5,])))

	etr_params = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
						   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
						   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
						   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
	test_tau_income = tax.tau_income(r, w, b, n, factor, (e, etr_params))
	assert (np.allclose(test_tau_income, np.array([1.5, 1.5, 1.5, 1.5])))


def test_MTR_capital():

	r = np.array([0.5, 0.5, 0.5, 0.5])
	w = np.array([0.5, 0.5, 0.5, 0.5])
	b = np.array([0.5, 0.5, 0.5, 0.5])
	n = np.array([0.5, 0.5, 0.5, 0.5])
	factor = 1
	e = np.array([0.5, 0.5, 0.5, 0.5])
	etr_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	mtry_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	analytical_mtrs = 1

	test_MTR_capital = tax.MTR_capital(r, w, b, n, factor, (e, etr_params, mtry_params, analytical_mtrs))
	assert (np.allclose(test_MTR_capital, np.array([1.5, 1.5, 1.5, 1.5])))


def test_MTR_labor():

	r = np.array([0.5, 0.5, 0.5, 0.5])
	w = np.array([0.5, 0.5, 0.5, 0.5])
	b = np.array([0.5, 0.5, 0.5, 0.5])
	n = np.array([0.5, 0.5, 0.5, 0.5])
	factor = 1
	e = np.array([0.5, 0.5, 0.5, 0.5])
	etr_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	mtry_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	analytical_mtrs = 1

	test_MTR_labor = tax.MTR_labor(r, w, b, n, factor, (e, etr_params, mtry_params, analytical_mtrs))
	assert (np.allclose(test_MTR_labor, np.array([1.5, 1.5, 1.5, 1.5])))

def test_get_biz_tax():

	w = np.array([0.5, 0.5])
	Y = np.array([2.0, 2.0])
	L = np.array([1.0, 1.0])
	K = np.array([1.0, 1.0])
	tau_b = 1
	delta_tau = 1
	biz_tax = tax.get_biz_tax(w, Y, L, K, (tau_b, delta_tau))
	assert (np.allclose(biz_tax, np.array([0.5, 0.5])))


def	test_total_taxes():

    r = np.array([0.5, 0.5, 0.5, 0.5])
    w = np.array([0.5, 0.5, 0.5, 0.5])
    b = np.array([0.5, 0.5, 0.5, 0.5])
    n = np.array([0.5, 0.5, 0.5, 0.5])
    BQ = np.array([0.5, 0.5, 0.5, 0.5])
    factor = 1
    T_H = np.array([0.5, 0.5, 0.5, 0.5])
    j = 0
    shift = 1
    e = np.array([0.5, 0.5, 0.5, 0.5])
    lambdas = np.array([0.5, 0.5, 0.5, 0.5])
    retire = 1
    etr_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    h_wealth = 1
    p_wealth = 2
    m_wealth = 3
    tau_payroll = 4
    theta = np.array([ 0.225])
    tau_bq = np.array([1])
    J = 1
    S = 1

    # method = ss
    method = 'SS'
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift, (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S))
    assert (np.allclose(total_taxes, np.array([ 1.59285714,  1.59285714,  1.59285714,  1.59285714])))

    # method = TPI_scalar
    method = 'TPI_scalar'
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift, (e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S))
    assert (np.allclose(total_taxes, np.array([ 1.20535714, 1.20535714, 1.20535714, 1.20535714])))


    # method = TPI
    method = 'TPI'
    retire = 2
    j=0
    total_taxes = tax.total_taxes(r, w, b, n, BQ, factor, T_H, j, shift, (
    e, lambdas, method, retire, etr_params, h_wealth, p_wealth, m_wealth, tau_payroll, theta, tau_bq, J, S))
    assert (np.allclose(total_taxes, np.array([1.59285714, 1.59285714, 1.59285714, 1.59285714])))


    # shift = 0
    method = 'TPI'
    shift=0
    retire = 2
    j=0
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
