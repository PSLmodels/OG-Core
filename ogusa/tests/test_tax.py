import numpy as np
import copy
import pytest
from ogusa import tax
from ogusa.parameters import Specifications


p = Specifications()
new_param_values = {
    'S': 4,
    'J': 1,
    'T': 4,
}
p.update_specifications(new_param_values)
p.retire = [3, 3, 3, 3, 3, 3, 3, 3]
p1 = copy.deepcopy(p)
p2 = copy.deepcopy(p)
p3 = copy.deepcopy(p)
# Use just a column of e
p1.e = np.transpose(np.array([[0.1, 0.3, 0.5, 0.2], [0.1, 0.3, 0.5, 0.2]]))
# e has two dimensions
p2.e = np.array([[0.4, 0.3], [0.5, 0.4], [.6, .4], [.4, .3]])
p3.e = np.array([[0.35, 0.3], [0.55, 0.4], [.65, .4], [.45, .3]])
p5 = copy.deepcopy(p3)
p5.PIA_minpayment = 125.0
wss = 0.5
n1 = np.array([0.5, 0.5, 0.5, 0.5])
n2 = nssmat = np.array([[0.4, 0.4], [0.4, 0.4], [0.4, 0.4], [0.4, 0.4]])
n3 = nssmat = np.array([[0.3, .35], [0.3, .35], [0.3, .35], [0.3, .35]])
factor1 = 100000
factor3 = 10000
factor4 = 1000
expected1 = np.array([0.042012])
expected2 = np.array([0.042012, 0.03842772])
expected3 = np.array([0.1145304, 0.0969304])
expected4 = np.array([0.1755, 0.126])
expected5 = np.array([0.1755, 0.126 * 1.1904761904761905])

test_data = [(n1, wss, factor1, 0, p1, expected1),
             (n2, wss, factor1, None, p2, expected2),
             (n3, wss, factor3, None, p3, expected3),
             (n3, wss, factor4, None, p3, expected4),
             (n3, wss, factor4, None, p5, expected5)]


@pytest.mark.parametrize('n,w,factor,j,p,expected', test_data,
                         ids=['1D e', '2D e', 'AIME case 2',
                              'AIME case 3', 'Min PIA case'])
def test_replacement_rate_vals(n, w, factor, j, p, expected):
    # Test replacement rate function, making sure to trigger all three
    # cases of AIME

    theta = tax.replacement_rate_vals(n, w, factor, j, p)
    assert np.allclose(theta, expected)


b1 = np.array([0.1, 0.5, 0.9])
p1 = Specifications()
new_param_values = {
    'S': 3,
    'J': 1,
    'T': 3,
    'h_wealth': [2],
    'p_wealth': [3],
    'm_wealth': [4]
}
p1.update_specifications(new_param_values)
expected1 = np.array([0.14285714, 0.6, 0.93103448])
p2 = Specifications()
new_param_values2 = {
    'S': 3,
    'J': 1,
    'T': 3,
    'h_wealth': [1.2, 1.1, 2.3],
    'p_wealth': [2.2, 2.3, 1.8],
    'm_wealth': [3, 4, 3]
}
p2.update_specifications(new_param_values2)
expected2 = np.array([0.084615385, 0.278021978, 0.734911243])

test_data = [(b1, p1, expected1),
             (b1, p2, expected2)]


@pytest.mark.parametrize('b,p,expected', test_data,
                         ids=['constant params', 'vary params'])
def test_ETR_wealth(b, p, expected):
    # Test wealth tax computation
    tau_w = tax.ETR_wealth(b, p.h_wealth[:p.T], p.m_wealth[:p.T],
                           p.p_wealth[:p.T])

    assert np.allclose(tau_w, expected)


b1 = np.array([0.2, 0.6, 0.8])
p1 = Specifications()
new_param_values = {
    'S': 3,
    'J': 1,
    'T': 3,
    'h_wealth': [3],
    'p_wealth': [4],
    'm_wealth': [5]
}
p1.update_specifications(new_param_values)
expected1 = np.array([1.91326531, 1.29757785, 1.09569028])
b2 = np.array([0.1, 0.5, 0.9])
p2 = Specifications()
new_param_values2 = {
    'S': 3,
    'J': 1,
    'T': 3,
    'h_wealth': [1.2, 1.1, 2.3],
    'p_wealth': [2.2, 2.3, 1.8],
    'm_wealth': [3, 4, 3]
}
p2.update_specifications(new_param_values2)
expected2 = np.array([0.813609467, 0.488829851, 0.483176359])

test_data = [(b1, p1, expected1),
             (b2, p2, expected2)]


@pytest.mark.parametrize('b,p,expected', test_data,
                         ids=['constant params', 'vary params'])
def test_MTR_wealth(b, p, expected):
    # Test marginal tax rate on wealth
    tau_w_prime = tax.MTR_wealth(b, p.h_wealth[:p.T], p.m_wealth[:p.T],
                                 p.p_wealth[:p.T])

    assert np.allclose(tau_w_prime, expected)


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
    p = Specifications()
    new_param_values = {
        'tau_b': [0.20],
        'delta_tau_annual': [0.06]
    }
    p.update_specifications(new_param_values)
    p.T = 3
    w = np.array([1.2, 1.1, 1.2])
    Y = np.array([3.0, 7.0, 3.0])
    L = np.array([2.0, 3.0, 2.0])
    K = np.array([5.0, 6.0, 5.0])
    biz_tax = tax.get_biz_tax(w, Y, L, K, p, 'TPI')
    assert np.allclose(biz_tax, np.array([0.06, 0.668, 0.06]))


# Set parameter class for each case
p = Specifications()
p.tax_func_type = 'DEP'
p.J = 1
p.S = 3
p.lambdas = np.array([1.0])
p.e = np.array([0.5, 0.45, 0.3]).reshape(3, 1)
p.h_wealth = np.ones(p.T + p.S) * 1
p.p_wealth = np.ones(p.T + p.S) * 2
p.m_wealth = np.ones(p.T + p.S) * 3
p.tau_payroll = np.ones(p.T + p.S) * 0.15
p.tau_bq = np.ones(p.T + p.S) * 0.1
p.retire = (np.ones(p.T + p.S) * 2).astype(int)
p1 = copy.deepcopy(p)
p2 = copy.deepcopy(p)
p3 = copy.deepcopy(p)
p3.T = 3
p4 = copy.deepcopy(p)
p5 = copy.deepcopy(p)
p5.e = np.array([[0.3, 0.2], [0.5, 0.4], [0.45, 0.3]])
p5.J = 2
p5.T = 3
p5.lambdas = np.array([0.65, 0.35])
# set variables and other parameters for each case
r1 = 0.04
w1 = 1.2
b1 = np.array([0.4, 0.3, 0.5])
n1 = np.array([0.8, 0.4, 0.7])
BQ1 = np.array([0.3])
bq1 = BQ1 / p1.lambdas[0]
T_H1 = np.array([0.12])
theta1 = np.array([0.225])
etr_params1 = np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
     0.16, -0.15, 0.83],
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
     0.16, -0.15, 0.83],
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
     0.16, -0.15, 0.83]]), (1, p1.S, 12))
j1 = 0
shift1 = True
method1 = 'SS'

r2 = r1
w2 = w1
b2 = b1
n2 = n1
BQ2 = BQ1
bq2 = bq1
T_H2 = T_H1
theta2 = theta1
etr_params2 = etr_params1
j2 = 0
shift2 = True
method2 = 'TPI_scalar'

r3 = np.array([0.04, 0.045, 0.04])
w3 = np.array([1.2, 1.3, 1.1])
b3 = np.tile(np.reshape(np.array([0.4, 0.3, 0.5]), (1, p3.S, 1)),
             (p3.T, 1, p3.J))
n3 = np.tile(np.reshape(np.array([0.8, 0.4, 0.7]), (1, p3.S, 1)),
             (p3.T, 1, p3.J))
BQ3 = np.array([0.3, 0.4, 0.45])
bq3 = np.tile(np.reshape(BQ3 / p3.lambdas[0], (p3.T, 1)), (1, p3.S))
T_H3 = np.array([0.12, 0.1, 0.11])
theta3 = theta1
etr_params3 = np.tile(np.reshape(np.array(
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15, 0.16,
     -0.15, 0.83]), (1, 1, 12)), (p3.T, p3.S, 1))
j3 = 0
shift3 = True
method3 = 'TPI'

r4 = r3
w4 = w3
b4 = b3
n4 = n3
BQ4 = BQ3
bq4 = bq3
T_H4 = T_H3
theta4 = theta1
etr_params4 = etr_params3
j4 = 0
shift4 = False
method4 = 'TPI'

r5 = r3
w5 = w3
b5 = np.array([[[0.2, 0.3], [0.3, 0.35], [0.4, 0.35]],
              [[0.4, 0.3], [0.4, 0.35], [0.5, 0.35]],
              [[0.6, 0.4], [0.3, 0.4], [0.4, 0.22]]])
n5 = np.array([[[0.6, 0.5], [0.5, 0.55], [0.7, 0.8]],
              [[0.4, 0.43], [0.5, 0.66], [0.7, 0.7]],
              [[0.46, 0.44], [0.63, 0.64], [0.74, 0.72]]])
BQ5 = np.tile(np.reshape(np.array([[0.3, 0.35], [0.25, 0.3],
                                   [0.4, 0.45]]), (p5.T, 1, p5.J)),
              (1, p5.S, 1))
bq5 = BQ5 / p5.lambdas.reshape(1, 1, p5.J)
T_H5 = np.array([0.12, 0.1, 0.11])
theta5 = np.array([0.225, 0.3])
etr_params = np.tile(np.reshape(np.array([
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
     0.16, -0.15, 0.83],
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
     0.16, -0.15, 0.83],
    [0.001, 0.002, 0.003, 0.0015, 0.8, -0.14, 0.8, -0.15, 0.15,
     0.16, -0.15, 0.83]]), (1, p5.S, 12)), (p5.T, 1, 1))
etr_params5 = np.tile(np.reshape(etr_params, (p5.T, p5.S, 1, 12)),
                      (1, 1, p5.J, 1))
j5 = None
shift5 = False
method5 = 'TPI'

p6 = copy.deepcopy(p5)
p6.tau_bq = np.array([0.05, 0.2, 0.0])

p7 = copy.deepcopy(p5)
p7.tau_bq = np.array([0.05, 0.2, 0.0])
p7.retire = (np.array([1, 2, 2])).astype(int)

p8 = copy.deepcopy(p6)
p8.replacement_rate_adjust = [1.5, 0.6, 1.0]

factor = 105000
expected1 = np.array([0.47374766, -0.09027663, 0.03871394])
expected2 = np.array([0.20374766, -0.09027663, 0.03871394])
expected3 = np.array([[0.473747659, -0.090276635, 0.038713941],
                      [0.543420101, -0.064442513, 0.068204207],
                      [0.460680696, -0.05990653, 0.066228621]])
expected4 = np.array([[0.473747659, 0.179723365, 0.038713941],
                      [0.543420101, 0.228057487, 0.068204207],
                      [0.460680696, 0.18759347, 0.066228621]])
expected5 = np.array([[[0.16311573, 0.1583638], [0.27581667, 0.31559773],
                       [0.12283074, -0.02156221]],
                      [[0.1954706, 0.15747779], [0.3563044, 0.39808896],
                       [0.19657058, -0.05871855]],
                      [[0.31524401, 0.21763702], [0.34545346, 0.39350691],
                       [0.15958077, -0.0482051]]])
expected6 = np.array([[[0.16311573 - 0.023076923,  0.1583638 - 0.05],
                       [0.27581667 - 0.023076923, 0.31559773 - 0.05],
                       [0.12283074 - 0.023076923, -0.02156221 - 0.05]],
                      [[0.1954706 + 0.038461538, 0.15747779 + 0.085714286],
                       [0.3563044 + 0.038461538, 0.39808896 + 0.085714286],
                       [0.19657058 + 0.038461538, -0.05871855 + 0.085714286]],
                      [[0.31524401 - 0.061538462, 0.21763702 - 0.12857143],
                       [0.34545346 - 0.061538462, 0.39350691 - 0.12857143],
                       [0.15958077 - 0.061538462, -0.0482051 - 0.12857143]]])
expected7 = np.array([[[0.16311573 - 0.023076923,
                        0.1583638 - 0.05],
                       [0.27581667 - 0.023076923 - 0.27,
                        0.31559773 - 0.05 - 0.36],
                       [0.12283074 - 0.023076923,
                        -0.02156221 - 0.05]],
                      [[0.1954706 + 0.038461538, 0.15747779 + 0.085714286],
                       [0.3563044 + 0.038461538, 0.39808896 + 0.085714286],
                       [0.19657058 + 0.038461538, -0.05871855 + 0.085714286]],
                      [[0.31524401 - 0.061538462, 0.21763702 - 0.12857143],
                       [0.34545346 - 0.061538462, 0.39350691 - 0.12857143],
                       [0.15958077 - 0.061538462, -0.0482051 - 0.12857143]]])
expected8 = np.array([[[0.16311573 - 0.023076923,  0.1583638 - 0.05],
                       [0.27581667 - 0.023076923, 0.31559773 - 0.05],
                       [0.12283074 - 0.023076923 - 0.135,
                        -0.02156221 - 0.05 - 0.18]],
                      [[0.1954706 + 0.038461538, 0.15747779 + 0.085714286],
                       [0.3563044 + 0.038461538, 0.39808896 + 0.085714286],
                       [0.19657058 + 0.038461538 + 0.117,
                        -0.05871855 + 0.085714286 + 0.156]],
                      [[0.31524401 - 0.061538462, 0.21763702 - 0.12857143],
                       [0.34545346 - 0.061538462, 0.39350691 - 0.12857143],
                       [0.15958077 - 0.061538462, -0.0482051 - 0.12857143]]])

test_data = [(r1, w1, b1, n1, bq1, factor, T_H1, theta1, None, j1, shift1,
              method1, p1.e[:, j1], etr_params1[-1, :, :],
              p1, expected1),
             (r2, w2, b2, n2, bq2, factor, T_H2, theta2, None, j2, shift2,
              method2, p2.e[:, j2], etr_params2, p2, expected2),
             (r3, w3, b3[:, :, j3], n3[:, :, j3], bq3, factor, T_H3,
              theta3, 0, j3, shift3, method3, p3.e[:, j3], etr_params3, p3,
              expected3),
             (r4, w4, b4[:, :, j4], n4[:, :, j4], bq4, factor, T_H4,
              theta4, 0, j4, shift4, method4, p4.e[:, j4],
              etr_params4, p4, expected4),
             (r5, w5, b5, n5, bq5, factor, T_H5, theta5, 0, j5, shift5,
              method5, p5.e, etr_params5, p5, expected5),
             (r5, w5, b5, n5, bq5, factor, T_H5, theta5, 0, j5, shift5,
              method5, p5.e, etr_params5, p6, expected6),
             (r5, w5, b5, n5, bq5, factor, T_H5, theta5, 0, j5, shift5,
              method5, p5.e, etr_params5, p7, expected7),
             (r5, w5, b5, n5, bq5, factor, T_H5, theta5, 0, j5, shift5,
              method5, p5.e, etr_params5, p8, expected8)]


@pytest.mark.parametrize('r,w,b,n,bq,factor,T_H,theta,t,j,shift,method,'
                         + 'e,etr_params,p,expected',
                         test_data, ids=['SS', 'TPI Scalar',
                                         'TPI shift = True',
                                         'TPI shift = False', 'TPI 3D',
                                         'TPI 3D,vary tau_bq',
                                         'TPI 3D,vary retire',
                                         'TPI 3D,vary replacement rate'])
def test_total_taxes(r, w, b, n, bq, factor, T_H, theta, t, j, shift,
                     method, e, etr_params, p, expected):
    # Test function that computes total net taxes for the household
    # method = ss
    total_taxes = tax.total_taxes(r, w, b, n, bq, factor, T_H, theta, t,
                                  j, shift, method, e, etr_params, p)
    assert np.allclose(total_taxes, expected)
