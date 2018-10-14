import pytest
import numpy as np
from ogusa import household
from ogusa.pb_api import Specifications

test_data = [(0.1, 1, 10),
             (0.2, 2.5, 55.90169944),
             (np.array([0.5, 6.2, 1.5]), 3.2,
              np.array([9.18958684, 0.002913041, 0.273217159]))]


@pytest.mark.parametrize('c,sigma,expected', test_data,
                         ids=['Scalar 0', 'Scalar 1', 'Vector'])
def test_marg_ut_cons(c, sigma, expected):
    # Test marginal utility of consumption calculation
    test_value = household.marg_ut_cons(c, sigma)

    assert np.allclose(test_value, expected)


# Tuples in order: n, p, expected result
p1 = Specifications()
p1.b_ellipse = 0.527
p1.upsilon = 1.497
p1.ltilde = 1.0
p1.chi_n = 3.3

p2 = Specifications()
p2.b_ellipse = 0.527
p2.upsilon = 0.9
p2.ltilde = 1.0
p2.chi_n = 3.3

p3 = Specifications()
p3.b_ellipse = 0.527
p3.upsilon = 0.9
p3.ltilde = 2.3
p3.chi_n = 3.3

p4 = Specifications()
p4.b_ellipse = 2.6
p4.upsilon = 1.497
p4.ltilde = 1.0
p4.chi_n = 3.3

test_data = [(0.87, p1, 2.825570309),
             (0.0, p1, 0.0009117852028298067),
             (0.99999, p1, 69.52423604),
             (0.00001, p1, 0.005692782),
             (0.8, p2, 1.471592068),
             (0.8, p3, 0.795937549),
             (0.8, p4, 11.66354267),
             (np.array([[0.8, 0.9, 0.3], [0.5, 0.2, 0.99]]),
              p1,
              np.array([[2.364110379, 3.126796062, 1.014935377],
                        [1.4248841, 0.806333875, 6.987729463]]))]


@pytest.mark.parametrize('n,params,expected', test_data,
                         ids=['1', '2', '3', '4', '5', '6', '7', '8'])
def test_marg_ut_labor(n, params, expected):
    # Test marginal utility of labor calculation
    test_value = household.marg_ut_labor(n, params.chi_n, params)

    assert np.allclose(test_value, expected)


p1 = Specifications()
p1.e = 0.99
p1.lambdas = np.array([0.25])
p1.g_y = 0.03
r1 = 0.05
w1 = 1.2
b1 = 0.5
b_splus1_1 = 0.55
n1 = 0.8
BQ1 = 0.1
net_tax1 = 0.02
j1 = None

p2 = Specifications()
p2.e = np.array([0.99, 1.5, 0.2])
p2.lambdas = np.array([0.25])
p2.g_y = 0.03
r2 = np.array([0.05, 0.04, 0.09])
w2 = np.array([1.2, 0.8, 2.5])
b2 = np.array([0.5, 0.99, 9])
b_splus1_2 = np.array([0.55, 0.2, 4])
n2 = np.array([0.8, 3.2, 0.2])
BQ2 = np.array([0.1, 2.4, 0.2])
net_tax2 = np.array([0.02, 0.5, 1.4])
j2 = None

p3 = Specifications()
p3.e = np.array([[1.0, 2.1], [0.4, 0.5], [1.6, 0.9]])
p3.lambdas = np.array([0.4, 0.6])
p3.g_y = 0.01
r3 = 0.11
w3 = 0.75
b3 = np.array([[0.56, 0.7], [0.4, 0.95], [2.06, 1.7]])
b_splus1_3 = np.array([[0.4, 0.6], [0.33, 1.95], [1.6, 2.7]])
n3 = np.array([[0.9, 0.5], [0.8, 1.1], [0, 0.77]])
BQ3 = np.array([1.3, 0.3])
net_tax3 = np.array([[0.1, 1.1], [0.4, 0.44], [0.6, 1.7]])
j3 = None

p4 = Specifications()
p4.e = np.array([[1.0, 2.1], [0.4, 0.5], [1.6, 0.9]])
p4.lambdas = np.array([0.7, 0.3])
p4.g_y = 0.05
r4 = np.tile(np.reshape(np.array([0.11, 0.02, 0.08, 0.05]),
                        (4, 1, 1)), (1, 3, 2))
w4 = np.tile(np.reshape(np.array([0.75, 1.3, 0.9, 0.7]),
                        (4, 1, 1)), (1, 3, 2))
b4 = np.array([np.array([[0.5, 0.55], [0.6, 0.9], [0.9, .4]]),
               np.array([[7.1, 8.0], [1.0, 2.1], [9.1, 0.1]]),
               np.array([[0.4, 0.2], [0.34, 0.56], [0.3, 0.6]]),
               np.array([[0.1, 0.2], [0.4, 0.5], [0.555, 0.76]])])
b_splus1_4 = np.array([np.array([[0.4, 0.2], [1.4, 1.5], [0.5, 0.6]]),
                       np.array([[7.1, 8.0], [0.4, 0.9], [9.1, 10]]),
                       np.array([[0.15, 0.52], [0.44, 0.85], [0.5, 0.6]]),
                       np.array([[4.1, 2.0], [0.65, 0.65], [0.25, 0.56]])])
n4 = np.array([np.array([[0.8, 0.9], [0.4, 0.5], [0.55, 0.66]]),
               np.array([[0.7, 0.8], [0.2, 0.1], [0, 0.4]]),
               np.array([[0.1, 0.2], [1.4, 1.5], [0.5, 0.6]]),
               np.array([[0.4, 0.6], [0.99, 0.44], [0.35, 0.65]])])
BQ4 = np.tile(np.reshape(np.array([[0.1, 1.1], [0.4, 1.0], [0.6, 1.7],
                                   [0.9, 2.0]]), (4, 1, 2)), (1, 3, 1))
net_tax4 = np.array([np.array([[0.01, 0.02], [0.4, 0.5], [0.05, 0.06]]),
                     np.array([[0.17, 0.18], [0.08, .02], [0.9, 0.10]]),
                     np.array([[1.0, 2.0], [0.04, 0.25], [0.15, 0.16]]),
                     np.array([[0.11, 0.021], [0.044, 0.025],
                               [0.022, 0.032]])])
j4 = None

test_data = [((r1, w1, b1, b_splus1_1, n1, BQ1, net_tax1, j1, p1),
              1.288650006),
             ((r2, w2, b2, b_splus1_2, n2, BQ2, net_tax2, j2, p2),
              np.array([1.288650006, 13.76350909, 5.188181864])),
             ((r3, w3, b3, b_splus1_3, n3, BQ3, net_tax3, j3, p3),
              np.array([[4.042579933, 0.3584699],
                        [3.200683445, -0.442597826],
                        [3.320519733, -1.520385451]])),
             ((r4, w4, b4, b_splus1_4, n4, BQ4, net_tax4, j4, p4),
              np.array([np.array([[0.867348704, 5.464412447],
                                  [-0.942922392, 2.776260022],
                                  [1.226221595, 3.865404009]]),
                        np.array([[1.089403787, 5.087164562],
                                  [1.194920133, 4.574189347],
                                  [-0.613138406, -6.70937763]]),
                        np.array([[0.221452193, 3.714005697],
                                  [1.225783575, 5.802886235],
                                  [1.225507309, 6.009904009]]),
                        np.array([[-2.749497209, 5.635124474],
                                  [1.255588073, 6.637340454],
                                  [1.975646512, 7.253454853]])]))]


@pytest.mark.parametrize('model_args,expected', test_data,
                         ids=['scalar', 'vector', 'matrix', '3D array'])
def test_get_cons(model_args, expected):
    # Test consumption calculation
    r, w, b, b_splus1, n, BQ, net_tax, j, p = model_args
    test_value = household.get_cons(r, w, b, b_splus1, n, BQ, net_tax,
                                    p.e, j, p)

    assert np.allclose(test_value, expected)


# Define variables for test of SS version
p1 = Specifications()
p1.e = np.array([1.0, 0.9, 1.4]).reshape(3, 1)
p1.sigma = 2.0
p1.beta = 0.96
p1.g_y = 0.03
p1.chi_b = np.array([1.5])
p1.tau_bq = 0.0
p1.rho = np.array([0.1, 0.2, 1.0])
p1.lambdas = np.array([1.0])
p1.J = 1
p1.S = 3
p1.T = 3
p1.analytical_mtrs = False
etr_params = np.array([np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.33, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.20, 0]]),
                       np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0]]),
                       np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0]])])
mtry_params = np.array([np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.28, 0]]),
                        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.49, 0]]),
                        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.32, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70, 0]])])
p1.h_wealth = 0.1
p1.m_wealth = 1.0
p1.p_wealth = 0.0
p1.tau_payroll = 0.15
p1.retire = 2

test_params_ss = p1
r = 0.05
w = 1.2
b = np.array([0.0, 0.8, 0.5])
b_splus1 = np.array([0.8, 0.5, 0.1])
n = np.array([0.9, 0.8, 0.5])
BQ = 0.1
factor = 120000
T_H = 0.22
theta = np.array([0.1])
method = 'SS'
j = None
test_vars_ss = (r, w, b, b_splus1, n, BQ, factor, T_H, theta,
                etr_params[-1, :, :], mtry_params[-1, :, :], j, method)
expected_ss = np.array([9.940309897, -1.004780793, -140.5545878])

# Define variables/params for test of TPI version
method_tpi = 'TPI'
test_params_tpi = p1
j = 0
r_vec = np.array([0.05, 0.03, 0.04])
w_vec = np.array([1.2, 0.9, 0.8])
BQ_vec = np.array([0.1, 0.05, 0.15])
T_H_vec = np.array([0.22, 0.15, 0.0])
etr_params_tpi = np.empty((p1.S, etr_params.shape[2]))
mtry_params_tpi = np.empty((p1.S, mtry_params.shape[2]))
for i in range(etr_params.shape[2]):
    etr_params_tpi[:, i] = np.diag(np.transpose(etr_params[:, :p1.S, i]))
    mtry_params_tpi[:, i] = np.diag(np.transpose(mtry_params[:, :p1.S, i]))
test_vars_tpi = (r_vec, w_vec, b, b_splus1, n, BQ_vec, factor,
                 T_H_vec, theta, etr_params_tpi,
                 mtry_params_tpi, j, method_tpi)
expected_tpi = np.array([300.977031, 2.719866638, -139.9187228])
test_data = [(test_vars_ss, test_params_ss, expected_ss),
             (test_vars_tpi, test_params_tpi, expected_tpi)]


@pytest.mark.parametrize('model_vars,params,expected', test_data,
                         ids=['SS', 'TPI'])
def test_FOC_savings(model_vars, params, expected):
    # Test FOC condition for household's choice of savings
    (r, w, b, b_splus1, n, BQ, factor, T_H, theta, etr_params,
     mtry_params, j, method) = model_vars
    if j is not None:
        test_value = household.FOC_savings(
            r, w, b, b_splus1, n, BQ, factor, T_H, theta,
            params.e[:, j], params.rho, params.retire, etr_params,
            mtry_params, j, params, method)
    else:
        test_value = household.FOC_savings(
            r, w, b, b_splus1, n, BQ, factor, T_H, theta,
            np.squeeze(params.e), params.rho, params.retire,
            etr_params, mtry_params, j, params, method)
    assert np.allclose(test_value, expected)


# Define variables for test of SS version
p1 = Specifications()
p1.rho = np.array([0.1, 0.2, 1.0])
p1.e = np.array([1.0, 0.9, 1.4]).reshape(3, 1)
p1.sigma = 1.5
p1.g_y = 0.04
p1.b_ellipse = 0.527
p1.upsilon = 1.45
p1.chi_n = 0.75
p1.ltilde = 1.2
p1.tau_bq = np.array([0.0])
p1.lambdas = np.array([1.0])
p1.J = 1
p1.S = 3
p1.T = 3
p1.analytical_mtrs = False
etr_params = np.array([np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.33, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.20, 0]]),
                       np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0]]),
                       np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0]])])
mtrx_params = np.array([np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.44, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18, 0]]),
                        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0]]),
                        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.77, 0]])])
p1.h_wealth = 0.1
p1.m_wealth = 1.0
p1.p_wealth = 0.0
p1.tau_payroll = 0.15
p1.retire = 2
theta = np.array([0.1])
j = 0
method = 'SS'
r = 0.05
w = 1.2
b = np.array([0.0, 0.8, 0.5])
b_splus1 = np.array([0.8, 0.5, 0.1])
n = np.array([0.9, 0.8, 0.5])
BQ = 0.1
factor = 120000
T_H = 0.22
test_params_ss = p1
test_vars_ss = (r, w, b, b_splus1, n, BQ, factor, T_H, theta,
                etr_params[-1, :, :], mtrx_params[-1, :, :], j, method)
expected_ss = np.array([4.776470285, 0.140755217, -0.141968521])

# Define variables/params for test of TPI version
method_tpi = 'TPI'
test_params_tpi = p1
j = 0
r_vec = np.array([0.05, 0.03, 0.04])
w_vec = np.array([1.2, 0.9, 0.8])
BQ_vec = np.array([0.1, 0.05, 0.15])
T_H_vec = np.array([0.22, 0.15, 0.0])
etr_params_tpi = np.empty((p1.S, etr_params.shape[2]))
mtrx_params_tpi = np.empty((p1.S, mtrx_params.shape[2]))
for i in range(etr_params.shape[2]):
    etr_params_tpi[:, i] = np.diag(np.transpose(etr_params[:, :p1.S, i]))
    mtrx_params_tpi[:, i] = np.diag(np.transpose(mtrx_params[:, :p1.S, i]))
test_vars_tpi = (r_vec, w_vec, b, b_splus1, n, BQ_vec, factor,
                 T_H_vec, theta, etr_params_tpi,
                 mtrx_params_tpi, j, method_tpi)
expected_tpi = np.array([69.39898495, 1.930915718, -0.133902455])

test_data = [(test_vars_ss, test_params_ss, expected_ss),
             (test_vars_tpi, test_params_tpi, expected_tpi)]


@pytest.mark.parametrize('model_vars,params,expected', test_data,
                         ids=['SS', 'TPI'])
def test_FOC_labor(model_vars, params, expected):
    # Test FOC condition for household's choice of labor supply
    (r, w, b, b_splus1, n, BQ, factor, T_H, theta, etr_params,
     mtrx_params, j, method) = model_vars
    test_value = household.FOC_labor(
        r, w, b, b_splus1, n, BQ, factor, T_H, theta, params.chi_n,
        params.e[:, j], params.retire, etr_params, mtrx_params, j,
        params, method)

    assert np.allclose(test_value, expected)


# def test_constraint_checker_SS():
#
#     assert np.allclose()
#
#
# def test_constraint_checker_TPI():
#
#     assert np.allclose()
