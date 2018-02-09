import pytest
import numpy as np
from ogusa import household


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


# Note that params tuple in order: b_ellipse, upsilon, ltilde, chi_n
test_data = [(0.87, (0.527, 1.497, 1.0, 3.3), 2.825570309),
             (0.0, (0.527, 1.497, 1.0, 3.3), 0),
             (0.99999, (0.527, 1.497, 1.0, 3.3), 69.52423604),
             (0.00001, (0.527, 1.497, 1.0, 3.3), 0.005692782),
             (0.8, (0.527, 0.9, 1.0, 3.3), 1.471592068),
             (0.8, (0.527, 0.9, 2.3, 3.3), 0.795937549),
             (0.8, (2.6, 1.497, 1.0, 3.3), 11.66354267),
             (np.array([[0.8, 0.9, 0.3], [0.5, 0.2, 0.99]]),
              (0.527, 1.497, 1.0, 3.3),
              np.array([[2.364110379, 3.126796062, 1.014935377],
                        [1.4248841, 0.806333875, 6.987729463]]))]


@pytest.mark.parametrize('n,params,expected', test_data,
                         ids=['1', '2', '3', '4', '5', '6', '7', '8'])
def test_marg_ut_labor(n, params, expected):
    # Test marginal utility of labor calculation
    test_value = household.marg_ut_labor(n, params)

    assert np.allclose(test_value, expected)


# model_vars are in order: r, w, b, b_splus1, n, BQ, net_tax
# params are in order: e, lambdas, g_y
test_data = [((0.05, 1.2, 0.5, 0.55, 0.8, 0.1, 0.02), (0.99, 0.25, 0.03),
              1.288650006),
             ((np.array([0.05, 0.04, 0.09]), np.array([1.2, 0.8, 2.5]),
               np.array([0.5, 0.99, 9]), np.array([0.55, 0.2, 4]),
               np.array([0.8, 3.2, 0.2]), np.array([0.1, 2.4, 0.2]),
               np.array([0.02, 0.5, 1.4])),
              (np.array([0.99, 1.5, 0.2]), 0.25, 0.03),
              np.array([1.288650006, 13.76350909, 5.188181864])),
             ((0.11, 0.75,
               np.array([[0.56, 0.7], [0.4, 0.95], [2.06, 1.7]]),
               np.array([[0.4, 0.6], [0.33, 1.95], [1.6, 2.7]]),
               np.array([[0.9, 0.5], [0.8, 1.1], [0, 0.77]]),
               np.array([1.3, 0.3]),
               np.array([[0.1, 1.1], [0.4, 0.44], [0.6, 1.7]])),
              (np.array([[1.0, 2.1], [0.4, 0.5], [1.6, 0.9]]),
               np.array([0.4, 0.6]), 0.01),
              np.array([[4.042579933, 0.3584699],
                        [3.200683445, -0.442597826],
                        [3.320519733, -1.520385451]])),
             ((np.tile(np.reshape(np.array([0.11, 0.02, 0.08, 0.05]),
                                  (4, 1, 1)), (1, 3, 2)),
               np.tile(np.reshape(np.array([0.75, 1.3, 0.9, 0.7]),
                                  (4, 1, 1)), (1, 3, 2)),
               np.array([np.array([[0.5, 0.55], [0.6, 0.9], [0.9, .4]]),
                         np.array([[7.1, 8.0], [1.0, 2.1], [9.1, 0.1]]),
                         np.array([[0.4, 0.2], [0.34, 0.56], [0.3, 0.6]]),
                         np.array([[0.1, 0.2], [0.4, 0.5], [0.555, 0.76]])]),
               np.array([np.array([[0.4, 0.2], [1.4, 1.5], [0.5, 0.6]]),
                         np.array([[7.1, 8.0], [0.4, 0.9], [9.1, 10]]),
                         np.array([[0.15, 0.52], [0.44, 0.85], [0.5, 0.6]]),
                         np.array([[4.1, 2.0], [0.65, 0.65], [0.25, 0.56]])]),
               np.array([np.array([[0.8, 0.9], [0.4, 0.5], [0.55, 0.66]]),
                         np.array([[0.7, 0.8], [0.2, 0.1], [0, 0.4]]),
                         np.array([[0.1, 0.2], [1.4, 1.5], [0.5, 0.6]]),
                         np.array([[0.4, 0.6], [0.99, 0.44], [0.35, 0.65]])]),
               np.tile(np.reshape(np.array([[0.1, 1.1], [0.4, 1.0],
                                            [0.6, 1.7], [0.9, 2.0]]),
                                  (4, 1, 2)), (1, 3, 1)),
               np.array([np.array([[0.01, 0.02], [0.4, 0.5], [0.05, 0.06]]),
                         np.array([[0.17, 0.18], [0.08, .02], [0.9, 0.10]]),
                         np.array([[1.0, 2.0], [0.04, 0.25], [0.15, 0.16]]),
                         np.array([[0.11, 0.021], [0.044, 0.025],
                                   [0.022, 0.032]])])),
              (np.array([[1.0, 2.1], [0.4, 0.5], [1.6, 0.9]]),
               np.array([0.7, 0.3]), 0.05),
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


@pytest.mark.parametrize('model_vars,params,expected', test_data,
                         ids=['scalar', 'vector', 'matrix', '3D array'])
def test_get_cons(model_vars, params, expected):
    # Test consumption calculation
    r, w, b, b_splus1, n, BQ, net_tax = model_vars
    test_value = household.get_cons(r, w, b, b_splus1, n, BQ, net_tax,
                                    params)

    assert np.allclose(test_value, expected)


# model_vars in order: r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H
# params in order: e, sigma, beta, g_y, chi_b, theta, tau_bq, rho,
# lambdas, j, J, S, analytical_mtrs, etr_params, mtry_params, h_wealth,
# p_wealth, m_wealth, tau_payroll, retire, method
# Define variables for test of SS version
e = np.array([1.0, 0.9, 1.4])
sigma = 2.0
beta = 0.96
g_y = 0.03
chi_b = 1.5
theta = np.array([0.1])
tau_bq = np.array([0.0])
rho = np.array([0.1, 0.2, 1.0])
lambdas = 1.0
j = 0
J = 1
S = 3
analytical_mtrs = False
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
h_wealth = 0.1
m_wealth = 1.0
p_wealth = 0.0
tau_payroll = 0.15
retire = 2
method = 'SS'
test_params_ss = (e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas,
               j, J, S, analytical_mtrs, etr_params[-1,:,:],
               mtry_params[-1,:,:], h_wealth, p_wealth, m_wealth,
               tau_payroll, retire, method)
r = 0.05
w = 1.2
b = np.array([0.0, 0.8, 0.5])
b_splus1 = np.array([0.8, 0.5, 0.1])
b_splus2 = np.array([0.5, 0.1, 0.0])
n = np.array([0.9, 0.8, 0.5])
BQ = 0.1
factor = 120000
T_H = 0.22
test_vars_ss = (r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H)
expected_ss = np.array([9.940309897, -1.004780793, -140.5545878])

# Define variables/params for test of TPI version
method_tpi = 'TPI'
etr_params_tpi = np.empty((S, etr_params.shape[2]))
mtry_params_tpi = np.empty((S, mtry_params.shape[2]))
for i in range(etr_params.shape[2]):
    etr_params_tpi[:, i] = np.diag(np.transpose(etr_params[:, :S, i]))
    mtry_params_tpi[:, i] = np.diag(np.transpose(mtry_params[:, :S, i]))
test_params_tpi = (e, sigma, beta, g_y, chi_b, theta, tau_bq, rho,
                   lambdas, j, J, S, analytical_mtrs, etr_params_tpi,
                   mtry_params_tpi, h_wealth, p_wealth, m_wealth,
                   tau_payroll, retire, method_tpi)
r_vec = np.array([0.05, 0.03, 0.04])
w_vec = np.array([1.2, 0.9, 0.8])
BQ_vec = np.array([0.1, 0.05, 0.15])
T_H_vec = np.array([0.22, 0.15, 0.0])
test_vars_tpi = (r_vec, w_vec, b, b_splus1, b_splus2, n, BQ_vec, factor,
                 T_H_vec)
expected_tpi = np.array([300.977031, 2.719866638, -139.9187228])

test_data = [(test_vars_ss, test_params_ss, expected_ss),
             (test_vars_tpi, test_params_tpi, expected_tpi)]

@pytest.mark.parametrize('model_vars,params,expected', test_data,
                         ids=['SS', 'TPI'])
def test_FOC_savings(model_vars, params, expected):
    # Test FOC condition for household's choice of savings
    r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H = model_vars
    test_value = household.FOC_savings(r, w, b, b_splus1, b_splus2, n,
                                       BQ, factor, T_H, params)

    assert np.allclose(test_value, expected)


def test_FOC_labor():

    assert np.allclose()


# def test_constraint_checker_SS():
#
#     assert np.allclose()
#
#
# def test_constraint_checker_TPI():
#
#     assert np.allclose()
