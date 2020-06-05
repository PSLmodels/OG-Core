import pytest
import numpy as np
import copy
from ogusa import aggregates as aggr
from ogusa.parameters import Specifications


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
    'eta': (np.ones((40, 2)) / (40 * 2)),
    'lambdas': [0.6, 0.4]
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
n = np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
L_loop = np.ones(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
for t in range(p.T):
    for i in range(p.S):
        for k in range(p.J):
            L_loop[t, i, k] *= (p.omega[t, i] * p.lambdas[k] *
                                n[t, i, k] * p.e[i, k])
expected1 = L_loop[-1, :, :].sum()
expected2 = L_loop.sum(1).sum(1)
test_data = [(n[-1, :, :], p, 'SS', expected1), (n, p, 'TPI', expected2)]


@pytest.mark.parametrize('n,p,method,expected', test_data, ids=['SS', 'TPI'])
def test_get_L(n, p, method, expected):
    """
        Test aggregate labor function.
    """
    L = aggr.get_L(n, p, method)
    assert (np.allclose(L, expected))


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
    'eta': (np.ones((40, 2)) / (40 * 2)),
    'lambdas': [0.6, 0.4],
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
b_splus1 = 10 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
K_p1 = 0.9 + np.random.rand(p.T)
K = 0.9 + np.random.rand(p.T)
omega_extended = np.append(p.omega_SS[1:], [0.0])
imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
part2 = (((b_splus1[-1, :, :] *
           np.transpose((omega_extended * imm_extended) *
           p.lambdas)).sum()) / (1 + p.g_n_ss))
aggI_SS = ((1 + p.g_n_ss) * np.exp(p.g_y) * (K_p1[-1] - part2) -
           (1.0 - p.delta) * K[-1])
omega_shift = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
                        axis=1)
imm_shift = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
                      axis=1)
part2 = ((((b_splus1 * np.squeeze(p.lambdas)) *
          np.tile(np.reshape(imm_shift * omega_shift,
                             (p.T, p.S, 1)),
                  (1, 1, p.J))).sum(1).sum(1)) /
         (1 + np.squeeze(np.hstack((p.g_n[1:p.T], p.g_n_ss)))))
aggI_TPI = ((1 + np.squeeze(np.hstack((p.g_n[1:p.T], p.g_n_ss))))
            * np.exp(p.g_y) * (K_p1 - part2) - (1.0 - p.delta) * K)
test_data = [(b_splus1[-1, :, :], K_p1[-1], K[-1], p, 'SS', aggI_SS),
             (b_splus1, K_p1, K, p, 'TPI', aggI_TPI)]
aggI_total_SS = ((1 + p.g_n_ss) * np.exp(p.g_y) * (K[-1]) -
                 (1.0 - p.delta) * K[-1])
aggI_total_TPI = (((1 + np.squeeze(np.hstack((p.g_n[1:p.T], p.g_n_ss))))
                   * np.exp(p.g_y) * K_p1) - (1.0 - p.delta) * K)
test_data = [(b_splus1[-1, :, :], K_p1[-1], K[-1], p, 'SS', aggI_SS),
             (b_splus1, K_p1, K, p, 'TPI', aggI_TPI),
             (None, K[-1], K[-1], p, 'total_ss', aggI_total_SS),
             (None, K_p1, K, p, 'total_tpi', aggI_total_TPI)]


@pytest.mark.parametrize('b_splus1,K_p1,K,p,method,expected', test_data,
                         ids=['SS', 'TPI', 'total_ss', 'total_tpi'])
def test_get_I(b_splus1, K_p1, K, p, method, expected):
    """
        Text aggregate investment function.
    """
    aggI = aggr.get_I(b_splus1, K_p1, K, p, method)
    assert (np.allclose(aggI, expected))


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
    'eta': (np.ones((40, 2)) / (40 * 2)),
    'lambdas': [0.6, 0.4],
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
p.omega_S_preTP = p.omega[0, :]
b = -0.1 + (7 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
omega_extended = np.append(p.omega[:p.T, 1:], np.zeros((p.T, 1)),
                           axis=1)
imm_extended = np.append(p.imm_rates[:p.T, 1:], np.zeros((p.T, 1)),
                         axis=1)
B_test = ((b * np.squeeze(p.lambdas) *
           np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                   (1, 1, p.J))) +
          (b * np.squeeze(p.lambdas) *
           np.tile(np.reshape(omega_extended *
                              imm_extended, (p.T, p.S, 1)),
                   (1, 1, p.J))))
expected1 = B_test[-1, :, :].sum() / (1.0 + p.g_n_ss)
expected2 = B_test.sum(1).sum(1) / (1.0 + np.hstack((p.g_n[1:p.T], p.g_n_ss)))
expected3 = B_test[0, :, :].sum() / (1.0 + p.g_n[0])
test_data = [(b[-1, :, :], p, 'SS', False, expected1),
             (b, p, 'TPI', False, expected2),
             (b[0, :, :], p, 'SS', True, expected3)]


@pytest.mark.parametrize('b,p,method,PreTP,expected', test_data,
                         ids=['SS', 'TPI', 'Pre-TP'])
def test_get_B(b, p, method, PreTP, expected):
    """
    Test aggregate savings function.
    """
    B = aggr.get_B(b, p, method, PreTP)
    assert np.allclose(B, expected)


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
    'eta': (np.ones((40, 2)) / (40 * 2)),
    'lambdas': [0.6, 0.4],
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
p.omega_S_preTP = p.omega[0, :]
# set values for some variables
r = 0.5 + 0.5 * np.random.rand(p.T)
b_splus1 = 0.06 + 7 * np.random.rand(p.T, p.S, p.J)
pop = np.append(p.omega_S_preTP.reshape(1, p.S),
                p.omega[:p.T - 1, :], axis=0)
BQ_presum = ((b_splus1 * np.squeeze(p.lambdas)) *
             np.tile(np.reshape(p.rho * pop, (p.T, p.S, 1)),
                     (1, 1, p.J)))
growth_adj = (1.0 + r) / (1.0 + p.g_n[:p.T])

expected1 = BQ_presum[-1, :, :].sum(0) * growth_adj[-1]
expected2 = BQ_presum[-1, :, 1].sum(0) * growth_adj[-1]
expected3 = (BQ_presum.sum(1) *
             np.tile(np.reshape(growth_adj, (p.T, 1)), (1, p.J)))
expected4 = BQ_presum[:, :, 1].sum(1) * growth_adj
expected5 = BQ_presum[0, :, :].sum(0) * growth_adj[0]
expected6 = BQ_presum[0, :, 1].sum(0) * growth_adj[0]

p2 = copy.deepcopy(p)
p2.use_zeta = True
expected7 = BQ_presum[-1, :, 1].sum() * growth_adj[-1]
expected8 = (BQ_presum[:, :, 1].sum(1) * growth_adj)
expected9 = (BQ_presum.sum(1) *
             np.tile(np.reshape(growth_adj, (p.T, 1)), (1, p.J))).sum(1)
test_data = [(r[-1], b_splus1[-1, :, :], None, p, 'SS', False,
              expected1),
             (r[-1], b_splus1[-1, :, 1], 1, p, 'SS', False, expected2),
             (r, b_splus1, None, p, 'TPI', False, expected3),
             (r, b_splus1[:, :, 1], 1, p, 'TPI', False, expected4),
             (r[0], b_splus1[0, :, :], None, p, 'SS', True, expected5),
             (r[0], b_splus1[0, :, 1], 1, p, 'SS', True, expected6),
             (r[-1], b_splus1[-1, :, 1], 1, p2, 'SS', False, expected7),
             (r, b_splus1[:, :, 1], 1, p2, 'TPI', False, expected8),
             (r, b_splus1, None, p2, 'TPI', False, expected9)]


@pytest.mark.parametrize('r,b_splus1,j,p,method,PreTP,expected',
                         test_data, ids=[
                             'SS, all j', 'SS, one j', 'TPI, all j',
                             'TPI, one j', 'Pre-TP, all j',
                             'Pre-TP, one j', 'Use zeta, SS, one j',
                             'Use zeta, TPI, one j',
                             'Use zeta, TPI, all j'])
def test_get_BQ(r, b_splus1, j, p, method, PreTP, expected):
    """
    Test of aggregate bequest function.
    """
    BQ = aggr.get_BQ(r, b_splus1, j, p, method, PreTP)
    assert np.allclose(BQ, expected)


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
    'eta': (np.ones((40, 2)) / (40 * 2)),
    'lambdas': [0.6, 0.4],
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
# make up some consumption values for testing
c = 0.1 + 0.5 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
aggC_presum = ((c * np.squeeze(p.lambdas)) *
               np.tile(np.reshape(p.omega[:p.T, :], (p.T, p.S, 1)),
                       (1, 1, p.J)))
expected1 = aggC_presum[-1, :, :].sum()
expected2 = aggC_presum.sum(1).sum(1)
test_data = [(c[-1, :, :], p, 'SS', expected1),
             (c, p, 'TPI', expected2)]


@pytest.mark.parametrize('c,p,method,expected', test_data,
                         ids=['SS', 'TPI'])
def test_get_C(c, p, method, expected):
    """
    Test aggregate consumption function.
    """
    C = aggr.get_C(c, p, method)
    assert np.allclose(C, expected)


p = Specifications()
dim4 = 12
new_param_values = {
    'T': 30,
    'S': 20,
    'J': 2,
    'eta': (np.ones((20, 2)) / (20 * 2)),
    'lambdas': [0.6, 0.4],
    'tau_bq': [0.17],
    'tau_payroll': [0.5],
    'h_wealth': [0.1],
    'p_wealth': [0.2],
    'm_wealth': [1.0],
    'cit_rate': [0.2],
    'delta_tau_annual': [float(1 - ((1 - 0.0975) **
                                    (20 / (p.ending_age -
                                           p.starting_age))))]
}
p.update_specifications(new_param_values)
# make up some consumption values for testing
# Assign values to variables for tests
random_state = np.random.RandomState(10)
r = 0.067 + (0.086 - 0.067) * random_state.rand(p.T)
w = 0.866 + (0.927 - 0.866) * random_state.rand(p.T)
b = 6.94 * random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
c = np.ones((p.T, p.S, p.J)) * 2.2
n = (0.191 + (0.503 - 0.191) *
     random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
BQ = (0.032 + (0.055 - 0.032) *
      random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
bq = BQ / p.lambdas.reshape(1, 1, p.J)
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

p3 = Specifications()
dim4 = 12
new_param_values3 = {
    'T': 30,
    'S': 20,
    'J': 2,
    'eta': (np.ones((20, 2)) / (20 * 2)),
    'lambdas': [0.6, 0.4],
    'tau_bq': [0.17],
    'tau_payroll': [0.5],
    'h_wealth': [0.1],
    'p_wealth': [0.2],
    'm_wealth': [1.0],
    'cit_rate': [0.2],
    'replacement_rate_adjust': [1.5, 1.5, 1.5, 1.6, 1.0],
    'delta_tau_annual': [float(1 - ((1 - 0.0975) **
                                    (20 / (p3.ending_age -
                                           p3.starting_age))))]
}
p3.update_specifications(new_param_values3)
p3.e = p.e
p3.omega = p.omega
p3.omega_SS = p.omega_SS
expected1 = 0.5370699180829722
expected2 = np.array(
            [0.50260639, 0.48109794, 0.5059882, 0.50527725, 0.57985594,
             0.59290848, 0.52345093, 0.52404633, 0.54382821, 0.55482053,
             0.51400707, 0.50237146, 0.4868004, 0.55008867, 0.49817611,
             0.58803381, 0.54893319, 0.5484411, 0.66892545, 0.56201835,
             0.58842445, 0.54289658, 0.50051496, 0.47262093, 0.50623643,
             0.55579704, 0.49693837, 0.56426605, 0.51268459,
             0.52148645])
expected3 = np.array(
            [0.471705, 0.45212442, 0.47401651, 0.47099882, 0.57985594,
             0.59290848, 0.52345093, 0.52404633, 0.54382821, 0.55482053,
             0.51400707, 0.50237146, 0.4868004, 0.55008867, 0.49817611,
             0.58803381, 0.54893319, 0.5484411, 0.66892545, 0.56201835,
             0.58842445, 0.54289658, 0.50051496, 0.47262093, 0.50623643,
             0.55579704, 0.49693837, 0.56426605, 0.51268459,
             0.52148645])
test_data = [(r[0], w[0], b[0, :, :], n[0, :, :], bq[0, :, :],
              c[0, :, :], Y[0], L[0], K[0], factor, theta,
              etr_params[-1, :, :, :], p, 'SS', expected1),
             (r, w, b, n, bq, c, Y, L, K, factor, theta, etr_params, p,
              'TPI', expected2),
             (r, w, b, n, bq, c, Y, L, K, factor, theta, etr_params, p3,
              'TPI', expected3)]


@pytest.mark.parametrize(
    'r,w,b,n,bq,c,Y,L,K,factor,theta,etr_params,p,method,expected',
    test_data, ids=['SS', 'TPI', 'TPI, replace rate adjust'])
def test_revenue(r, w, b, n, bq, c, Y, L, K, factor, theta, etr_params,
                 p, method, expected):
    """
    Test aggregate revenue function.
    """
    revenue, _, agg_pension_outlays, _, _, _, _, _, _ = aggr.revenue(
        r, w, b, n, bq, c, Y, L, K, factor, theta, etr_params, p, method)
    expected += agg_pension_outlays

    assert(np.allclose(revenue, expected))


test_data = [(0.04, 0.02, 2.0, 4.0, 0.026666667),
             (np.array([0.05, 0.03]), np.array([0.02, 0.01]),
              np.array([3.0, 4.0]), np.array([7.0, 6.0]),
              np.array([0.029, 0.018])),
             (0.04, 0.02, 2.0, 0.0, 0.04)]


@pytest.mark.parametrize('r,r_gov,B,D,expected', test_data,
                         ids=['scalar', 'vector', 'no debt'])
def test_get_r_hh(r, r_gov, B, D, expected):
    """
    Test function to compute interet rate on household portfolio.
    """
    r_hh_test = aggr.get_r_hh(r, r_gov, B, D)

    assert(np.allclose(r_hh_test, expected))


def test_resource_constraint():
    """
    Test resource constraint equation.
    """
    p = Specifications()
    p.delta = 0.05
    Y = np.array([48, 55, 2, 99, 8])
    C = np.array([33, 44, 0.4, 55, 6])
    G = np.array([4, 5, 0.01, 22, 0])
    I = np.array([20, 5, 0.6, 10, 1])
    K_f = np.array([0, 0, 0.2, 3, 0.05])
    new_borrowing_f = np.array([0, 0.1, 0.3, 4, 0.5])
    debt_service_f = np.array([0.1, 0.1, 0.3, 2, 0.02])
    r = np.array([0.03, 0.04, 0.03, 0.06, 0.01])
    expected = np.array([-9.1, 1, 0.974, 13.67, 1.477])
    test_RC = aggr.resource_constraint(Y, C, G, I, K_f, new_borrowing_f,
                                       debt_service_f, r, p)

    assert(np.allclose(test_RC, expected))


def test_get_K_splits():
    '''
    Test of the get_K_splits function.
    '''
    B = 2.2
    K_demand_open = 0.5
    D_d = 1.1
    zeta_K = 0.2

    expected_K_d = 1.1
    expected_K_f = 0.2 * (0.5 - (2.2 - 1.1))
    expected_K = expected_K_d + expected_K_f

    test_K, test_K_d, test_K_f = aggr.get_K_splits(
        B, K_demand_open, D_d, zeta_K)

    np.allclose(test_K, expected_K)
    np.allclose(test_K_d, expected_K_d)
    np.allclose(test_K_f, expected_K_f)


def test_get_K_splits_negative_K_d():
    '''
    Test of the get_K_splits function for case where K_d < 0.
    '''
    B = 2.2
    K_demand_open = 0.5
    D_d = 2.3
    zeta_K = 0.2

    expected_K_d = 0.05
    expected_K_f = 0.2 * (0.5 - (2.2 - 2.3))
    expected_K = expected_K_d + expected_K_f

    test_K, test_K_d, test_K_f = aggr.get_K_splits(
        B, K_demand_open, D_d, zeta_K)

    np.allclose(test_K, expected_K)
    np.allclose(test_K_d, expected_K_d)
    np.allclose(test_K_f, expected_K_f)
