import pytest
import numpy as np
from ogusa import aggregates as aggr
from ogusa.parameters import Specifications


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
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


@pytest.mark.parametrize('b_splus1,K_p1,K,p,method,expected', test_data,
                         ids=['SS', 'TPI'])
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
    'lambdas': [0.6, 0.4],
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
b = -0.1 + (7 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
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
expected1 = K_test[-1, :, :].sum() / (1.0 + p.g_n_ss)
expected2 = K_test.sum(1).sum(1) / (1.0 + np.hstack((p.g_n[1:p.T], p.g_n_ss)))
test_data = [(b[-1, :, :], p, 'SS', expected1),
             (b, p, 'TPI', expected2)]


@pytest.mark.parametrize('b,p,method,expected', test_data,
                         ids=['SS', 'TPI'])
def test_get_K(b, p, method, expected):
    """
    Test aggregate capital function.
    """
    K = aggr.get_K(b, p, method, False)
    assert np.allclose(K, expected)


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
    'lambdas': [0.6, 0.4],
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
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
test_data = [(r[-1], b_splus1[-1, :, :], None, p, 'SS', expected1),
             (r[-1], b_splus1[-1, :, 1], 1, p, 'SS', expected2),
             (r, b_splus1, None, p, 'TPI', expected3),
             (r, b_splus1[:, :, 1], 1, p, 'TPI', expected4)]


@pytest.mark.parametrize('r,b_splus1,j,p,method,expected', test_data,
                         ids=['SS, all j', 'SS, one j', 'TPI, all j',
                              'TPI, one j'])
def test_get_BQ(r, b_splus1, j, p, method, expected):
    """
    Test of aggregate bequest function.
    """
    BQ = aggr.get_BQ(r, b_splus1, j, p, method, False)
    assert np.allclose(BQ, expected)


p = Specifications()
new_param_values = {
    'T': 160,
    'S': 40,
    'J': 2,
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
    'lambdas': [0.6, 0.4],
    'tau_bq': [0.17],
    'tau_payroll': [0.5],
    'h_wealth': [0.1],
    'p_wealth': [0.2],
    'm_wealth': [1.0],
    'tau_b': [0.2],
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
expected1 = 0.5562489534339288
expected2 = [0.52178543, 0.49977116, 0.52015768, 0.52693363, 0.59695398,
             0.61360011, 0.54679056, 0.54096669, 0.56301133, 0.5729165,
             0.52734917, 0.51432562, 0.50060814, 0.5633982,  0.51509517,
             0.60189683, 0.56766507, 0.56439768, 0.68919173, 0.57765917,
             0.60292137, 0.56621788, 0.51913478, 0.48952262, 0.52142782,
             0.5735005, 0.51166718, 0.57939994, 0.52585236, 0.53767652]
test_data = [(r[0], w[0], b[0, :, :], n[0, :, :], BQ[0, :, :], Y[0],
              L[0], K[0], factor, theta, etr_params[-1, :, :, :], p,
              'SS', expected1),
             (r, w, b, n, BQ, Y, L, K, factor, theta, etr_params, p,
              'TPI', expected2)]


@pytest.mark.parametrize('r,w,b,n,BQ,Y,L,K,factor,theta,etr_params,p,' +
                         'method,expected', test_data, ids=['SS', 'TPI'])
def test_revenue(r, w, b, n, BQ, Y, L, K, factor, theta, etr_params, p,
                 method, expected):
    """
    Test aggregate revenue function.
    """
    revenue, _, _, _, _, _ = aggr.revenue(r, w, b, n, BQ, Y, L, K,
                                          factor, theta, etr_params, p,
                                          method)
    assert(np.allclose(revenue, expected))
