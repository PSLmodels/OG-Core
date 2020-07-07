import pytest
from ogusa import firm
import numpy as np
from ogusa.parameters import Specifications


p1 = Specifications()
new_param_values = {
    'Z': [2.0],
    'gamma': 0.5,
    'epsilon': 1.0
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values)
L1 = np.array([4.0])
K1 = np.array([9.0])
expected1 = np.array([12.0])
p2 = Specifications()
new_param_values2 = {
    'Z': [2.0],
    'gamma': 0.5,
    'epsilon': 0.2
}
expected2 = np.array([18.84610765])
p3 = Specifications()
new_param_values3 = {
    'Z': [2.0],
    'gamma': 0.5,
    'epsilon': 1.2
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
L3 = np.array([1 / 12.0])
K3 = np.array([1 / 4.0])
expected3 = np.array([0.592030917])
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
p4 = Specifications()
new_param_values4 = {
    'Z': [2.0],
    'gamma': 0.5,
    'epsilon': 1.0,
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p4.J)) / (3 * p4.J))
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
L4 = np.array([4.0, 4.0, 4.0])
K4 = np.array([9.0, 9.0, 9.0])
expected4 = np.array([12.0, 12.0, 12.0])
p5 = Specifications()
new_param_values5 = {
    'Z': [1.5, 2.5, 0.6],
    'gamma': 0.5,
    'epsilon': 1.0,
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p5.J)) / (3 * p5.J))
}
# update parameters instance with new values for test
p5.update_specifications(new_param_values5)
expected5 = np.array([9.0, 15.0, 3.6])


@pytest.mark.parametrize('K,L,p,method,expected',
                         [(K1, L1, p1, 'SS', expected1),
                          (K1, L1, p2, 'SS', expected2),
                          (K3, L3, p3, 'SS', expected3),
                          (K4, L4, p4, 'TPI', expected4),
                          (K4, L4, p5, 'TPI', expected5)],
                         ids=['epsilon=1.0,SS', 'epsilon=0.2,SS',
                              'epsilon=1.2,SS', 'epsilon=1.0,TP',
                              'epsilon=1.0,TP,varyZ'])
def test_get_Y(K, L, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    Y = firm.get_Y(K, L, p, method)
    assert (np.allclose(Y, expected, atol=1e-6))


p1 = Specifications()
new_param_values1 = {
    'Z': [0.5],
    'gamma': 0.5,
    'delta_annual': 0.25,
    'cit_rate': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 1.2
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
p1.g_y = 0.03
p1.g_n_ss = 0.02
# assign values for Y and K variables
Y1 = np.array([2.0])
K1 = np.array([1.0])
V1 = K1
Kp1_1 = K1
Vp1_1 = Kp1_1
r1 = 0.05
z1 = 0.074375
K_tau1 = 0.714285714
X1 = 0.053125
Xp1_1 = 0.053125
expected1 = np.array([0.610908068])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.5,
    'cit_rate': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 0.5,
    'delta_annual': 0.5
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
p2.g_y = 0.03
p2.g_n_ss = 0.02
X2 = 0.10625
Xp1_2 = 0.10625
expected2 = np.array([1.547552448])
p3 = Specifications()
new_param_values3 = {
    'Z': [0.5],
    'gamma': 0.5,
    'cit_rate': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 1.0,
    'delta_annual': 0.5
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
p3.g_y = 0.03
p3.g_n_ss = 0.02
X3 = 0.10625
Xp1_3 = 0.10625
expected3 = np.array([0.523776224])
p4 = Specifications()
new_param_values4 = {
    'Z': [0.5],
    'gamma': 0.5,
    'cit_rate': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'delta_annual': 0.5,
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p4.J)) / (3 * p4.J))
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
p4.g_y = 0.03
p4.g_n_ss = 0.02
p4.g_n = np.ones(p4.T + p4.S) * 0.02
Y4 = np.array([3.0, 3.2, 3.8])
K4 = np.array([1.8, 1.2, 1.0])
V4 = K4
Kp1_4 = np.array([1.2, 1.0, 1.0])
Vp1_4 = Kp1_4
X4 = np.array([0.145715708, 0.085087555, 0.085086933])
Xp1_4 = np.array([0.085087555, 0.085086933, 0.080953171])
expected4 = np.array([-0.238048338, 0.115078721, 0.521122772])

p5 = Specifications()
new_param_values5 = {
    'Z': [1.5, 2.5, 0.6],
    'gamma': 0.5,
    'cit_rate': [0.2, 0.0, 0.5],
    'delta_tau_annual': [0.35, 0.2, 0.1],
    'epsilon': 1.2,
    'delta_annual': 0.5,
    'psi': 1.4,
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p5.J)) / (3 * p5.J))
}
# update parameters instance with new values for test
p5.update_specifications(new_param_values5)
p5.g_y = 0.03
p5.g_n_ss = 0.02
p5.g_n = np.ones(p4.T + p4.S) * 0.02
p5.mu = 1.051063615
Y5 = np.array([3.0, 3.2, 3.8])
K5 = np.array([1.8, 1.2, 1.0])
V5 = K5
Kp1_5 = np.array([1.2, 1.0, 1.0])
Vp1_5 = Kp1_5
X5 = np.array([0.058286283, 0, 0.085048166])
Xp1_5 = np.array([0, 0.085048166, 0.085878385])
expected5 = np.array([-0.454225476, 0.259199506, 0.567988021])

p6 = Specifications()
new_param_values6 = {
    'Z': [0.5],
    'gamma': 0.5,
    'delta_annual': 0.25,
    'cit_rate': [0.5],
    'delta_tau_annual': [0.0],
    'epsilon': 1.2
}
# update parameters instance with new values for test
p6.update_specifications(new_param_values6)
# assign values for Y and K variables
Y6 = np.array([2.0])
K6 = np.array([1.0])
expected6 = np.array([0.565172327])


@pytest.mark.parametrize('Y,K,Kp1,V,Vp1,X,Xp1,p,method,expected', [
    (Y1, K1, Kp1_1, V1, Vp1_1, X1, Xp1_1, p1, 'SS', expected1),
    (Y1, K1, Kp1_1, V1, Vp1_1, X2, Xp1_2, p2, 'SS', expected2),
    (Y1, K1, Kp1_1, V1, Vp1_1, X3, Xp1_3, p3, 'SS', expected3),
    (Y4, K4, Kp1_4, V4, Vp1_4, X4, Xp1_4, p4, 'TPI', expected4),
    (Y5, K5, Kp1_5, V5, Vp1_5, X5, Xp1_5, p5, 'TPI', expected5),
    (Y6, K6, Kp1_1, K6, Kp1_1, 0.0, 0.0, p6, 'SS', expected6)],
                         ids=['epsilon=1.2,SS', 'epsilon=0.5,SS',
                              'epsilon=1.0,SS', 'epsilon=1.2,TP',
                              'epsilon=1.2,TP,varyParams',
                              'epsilon=1.2,SS,delta_tau=0'])
def test_get_r(Y, K, Kp1, V, Vp1, X, Xp1, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    r = firm.get_r(Y, K, Kp1, V, Vp1, X, Xp1, p, method)
    assert (np.allclose(r, expected))


p1 = Specifications()
new_param_values1 = {
    'Z': [0.5],
    'gamma': 0.5,
    'epsilon': 0.2
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
Y1 = np.array([2.0])
L1 = np.array([1.0])
expected1 = np.array([16.])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.5,
    'epsilon': 1.5
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.793700526])
p3 = Specifications()
new_param_values3 = {
    'Z': [0.5],
    'gamma': 0.5,
    'epsilon': 1.0
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([1.0])
p4 = Specifications()
new_param_values4 = {
    'Z': [0.5, 0.47],
    'gamma': 0.5,
    'epsilon': 1.2,
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p4.J)) / (3 * p4.J))
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
Y4 = np.array([2.0, 2.0, 2.0])
L4 = np.array([1.0, 1.0, 1.0])
expected4 = np.array([0.890898718, 0.881758476, 0.881758476])


@pytest.mark.parametrize('Y,L,p,method,expected',
                         [(Y1, L1, p1, 'SS', expected1),
                          (Y1, L1, p2, 'SS', expected2),
                          (Y1, L1, p3, 'SS', expected3),
                          (Y4, L4, p4, 'TPI', expected4)],
                         ids=['epsilon=0.2,SS', 'epsilon=1.5,SS',
                              'epsilon=1.0,SS', 'epsilon=1.2,TP'])
def test_get_w(Y, L, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    w = firm.get_w(Y, L, p, method)
    assert (np.allclose(w, expected, atol=1e-6))


p1 = Specifications()
new_param_values1 = {
    'Z': [0.5],
    'gamma': 0.4,
    'epsilon': 0.8,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.01])
expected1 = np.array([10.30175902])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([215.1799075])
p3 = Specifications()
new_param_values3 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'epsilon': 1.0,
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([10.33169079])
p4 = Specifications()
new_param_values4 = {
    'Z': [0.5, 0.1, 1.1],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'epsilon': 0.5,
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)],
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p4.J)) / (3 * p4.J))
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
r4 = np.array([0.01, 0.04, 0.55])
expected4 = np.array([0.465031434, -0.045936078, 0.575172024])


@pytest.mark.parametrize('r,p,method,expected',
                         [(r1, p1, 'SS', expected1),
                          (r1, p2, 'SS', expected2),
                          (r1, p3, 'SS', expected3),
                          (r4, p4, 'TPI', expected4)],
                         ids=['epsilon=0.8,SS', 'epsilon=1.2,SS',
                              'epsilon=1.0,SS', 'epsilon=0.5,TP'])
def test_get_KLratio_from_r(r, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    KLratio = firm.get_KLratio_from_r(r, p, method)
    assert (np.allclose(KLratio, expected, atol=1e-6))


p1 = Specifications()
new_param_values1 = {
    'Z': [0.5],
    'gamma': 0.4,
    'epsilon': 0.8,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.04])
expected1 = np.array([1.265762107])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'epsilon': 1.0,
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.550887455])
p3 = Specifications()
new_param_values3 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([2.855428923])
p4 = Specifications()
new_param_values4 = {
    'Z': [0.5, 1.0, 4.0],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)],
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p4.J)) / (3 * p4.J))
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
r4 = np.array([0.04, 0.04, 0.04])
expected4 = np.array([0.380178134, 1.19149279, 17.8375083])


@pytest.mark.parametrize('r,p,method,expected',
                         [(r1, p1, 'SS', expected1),
                          (r1, p2, 'SS', expected2),
                          (r1, p3, 'SS', expected3),
                          (r4, p4, 'TPI', expected4)],
                         ids=['epsilon=0.8,SS', 'epsilon=1.0,SS',
                              'epsilon=1.2,SS', 'epsilon=1.2,TP'])
def test_get_w_from_r(r, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    w = firm.get_w_from_r(r, p, method)
    assert (np.allclose(w, expected, atol=1e-6))


p1 = Specifications()
new_param_values1 = {
    'gamma': 0.5,
    'cit_rate': [0.75],
    'delta_annual': 0.15,
    'delta_tau_annual': [0.03],
    'Z': [2.0],
    'epsilon': 1.2
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
L1 = np.array([2.0])
r1 = np.array([1.0])
expected1 = np.array([5.74454599])
p2 = Specifications()
new_param_values2 = {
    'gamma': 0.5,
    'cit_rate': [0.75],
    'delta_annual': 0.15,
    'delta_tau_annual': [0.03],
    'Z': [2.0],
    'epsilon': 1.0
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([1.1589348])
p3 = Specifications()
new_param_values3 = {
    'gamma': 0.5,
    'epsilon': 0.4,
    'Z': [4.0],
    'cit_rate': [0.0],
    'delta_tau_annual': [0.5],
    'delta_annual': 0.05
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([4.577211711])
p4 = Specifications()
new_param_values4 = {
    'gamma': 0.5,
    'epsilon': 0.4,
    'Z': [4.0, 3.0],
    'delta_tau_annual': [0.5],
    'delta_annual': 0.05,
    'cit_rate': [0.5],
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p4.J)) / (3 * p4.J))
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
L4 = np.array([2.0, 2.0, 2.0])
r4 = np.array([1.0, 1.0, 1.0])
expected4 = np.array([3.39707089, 2.85348453, 2.85348453])


@pytest.mark.parametrize('L,r,p,method,expected',
                         [(L1, r1, p1, 'SS', expected1),
                          (L1, r1, p2, 'SS', expected2),
                          (L1, r1, p3, 'SS', expected3),
                          (L4, r4, p4, 'TPI', expected4)],
                         ids=['epsilon=1.2,SS', 'epsilon=1.0,SS',
                              'epsilon=0.4,SS', 'epsilon=0.4,TP'])
def test_get_K(L, r, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    K = firm.get_K(L, r, p, method)
    assert (np.allclose(K, expected, atol=1e-6))


p1 = Specifications()
p1.psi = 4.0
p1.g_n_ss = 0.01
p1.g_y = 0.03
p1.delta = 0.05
p1.mu = 0.090759079
K_1 = 5
Kp1_1 = 5
expected_Psi_1 = 0.0
expected_dPsidK_1 = 0.0
expected_dPsidKp1_1 = 0.0

p2 = Specifications()
p2.psi = 2.0
p2.g_n_ss = 0.0
p2.g_y = 0.03
p2.delta = 0.05
p2.mu = 0.05
K_2 = 6
Kp1_2 = 6
expected_Psi_2 = 0.011527985
expected_dPsidK_2 = -0.122196836
expected_dPsidKp1_2 = 0.102296044


p3 = Specifications()
p3.psi = 4.0
p3.g_n_ss = 0.0
p3.g_n = np.array([-0.01, 0.02, 0.03, 0.0])
p3.T = 3
p3.g_y = 0.04
p3.delta = 0.05
p3.mu = 0.05
K_3 = np.array([4, 4.5, 5.5])
Kp1_3 = np.array([4.5, 5.5, 5])
expected_Psi_3 = np.array([0.309124823, 0.534408906, -1.520508524])
expected_dPsidK_3 = np.array([-0.805820108, -0.846107505, 2.657143029])
expected_dPsidKp1_3 = np.array([0.479061039, 0.43588367, -62.31580895])


@pytest.mark.parametrize('K,Kp1,p,method,expected',
                         [(K_1, Kp1_1, p1, 'SS', expected_Psi_1),
                          (K_2, Kp1_2, p2, 'SS', expected_Psi_2),
                          (K_3, Kp1_3, p3, 'TPI', expected_Psi_3)],
                         ids=['Zero cost', 'Non-zero cost', 'TPI'])
def test_adj_cost(K, Kp1, p, method, expected):
    '''
    Test of the firm capital adjustment cost function.
    '''
    test_val = firm.adj_cost(K, Kp1, p, method)
    assert np.allclose(test_val, expected)


@pytest.mark.parametrize('K,Kp1,p,method,expected',
                         [(K_1, Kp1_1, p1, 'SS', expected_dPsidK_1),
                          (K_2, Kp1_2, p2, 'SS', expected_dPsidK_2),
                          (K_3, Kp1_3, p3, 'TPI', expected_dPsidK_3)],
                         ids=['Zero cost', 'Non-zero cost', 'TPI'])
def test_adj_cost_dK(K, Kp1, p, method, expected):
    '''
    Test of the firm capital adjustment cost function.
    '''
    test_val = firm.adj_cost_dK(K, Kp1, p, method)
    assert np.allclose(test_val, expected)


@pytest.mark.parametrize('K,Kp1,p,method,expected',
                         [(K_1, Kp1_1, p1, 'SS', expected_dPsidKp1_1),
                          (K_2, Kp1_2, p2, 'SS', expected_dPsidKp1_2),
                          (K_3, Kp1_3, p3, 'TPI', expected_dPsidKp1_3)],
                         ids=['Zero cost', 'Non-zero cost', 'TPI'])
def test_adj_cost_dKp1(K, Kp1, p, method, expected):
    '''
    Test of the firm capital adjustment cost function.
    '''
    test_val = firm.adj_cost_dKp1(K, Kp1, p, method)
    assert np.allclose(test_val, expected)


p1 = Specifications()
p1.tau_b = np.array([0.35, 0.35, 0.35, 0.35])
p1.delta_tau = np.array([0.05, 0.05, 0.05, 0.05])
p1.T = 4
p1.S = 3
r1 = 0.04
method1 = 'SS'
expected_val1 = 0.194444444
p2 = Specifications()
p2.tau_b = np.array([0.35, 0.35, 0.35, 0.35])
p2.delta_tau = np.array([0.05, 0.05, 0.05, 0.05])
p2.T = 4
p2.S = 3
r2 = np.array([0.04, 0.04, 0.04, 0.04])
method2 = 'TPI'
expected_val2 = np.array([0.194444444, 0.194444444, 0.194444444,
                          0.194444444, 0.194444444, 0.194444444])
p3 = Specifications()
p3.tau_b = np.array([0.35, 0.32, 0.32, 0.32])
p3.delta_tau = np.array([0.05, 0.055, 0.055, 0.06])
p3.T = 4
p3.S = 3
r3 = np.array([0.04, 0.045, 0.05, 0.05])
method3 = 'TPI'
expected_val3 = np.array([0.174058286, 0.173852814, 0.174545455,
                          0.174545455, 0.174545455, 0.174545455])


@pytest.mark.parametrize('r,method,p,expected_val', [
    (r1, method1, p1, expected_val1), (r2, method2, p2, expected_val2),
    (r3, method3, p3, expected_val3)],
                         ids=['SS', 'TPI, constant params',
                              'TPI, varying params'])
def test_get_NPV_depr(r, method, p, expected_val):
    '''
    Test of firm.get_NPV_depr() function.
    '''
    test_val = firm.get_NPV_depr(r, p, method)

    assert np.allclose(test_val, expected_val)


def test_get_K_from_Y():
    '''
    test of firm.get_K_from_Y()
    '''
    expected_val = 12.95001519647885
    r = 0.05
    Y = 3.8
    p = Specifications()
    test_val = firm.get_K_from_Y(Y, r, p, 'SS')

    assert np.allclose(test_val, expected_val)


p1 = Specifications()
K0_1 = 1.3
V1 = 2
K_tau0_1 = 1.2
z1 = 0.08
p1.delta = 0.05
p1.delta_tau = np.ones(p1.T + p1.S) * 0.065
p1.tau_b = np.ones(p1.T + p1.S) * 0.21
p1.mu = 0.06
p1.psi = 1.2
expected_val1 = (2.037617555, 1.567398119)
p2 = Specifications()
p2.delta = 0.05
p2.delta_tau = np.ones(p2.T + p2.S) * 0.0
p2.tau_b = np.ones(p2.T + p2.S) * 0.21
p2.mu = 0.06
p2.psi = 1.2
expected_val2 = (2.0, 0.0)
K0_3 = 1.3
V3 = np.array([2, 2.2, 2.3])
K_tau0_3 = 1.2
z3 = np.array([0.08, 0.08, 0.08])
p3 = Specifications()
p3.T = 3
p3.S = 3
p3.delta = 0.05
p3.delta_tau = np.ones(p3.T + p3.S) * 0.065
p3.g_n = np.ones(p3.T + p3.S) * 0.02
p3.g_y = 0.03
p3.tau_b = np.ones(p3.T + p3.S) * 0.21
p3.mu = 0.06
p3.psi = 1.2
expected_val3 = (np.array([1.3, 1.6633939, 1.9097573]),
                 np.array([1.2, 3.058132822, 3.662300635]))


@pytest.mark.parametrize('K0,V,K_tau0,z,p,method,expected_val', [
    (K0_1, V1, K_tau0_1, z1, p1, 'SS', expected_val1),
    (K0_1, V1, K_tau0_1, z1, p2, 'SS', expected_val2),
    (K0_3, V3, K_tau0_3, z3, p3, 'TPI', expected_val3)],
                         ids=['SS', 'SS,delta_tau=0', 'TPI'])
def test_get_K_demand(K0, V, K_tau0, z, p, method, expected_val):
    '''
    Test of firm.get_K_demand()
    '''
    test_val = firm.get_K_demand(K0, V, K_tau0, z, p, method)

    assert np.allclose(test_val, expected_val)


def test_get_K_tau_p1():
    '''
    Test of firm.get_K_tau_p1()
    '''
    expected_val = 1.978947755
    K_tau = 1.2
    I = 1.0
    delta_tau = 0.1
    g_np1 = 0.02
    g_y = 0.03
    test_val = firm.get_K_tau_p1(K_tau, I, delta_tau, g_np1, g_y)

    assert np.allclose(test_val, expected_val)


def test_FOC_I():
    '''
    Test of firm.FOC_I()
    '''
    expected_val = 0.186416301
    p = Specifications()
    Kp1 = 3.8
    K = 3.2
    Vp1 = 4.5
    K_tau = 2.4
    z = 0.19
    p.delta = 0.05
    p.psi = 1.2
    p.mu = 0.2
    p.tau_b = np.ones(p.T + p.S) * 0.35
    p.delta_tau = np.ones(p.T + p.S) * 0.06
    p.g_n = np.ones(p.T + p.S) * 0.01

    args = (K, Vp1, K_tau, z, p, 0)
    test_val = firm.FOC_I(Kp1, *args)

    assert np.allclose(test_val, expected_val)


def test_get_X():
    '''
    Test of firm.get_X()
    '''
    expected_val = 0.4
    Z = 0.2
    K_tau = 2.0

    test_val = firm.get_X(Z, K_tau)

    assert np.allclose(test_val, expected_val)


def test_get_q():
    '''
    Test of firm.get_q()
    '''
    expected_val = 1.3
    K = 1.0
    V = 2.0
    X = 0.7

    test_val = firm.get_q(K, V, X)

    assert np.allclose(test_val, expected_val)
