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
    'S': 3
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
    'S': 3
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
    'delta_annual': [0.25],
    'tau_b': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 1.2
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
# assign values for Y and K variables
Y1 = np.array([2.0])
K1 = np.array([1.0])
expected1 = np.array([0.370449359])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.5,
    'tau_b': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 0.5,
    'delta_annual': [0.5]
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.675])
p3 = Specifications()
new_param_values3 = {
    'Z': [0.5],
    'gamma': 0.5,
    'tau_b': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 1.0,
    'delta_annual': [0.5]
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([0.175])
p4 = Specifications()
new_param_values4 = {
    'Z': [0.5],
    'gamma': 0.5,
    'tau_b': [0.5],
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'delta_annual': [0.5],
    'T': 3,
    'S': 3
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
Y4 = np.array([3.0, 3.2, 3.8])
K4 = np.array([1.8, 1.2, 1.0])
expected4 = np.array([-0.117344327, 0.066121991, 0.260484002])

p5 = Specifications()
new_param_values5 = {
    'Z': [1.5, 2.5, 0.6],
    'gamma': 0.5,
    'tau_b': [0.2, 0.0, 0.5],
    'delta_tau_annual': [0.35, 0.2, 0.1],
    'epsilon': 1.2,
    'delta_annual': [0.5],
    'T': 3,
    'S': 3
}
# update parameters instance with new values for test
p5.update_specifications(new_param_values5)
expected5 = np.array([-0.064719663, 0.480604113, 0.25383862])


@pytest.mark.parametrize('Y,K,p,method,expected',
                         [(Y1, K1, p1, 'SS', expected1),
                          (Y1, K1, p2, 'SS', expected2),
                          (Y1, K1, p3, 'SS', expected3),
                          (Y4, K4, p4, 'TPI', expected4),
                          (Y4, K4, p5, 'TPI', expected5)],
                         ids=['epsilon=1.2,SS', 'epsilon=0.5,SS',
                              'epsilon=1.0,SS', 'epsilon=1.2,TP',
                              'epsilon=1.2,TP,varyParams'])
def test_get_r(Y, K, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    r = firm.get_r(Y, K, p, method)
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
    'S': 3
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
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'tau_b': [0.0357]
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.01])
expected1 = np.array([10.30175902])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'tau_b': [0.0357]
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([215.1799075])
p3 = Specifications()
new_param_values3 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'epsilon': 1.0,
    'tau_b': [0.0357]
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([10.33169079])
p4 = Specifications()
new_param_values4 = {
    'Z': [0.5, 0.1, 1.1],
    'gamma': 0.4,
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'epsilon': 0.5,
    'tau_b': [0.0357],
    'T': 3,
    'S': 3
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
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'tau_b': [0.0357]
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.04])
expected1 = np.array([1.265762107])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'epsilon': 1.0,
    'tau_b': [0.0357]
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.550887455])
p3 = Specifications()
new_param_values3 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'tau_b': [0.0357]
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([2.855428923])
p4 = Specifications()
new_param_values4 = {
    'Z': [0.5, 1.0, 4.0],
    'gamma': 0.4,
    'delta_annual': [0.05],
    'delta_tau_annual': [0.35],
    'epsilon': 1.2,
    'tau_b': [0.0357],
    'T': 3,
    'S': 3
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
    'tau_b': [0.75],
    'delta_annual': [0.15],
    'delta_tau_annual': [0.03],
    'Z': [2.0],
    'epsilon': 1.2
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
L1 = np.array([2.0])
r1 = np.array([1.0])
expected1 = np.array([0.325726586])
p2 = Specifications()
new_param_values2 = {
    'gamma': 0.5,
    'tau_b': [0.75],
    'delta_annual': [0.15],
    'delta_tau_annual': [0.03],
    'Z': [2.0],
    'epsilon': 1.0
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.098327933])
p3 = Specifications()
new_param_values3 = {
    'gamma': 0.5,
    'epsilon': 0.4,
    'Z': [4.0],
    'tau_b': [0.0],
    'delta_tau_annual': [0.5],
    'delta_annual': [0.05]
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
    'delta_annual': [0.05],
    'tau_b': [0.5],
    'T': 3,
    'S': 3
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
L4 = np.array([2.0, 2.0, 2.0])
r4 = np.array([1.0, 1.0, 1.0])
expected4 = np.array([2.801139363, 2.303692012, 2.303692012])


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
