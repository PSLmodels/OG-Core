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
L1 = np.array([4.0, 4.0])
K1 = np.array([9.0, 9.0])
expected1 = np.array([12.0, 12.0])
p2 = Specifications()
new_param_values2 = {
    'Z': [2.0],
    'gamma': 0.5,
    'epsilon': 0.2
}
expected2 = np.array([18.84610765, 18.84610765])
p3 = Specifications()
new_param_values3 = {
    'Z': [2.0],
    'gamma': 0.5,
    'epsilon': 0.2
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
L3 = np.array([1 / 12.0, 1 / 12.0])
K3 = np.array([1 / 4.0, 1 / 4.0])
expected3 = np.array([0.39518826, 0.39518826])
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


@pytest.mark.parametrize('K,L,p,method,expected',
                         [(K1, L1, p1, 'SS', expected1),
                          (K1, L1, p2, 'SS', expected2),
                          (K3, L3, p3, 'SS', expected3),
                          (K4, L4, p4, 'TPI', expected4)],
                         ids=['epsilon=1.0,SS', 'epsilon=0.5,SS',
                              'epsilon=1.5,SS', 'epsilon=1.0,TP'])
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
    'tau_b': 0.5,
    'delta_tau_annual': 0.35,
    'epsilon': 0.2
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
# assign values for Y and K variables
Y1 = np.array([2.0, 2.0])
K1 = np.array([1.0, 1.0])
expected1 = np.array([7.925, 7.925])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.5,
    'tau_b': 0.5,
    'delta_tau_annual': 0.35,
    'epsilon': 0.5,
    'delta_annual': 0.5
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.675, 0.675])


@pytest.mark.parametrize('Y,K,p,method,expected',
                         [(Y1, K1, p1, 'SS', expected1),
                          (Y1, K1, p2, 'SS', expected2)],
                         ids=['epsilon=0.2,SS', 'epsilon=0.5,SS'])
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
Y1 = np.array([2.0, 2.0])
L1 = np.array([1.0, 1.0])
expected1 = np.array([16., 16.])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.5,
    'epsilon': 0.5
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([2.0, 2.0])


@pytest.mark.parametrize('Y,L,p,method,expected',
                         [(Y1, L1, p1, 'SS', expected1),
                          (Y1, L1, p2, 'SS', expected2)],
                         ids=['epsilon=0.2,SS', 'epsilon=0.5,SS'])
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
    'delta_tau_annual': 0.35
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.01, 0.04, 0.55])
expected1 = np.array([10.30175902, 6.04917808, 0.51040376])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': 0.35,
    'epsilon': 0.2
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([1.18477314, 1.06556941, 0.62169561])


@pytest.mark.parametrize('r,p,method,expected',
                         [(r1, p1, 'SS', expected1),
                          (r1, p2, 'SS', expected2)],
                         ids=['epsilon=0.8,SS', 'epsilon=0.2,SS'])
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
    'delta_tau_annual': 0.35
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.01, 0.04, 0.55])
expected1 = np.array([1.509317, 1.26576211, 0.43632069])
p2 = Specifications()
new_param_values2 = {
    'Z': [0.5],
    'gamma': 0.4,
    'delta_annual': 0.05,
    'delta_tau_annual': 0.35,
    'epsilon': 0.2
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.87329195, 0.83846042, 0.42967956])


@pytest.mark.parametrize('r,p,method,expected',
                         [(r1, p1, 'SS', expected1),
                          (r1, p2, 'SS', expected2)],
                         ids=['epsilon=0.8,SS', 'epsilon=0.2,SS'])
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
    'tau_b': 0.75,
    'delta_annual': 0.15,
    'delta_tau_annual': 0.03,
    'Z': [2.0],
    'epsilon': 1.0
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
L1 = np.array([2.0, 2.0])
r1 = np.array([1.0, 1.0])
expected1 = np.array([0.09832793, 0.09832793])
p2 = Specifications()
new_param_values2 = {
    'gamma': 0.5,
    'tau_b': 0.75,
    'delta_annual': 0.15,
    'delta_tau_annual': 0.03,
    'Z': [2.0],
    'epsilon': 0.2
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.91363771, 0.91363771])
p3 = Specifications()
new_param_values3 = {
    'gamma': 0.5,
    'epsilon': 0.5,
    'Z': [4.0],
    'tau_b': 0.0,
    'delta_tau_annual': 0.5,
    'delta_annual': 0.05
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([5.80720058, 5.80720058])
p4 = Specifications()
new_param_values4 = {
    'gamma': 0.5,
    'epsilon': 0.5,
    'Z': [4.0],
    'delta_tau_annual': 0.5,
    'delta_annual': 0.05,
    'tau_b': 0.5
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values)
expected4 = np.array([4.32455532, 4.32455532])


@pytest.mark.parametrize('L,r,p,method,expected',
                         [(L1, r1, p1, 'SS', expected1),
                          (L1, r1, p2, 'SS', expected2),
                          (L1, r1, p3, 'SS', expected3),
                          (L1, r1, p4, 'SS', expected4)],
                         ids=['epsilon=1.0,SS', 'epsilon=0.2,SS',
                              'tau_b=0.0,SS', 'tau_b=0.5,SS'])
def test_get_K(L, r, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    K = firm.get_K(L, r, p, method)
    assert (np.allclose(K, expected, atol=1e-6))
