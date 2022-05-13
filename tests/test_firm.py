import pytest
from ogcore import firm
import numpy as np
from ogcore.parameters import Specifications


p1 = Specifications()
new_param_values = {
    'Z': [[2.0]],
    'gamma': [0.5],
    'epsilon': [1.0]
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values)
L1 = np.array([4.0])
K1 = np.array([9.0])
K_g1 = np.array([0.0])
expected1 = np.array([12.0])
p2 = Specifications()
new_param_values2 = {
    'Z': [[2.0]],
    'gamma': [0.5],
    'epsilon': [0.2]
}
p2.update_specifications(new_param_values2)
expected2 = np.array([18.84610765])
p3 = Specifications()
new_param_values3 = {
    'Z': [[2.0]],
    'gamma': [0.5],
    'epsilon': [1.2]
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
L3 = np.array([1 / 12.0])
K3 = np.array([1 / 4.0])
K_g3 = np.array([0.0])
expected3 = np.array([0.592030917])
# update parameters instance with new values for test
p4 = Specifications()
new_param_values4 = {
    'Z': [[2.0]],
    'gamma': [0.5],
    'epsilon': [1.0],
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p4.J)) / (3 * p4.J))
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
L4 = np.array([4.0, 4.0, 4.0]).reshape(3, 1)
K4 = np.array([9.0, 9.0, 9.0]).reshape(3, 1)
K_g4 = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
expected4 = np.array([12.0, 12.0, 12.0]).reshape(3, 1)
p5 = Specifications()
new_param_values5 = {
    'Z': [[1.5], [2.5], [0.6]],
    'gamma': [0.5],
    'epsilon': [1.0],
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p5.J)) / (3 * p5.J))
}
# update parameters instance with new values for test
p5.update_specifications(new_param_values5)
expected5 = np.array([9.0, 15.0, 3.6]).reshape(3, 1)
p6 = Specifications()
new_param_values6 = {
    'Z': [[1.5], [2.5], [0.6]],
    'gamma': [0.5],
    'epsilon': [1.0],
    'T': 3,
    'S': 3,
    'eta': (np.ones((3, p5.J)) / (3 * p5.J)),
    'gamma_g': [0.2],
    'initial_Kg_ratio': 0.01
}
# update parameters instance with new values for test
p6.update_specifications(new_param_values6)
K_g6 = np.array([1.2, 3.0, 0.9]).reshape(3, 1)
expected6 = np.array([7.07402777, 14.16131267, 2.671400509]).reshape(3, 1)

p9 = Specifications()
new_param_values9 = {
    'Z': [[1.5, 1.5, 1.5], [2.5, 2.5, 2.5], [0.6, 0.6, 0.6]],
    'gamma': [0.5, 0.3, 0.4],
    'epsilon': [1.0, 1.0, 1.0],
    'T': 3,
    'S': 3,
    'M': 3,
    'eta': (np.ones((3, p5.J)) / (3 * p5.J)),
    'gamma_g': [0.2, 0.1, 0.25],
    'initial_Kg_ratio': 0.01
}
# update parameters instance with new values for test
p9.update_specifications(new_param_values9)
expected9 = np.array([
        [7.07402777,6.784504444, 6.141925883],
        [14.16131267, 12.39255576, 12.87177155],
        [2.671400509, 2.636842858, 2.286282428]])

p10 = Specifications()
new_param_values9 = {
    'Z': [[1.5, 1.5, 1.5], [2.5, 2.5, 2.5], [0.6, 0.6, 0.6]],
    'gamma': [0.5, 0.3, 0.4],
    'epsilon': [0.3, 0.4, 0.45],
    'T': 3,
    'S': 3,
    'M': 3,
    'eta': (np.ones((3, p5.J)) / (3 * p5.J)),
    'gamma_g': [0.1, 0.2, 0.25],
    'initial_Kg_ratio': 0.01
}
# update parameters instance with new values for test
p10.update_specifications(new_param_values9)
expected10 = np.tile(
    np.array([7.07402777, 14.16131267, 2.671400509]).reshape(3, 1),
    (1, 3))


@pytest.mark.parametrize('K,K_g,L,p,method,expected',
                         [(K1, K_g1, L1, p1, 'SS', expected1),
                          (K1, K_g1, L1, p2, 'SS', expected2),
                          (K3, K_g3, L3, p3, 'SS', expected3),
                          (K4, K_g4, L4, p4, 'TPI', expected4),
                          (K4, K_g4, L4, p5, 'TPI', expected5),
                          (K4, K_g6, L4, p6, 'TPI', expected6),
                        #   (K4, K_g6, L4, p7, 'TPI', expected7),
                        #   (K4, K_g6, L4, p8, 'TPI', expected8),
                          (K4, K_g6, L4, p9, 'TPI', expected9),
                        #   (K4, K_g6, L4, p10, 'TPI', expected10)
                          ],
                         ids=['epsilon=1.0,SS', 'epsilon=0.2,SS',
                              'epsilon=1.2,SS', 'epsilon=1.0,TP',
                              'epsilon=1.0,TP,varyZ',
                              'epsilon=1.0,TP,varyZ,non-zeroKg',
                            #   'M>1, SS, eps=1',
                            #   'M>1, SS, eps<1',
                              'M>1, TPI, eps=1',
                            #   'M>1, TPI, eps<1'
                              ])
def test_get_Y(K, K_g, L, p, method, expected):
    """
        choose values that simplify the calculations and are similar to
        observed values
    """
    Y = firm.get_Y(K, K_g, L, p, None, method)
    assert (np.allclose(Y, expected, atol=1e-6))


# p1 = Specifications()
# new_param_values1 = {
#     'Z': [0.5],
#     'gamma': 0.5,
#     'delta_annual': 0.25,
#     'cit_rate': [0.5],
#     'delta_tau_annual': [0.35],
#     'epsilon': 1.2
# }
# # update parameters instance with new values for test
# p1.update_specifications(new_param_values1)
# # assign values for Y and K variables
# Y1 = np.array([2.0])
# K1 = np.array([1.0])
# expected1 = np.array([0.59492233])
# p2 = Specifications()
# new_param_values2 = {
#     'Z': [[0.5]],
#     'gamma': [0.5],
#     'cit_rate': [0.5],
#     'delta_tau_annual': [0.35],
#     'epsilon': [0.5],
#     'delta_annual': 0.5
# }
# # update parameters instance with new values for test
# p2.update_specifications(new_param_values2)
# expected2 = np.array([1.35975])
# p3 = Specifications()
# new_param_values3 = {
#     'Z': [[0.5]],
#     'gamma': [0.5],
#     'cit_rate': [0.5],
#     'delta_tau_annual': [0.35],
#     'epsilon': [1.0],
#     'delta_annual': 0.5
# }
# # update parameters instance with new values for test
# p3.update_specifications(new_param_values3)
# expected3 = np.array([0.44475])
# p4 = Specifications()
# new_param_values4 = {
#     'Z': [[0.5]],
#     'gamma': [0.5],
#     'cit_rate': [0.5],
#     'delta_tau_annual': [0.35],
#     'epsilon': [1.2],
#     'delta_annual': 0.5,
#     'T': 3,
#     'S': 3,
#     'eta': (np.ones((3, p4.J)) / (3 * p4.J))
# }
# # update parameters instance with new values for test
# p4.update_specifications(new_param_values4)
# Y4 = np.array([3.0, 3.2, 3.8])
# K4 = np.array([1.8, 1.2, 1.0])
# expected4 = np.array([-0.21473161, 0.12101175, 0.47669423])

# p5 = Specifications()
# new_param_values5 = {
#     'Z': [[1.5, 2.5, 0.6]],
#     'gamma': 0.5,
#     'cit_rate': [[0.2, 0.0, 0.5]],
#     'delta_tau_annual': [0.35, 0.2, 0.1],
#     'epsilon': 1.2,
#     'delta_annual': 0.5,
#     'T': 3,
#     'S': 3,
#     'eta': (np.ones((3, p5.J)) / (3 * p5.J))
# }
# # update parameters instance with new values for test
# p5.update_specifications(new_param_values5)
# expected5 = np.array([-0.07814687, 0.48060411, 0.51451412])


# @pytest.mark.parametrize('Y,K,p,method,expected',
#                          [(Y1, K1, p1, 'SS', expected1),
#                           (Y1, K1, p2, 'SS', expected2),
#                           (Y1, K1, p3, 'SS', expected3),
#                           (Y4, K4, p4, 'TPI', expected4),
#                           (Y4, K4, p5, 'TPI', expected5)],
#                          ids=['epsilon=1.2,SS', 'epsilon=0.5,SS',
#                               'epsilon=1.0,SS', 'epsilon=1.2,TP',
#                               'epsilon=1.2,TP,varyParams'])
# def test_get_r(Y, K, p, method, expected):
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     r = firm.get_r(Y, K, p, method)
#     assert (np.allclose(r, expected))


# p1 = Specifications()
# new_param_values1 = {
#     'Z': [0.5],
#     'gamma': 0.5,
#     'epsilon': 0.2
# }
# # update parameters instance with new values for test
# p1.update_specifications(new_param_values1)
# Y1 = np.array([2.0])
# L1 = np.array([1.0])
# expected1 = np.array([16.])
# p2 = Specifications()
# new_param_values2 = {
#     'Z': [0.5],
#     'gamma': 0.5,
#     'epsilon': 1.5
# }
# # update parameters instance with new values for test
# p2.update_specifications(new_param_values2)
# expected2 = np.array([0.793700526])
# p3 = Specifications()
# new_param_values3 = {
#     'Z': [0.5],
#     'gamma': 0.5,
#     'epsilon': 1.0
# }
# # update parameters instance with new values for test
# p3.update_specifications(new_param_values3)
# expected3 = np.array([1.0])
# p4 = Specifications()
# new_param_values4 = {
#     'Z': [0.5, 0.47],
#     'gamma': 0.5,
#     'epsilon': 1.2,
#     'T': 3,
#     'S': 3,
#     'eta': (np.ones((3, p4.J)) / (3 * p4.J))
# }
# # update parameters instance with new values for test
# p4.update_specifications(new_param_values4)
# Y4 = np.array([2.0, 2.0, 2.0])
# L4 = np.array([1.0, 1.0, 1.0])
# expected4 = np.array([0.890898718, 0.881758476, 0.881758476])


# @pytest.mark.parametrize('Y,L,p,method,expected',
#                          [(Y1, L1, p1, 'SS', expected1),
#                           (Y1, L1, p2, 'SS', expected2),
#                           (Y1, L1, p3, 'SS', expected3),
#                           (Y4, L4, p4, 'TPI', expected4)],
#                          ids=['epsilon=0.2,SS', 'epsilon=1.5,SS',
#                               'epsilon=1.0,SS', 'epsilon=1.2,TP'])
# def test_get_w(Y, L, p, method, expected):
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     w = firm.get_w(Y, L, p, method)
#     assert (np.allclose(w, expected, atol=1e-6))


# p1 = Specifications()
# new_param_values1 = {
#     'Z': [0.5],
#     'gamma': 0.4,
#     'epsilon': 0.8,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
# }
# # update parameters instance with new values for test
# p1.update_specifications(new_param_values1)
# r1 = np.array([0.01])
# expected1 = np.array([10.30175902])
# p2 = Specifications()
# new_param_values2 = {
#     'Z': [0.5],
#     'gamma': 0.4,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'epsilon': 1.2,
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
# }
# # update parameters instance with new values for test
# p2.update_specifications(new_param_values2)
# expected2 = np.array([215.1799075])
# p3 = Specifications()
# new_param_values3 = {
#     'Z': [0.5],
#     'gamma': 0.4,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'epsilon': 1.0,
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
# }
# # update parameters instance with new values for test
# p3.update_specifications(new_param_values3)
# expected3 = np.array([10.33169079])
# p4 = Specifications()
# new_param_values4 = {
#     'Z': [0.5, 0.1, 1.1],
#     'gamma': 0.4,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'epsilon': 0.5,
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)],
#     'T': 3,
#     'S': 3,
#     'eta': (np.ones((3, p4.J)) / (3 * p4.J))
# }
# # update parameters instance with new values for test
# p4.update_specifications(new_param_values4)
# r4 = np.array([0.01, 0.04, 0.55])
# expected4 = np.array([0.465031434, -0.045936078, 0.575172024])


# @pytest.mark.parametrize('r,p,method,expected',
#                          [(r1, p1, 'SS', expected1),
#                           (r1, p2, 'SS', expected2),
#                           (r1, p3, 'SS', expected3),
#                           (r4, p4, 'TPI', expected4)],
#                          ids=['epsilon=0.8,SS', 'epsilon=1.2,SS',
#                               'epsilon=1.0,SS', 'epsilon=0.5,TP'])
# def test_get_KLratio_from_r(r, p, method, expected):
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     KLratio = firm.get_KLratio_KLonly(r, p, method)
#     assert (np.allclose(KLratio, expected, atol=1e-6))


# expected4 = np.array([0.465031434, 0.045936078, 0.575172024])


# @pytest.mark.parametrize('r,p,method,expected',
#                          [(r1, p1, 'SS', expected1),
#                           (r1, p2, 'SS', expected2),
#                           (r1, p3, 'SS', expected3),
#                           (r4, p4, 'TPI', expected4)],
#                          ids=['epsilon=0.8,SS', 'epsilon=1.2,SS',
#                               'epsilon=1.0,SS', 'epsilon=0.5,TP'])
# def test_get_KLratio(r, p, method, expected):
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     w = firm.get_w_from_r(r, p, method)
#     KLratio = firm.get_KLratio(r, w, p, method)
#     assert (np.allclose(KLratio, expected, atol=1e-6))


# p1 = Specifications()
# new_param_values1 = {
#     'Z': [0.5],
#     'gamma': 0.4,
#     'epsilon': 0.8,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
# }
# # update parameters instance with new values for test
# p1.update_specifications(new_param_values1)
# r1 = np.array([0.04])
# expected1 = np.array([1.265762107])
# p2 = Specifications()
# new_param_values2 = {
#     'Z': [0.5],
#     'gamma': 0.4,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'epsilon': 1.0,
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
# }
# # update parameters instance with new values for test
# p2.update_specifications(new_param_values2)
# expected2 = np.array([0.550887455])
# p3 = Specifications()
# new_param_values3 = {
#     'Z': [0.5],
#     'gamma': 0.4,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'epsilon': 1.2,
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)]
# }
# # update parameters instance with new values for test
# p3.update_specifications(new_param_values3)
# expected3 = np.array([2.855428923])
# p4 = Specifications()
# new_param_values4 = {
#     'Z': [0.5, 1.0, 4.0],
#     'gamma': 0.4,
#     'delta_annual': 0.05,
#     'delta_tau_annual': [0.35],
#     'epsilon': 1.2,
#     'cit_rate': [(0.0357 / 0.55) * (0.055 / 0.017)],
#     'T': 3,
#     'S': 3,
#     'eta': (np.ones((3, p4.J)) / (3 * p4.J))
# }
# # update parameters instance with new values for test
# p4.update_specifications(new_param_values4)
# r4 = np.array([0.04, 0.04, 0.04])
# expected4 = np.array([0.380178134, 1.19149279, 17.8375083])


# @pytest.mark.parametrize('r,p,method,expected',
#                          [(r1, p1, 'SS', expected1),
#                           (r1, p2, 'SS', expected2),
#                           (r1, p3, 'SS', expected3),
#                           (r4, p4, 'TPI', expected4)],
#                          ids=['epsilon=0.8,SS', 'epsilon=1.0,SS',
#                               'epsilon=1.2,SS', 'epsilon=1.2,TP'])
# def test_get_w_from_r(r, p, method, expected):
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     w = firm.get_w_from_r(r, p, method)
#     assert (np.allclose(w, expected, atol=1e-6))


# p1 = Specifications()
# new_param_values1 = {
#     'gamma': 0.5,
#     'cit_rate': [0.75],
#     'delta_annual': 0.15,
#     'delta_tau_annual': [0.03],
#     'Z': [2.0],
#     'epsilon': 1.2
# }
# # update parameters instance with new values for test
# p1.update_specifications(new_param_values1)
# L1 = np.array([2.0])
# r1 = np.array([1.0])
# expected1 = np.array([5.74454599])
# p2 = Specifications()
# new_param_values2 = {
#     'gamma': 0.5,
#     'cit_rate': [0.75],
#     'delta_annual': 0.15,
#     'delta_tau_annual': [0.03],
#     'Z': [2.0],
#     'epsilon': 1.0
# }
# # update parameters instance with new values for test
# p2.update_specifications(new_param_values2)
# expected2 = np.array([1.1589348])
# p3 = Specifications()
# new_param_values3 = {
#     'gamma': 0.5,
#     'epsilon': 0.4,
#     'Z': [4.0],
#     'cit_rate': [0.0],
#     'delta_tau_annual': [0.5],
#     'delta_annual': 0.05
# }
# # update parameters instance with new values for test
# p3.update_specifications(new_param_values3)
# expected3 = np.array([4.577211711])
# p4 = Specifications()
# new_param_values4 = {
#     'gamma': 0.5,
#     'epsilon': 0.4,
#     'Z': [4.0, 3.0],
#     'delta_tau_annual': [0.5],
#     'delta_annual': 0.05,
#     'cit_rate': [0.5],
#     'T': 3,
#     'S': 3,
#     'eta': (np.ones((3, p4.J)) / (3 * p4.J))
# }
# # update parameters instance with new values for test
# p4.update_specifications(new_param_values4)
# L4 = np.array([2.0, 2.0, 2.0])
# r4 = np.array([1.0, 1.0, 1.0])
# expected4 = np.array([3.39707089, 2.85348453, 2.85348453])


# @pytest.mark.parametrize('L,r,p,method,expected',
#                          [(L1, r1, p1, 'SS', expected1),
#                           (L1, r1, p2, 'SS', expected2),
#                           (L1, r1, p3, 'SS', expected3),
#                           (L4, r4, p4, 'TPI', expected4)],
#                          ids=['epsilon=1.2,SS', 'epsilon=1.0,SS',
#                               'epsilon=0.4,SS', 'epsilon=0.4,TP'])
# def test_get_K_KLonly(L, r, p, method, expected):
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     K = firm.get_K_KLonly(L, r, p, method)
#     assert (np.allclose(K, expected, atol=1e-6))


# @pytest.mark.parametrize('L,r,p,method,expected',
#                          [(L1, r1, p1, 'SS', expected1),
#                           (L1, r1, p2, 'SS', expected2),
#                           (L1, r1, p3, 'SS', expected3),
#                           (L4, r4, p4, 'TPI', expected4)],
#                          ids=['epsilon=1.2,SS', 'epsilon=1.0,SS',
#                               'epsilon=0.4,SS', 'epsilon=0.4,TP'])
# def test_get_K(L, r, p, method, expected):
#     """
#         choose values that simplify the calculations and are similar to
#         observed values
#     """
#     w = firm.get_w_from_r(r, p, method)
#     K = firm.get_K(r, w, L, p, method)
#     assert (np.allclose(K, expected, atol=1e-6))


# Y1 = 2.0
# x1 = 1.0
# expected1 = 1.0
# Y2 = np.array([2.0, 1.0])
# x2 = np.array([1.0, 0.5])
# expected2 = np.array([0.6, 0.6])


# @pytest.mark.parametrize('Y,x,share,method,expected',
#                          [(Y1, x1, 0.5, 'SS', expected1),
#                           (Y2, x2, 0.3, 'TPI', expected2),
#                           (Y2, np.zeros_like(Y2), 0.5, 'TPI', np.zeros_like(Y2)),
#                           (Y1, np.zeros_like(Y1), 0.5, 'SS', np.zeros_like(Y1))],
#                          ids=['SS', 'TPI', 'x=0, TPI', 'x=0,SS'])
# def test_get_MPx(Y, x, share, method, expected):
#     """
#     Test of the marginal product function
#     """
#     p = Specifications()
#     p.Z = np.ones(2)
#     mpx = firm.get_MPx(Y, x, share, p, method)

#     assert (np.allclose(mpx, expected, atol=1e-6))
