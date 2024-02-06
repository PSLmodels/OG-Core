from math import exp
import pytest
from ogcore import firm
import numpy as np
from ogcore.parameters import Specifications


p1 = Specifications()
new_param_values = {"Z": [[2.0]], "gamma": [0.5], "epsilon": [1.0]}
# update parameters instance with new values for test
p1.update_specifications(new_param_values)
L1 = np.array([4.0])
K1 = np.array([9.0])
K_g1 = np.array([0.0])
expected1 = np.array([12.0])
p2 = Specifications()
new_param_values2 = {"Z": [[2.0]], "gamma": [0.5], "epsilon": [0.2]}
p2.update_specifications(new_param_values2)
expected2 = np.array([18.84610765])
p3 = Specifications()
new_param_values3 = {"Z": [[2.0]], "gamma": [0.5], "epsilon": [1.2]}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
L3 = np.array([1 / 12.0])
K3 = np.array([1 / 4.0])
K_g3 = np.array([0.0])
expected3 = np.array([0.592030917])
# update parameters instance with new values for test
p4 = Specifications()
rho_vec = np.zeros((1, 3))
rho_vec[0, -1] = 1.0
new_param_values4 = {
    "Z": [[2.0]],
    "gamma": [0.5],
    "epsilon": [1.0],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
L4 = np.array([4.0, 4.0, 4.0]).reshape(3, 1)
K4 = np.array([9.0, 9.0, 9.0]).reshape(3, 1)
K_g4 = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
expected4 = np.array([12.0, 12.0, 12.0]).reshape(3, 1)
p5 = Specifications()
new_param_values5 = {
    "Z": [[1.5], [2.5], [0.6]],
    "gamma": [0.5],
    "epsilon": [1.0],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p5.J)) / (3 * p5.J)),
    "e": np.ones((3, p5.J)),
}
# update parameters instance with new values for test
p5.update_specifications(new_param_values5)
expected5 = np.array([9.0, 15.0, 3.6]).reshape(3, 1)
p6 = Specifications()
new_param_values6 = {
    "Z": [[1.5], [2.5], [0.6]],
    "gamma": [0.5],
    "epsilon": [1.0],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p5.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p5.J)) / (3 * p5.J)),
    "gamma_g": [0.2],
    "initial_Kg_ratio": 0.01,
}
# update parameters instance with new values for test
p6.update_specifications(new_param_values6)
K_g6 = np.array([1.2, 3.0, 0.9]).reshape(3, 1)
expected6 = np.array([7.07402777, 14.16131267, 2.671400509]).reshape(3, 1)

# update parameters instance with new values for test
p7 = Specifications()
new_param_values7 = {
    "Z": [[2.0, 1.5]],
    "gamma": [0.5, 0.3],
    "gamma_g": [0.2, 0.1],
    "epsilon": [1.0, 1.0],
    "T": 3,
    "S": 3,
    "M": 2,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
    "initial_Kg_ratio": 0.01,
}
# update parameters instance with new values for test
p7.update_specifications(new_param_values7)
L7 = np.array([4.0, 4.0])
K7 = np.array([9.0, 9.0])
K_g7 = np.array([0.3])
expected7 = np.array([7.148147389, 5.906254166])

# update parameters instance with new values for test
p8 = Specifications()
new_param_values8 = {
    "Z": [[2.0, 1.5]],
    "gamma": [0.5, 0.3],
    "gamma_g": [0.2, 0.1],
    "epsilon": [0.6, 0.7],
    "T": 3,
    "S": 3,
    "M": 2,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
    "initial_Kg_ratio": 0.01,
}
# update parameters instance with new values for test
p8.update_specifications(new_param_values8)
expected8 = np.array([13.58741333, 12.8445788])


p9 = Specifications()
new_param_values9 = {
    "Z": [[1.5, 1.5, 1.5], [2.5, 2.5, 2.5], [0.6, 0.6, 0.6]],
    "gamma": [0.5, 0.3, 0.4],
    "epsilon": [1.0, 1.0, 1.0],
    "T": 3,
    "S": 3,
    "M": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p5.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p5.J)) / (3 * p5.J)),
    "gamma_g": [0.2, 0.1, 0.25],
    "initial_Kg_ratio": 0.01,
}
# update parameters instance with new values for test
p9.update_specifications(new_param_values9)
L9 = np.tile(np.array([4.0, 4.0, 4.0]).reshape(3, 1), (1, 3))
K9 = np.tile(np.array([9.0, 9.0, 9.0]).reshape(3, 1), (1, 3))
expected9 = np.array(
    [
        [7.07402777, 6.784504444, 6.141925883],
        [14.16131267, 12.39255576, 12.87177155],
        [2.671400509, 2.636842858, 2.286282428],
    ]
)

p10 = Specifications()
new_param_values10 = {
    "Z": [[1.5, 1.5, 1.5], [2.5, 2.5, 2.5], [0.6, 0.6, 0.6]],
    "gamma": [0.5, 0.3, 0.4],
    "epsilon": [0.3, 0.4, 0.45],
    "T": 3,
    "S": 3,
    "M": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p5.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p5.J)) / (3 * p5.J)),
    "gamma_g": [0.2, 0.1, 0.25],
    "initial_Kg_ratio": 0.01,
}
# update parameters instance with new values for test
p10.update_specifications(new_param_values10)
expected10 = np.array(
    [
        [15.41106022, 38.83464768, 4.946631616],
        [13.02348889, 22.39766006, 5.097163565],
        [14.31423941, 35.75229301, 4.789115236],
    ]
).T


# TODO: finish the below, then need to add tests of m not None
# for both SS and TPI...
@pytest.mark.parametrize(
    "K,K_g,L,p,method, m, expected",
    [
        (K1, K_g1, L1, p1, "SS", None, expected1),
        (K1, K_g1, L1, p2, "SS", None, expected2),
        (K3, K_g3, L3, p3, "SS", None, expected3),
        (K4, K_g4, L4, p4, "TPI", None, expected4),
        (K4, K_g4, L4, p5, "TPI", None, expected5),
        (K4, K_g6, L4, p6, "TPI", None, expected6),
        (K7, K_g7, L7, p7, "SS", None, expected7),
        (K7, K_g7, L7, p8, "SS", None, expected8),
        (K9, K_g6, L9, p9, "TPI", None, expected9),
        (K9, K_g6, L9, p10, "TPI", None, expected10),
        (K7[0], K_g7, L7[0], p7, "SS", 0, expected7[0]),
        (K7[0], K_g7, L7[0], p8, "SS", 0, expected8[0]),
        (K9[:, 0], np.squeeze(K_g6), L9[:, 0], p9, "TPI", 0, expected9[:, 0]),
        (
            K9[:, 0],
            np.squeeze(K_g6),
            L9[:, 0],
            p10,
            "TPI",
            0,
            expected10[:, 0],
        ),
    ],
    ids=[
        "epsilon=1.0,SS",
        "epsilon=0.2,SS",
        "epsilon=1.2,SS",
        "epsilon=1.0,TP",
        "epsilon=1.0,TP,varyZ",
        "epsilon=1.0,TP,varyZ,non-zeroKg",
        "M>1, SS, eps=1",
        "M>1, SS, eps<1",
        "M>1, TPI, eps=1",
        "M>1, TPI, eps<1",
        "M>1, SS, eps=1, m not None",
        "M>1, SS, eps<1, m not None",
        "M>1, TPI, eps=1, m not None",
        "M>1, TPI, eps<1, m not None",
    ],
)
def test_get_Y(K, K_g, L, p, method, m, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    Y = firm.get_Y(K, K_g, L, p, method, m)

    assert np.allclose(Y, expected, atol=1e-6)


p1 = Specifications()
new_param_values1 = {
    "Z": [[0.5]],
    "gamma": [0.5],
    "delta_annual": 0.25,
    "cit_rate": [[0.5]],
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.2],
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
# assign values for Y and K variables
Y1 = np.array([2.0])
K1 = np.array([1.0])
expected1 = np.array([0.59492233])
p2 = Specifications()
new_param_values2 = {
    "Z": [[0.5]],
    "gamma": [0.5],
    "cit_rate": [[0.5]],
    "delta_tau_annual": [[0.35]],
    "epsilon": [0.5],
    "delta_annual": 0.5,
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([1.35975])
p3 = Specifications()
new_param_values3 = {
    "Z": [[0.5]],
    "gamma": [0.5],
    "cit_rate": [[0.5]],
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.0],
    "delta_annual": 0.5,
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([0.44475])
p4 = Specifications()
new_param_values4 = {
    "Z": [[0.5]],
    "gamma": [0.5],
    "cit_rate": [[0.5]],
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.2],
    "delta_annual": 0.5,
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
Y4 = np.array([3.0, 3.2, 3.8])
K4 = np.array([1.8, 1.2, 1.0])
expected4 = np.array([-0.21473161, 0.12101175, 0.47669423])

p5 = Specifications()
new_param_values5 = {
    "Z": [[1.5], [2.5], [0.6]],
    "gamma": [0.5],
    "cit_rate": [[0.2], [0.0], [0.5]],
    "delta_tau_annual": [[0.35], [0.2], [0.1]],
    "epsilon": [1.2],
    "delta_annual": 0.5,
    "inv_tax_credit": [[0.07]],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p5.J)) / (3 * p5.J)),
}
# update parameters instance with new values for test
p5.update_specifications(new_param_values5)
expected5 = np.array([-0.07814687, 0.48060411, 0.51451412]) + 0.07
p_m = np.ones((p5.T, p5.M))


@pytest.mark.parametrize(
    "Y,K,p_m,p,method,expected",
    [
        (Y1, K1, p_m[-1, :], p1, "SS", expected1),
        (Y1, K1, p_m[-1, :], p2, "SS", expected2),
        (Y1, K1, p_m[-1, :], p3, "SS", expected3),
        (
            Y4.reshape(3, 1),
            K4.reshape(3, 1),
            p_m,
            p4,
            "TPI",
            expected4.reshape(3, 1),
        ),
        (
            Y4.reshape(3, 1),
            K4.reshape(3, 1),
            p_m,
            p5,
            "TPI",
            expected5.reshape(3, 1),
        ),
    ],
    ids=[
        "epsilon=1.2,SS",
        "epsilon=0.5,SS",
        "epsilon=1.0,SS",
        "epsilon=1.2,TP",
        "epsilon=1.2,TP,varyParams",
    ],
)
def test_get_r(Y, K, p_m, p, method, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    r = firm.get_r(Y, K, p_m, p, method)
    print("R shapes = ", r.shape, expected.shape)
    assert np.allclose(r, expected)


p1 = Specifications()
new_param_values1 = {"Z": [[0.5]], "gamma": [0.5], "epsilon": [0.2]}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
Y1 = np.array([2.0])
L1 = np.array([1.0])
expected1 = np.array([16.0])
p2 = Specifications()
new_param_values2 = {"Z": [[0.5]], "gamma": [0.5], "epsilon": [1.5]}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.793700526])
p3 = Specifications()
new_param_values3 = {"Z": [[0.5]], "gamma": [0.5], "epsilon": [1.0]}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([1.0])
p4 = Specifications()
new_param_values4 = {
    "Z": [[0.5], [0.47]],
    "gamma": [0.5],
    "epsilon": [1.2],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
Y4 = np.array([2.0, 2.0, 2.0])
L4 = np.array([1.0, 1.0, 1.0])
expected4 = np.array([0.890898718, 0.881758476, 0.881758476])
p_m = np.ones((p4.T, p4.M))


@pytest.mark.parametrize(
    "Y,L,p_m,p,method,expected",
    [
        (Y1, L1, p_m[-1, :], p1, "SS", expected1),
        (Y1, L1, p_m[-1, :], p2, "SS", expected2),
        (Y1, L1, p_m[-1, :], p3, "SS", expected3),
        (
            Y4.reshape(3, 1),
            L4.reshape(3, 1),
            p_m,
            p4,
            "TPI",
            expected4.reshape(3, 1),
        ),
    ],
    ids=[
        "epsilon=0.2,SS",
        "epsilon=1.5,SS",
        "epsilon=1.0,SS",
        "epsilon=1.2,TP",
    ],
)
def test_get_w(Y, L, p_m, p, method, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    w = firm.get_w(Y, L, p_m, p, method)
    assert np.allclose(w, expected, atol=1e-6)


p1 = Specifications()
new_param_values1 = {
    "Z": [[0.5]],
    "gamma": [0.4],
    "epsilon": [0.8],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.01])
expected1 = np.array([10.30175902])
p2 = Specifications()
new_param_values2 = {
    "Z": [[0.5]],
    "gamma": [0.4],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.2],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([215.1799075])
p3 = Specifications()
new_param_values3 = {
    "Z": [[0.5]],
    "gamma": [0.4],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.0],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([10.33169079])
p4 = Specifications()
new_param_values4 = {
    "Z": [[0.5], [0.1], [1.1]],
    "gamma": [0.4],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "epsilon": [0.5],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
r4 = np.array([0.01, 0.04, 0.55])
expected4 = np.array([0.465031434, -0.045936078, 0.575172024])


@pytest.mark.parametrize(
    "r,p,method,expected",
    [
        (r1, p1, "SS", expected1),
        (r1, p2, "SS", expected2),
        (r1, p3, "SS", expected3),
        (r4, p4, "TPI", expected4),
    ],
    ids=[
        "epsilon=0.8,SS",
        "epsilon=1.2,SS",
        "epsilon=1.0,SS",
        "epsilon=0.5,TP",
    ],
)
def test_get_KLratio_from_r(r, p, method, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    KLratio = firm.get_KLratio_KLonly(r, p, method)
    assert np.allclose(KLratio, expected, atol=1e-6)


expected4 = np.array([0.465031434, 0.045936078, 0.575172024])


@pytest.mark.parametrize(
    "r,p,method,expected",
    [
        (r1, p1, "SS", expected1),
        (r1, p2, "SS", expected2),
        (r1, p3, "SS", expected3),
        (r4, p4, "TPI", expected4),
    ],
    ids=[
        "epsilon=0.8,SS",
        "epsilon=1.2,SS",
        "epsilon=1.0,SS",
        "epsilon=0.5,TP",
    ],
)
def test_get_KLratio(r, p, method, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    w = firm.get_w_from_r(r, p, method)
    KLratio = firm.get_KLratio(r, w, p, method)
    assert np.allclose(KLratio, expected, atol=1e-6)


p1 = Specifications()
new_param_values1 = {
    "Z": [[0.5]],
    "gamma": [0.4],
    "epsilon": [0.8],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
r1 = np.array([0.04])
expected1 = np.array([1.265762107])
p2 = Specifications()
new_param_values2 = {
    "Z": [[0.5]],
    "gamma": [0.4],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.0],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([0.550887455])
p3 = Specifications()
new_param_values3 = {
    "Z": [[0.5]],
    "gamma": [0.4],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.2],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([2.855428923])
p4 = Specifications()
new_param_values4 = {
    "Z": [[0.5], [1.0], [4.0]],
    "gamma": [0.4],
    "delta_annual": 0.05,
    "delta_tau_annual": [[0.35]],
    "epsilon": [1.2],
    "cit_rate": [[(0.0357 / 0.55) * (0.055 / 0.017)]],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
r4 = np.array([0.04, 0.04, 0.04])
expected4 = np.array([0.380178134, 1.19149279, 17.8375083])


@pytest.mark.parametrize(
    "r,p,method,expected",
    [
        (r1, p1, "SS", expected1),
        (r1, p2, "SS", expected2),
        (r1, p3, "SS", expected3),
        (r4, p4, "TPI", expected4),
    ],
    ids=[
        "epsilon=0.8,SS",
        "epsilon=1.0,SS",
        "epsilon=1.2,SS",
        "epsilon=1.2,TP",
    ],
)
def test_get_w_from_r(r, p, method, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    w = firm.get_w_from_r(r, p, method)
    assert np.allclose(w, expected, atol=1e-6)


p1 = Specifications()
new_param_values1 = {
    "gamma": [0.5],
    "cit_rate": [[0.75]],
    "delta_annual": 0.15,
    "delta_tau_annual": [[0.03]],
    "Z": [[2.0]],
    "epsilon": [1.2],
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
L1 = np.array([2.0])
r1 = np.array([1.0])
expected1 = np.array([5.74454599])
p2 = Specifications()
new_param_values2 = {
    "gamma": [0.5],
    "cit_rate": [[0.75]],
    "delta_annual": 0.15,
    "delta_tau_annual": [[0.03]],
    "Z": [[2.0]],
    "epsilon": [1.0],
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected2 = np.array([1.1589348])
p3 = Specifications()
new_param_values3 = {
    "gamma": [0.5],
    "epsilon": [0.4],
    "Z": [[4.0]],
    "cit_rate": [[0.0]],
    "delta_tau_annual": [[0.5]],
    "delta_annual": 0.05,
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)
expected3 = np.array([4.577211711])
p4 = Specifications()
new_param_values4 = {
    "gamma": [0.5],
    "epsilon": [0.4],
    "Z": [[4.0], [3.0]],
    "delta_tau_annual": [[0.5]],
    "delta_annual": 0.05,
    "cit_rate": [[0.5]],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p4.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p4.J)) / (3 * p4.J)),
}
# update parameters instance with new values for test
p4.update_specifications(new_param_values4)
L4 = np.array([2.0, 2.0, 2.0])
r4 = np.array([1.0, 1.0, 1.0])
expected4 = np.array([3.39707089, 2.85348453, 2.85348453])


@pytest.mark.parametrize(
    "L,r,p,method,expected",
    [
        (L1, r1, p1, "SS", expected1),
        (L1, r1, p2, "SS", expected2),
        (L1, r1, p3, "SS", expected3),
        (L4, r4, p4, "TPI", expected4),
    ],
    ids=[
        "epsilon=1.2,SS",
        "epsilon=1.0,SS",
        "epsilon=0.4,SS",
        "epsilon=0.4,TP",
    ],
)
def test_get_K_KLonly(L, r, p, method, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    K = firm.get_K_KLonly(L, r, p, method)
    assert np.allclose(K, expected, atol=1e-6)


@pytest.mark.parametrize(
    "L,r,p,method,expected",
    [
        (L1, r1, p1, "SS", expected1),
        (L1, r1, p2, "SS", expected2),
        (L1, r1, p3, "SS", expected3),
        (L4, r4, p4, "TPI", expected4),
    ],
    ids=[
        "epsilon=1.2,SS",
        "epsilon=1.0,SS",
        "epsilon=0.4,SS",
        "epsilon=0.4,TP",
    ],
)
def test_get_K(L, r, p, method, expected):
    """
    choose values that simplify the calculations and are similar to
    observed values
    """
    w = firm.get_w_from_r(r, p, method)
    K = firm.get_K(r, w, L, p, method)
    assert np.allclose(K, expected, atol=1e-6)


Y1 = np.array([18.84610765])
Y2 = np.array([12.0])
Y3 = np.array([18.84610765, 18.84610765, 18.84610765])
Y4 = np.array([12.0, 12.0, 12.0])
x1 = np.array([9.0])
x2 = np.array([9.0, 9.0, 9.0])
p1 = Specifications()
new_param_values1 = {
    "gamma": [0.5],
    "epsilon": [0.2],
    "Z": [[2.0]],
    "T": 3,
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
p2 = Specifications()
new_param_values2 = {
    "gamma": [0.5],
    "epsilon": [1.0],
    "Z": [[2.0]],
    "T": 3,
    "e": p2.e[0, :, :],
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
expected1 = np.array([0.078636799])
expected2 = np.array([0.666666667])
expected3 = np.array([0.078636799, 0.078636799, 0.078636799])
expected4 = np.array([0.666666667, 0.666666667, 0.666666667])


@pytest.mark.parametrize(
    "Y,x,share,p,method,expected",
    [
        (Y1, x1, 1 - p1.gamma[-1] - p1.gamma_g[-1], p1, "SS", expected1),
        (Y3, x2, 1 - p1.gamma[-1] - p1.gamma_g[-1], p1, "TPI", expected3),
        (Y2, x1, 1 - p2.gamma[-1] - p2.gamma_g[-1], p2, "SS", expected2),
        (Y4, x2, 1 - p1.gamma[-1] - p2.gamma_g[-1], p2, "TPI", expected4),
        (Y2, np.zeros_like(Y2), 0.5, p1, "SS", np.zeros_like(Y2)),
        (Y3, np.zeros_like(Y3), 0.5, p1, "TPI", np.zeros_like(Y3)),
    ],
    ids=["SS", "TPI", "SS, eps=1", "TPI, eps=1", "x=0, SS", "x=0,TPI"],
)
def test_get_MPx(Y, x, share, p, method, expected):
    """
    Test of the marginal product function
    """
    mpx = firm.get_MPx(Y, x, share, p, method)

    assert np.allclose(mpx, expected, atol=1e-6)


r1 = 0.05
r2 = np.array([0.05, 0.05, 0.05])
pm1 = np.array([1.2])
pm2 = np.array([[1.2], [1.2], [1.2]])
p1 = Specifications()
new_param_values1 = {
    "gamma": [0.5],
    "epsilon": [0.2],
    "Z": [[2.0]],
    "delta_tau_annual": [[0.35]],
    "delta_annual": 0.5,
    "cit_rate": [[0.5]],
    "adjustment_factor_for_cit_receipts": [1.0],
    "c_corp_share_of_assets": 1.0,
    "T": 3,
    "e": p1.e[0, :, :],
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
p2 = Specifications()
new_param_values2 = {
    "gamma": [0.5],
    "epsilon": [1.0],
    "Z": [[2.0]],
    "delta_tau_annual": [[0.35]],
    "delta_annual": 0.25,
    "cit_rate": [[0.5]],
    "adjustment_factor_for_cit_receipts": [1.0],
    "c_corp_share_of_assets": 1.0,
    "T": 3,
    "e": p2.e[0, :, :],
}
# update parameters instance with new values for test
p2.update_specifications(new_param_values2)
p3 = Specifications()
new_param_values3 = {
    "gamma": [0.5, 0.5],
    "epsilon": [1.0, 1.0],
    "Z": [[2.0]],
    "delta_tau_annual": [[0.35]],
    "delta_annual": 0.25,
    "inv_tax_credit": [[0.03]],
    "cit_rate": [[0.5]],
    "adjustment_factor_for_cit_receipts": [1.0],
    "c_corp_share_of_assets": 1.0,
    "T": 3,
    "M": 2,
    "e": p3.e[0, :, :],
}
# update parameters instance with new values for test
p3.update_specifications(new_param_values3)

coc_expected1 = np.array([0.75])
coc_expected2 = np.array([0.75, 0.75, 0.75])
coc_expected3 = np.array([0.25, 0.25]) - (0.03 * p3.delta / (1 - 0.5))
coc_expected4 = np.array([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]) - (
    0.03 * p3.delta / (1 - 0.5)
)
ky_expected1 = np.array([0.315478672])
ky_expected2 = np.array([2.4, 2.4, 2.4])
ky_expected3 = np.array([2.4])


@pytest.mark.parametrize(
    "r,p,method,m,expected",
    [
        (r1, p1, "SS", -1, coc_expected1),
        (r2, p1, "TPI", -1, coc_expected2),
        (r1, p3, "SS", None, coc_expected3),
        (r2, p3, "TPI", None, coc_expected4),
    ],
    ids=["SS", "TPI", "SS, m=None", "TPI, m=None"],
)
def test_get_cost_of_capital(r, p, method, m, expected):
    """
    Test of the cost of capital function
    """
    coc = firm.get_cost_of_capital(r, p, method, m)
    assert np.allclose(coc, expected, atol=1e-6)


@pytest.mark.parametrize(
    "r,p_m,p,method,m,expected",
    [
        (r1, pm1, p1, "SS", -1, ky_expected1),
        (r2, pm2, p2, "TPI", -1, ky_expected2),
        (r1, pm1, p2, "SS", -1, ky_expected3),
    ],
    ids=["SS", "TPI", "SS, epsilon=1.0"],
)
def test_get_KY_ratio(r, p_m, p, method, m, expected):
    """
    Test of the ratio of KY function
    """
    KY_ratio = firm.get_KY_ratio(r, p_m, p, method, m)
    assert np.allclose(KY_ratio, expected, atol=1e-6)


w1 = 1.3
w2 = np.array([1.3, 1.3, 1.3])
Y1 = np.array([18.84610765])
Y2 = np.array([12])
Y3 = np.array([18.84610765, 18.84610765, 18.84610765])
Y4 = np.array([12, 12, 12])
L1 = np.array([9.0])
L2 = np.array([9.0, 9.0, 9.0])
pm_expected1 = np.array([16.53170028])
pm_expected2 = np.array([16.53170028, 16.53170028, 16.53170028])
pm_expected3 = np.array([1.95])
pm_expected4 = np.array([1.95, 1.95, 1.95])


@pytest.mark.parametrize(
    "w,Y,L,p,method,expected",
    [
        (w1, Y1, L1, p1, "SS", pm_expected1),
        (w2, Y3, L2, p1, "TPI", pm_expected2),
        (w1, Y2, L1, p2, "SS", pm_expected3),
        (w2, Y4, L2, p2, "TPI", pm_expected4),
    ],
    ids=["SS", "TPI", "SS, epsilon=1.0", "TPI, epsilon=1.0"],
)
def test_get_pm(w, Y, L, p, method, expected):
    """
    Test of the function that computes goods prices
    """
    pm = firm.get_pm(w, Y, L, p, method)
    assert np.allclose(pm, expected, atol=1e-6)


Y1 = np.array([18.84610765])
Y2 = np.array([12])
K1 = np.array([4])
Kg = 0
Kg2 = np.zeros(3)
Y3 = np.array([18.84610765, 18.84610765, 18.84610765])
Y4 = np.array([12, 12, 12])
K2 = np.array([4, 4, 4])
L_expected1 = 9.0
L_expected2 = np.array([9.0, 9.0, 9.0])
Y5 = np.array([7.07402777, 14.16131267, 2.671400509])
K5 = np.array([9.0, 9.0, 9.0])
Kg5 = np.array([1.2, 3, 0.9])
L_expected5 = np.array([4.0, 4.0, 4.0])
p5 = Specifications()
new_param_values5 = {
    "gamma": [0.5],
    "gamma_g": [0.2],
    "epsilon": [1.0],
    "Z": [[1.5], [2.5], [0.6]],
    "delta_tau_annual": [[0.35]],
    "delta_annual": 0.05,
    "cit_rate": [[0.3]],
    "adjustment_factor_for_cit_receipts": [1.0],
    "c_corp_share_of_assets": 1.0,
    "initial_Kg_ratio": 0.01,
    "T": 3,
    "e": p5.e[0, :, :],
}
# update parameters instance with new values for test
p5.update_specifications(new_param_values5)

p6 = Specifications()
new_param_values6 = {
    "gamma": [0.4],
    "gamma_g": [0.25],
    "epsilon": [0.3],
    "Z": [[0.6]],
    "delta_tau_annual": [[0.35]],
    "delta_annual": 0.05,
    "cit_rate": [[0.3]],
    "adjustment_factor_for_cit_receipts": [1.0],
    "c_corp_share_of_assets": 1.0,
    "initial_Kg_ratio": 0.01,
    "T": 3,
    "e": p6.e[0, :, :],
}
# update parameters instance with new values for test
p6.update_specifications(new_param_values6)
Y6 = np.ones(3) * 3.731865484
K6 = np.ones(3) * 9.0
Kg6 = np.ones(3) * 0.9


@pytest.mark.parametrize(
    "Y,K,Kg,p,method,expected",
    [
        (Y1, K1, Kg, p1, "SS", L_expected1),
        (Y3, K2, Kg2, p1, "TPI", L_expected2),
        (Y2, K1, Kg, p2, "SS", L_expected1),
        (Y4, K2, Kg2, p2, "TPI", L_expected2),
        (Y5, K5, Kg5, p5, "TPI", L_expected5),
        (Y6, K6, Kg6, p6, "TPI", L_expected5),
    ],
    ids=[
        "SS",
        "TPI",
        "SS, epsilon=1.0",
        "TPI, epsilon=1.0",
        "TPI, eps=1, Kg>0",
        "TPI, eps!=1, Kg>0",
    ],
)
def test_solve_L(Y, K, Kg, p, method, expected):
    """
    Test of the function that solves for labor supply
    """
    L = firm.solve_L(Y, K, Kg, p, method)
    assert np.allclose(L, expected, atol=1e-6)


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


@pytest.mark.parametrize(
    "K,Kp1,p,method,expected",
    [
        (K_1, Kp1_1, p1, "SS", expected_Psi_1),
        (K_2, Kp1_2, p2, "SS", expected_Psi_2),
        (K_3, Kp1_3, p3, "TPI", expected_Psi_3),
    ],
    ids=["Zero cost", "Non-zero cost", "TPI"],
)
def test_adj_cost(K, Kp1, p, method, expected):
    """
    Test of the firm capital adjustment cost function.
    """
    test_val = firm.adj_cost(K, Kp1, p, method)
    assert np.allclose(test_val, expected)
