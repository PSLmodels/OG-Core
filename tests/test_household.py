import pytest
import numpy as np
import copy
import os
from ogcore import household, utils
from ogcore.parameters import Specifications

CUR_PATH = os.path.abspath(os.path.dirname(__file__))

test_data = [
    (0.1, 1, 10),
    (0.2, 2.5, 55.90169944),
    (
        np.array([0.5, 6.2, 1.5]),
        3.2,
        np.array([9.18958684, 0.002913041, 0.273217159]),
    ),
]


@pytest.mark.parametrize(
    "c,sigma,expected", test_data, ids=["Scalar 0", "Scalar 1", "Vector"]
)
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

test_data = [
    (0.87, p1, 2.825570309),
    (0.0, p1, 0.0009117852028298067),
    (0.99999, p1, 69.52423604),
    (0.00001, p1, 0.005692782),
    (0.8, p2, 1.471592068),
    (0.8, p3, 0.795937549),
    (0.8, p4, 11.66354267),
    (
        np.array([[0.8, 0.9, 0.3], [0.5, 0.2, 0.99]]),
        p1,
        np.array(
            [
                [2.364110379, 3.126796062, 1.014935377],
                [1.4248841, 0.806333875, 6.987729463],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "n,params,expected",
    test_data,
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
)
def test_marg_ut_labor(n, params, expected):
    # Test marginal utility of labor calculation
    test_value = household.marg_ut_labor(n, params.chi_n, params)

    assert np.allclose(test_value, expected)


p1 = Specifications()
p1.zeta = np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]])
p1.S = 3
p1.J = 2
p1.T = 3
p1.lambdas = np.array([0.6, 0.4])
p1.omega_SS = np.array([0.25, 0.25, 0.5])
p1.omega = np.tile(p1.omega_SS.reshape((1, p1.S)), (p1.T, 1))
BQ1 = 2.5
p1.use_zeta = True
expected1 = np.array([[1.66666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]])
p2 = Specifications()
p2.zeta = np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]])
p2.S = 3
p2.rho = np.array([[0.0, 0.0, 1.0]])
p2.J = 2
p2.T = 3
p2.lambdas = np.array([0.6, 0.4])
p2.omega_SS = np.array([0.25, 0.25, 0.5])
p2.omega = np.tile(p2.omega_SS.reshape((1, p2.S)), (p2.T, 1))
p2.use_zeta = True
BQ2 = np.array([2.5, 0.8, 3.6])
expected2 = np.array([7.5, 10.0, 0.0])
expected3 = np.array(
    [
        [[1.666666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]],
        [[0.533333333, 2.4], [0.8, 3.2], [0.133333333, 0.0]],
        [[2.4, 10.8], [3.6, 14.4], [0.6, 0.0]],
    ]
)
expected4 = np.array([[7.5, 10.0, 0.0], [2.4, 3.2, 0.0], [10.8, 14.4, 0.0]])
p3 = Specifications()
p3.S = 3
p3.rho = np.array([[0.0, 0.0, 1.0]])
p3.J = 2
p3.T = 3
p3.lambdas = np.array([0.6, 0.4])
p3.omega_SS = np.array([0.25, 0.25, 0.5])
p3.omega = np.tile(p2.omega_SS.reshape((1, p2.S)), (p2.T, 1))
p3.use_zeta = False
BQ3 = np.array([1.1, 0.8])
BQ4 = np.array([[1.1, 0.8], [3.2, 4.6], [2.5, 0.1]])
expected5 = np.array(
    [[1.833333333, 2.0], [1.833333333, 2.0], [1.833333333, 2.0]]
)
expected6 = np.array([2.0, 2.0, 2.0])
expected7 = np.array(
    [
        [[1.833333333, 2.0], [1.833333333, 2.0], [1.833333333, 2.0]],
        [[5.333333333, 11.5], [5.333333333, 11.5], [5.333333333, 11.5]],
        [[4.166666667, 0.25], [4.166666667, 0.25], [4.166666667, 0.25]],
    ]
)
expected8 = np.array([[2.0, 2.0, 2.0], [11.5, 11.5, 11.5], [0.25, 0.25, 0.25]])
test_data = [
    (BQ1, None, p1, "SS", expected1),
    (BQ1, 1, p1, "SS", expected2),
    (BQ2, None, p2, "TPI", expected3),
    (BQ2, 1, p2, "TPI", expected4),
    (BQ3, None, p3, "SS", expected5),
    (BQ3, 1, p3, "SS", expected6),
    (BQ4, None, p3, "TPI", expected7),
    (BQ4, 1, p3, "TPI", expected8),
]


@pytest.mark.parametrize(
    "BQ,j,p,method,expected",
    test_data,
    ids=[
        "SS, use zeta, all j",
        "SS, use zeta, one j",
        "TPI, use zeta, all j",
        "TPI, use zeta, one j",
        "SS, not use zeta, all j",
        "SS, not use zeta, one j",
        "TPI, not use zeta, all j",
        "TPI, not use zeta, one j",
    ],
)
def test_get_bq(BQ, j, p, method, expected):
    # Test the get_bq function
    test_value = household.get_bq(BQ, j, p, method)
    print("Test value = ", test_value)
    assert np.allclose(test_value, expected)


p1 = Specifications()
p1.eta = np.tile(
    np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]]).reshape(1, p2.S, p2.J),
    (p2.T, 1, 1),
)
p1.S = 3
p1.J = 2
p1.T = 3
p1.lambdas = np.array([0.6, 0.4])
p1.omega_SS = np.array([0.25, 0.25, 0.5])
p1.omega = np.tile(p1.omega_SS.reshape((1, p1.S)), (p1.T, 1))
TR1 = 2.5
expected1 = np.array([[1.66666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]])
p2 = Specifications()
p2.S = 3
p2.rho = np.array([[0.0, 0.0, 1.0]])
p2.J = 2
p2.T = 3
p2.eta = np.tile(
    np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]]).reshape(1, p2.S, p2.J),
    (p2.T, 1, 1),
)
p2.lambdas = np.array([0.6, 0.4])
p2.omega_SS = np.array([0.25, 0.25, 0.5])
p2.omega = np.tile(p2.omega_SS.reshape((1, p2.S)), (p2.T, 1))
TR2 = np.array([2.5, 0.8, 3.6])
expected2 = np.array([7.5, 10.0, 0.0])
expected3 = np.array(
    [
        [[1.666666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]],
        [[0.533333333, 2.4], [0.8, 3.2], [0.133333333, 0.0]],
        [[2.4, 10.8], [3.6, 14.4], [0.6, 0.0]],
    ]
)
expected4 = np.array([[7.5, 10.0, 0.0], [2.4, 3.2, 0.0], [10.8, 14.4, 0.0]])
test_data = [
    (TR1, None, p1, "SS", expected1),
    (TR1, 1, p1, "SS", expected2),
    (TR2, None, p2, "TPI", expected3),
    (TR2, 1, p2, "TPI", expected4),
]


@pytest.mark.parametrize(
    "TR,j,p,method,expected",
    test_data,
    ids=["SS, all j", "SS, one j", "TPI, all j", "TPI, one j"],
)
def test_get_tr(TR, j, p, method, expected):
    # Test the get_tr function
    test_value = household.get_tr(TR, j, p, method)
    print("Test value = ", test_value)
    assert np.allclose(test_value, expected)


# Set up test for get_rm
(
    RM1,
    RM2,
    RM3,
    RM4,
    expected_rm1,
    expected_rm2,
    expected_rm3,
    expected_rm4,
) = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "RMrm_test_tuple.pkl")
)
j1 = 0
p_rm_1 = Specifications()
p_rm_2 = copy.deepcopy(p_rm_1)
p_rm_2.alpha_RM_1 = 0.05
p_rm_2.alpha_RM_T = 0.05
p_rm_2.g_RM = ((np.exp(p_rm_2.g_y) * (1 + p_rm_2.g_n_ss)) - 1) * np.ones(
    p_rm_2.T + p_rm_2.S
)
j2 = 3
j3 = 4
j4 = 6
p_rm_4 = copy.deepcopy(p_rm_2)
p_rm_4.g_RM = (
    (np.exp(p_rm_4.g_y) * (1 + p_rm_4.g_n_ss)) - 1 + 0.005
) * np.ones(p_rm_4.T + p_rm_4.S)
test_data_rm = [
    (RM1, j1, p_rm_1, "SS", expected_rm1),
    (RM2, j2, p_rm_2, "SS", expected_rm2),
    (RM3, j3, p_rm_2, "TPI", expected_rm3),
    (RM4, j4, p_rm_4, "TPI", expected_rm4),
]


@pytest.mark.parametrize(
    "RM,j,p,method,expected",
    test_data_rm,
    ids=["SS, zero", "SS, rm>0", "TPI, model growth", "TPI, bigger growth"],
)
def test_get_rm(RM, j, p, method, expected):
    # Test the get_rm function
    rm = household.get_rm(RM, j, p, method)
    print("Test value = ", rm)
    assert np.allclose(rm, expected)


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
rm1 = 0.0
tau_c1 = 0.05
bq1 = BQ1 / p1.lambdas
net_tax1 = 0.02
j1 = None

p2 = Specifications()
p2.e = np.array([0.99, 1.5, 0.2])
p2.lambdas = np.array([0.25])
p2.g_y = 0.03
p2.T = 3
p2.J = 1
r2 = np.array([0.05, 0.04, 0.09])
w2 = np.array([1.2, 0.8, 2.5])
b2 = np.array([0.5, 0.99, 9])
b_splus1_2 = np.array([0.55, 0.2, 4])
n2 = np.array([0.8, 3.2, 0.2])
tau_c2 = np.array([0.08, 0.32, 0.02])
BQ2 = np.array([0.1, 2.4, 0.2])
bq2 = BQ2 / p2.lambdas
rm2 = np.zeros_like(bq2)
net_tax2 = np.array([0.02, 0.5, 1.4])
j2 = None

p3 = Specifications()
p3.e = np.array([[1.0, 2.1], [0.4, 0.5], [1.6, 0.9]])
p3.lambdas = np.array([0.4, 0.6])
p3.g_y = 0.01
p3.S = 3
p3.rho = np.array([[0.0, 0.0, 1.0]])
p3.J = 2
r3 = 0.11
w3 = 0.75
b3 = np.array([[0.56, 0.7], [0.4, 0.95], [2.06, 1.7]])
b_splus1_3 = np.array([[0.4, 0.6], [0.33, 1.95], [1.6, 2.7]])
n3 = np.array([[0.9, 0.5], [0.8, 1.1], [0, 0.77]])
BQ3 = np.array([1.3, 0.3])
bq3 = BQ3 / p3.lambdas.reshape(p3.J)
rm3 = np.zeros_like(bq3)
tau_c3 = np.array([[0.09, 0.05], [0.08, 0.11], [0.0, 0.077]])
net_tax3 = np.array([[0.1, 1.1], [0.4, 0.44], [0.6, 1.7]])
j3 = None

p4 = Specifications()
p4.e = np.array([[1.0, 2.1], [0.4, 0.5], [1.6, 0.9]])
p4.lambdas = np.array([0.7, 0.3])
p4.g_y = 0.05
r4 = np.tile(
    np.reshape(np.array([0.11, 0.02, 0.08, 0.05]), (4, 1, 1)), (1, 3, 2)
)
w4 = np.tile(np.reshape(np.array([0.75, 1.3, 0.9, 0.7]), (4, 1, 1)), (1, 3, 2))
b4 = np.array(
    [
        np.array([[0.5, 0.55], [0.6, 0.9], [0.9, 0.4]]),
        np.array([[7.1, 8.0], [1.0, 2.1], [9.1, 0.1]]),
        np.array([[0.4, 0.2], [0.34, 0.56], [0.3, 0.6]]),
        np.array([[0.1, 0.2], [0.4, 0.5], [0.555, 0.76]]),
    ]
)
b_splus1_4 = np.array(
    [
        np.array([[0.4, 0.2], [1.4, 1.5], [0.5, 0.6]]),
        np.array([[7.1, 8.0], [0.4, 0.9], [9.1, 10]]),
        np.array([[0.15, 0.52], [0.44, 0.85], [0.5, 0.6]]),
        np.array([[4.1, 2.0], [0.65, 0.65], [0.25, 0.56]]),
    ]
)
n4 = np.array(
    [
        np.array([[0.8, 0.9], [0.4, 0.5], [0.55, 0.66]]),
        np.array([[0.7, 0.8], [0.2, 0.1], [0, 0.4]]),
        np.array([[0.1, 0.2], [1.4, 1.5], [0.5, 0.6]]),
        np.array([[0.4, 0.6], [0.99, 0.44], [0.35, 0.65]]),
    ]
)
BQ4 = np.tile(
    np.reshape(
        np.array([[0.1, 1.1], [0.4, 1.0], [0.6, 1.7], [0.9, 2.0]]), (4, 1, 2)
    ),
    (1, 3, 1),
)
bq4 = BQ4 / p4.lambdas.reshape(1, 1, 2)
rm4 = np.zeros_like(bq4)
tau_c4 = np.array(
    [
        np.array([[0.02, 0.03], [0.04, 0.05], [0.055, 0.066]]),
        np.array([[0.07, 0.08], [0.02, 0.01], [0.0, 0.04]]),
        np.array([[0.01, 0.02], [0.14, 0.15], [0.05, 0.06]]),
        np.array([[0.04, 0.06], [0.099, 0.044], [0.035, 0.065]]),
    ]
)
net_tax4 = np.array(
    [
        np.array([[0.01, 0.02], [0.4, 0.5], [0.05, 0.06]]),
        np.array([[0.17, 0.18], [0.08, 0.02], [0.9, 0.10]]),
        np.array([[1.0, 2.0], [0.04, 0.25], [0.15, 0.16]]),
        np.array([[0.11, 0.021], [0.044, 0.025], [0.022, 0.032]]),
    ]
)
j4 = None

test_data = [
    (
        (r1, w1, b1, b_splus1_1, n1, bq1, rm1, net_tax1, p1),
        1.288650006,
    ),
    (
        (r2, w2, b2, b_splus1_2, n2, bq2, rm2, net_tax2, p2),
        np.array([1.288650006, 13.76350909, 5.188181864]),
    ),
    (
        (r3, w3, b3, b_splus1_3, n3, bq3, rm3, net_tax3, p3),
        np.array(
            [
                [4.042579933, 0.3584699],
                [3.200683445, -0.442597826],
                [3.320519733, -1.520385451],
            ]
        ),
    ),
    (
        (r4, w4, b4, b_splus1_4, n4, bq4, rm4, net_tax4, p4),
        np.array(
            [
                np.array(
                    [
                        [0.867348704, 5.464412447],
                        [-0.942922392, 2.776260022],
                        [1.226221595, 3.865404009],
                    ]
                ),
                np.array(
                    [
                        [1.089403787, 5.087164562],
                        [1.194920133, 4.574189347],
                        [-0.613138406, -6.70937763],
                    ]
                ),
                np.array(
                    [
                        [0.221452193, 3.714005697],
                        [1.225783575, 5.802886235],
                        [1.225507309, 6.009904009],
                    ]
                ),
                np.array(
                    [
                        [-2.749497209, 5.635124474],
                        [1.255588073, 6.637340454],
                        [1.975646512, 7.253454853],
                    ]
                ),
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "model_args,expected",
    test_data,
    ids=["scalar", "vector", "matrix", "3D array"],
)
def test_get_cons(model_args, expected):
    # Test consumption calculation
    r, w, b, b_splus1, n, bq, rm, net_tax, p = model_args
    p_tilde = np.ones_like(w)
    test_value = household.get_cons(
        r, w, p_tilde, b, b_splus1, n, bq, rm, net_tax, p.e, p
    )

    assert np.allclose(test_value, expected)


"""
-------------------------------------------------------------------------------
test_FOC_savings() function
-------------------------------------------------------------------------------
"""
# Define variables for test of SS version
p1 = Specifications()
p1.e = np.array([1.0, 0.9, 1.4]).reshape(3, 1)
p1.sigma = 2.0
p1.J = 1
p1.beta = np.ones(p1.J) * 0.96
p1.g_y = 0.03
p1.chi_b = np.array([1.5])
p1.tau_bq = np.array([0.0])
p1.rho = np.array([[0.1, 0.2, 1.0]])
p1.lambdas = np.array([1.0])
p1.S = 3
p1.T = 3
p1.labor_income_tax_noncompliance_rate = np.zeros((p1.T + p1.S, p1.J))
p1.capital_income_tax_noncompliance_rate = np.zeros((p1.T + p1.S, p1.J))
p1.analytical_mtrs = False
etr_params = np.array(
    [
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.33, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.20, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0],
            ]
        ),
    ]
)
mtry_params = np.array(
    [
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.28, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.49, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.32, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.70, 0],
            ]
        ),
    ]
)
p1.h_wealth = np.array([0.1])
p1.m_wealth = np.array([1.0])
p1.p_wealth = np.array([0.0])
p1.tau_payroll = np.array([0.15])
p1.retire = np.array([2]).astype(int)

test_params_ss = p1
r = 0.05
w = 1.2
b = np.array([0.0, 0.8, 0.5])
b_splus1 = np.array([0.8, 0.5, 0.1])
n = np.array([0.9, 0.8, 0.5])
bq = 0.1
rm = 0.0
factor = 120000
tr = 0.22
ubi_ss = np.zeros(p1.S)
theta = np.array([0.1])
method = "SS"
j = None
test_vars_ss = (
    r,
    w,
    b,
    b_splus1,
    n,
    bq,
    rm,
    factor,
    tr,
    ubi_ss,
    theta,
    etr_params[-1, :, :],
    mtry_params[-1, :, :],
    None,
    j,
    method,
)
test_vars_ss0 = (
    r,
    w,
    b,
    b_splus1,
    n,
    bq,
    rm,
    factor,
    tr,
    ubi_ss,
    theta,
    etr_params[-1, :, :],
    mtry_params[-1, :, :],
    None,
    0,
    method,
)
expected_ss = np.array([9.9403099, -1.00478079, -140.55458776])

# Define variables/params for test of TPI version
method_tpi = "TPI"
test_params_tpi = copy.deepcopy(p1)
test_params_tpi.tau_payroll = np.array([0.15, 0.15, 0.15])
test_params_tpi.tau_bq = np.array([0.0, 0.0, 0.0])
test_params_tpi.retire = np.array([2, 2, 2]).astype(int)
test_params_tpi.h_wealth = np.array([0.1, 0.1, 0.1])
test_params_tpi.m_wealth = np.array([1.0, 1.0, 1.0])
test_params_tpi.p_wealth = np.array([0.0, 0.0, 0.0])
j = 0
r_vec = np.array([0.05, 0.03, 0.04])
w_vec = np.array([1.2, 0.9, 0.8])
b_path = np.tile(np.reshape(np.array([0.0, 0.8, 0.5]), (1, 3)), (3, 1))
b_splus1_path = np.tile(np.reshape(np.array([0.8, 0.5, 0.1]), (1, 3)), (3, 1))
n_path = np.tile(np.reshape(np.array([0.9, 0.8, 0.5]), (1, 3)), (3, 1))
bq_vec = np.array([0.1, 0.05, 0.15])
rm_vec = np.zeros_like(bq_vec)
tr_vec = np.array([0.22, 0.15, 0.0])
etr_params_tpi = np.empty((p1.S, etr_params.shape[2]))
mtry_params_tpi = np.empty((p1.S, mtry_params.shape[2]))
for i in range(etr_params.shape[2]):
    etr_params_tpi[:, i] = np.diag(np.transpose(etr_params[:, : p1.S, i]))
    mtry_params_tpi[:, i] = np.diag(np.transpose(mtry_params[:, : p1.S, i]))
test_vars_tpi = (
    r_vec,
    w_vec,
    np.diag(b_path),
    np.diag(b_splus1_path),
    np.diag(n_path),
    bq_vec,
    rm_vec,
    factor,
    tr_vec,
    ubi_ss,
    theta,
    etr_params_tpi,
    mtry_params_tpi,
    0,
    j,
    method_tpi,
)
expected_tpi = np.array([300.97703103, 2.71986664, -139.91872277])

# create parameter objects with non-zero tax noncompliance
test_params_ss_noncomply = copy.deepcopy(test_params_ss)
test_params_ss_noncomply.labor_income_tax_noncompliance_rate = (
    np.ones((test_params_ss.T, test_params_ss.J)) * 0.05
)
test_params_ss_noncomply.capital_income_tax_noncompliance_rate = (
    np.ones((test_params_ss.T, test_params_ss.J)) * 0.05
)
test_params_tpi_noncomply = copy.deepcopy(test_params_tpi)
test_params_tpi_noncomply.labor_income_tax_noncompliance_rate = (
    np.ones((test_params_tpi.T, test_params_tpi.J)) * 0.05
)
test_params_tpi_noncomply.capital_income_tax_noncompliance_rate = (
    np.ones((test_params_tpi.T, test_params_tpi.J)) * 0.05
)

expected_ss_noncomply = np.array([9.57729582, -0.99595713, -140.57731873])
expected_tpi_noncomply = np.array([173.72734003, 2.16357338, -139.95857116])


# Define variables for test of SS and TPI with non-zero wealth tax
test_params_ss_tau_w = copy.deepcopy(p1)
test_params_ss_tau_w.h_wealth = np.array([0.305509])
test_params_ss_tau_w.m_wealth = np.array([2.16051])
test_params_ss_tau_w.p_wealth = np.array([0.025])
expected_ss_tau_w = np.array([9.94107316, -1.00174574, -140.5535989])

test_params_tpi_tau_w = copy.deepcopy(test_params_tpi)
test_params_tpi_tau_w.h_wealth = np.array([0.305509, 0.305509, 0.305509])
test_params_tpi_tau_w.m_wealth = np.array([2.16051, 2.16051, 2.16051])
test_params_tpi_tau_w.p_wealth = np.array([0.025, 0.025, 0.025])
expected_tpi_tau_w = np.array([300.95971044, 2.76460318, -139.91614123])

test_data = [
    (test_vars_ss, test_params_ss, expected_ss),
    (test_vars_tpi, test_params_tpi, expected_tpi),
    (test_vars_ss, test_params_ss_tau_w, expected_ss_tau_w),
    (test_vars_tpi, test_params_tpi_tau_w, expected_tpi_tau_w),
    (test_vars_ss0, test_params_ss, expected_ss),
    (test_vars_ss0, test_params_ss_noncomply, expected_ss_noncomply),
    (test_vars_ss, test_params_ss_noncomply, expected_ss_noncomply),
    (test_vars_tpi, test_params_tpi_noncomply, expected_tpi_noncomply),
]


@pytest.mark.parametrize(
    "model_vars,in_params,expected",
    test_data,
    ids=[
        "SS",
        "TPI",
        "SS - wealth tax",
        "TPI - wealth tax",
        "SS - j =0",
        "SS, j=0, noncomply",
        "SS, j=None, noncomply",
        "TPI, j=0, noncomply",
    ],
)
def test_FOC_savings(model_vars, in_params, expected):
    # Test FOC condition for household's choice of savings
    (
        r,
        w,
        b,
        b_splus1,
        n,
        BQ,
        rm,
        factor,
        tr,
        ubi,
        theta,
        etr_params,
        mtry_params,
        t,
        j,
        method,
    ) = model_vars
    params = copy.deepcopy(in_params)
    # reshape e matrix to be 3D
    params.e = np.tile(
        params.e.reshape(1, params.S, params.J), (params.T, 1, 1)
    )
    if method == "TPI":
        p_tilde = np.ones_like(w)
    else:
        p_tilde = np.array([1.0])
    if j is not None:
        test_value = household.FOC_savings(
            r,
            w,
            p_tilde,
            b,
            b_splus1,
            n,
            BQ,
            rm,
            factor,
            tr,
            ubi,
            theta,
            params.rho[-1, :],
            etr_params,
            mtry_params,
            t,
            j,
            params,
            method,
        )
    else:
        test_value = household.FOC_savings(
            r,
            w,
            p_tilde,
            b,
            b_splus1,
            n,
            BQ,
            rm,
            factor,
            tr,
            ubi,
            theta,
            params.rho[-1, :],
            etr_params,
            mtry_params,
            t,
            j,
            params,
            method,
        )
    print(test_value)
    assert np.allclose(test_value, expected)


"""
-------------------------------------------------------------------------------
test_FOC_labor() function
-------------------------------------------------------------------------------
"""
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
p1.labor_income_tax_noncompliance_rate = np.zeros((p1.T + p1.S, p1.J))
p1.capital_income_tax_noncompliance_rate = np.zeros((p1.T + p1.S, p1.J))
p1.analytical_mtrs = False
etr_params = np.array(
    [
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.33, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.20, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45, 0],
            ]
        ),
    ]
)
mtrx_params = np.array(
    [
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.44, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.77, 0],
            ]
        ),
    ]
)
p1.h_wealth = np.array([0.1])
p1.m_wealth = np.array([1.0])
p1.p_wealth = np.array([0.0])
p1.tau_payroll = np.array([0.15])
p1.retire = np.array([2]).astype(int)
theta = np.array([0.1])
j = 0
method = "SS"
r = 0.05
w = 1.2
b = np.array([0.0, 0.8, 0.5])
b_splus1 = np.array([0.8, 0.5, 0.1])
n = np.array([0.9, 0.8, 0.5])
bq = 0.1
rm = 0.0
factor = 120000
tr = 0.22
ubi_ss = np.zeros(p1.S)
test_params_ss = p1
test_vars_ss = (
    r,
    w,
    b,
    b_splus1,
    n,
    bq,
    rm,
    factor,
    tr,
    ubi_ss,
    theta,
    etr_params[-1, :, :],
    mtrx_params[-1, :, :],
    None,
    j,
    method,
)
expected_ss = np.array([4.77647028, 0.14075522, -0.14196852])

# Define variables/params for test of TPI version
method_tpi = "TPI"
test_params_tpi = copy.deepcopy(p1)
j = 0
test_params_tpi.retire = np.array([2, 2, 2]).astype(int)
test_params_tpi.h_wealth = np.array([0.1, 0.1, 0.1])
test_params_tpi.m_wealth = np.array([1.0, 1.0, 1.0])
test_params_tpi.p_wealth = np.array([0.0, 0.0, 0.0])
test_params_tpi.tau_payroll = np.array([0.15, 0.15, 0.15])
test_params_tpi.tau_bq = np.array([0.0, 0.0, 0.0])
r_vec = np.array([0.05, 0.03, 0.04])
w_vec = np.array([1.2, 0.9, 0.8])
b_path = np.tile(np.reshape(np.array([0.0, 0.8, 0.5]), (1, 3)), (3, 1))
b_splus1_path = np.tile(np.reshape(np.array([0.8, 0.5, 0.1]), (1, 3)), (3, 1))
n_path = np.tile(np.reshape(np.array([0.9, 0.8, 0.5]), (1, 3)), (3, 1))
bq_vec = np.tile(np.array([0.1, 0.05, 0.15]).reshape(3, 1), (1, 3))
rm_vec = np.zeros_like(bq_vec)
tr_vec = np.tile(np.array([0.22, 0.15, 0.0]).reshape(3, 1), (1, 3))
etr_params_tpi = np.empty((p1.S, etr_params.shape[2]))
mtrx_params_tpi = np.empty((p1.S, mtrx_params.shape[2]))
etr_params_tpi = etr_params
mtrx_params_tpi = mtrx_params
test_vars_tpi = (
    r_vec,
    w_vec,
    b_path,
    b_splus1_path,
    n_path,
    bq_vec,
    rm_vec,
    factor,
    tr_vec,
    ubi_ss,
    theta,
    etr_params_tpi,
    mtrx_params_tpi,
    0,
    j,
    method_tpi,
)
expected_tpi = np.array(
    [
        [6.93989849e01, 7.03703184e-03, 4.32040026e-01],
        [5.07350175e01, 1.93091572e00, -3.18176601e-01],
        [2.51596643e05, 2.15801427e-01, -1.33902455e-01],
    ]
)

# Define variables/params for test of TPI version
method_tpi = "TPI"
test_params_tau_pay = copy.deepcopy(p1)
test_params_tau_pay.retire = np.array([2, 2, 2]).astype(int)
test_params_tau_pay.h_wealth = np.array([0.1, 0.1, 0.1])
test_params_tau_pay.m_wealth = np.array([1.0, 1.0, 1.0])
test_params_tau_pay.p_wealth = np.array([0.0, 0.0, 0.0])
test_params_tau_pay.tau_payroll = np.array([0.11, 0.05, 0.33])
test_params_tau_pay.tau_bq = np.array([0.0, 0.0, 0.0])
expected_tau_pay = np.array(
    [
        [2.83370314e01, 2.49648863e-02, 4.47443095e-01],
        [1.47067455e01, 1.73279932e00, -1.80823501e-01],
        [3.50954120e05, 1.39637342e-01, -4.15072835e-01],
    ]
)

# create parameter objects with non-zero tax noncompliance
test_params_ss_noncomply = copy.deepcopy(test_params_ss)
test_params_ss_noncomply.labor_income_tax_noncompliance_rate = (
    np.ones((test_params_ss.T, test_params_ss.J)) * 0.05
)
test_params_ss_noncomply.capital_income_tax_noncompliance_rate = (
    np.ones((test_params_ss.T, test_params_ss.J)) * 0.05
)
test_params_tpi_noncomply = copy.deepcopy(test_params_tpi)
test_params_tpi_noncomply.labor_income_tax_noncompliance_rate = (
    np.ones((test_params_tpi.T, test_params_tpi.J)) * 0.05
)
test_params_tpi_noncomply.capital_income_tax_noncompliance_rate = (
    np.ones((test_params_tpi.T, test_params_tpi.J)) * 0.05
)

expected_ss_noncomply = np.array([4.69251429, 0.14527838, -0.09559029])
expected_tpi_noncomply = np.array(
    [
        [4.41773424e01, 2.08215139e-02, 4.34837339e-01],
        [5.10759447e01, 1.71845336e00, -2.53093346e-01],
        [2.48092106e05, 2.21159842e-01, -8.36295540e-02],
    ]
)

test_data = [
    (test_vars_ss, test_params_ss, expected_ss),
    (test_vars_tpi, test_params_tpi, expected_tpi),
    (test_vars_tpi, test_params_tau_pay, expected_tau_pay),
    (test_vars_ss, test_params_ss_noncomply, expected_ss_noncomply),
    (test_vars_tpi, test_params_tpi_noncomply, expected_tpi_noncomply),
]


@pytest.mark.parametrize(
    "model_vars,params,expected",
    test_data,
    ids=["SS", "TPI", "vary tau_payroll", "SS, noncomply", "TPI, noncomply"],
)
def test_FOC_labor(model_vars, params, expected):
    # Test FOC condition for household's choice of labor supply
    (
        r,
        w,
        b,
        b_splus1,
        n,
        bq,
        rm,
        factor,
        tr,
        ubi,
        theta,
        etr_params,
        mtrx_params,
        t,
        j,
        method,
    ) = model_vars
    # reshape e matrix for 3D
    params.e = np.tile(
        params.e.reshape(1, params.S, params.J), (params.T, 1, 1)
    )
    if method == "TPI":
        p_tilde = np.ones_like(w)
    else:
        p_tilde = np.array([1.0])
    test_value = household.FOC_labor(
        r,
        w,
        p_tilde,
        b,
        b_splus1,
        n,
        bq,
        rm,
        factor,
        tr,
        ubi,
        theta,
        params.chi_n,
        etr_params,
        mtrx_params,
        t,
        j,
        params,
        method,
    )

    assert np.allclose(test_value, expected)


def test_get_y():
    """
    Test of household.get_y() function.
    """
    r_p = np.array([0.05, 0.04, 0.09])
    w = np.array([1.2, 0.8, 2.5])
    b_s = np.array([0.5, 0.99, 9])
    n = np.array([0.8, 3.2, 0.2])
    expected_y = np.array([0.9754, 3.8796, 0.91])
    p = Specifications()
    # p.update_specifications({'S': 4, 'J': 1})
    p.S = 3
    p.rho = np.array([[0.0, 0.0, 1.0]])
    p.e = np.array([0.99, 1.5, 0.2])
    p.e = np.tile(p.e.reshape(1, p.S, 1), (p.T, 1, 1))

    test_y = household.get_y(r_p, w, b_s, n, p, "SS")
    # TODO: test with "TPI"

    assert np.allclose(test_y, expected_y)


bssmat0 = np.array([[0.1, 0.2], [0.3, 0.4]])
nssmat0 = np.array([[0.1, 0.2], [0.3, 0.4]])
cssmat0 = np.array([[0.1, 0.2], [0.3, 0.4]])

bssmat1 = np.array([[-0.1, -0.2], [-0.3, -0.4]])
nssmat1 = np.array([[-0.1, -0.2], [3.3, 4.4]])
cssmat1 = np.array([[-0.1, -0.2], [-0.3, -0.4]])
test_data = [
    (bssmat0, nssmat0, cssmat0, 1.0),
    (bssmat1, nssmat1, cssmat1, 1.0),
]


@pytest.mark.parametrize(
    "bssmat,nssmat,cssmat,ltilde", test_data, ids=["passing", "failing"]
)
def test_constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    household.constraint_checker_SS(bssmat, nssmat, cssmat, ltilde)
    assert True


@pytest.mark.parametrize(
    "bssmat,nssmat,cssmat,ltilde", test_data, ids=["passing", "failing"]
)
def test_constraint_checker_TPI(bssmat, nssmat, cssmat, ltilde):
    household.constraint_checker_TPI(bssmat, nssmat, cssmat, 10, ltilde)
    assert True


def test_ci():
    """
    Test of the get_ci function
    """
    c_s = np.array([2.0, 3.0, 5.0, 7.0]).reshape(4, 1)
    p_i = np.array([1.1, 0.8, 1.0])
    p_tilde = np.array([2.3])
    tau_c = np.array([0.2, 0.3, 0.5])
    alpha_c = np.array([0.5, 0.3, 0.2])
    expected_ci = np.array(
        [
            [1.742424242, 2.613636364, 4.356060606, 6.098484848],
            [1.326923077, 1.990384615, 3.317307692, 4.644230769],
            [0.613333333, 0.92, 1.533333333, 2.146666667],
        ]
    ).reshape(3, 4, 1)

    test_ci = household.get_ci(c_s, p_i, p_tilde, tau_c, alpha_c)

    assert np.allclose(test_ci, expected_ci)
