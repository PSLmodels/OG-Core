import pytest
import numpy as np
import copy
from ogcore import aggregates as aggr
from ogcore.parameters import Specifications


p = Specifications()
new_param_values = {
    "T": 160,
    "S": 40,
    "J": 2,
    "eta": (np.ones((40, 2)) / (40 * 2)),
    "lambdas": [0.6, 0.4],
    "omega": np.ones((160, 40)) / 40,
    "omega_SS": np.ones(40) / 40,
    "e": np.ones((40, 2)),
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
n = np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
L_loop = np.ones(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
for t in range(p.T):
    for i in range(p.S):
        for k in range(p.J):
            L_loop[t, i, k] *= (
                p.omega[t, i] * p.lambdas[k] * n[t, i, k] * p.e[i, k]
            )
expected1 = L_loop[-1, :, :].sum()
expected2 = L_loop.sum(1).sum(1)
test_data = [(n[-1, :, :], p, "SS", expected1), (n, p, "TPI", expected2)]


@pytest.mark.parametrize("n,p,method,expected", test_data, ids=["SS", "TPI"])
def test_get_L(n, p, method, expected):
    """
    Test aggregate labor function.
    """
    L = aggr.get_L(n, p, method)
    assert np.allclose(L, expected)


p = Specifications()
new_param_values = {
    "T": 160,
    "S": 40,
    "J": 2,
    "eta": (np.ones((40, 2)) / (40 * 2)),
    "lambdas": [0.6, 0.4],
    "omega": np.ones((160, 40)) / 40,
    "omega_SS": np.ones(40) / 40,
    "imm_rates": np.zeros((160, 40)),
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
b_splus1 = 10 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
K_p1 = 0.9 + np.random.rand(p.T)
K = 0.9 + np.random.rand(p.T)
omega_extended = np.append(p.omega_SS[1:], [0.0])
imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
part2 = (
    (
        b_splus1[-1, :, :]
        * np.transpose((omega_extended * imm_extended) * p.lambdas)
    ).sum()
) / (1 + p.g_n_ss)
aggI_SS = (1 + p.g_n_ss) * np.exp(p.g_y) * (K_p1[-1] - part2) - (
    1.0 - p.delta
) * K[-1]
omega_shift = np.append(p.omega[: p.T, 1:], np.zeros((p.T, 1)), axis=1)
imm_shift = np.append(p.imm_rates[: p.T, 1:], np.zeros((p.T, 1)), axis=1)
part2 = (
    (
        (b_splus1 * np.squeeze(p.lambdas))
        * np.tile(
            np.reshape(imm_shift * omega_shift, (p.T, p.S, 1)), (1, 1, p.J)
        )
    )
    .sum(1)
    .sum(1)
) / (1 + np.squeeze(np.hstack((p.g_n[1 : p.T], p.g_n_ss))))
aggI_TPI = (1 + np.squeeze(np.hstack((p.g_n[1 : p.T], p.g_n_ss)))) * np.exp(
    p.g_y
) * (K_p1 - part2) - (1.0 - p.delta) * K
test_data = [
    (b_splus1[-1, :, :], K_p1[-1], K[-1], p, "SS", aggI_SS),
    (b_splus1, K_p1, K, p, "TPI", aggI_TPI),
]
aggI_total_SS = (1 + p.g_n_ss) * np.exp(p.g_y) * (K[-1]) - (1.0 - p.delta) * K[
    -1
]
aggI_total_TPI = (
    (1 + np.squeeze(np.hstack((p.g_n[1 : p.T], p.g_n_ss))))
    * np.exp(p.g_y)
    * K_p1
) - (1.0 - p.delta) * K
test_data = [
    (b_splus1[-1, :, :], K_p1[-1], K[-1], p, "SS", aggI_SS),
    (b_splus1, K_p1, K, p, "TPI", aggI_TPI),
    (None, K[-1], K[-1], p, "total_ss", aggI_total_SS),
    (None, K_p1, K, p, "total_tpi", aggI_total_TPI),
]


@pytest.mark.parametrize(
    "b_splus1,K_p1,K,p,method,expected",
    test_data,
    ids=["SS", "TPI", "total_ss", "total_tpi"],
)
def test_get_I(b_splus1, K_p1, K, p, method, expected):
    """
    Text aggregate investment function.
    """
    aggI = aggr.get_I(b_splus1, K_p1, K, p, method)
    assert np.allclose(aggI, expected)


p = Specifications()
new_param_values = {
    "T": 160,
    "S": 40,
    "J": 2,
    "eta": (np.ones((40, 2)) / (40 * 2)),
    "lambdas": [0.6, 0.4],
    "omega": np.ones((160, 40)) / 40,
    "omega_SS": np.ones(40) / 40,
    "imm_rates": np.zeros((160, 40)),
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
p.omega_S_preTP = p.omega[0, :]
b = -0.1 + (7 * np.random.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J))
omega_extended = np.append(p.omega[: p.T, 1:], np.zeros((p.T, 1)), axis=1)
imm_extended = np.append(p.imm_rates[: p.T, 1:], np.zeros((p.T, 1)), axis=1)
B_test = (
    b
    * np.squeeze(p.lambdas)
    * np.tile(np.reshape(p.omega[: p.T, :], (p.T, p.S, 1)), (1, 1, p.J))
) + (
    b
    * np.squeeze(p.lambdas)
    * np.tile(
        np.reshape(omega_extended * imm_extended, (p.T, p.S, 1)), (1, 1, p.J)
    )
)
expected1 = B_test[-1, :, :].sum() / (1.0 + p.g_n_ss)
expected2 = B_test.sum(1).sum(1) / (
    1.0 + np.hstack((p.g_n[1 : p.T], p.g_n_ss))
)
expected3 = B_test[0, :, :].sum() / (1.0 + p.g_n[0])
test_data = [
    (b[-1, :, :], p, "SS", False, expected1),
    (b, p, "TPI", False, expected2),
    (b[0, :, :], p, "SS", True, expected3),
]


@pytest.mark.parametrize(
    "b,p,method,PreTP,expected", test_data, ids=["SS", "TPI", "Pre-TP"]
)
def test_get_B(b, p, method, PreTP, expected):
    """
    Test aggregate savings function.
    """
    B = aggr.get_B(b, p, method, PreTP)
    assert np.allclose(B, expected)


p = Specifications()
new_param_values = {
    "T": 160,
    "S": 40,
    "J": 2,
    "eta": (np.ones((40, 2)) / (40 * 2)),
    "lambdas": [0.6, 0.4],
    "omega": np.ones((160, 40)) / 40,
    "omega_SS": np.ones(40) / 40,
    "imm_rates": np.zeros((160, 40)),
    "rho": np.zeros(40),
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
p.omega_S_preTP = p.omega[0, :]
# set values for some variables
r = 0.5 + 0.5 * np.random.rand(p.T)
b_splus1 = 0.06 + 7 * np.random.rand(p.T, p.S, p.J)
pop = np.append(p.omega_S_preTP.reshape(1, p.S), p.omega[: p.T - 1, :], axis=0)
BQ_presum = (b_splus1 * np.squeeze(p.lambdas)) * np.tile(
    np.reshape(p.rho * pop, (p.T, p.S, 1)), (1, 1, p.J)
)
growth_adj = (1.0 + r) / (1.0 + p.g_n[: p.T])

expected1 = BQ_presum[-1, :, :].sum(0) * growth_adj[-1]
expected2 = BQ_presum[-1, :, 1].sum(0) * growth_adj[-1]
expected3 = BQ_presum.sum(1) * np.tile(
    np.reshape(growth_adj, (p.T, 1)), (1, p.J)
)
expected4 = BQ_presum[:, :, 1].sum(1) * growth_adj
expected5 = BQ_presum[0, :, :].sum(0) * growth_adj[0]
expected6 = BQ_presum[0, :, 1].sum(0) * growth_adj[0]

p2 = copy.deepcopy(p)
p2.use_zeta = True
expected7 = BQ_presum[-1, :, 1].sum() * growth_adj[-1]
expected8 = BQ_presum[:, :, 1].sum(1) * growth_adj
expected9 = (
    BQ_presum.sum(1) * np.tile(np.reshape(growth_adj, (p.T, 1)), (1, p.J))
).sum(1)
test_data = [
    (r[-1], b_splus1[-1, :, :], None, p, "SS", False, expected1),
    (r[-1], b_splus1[-1, :, 1], 1, p, "SS", False, expected2),
    (r, b_splus1, None, p, "TPI", False, expected3),
    (r, b_splus1[:, :, 1], 1, p, "TPI", False, expected4),
    (r[0], b_splus1[0, :, :], None, p, "SS", True, expected5),
    (r[0], b_splus1[0, :, 1], 1, p, "SS", True, expected6),
    (r[-1], b_splus1[-1, :, 1], 1, p2, "SS", False, expected7),
    (r, b_splus1[:, :, 1], 1, p2, "TPI", False, expected8),
    (r, b_splus1, None, p2, "TPI", False, expected9),
]


@pytest.mark.parametrize(
    "r,b_splus1,j,p,method,PreTP,expected",
    test_data,
    ids=[
        "SS, all j",
        "SS, one j",
        "TPI, all j",
        "TPI, one j",
        "Pre-TP, all j",
        "Pre-TP, one j",
        "Use zeta, SS, one j",
        "Use zeta, TPI, one j",
        "Use zeta, TPI, all j",
    ],
)
def test_get_BQ(r, b_splus1, j, p, method, PreTP, expected):
    """
    Test of aggregate bequest function.
    """
    BQ = aggr.get_BQ(r, b_splus1, j, p, method, PreTP)
    assert np.allclose(BQ, expected)


p = Specifications()
new_param_values = {
    "T": 160,
    "S": 40,
    "J": 2,
    "M": 3,
    "eta": (np.ones((40, 2)) / (40 * 2)),
    "lambdas": [0.6, 0.4],
    "omega": np.ones((160, 40)) / 40,
    "omega_SS": np.ones(40) / 40,
    "imm_rates": np.zeros((160, 40)),
}
# update parameters instance with new values for test
p.update_specifications(new_param_values)
# make up some consumption values for testing
c = 0.1 + 0.5 * np.random.rand(p.T * p.M * p.S * p.J).reshape(
    p.T, p.M, p.S, p.J
)
aggC_presum = (c * np.squeeze(p.lambdas)) * np.tile(
    np.reshape(p.omega[: p.T, :], (p.T, 1, p.S, 1)), (1, p.M, 1, p.J)
)
expected1 = aggC_presum[-1, -1, :, :].sum(-1).sum(-1)
expected2 = aggC_presum[:, -1, :, :].sum(-1).sum(-1)
test_data = [
    (c[-1, -1, :, :], p, "SS", expected1),
    (c[:, -1, :, :], p, "TPI", expected2),
]


@pytest.mark.parametrize("c,p,method,expected", test_data, ids=["SS", "TPI"])
def test_get_C(c, p, method, expected):
    """
    Test aggregate consumption function.
    """
    C = aggr.get_C(c, p, method)
    assert np.allclose(C, expected)


"""
-------------------------------------------------------------------------------
CI test of revenue() function
-------------------------------------------------------------------------------
"""
p = Specifications()
num_tax_params = 12
new_param_values = {
    "T": 30,
    "S": 20,
    "J": 2,
    "eta": (np.ones((20, 2)) / (20 * 2)),
    "lambdas": [0.6, 0.4],
    "tau_bq": [0.17],
    "tau_payroll": [0.5],
    "h_wealth": [0.1],
    "p_wealth": [0.2],
    "m_wealth": [1.0],
    "cit_rate": [[0.2]],
    "delta_tau_annual": [
        [float(1 - ((1 - 0.0975) ** (20 / (p.ending_age - p.starting_age))))]
    ],
    "omega": np.ones((30, 20)) / 20,
    "omega_SS": np.ones(20) / 20,
    "imm_rates": np.zeros((30, 20)),
}
p.update_specifications(new_param_values)
# make up some consumption values for testing
# Assign values to variables for tests
random_state = np.random.RandomState(10)
r = 0.067 + (0.086 - 0.067) * random_state.rand(p.T)
w = 0.866 + (0.927 - 0.866) * random_state.rand(p.T)
b = 6.94 * random_state.rand(p.T * p.S * p.J).reshape(p.T, p.S, p.J)
c = np.ones((p.T, p.S, p.J)) * 2.2
n = 0.191 + (0.503 - 0.191) * random_state.rand(p.T * p.S * p.J).reshape(
    p.T, p.S, p.J
)
BQ = 0.032 + (0.055 - 0.032) * random_state.rand(p.T * p.S * p.J).reshape(
    p.T, p.S, p.J
)
bq = BQ / p.lambdas.reshape(1, 1, p.J)
Y = 0.561 + (0.602 - 0.561) * random_state.rand(p.T).reshape(p.T, 1)
L = 0.416 + (0.423 - 0.416) * random_state.rand(p.T).reshape(p.T, 1)
K = 0.957 + (1.163 - 0.957) * random_state.rand(p.T).reshape(p.T, 1)
ubi = np.zeros((p.T, p.S, p.J))
factor = 140000.0
# update parameters instance with new values for test
p.e = 0.263 + (2.024 - 0.263) * random_state.rand(p.S * p.J).reshape(p.S, p.J)
p.omega = 0.039 * random_state.rand(p.T * p.S * 1).reshape(p.T, p.S)
p.omega = p.omega / p.omega.sum(axis=1).reshape(p.T, 1)
p.omega_SS = p.omega[-1, :]
etr_params = 0.22 * random_state.rand(
    p.T * p.S * p.J * num_tax_params
).reshape(p.T, p.S, p.J, num_tax_params)
theta = 0.101 + (0.156 - 0.101) * random_state.rand(p.J)

p3 = Specifications()
new_param_values3 = {
    "T": 30,
    "S": 20,
    "J": 2,
    "eta": (np.ones((20, 2)) / (20 * 2)),
    "lambdas": [0.6, 0.4],
    "tau_bq": [0.17],
    "tau_payroll": [0.5],
    "h_wealth": [0.1],
    "p_wealth": [0.2],
    "m_wealth": [1.0],
    "cit_rate": [[0.2]],
    "replacement_rate_adjust": [1.5, 1.5, 1.5, 1.6, 1.0],
    "delta_tau_annual": [
        [float(1 - ((1 - 0.0975) ** (20 / (p3.ending_age - p3.starting_age))))]
    ],
    "omega": np.ones((30, 20)) / 20,
    "omega_SS": np.ones(20) / 20,
    "imm_rates": np.zeros((30, 20)),
}
p3.update_specifications(new_param_values3)
p3.e = p.e
p3.omega = p.omega
p3.omega_SS = p.omega_SS

p_u = Specifications()
new_param_values_ubi = {
    "T": 30,
    "S": 20,
    "J": 2,
    "eta": (np.ones((20, 2)) / (20 * 2)),
    "lambdas": [0.6, 0.4],
    "tau_bq": [0.17],
    "tau_payroll": [0.5],
    "h_wealth": [0.1],
    "p_wealth": [0.2],
    "m_wealth": [1.0],
    "cit_rate": [[0.2]],
    "delta_tau_annual": [
        [
            float(
                1
                - ((1 - 0.0975) ** (20 / (p_u.ending_age - p_u.starting_age)))
            )
        ]
    ],
    "ubi_nom_017": 1000,
    "ubi_nom_1864": 1500,
    "ubi_nom_65p": 500,
}
p_u.update_specifications(new_param_values_ubi)
# make up some consumption values for testing
# Assign values to variables for tests
random_state = np.random.RandomState(10)
r_u = 0.067 + (0.086 - 0.067) * random_state.rand(p_u.T)
w_u = 0.866 + (0.927 - 0.866) * random_state.rand(p_u.T)
b_u = 6.94 * random_state.rand(p_u.T * p_u.S * p_u.J).reshape(
    p_u.T, p_u.S, p_u.J
)
c_u = np.ones((p_u.T, p_u.S, p_u.J)) * 2.2
n_u = 0.191 + (0.503 - 0.191) * random_state.rand(
    p_u.T * p_u.S * p_u.J
).reshape(p_u.T, p_u.S, p_u.J)
BQ_u = 0.032 + (0.055 - 0.032) * random_state.rand(
    p_u.T * p_u.S * p_u.J
).reshape(p_u.T, p_u.S, p_u.J)
bq_u = BQ_u / p_u.lambdas.reshape(1, 1, p_u.J)
Y_u = 0.561 + (0.602 - 0.561) * random_state.rand(p_u.T).reshape(p_u.T, 1)
L_u = 0.416 + (0.423 - 0.416) * random_state.rand(p_u.T).reshape(p_u.T, 1)
K_u = 0.957 + (1.163 - 0.957) * random_state.rand(p_u.T).reshape(p_u.T, 1)
factor_u = 140000.0
ubi_u = p_u.ubi_nom_array / factor_u
# update parameters instance with new values for test
p_u.e = 0.263 + (2.024 - 0.263) * random_state.rand(p_u.S * p_u.J).reshape(
    p_u.S, p_u.J
)
p_u.omega = 0.039 * random_state.rand(p_u.T * p_u.S * 1).reshape(p_u.T, p_u.S)
p_u.omega = p_u.omega / p_u.omega.sum(axis=1).reshape(p_u.T, 1)
p_u.omega_SS = p_u.omega[-1, :]
etr_params_u = 0.22 * random_state.rand(
    p_u.T * p_u.S * p_u.J * num_tax_params
).reshape(p_u.T, p_u.S, p_u.J, num_tax_params)
theta_u = 0.101 + (0.156 - 0.101) * random_state.rand(p_u.J)
# vector of output prices
p_m = np.ones((p.T, p.M))

expected1 = 0.5688319028341413
expected2 = np.array(
    [
        0.58978896,
        0.5318829,
        0.58291302,
        0.56616446,
        0.60152253,
        0.63685373,
        0.60718972,
        0.56236328,
        0.56929121,
        0.60536959,
        0.58763365,
        0.59627562,
        0.55409009,
        0.56782614,
        0.56400569,
        0.6636463,
        0.59160813,
        0.64735391,
        0.72066489,
        0.64096484,
        0.61899218,
        0.58806093,
        0.54783766,
        0.5393597,
        0.55685316,
        0.65395071,
        0.58946501,
        0.64129696,
        0.58759922,
        0.5580478,
    ]
)
expected3 = np.array(
    [
        0.58978896,
        0.5318829,
        0.58291302,
        0.56616446,
        0.60152253,
        0.63685373,
        0.60718972,
        0.56236328,
        0.56929121,
        0.60536959,
        0.58763365,
        0.59627562,
        0.55409009,
        0.56782614,
        0.56400569,
        0.6636463,
        0.59160813,
        0.64735391,
        0.72066489,
        0.64096484,
        0.61899218,
        0.58806093,
        0.54783766,
        0.5393597,
        0.55685316,
        0.65395071,
        0.58946501,
        0.64129696,
        0.58759922,
        0.5580478,
    ]
)
expected4 = 0.5688319028341413
expected5 = np.array(
    [
        0.58978896,
        0.5318829,
        0.58291302,
        0.56616446,
        0.60152253,
        0.63685373,
        0.60718972,
        0.56236328,
        0.56929121,
        0.60536959,
        0.58763365,
        0.59627562,
        0.55409009,
        0.56782614,
        0.56400569,
        0.6636463,
        0.59160813,
        0.64735391,
        0.72066489,
        0.64096484,
        0.61899218,
        0.58806093,
        0.54783766,
        0.5393597,
        0.55685316,
        0.65395071,
        0.58946501,
        0.64129696,
        0.58759922,
        0.5580478,
    ]
)
test_data = [
    (
        r[0],
        w[0],
        b[0, :, :],
        n[0, :, :],
        bq[0, :, :],
        c[0, :, :],
        Y[0],
        L[0],
        K[0],
        p_m[-1, :],
        factor,
        ubi[0, :, :],
        theta,
        etr_params[-1, :, :, :],
        p,
        None,
        "SS",
        expected1,
    ),
    (
        r,
        w,
        b,
        n,
        bq,
        c,
        Y,
        L,
        K,
        p_m,
        factor,
        ubi,
        theta,
        etr_params,
        p,
        None,
        "TPI",
        expected2,
    ),
    (
        r,
        w,
        b,
        n,
        bq,
        c,
        Y,
        L,
        K,
        p_m,
        factor,
        ubi,
        theta,
        etr_params,
        p3,
        None,
        "TPI",
        expected3,
    ),
    (
        r_u[0],
        w_u[0],
        b_u[0, :, :],
        n_u[0, :, :],
        bq_u[0, :, :],
        c_u[0, :, :],
        Y_u[0],
        L_u[0],
        K_u[0],
        p_m[-1, :],
        factor_u,
        ubi_u[0, :, :],
        theta_u,
        etr_params_u[-1, :, :, :],
        p_u,
        None,
        "SS",
        expected4,
    ),
    (
        r_u,
        w_u,
        b_u,
        n_u,
        bq_u,
        c_u,
        Y_u,
        L_u,
        K_u,
        p_m,
        factor_u,
        ubi_u,
        theta_u,
        etr_params_u,
        p_u,
        None,
        "TPI",
        expected5,
    ),
]


@pytest.mark.parametrize(
    "r,w,b,n,bq,c,Y,L,K,p_m,factor,ubi,theta,etr_params,p,m,method,expected",
    test_data,
    ids=["SS", "TPI", "TPI, replace rate adjust", "SS UBI>0", "TPI UBI>0"],
)
def test_revenue(
    r,
    w,
    b,
    n,
    bq,
    c,
    Y,
    L,
    K,
    p_m,
    factor,
    ubi,
    theta,
    etr_params,
    p,
    m,
    method,
    expected,
):
    """
    Test aggregate revenue function.
    """
    revenue, _, _, _, _, _, _, _, _, _ = aggr.revenue(
        r,
        w,
        b,
        n,
        bq,
        c,
        Y,
        L,
        K,
        p_m,
        factor,
        ubi,
        theta,
        etr_params,
        p,
        m,
        method,
    )
    print("Rev: ", revenue)
    print("Exp: ", expected)
    assert np.allclose(revenue, expected)


test_data = [
    (
        0.04,
        0.02,
        np.array([1.0]),
        np.array([2.0]),
        0.0,
        4.0,
        np.array([0.0]),
        "SS",
        0.026666667,
    ),
    (
        np.array([0.05, 0.03, 0.03]),
        np.array([0.02, 0.01, 0.01]),
        np.array([1.0, 1.0, 1.0]),
        np.array([3.0, 4.0, 4.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([7.0, 6.0, 6.0]),
        np.array([0.0, 0.0, 0.0]),
        "TPI",
        np.array([0.029, 0.018, 0.018]),
    ),
    (
        0.04,
        0.02,
        np.array([1.0]),
        np.array([2.0]),
        0.0,
        0.0,
        np.array([0.0]),
        "SS",
        0.04,
    ),
    (
        np.array([0.05, 0.03, 0.03]),
        np.array([0.02, 0.01, 0.01]),
        np.array([1.0, 1.0, 1.0]),
        np.array([3.0, 4.0, 4.0]),
        np.array([1.0, 2.0, 2.0]),
        np.array([7.0, 6.0, 6.0]),
        np.array([0.04, 0.2, 0.2]),
        "TPI",
        np.array([0.0328572, 0.056572, 0.056572]),
    ),
    (
        0.04,
        0.02,
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
        0.0,
        0.0,
        np.array([0.0]),
        "SS",
        np.array([0.04, 0.04]),
    ),
    (
        np.array([0.05, 0.03, 0.03]),
        np.array([0.02, 0.01, 0.01]),
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
        np.array([1.0, 2.0, 2.0]),
        np.array([7.0, 6.0, 6.0]),
        np.array([[0.04, 0.04, 0.04], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
        "TPI",
        np.array([0.04410725, 0.08762, 0.08762]),
    ),
]


@pytest.mark.parametrize(
    "r,r_gov,p_m,K_vec,K_g,D,MPKg_vec,method,expected",
    test_data,
    ids=[
        "SS, M=1",
        "TPI, M=1",
        "no debt",
        "TPI,MPKg>0",
        "SS, M>1",
        "TPI, M>1",
    ],
)
def test_get_r_p(r, r_gov, p_m, K_vec, K_g, D, MPKg_vec, method, expected):
    """
    Test function to compute interest rate on household portfolio.
    """
    p = Specifications()
    if method == "TPI" and p_m.ndim > 1:
        M = p_m.shape[-1]
    elif method == "SS":
        M = len(p_m)
    else:
        M = 1
    p.update_specifications({"T": 3, "M": M})

    r_p_test = aggr.get_r_p(r, r_gov, p_m, K_vec, K_g, D, MPKg_vec, p, method)

    assert np.allclose(r_p_test, expected)


def test_resource_constraint():
    """
    Test resource constraint equation.
    """
    Y = np.array([48, 55, 2, 99, 8])
    C = np.array([33, 44, 0.4, 55, 6])
    G = np.array([4, 5, 0.01, 22, 0])
    I_d = np.array([20, 5, 0.6, 10, 1])
    I_g = np.zeros_like(I_d)
    net_capital_flows = np.array([0.1, 0, 0.016, -1.67, -0.477])
    expected = np.array([-9.1, 1, 0.974, 13.67, 1.477])
    test_RC = aggr.resource_constraint(Y, C, G, I_d, I_g, net_capital_flows)

    assert np.allclose(test_RC, expected)


def test_get_capital_outflows():
    """
    Test of the get_captial_outflows function.
    """
    p = Specifications()
    p.delta = 0.05
    K_f = np.array([0, 0, 0.2, 3, 0.05])
    new_borrowing_f = np.array([0, 0.1, 0.3, 4, 0.5])
    debt_service_f = np.array([0.1, 0.1, 0.3, 2, 0.02])
    r = np.array([0.03, 0.04, 0.03, 0.06, 0.01])
    expected = np.array([0.1, 0, 0.016, -1.67, -0.477])

    test_flow = aggr.get_capital_outflows(
        r, K_f, new_borrowing_f, debt_service_f, p
    )

    assert np.allclose(test_flow, expected)


def test_get_K_splits():
    """
    Test of the get_K_splits function.
    """
    B = 2.2
    K_demand_open = 0.5
    D_d = 1.1
    zeta_K = 0.2

    expected_K_d = 1.1
    expected_K_f = 0.2 * (0.5 - (2.2 - 1.1))
    expected_K = expected_K_d + expected_K_f

    test_K, test_K_d, test_K_f = aggr.get_K_splits(
        B, K_demand_open, D_d, zeta_K
    )

    np.allclose(test_K, expected_K)
    np.allclose(test_K_d, expected_K_d)
    np.allclose(test_K_f, expected_K_f)


def test_get_K_splits_negative_K_d():
    """
    Test of the get_K_splits function for case where K_d < 0.
    """
    B = 2.2
    K_demand_open = 0.5
    D_d = 2.3
    zeta_K = 0.2

    expected_K_d = 0.05
    expected_K_f = 0.2 * (0.5 - (2.2 - 2.3))
    expected_K = expected_K_d + expected_K_f

    test_K, test_K_d, test_K_f = aggr.get_K_splits(
        B, K_demand_open, D_d, zeta_K
    )

    np.allclose(test_K, expected_K)
    np.allclose(test_K_d, expected_K_d)
    np.allclose(test_K_f, expected_K_f)


alpha_c = np.array([0.5, 0.3, 0.2])
tau_c = np.array([0.09, 0.07, 0.15])
p_m_ss = np.array([1.2, 1.3, 2.5])
p_m_tpi = np.tile(p_m_ss.reshape(1, 3), (3, 1))
expected_ss = 4.367191169
expected_tpi = np.array([4.367191169, 4.367191169, 4.367191169])


@pytest.mark.parametrize(
    "p_m,tau_c,alpha_c,method,expected",
    [
        (p_m_ss, tau_c, alpha_c, "SS", expected_ss),
        (p_m_tpi, tau_c, alpha_c, "TPI", expected_tpi),
    ],
    ids=[
        "SS",
        "TPI",
    ],
)
def test_get_ptilde(p_m, tau_c, alpha_c, method, expected):
    """
    Test of the get_ptilde function.
    """

    test_vals = aggr.get_ptilde(p_m, tau_c, alpha_c, method)

    assert np.allclose(test_vals, expected)
