import numpy as np
import copy
import pytest
from ogcore import tax
from ogcore.parameters import Specifications


b1 = np.array([0.1, 0.5, 0.9])
p1 = Specifications()
rho_vec = np.zeros((1, 3))
rho_vec[0, -1] = 1.0
new_param_values = {
    "S": 3,
    "rho": rho_vec.tolist(),
    "lambdas": [1.0],
    "J": 1,
    "T": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, 1)),
    "eta": (np.ones((3, 1)) / (3 * 1)),
    "h_wealth": [2],
    "p_wealth": [3],
    "m_wealth": [4],
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
}
p1.update_specifications(new_param_values)
expected1 = np.array([0.14285714, 0.6, 0.93103448])
p2 = Specifications()
new_param_values2 = {
    "S": 3,
    "rho": rho_vec.tolist(),
    "lambdas": [1.0],
    "J": 1,
    "T": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, 1)),
    "eta": (np.ones((3, 1)) / (3 * 1)),
    "h_wealth": [1.2, 1.1, 2.3],
    "p_wealth": [2.2, 2.3, 1.8],
    "m_wealth": [3, 4, 3],
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
}
p2.update_specifications(new_param_values2)
expected2 = np.array([0.084615385, 0.278021978, 0.734911243])

test_data = [(b1, p1, expected1), (b1, p2, expected2)]


@pytest.mark.parametrize(
    "b,p,expected", test_data, ids=["constant params", "vary params"]
)
def test_ETR_wealth(b, p, expected):
    # Test wealth tax computation
    tau_w = tax.ETR_wealth(
        b, p.h_wealth[: p.T], p.m_wealth[: p.T], p.p_wealth[: p.T]
    )

    assert np.allclose(tau_w, expected)


b1 = np.array([0.2, 0.6, 0.8])
p1 = Specifications()
new_param_values = {
    "S": 3,
    "rho": rho_vec.tolist(),
    "lambdas": [1.0],
    "J": 1,
    "T": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, 1)),
    "eta": (np.ones((3, 1)) / (3 * 1)),
    "h_wealth": [3],
    "p_wealth": [4],
    "m_wealth": [5],
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
}
p1.update_specifications(new_param_values)
expected1 = np.array([0.81122449, 1.837370242, 2.173849525])
b2 = np.array([0.1, 0.5, 0.9])
p2 = Specifications()
new_param_values2 = {
    "S": 3,
    "rho": rho_vec.tolist(),
    "lambdas": [1.0],
    "J": 1,
    "T": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, 1)),
    "eta": (np.ones((3, 1)) / (3 * 1)),
    "h_wealth": [1.2, 1.1, 2.3],
    "p_wealth": [2.2, 2.3, 1.8],
    "m_wealth": [3, 4, 3],
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
}
p2.update_specifications(new_param_values2)
expected2 = np.array([0.165976331, 0.522436904, 1.169769966])

test_data = [(b1, p1, expected1), (b2, p2, expected2)]


@pytest.mark.parametrize(
    "b,p,expected", test_data, ids=["constant params", "vary params"]
)
def test_MTR_wealth(b, p, expected):
    # Test marginal tax rate on wealth
    tau_w_prime = tax.MTR_wealth(
        b, p.h_wealth[: p.T], p.m_wealth[: p.T], p.p_wealth[: p.T]
    )

    assert np.allclose(tau_w_prime, expected)


p1 = Specifications()
p1.S = 2
p1.J = 1
p1.labor_income_tax_noncompliance_rate = np.zeros((p1.T, p1.S, p1.J))
p1.capital_income_tax_noncompliance_rate = np.zeros((p1.T, p1.S, p1.J))
p1.e = np.array([0.5, 0.45])
p1.tax_func_type = "DEP"
etr_params1 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
        ]
    ),
    (1, p1.S, 12),
)

p2 = Specifications()
p2.S = 2
p2.J = 1
p2.labor_income_tax_noncompliance_rate = np.zeros((p2.T, p2.S, p2.J))
p2.capital_income_tax_noncompliance_rate = np.zeros((p2.T, p2.S, p2.J))
p2.e = np.array([0.5, 0.45])
p2.tax_func_type = "GS"
etr_params2 = np.reshape(
    np.array(
        [
            [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
    (1, p2.S, 12),
)

p3 = Specifications()
p3.S = 2
p3.J = 1
p3.labor_income_tax_noncompliance_rate = np.zeros((p3.T, p3.S, p3.J))
p3.capital_income_tax_noncompliance_rate = np.zeros((p3.T, p3.S, p3.J))
p3.e = np.array([0.5, 0.45])
p3.tax_func_type = "DEP_totalinc"
etr_params3 = np.reshape(
    np.array(
        [
            [0.001, 0.002, 0.8, -0.14, 0.15, -0.15],
            [0.001, 0.002, 0.8, -0.14, 0.15, -0.15],
        ]
    ),
    (1, p3.S, 6),
)

p4 = Specifications()
p4.S = 3
p4.J = 1
p4.labor_income_tax_noncompliance_rate = np.zeros((p4.T, p4.S, p4.J))
p4.capital_income_tax_noncompliance_rate = np.zeros((p4.T, p4.S, p4.J))
p4.e = np.array([0.5, 0.45, 0.3])
p4.tax_func_type = "DEP"
etr_params4 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.002,
                0.001,
                0.002,
                0.04,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.011,
                0.001,
                0.003,
                0.06,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
        ]
    ),
    (1, p4.S, 12),
)
p5 = copy.deepcopy(p1)
p5.labor_income_tax_noncompliance_rate = np.ones((p5.T, p5.S, p5.J)) * 0.05
p5.capital_income_tax_noncompliance_rate = np.ones((p5.T, p5.S, p5.J)) * 0.05


@pytest.mark.parametrize(
    "b,n,etr_params,params,expected",
    [
        (
            np.array([0.4, 0.4]),
            np.array([0.5, 0.4]),
            etr_params1,
            p1,
            np.array([0.80167091, 0.80167011]),
        ),
        (
            np.array([0.4, 0.4]),
            np.array([0.5, 0.4]),
            etr_params2,
            p2,
            np.array([0.395985449, 0.395980186]),
        ),
        (
            np.array([0.4, 0.4]),
            np.array([0.5, 0.4]),
            etr_params3,
            p3,
            np.array([0.799999059, 0.799998254]),
        ),
        (
            np.reshape(np.array([0.4, 0.3, 0.5]), (3)),
            np.reshape(np.array([0.8, 0.4, 0.7]), (3)),
            etr_params4,
            p4,
            np.array([0.80167144, 0.80163711, 0.8016793]),
        ),
        (
            np.array([0.4, 0.4]),
            np.array([0.5, 0.4]),
            etr_params1,
            p5,
            np.array([0.80167091 * 0.95, 0.80167011 * 0.95]),
        ),
    ],
    ids=["DEP", "GS", "DEP_totalinc", "DEP, >1 dim", "GS, noncomply"],
)
def test_ETR_income(b, n, etr_params, params, expected):
    # Test income tax function
    r = 0.04
    w = 1.2
    factor = 100000
    test_ETR_income = tax.ETR_income(
        r,
        w,
        b,
        n,
        factor,
        params.e,
        etr_params,
        params.labor_income_tax_noncompliance_rate,
        params.capital_income_tax_noncompliance_rate,
        params,
    )
    assert np.allclose(test_ETR_income, expected)


p1 = Specifications()
p1.e = np.array([0.5, 0.45, 0.3])
p1.S = 3
p1.J = 1
p1.labor_income_tax_noncompliance_rate = np.zeros((p1.T, p1.S, p1.J))
p1.capital_income_tax_noncompliance_rate = np.zeros((p1.T, p1.S, p1.J))
p1.tax_func_type = "DEP"
p1.analytical_mtrs = True
etr_params1 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.002,
                0.001,
                0.002,
                0.04,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.011,
                0.001,
                0.003,
                0.06,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
        ]
    ),
    (1, p1.S, 12),
)
mtrx_params1 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.68,
                0.8,
                0.96,
                -0.17,
                -0.42,
                0.18,
                0.43,
                -0.42,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.65,
                0.8,
                0.90,
                -0.17,
                -0.42,
                0.18,
                0.33,
                -0.12,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.56,
                0.8,
                0.65,
                -0.17,
                -0.42,
                0.18,
                0.38,
                -0.22,
            ],
        ]
    ),
    (1, p1.S, 12),
)
p2 = Specifications()
p2.e = np.array([0.5, 0.45, 0.3])
p2.S = 3
p2.J = 1
p2.labor_income_tax_noncompliance_rate = np.zeros((p2.T, p2.S, p2.J))
p2.capital_income_tax_noncompliance_rate = np.zeros((p2.T, p2.S, p2.J))
p2.tax_func_type = "DEP"
p2.analytical_mtrs = True
etr_params2 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.002,
                0.001,
                0.002,
                0.04,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.011,
                0.001,
                0.003,
                0.06,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
        ]
    ),
    (1, p2.S, 12),
)
mtry_params2 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.68,
                0.8,
                0.96,
                -0.17,
                -0.42,
                0.18,
                0.43,
                -0.42,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.65,
                0.8,
                0.90,
                -0.17,
                -0.42,
                0.18,
                0.33,
                -0.12,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.56,
                0.8,
                0.65,
                -0.17,
                -0.42,
                0.18,
                0.38,
                -0.22,
            ],
        ]
    ),
    (1, p2.S, 12),
)
p3 = Specifications()
p3.e = np.array([0.5, 0.45, 0.3])
p3.S = 3
p3.J = 1
p3.labor_income_tax_noncompliance_rate = np.zeros((p3.T, p3.S, p3.J))
p3.capital_income_tax_noncompliance_rate = np.zeros((p3.T, p3.S, p3.J))
p3.tax_func_type = "DEP"
p3.analytical_mtrs = False
etr_params3 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.002,
                0.001,
                0.002,
                0.04,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.011,
                0.001,
                0.003,
                0.06,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
        ]
    ),
    (1, p3.S, 12),
)
mtrx_params3 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.68,
                0.8,
                0.96,
                -0.17,
                -0.42,
                0.18,
                0.43,
                -0.42,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.65,
                0.8,
                0.90,
                -0.17,
                -0.42,
                0.18,
                0.33,
                -0.12,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.56,
                0.8,
                0.65,
                -0.17,
                -0.42,
                0.18,
                0.38,
                -0.22,
            ],
        ]
    ),
    (1, p3.S, 12),
)

p4 = Specifications()
p4.e = np.array([0.5, 0.45, 0.3])
p4.S = 3
p4.J = 1
p4.labor_income_tax_noncompliance_rate = np.zeros((p4.T, p4.S, p4.J))
p4.capital_income_tax_noncompliance_rate = np.zeros((p4.T, p4.S, p4.J))
p4.tax_func_type = "GS"
p4.analytical_mtrs = False
etr_params4 = np.reshape(
    np.array(
        [
            [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.6, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
    (1, p4.S, 12),
)
mtrx_params4 = np.reshape(
    np.array(
        [
            [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.396, 0.7, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.6, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
    (1, p4.S, 12),
)

p5 = Specifications()
p5.e = np.array([0.5, 0.45, 0.3])
p5.S = 3
p5.J = 1
p5.labor_income_tax_noncompliance_rate = np.zeros((p5.T, p5.S, p5.J))
p5.capital_income_tax_noncompliance_rate = np.zeros((p5.T, p5.S, p5.J))
p5.tax_func_type = "DEP_totalinc"
p5.analytical_mtrs = True
etr_params5 = np.reshape(
    np.array(
        [
            [0.001, 0.002, 0.8, -0.14, 0.15, -0.15],
            [0.002, 0.001, 0.8, -0.14, 0.15, -0.15],
            [0.011, 0.001, 0.8, -0.14, 0.15, -0.15],
        ]
    ),
    (1, p5.S, 6),
)
mtrx_params5 = np.reshape(
    np.array(
        [
            [0.001, 0.002, 0.68, 0.18, -0.42, 0.96],
            [0.001, 0.002, 0.65, -0.17, 0.18, -0.12],
            [0.001, 0.002, 0.56, -0.17, 0.18, -0.22],
        ]
    ),
    (1, p5.S, 6),
)

p6 = Specifications()
p6.e = np.array([0.5, 0.45, 0.3])
p6.S = 3
p6.J = 1
p6.labor_income_tax_noncompliance_rate = np.zeros((p6.T, p6.S, p6.J))
p6.capital_income_tax_noncompliance_rate = np.zeros((p6.T, p6.S, p6.J))
p6.tax_func_type = "DEP_totalinc"
p6.analytical_mtrs = False
etr_params6 = np.reshape(
    np.array(
        [
            [0.001, 0.002, 0.8, -0.14, 0.15, -0.15],
            [0.002, 0.001, 0.8, -0.14, 0.15, -0.15],
            [0.011, 0.001, 0.8, -0.14, 0.15, -0.15],
        ]
    ),
    (1, p6.S, 6),
)
mtrx_params6 = np.reshape(
    np.array(
        [
            [0.001, 0.002, 0.68, -0.17, 0.18, -0.42],
            [0.001, 0.002, 0.65, -0.17, 0.18, -0.12],
            [0.001, 0.002, 0.56, -0.17, 0.18, -0.22],
        ]
    ),
    (1, p6.S, 6),
)
p7 = copy.deepcopy(p4)
p7.labor_income_tax_noncompliance_rate = np.ones((p7.T, p7.S, p7.J)) * 0.05
p7.capital_income_tax_noncompliance_rate = np.ones((p7.T, p7.S, p7.J)) * 0.05


@pytest.mark.parametrize(
    "etr_params,mtr_params,params,mtr_capital,expected",
    [
        (
            etr_params1,
            mtrx_params1,
            p1,
            False,
            np.array([0.801675428, 0.801647645, 0.801681744]),
        ),
        (
            etr_params2,
            mtry_params2,
            p2,
            True,
            np.array([0.8027427, 0.80335305, 0.80197745]),
        ),
        (
            etr_params3,
            mtrx_params3,
            p3,
            False,
            np.array([0.45239409, 0.73598958, 0.65126073]),
        ),
        (
            etr_params4,
            mtrx_params4,
            p4,
            False,
            np.array([0.395999995, 0.395999983, 0.599999478]),
        ),
        (
            etr_params5,
            mtrx_params5,
            p5,
            False,
            np.array([0.800001028, 0.800002432, 0.800000311]),
        ),
        (
            etr_params6,
            mtrx_params6,
            p6,
            False,
            np.array([0.439999714, 0.709998696, 0.519999185]),
        ),
        (
            etr_params4,
            mtrx_params4,
            p7,
            False,
            np.array(
                [0.395999995 * 0.95, 0.395999983 * 0.95, 0.599999478 * 0.95]
            ),
        ),
    ],
    ids=[
        "DEP, analytical mtr, labor income",
        "DEP, analytical mtr, capital income",
        "DEP, not analytical mtr",
        "GS",
        "DEP_totalinc, analytical mtr",
        "DEP_totalinc, not analytical mtr",
        "GS, noncomply",
    ],
)
def test_MTR_income(etr_params, mtr_params, params, mtr_capital, expected):
    # Test the MTR on income function
    r = 0.04
    w = 1.2
    b = np.array([0.4, 0.3, 0.5])
    n = np.array([0.8, 0.4, 0.7])
    factor = 110000
    if mtr_capital:
        noncompliance_rate = params.capital_income_tax_noncompliance_rate
    else:
        noncompliance_rate = params.labor_income_tax_noncompliance_rate

    test_mtr = tax.MTR_income(
        r,
        w,
        b,
        n,
        factor,
        mtr_capital,
        params.e,
        etr_params,
        mtr_params,
        noncompliance_rate,
        params,
    )
    assert np.allclose(test_mtr, expected)


p1 = Specifications()
new_param_values1 = {
    "cit_rate": [[0.20]],
    "delta_tau_annual": [[0.0023176377601205056]],
    "inv_tax_credit": [[0.02]],
    "T": 3,
    "S": 3,
    "chi_n": np.ones(3),
    "e": np.ones((3, p1.J)),
    "rho": rho_vec.tolist(),
    "eta": (np.ones((3, p1.J)) / (3 * p1.J)),
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
}
# update parameters instance with new values for test
p1.update_specifications(new_param_values1)
w = np.array([1.2, 1.1, 1.2])
Y = np.array([[3.0], [7.0], [3.0]])
p_m = np.ones((p1.T, 1))
L = np.array([[2.0], [3.0], [2.0]])
K = np.array([[5.0], [6.0], [5.0]])
inv_tax_credit_amounts = (
    np.array([[p1.delta * 5.0], [p1.delta * 6.0], [p1.delta * 5.0]]) * 0.02
)
expected1 = np.array([[0.0102], [0.11356], [0.0102]]) - inv_tax_credit_amounts


@pytest.mark.parametrize(
    "w,Y,L,K,p_m,p,m,method,expected",
    [
        (w.reshape(3, 1), Y, L, K, p_m, p1, None, "TPI", expected1),
        (w, Y[:, 0], L[:, 0], K[:, 0], p_m, p1, 0, "TPI", expected1[:, 0]),
        (
            w[-1],
            Y[-1, :],
            L[-1, :],
            K[-1, :],
            p_m[-1, :],
            p1,
            None,
            "SS",
            expected1[-1, :],
        ),
        (
            w[-1],
            Y[-1, 0],
            L[-1, 0],
            K[-1, 0].reshape(1, 1),
            p_m[-1, :],
            p1,
            0,
            "SS",
            expected1[-1, 0],
        ),
    ],
    ids=[
        "TPI, m is None",
        "TPI, m is not None",
        "SS, m is None",
        "SS, m is not None",
    ],
)
def test_get_biz_tax(w, Y, L, K, p_m, p, m, method, expected):
    # Test function for business tax receipts
    biz_tax = tax.get_biz_tax(w, Y, L, K, p_m, p, m, method)
    assert np.allclose(biz_tax, expected)


"""
-------------------------------------------------------------------------------
Test tax.py net_taxes() function
-------------------------------------------------------------------------------
"""
# Set parameter class for each case
p = Specifications()
p.tax_func_type = "DEP"
p.J = 1
p.S = 3
p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.S, p.J))
p.capital_income_tax_noncompliance_rate = np.zeros((p.T, p.S, p.J))
p.lambdas = np.array([1.0])
p.e = np.array([0.5, 0.45, 0.3]).reshape(3, 1)
p.h_wealth = np.ones(p.T + p.S) * 1
p.p_wealth = np.ones(p.T + p.S) * 2
p.m_wealth = np.ones(p.T + p.S) * 3
p.tau_payroll = np.ones(p.T + p.S) * 0.15
p.tau_bq = np.ones(p.T + p.S) * 0.1
p.retire = (np.ones(p.T + p.S) * 2).astype(int)
p1 = copy.deepcopy(p)
p2 = copy.deepcopy(p)
p3 = copy.deepcopy(p)
p3.T = 3
p3.labor_income_tax_noncompliance_rate = np.zeros((p3.T, p3.S, p3.J))
p3.capital_income_tax_noncompliance_rate = np.zeros((p3.T, p3.S, p3.J))
p4 = copy.deepcopy(p)
p5 = copy.deepcopy(p)
p5.e = np.array([[0.3, 0.2], [0.5, 0.4], [0.45, 0.3]])
p5.J = 2
p5.T = 3
p5.labor_income_tax_noncompliance_rate = np.zeros((p5.T, p5.S, p5.J))
p5.capital_income_tax_noncompliance_rate = np.zeros((p5.T, p5.S, p5.J))
p5.lambdas = np.array([0.65, 0.35])
# set variables and other parameters for each case
r1 = 0.04
w1 = 1.2
b1 = np.array([0.4, 0.3, 0.5])
n1 = np.array([0.8, 0.4, 0.7])
BQ1 = np.array([0.3])
bq1 = BQ1 / p1.lambdas[0]
tr1 = np.array([0.12])
theta1 = np.array([0.225])
etr_params1 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
        ]
    ),
    (1, p1.S, 12),
)
j1 = 0
shift1 = True
method1 = "SS"

r2 = r1
w2 = w1
b2 = b1
n2 = n1
BQ2 = BQ1
bq2 = bq1
tr2 = tr1
theta2 = theta1
etr_params2 = etr_params1
j2 = 0
shift2 = True
method2 = "TPI_scalar"

r3 = np.array([0.04, 0.045, 0.04])
w3 = np.array([1.2, 1.3, 1.1])
b3 = np.tile(
    np.reshape(np.array([0.4, 0.3, 0.5]), (1, p3.S, 1)), (p3.T, 1, p3.J)
)
n3 = np.tile(
    np.reshape(np.array([0.8, 0.4, 0.7]), (1, p3.S, 1)), (p3.T, 1, p3.J)
)
BQ3 = np.array([0.3, 0.4, 0.45])
bq3 = np.tile(np.reshape(BQ3 / p3.lambdas[0], (p3.T, 1)), (1, p3.S))
tr3 = np.tile(np.reshape(np.array([0.12, 0.1, 0.11]), (p3.T, 1)), (1, p3.S))
theta3 = theta1
etr_params3 = np.tile(
    np.reshape(
        np.array(
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ]
        ),
        (1, 1, 12),
    ),
    (p3.T, p3.S, 1),
)
j3 = 0
shift3 = True
method3 = "TPI"

r4 = r3
w4 = w3
b4 = b3
n4 = n3
BQ4 = BQ3
bq4 = bq3
tr4 = tr3
theta4 = theta1
etr_params4 = etr_params3
j4 = 0
shift4 = False
method4 = "TPI"

r5 = r3
w5 = w3
b5 = np.array(
    [
        [[0.2, 0.3], [0.3, 0.35], [0.4, 0.35]],
        [[0.4, 0.3], [0.4, 0.35], [0.5, 0.35]],
        [[0.6, 0.4], [0.3, 0.4], [0.4, 0.22]],
    ]
)
n5 = np.array(
    [
        [[0.6, 0.5], [0.5, 0.55], [0.7, 0.8]],
        [[0.4, 0.43], [0.5, 0.66], [0.7, 0.7]],
        [[0.46, 0.44], [0.63, 0.64], [0.74, 0.72]],
    ]
)
BQ5 = np.tile(
    np.reshape(
        np.array([[0.3, 0.35], [0.25, 0.3], [0.4, 0.45]]), (p5.T, 1, p5.J)
    ),
    (1, p5.S, 1),
)
bq5 = BQ5 / p5.lambdas.reshape(1, 1, p5.J)
tr5 = np.tile(
    np.reshape(np.array([0.12, 0.1, 0.11]), (p5.T, 1, 1)), (1, p5.S, p5.J)
)
theta5 = np.array([0.225, 0.3])
etr_params = np.tile(
    np.reshape(
        np.array(
            [
                [
                    0.001,
                    0.002,
                    0.003,
                    0.0015,
                    0.8,
                    0.8,
                    0.83,
                    -0.14,
                    -0.15,
                    0.15,
                    0.16,
                    -0.15,
                ],
                [
                    0.001,
                    0.002,
                    0.003,
                    0.0015,
                    0.8,
                    0.8,
                    0.83,
                    -0.14,
                    -0.15,
                    0.15,
                    0.16,
                    -0.15,
                ],
                [
                    0.001,
                    0.002,
                    0.003,
                    0.0015,
                    0.8,
                    0.8,
                    0.83,
                    -0.14,
                    -0.15,
                    0.15,
                    0.16,
                    -0.15,
                ],
            ]
        ),
        (1, p5.S, 12),
    ),
    (p5.T, 1, 1),
)
etr_params5 = np.tile(
    np.reshape(etr_params, (p5.T, p5.S, 1, 12)), (1, 1, p5.J, 1)
)

j5 = None
shift5 = False
method5 = "TPI"

p6 = copy.deepcopy(p5)
p6.tau_bq = np.array([0.05, 0.2, 0.0])

p7 = copy.deepcopy(p5)
p7.tau_bq = np.array([0.05, 0.2, 0.0])
p7.retire = (np.array([1, 2, 2])).astype(int)

p8 = copy.deepcopy(p6)
p8.replacement_rate_adjust = [1.5, 0.6, 1.0]

factor = 105000
ubi1 = np.zeros((p1.T, p1.S, p1.J))
ubi2 = np.zeros((p2.T, p2.S, p2.J))
ubi3 = np.zeros((p3.T, p3.S, p3.J))
ubi4 = np.zeros((p4.T, p4.S, p4.J))
ubi5 = np.zeros((p5.T, p5.S, p5.J))

p_u = Specifications()
new_param_values_ubi = {
    "T": 3,
    "S": 3,
    "J": 2,
    "chi_n": np.ones(3),
    "e": np.ones((3, 2)),
    "rho": rho_vec.tolist(),
    "lambdas": [0.65, 0.35],
    "eta": (np.ones((3, 2)) / (3 * 2)),
    "ubi_nom_017": 1000,
    "ubi_nom_1864": 1500,
    "ubi_nom_65p": 500,
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
}
p_u.update_specifications(new_param_values_ubi)
p_u.tax_func_type = "DEP"
p_u.e = np.array([[0.3, 0.2], [0.5, 0.4], [0.45, 0.3]])
p_u.h_wealth = np.ones(p_u.T + p_u.S) * 1
p_u.p_wealth = np.ones(p_u.T + p_u.S) * 2
p_u.m_wealth = np.ones(p_u.T + p_u.S) * 3
p_u.tau_payroll = np.ones(p_u.T + p_u.S) * 0.15
p_u.tau_bq = np.ones(p_u.T + p_u.S) * 0.1
p_u.retire = (np.ones(p_u.T + p_u.S) * 2).astype(int)

factor_u = 105000
ubi_u = p_u.ubi_nom_array / factor_u

# set variables and other parameters for each case
r9 = 0.04
w9 = 1.2
b9 = np.array([0.4, 0.3, 0.5])
n9 = np.array([0.8, 0.4, 0.7])
BQ9 = np.array([0.3])
bq9 = BQ9 / p_u.lambdas[0]
tr9 = np.array([0.12])
theta9 = np.array([0.225])
etr_params9 = np.reshape(
    np.array(
        [
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ],
        ]
    ),
    (1, p_u.S, 12),
)
j9 = 0
shift9 = True
method9 = "SS"

r10 = r9
w10 = w9
b10 = b9
n10 = n9
BQ10 = BQ9
bq10 = bq9
tr10 = tr9
theta10 = theta9
etr_params10 = etr_params9
j10 = 0
shift10 = True
method10 = "TPI_scalar"

r11 = np.array([0.04, 0.045, 0.04])
w11 = np.array([1.2, 1.3, 1.1])
b11 = np.tile(
    np.reshape(np.array([0.4, 0.3, 0.5]), (1, p_u.S, 1)), (p_u.T, 1, p_u.J)
)
n11 = np.tile(
    np.reshape(np.array([0.8, 0.4, 0.7]), (1, p_u.S, 1)), (p_u.T, 1, p_u.J)
)
BQ11 = np.array([0.3, 0.4, 0.45])
bq11 = np.tile(np.reshape(BQ11 / p_u.lambdas[0], (p_u.T, 1)), (1, p_u.S))
tr11 = np.tile(np.reshape(np.array([0.12, 0.1, 0.11]), (p_u.T, 1)), (1, p_u.S))
theta11 = theta9
etr_params11 = np.tile(
    np.reshape(
        np.array(
            [
                0.001,
                0.002,
                0.003,
                0.0015,
                0.8,
                0.8,
                0.83,
                -0.14,
                -0.15,
                0.15,
                0.16,
                -0.15,
            ]
        ),
        (1, 1, 12),
    ),
    (p_u.T, p_u.S, 1),
)
j11 = 0
shift11 = True
method11 = "TPI"


p12 = copy.deepcopy(p1)
p12.labor_income_tax_noncompliance_rate = (
    np.ones((p12.T + p12.S, p12.J)) * 0.05
)
p12.capital_income_tax_noncompliance_rate = (
    np.ones((p12.T + p12.S, p12.J)) * 0.05
)
p13 = copy.deepcopy(p5)
p13.labor_income_tax_noncompliance_rate = (
    np.ones((p13.T + p13.S, p13.J)) * 0.05
)
p13.capital_income_tax_noncompliance_rate = (
    np.ones((p13.T + p13.S, p13.J)) * 0.05
)

expected1 = np.array([0.47374766, -0.09027663, 0.03871394])
expected2 = np.array([0.20374766, -0.09027663, 0.03871394])
expected3 = np.array(
    [
        [0.473747659, -0.090276635, 0.038713941],
        [0.543420101, -0.064442513, 0.068204207],
        [0.460680696, -0.05990653, 0.066228621],
    ]
)
expected4 = np.array(
    [
        [0.473747659, 0.179723365, 0.038713941],
        [0.543420101, 0.228057487, 0.068204207],
        [0.460680696, 0.18759347, 0.066228621],
    ]
)
expected5 = np.array(
    [
        [
            [0.16311573, 0.1583638],
            [0.27581667, 0.31559773],
            [0.12283074, -0.02156221],
        ],
        [
            [0.1954706, 0.15747779],
            [0.3563044, 0.39808896],
            [0.19657058, -0.05871855],
        ],
        [
            [0.31524401, 0.21763702],
            [0.34545346, 0.39350691],
            [0.15958077, -0.0482051],
        ],
    ]
)
expected6 = np.array(
    [
        [
            [0.16311573 - 0.023076923, 0.1583638 - 0.05],
            [0.27581667 - 0.023076923, 0.31559773 - 0.05],
            [0.12283074 - 0.023076923, -0.02156221 - 0.05],
        ],
        [
            [0.1954706 + 0.038461538, 0.15747779 + 0.085714286],
            [0.3563044 + 0.038461538, 0.39808896 + 0.085714286],
            [0.19657058 + 0.038461538, -0.05871855 + 0.085714286],
        ],
        [
            [0.31524401 - 0.061538462, 0.21763702 - 0.12857143],
            [0.34545346 - 0.061538462, 0.39350691 - 0.12857143],
            [0.15958077 - 0.061538462, -0.0482051 - 0.12857143],
        ],
    ]
)
expected7 = np.array(
    [
        [
            [0.16311573 - 0.023076923, 0.1583638 - 0.05],
            [0.27581667 - 0.023076923 - 0.27, 0.31559773 - 0.05 - 0.36],
            [0.12283074 - 0.023076923, -0.02156221 - 0.05],
        ],
        [
            [0.1954706 + 0.038461538, 0.15747779 + 0.085714286],
            [0.3563044 + 0.038461538, 0.39808896 + 0.085714286],
            [0.19657058 + 0.038461538, -0.05871855 + 0.085714286],
        ],
        [
            [0.31524401 - 0.061538462, 0.21763702 - 0.12857143],
            [0.34545346 - 0.061538462, 0.39350691 - 0.12857143],
            [0.15958077 - 0.061538462, -0.0482051 - 0.12857143],
        ],
    ]
)
expected8 = np.array(
    [
        [
            [0.16311573 - 0.023076923, 0.1583638 - 0.05],
            [0.27581667 - 0.023076923, 0.31559773 - 0.05],
            [0.12283074 - 0.023076923 - 0.135, -0.02156221 - 0.05 - 0.18],
        ],
        [
            [0.1954706 + 0.038461538, 0.15747779 + 0.085714286],
            [0.3563044 + 0.038461538, 0.39808896 + 0.085714286],
            [
                0.19657058 + 0.038461538 + 0.117,
                -0.05871855 + 0.085714286 + 0.156,
            ],
        ],
        [
            [0.31524401 - 0.061538462, 0.21763702 - 0.12857143],
            [0.34545346 - 0.061538462, 0.39350691 - 0.12857143],
            [0.15958077 - 0.061538462, -0.0482051 - 0.12857143],
        ],
    ]
)
expected9 = np.array([0.28384671, -0.07461627, 0.15144631])
expected10 = np.array([0.01384671, -0.07461627, 0.15144631])
expected11 = np.array(
    [
        [0.30869878, -0.0497642, 0.17629838],
        [0.39311429, 0.00794409, 0.24575229],
        [0.35257605, 0.02042006, 0.23553789],
    ]
)
expected12 = np.array([0.453866159, -0.09027663 - 0.009138896, 0.027811102])
expected13 = np.array(
    [
        [
            [0.15413763, 0.15307288],
            [0.2633108, 0.30445456],
            [0.10703778, -0.03366739],
        ],
        [
            [0.18849603, 0.15245539],
            [0.34255564, 0.38370095],
            [0.17925424, -0.07029269],
        ],
        [
            [0.30819723, 0.21311562],
            [0.33108374, 0.38157802],
            [0.14425679, -0.05808117],
        ],
    ]
)

test_data = [
    (
        r1,
        w1,
        b1,
        n1,
        bq1,
        factor,
        tr1,
        ubi1[0, :, :],
        theta1,
        None,
        j1,
        shift1,
        method1,
        p1.e[:, j1],
        etr_params1[-1, :, :],
        p1,
        expected1,
    ),
    (
        r2,
        w2,
        b2,
        n2,
        bq2,
        factor,
        tr2,
        ubi2[0, :, :],
        theta2,
        None,
        j2,
        shift2,
        method2,
        p2.e[:, j2],
        etr_params2,
        p2,
        expected2,
    ),
    (
        r3,
        w3,
        b3[:, :, j3],
        n3[:, :, j3],
        bq3,
        factor,
        tr3,
        ubi3,
        theta3,
        0,
        j3,
        shift3,
        method3,
        p3.e[:, j3],
        etr_params3,
        p3,
        expected3,
    ),
    (
        r4,
        w4,
        b4[:, :, j4],
        n4[:, :, j4],
        bq4,
        factor,
        tr4,
        ubi4,
        theta4,
        0,
        j4,
        shift4,
        method4,
        p4.e[:, j4],
        etr_params4,
        p4,
        expected4,
    ),
    (
        r5,
        w5,
        b5,
        n5,
        bq5,
        factor,
        tr5,
        ubi5,
        theta5,
        0,
        j5,
        shift5,
        method5,
        p5.e,
        etr_params5,
        p5,
        expected5,
    ),
    (
        r5,
        w5,
        b5,
        n5,
        bq5,
        factor,
        tr5,
        ubi5,
        theta5,
        0,
        j5,
        shift5,
        method5,
        p5.e,
        etr_params5,
        p6,
        expected6,
    ),
    (
        r5,
        w5,
        b5,
        n5,
        bq5,
        factor,
        tr5,
        ubi5,
        theta5,
        0,
        j5,
        shift5,
        method5,
        p5.e,
        etr_params5,
        p7,
        expected7,
    ),
    (
        r5,
        w5,
        b5,
        n5,
        bq5,
        factor,
        tr5,
        ubi5,
        theta5,
        0,
        j5,
        shift5,
        method5,
        p5.e,
        etr_params5,
        p8,
        expected8,
    ),
    (
        r9,
        w9,
        b9,
        n9,
        bq9,
        factor_u,
        tr9,
        ubi_u[0, :, j9],
        theta9,
        None,
        j9,
        shift9,
        method9,
        p_u.e[:, j9],
        etr_params9[-1, :, :],
        p_u,
        expected9,
    ),
    (
        r10,
        w10,
        b10,
        n10,
        bq10,
        factor_u,
        tr10,
        ubi_u[0, :, j10],
        theta10,
        None,
        j10,
        shift10,
        method10,
        p_u.e[:, j10],
        etr_params10,
        p_u,
        expected10,
    ),
    (
        r11,
        w11,
        b11[:, :, j11],
        n11[:, :, j11],
        bq11,
        factor_u,
        tr11,
        ubi_u[: p_u.T, :, j11],
        theta11,
        0,
        j11,
        shift11,
        method11,
        p_u.e[:, j11],
        etr_params11,
        p_u,
        expected11,
    ),
    (
        r1,
        w1,
        b1,
        n1,
        bq1,
        factor,
        tr1,
        ubi1[0, :, :],
        theta1,
        None,
        j1,
        shift1,
        method1,
        p1.e[:, j1],
        etr_params1[-1, :, :],
        p12,
        expected12,
    ),
    (
        r5,
        w5,
        b5,
        n5,
        bq5,
        factor,
        tr5,
        ubi5,
        theta5,
        0,
        j5,
        shift5,
        method5,
        p5.e,
        etr_params5,
        p13,
        expected13,
    ),
]


@pytest.mark.parametrize(
    "r,w,b,n,bq,factor,tr,ubi,theta,t,j,shift,method,"
    + "e,etr_params,p,expected",
    test_data,
    ids=[
        "SS",
        "TPI Scalar",
        "TPI shift = True",
        "TPI shift = False",
        "TPI 3D",
        "TPI 3D,vary tau_bq",
        "TPI 3D,vary retire",
        "TPI 3D,vary replacement rate",
        "SS UBI>0",
        "TPI scalar UBI>0",
        "TPI UBI>0",
        "SS, noncomply",
        "TPI 3D. noncomply",
    ],
)
def test_net_taxes(
    r,
    w,
    b,
    n,
    bq,
    factor,
    tr,
    ubi,
    theta,
    t,
    j,
    shift,
    method,
    e,
    etr_params,
    p,
    expected,
):
    # Test function that computes total net taxes for the household
    net_taxes = tax.net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        theta,
        t,
        j,
        shift,
        method,
        e,
        etr_params,
        p,
    )

    assert np.allclose(net_taxes, expected)
