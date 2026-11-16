import numpy as np
import copy
import pytest
from ogcore import pensions, tax
from ogcore.parameters import Specifications

p = Specifications()
rho_vec = np.zeros((1, 4, 1))
rho_vec[0, -1, :] = 1.0
new_param_values = {
    "S": 4,
    "rho": rho_vec.tolist(),
    "rho_preTP": rho_vec[0, :, :].tolist(),
    "lambdas": [1.0],
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
    "income_tax_filer": [[1]],
    "wealth_tax_filer": [[1]],
    "replacement_rate_adjust": [[1.0]],
    "J": 1,
    "T": 4,
    "chi_n": np.ones(4),
    "eta": (np.ones((4, 1)) / (4 * 1)),
    "e": np.ones((4, 1)),
    "omega": (np.ones((1, 4, 1)) / 4).tolist(),
    "omega_SS": (np.ones((4, 1)) / 4).tolist(),
    "omega_S_preTP": (np.ones((4, 1)) / 4).tolist(),
    "imm_rates": np.zeros((1, 4, 1)).tolist(),
    "imm_rates_preTP": np.zeros((4, 1)).tolist(),
}
p.update_specifications(new_param_values)
p.retire = np.full((p.T + p.S, p.S), 3)
p1 = Specifications()
p1.update_specifications(
    {
        "S": 4,
        "rho": np.tile(rho_vec, (1, 1, 2)).tolist(),
        "rho_preTP": np.tile(rho_vec[0, :, :], (1, 2)).tolist(),
        "lambdas": [0.5, 0.5],
        "labor_income_tax_noncompliance_rate": [[0.0]],
        "capital_income_tax_noncompliance_rate": [[0.0]],
        "income_tax_filer": [[1]],
        "wealth_tax_filer": [[1]],
        "replacement_rate_adjust": [[1.0]],
        "J": 2,
        "T": 4,
        "chi_n": np.ones(4),
        "eta": (np.ones((4, 2)) / (4 * 2)),
        "e": np.ones((4, 2)),
        "omega": (np.ones((1, 4, 2)) / (4 * 2)).tolist(),
        "omega_SS": (np.ones((4, 2)) / (4 * 2)).tolist(),
        "omega_S_preTP": (np.ones((4, 2)) / (4 * 2)).tolist(),
        "imm_rates": np.zeros((1, 4, 2)).tolist(),
        "imm_rates_preTP": np.zeros((4, 2)).tolist(),
    }
)
p1.retire = np.full((p1.T + p1.S, p1.S), 3)
p2 = copy.deepcopy(p1)
p3 = copy.deepcopy(p1)
# Use just a column of e
p1.e = np.transpose(np.array([[0.1, 0.3, 0.5, 0.2], [0.1, 0.3, 0.5, 0.2]]))
# e has two dimensions
p2.e = np.array([[0.4, 0.3], [0.5, 0.4], [0.6, 0.4], [0.4, 0.3]])
p3.e = np.array([[0.35, 0.3], [0.55, 0.4], [0.65, 0.4], [0.45, 0.3]])
p5 = copy.deepcopy(p3)
p5.PIA_minpayment = 125.0
wss = 0.5
n1 = np.array([0.5, 0.5, 0.5, 0.5])
n2 = nssmat = np.array([[0.4, 0.4], [0.4, 0.4], [0.4, 0.4], [0.4, 0.4]])
n3 = nssmat = np.array([[0.3, 0.35], [0.3, 0.35], [0.3, 0.35], [0.3, 0.35]])
factor1 = 100000
factor3 = 10000
factor4 = 1000
expected1 = np.array([0.042012])
expected2 = np.array([0.042012, 0.03842772])
expected3 = np.array([0.1145304, 0.0969304])
expected4 = np.array([0.1755, 0.126])
expected5 = np.array([0.1755, 0.126 * 1.1904761904761905])
p6 = copy.deepcopy(p5)
p6.baseline = False
p6.baseline_theta = True
p6.SS_theta = expected5
p7 = copy.deepcopy(p1)
p7.baseline = False
p7.baseline_theta = True
p7.SS_theta = expected1

test_data = [
    (n1, wss, factor1, 0, p1, expected1),
    (n2, wss, factor1, None, p2, expected2),
    (n3, wss, factor3, None, p3, expected3),
    (n3, wss, factor4, None, p3, expected4),
    (n3, wss, factor4, None, p5, expected5),
    (n3, wss, factor4, None, p6, expected5),
    (n1, wss, factor1, 0, p7, expected1),
]


@pytest.mark.parametrize(
    "n,w,factor,j,p_in,expected",
    test_data,
    ids=[
        "1D e",
        "2D e",
        "AIME case 2",
        "AIME case 3",
        "Min PIA case",
        "use_baseline_theta",
        "use_baseline_theta, j not None",
    ],
)
def test_replacement_rate_vals(n, w, factor, j, p_in, expected):
    # Test replacement rate function, making sure to trigger all three
    # cases of AIME
    # make e 3D
    p = copy.deepcopy(p_in)
    # p.e = np.tile(np.reshape(p.e, (1, p.S, p.J)), (p.T, 1, 1))
    p.e = np.tile(
        np.reshape(p.e, (1, p.e.shape[0], p.e.shape[1])), (p.T, 1, 1)
    )
    theta = pensions.replacement_rate_vals(n, w, factor, j, p)
    assert np.allclose(theta, expected)


p = Specifications()
p.S = 7
p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.2
p.retire = 4
p.avg_earn_num_years = 50
p.yr_contrib = 55
p.g_y = np.ones(p.T) * 0.03
j = 1
w = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
equiv_periods = int(round((p.S / 80.0) * p.avg_earn_num_years)) - 1
equiv_contrib_periods = int(round((p.S / 80.0) * p.yr_contrib)) - 1
DB = np.zeros(p.S)
DB_loop_expected1 = np.array(
    [0, 0, 0, 0, 0.337864778, 0.327879365, 0.318189065]
)

# the loop takes the household's own retirement period, averaging
# window, replacement rate, and credited periods at each age u
args1 = (
    w,
    e,
    n,
    np.full(p.S, p.retire),
    p.S,
    p.g_y,
    DB,
    np.full(p.S, equiv_periods),
    np.full(p.S, 0.2),  # rate in effect at each age u
    np.full(p.S, equiv_contrib_periods),
)

test_data = [(args1, DB_loop_expected1)]


@pytest.mark.parametrize(
    "args,DB_loop_expected", test_data, ids=["SS/Complete"]
)
def test_DB_1dim_loop(args, DB_loop_expected):
    """
    Test of the pensions.DB_1dim_loop() function.
    """

    DB_loop = pensions.DB_1dim_loop(*args)
    assert np.allclose(DB_loop, DB_loop_expected)


p = Specifications()
p.S = 7
p.retire = 4
per_rmn = p.S
p.avg_earn_num_years = 50
p.yr_contrib = 55
p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.2
p.g_y = np.ones(p.T) * 0.03
w = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
deriv_DB_loop_expected = np.array(
    [0.352, 0.3256, 0.2904, 0.232, 0.0, 0.0, 0.0]
)
d_theta_empty = np.zeros_like(w)
equiv_periods = int(round((p.S / 80.0) * p.avg_earn_num_years)) - 1
equiv_contrib_periods = int(round((p.S / 80.0) * p.yr_contrib)) - 1
args3 = (
    w,
    e,
    p.S,
    p.retire,
    per_rmn,
    equiv_periods,
    np.full(per_rmn, 0.2),  # rate in effect at each remaining period
    equiv_contrib_periods,
)

test_data = [(args3, deriv_DB_loop_expected)]


@pytest.mark.parametrize("args,deriv_DB_loop_expected", test_data)
def test_deriv_DB_loop(args, deriv_DB_loop_expected):
    """
    Test of the pensions.deriv_DB_loop() function.
    """
    w, e, S, retire, per_rmn, last_career_yrs, rep_rate_py, yr_contrib = args
    deriv_DB_loop = pensions.deriv_DB_loop(
        w, e, S, retire, per_rmn, last_career_yrs, rep_rate_py, yr_contrib
    )

    assert np.allclose(deriv_DB_loop, deriv_DB_loop_expected)


p = Specifications()
p.S = 7
p.retire = 4
p.vpoint = 0.4
w = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
p.g_y = np.ones(p.T) * 0.03
factor = 2
d_theta_empty = np.zeros_like(w)
deriv_PS_loop_expected1 = np.array(
    [0.003168, 0.0029304, 0.0026136, 0.002088, 0, 0, 0]
)
args3 = (w, e, p.S, p.retire, per_rmn, d_theta_empty, p.vpoint, factor)

test_data = [(args3, deriv_PS_loop_expected1)]


@pytest.mark.parametrize(
    "args,deriv_PS_loop_expected", test_data, ids=["SS/Complete"]
)
def test_deriv_PS_loop(args, deriv_PS_loop_expected):
    """
    Test of the pensions.deriv_PS_loop() function.
    """
    w, e, S, retire, per_rmn, d_theta_empty, vpoint, factor = args

    deriv_PS_loop = pensions.deriv_PS_loop(
        w, e, S, retire, per_rmn, d_theta_empty, vpoint, factor
    )

    assert np.allclose(deriv_PS_loop, deriv_PS_loop_expected)


#############non-zero d_theta: case 1############
p = Specifications()
p.S = 7
p.retire = 4
p.avg_earn_num_years = 50
p.yr_contrib = 55
p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.2
p.g_y = np.ones(p.T) * 0.03
n_ddb1 = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
w_ddb1 = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e_ddb1 = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
per_rmn = n_ddb1.shape[0]
d_theta_empty = np.zeros_like(per_rmn)
deriv_DB_expected1 = np.array([0.33, 0.30525, 0.27225, 0.2175, 0.0, 0.0, 0.0])
args_ddb1 = (w_ddb1, e_ddb1, per_rmn, 0, 1, p)

#############non-zero d_theta: case 2############
p2 = Specifications()
p2.S = 7
p2.retire = 5
p2.last_career_yrs = 2
p2.yr_contrib = 55
p2.alpha_db = np.ones((p2.T + p2.S, p2.J)) * 0.2
p2.g_y = 0.03
n_ddb2 = np.array([0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
w_ddb1 = np.array([1.1, 1.21, 1, 1.01, 0.99, 0.8])
e_ddb1 = np.array([1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
per_rmn = n_ddb2.shape[0]
d_theta_empty = np.zeros_like(per_rmn)
# p2 keeps the default avg_earn_num_years=35 (3 periods), 5 periods credited
deriv_DB_expected2 = np.array([0.407, 0.363, 0.29, 0.2929, 0.0, 0.0])
args_ddb2 = (w_ddb1, e_ddb1, per_rmn, 0, 1, p2)

# Rate that varies over time: the derivative for a household of age s
# at time t is evaluated at the rate in effect when it retires, period
# t + R - s.  With S=7, R=4 and the rate doubling from period 2 on, the
# household of age 0 at t=0 retires in period 4 (0.4), age 1 in period 3
# (0.4), age 2 in period 2 (0.4), age 3 in period 1 (0.2).
p3 = copy.deepcopy(p)
p3.alpha_db = np.ones((p3.T + p3.S, p3.J)) * 0.2
p3.alpha_db[2:, :] = 0.4
deriv_DB_expected3 = deriv_DB_expected1 * np.array(
    [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
)
args_ddb3 = (
    np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8]),
    np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6]),
    7,
    0,
    1,
    p3,
)

test_data = [
    (args_ddb1, deriv_DB_expected1),
    (args_ddb2, deriv_DB_expected2),
    (args_ddb3, deriv_DB_expected3),
]


@pytest.mark.parametrize(
    "args,deriv_DB_expected",
    test_data,
    ids=[
        "non-zero d_theta: case 1",
        "non-zero d_theta: case 2",
        "time-varying alpha_db",
    ],
)
def test_deriv_DB(args, deriv_DB_expected):
    """
    Test of the pensions.deriv_DB() function.
    """
    w, e, per_rmn, t, j, p = args
    deriv_DB = pensions.deriv_DB(w, e, per_rmn, t, j, p)

    assert np.allclose(deriv_DB, deriv_DB_expected)


#############PS deriv SS or complete lifetimes############
p = Specifications()
p.S = 7
p.retire = 4
p.vpoint = 0.4
omegas = 1 / (p.S) * np.ones(p.S)
p.omega_SS = omegas
p.g_y = np.ones(p.T) * 0.03
per_rmn_dps1 = p.S
factor = 2
w = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
deriv_PS_expected1 = np.array(
    [0.003168, 0.0029304, 0.0026136, 0.002088, 0, 0, 0]
)
args_dps1 = (w, e, per_rmn_dps1, factor, p)

##############PS deriv incomplete lifetimes############
p2 = Specifications()
p2.S = 7
p2.retire = 4
p2.vpoint = 0.4
omegas = 1 / (p2.S) * np.ones(p2.S)
p2.omega_SS = omegas
p2.g_y = 0.03
per_rmn_dps2 = 5
factor = 2
w = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
deriv_PS_expected2 = np.array([0.0026136, 0.002088, 0, 0, 0])
args_dps2 = (w, e, per_rmn_dps2, factor, p2)
test_data = [(args_dps1, deriv_PS_expected1), (args_dps2, deriv_PS_expected2)]


@pytest.mark.parametrize(
    "args,deriv_PS_expected", test_data, ids=["SS/Complete", "Incomplete"]
)
def test_deriv_S(args, deriv_PS_expected):
    """
    Test of the pensions.deriv_PS() function.
    """
    w, e, per_rmn, factor, p = args

    deriv_PS = pensions.deriv_PS(w, e, per_rmn, factor, p)

    assert np.allclose(deriv_PS, deriv_PS_expected)


#############complete lifetimes, S = 4###################
p = Specifications()
p.S = 4
p.retire = 2
per_rmn = p.S
p.g_y = np.ones(p.T) * np.ones(p.T) * 0.03
p.g_n = np.ones(p.T) * 0.0
p.g_n_ss = 0.0
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
p.tau_p = 0.3
p.k_ret = 0.4615
p.mort_rates_SS = np.array([0.01, 0.05, 0.3, 1])
w = np.array([1.2, 1.1, 1.21, 1])
e = np.array([1.1, 1.11, 0.9, 0.87])
r = np.ones(p.T) * 0.02
# annuity factor 1 + 0.7 / 1.02 (survival timing fixed in Issue #1268)
d_NDC_expected1 = np.array([0.33317153, 0.29907546, 0.0, 0.0])
args1 = (r, w, e, None, per_rmn, p)

#############Incomplete lifetimes###################
p2 = Specifications()
p2.S = 4
p2.retire = 2
per_rmn2 = 3
p2.g_y = np.ones(p2.T) * 0.04
p2.g_n = np.ones(p2.T) * 0.0
p2.g_n_ss = 0.0
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "LR GDP"
p2.tau_p = 0.3
p2.k_ret = 0.4615
p2.mort_rates_SS = np.array([0.1, 0.2, 0.4, 0.6, 1.0])
w2 = np.array([1.1, 1.21, 1.25])
e2 = np.array([1.11, 0.9, 1.0])
r2 = np.ones(p2.T) * 0.04
d_NDC_expected2 = np.array([0.32852794, 0.0, 0.0])
args2 = (r2, w2, e2, None, per_rmn2, p2)

test_data = [(args1, d_NDC_expected1), (args2, d_NDC_expected2)]


@pytest.mark.parametrize(
    "args,d_NDC_expected", test_data, ids=["SS/Complete", "Incomplete"]
)
def test_deriv_NDC(args, d_NDC_expected):
    """
    Test of the pensions.deriv_NDC() function.
    """
    r, w, e, Y, per_rmn, p = args
    d_NDC = pensions.deriv_NDC(r, w, e, Y, per_rmn, p)

    assert np.allclose(d_NDC, d_NDC_expected)


## For all parameterizations below ##
t = 1
shift = False
method = "SS"
theta = None

################pension benefit: DB############
p = Specifications()
p.pension_system = "Defined Benefits"
p.S = 7
p.retire = 4
p.avg_earn_num_years = 50
p.yr_contrib = 55
p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.2
p.g_y = np.ones(p.T) * 0.03
w_db = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e_db = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n_db = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
Y = None
j = 1
factor = 2
omegas = None
lambdas = 1
pension_expected_db = [0, 0, 0, 0, 0.433821727, 0.421000358, 0.408557917]
args_pb = (r, w_db, n_db, Y, theta, t, j, shift, method, e_db, factor, p)

################pension benefit: NDC############
p2 = Specifications()
p2.pension_system = "Notional Defined Contribution"
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "r"
p2.S = 4
p2.retire = 2
w_ndc = np.array([1.2, 1.1, 1.21, 1])
e_ndc = np.array([1.1, 1.11, 0.9, 0.87])
n_ndc = np.array([0.4, 0.45, 0.4, 0.3])
Y = None
j = 1
p2.g_y = np.ones(p2.T) * 0.03
p2.g_n_ss = 0.0
r = np.ones(p2.T) * 0.03
factor = 2
p2.tau_p = 0.3
p2.k_ret = 0.4615
p2.mort_rates_SS = np.array([0.01, 0.05, 0.3, 0.4, 1])
omegas = None
lambdas = 1
pension_expected_ndc = [0, 0, 0.25751495, 0.24990423]
args_ndc = (r, w_ndc, n_ndc, Y, theta, t, j, shift, method, e_ndc, factor, p2)

################pension benefit: PS############
p3 = Specifications()
p3.pension_system = "Points System"
p3.S = 7
p3.retire = 4
w_ps = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e_ps = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n_ps = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
omegas = (1 / p3.S) * np.ones(p3.S)
p3.omega_SS = omegas
p3.vpoint = 0.4
factor = 2
Y = None
lambdas = 1
j = 1
p3.g_y = np.ones(p3.T) * 0.03
pension_expected_ps = [0, 0, 0, 0, 0.004164689, 0.004041603, 0.003922156]
args_ps = (r, w_db, n_db, Y, theta, t, j, shift, method, e_db, factor, p3)

## SS ##
p4 = Specifications()
p4.pension_system = "US-Style Social Security"
p4.S = 7
p4.retire = np.full((p4.T + p4.S, p4.S), 4)
w_ss = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e_ss = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n_ss = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
omegas = (1 / p4.S) * np.ones(p4.S)
theta = np.array([0.4, 0.4])
p4.replacement_rate_adjust = np.ones((p4.T, p4.J))
pension_expected_ss = [0, 0, 0, 0, 0.404, 0.396, 0.32]
method = "TPI"
shift = False
# the cohort (1D) path takes the unadjusted scalar theta for group j
args_ss = (
    r,
    w_ss,
    n_ss,
    Y,
    theta[j],
    t,
    j,
    shift,
    method,
    e_ss,
    factor,
    p4,
)


test_data = [
    (args_pb, pension_expected_db),
    (args_ndc, pension_expected_ndc),
    (args_ps, pension_expected_ps),
    (args_ss, pension_expected_ss),
]


@pytest.mark.parametrize(
    "args,pension_expected", test_data, ids=["DB", "NDC", "PS", "SS"]
)
def test_pension_amount(args, pension_expected):
    """
    Test of pensions.pension_amount
    """
    r, w, n, Y, theta, t, j, shift, method, e, factor, p = args

    pension = pensions.pension_amount(
        r, w, n, Y, theta, t, j, shift, method, e, factor, p
    )
    assert np.allclose(pension, pension_expected)


############SS or complete lifetimes############
p = Specifications()
p.S = 3
p.retire = 2
per_rmn = p.S
p.g_n_ss = 0.0
p.g_y = np.ones(p.T) * np.ones(p.T) * 0.03
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
w = np.array([1.2, 1.1, 1.21])
e = np.array([1.1, 1.11, 0.9])
r = np.ones(p.T) * 0.02
p.tau_p = 0.3
p.k_ret = 0.4615
delta_ret = 0.816476822
g_ndc = p.g_y[-1] + p.g_n_ss
p.mort_rates_SS = np.array([0.1, 0.2, 0.4, 1.0])
deriv_NDC_loop_expected1 = np.array([0.33302457, 0.29907546, 0.0])
d_theta_empty = np.zeros_like(w)
args3 = (w, e, per_rmn, g_ndc, delta_ret, d_theta_empty, p)

################Incomplete lifetimes#################
p2 = Specifications()
p2.S = 4
p2.retire = 2
per_rmn = 3
w2 = np.array([1.1, 1.21, 1.25])
e2 = np.array([1.11, 0.9, 1.0])
p2.g_y = np.ones(p.T) * 0.04
p2.g_n_ss = 0.0
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "r"
r = np.array([0.03, 0.04, 0.03])
p2.tau_p = 0.3
p2.k_ret = 0.4615
delta_ret2 = 0.896520810
g_ndc2 = p2.g_y[-1] + p2.g_n_ss
p2.mort_rates_SS = np.array([0.2, 0.4, 1.0])
deriv_NDC_loop_expected2 = np.array([0.32839557, 0.0, 0.0])
d_theta_empty = np.zeros(per_rmn)
args4 = (w2, e2, per_rmn2, g_ndc2, delta_ret2, d_theta_empty, p2)

test_data = [
    (args3, deriv_NDC_loop_expected1),
    (args4, deriv_NDC_loop_expected2),
]


@pytest.mark.parametrize(
    "args,deriv_NDC_loop_expected", test_data, ids=["SS", "incomplete"]
)
def test_deriv_NDC_loop(args, deriv_NDC_loop_expected):
    """
    Test of the pensions.deriv_NDC_loop() function.
    """
    w, e, per_rmn, g_ndc_value, delta_ret_value, d_theta, p = args

    print("TESTING", p.tau_p, delta_ret_value, g_ndc_value)
    deriv_NDC_loop = pensions.deriv_NDC_loop(
        w,
        e,
        per_rmn,
        p.S,
        p.retire,
        p.tau_p,
        g_ndc_value,
        delta_ret_value,
        d_theta,
    )

    assert np.allclose(deriv_NDC_loop, deriv_NDC_loop_expected)


############SS or complete lifietimes############
p = Specifications()
p.S = 4
p.retire = 2
p.g_y = np.ones(p.T) * np.ones(p.T) * 0.04
p.g_n_ss = 0.0
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
r = 0.02
g_ndc_amount = p.g_y[-1] + p.g_n_ss
p.mort_rates_SS = np.array([0.1, 0.2, 0.4, 0.6, 1.0])
dir_delta_s_empty = np.zeros(p.S - p.retire + 1)
# 1 + (1 - 0.4) / 1.04: payments at ages 2 and 3
dir_delta_ret_expected1 = 1.576923077
surv_rates = np.zeros(p.S - p.retire + 1)
surv_rates = 1 - p.mort_rates_SS
args5 = (surv_rates, g_ndc_amount, dir_delta_s_empty, p)
test_data = [(args5, dir_delta_ret_expected1)]


@pytest.mark.parametrize("args,dir_delta_ret_expected", test_data, ids=["SS"])
def test_delta_ret_loop(args, dir_delta_ret_expected):
    """
    Test of the pensions.delta_ret_loop() function.
    """
    surv_rates, g_ndc_value, dir_delta_s_empty, p = args
    dir_delta = pensions.delta_ret_loop(p.S, p.retire, surv_rates, g_ndc_value)

    assert np.allclose(dir_delta, dir_delta_ret_expected)


#####################SS / Complete lifetimes############
p = Specifications()
p.S = 7
p.retire = 4
p.avg_earn_num_years = 50
p.yr_contrib = 55
p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.2
p.g_y = np.ones(p.T) * 0.03
j = 1
w = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
DB_expected1 = np.array([0, 0, 0, 0, 0.433821727, 0.421000358, 0.408557917])
args1 = (w, e, n, None, j, "SS", p)

# Same steady state, but with a replacement rate that differs across
# lifetime income groups: group j=1 keeps 0.2 and the other groups get
# a very different rate, so the result for j=1 is unchanged only if the
# group-specific rate is used.
p_J = copy.deepcopy(p)
p_J.alpha_db = np.ones((p_J.T + p_J.S, p_J.J)) * 0.9
p_J.alpha_db[:, 1] = 0.2
args1_J = (w, e, n, None, j, "SS", p_J)

#####################Incomplete############
p2 = Specifications()
p2.S = 7
p2.retire = 4
p2.avg_earn_num_years = 50
p2.yr_contrib = 55
p2.alpha_db = np.ones((p2.T + p2.S, p2.J)) * 0.2
p2.g_y = np.ones(p2.T) * 0.03
j = 1
w2 = np.array([1.21, 1.0, 1.01, 0.99, 0.8])
e2 = np.array(
    [
        [1.1, 1.1],
        [1.11, 1.11],
        [0.9, 0.9],
        [0.87, 0.87],
        [0.87, 0.87],
        [0.7, 0.7],
        [0.6, 0.6],
    ]
)
n2 = np.array([0.4, 0.42, 0.3, 0.2, 0.2])
p2.w_preTP = np.array([1.05])
p2.n_preTP = np.array(
    [
        [0.4, 0.4],
        [0.3, 0.3],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
        [0.45, 0.45],
        [0.5, 0.5],
    ]
)
p2.e = e2
DB_expected2 = np.array([0, 0, 0.40132004, 0.389459241, 0.377948981])
args2 = (w2, e2, n2, 0, j, "TPI", p2)

# Incomplete lifetime straddling a change in the replacement rate.  This
# household is age 2 at t=0 (S=7, 5 periods remaining) so its benefits
# at ages 4, 5, 6 are paid in periods 2, 3, 4.  The rate doubles from
# period 3 on, so the age-4 benefit is unchanged and the age-5 and
# age-6 benefits double: the rate in effect when the benefit is paid
# applies, not the rate at retirement.
p2_tv = copy.deepcopy(p2)
p2_tv.alpha_db = np.ones((p2_tv.T + p2_tv.S, p2_tv.J)) * 0.2
p2_tv.alpha_db[3:, :] = 0.4
DB_expected2_tv = DB_expected2 * np.array([1.0, 1.0, 1.0, 2.0, 2.0])
args2_tv = (w2, e2, n2, 0, j, "TPI", p2_tv)

test_data = [
    (args1, DB_expected1),
    (args1_J, DB_expected1),
    (args2, DB_expected2),
    (args2_tv, DB_expected2_tv),
]


@pytest.mark.parametrize(
    "args,DB_expected",
    test_data,
    ids=[
        "SS/Complete",
        "SS/Complete, alpha_db varies by J",
        "Incomplete",
        "Incomplete, alpha_db varies over time",
    ],
)
def test_DB(args, DB_expected):
    """
    Test of the pensions.DB_amount() function.
    """
    w, e, n, t, j, method, p = args
    DB = pensions.DB_amount(w, e, n, t, j, method, p)

    assert np.allclose(DB, DB_expected)


def test_DB_cross_section_varies_by_J():
    """
    In the steady state with all groups computed at once (S x J inputs),
    each group's DB benefit should scale with its own replacement rate.
    """
    p_xs = Specifications()
    p_xs.S = 7
    p_xs.J = 3
    p_xs.retire = 4
    p_xs.avg_earn_num_years = 50
    p_xs.yr_contrib = 55
    p_xs.g_y = 0.03
    alpha_J = np.array([0.1, 0.2, 0.4])
    p_xs.alpha_db = np.tile(alpha_J.reshape(1, 3), (p_xs.T + p_xs.S, 1))
    w_xs = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
    e_xs = np.tile(
        np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6]).reshape(7, 1),
        (1, 3),
    )
    n_xs = np.tile(
        np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2]).reshape(7, 1), (1, 3)
    )
    DB = pensions.DB_amount(w_xs, e_xs, n_xs, None, None, "SS", p_xs)
    assert DB.shape == (7, 3)
    # identical earnings histories, so benefits differ only by the rate
    assert np.allclose(DB[:, 1], DB_expected1)
    assert np.allclose(DB[:, 0], DB_expected1 * 0.5)
    assert np.allclose(DB[:, 2], DB_expected1 * 2.0)


def test_DB_time_path_consistent_with_cohort_solves():
    """
    With a replacement rate that varies over time and by group, the DB
    benefits computed for the full time path at once (the T x S x J
    branch used for aggregates) must agree with the benefits computed
    cohort by cohort (the partial- and full-lifetime branches used in
    the household solves), since both index the rate by the period in
    which the benefit is paid.
    """
    p_tp = Specifications()
    p_tp.S = 7
    p_tp.J = 2
    p_tp.T = 12
    p_tp.retire = 4
    p_tp.avg_earn_num_years = 50
    p_tp.yr_contrib = 55
    p_tp.g_y = 0.03
    rng = np.random.RandomState(3)
    p_tp.alpha_db = 0.1 + 0.3 * rng.rand(p_tp.T + p_tp.S, p_tp.J)
    p_tp.e = np.tile(
        np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6]).reshape(1, 7, 1),
        (p_tp.T + p_tp.S, 1, 2),
    )
    p_tp.e[:, :, 1] *= 1.5
    # constant wage and labor supply over time so that the pre-time-path
    # history used by the two branches is the same
    w_path = np.ones(p_tp.T) * 1.2
    n_ss = np.tile(
        np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2]).reshape(7, 1), (1, 2)
    )
    n_path = np.tile(n_ss.reshape(1, 7, 2), (p_tp.T, 1, 1))
    p_tp.n_preTP = n_ss
    DB_3D = pensions.DB_amount(w_path, None, n_path, 0, None, "TPI", p_tp)
    assert DB_3D.shape == (p_tp.T, p_tp.S, p_tp.J)
    for j in range(p_tp.J):
        # cohorts alive at t=0, of age a, with S - a periods remaining
        for a in range(1, p_tp.S):
            per_rmn = p_tp.S - a
            DB_1D = pensions.DB_amount(
                w_path[:per_rmn],
                p_tp.e[-1, a:, j],
                n_ss[a:, j],
                0,
                j,
                "TPI",
                p_tp,
            )
            path_vals = np.array(
                [DB_3D[tt, a + tt, j] for tt in range(per_rmn)]
            )
            assert np.allclose(DB_1D, path_vals), (j, a)
        # cohorts born at t >= 0 with full lifetimes inside the path
        for t0 in range(p_tp.T - p_tp.S + 1):
            DB_1D = pensions.DB_amount(
                w_path[t0 : t0 + p_tp.S],
                p_tp.e[-1, :, j],
                n_ss[:, j],
                t0,
                j,
                "TPI",
                p_tp,
            )
            path_vals = np.array([DB_3D[t0 + u, u, j] for u in range(p_tp.S)])
            assert np.allclose(DB_1D, path_vals), (j, t0)


def test_DB_partial_lifetime_uses_time_varying_e():
    """
    For a cohort alive when the time path begins (the partial-lifetime
    branch), the earnings profile used in the benefit formula must be
    the one in effect in the period each age is lived in, not the
    steady-state profile.  Check this against the full-lifetime branch
    called with the lifetime profile built by hand, and check that the
    answer differs from the one using the steady-state profile.
    """
    p_tv = Specifications()
    p_tv.S = 7
    p_tv.J = 2
    p_tv.T = 12
    p_tv.retire = 4
    p_tv.avg_earn_num_years = 50
    p_tv.yr_contrib = 55
    p_tv.g_y = 0.03
    p_tv.alpha_db = np.ones((p_tv.T + p_tv.S, p_tv.J)) * 0.2
    e_ss = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
    # profile drifts up over the path, settling at the steady state
    scale = np.linspace(0.6, 1.0, p_tv.T).reshape(p_tv.T, 1, 1)
    p_tv.e = scale * np.tile(e_ss.reshape(1, 7, 1), (p_tv.T, 1, 2))
    p_tv.e[:, :, 1] *= 1.5
    w_path = np.linspace(1.2, 1.3, p_tv.T)
    n_ss = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
    p_tv.n_preTP = np.tile(n_ss.reshape(7, 1), (1, 2))
    j = 1
    for t in range(0, 3):
        for a in range(1, p_tv.S):
            per_rmn = p_tv.S - a
            birth = t - a
            # lifetime path as seen by DB_amount's partial branch
            w_S = np.append(np.full(a, w_path[t]), w_path[t : t + per_rmn])
            n_S = np.append(p_tv.n_preTP[:a, j], n_ss[a:])
            periods = np.clip(birth + np.arange(p_tv.S), 0, p_tv.T - 1)
            e_S = p_tv.e[periods, np.arange(p_tv.S), j]
            expected = pensions.DB_amount(
                w_S, e_S, n_S, birth, j, "TPI", p_tv
            )[-per_rmn:]
            DB_partial = pensions.DB_amount(
                w_path[t : t + per_rmn],
                e_S[a:],
                n_ss[a:],
                t,
                j,
                "TPI",
                p_tv,
            )
            assert np.allclose(DB_partial, expected), (t, a)
            if per_rmn > p_tv.S - p_tv.retire:
                # a retiree whose benefit was computed before the SS
                # profile applies: the SS profile would give a different
                # (here larger) benefit
                DB_ss_e = pensions.DB_amount(
                    w_S, p_tv.e[-1, :, j], n_S, birth, j, "TPI", p_tv
                )[-per_rmn:]
                assert not np.allclose(DB_partial, DB_ss_e), (t, a)


################pension benefit derivative: DB############
p = Specifications()
p.pension_system = "Defined Benefits"
p.S = 7
p.retire = 4
per_rmn = p.S
p.avg_earn_num_years = 50
p.yr_contrib = 55
p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.2
w_ddb = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e_ddb = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
p.g_y = np.ones(p.T) * 0.03
Y = None
r_ddb = np.ones(p.T) * 0.03
factor = 2
d_theta_expected_ddb = np.array(
    [0.33, 0.30525, 0.27225, 0.2175, 0.0, 0.0, 0.0]
)
args_ddb = (r_ddb, w_ddb, e_ddb, Y, per_rmn, factor, 0, 1, p)

################pension benefit derivative: NDC############
p2 = Specifications()
p2.pension_system = "Notional Defined Contribution"
p2.S = 4
p2.retire = 2
per_rmn = p2.S
w_dndc = np.array([1.2, 1.1, 1.21, 1])
e_dndc = np.array([1.1, 1.11, 0.9, 0.87])
p2.g_y = np.ones(p2.T) * 0.03
p2.g_n_ss = 0.0
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "r"
r_dndc = np.ones(p2.T) * 0.02
p2.tau_p = 0.3
p2.k_ret = 0.4615
p2.mort_rates_SS = np.array([0.01, 0.05, 0.3, 1])
d_theta_expected_dndc = np.array([0.33317153, 0.29907546, 0, 0])
Y = None
factor = 2
args_dndc = (r_dndc, w_dndc, e_dndc, Y, per_rmn, factor, 0, 1, p2)


################pension benefit derivative: PS############
p3 = Specifications()
p3.pension_system = "Points System"
p3.S = 7
p3.retire = 4
w_dps = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e_dps = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
p3.g_y = np.ones(p3.T) * 0.03
Y = None
factor = 2
r_dps = np.ones(p3.T) * 0.03
p3.vpoint = 0.4
omegas = 1 / (p3.S) * np.ones(p3.S)
p3.omega_SS = omegas
per_rmn_ps = 5
d_theta_expected_dps = np.array([0.0026136, 0.002088, 0, 0, 0])
args_dps = (r_dps, w_dps, e_dps, Y, per_rmn_ps, factor, 0, 1, p3)

test_data = [
    (args_ddb, d_theta_expected_ddb),
    (args_dndc, d_theta_expected_dndc),
    (args_dps, d_theta_expected_dps),
]


@pytest.mark.parametrize(
    "args,d_theta_expected", test_data, ids=["DB", "NDC", "PS"]
)
def test_deriv_theta(args, d_theta_expected):
    """
    Test of pensions.deriv_theta
    """
    r, w, e, Y, per_rmn, factor, t, j, p = args
    d_theta = pensions.deriv_theta(r, w, e, Y, per_rmn, factor, t, j, p)
    assert np.allclose(d_theta, d_theta_expected)


#############complete lifetimes, S = 4###################
p = Specifications()
p.S = 4
p.retire = 2
j = 1
w = np.array([1.2, 1.1, 1.21, 1])
e = np.array([1.1, 1.11, 0.9, 0.87])
n = np.array([0.4, 0.45, 0.4, 0.3])
p.g_y = np.ones(p.T) * 0.03
p.g_n_ss = 0.0
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
r = np.ones(p.T) * 0.03
p.tau_p = 0.3
p.k_ret = 0.4615
p.mort_rates_SS = np.array([0.01, 0.05, 0.3, 0.4, 1])
NDC_expected1 = np.array([0, 0, 0.25751495, 0.24990423])
# NDC_expected1 = np.array([0, 0, 0.279756794, 0.271488732])
args1 = (w, e, n, r, None, None, j, "SS", p)

#############incomplete lifetimes###################
p2 = Specifications()
p2.S = 4
p2.retire = 2
j = 1
w = np.array([1.1, 1.21, 1])
e = np.array([[1.0, 1.0], [1.11, 1.11], [0.9, 0.9], [0.87, 0.87]])
n = np.array([0.45, 0.4, 0.3])
p2.w_preTP = np.array([1.05])
p2.n_preTP = np.array([[0.4, 0.4], [0.2, 0.2], [0.3, 0.3], [0.5, 0.5]])
p2.g_y = np.ones(p2.T) * 0.03
p2.g_n_ss = 0.0
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "r"
r = np.ones(p2.T) * 0.03
p2.tau_p = 0.3
p2.k_ret = 0.4615
p2.mort_rates_SS = np.array([0.01, 0.05, 0.3, 0.4, 1])
p2.e = e
# Pre-time-path wages are anchored to the period-0 wage of the passed
# path (w[0] = 1.1), not the old w_preTP attribute. Earlier expected
# values under the old convention, kept for reference:
# NDC_expected2 = np.array([0, 0.25185784, 0.24441432])
# TODO: why move from numbers below to those above ?  Diff in numpy rounding??
# NDC_expected2 = np.array([0, 0.251721214, 0.244281728])
NDC_expected2 = np.array([0, 0.23648259, 0.22949347])
args2 = (w, e, n, r, None, 0, j, "TPI", p2)

test_data = [(args1, NDC_expected1), (args2, NDC_expected2)]


@pytest.mark.parametrize(
    "args,NDC_expected", test_data, ids=["SS/Complete", "Incomplete"]
)
def test_NDC(args, NDC_expected):
    """
    Test of the pensions.NDC() function.
    """
    w, e, n, r, Y, t, j, method, p = args
    NDC = pensions.NDC_amount(w, e, n, r, Y, t, j, method, p)
    assert np.allclose(NDC, NDC_expected)


#############complete lifetimes, S = 7###################
p = Specifications()
p.S = 7
p.retire = 4
p.vpoint = 0.4
j = 1
w = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
p.g_y = np.ones(p.T) * 0.03
factor = 2
points_py_s = np.zeros(p.retire)
L_inc_avg_s = np.zeros(p.retire)
L_inc_avg = np.zeros(1)
PS = np.zeros(p.S)
PS_loop_expected = np.array(
    [0, 0, 0, 0, 0.004164689, 0.004041603, 0.003922156]
)
args1 = (w, e, n, p.retire, p.S, p.g_y, p.vpoint, factor, L_inc_avg_s, PS)

test_data = [(args1, PS_loop_expected)]


@pytest.mark.parametrize(
    "args,PS_loop_expected", test_data, ids=["SS/Complete"]
)
def test_PS_1dim_loop(args, PS_loop_expected):
    """
    Test of the pensions.PS_1dim_loop() function.
    """
    w, e, n, S_ret, S, g_y, vpoint, factor, L_inc_avg_s, PS = args
    PS_loop = pensions.PS_1dim_loop(
        w, e, n, S_ret, S, g_y, vpoint, factor, L_inc_avg_s, PS
    )
    assert np.allclose(PS_loop, PS_loop_expected)


#####################SS / Complete lifetimes############
p = Specifications()
p.S = 7
p.retire = 4
p.g_y = np.ones(p.T) * 0.03
j = 1
lambdas = 1
w = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
omegas = (1 / p.S) * np.ones(p.S)
p.omega_SS = omegas
factor = 2
points_py_s = np.zeros(p.retire)
L_inc_avg_s = np.zeros(p.retire)
L_inc_avg = np.zeros(1)
PS = np.zeros(p.S)
p.vpoint = 0.4
PS_expected1 = np.array([0, 0, 0, 0, 0.004164689, 0.004041603, 0.003922156])
args1 = (w, e, n, j, factor, p)

####################Incomplete############
p2 = Specifications()
p2.S = 7
p2.retire = 4
p2.omega_SS = omegas
factor = 2
p2.g_y = np.ones(p2.T) * 0.03
p2.vpoint = 0.4
j = 1
lambdas = 1
points_py_s = np.zeros(p.retire)
L_inc_avg_s = np.zeros(p.retire)
L_inc_avg = np.zeros(1)
PS = np.zeros(p.S)
w2 = np.array([1.21, 1.0, 1.01, 0.99, 0.8])
e2 = np.array(
    [
        [1.1, 1.1],
        [1.11, 1.11],
        [0.9, 0.9],
        [0.87, 0.87],
        [0.87, 0.87],
        [0.7, 0.7],
        [0.6, 0.6],
    ]
)
n2 = np.array([0.4, 0.42, 0.3, 0.2, 0.2])
p2.w_preTP = np.array([1.05])
p2.n_preTP = np.array(
    [
        [0.4, 0.4],
        [0.3, 0.3],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
        [0.45, 0.45],
        [0.5, 0.5],
    ]
)
p2.e = e2
# Pre-time-path wages anchored to the period-0 wage (w2[0] = 1.21)
PS_expected2 = np.array([0, 0, 0.003852672, 0.003738809, 0.00362831])
args2 = (w2, e2, n2, j, factor, p2)

test_data = [(args1, PS_expected1), (args2, PS_expected2)]


@pytest.mark.parametrize(
    "args,PS_expected", test_data, ids=["SS/Complete", "Incomplete"]
)
def test_get_PS(args, PS_expected):
    """
    Test of the pensions.get_PS() function.
    """
    w, e, n, j, factor, p = args
    PS = pensions.PS_amount(w, e, n, j, factor, p)
    print("PS inside of the test", PS)
    assert np.allclose(PS, PS_expected)


# Regression tests for the array/scalar bugs hit when the non-US pension
# systems run against a real Specifications object (Issues #1014 and #1075):
# p.retire is an array (time-varying since PR #433) but was passed as the
# scalar S_ret into the numba loops; p.g_y is a scalar but the loops index
# it; and in the SS the wage is a scalar but the loops index it as a path.
# The mock-parameter tests above pre-scalarize these inputs, so they never
# exercised the real interface.
@pytest.mark.parametrize(
    "system,updates",
    [
        (
            "Defined Benefits",
            {"alpha_db": [[0.02]], "yr_contrib": 35, "avg_earn_num_years": 40},
        ),
        ("Points System", {"vpoint": 0.5}),
        # The NDC growth-rate settings (ndc_growth_rate,
        # dir_growth_rate) became parameters with Issue #1169; tau_p
        # must be positive for the pension to be nonzero
        (
            "Notional Defined Contribution",
            {
                "tau_p": 0.1,
                "ndc_growth_rate": "LR GDP",
                "dir_growth_rate": "r",
            },
        ),
        (
            "Hybrid DB-NDC",
            {
                "alpha_db": [[0.02]],
                "yr_contrib": 35,
                "avg_earn_num_years": 40,
                "tau_p": 0.1,
                "ndc_growth_rate": "LR GDP",
                "dir_growth_rate": "r",
                "factor1_db_ndc_hybrid": [0.5],
                "factor2_db_ndc_hybrid": [1.5],
                "avg_labor_inc_t1_model_units": 0.8,
            },
        ),
    ],
    ids=["DB", "PS", "NDC", "Hybrid"],
)
def test_pension_amount_with_real_specifications(system, updates):
    """
    pension_amount must accept a real Specifications object in the SS:
    retire is an array, g_y a scalar, and the steady-state wage a scalar.
    """
    p = Specifications()
    p.update_specifications(dict(updates, pension_system=system))
    j = 0
    w = 1.2  # steady-state wage is a scalar
    r, Y, factor = 0.05, 1.0, 100000.0
    n = 0.4 * np.ones(p.S)
    e = p.e[-1, :, j]
    theta = np.zeros(p.J)
    pension = pensions.pension_amount(
        r, w, n, Y, theta, None, j, False, "SS", e, factor, p
    )
    S_ret = int(p.retire[-1, -1])
    assert pension.shape == (p.S,)
    assert np.all(pension[:S_ret] == 0.0)
    assert np.all(pension[S_ret:] > 0.0)


@pytest.mark.parametrize(
    "system,updates",
    [
        ("US-Style Social Security", {}),
        ("Defined Benefits", {"alpha_db": [[0.01]], "yr_contrib": 40}),
        (
            "Points System",
            {"vpoint": 0.4, "points_growth_rate": "LR GDP"},
        ),
        (
            "Notional Defined Contribution",
            {
                "tau_p": 0.1,
                "ndc_growth_rate": "LR GDP",
                "dir_growth_rate": "r",
            },
        ),
        (
            "Hybrid DB-NDC",
            {
                "alpha_db": [[0.02]],
                "yr_contrib": 35,
                "avg_earn_num_years": 40,
                "tau_p": 0.1,
                "ndc_growth_rate": "LR GDP",
                "dir_growth_rate": "r",
                "factor1_db_ndc_hybrid": [0.5],
                "factor2_db_ndc_hybrid": [1.5],
                "avg_labor_inc_t1_model_units": 0.8,
            },
        ),
    ],
    ids=["SS", "DB", "PS", "NDC", "Hybrid"],
)
def test_replacement_rate_adjust_applies_to_every_system(system, updates):
    """
    replacement_rate_adjust must scale benefits under every pension system.

    It was previously read only inside SS_amount, so setting it under any
    system other than US-Style Social Security was silently ignored.
    """

    def pension_for(adjust):
        p = Specifications()
        p.update_specifications(dict(updates, pension_system=system))
        adjust_arr = np.asarray(p.replacement_rate_adjust).copy()
        adjust_arr[:, :] = adjust
        p.replacement_rate_adjust = adjust_arr
        j = 0
        return pensions.pension_amount(
            0.05,
            1.2,
            0.4 * np.ones(p.S),
            1.0,
            0.1,
            None,
            j,
            False,
            "SS",
            p.e[-1, :, j],
            100000.0,
            p,
        )

    full = pension_for(1.0)
    halved = pension_for(0.5)
    assert np.any(full > 0.0), "test would be vacuous with a zero benefit"
    assert np.allclose(halved, 0.5 * full)


HYBRID_UPDATES = {
    "alpha_db": [[0.02]],
    "yr_contrib": 35,
    "avg_earn_num_years": 40,
    "tau_p": 0.1,
    "ndc_growth_rate": "LR GDP",
    "dir_growth_rate": "r",
}


def _hybrid_params(factor1, factor2, avg_inc=0.8):
    """
    Specifications for the hybrid system; factor1 and factor2 may be
    single values (held constant over time) or lists of values by year,
    and avg_inc is the average labor income in the first model period,
    in model units.
    """

    def _path(f):
        # a single entered value would be indexed to remove trend
        # growth, so repeat it to hold the factor constant
        f = np.atleast_1d(f).tolist()
        return f * 2 if len(f) == 1 else f

    p = Specifications()
    p.update_specifications(
        dict(
            HYBRID_UPDATES,
            pension_system="Hybrid DB-NDC",
            factor1_db_ndc_hybrid=_path(factor1),
            factor2_db_ndc_hybrid=_path(factor2),
            avg_labor_inc_t1_model_units=avg_inc,
        )
    )
    # labor supply before the time path, normally set by the TPI solver
    p.n_preTP = 0.4 * np.ones((p.S, p.J))
    return p


def test_earnings_band_at():
    """
    earnings_band_at returns the part of labor income between the
    cutoffs in effect in the period it was earned, and leaves it
    unchanged with lo = 0 and hi = inf. Periods outside the paths read
    their first or last element.
    """
    x = np.array([0.2, 0.7, 1.2, 3.0])

    def band(x, lo, hi, period=0):
        return pensions.earnings_band_at(
            x, np.atleast_1d(float(lo)), np.atleast_1d(float(hi)), period
        )

    assert np.allclose(band(x, 0.0, np.inf), x)
    assert np.allclose(band(x, 0.0, 1.0), [0.2, 0.7, 1.0, 1.0])
    assert np.allclose(band(x, 1.0, 2.0), [0.0, 0.0, 0.2, 1.0])
    assert np.isclose(band(1.2, 1.0, 2.0), 0.2)
    lo, hi = np.array([0.0, 1.0]), np.array([1.0, 2.0])
    assert np.allclose(
        pensions.earnings_band_at(x, lo, hi, 0), [0.2, 0.7, 1.0, 1.0]
    )
    assert np.allclose(
        pensions.earnings_band_at(x, lo, hi, 1), [0.0, 0.0, 0.2, 1.0]
    )
    assert np.allclose(
        pensions.earnings_band_at(x, lo, hi, -3),
        pensions.earnings_band_at(x, lo, hi, 0),
    )
    assert np.allclose(
        pensions.earnings_band_at(x, lo, hi, 50),
        pensions.earnings_band_at(x, lo, hi, 1),
    )


@pytest.mark.parametrize("ndim", [1, 2], ids=["S", "SxJ"])
def test_DB_NDC_hybrid_amount_SS(ndim):
    """
    In the steady state the hybrid benefit equals the DB benefit on labor
    income capped at factor1 * avg_inc plus the NDC benefit on labor
    income between factor1 * avg_inc and factor2 * avg_inc.
    """
    factor1, factor2 = 0.5, 1.5
    p = _hybrid_params(factor1, factor2)
    r, w, ndc_rates = 0.05, 1.2, None
    if ndim == 1:
        # the top income group, whose earnings reach both bands
        j = p.J - 1
        e = p.e[-1, :, j]
        n = 0.4 * np.ones(p.S)
    else:
        j = None
        e = p.e[-1, :, :]
        n = 0.4 * np.ones((p.S, p.J))
    hybrid = pensions.DB_NDC_hybrid_amount(
        w, e, n, r, ndc_rates, None, j, "SS", p
    )
    # same benefits computed by hand from the labor supply that delivers
    # each band's labor income
    cut1 = factor1 * p.avg_labor_inc_t1_model_units
    cut2 = factor2 * p.avg_labor_inc_t1_model_units
    inc = w * e * n
    n_db = np.minimum(inc, cut1) / (w * e)
    n_ndc = np.clip(inc - cut1, 0.0, cut2 - cut1) / (w * e)
    db = pensions.DB_amount(w, e, n_db, None, j, "SS", p)
    ndc = pensions.NDC_amount(w, e, n_ndc, r, ndc_rates, None, j, "SS", p)
    assert np.any(ndc > 0.0) and np.any(db > 0.0)
    assert np.allclose(hybrid, db + ndc)
    # and the band cutoffs bind: both parts are below the uncapped ones
    assert np.all(db <= pensions.DB_amount(w, e, n, None, j, "SS", p) + 1e-12)


@pytest.mark.parametrize("method", ["SS", "TPI"])
def test_DB_NDC_hybrid_amount_limits(method):
    """
    With factor1 above all earnings the hybrid is the pure DB system, and
    with factor1 = 0 and factor2 above all earnings it is the pure NDC
    system. Checked on the full time-path array and the SS cross section.
    """
    big = 100.0
    r, ndc_rates = 0.05, None
    p_db = _hybrid_params(big, big)
    p_ndc = _hybrid_params(0.0, big)
    if method == "SS":
        w, t, j = 1.2, None, None
        e = p_db.e[-1, :, :]
        n = 0.4 * np.ones((p_db.S, p_db.J))
    else:
        # the full time-path array has at least S periods
        T = p_db.S + 5
        w, t, j = np.linspace(1.0, 1.2, T), 0, None
        # time-varying credited and discount rates along the path
        n_per = p_db.T + p_db.S
        ndc_rates = pensions.NDCRates(
            np.linspace(0.01, 0.03, n_per), np.linspace(0.03, 0.02, n_per)
        )
        e = p_db.e[-1, :, :]
        n = 0.4 * np.ones((T, p_db.S, p_db.J))
    assert np.allclose(
        pensions.DB_NDC_hybrid_amount(
            w, e, n, r, ndc_rates, t, j, method, p_db
        ),
        pensions.DB_amount(w, e, n, t, j, method, p_db),
    )
    assert np.allclose(
        pensions.DB_NDC_hybrid_amount(
            w, e, n, r, ndc_rates, t, j, method, p_ndc
        ),
        pensions.NDC_amount(w, e, n, r, ndc_rates, t, j, method, p_ndc),
    )


def test_DB_NDC_hybrid_amount_partial_path():
    """
    A cohort alive when the time path begins (partial lifetime path) is
    handled, and matches the DB limit when factor1 exceeds all earnings.
    """
    r, ndc_rates, j, t = 0.05, None, 0, 0
    p = _hybrid_params(100.0, 100.0)
    per_rmn = p.S // 2
    w = np.ones(per_rmn) * 1.1
    n = 0.4 * np.ones(per_rmn)
    e = p.e[-1, -per_rmn:, j]
    hybrid = pensions.DB_NDC_hybrid_amount(
        w, e, n, r, ndc_rates, t, j, "TPI", p
    )
    assert hybrid.shape == (per_rmn,)
    assert np.allclose(hybrid, pensions.DB_amount(w, e, n, t, j, "TPI", p))


@pytest.mark.parametrize("method", ["SS", "TPI"])
def test_DB_NDC_hybrid_amount_cutoffs_in_levels(method):
    """
    The cutoffs are factor * avg_labor_inc_t1_model_units, levels of
    labor income that do not move with the wage: scaling the factors by
    c and the average income by 1 / c leaves the benefit unchanged, and
    raising the wage moves more income above the cutoffs, so the
    benefit rises less than in proportion.
    """
    r, ndc_rates = 0.05, None
    p = _hybrid_params(0.5, 1.5, avg_inc=0.8)
    p_scaled = _hybrid_params(0.25, 0.75, avg_inc=1.6)
    if method == "SS":
        w, t, j = 1.2, None, None
        e = p.e[-1, :, :]
        n = 0.4 * np.ones((p.S, p.J))
    else:
        T = p.S + 5
        w, t, j = np.linspace(1.0, 1.2, T), 0, None
        e = p.e[-1, :, :]
        n = 0.4 * np.ones((T, p.S, p.J))
    base = pensions.DB_NDC_hybrid_amount(
        w, e, n, r, ndc_rates, t, j, method, p
    )
    assert np.any(base > 0.0)
    assert np.allclose(
        base,
        pensions.DB_NDC_hybrid_amount(
            w, e, n, r, ndc_rates, t, j, method, p_scaled
        ),
    )
    doubled = pensions.DB_NDC_hybrid_amount(
        2.0 * np.asarray(w), e, n, r, ndc_rates, t, j, method, p
    )
    assert np.all(doubled <= 2.0 * base + 1e-12)
    assert np.any(doubled < 2.0 * base - 1e-12)


def test_hybrid_factors_scalar_is_indexed():
    """
    A single value entered for a hybrid factor (as in the defaults) is
    the initial value of a path with trend growth removed,
    init_val / e^{g_y t}; two or more values are used as given.
    """
    p = Specifications()
    t = np.arange(p.T + p.S)
    assert np.allclose(p.factor1_db_ndc_hybrid, 1.0 * np.exp(-p.g_y * t))
    assert np.allclose(p.factor2_db_ndc_hybrid, 2.0 * np.exp(-p.g_y * t))
    assert p.avg_labor_inc_t1_model_units == 1.0
    p.update_specifications(
        {
            "factor1_db_ndc_hybrid": [0.5],
            "factor2_db_ndc_hybrid": [1.5, 1.5],
        }
    )
    assert np.allclose(p.factor1_db_ndc_hybrid, 0.5 * np.exp(-p.g_y * t))
    assert np.allclose(p.factor2_db_ndc_hybrid, 1.5)


def test_DB_NDC_hybrid_amount_bad_factors():
    """factor2_db_ndc_hybrid below factor1_db_ndc_hybrid raises."""
    p = _hybrid_params(1.0, 2.0)
    p.factor1_db_ndc_hybrid = 3.0
    with pytest.raises(ValueError):
        pensions.DB_NDC_hybrid_amount(
            1.2,
            p.e[-1, :, 0],
            0.4 * np.ones(p.S),
            0.05,
            None,
            None,
            0,
            "SS",
            p,
        )


@pytest.mark.local
def test_SS_solve_defined_benefits(tmp_path):
    """
    The steady state must solve with the Defined Benefits pension system
    and produce positive aggregate pension outlays (Issue #1014 asked for
    model-run tests of each pension system; the analogous NDC and Points
    System runs should be added with the resolution of that issue).
    """
    from ogcore.execute import runner
    from ogcore import utils

    p = Specifications(baseline=True, num_workers=1)
    p.update_specifications(
        {
            "pension_system": "Defined Benefits",
            "alpha_db": [[0.02]],
            "yr_contrib": 35,
            "avg_earn_num_years": 40,
        }
    )
    p.baseline_dir = p.output_base = str(tmp_path)
    runner(p, time_path=False, client=None)
    ss = utils.safe_read_pickle(str(tmp_path / "SS" / "SS_vars.pkl"))
    Y = np.asarray(ss["Y"]).sum()
    assert ss["agg_pension_outlays"] > 0.0
    assert 0.0 < ss["agg_pension_outlays"] / Y < 1.0


def test_hybrid_factors_time_varying_parameters():
    """
    The hybrid cutoffs are paths of length T + S: values given by year
    are held at the last value entered, the two lists may have
    different lengths, and factor2 below factor1 in any period raises.
    """
    p = _hybrid_params([0.5, 0.6, 0.7], [1.5, 1.6])
    assert p.factor1_db_ndc_hybrid.shape == (p.T + p.S,)
    assert p.factor2_db_ndc_hybrid.shape == (p.T + p.S,)
    assert np.allclose(p.factor1_db_ndc_hybrid[:4], [0.5, 0.6, 0.7, 0.7])
    assert np.allclose(p.factor2_db_ndc_hybrid[:4], [1.5, 1.6, 1.6, 1.6])
    assert np.allclose(p.factor1_db_ndc_hybrid[-1], 0.7)
    with pytest.raises(ValueError, match="in every period"):
        _hybrid_params([0.5, 0.6, 2.0], [1.5, 1.6])


def _time_varying_hybrid_params():
    """Cutoffs that change in each of the first S + 10 years."""
    p0 = Specifications()
    n_years = p0.S + 10
    factor1 = np.linspace(0.3, 0.9, n_years)
    factor2 = np.linspace(1.0, 2.0, n_years)
    return _hybrid_params(factor1, factor2)


def test_DB_NDC_hybrid_amount_time_varying_cohort():
    """
    Along the time path a cohort's benefit uses the cutoffs in effect in
    the period of each earning: it equals the DB and NDC benefits on the
    labor supply that delivers each band's effective earnings, with the
    cutoffs read down the cohort's own path.
    """
    p = _time_varying_hybrid_params()
    r, ndc_rates, t = 0.05, None, 3
    j = p.J - 1
    w = np.linspace(1.0, 1.3, p.S)
    e = p.e[-1, :, j]
    n = 0.4 * np.ones(p.S)
    hybrid = pensions.DB_NDC_hybrid_amount(
        w, e, n, r, ndc_rates, t, j, "TPI", p
    )
    # the household is age 0 in period t, so age s is lived in t + s
    f1 = p.factor1_db_ndc_hybrid[t : t + p.S]
    f2 = p.factor2_db_ndc_hybrid[t : t + p.S]
    cut1 = f1 * p.avg_labor_inc_t1_model_units
    cut2 = f2 * p.avg_labor_inc_t1_model_units
    inc = w * e * n
    n_db = np.minimum(inc, cut1) / (w * e)
    n_ndc = np.clip(inc - cut1, 0.0, cut2 - cut1) / (w * e)
    db = pensions.DB_amount(w, e, n_db, t, j, "TPI", p)
    ndc = pensions.NDC_amount(w, e, n_ndc, r, ndc_rates, t, j, "TPI", p)
    assert np.any(db > 0.0) and np.any(ndc > 0.0)
    assert np.allclose(hybrid, db + ndc)
    # the time variation matters: holding the cutoffs at their period-t
    # values gives a different benefit
    p_const = _hybrid_params(f1[0], f2[0])
    const = pensions.DB_NDC_hybrid_amount(
        w, e, n, r, ndc_rates, t, j, "TPI", p_const
    )
    assert not np.allclose(hybrid, const)


def test_DB_NDC_hybrid_amount_time_varying_3d_matches_cross_sections():
    """
    With constant wages and labor supply, the full time-path array the
    aggregates use agrees with the cross section of each period, so the
    two read the time-varying cutoffs at the same periods.
    """
    p = _time_varying_hybrid_params()
    r, ndc_rates = 0.05, None
    T = p.S + 5
    w_val = 1.1
    e = p.e[-1, :, :]
    n3 = 0.4 * np.ones((T, p.S, p.J))
    hybrid3 = pensions.DB_NDC_hybrid_amount(
        np.full(T, w_val), e, n3, r, ndc_rates, 0, None, "TPI", p
    )
    assert hybrid3.shape == (T, p.S, p.J)
    for tt in (0, 7, 40, T - 1):
        hybrid2 = pensions.DB_NDC_hybrid_amount(
            np.full(p.S, w_val),
            e,
            0.4 * np.ones((p.S, p.J)),
            r,
            ndc_rates,
            tt,
            None,
            "TPI",
            p,
        )
        assert np.allclose(hybrid3[tt], hybrid2)
    # and the benefits change over time as the cutoffs do
    assert not np.allclose(hybrid3[0], hybrid3[40])


@pytest.mark.local
def test_SS_solve_hybrid_db_ndc(tmp_path):
    """
    The steady state must solve with the hybrid DB-NDC pension system
    and produce positive aggregate pension outlays.
    """
    from ogcore.execute import runner
    from ogcore import utils

    p = Specifications(baseline=True, num_workers=1)
    p.update_specifications(
        dict(
            HYBRID_UPDATES,
            pension_system="Hybrid DB-NDC",
            factor1_db_ndc_hybrid=[0.5],
            factor2_db_ndc_hybrid=[1.5],
            avg_labor_inc_t1_model_units=0.8,
        )
    )
    p.baseline_dir = p.output_base = str(tmp_path)
    runner(p, time_path=False, client=None)
    ss = utils.safe_read_pickle(str(tmp_path / "SS" / "SS_vars.pkl"))
    Y = np.asarray(ss["Y"]).sum()
    assert ss["agg_pension_outlays"] > 0.0
    assert 0.0 < ss["agg_pension_outlays"] / Y < 1.0


def test_pension_periods_conversion_and_bounds():
    """
    The calendar-year pension parameters are converted to model periods
    by rounding S/80 * years, with no shift, so the average is taken over
    ``avg_earn_num_years`` years and ``yr_contrib`` years are credited.
    A window shorter than one model period or longer than the working
    life (which would index before the start of the lifetime) raises.
    """
    p = Specifications()
    p.update_specifications(
        {
            "pension_system": "Defined Benefits",
            "alpha_db": [[0.02]],
            "yr_contrib": 40,
            "avg_earn_num_years": 35,
        }
    )
    ny, cy, S_ret = pensions.pension_periods(p)
    assert ny.shape == cy.shape == S_ret.shape == (p.T + p.S, p.S)
    assert (ny[-1, -1], cy[-1, -1], S_ret[-1, -1]) == (
        35,
        40,
        int(p.retire[-1, -1]),
    )
    n = np.linspace(0.5, 0.2, p.S)
    e = np.asarray(p.e)[-1, :, 0]
    assert np.isfinite(pensions.DB_amount(1.0, e, n, None, 0, "SS", p)).all()
    # window exactly the working life is allowed
    p.avg_earn_num_years = int(p.retire[-1, -1])
    assert np.all(pensions.pension_periods(p)[0] == int(p.retire[-1, -1]))
    # one year longer would index before the first period of life
    p.avg_earn_num_years = int(p.retire[-1, -1]) + 1
    with pytest.raises(ValueError, match="working life"):
        pensions.DB_amount(1.0, e, n, None, 0, "SS", p)
    with pytest.raises(ValueError, match="working life"):
        pensions.replacement_rate_vals(n, 1.0, 1e5, 0, p)
    # a one-year window is one period under Social Security, not 0/0
    p.avg_earn_num_years = 1
    theta = pensions.replacement_rate_vals(n, 1.0, 1e5, 0, p)
    assert np.isfinite(theta).all()
    # fewer than one model period
    p.S = 40
    with pytest.raises(ValueError, match="at least one model period"):
        pensions.pension_periods(p)


def test_DB_amount_first_doughnut_ring_t_none():
    """
    TPI.firstdoughnutring solves the household in its last period of life
    at t = 0 and passes t = None with method 'TPI_scalar'.  DB_amount must
    treat that as period 0 rather than raise on ``t - (S - per_rmn)``.
    """
    p = Specifications()
    p.update_specifications(
        {
            "pension_system": "Defined Benefits",
            "alpha_db": [[0.02]],
            "yr_contrib": 35,
            "avg_earn_num_years": 40,
        }
    )
    p.n_preTP = np.full((p.S, p.J), 0.4)
    j = 0
    w, n, e = 1.2, np.array([0.3]), np.asarray(p.e)[-1, -1:, j]
    scalar = pensions.DB_amount(w, e, n, None, j, "TPI_scalar", p)
    period0 = pensions.DB_amount(w, e, n, 0, j, "TPI_scalar", p)
    assert np.isfinite(scalar).all()
    assert np.allclose(scalar, period0)
    # the oldest household at t = 0 is paid under the period-0 rate
    alpha_path = np.full((p.T + p.S, p.J), 0.02)
    alpha_path[0, :] = 0.05
    p.alpha_db = alpha_path
    raised = pensions.DB_amount(w, e, n, None, j, "TPI_scalar", p)
    assert np.allclose(raised, period0 * 0.05 / 0.02)


def test_replacement_rate_adjustment_indexes_by_cohort_year():
    """
    On the time path the adjustment must vary with each cohort's own year.

    A single adjust[t] applied to the whole path would pass a constant-
    adjustment test and silently flatten a glide, so this checks the
    per-row offset directly.
    """
    p = Specifications()
    j, t, length = 0, 3, 5
    adjust = np.ones((p.T + p.S, p.J))
    adjust[:, j] = np.arange(p.T + p.S) + 1.0
    p.replacement_rate_adjust = adjust
    expected = np.arange(t, t + length) + 1.0

    two_d = pensions.replacement_rate_adjustment(
        np.ones((length, p.S)), t, j, "TPI", p
    )
    assert np.allclose(two_d.flatten(), expected)

    three_d = pensions.replacement_rate_adjustment(
        np.ones((length, p.S, p.J)), t, None, "TPI", p
    )
    assert np.allclose(three_d[:, 0, j], expected)

    # a 1D array is one cohort's remaining lifetime from period t, so
    # it takes the same per-year slice rather than a single adjust[t]
    one_d = pensions.replacement_rate_adjustment(
        np.ones(length), t, j, "TPI", p
    )
    assert np.allclose(one_d, expected)

    # the steady state takes the terminal value
    ss = pensions.replacement_rate_adjustment(np.ones(p.S), None, j, "SS", p)
    assert np.isclose(ss, adjust[-1, j])


def _glide_params(j, switch_year, cut, regimes=None):
    """
    Specifications with replacement_rate_adjust = 1 before switch_year
    and = cut from switch_year on, for lifetime income group j only,
    and optionally a list of pension regimes.
    """
    p = Specifications()
    if regimes is not None:
        p.update_specifications({"pension_regimes": regimes})
    adjust = np.ones((p.T + p.S, p.J))
    adjust[switch_year:, j] = cut
    p.replacement_rate_adjust = adjust
    return p


# retirement age 67 from 2030 for everyone at or below age 54 in 2030;
# older households keep the base retirement age of 65
GRANDFATHERED_67 = [
    {"year": 2030, "max_age_at_start": 54, "retirement_age": 67}
]


def test_retirement_mask_follows_each_cohort():
    """
    retirement_mask must read each household's own retirement age: on
    the household (cohort) path that is the diagonal of p.retire, and
    the cross-section masks must agree with it cell by cell.
    """
    p = Specifications()
    p.update_specifications({"pension_regimes": GRANDFATHERED_67})
    S, T = p.S, p.T
    ages = np.arange(S)
    base, raised = 44, 46  # model periods for ages 65 and 67
    assert np.unique(p.retire).tolist() == [base, raised]

    # steady state: the final row, one age earlier when shift is True
    ss = pensions.retirement_mask(None, None, False, "SS", 1, p)
    assert np.array_equal(ss, ages >= raised)
    ss_shift = pensions.retirement_mask(None, None, True, "SS", 1, p)
    assert np.array_equal(ss_shift, ages >= raised - 1)
    assert pensions.retirement_mask(None, None, False, "SS", 2, p).shape == (
        S,
        1,
    )

    # the oldest household in period 0 is retired
    assert pensions.retirement_mask(None, None, False, "TPI_scalar", 1, p)

    # a cohort born in 2032 (t = 7) is under the regime for its whole life
    born_later = pensions.retirement_mask(7, S, False, "TPI", 1, p)
    assert np.array_equal(born_later, ages >= raised)

    # households alive in period 0 solve partial lifetimes from t = 0.
    # Aged 60 in 2030 (55 in 2025, 45 periods left): above the cutoff,
    # so the base age applies. Aged 50 in 2030 (45 in 2025, 55 periods
    # left): the regime applies.
    for age_2025, expected in [(55, base), (45, raised)]:
        s0 = age_2025 - p.starting_age
        length = S - s0
        mask = pensions.retirement_mask(0, length, False, "TPI", 1, p)
        assert mask.shape == (length,)
        assert np.array_equal(mask, ages[s0:] >= expected)

    # cross sections agree with the cohort paths cell by cell
    two_d = pensions.retirement_mask(0, T, False, "TPI", 2, p)
    three_d = pensions.retirement_mask(0, T, False, "TPI", 3, p)
    assert two_d.shape == (T, S) and three_d.shape == (T, S, 1)
    assert np.array_equal(three_d[:, :, 0], two_d)
    assert np.array_equal(np.diagonal(two_d[7 : 7 + S, :]), born_later)
    s0 = 45 - p.starting_age
    assert np.array_equal(
        np.diagonal(two_d[:S, :], offset=s0),
        pensions.retirement_mask(0, S - s0, False, "TPI", 1, p),
    )
    # In 2045 (t = 20) the household aged 65 (age 50 in 2030) is under
    # the regime and not yet retired, while the one aged 66 (51 in
    # 2030) is retired. In 2029 (t = 4) age 65 is retired under the
    # base age.
    assert not two_d[20, base] and two_d[20, raised]
    assert two_d[4, base]

    with pytest.raises(ValueError, match="method"):
        pensions.retirement_mask(0, S, False, "nope", 1, p)


def test_household_path_pension_uses_year_of_payment():
    """
    On the household (cohort) path the replacement rate adjustment must
    be the factor for the year each benefit is paid, applied once.

    TPI used to pre-scale theta by the factor before passing it to the
    household problem, and SS_amount then scaled it again. For a uniform
    cut the household saw the factor squared. For a glide path it saw
    the factor for its entry year, so a cohort entering before the cut
    planned on full benefits that the aggregates never paid.
    """
    j, t, theta_j, cut = 2, 5, 0.4, 0.5
    switch_year = 30
    p = _glide_params(j, switch_year, cut)
    w = np.ones(p.S)
    n = 0.3 * np.ones(p.S)
    pension = pensions.pension_amount(
        np.full(p.S, 0.04),
        w,
        n,
        1.0,
        theta_j,
        t,
        j,
        False,
        "TPI",
        p.e[t, :, j],
        1.0,
        p,
    )
    cohort_retire = np.diagonal(p.retire[t : t + p.S, :])
    assert np.all(cohort_retire == cohort_retire[0])
    retire = int(cohort_retire[0])
    years_paid = t + np.arange(retire, p.S)
    expected = theta_j * p.replacement_rate_adjust[years_paid, j] * w[retire:]
    assert np.all(expected[years_paid >= switch_year] == theta_j * cut)
    assert np.all(expected[years_paid < switch_year] == theta_j)
    assert np.allclose(pension[:retire], 0.0)
    assert np.allclose(pension[retire:], expected)

    # the same holds through net_taxes, which is what the FOCs call
    z = np.zeros(p.S)
    num_params = len(p.etr_params[0][0])
    etr = np.array(
        [
            [p.etr_params[t + s][s][i] for i in range(num_params)]
            for s in range(p.S)
        ]
    )

    def net_tax(p_):
        return tax.net_taxes(
            np.full(p.S, 0.04),
            w,
            np.full(p.S, 0.5),
            n,
            z,
            1.0,
            z,
            z,
            theta_j,
            t,
            j,
            False,
            "TPI",
            p.e[t, :, j],
            etr,
            p_,
        )

    p_full = _glide_params(j, switch_year, 1.0)
    multiplier = 1 - (net_tax(p) - net_tax(p_full))[retire:] / theta_j
    assert np.allclose(multiplier, p.replacement_rate_adjust[years_paid, j])


@pytest.mark.parametrize(
    "regimes", [None, GRANDFATHERED_67], ids=["no regime", "grandfathered"]
)
def test_household_path_pension_matches_pension_mat(regimes):
    """
    The pension a cohort faces in its own household problem must equal the
    diagonal of the (T, S, J) pension array that TPI saves and aggregates,
    both for a cohort born along the path and for one alive at t = 0,
    whether or not a regime changes the retirement age by cohort.
    """
    j = 1
    p = _glide_params(j, 20, 0.6, regimes)
    S, T = p.S, p.T
    rng = np.random.default_rng(0)
    w_path = 1.0 + 0.1 * rng.random(T + S)
    n_path = 0.2 + 0.3 * rng.random((T + S, S, p.J))
    theta = 0.3 + 0.1 * np.arange(p.J)

    pension_mat = pensions.pension_amount(
        np.full(T, 0.04),
        w_path[:T],
        n_path[:T, :, :],
        1.0,
        theta,
        0,
        None,
        False,
        "TPI",
        p.e,
        1.0,
        p,
    )

    def cohort_pension(t, length):
        return pensions.pension_amount(
            np.full(length, 0.04),
            w_path[t : t + length],
            np.diag(n_path[t : t + S, :, j], S - length),
            1.0,
            theta[j],
            t,
            j,
            False,
            "TPI",
            p.e[t, :, j],
            1.0,
            p,
        )

    # a cohort born in period 7, and one aged 45 in 2025 (35 periods in,
    # 45 left) that the grandfathered regime does and does not cover
    first_retired = []
    for t, length in [(7, S), (0, S - 25), (0, S - 35)]:
        cohort = cohort_pension(t, length)
        expected = np.diagonal(pension_mat[t : t + S, :, j], S - length)
        assert np.any(cohort > 0.0), "vacuous with a zero benefit"
        assert np.allclose(cohort, expected)
        first_retired.append(S - length + np.argmax(cohort > 0.0))
    if regimes is None:
        assert first_retired == [44, 44, 44]
    else:
        # born in 2032 and aged 50 in 2030 retire at 67; aged 60 in 2030
        # keeps 65
        assert first_retired == [46, 46, 44]


@pytest.mark.local
def test_SS_solve_notional_defined_contribution(tmp_path):
    """
    The steady state must solve with the Notional Defined Contribution
    pension system and produce positive aggregate pension outlays (the
    NDC counterpart to test_SS_solve_defined_benefits, Issue #1014).
    """
    from ogcore.execute import runner
    from ogcore import utils

    p = Specifications(baseline=True, num_workers=1)
    p.update_specifications(
        {
            "pension_system": "Notional Defined Contribution",
            "tau_p": 0.1,
            "ndc_growth_rate": "LR GDP",
            "dir_growth_rate": "r",
        }
    )
    p.baseline_dir = p.output_base = str(tmp_path)
    runner(p, time_path=False, client=None)
    ss = utils.safe_read_pickle(str(tmp_path / "SS" / "SS_vars.pkl"))
    Y = np.asarray(ss["Y"]).sum()
    assert ss["agg_pension_outlays"] > 0.0
    assert 0.0 < ss["agg_pension_outlays"] / Y < 1.0


@pytest.mark.local
def test_SS_solve_points_system(tmp_path):
    """
    The steady state must solve with the Points System pension system
    and produce positive aggregate pension outlays (the Points System
    counterpart to test_SS_solve_defined_benefits, Issue #1014).
    """
    from ogcore.execute import runner
    from ogcore import utils

    p = Specifications(baseline=True, num_workers=1)
    p.update_specifications(
        {
            "pension_system": "Points System",
            "vpoint": 0.5,
        }
    )
    p.baseline_dir = p.output_base = str(tmp_path)
    runner(p, time_path=False, client=None)
    ss = utils.safe_read_pickle(str(tmp_path / "SS" / "SS_vars.pkl"))
    Y = np.asarray(ss["Y"]).sum()
    assert ss["agg_pension_outlays"] > 0.0
    assert 0.0 < ss["agg_pension_outlays"] / Y < 1.0


# A reform that moves cohorts aged 54 or younger in 2030 to a later
# retirement age and new DB and NDC parameters; older households keep
# the base system
DB_NDC_REFORM = [
    {
        "year": 2030,
        "max_age_at_start": 54,
        "retirement_age": 67,
        "alpha_db": 0.015,
        "avg_earn_num_years": 45,
        "yr_contrib": 38,
        "tau_p": 0.15,
        "k_ret": 0.3,
        "indR": 0.1,
    }
]


def _full_path_and_cohort_pensions(p, j, T_path, ndc_rates=None):
    """
    Pension benefits for the full (T_path, S, J) time path and, for a
    set of cohorts, from the household's own lifetime path, with a
    constant wage and labor supply so that both read the same history.
    """
    S = p.S
    p.n_preTP = np.full((S, p.J), 0.4)
    n3 = np.full((T_path, S, p.J), 0.4)
    w = np.full(T_path, 1.2)
    theta = np.zeros(p.J)
    full = pensions.pension_amount(
        np.full(T_path, 0.05),
        w,
        n3,
        ndc_rates,
        theta,
        0,
        None,
        False,
        "TPI",
        p.e,
        1e5,
        p,
    )
    cohorts = {}
    # households alive at t = 0 of age a (S - a periods remaining)
    for a in [0, 10, 25, 30, 35, 50, 60, 70, S - 1]:
        per_rmn = S - a
        cohorts[("alive", a)] = pensions.pension_amount(
            np.full(per_rmn, 0.05),
            np.full(per_rmn, 1.2),
            0.4 * np.ones(per_rmn),
            ndc_rates,
            theta[j],
            0,
            j,
            False,
            "TPI",
            p.e[-1, a:, j],
            1e5,
            p,
        )
    # households born in period t0 with full lifetimes
    for t0 in [0, 5, 20]:
        cohorts[("born", t0)] = pensions.pension_amount(
            np.full(S, 0.05),
            np.full(S, 1.2),
            0.4 * np.ones(S),
            ndc_rates,
            theta[j],
            t0,
            j,
            False,
            "TPI",
            p.e[-1, :, j],
            1e5,
            p,
        )
    return full, cohorts


@pytest.mark.parametrize(
    "system,updates",
    [
        (
            "Defined Benefits",
            {"alpha_db": [[0.02]], "yr_contrib": 35, "avg_earn_num_years": 40},
        ),
        (
            "Notional Defined Contribution",
            {
                "tau_p": 0.1,
                "ndc_growth_rate": "LR GDP",
                "dir_growth_rate": "r",
            },
        ),
    ],
    ids=["DB", "NDC"],
)
@pytest.mark.parametrize(
    "regimes", [None, DB_NDC_REFORM], ids=["base", "reform"]
)
def test_db_ndc_household_path_matches_full_path(system, updates, regimes):
    """
    Under the DB and NDC systems the benefit computed on a household's
    own lifetime path (the partial- and full-lifetime branches used in
    the household solves) must equal the diagonal of the (T, S, J)
    array used for the aggregates, both with the base parameters and
    under a grandfathered reform that changes the retirement age and
    the system parameters by cohort. Each cohort's first benefit must
    arrive at its own retirement age.
    """
    p = Specifications()
    revision = dict(updates, pension_system=system)
    if regimes is not None:
        revision["pension_regimes"] = regimes
    p.update_specifications(revision)
    j = 0
    S, T_path = p.S, 120
    full, cohorts = _full_path_and_cohort_pensions(p, j, T_path)
    assert full.shape == (T_path, S, p.J)
    assert np.all(full >= 0.0)
    for (kind, k), cohort in cohorts.items():
        if kind == "alive":
            a, t0 = k, 0
        else:
            a, t0 = 0, k
        length = len(cohort)
        on_path = min(length, T_path - t0)
        expected = np.array(
            [full[t0 + tt, a + tt, j] for tt in range(on_path)]
        )
        assert np.allclose(cohort[:on_path], expected), (kind, k)
        assert np.any(cohort > 0.0), (kind, k)
        first = a + np.argmax(cohort > 0.0)
        # the first age at which the cohort is at or past its own
        # retirement period, read down its diagonal of p.retire
        ages = np.arange(a, S)
        own = np.diagonal(p.retire[t0 : t0 + S, :], offset=a)
        assert first == ages[np.argmax(ages >= own)], (kind, k, first)
    if regimes is None:
        # everyone retires at 65 (period 44)
        assert np.all(p.retire == 44)
        assert np.argmax(cohorts[("alive", 30)] > 0.0) + 30 == 44
    else:
        # households aged 55 and 60 in 2030 (ages 30 and 35 model
        # periods in at t = 0) keep 65; aged 50 in 2030 and all later
        # cohorts move to 67
        assert np.argmax(cohorts[("alive", 35)] > 0.0) + 35 == 44
        assert np.argmax(cohorts[("alive", 30)] > 0.0) + 30 == 44
        assert np.argmax(cohorts[("alive", 25)] > 0.0) + 25 == 46
        assert np.argmax(cohorts[("born", 20)] > 0.0) == 46
        # the reform's parameters are in force for the new cohorts
        t = 2030 - p.start_year
        row = full[t + 30, :, j]
        assert row[46] > 0.0 and row[45] == 0.0


def test_db_regime_changes_benefit_by_cohort():
    """
    With a constant history, a regime that only changes the DB formula
    for cohorts at or below the cutoff age scales their benefits and
    leaves older cohorts' benefits unchanged.
    """
    base = Specifications()
    base.update_specifications(
        {
            "pension_system": "Defined Benefits",
            "alpha_db": [[0.02]],
            "yr_contrib": 40,
            "avg_earn_num_years": 40,
        }
    )
    reform = Specifications()
    reform.update_specifications(
        {
            "pension_system": "Defined Benefits",
            "alpha_db": [[0.02]],
            "yr_contrib": 40,
            "avg_earn_num_years": 40,
            "pension_regimes": [
                {
                    "year": 2030,
                    "max_age_at_start": 54,
                    "alpha_db": 0.01,
                    "yr_contrib": 30,
                }
            ],
        }
    )
    j, T_path = 0, 120
    full_base, _ = _full_path_and_cohort_pensions(base, j, T_path)
    full_reform, _ = _full_path_and_cohort_pensions(reform, j, T_path)
    t = 2030 - base.start_year
    E = base.starting_age
    # retirees aged 65 and over in 2030 were older than 54 and are paid
    # the base benefit in every later year, the covered cohorts get
    # 0.01 * 30 / (0.02 * 40) of it once they retire
    for tt in range(t, T_path):
        row_b, row_r = full_base[tt, :, j], full_reform[tt, :, j]
        cutoff_age = 54 - E + (tt - t)  # oldest covered age in period tt
        covered = np.arange(base.S) <= cutoff_age
        retired = row_b > 0.0
        assert np.allclose(
            row_r[retired & ~covered], row_b[retired & ~covered]
        )
        assert np.allclose(
            row_r[retired & covered], row_b[retired & covered] * 0.375
        )
    assert np.any(full_reform[-1, :, j] > 0.0)


def test_delta_ret_by_cohort():
    """
    delta_ret returns the steady-state coefficient by default and an
    array shaped like the retirement periods it is given, with the
    survivor and payment adjustments applied cell by cell.
    """
    p = Specifications()
    p.update_specifications(
        {
            "pension_system": "Notional Defined Contribution",
            "tau_p": 0.1,
            "dir_growth_rate": "LR GDP",
        }
    )
    ss = pensions.delta_ret(0.05, 1.0, p)
    assert isinstance(ss, float) and ss > 0.0
    S_ret = np.array([[44, 46], [46, 44]])
    out = pensions.delta_ret(0.05, 1.0, p, S_ret, 0.0, 0.0)
    assert out.shape == (2, 2)
    assert out[0, 0] == out[1, 1] == ss
    assert out[0, 1] == out[1, 0] > ss  # fewer years of payment
    # the adjustments shift the denominator; a repeated call gives the
    # same values (the survival rates are not modified in place)
    adj = pensions.delta_ret(0.05, 1.0, p, S_ret, 0.1, 0.3)
    assert np.allclose(1.0 / adj - 1.0 / out, 0.1 - 0.3)
    assert np.allclose(pensions.delta_ret(0.05, 1.0, p, S_ret, 0.0, 0.0), out)


def test_delta_ret_loop_survival_timing():
    """
    The annuity factor counts the payment at retirement age R with
    certainty and the payment at age R + k with the probability of
    surviving ages R, ..., R + k - 1 (1 - rho[s] is the probability of
    surviving from age s to s + 1). With two payment periods, 20%
    mortality after the first, certain death after the second, and no
    discounting, the factor is 1 + 0.8 (Issue #1268).
    """
    surv_rates = 1.0 - np.array([0.0, 0.0, 0.2, 1.0])
    assert np.isclose(pensions.delta_ret_loop(4, 2, surv_rates, 0.0), 1.8)
    # general case against the closed form
    S, R, g = 10, 6, 0.03
    mort = np.linspace(0.05, 1.0, S)
    expected = sum(
        np.prod(1.0 - mort[R : R + k]) / (1.0 + g) ** k for k in range(S - R)
    )
    assert np.isclose(pensions.delta_ret_loop(S, R, 1.0 - mort, g), expected)


def _time_varying_mortality(p):
    """
    Mortality rates that are lower early in the transition and converge
    to the (unchanged) steady-state rates by period 60.
    """
    scale = np.ones(p.T + p.S)
    scale[:60] = np.linspace(0.5, 1.0, 60)
    rho = p.rho[-1][None, :, :] * scale[:, None, None]
    rho[:, -1, :] = 1.0
    return rho


@pytest.mark.parametrize("method", ["steady state", "period", "cohort"])
def test_delta_ret_ndc_mortality(method):
    """
    ndc_mortality chooses the mortality rates in the annuity factor:
    the steady-state rates, the rates of the period of retirement, or
    the rates the retiring cohort faces as it ages. All three agree in
    the steady state.
    """
    p = Specifications()
    p.update_specifications(
        {
            "pension_system": "Notional Defined Contribution",
            "tau_p": 0.1,
            "dir_growth_rate": "LR GDP",
            "ndc_mortality": method,
        }
    )
    p.rho = _time_varying_mortality(p)
    lam = np.asarray(p.lambdas).ravel()
    mort = p.rho @ lam
    R, r, S = 44, 0.05, p.S
    g = pensions.g_dir(r, 1.0, p)
    cohort_k = float(pensions.cohort_array("k_ret", p)[-1, -1])
    k = cohort_k
    indR = float(pensions.cohort_array("indR", p)[-1, -1])

    def delta(surv):
        return 1.0 / (pensions.delta_ret_loop(S, R, surv, g) + indR - k)

    ss = delta(1.0 - mort[-1])
    # the steady state is the same under every option
    assert np.isclose(pensions.delta_ret(r, 1.0, p, R), ss)
    assert np.isclose(pensions.delta_ret(r, 1.0, p, R, t_ret=None), ss)
    t_ret = np.array([-50, -10, 0, 20, 45, 300, 500])
    out = pensions.delta_ret(
        r, 1.0, p, np.full(t_ret.shape, R), indR, cohort_k, t_ret
    )
    ages = np.arange(S)
    n_rows = p.T + p.S
    for tR, val in zip(t_ret, out):
        if method == "steady state":
            surv = 1.0 - mort[-1]
        elif method == "period":
            surv = 1.0 - mort[np.clip(tR, 0, n_rows - 1)]
        else:
            rows = np.clip(tR - R + ages, 0, n_rows - 1)
            surv = 1.0 - mort[rows, ages]
        assert np.isclose(val, delta(surv)), (method, tR)
    # cohorts retiring once mortality has converged get the
    # steady-state coefficient
    assert np.isclose(out[-1], ss)
    if method == "steady state":
        assert np.allclose(out, ss)
    else:
        # lower mortality early in the path means a longer expected
        # payout and a smaller coefficient
        assert out[2] < ss
    if method == "cohort":
        # a cohort retiring at period 20 lives through the converging
        # rates, so its survival is lower than the period-20 table
        p_period = Specifications()
        p_period.update_specifications({"ndc_mortality": "period"})
        p_period.rho = p.rho
        p_period.dir_growth_rate = p.dir_growth_rate
        assert out[3] > pensions.delta_ret(
            r, 1.0, p_period, R, indR, cohort_k, 20
        )


@pytest.mark.parametrize("method", ["period", "cohort"])
def test_ndc_household_path_matches_full_path_mortality(method):
    """
    With mortality that changes over the transition path, the NDC
    benefit computed on a household's own lifetime path must still
    equal the diagonal of the (T, S, J) array used for the aggregates,
    so that every branch of NDC_amount assigns the same retirement
    period to the household.
    """
    p = Specifications()
    p.update_specifications(
        {
            "pension_system": "Notional Defined Contribution",
            "tau_p": 0.1,
            "ndc_growth_rate": "LR GDP",
            "dir_growth_rate": "r",
            "ndc_mortality": method,
            "pension_regimes": DB_NDC_REFORM,
        }
    )
    p.rho = _time_varying_mortality(p)
    j, T_path = 0, 120
    full, cohorts = _full_path_and_cohort_pensions(p, j, T_path)
    # the coefficient varies across cohorts retiring at the same age
    assert not np.isclose(full[0, 50, j], full[T_path - 1, 50, j])
    for (kind, k), cohort in cohorts.items():
        a, t0 = (k, 0) if kind == "alive" else (0, k)
        on_path = min(len(cohort), T_path - t0)
        expected = np.array(
            [full[t0 + tt, a + tt, j] for tt in range(on_path)]
        )
        assert np.allclose(cohort[:on_path], expected), (kind, k)


def _ndc_params(**updates):
    p = Specifications()
    p.update_specifications(
        dict(
            {
                "pension_system": "Notional Defined Contribution",
                "tau_p": 0.1,
            },
            **updates,
        )
    )
    return p


def test_growth_rate_path_units():
    """
    The NDC growth rates are real growth realized from t - 1 to t,
    recovered from stationarized aggregates with productivity growth
    exp(g_y) and population growth g_n[t - 1] (g_n_preTP into period
    0). In the steady state GDP and the wage bill grow at
    exp(g_y)(1 + g_n_ss) - 1 and the average wage at exp(g_y) - 1.
    """
    p = _ndc_params()
    n_per = p.T + p.S
    g_y, g_n_ss = p.g_y, p.g_n_ss
    lr = np.exp(g_y) * (1 + g_n_ss) - 1
    for choice, expected in [
        ("LR GDP", lr),
        ("Curr GDP", lr),
        ("wage bill", lr),
        ("avg wage", np.exp(g_y) - 1),
        ("r", 0.04),
    ]:
        ss = pensions.growth_rate_path(choice, 0.04, 1.0, 1.0, 1.0, p, "SS")
        assert ss.shape == (1,) and np.isclose(ss[0], expected), choice
    rng = np.random.default_rng(0)
    Y = 1 + 0.1 * rng.random(p.T)  # a path that stops at T
    w = 1 + 0.1 * rng.random(n_per)
    L = 1 + 0.1 * rng.random(n_per)
    r = 0.03 + 0.01 * rng.random(n_per)
    g_n_prev = np.append(p.g_n_preTP, p.g_n[: n_per - 1])
    Y_ext = np.append(Y, np.full(p.S, Y[-1]))
    Y_prev = np.append(Y_ext[0], Y_ext[:-1])
    wL = w * L
    wL_prev = np.append(wL[0], wL[:-1])
    expected = {
        "Curr GDP": Y_ext / Y_prev * np.exp(g_y) * (1 + g_n_prev) - 1,
        "wage bill": wL / wL_prev * np.exp(g_y) * (1 + g_n_prev) - 1,
        "avg wage": wL / wL_prev * np.exp(g_y) - 1,
        "r": r,
        "LR GDP": np.array([lr]),
    }
    for choice, exp_path in expected.items():
        path = pensions.growth_rate_path(choice, r, Y, w, L, p, "TPI")
        assert np.allclose(path, exp_path), choice
    # constant stationarized aggregates give the long-run rates
    ones = np.ones(n_per)
    path = pensions.growth_rate_path("Curr GDP", r, ones, w, L, p, "TPI")
    assert np.allclose(path, np.exp(g_y) * (1 + g_n_prev) - 1)


def test_ndc_rates_path_averaging():
    """
    ndc_growth_avg_years averages the credited rate over that many
    years (model periods with S = 80) up to and including each period,
    padding the years before the path with the period-0 rate; the
    discount rate is not averaged.
    """
    p = _ndc_params(
        ndc_growth_rate="r", dir_growth_rate="r", ndc_growth_avg_years=5
    )
    n_per = p.T + p.S
    r = 0.03 + 0.02 * np.sin(np.arange(n_per) / 7.0)
    rates = pensions.ndc_rates_path(r, None, None, None, p, "TPI")
    assert isinstance(rates, pensions.NDCRates)
    assert np.allclose(rates.discount, r)
    padded = np.append(np.full(4, r[0]), r)
    expected = np.array([padded[t : t + 5].mean() for t in range(n_per)])
    assert np.allclose(rates.credit, expected)
    p.ndc_growth_avg_years = 1
    rates = pensions.ndc_rates_path(r, None, None, None, p, "TPI")
    assert np.allclose(rates.credit, r)


def test_NDC_1dim_loop_time_varying_credit():
    """
    A contribution made at age s compounds at the rates credited in
    periods t_birth + s + 1 through t_birth + R - 1.
    """
    S, R, t_birth = 6, 4, 3
    w = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    e = np.array([1.0, 1.2, 1.1, 0.9, 0.8, 0.7])
    n = np.array([0.4, 0.5, 0.45, 0.4, 0.2, 0.1])
    tau_p = np.full(S, 0.2)
    g_y = np.array([0.02])
    credit = 0.01 * np.arange(1, 16)  # rate credited in period t
    delta = np.full(S, 0.3)
    out = pensions.NDC_1dim_loop(
        w,
        e,
        n,
        np.full(S, R),
        S,
        g_y,
        tau_p,
        credit,
        t_birth,
        delta,
        np.zeros(S),
    )
    for u in range(S):
        if u < R:
            assert out[u] == 0.0
            continue
        expected = 0.3 * sum(
            0.2
            * w[s]
            * np.exp(-0.02 * (u - s))
            * e[s]
            * n[s]
            * np.prod(1 + credit[t_birth + s + 1 : t_birth + R])
            for s in range(R)
        )
        assert np.isclose(out[u], expected)


def test_delta_ret_discount_at_retirement():
    """
    Given a path of discount rates, each household's conversion
    coefficient discounts all payments at the rate of its retirement
    period (clipped to the path).
    """
    p = _ndc_params(dir_growth_rate="r", ndc_mortality="steady state")
    R = 44
    path = np.linspace(0.02, 0.06, 50)
    t_ret = np.array([-3, 0, 10, 49, 80])
    out = pensions.delta_ret(
        0.0, None, p, np.full(5, R), t_ret=t_ret, g_dir_path=path
    )
    for tR, val in zip(t_ret, out):
        g = path[np.clip(tR, 0, 49)]
        assert np.isclose(val, pensions.delta_ret(g, None, p, R)), tR
    # no path: the steady-state rate, here the last element of r
    assert np.isclose(
        pensions.delta_ret(np.array([0.1, 0.05]), None, p, R, t_ret=10),
        pensions.delta_ret(0.05, None, p, R),
    )


@pytest.mark.parametrize("choice", ["Curr GDP", "wage bill", "avg wage"])
def test_NDC_steady_state_growth_options(choice):
    """
    Every growth option works in the steady state (the 'Curr GDP'
    option used to return an array and fail), with or without the
    rates passed explicitly.
    """
    p = _ndc_params(ndc_growth_rate=choice, dir_growth_rate=choice)
    w = 1.2
    n = np.full((p.S, p.J), 0.4)
    e = p.e[-1]
    out = pensions.NDC_amount(w, e, n, 0.05, None, None, None, "SS", p)
    rates = pensions.ndc_rates_path(0.05, 1.0, w, 1.0, p, "SS")
    out2 = pensions.NDC_amount(w, e, n, 0.05, rates, None, None, "SS", p)
    assert np.allclose(out, out2)
    assert np.all(np.isfinite(out)) and np.all(out[-1] > 0)
    retired = np.argmax(out[:, 0] > 0)
    assert np.all(out[:retired] == 0)


@pytest.mark.parametrize(
    "credit_choice,dir_choice", [("Curr GDP", "r"), ("wage bill", "avg wage")]
)
def test_ndc_household_path_matches_full_path_rates(credit_choice, dir_choice):
    """
    With credited and discount rates that change over the transition
    path, the NDC benefit computed on a household's own lifetime path
    equals the diagonal of the (T, S, J) array used for the aggregates,
    and differs from the benefit under constant rates.
    """
    p = _ndc_params(
        ndc_growth_rate=credit_choice,
        dir_growth_rate=dir_choice,
        ndc_growth_avg_years=3,
        pension_regimes=DB_NDC_REFORM,
    )
    n_per = p.T + p.S
    t = np.arange(n_per)
    Y = 1 + 0.2 * np.exp(-t / 30.0) * np.sin(t / 5.0)
    w = 1 + 0.1 * np.cos(t / 9.0)
    L = 1 + 0.05 * np.sin(t / 4.0)
    r = 0.04 + 0.02 * np.exp(-t / 40.0)
    rates = pensions.ndc_rates_path(r, Y, w, L, p, "TPI")
    assert np.ptp(rates.credit) > 0 and np.ptp(rates.discount) > 0
    j, T_path = 0, 120
    full, cohorts = _full_path_and_cohort_pensions(p, j, T_path, rates)
    for (kind, k), cohort in cohorts.items():
        a, t0 = (k, 0) if kind == "alive" else (0, k)
        on_path = min(len(cohort), T_path - t0)
        expected = np.array(
            [full[t0 + tt, a + tt, j] for tt in range(on_path)]
        )
        assert np.allclose(cohort[:on_path], expected), (kind, k)
    const, _ = _full_path_and_cohort_pensions(p, j, T_path, None)
    assert not np.allclose(full[:60], const[:60])


def test_db_cohort_inputs_cache():
    """
    The Defined Benefits parameters along a household's lifetime are
    cached on the parameters object while the household problem solves.
    The cached values must equal the ones read directly, and replacing
    one of the arrays they come from must empty the cache.
    """
    p = Specifications()
    p.S = 7
    p.J = 2
    p.T = 12
    p.retire = 4
    p.avg_earn_num_years = 50
    p.yr_contrib = 55
    p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.2
    p.e = np.tile(
        np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6]).reshape(1, 7, 1),
        (p.T, 1, 2),
    )
    ny, cy, S_ret = pensions.pension_periods(p)
    alpha_db = pensions.cohort_array("alpha_db", p, by_j=True)
    for t, per_rmn, j in [(0, 3, 1), (2, 7, 0), (5, 7, 1)]:
        cached = pensions._db_cohort_inputs(p, t, per_rmn, j)
        expected = (
            pensions.cohort_path(S_ret, t, per_rmn, p),
            pensions.cohort_path(ny, t, per_rmn, p),
            pensions.cohort_path(cy, t, per_rmn, p),
            pensions.cohort_path(alpha_db, t, per_rmn, p)[:, j],
            pensions.cohort_path(p.e, t, per_rmn, p)[:, j],
        )
        for c, e in zip(cached, expected):
            assert np.array_equal(c, e)
        # a second call returns the cached arrays
        again = pensions._db_cohort_inputs(p, t, per_rmn, j)
        assert all(a is b for a, b in zip(again, cached))
    p.alpha_db = np.ones((p.T + p.S, p.J)) * 0.3
    replaced = pensions._db_cohort_inputs(p, 0, 3, 1)
    assert np.allclose(replaced[3], 0.3)
