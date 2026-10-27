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
p.retire = [3, 3, 3, 3, 3, 3, 3, 3]
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
p1.retire = [3, 3, 3, 3, 3, 3, 3, 3]
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
L_inc_avg = np.zeros(0)
L_inc_avg_s = np.zeros(equiv_periods)
DB = np.zeros(p.S)
DB_loop_expected1 = np.array(
    [0, 0, 0, 0, 0.337864778, 0.327879365, 0.318189065]
)

args1 = (
    w,
    e,
    n,
    p.retire,
    p.S,
    p.g_y,
    L_inc_avg_s,
    L_inc_avg,
    DB,
    equiv_periods,
    np.full(p.S, 0.2),  # rate in effect at each age u
    equiv_contrib_periods,
)

test_data = [(args1, DB_loop_expected1)]


@pytest.mark.parametrize(
    "args,DB_loop_expected", test_data, ids=["SS/Complete"]
)
def test_DB_1dim_loop(args, DB_loop_expected):
    """
    Test of the pensions.DB_1dim_loop() function.
    """

    (
        w,
        e,
        n,
        S_ret,
        S,
        g_y,
        L_inc_avg_s,
        L_inc_avg,
        DB,
        last_career_yrs,
        rep_rate_py,
        yr_contrib,
    ) = args
    DB_loop = pensions.DB_1dim_loop(
        w,
        e,
        n,
        S_ret,
        S,
        g_y,
        L_inc_avg_s,
        L_inc_avg,
        DB,
        last_career_yrs,
        rep_rate_py,
        yr_contrib,
    )
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
p.g_n_SS = 0.0
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
p.tau_p = 0.3
p.k_ret = 0.4615
p.mort_rates_SS = np.array([0.01, 0.05, 0.3, 1])
w = np.array([1.2, 1.1, 1.21, 1])
e = np.array([1.1, 1.11, 0.9, 0.87])
r = np.ones(p.T) * 0.02
d_NDC_expected1 = np.array([0.757437326, 0.680222841, 0.0, 0.0])
args1 = (r, w, e, None, per_rmn, p)

#############Incomplete lifetimes###################
p2 = Specifications()
p2.S = 4
p2.retire = 2
per_rmn2 = 3
p2.g_y = np.ones(p2.T) * 0.04
p2.g_n = np.ones(p2.T) * 0.0
p2.g_n_SS = 0.0
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "LR GDP"
p2.tau_p = 0.3
p2.k_ret = 0.4615
p2.mort_rates_SS = np.array([0.1, 0.2, 0.4, 0.6, 1.0])
w2 = np.array([1.1, 1.21, 1.25])
e2 = np.array([1.11, 0.9, 1.0])
r2 = np.ones(p2.T) * 0.04
d_NDC_expected2 = np.array([0.396808466, 0.0, 0.0])
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
p2.g_n_SS = 0.0
r = np.ones(p2.T) * 0.03
factor = 2
p2.tau_p = 0.3
p2.k_ret = 0.4615
p2.mort_rates_SS = np.array([0.01, 0.05, 0.3, 0.4, 1])
omegas = None
lambdas = 1
pension_expected_ndc = [0, 0, 0.27992856, 0.27165542]
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
p4.retire = (np.ones(p4.T) * 4).astype(int)
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
p.g_n_SS = 0.0
p.g_y = np.ones(p.T) * np.ones(p.T) * 0.03
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
w = np.array([1.2, 1.1, 1.21])
e = np.array([1.1, 1.11, 0.9])
r = np.ones(p.T) * 0.02
p.tau_p = 0.3
p.k_ret = 0.4615
delta_ret = 1.857010214
g_ndc = p.g_y[-1] + p.g_n_SS
p.mort_rates_SS = np.array([0.1, 0.2, 0.4, 1.0])
deriv_NDC_loop_expected1 = np.array([0.757437326, 0.680222841, 0.0])
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
p2.g_n_SS = 0.0
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "r"
r = np.array([0.03, 0.04, 0.03])
p2.tau_p = 0.3
p2.k_ret = 0.4615
delta_ret2 = 1.083288196
g_ndc2 = p2.g_y[-1] + p2.g_n_SS
p2.mort_rates_SS = np.array([0.2, 0.4, 1.0])
deriv_NDC_loop_expected2 = np.array([0.396808466, 0.0, 0.0])
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
p.g_n_SS = 0.0
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
r = 0.02
g_ndc_amount = p.g_y[-1] + p.g_n_SS
p.mort_rates_SS = np.array([0.1, 0.2, 0.4, 0.6, 1.0])
dir_delta_s_empty = np.zeros(p.S - p.retire + 1)
dir_delta_ret_expected1 = 1.384615385
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
    dir_delta = pensions.delta_ret_loop(
        p.S, p.retire, surv_rates, g_ndc_value, dir_delta_s_empty
    )

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
p2.g_n_SS = 0.0
p2.ndc_growth_rate = "LR GDP"
p2.dir_growth_rate = "r"
r_dndc = np.ones(p2.T) * 0.02
p2.tau_p = 0.3
p2.k_ret = 0.4615
p2.mort_rates_SS = np.array([0.01, 0.05, 0.3, 1])
d_theta_expected_dndc = np.array([0.75838653, 0.680222841, 0, 0])
# TODO: has to change first element from the below to above. Why?
# check by hand calculation spreadsheet
# d_theta_expected_dndc = np.array([0.757437326, 0.680222841, 0, 0])
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
p.g_n_SS = 0.0
p.ndc_growth_rate = "LR GDP"
p.dir_growth_rate = "r"
r = np.ones(p.T) * 0.03
p.tau_p = 0.3
p.k_ret = 0.4615
p.mort_rates_SS = np.array([0.01, 0.05, 0.3, 0.4, 1])
NDC_expected1 = np.array([0, 0, 0.27992856, 0.27165542])
# NDC_expected1 = np.array([0, 0, 0.279756794, 0.271488732])
args1 = (w, e, n, r, None, j, p)

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
p2.g_n_SS = 0.0
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
NDC_expected2 = np.array([0, 0.25705613, 0.24945897])
args2 = (w, e, n, r, None, j, p2)

test_data = [(args1, NDC_expected1), (args2, NDC_expected2)]


@pytest.mark.parametrize(
    "args,NDC_expected", test_data, ids=["SS/Complete", "Incomplete"]
)
def test_NDC(args, NDC_expected):
    """
    Test of the pensions.NDC() function.
    """
    w, e, n, r, Y, j, p = args
    NDC = pensions.NDC_amount(w, e, n, r, Y, j, p)
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
    ],
    ids=["DB", "PS", "NDC"],
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
    S_ret = int(p.retire[-1])
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
    ],
    ids=["SS", "DB", "PS", "NDC"],
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
    assert (ny, cy, S_ret) == (35, 40, int(p.retire[-1]))
    n = np.linspace(0.5, 0.2, p.S)
    e = np.asarray(p.e)[-1, :, 0]
    assert np.isfinite(pensions.DB_amount(1.0, e, n, None, 0, "SS", p)).all()
    # window exactly the working life is allowed
    p.avg_earn_num_years = int(p.retire[-1])
    assert pensions.pension_periods(p)[0] == int(p.retire[-1])
    # one year longer would index before the first period of life
    p.avg_earn_num_years = int(p.retire[-1]) + 1
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


def _glide_params(j, switch_year, cut):
    """
    Specifications with replacement_rate_adjust = 1 before switch_year
    and = cut from switch_year on, for lifetime income group j only.
    """
    p = Specifications()
    adjust = np.ones((p.T + p.S, p.J))
    adjust[switch_year:, j] = cut
    p.replacement_rate_adjust = adjust
    return p


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
    retire = p.retire[t]
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


def test_household_path_pension_matches_pension_mat():
    """
    The pension a cohort faces in its own household problem must equal the
    diagonal of the (T, S, J) pension array that TPI saves and aggregates.
    """
    j, t = 1, 7
    p = _glide_params(j, 20, 0.6)
    rng = np.random.default_rng(0)
    w_path = 1.0 + 0.1 * rng.random(p.T + p.S)
    n_path = 0.2 + 0.3 * rng.random((p.T + p.S, p.S, p.J))
    theta = 0.3 + 0.1 * np.arange(p.J)

    pension_mat = pensions.pension_amount(
        np.full(p.T, 0.04),
        w_path[: p.T],
        n_path[: p.T, :, :],
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
    cohort = pensions.pension_amount(
        np.full(p.S, 0.04),
        w_path[t : t + p.S],
        np.diag(n_path[t : t + p.S, :, j]),
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
    assert np.any(cohort > 0.0), "test would be vacuous with a zero benefit"
    assert np.allclose(cohort, np.diag(pension_mat[t : t + p.S, :, j]))


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
