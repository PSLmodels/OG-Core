import numpy as np
import copy
import pytest
from ogcore import pensions
from ogcore.parameters import Specifications


p = Specifications()
rho_vec = np.zeros((1, 4))
rho_vec[0, -1] = 1.0
new_param_values = {
    "S": 4,
    "rho": rho_vec.tolist(),
    "lambdas": [1.0],
    "labor_income_tax_noncompliance_rate": [[0.0]],
    "capital_income_tax_noncompliance_rate": [[0.0]],
    "J": 1,
    "T": 4,
    "chi_n": np.ones(4),
    "eta": (np.ones((4, 1)) / (4 * 1)),
    "e": np.ones((4, 1)),
}
p.update_specifications(new_param_values)
p.retire = [3, 3, 3, 3, 3, 3, 3, 3]
p1 = copy.deepcopy(p)
p2 = copy.deepcopy(p)
p3 = copy.deepcopy(p)
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

test_data = [
    (n1, wss, factor1, 0, p1, expected1),
    (n2, wss, factor1, None, p2, expected2),
    (n3, wss, factor3, None, p3, expected3),
    (n3, wss, factor4, None, p3, expected4),
    (n3, wss, factor4, None, p5, expected5),
]


@pytest.mark.parametrize(
    "n,w,factor,j,p_in,expected",
    test_data,
    ids=["1D e", "2D e", "AIME case 2", "AIME case 3", "Min PIA case"],
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
# p.update_specifications({
#     "S": 7,
#     "rep_rate_py":  0.2
# })
p.S = 7
p.rep_rate_py = 0.2
p.retire = 4
p.last_career_yrs = 3
p.yr_contr = 4
p.g_y = 0.03
j = 1
w = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
L_inc_avg = np.zeros(0)
L_inc_avg_s = np.zeros(p.last_career_yrs)
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
    p.last_career_yrs,
    p.rep_rate_py,
    p.yr_contr,
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
        yr_contr,
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
        yr_contr,
    )
    assert np.allclose(DB_loop, DB_loop_expected)


p = Specifications()
p.S = 7
p.retire = 4
per_rmn = p.S
p.last_career_yrs = 3
p.yr_contr = p.retire
p.rep_rate_py = 0.2
p.g_y = 0.03
w = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
deriv_DB_loop_expected = np.array(
    [0.352, 0.3256, 0.2904, 0.232, 0.0, 0.0, 0.0]
)
d_theta_empty = np.zeros_like(w)
args3 = (
    w,
    e,
    p.S,
    p.retire,
    per_rmn,
    p.last_career_yrs,
    p.rep_rate_py,
    p.yr_contr,
)

test_data = [(args3, deriv_DB_loop_expected)]


@pytest.mark.parametrize("args,deriv_DB_loop_expected", test_data)
def test_deriv_DB_loop(args, deriv_DB_loop_expected):
    """
    Test of the pensions.deriv_DB_loop() function.
    """
    (w, e, S, retire, per_rmn, last_career_yrs, rep_rate_py, yr_contr) = args
    deriv_DB_loop = pensions.deriv_DB_loop(
        w, e, S, retire, per_rmn, last_career_yrs, rep_rate_py, yr_contr
    )

    assert np.allclose(deriv_DB_loop, deriv_DB_loop_expected)


p = Specifications()
p.S = 7
p.retire = 4
p.vpoint = 0.4
w = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
p.g_y = 0.03
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
    (w, e, S, retire, per_rmn, d_theta_empty, vpoint, factor) = args

    deriv_PS_loop = pensions.deriv_PS_loop(
        w, e, S, retire, per_rmn, d_theta_empty, vpoint, factor
    )

    assert np.allclose(deriv_PS_loop, deriv_PS_loop_expected)


#############non-zero d_theta: case 1############
p = Specifications()
p.S = 7
p.retire = 4
p.last_career_yrs = 3
p.yr_contr = p.retire
p.rep_rate_py = 0.2
p.g_y = 0.03
n_ddb1 = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
w_ddb1 = np.array([1.2, 1.1, 1.21, 1, 1.01, 0.99, 0.8])
e_ddb1 = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
per_rmn = n_ddb1.shape[0]
d_theta_empty = np.zeros_like(per_rmn)
deriv_DB_expected1 = np.array([0.352, 0.3256, 0.2904, 0.232, 0.0, 0.0, 0.0])
args_ddb1 = (w_ddb1, e_ddb1, per_rmn, p)

#############non-zero d_theta: case 2############
p2 = Specifications()
p2.S = 7
p2.retire = 5
p2.last_career_yrs = 2
p2.yr_contr = p2.retire
p2.rep_rate_py = 0.2
p2.g_y = 0.03
n_ddb2 = np.array([0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
w_ddb1 = np.array([1.1, 1.21, 1, 1.01, 0.99, 0.8])
e_ddb1 = np.array([1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
per_rmn = n_ddb2.shape[0]
d_theta_empty = np.zeros_like(per_rmn)
deriv_DB_expected2 = np.array([0.6105, 0.5445, 0.435, 0.43935, 0.0, 0.0])
args_ddb2 = (w_ddb1, e_ddb1, per_rmn, p2)

test_data = [(args_ddb1, deriv_DB_expected1), (args_ddb2, deriv_DB_expected2)]


@pytest.mark.parametrize(
    "args,deriv_DB_expected",
    test_data,
    ids=["non-zero d_theta: case 1", "non-zero d_theta: case 2"],
)
def test_deriv_DB(args, deriv_DB_expected):
    """
    Test of the pensions.deriv_DB() function.
    """
    (w, e, per_rmn, p) = args
    deriv_DB = pensions.deriv_DB(w, e, per_rmn, p)

    assert np.allclose(deriv_DB, deriv_DB_expected)


#############PS deriv SS or complete lifetimes############
p = Specifications()
p.S = 7
p.retire = 4
p.vpoint = 0.4
omegas = 1 / (p.S) * np.ones(p.S)
p.omega_SS = omegas
p.g_y = 0.03
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
    (w, e, per_rmn, factor, p) = args

    deriv_PS = pensions.deriv_PS(w, e, per_rmn, factor, p)

    assert np.allclose(deriv_PS, deriv_PS_expected)


#############complete lifetimes, S = 4###################
p = Specifications()
p.S = 4
p.retire = 2
per_rmn = p.S
p.g_y = np.ones(p.T) * 0.03
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


# ################pension benefit: DB############
# p = Specifications()
# p.pension_system = 'DB'
# p.S = 7
# p.retire = 4
# p.last_career_yrs = 3
# p.yr_contr = p.retire
# p.rep_rate_py = 0.2
# p.g_y = np.ones(p.T) * 0.03
# w_db = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
# e_db = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
# n_db = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
# Y = None
# j = 1
# factor = 2
# omegas = None
# lambdas = 1
# pension_expected_db = [0, 0, 0, 0, 0.337864778, 0.327879365, 0.318189065]
# args_pb = (w_db, e_db, n_db, r, Y, lambdas, j, factor)

# ################pension benefit: NDC############
# p2 = Specifications()
# p2.pension_system = 'NDC'
# p2.ndc_growth_rate = 'LR GDP'
# p2.dir_growth_rate = 'r'
# p2.S = 4
# p2.retire = 2
# w_ndc = np.array([1.2, 1.1, 1.21, 1])
# e_ndc = np.array([1.1, 1.11, 0.9, 0.87])
# n_ndc = np.array([0.4, 0.45, 0.4, 0.3])
# Y = None
# j = 1
# p2.g_y = 0.03
# p2.g_n_SS = 0.0
# r = 0.03
# factor = 2
# p2.tau_p = 0.3
# p2.k_ret = 0.4615
# p2.mort_rates_SS = np.array([0.01, 0.05, 0.3, 0.4, 1])
# omegas = None
# lambdas = 1
# pension_expected_ndc = [0, 0, 0.279756794, 0.271488732]
# args_ndc = (w_ndc, e_ndc, n_ndc, r, Y, lambdas, j, factor)

# ################pension benefit: PS############
# p3 = Specifications()
# p3.pension_system = 'PS'
# p3.S = 7
# p3.S_ret = 4
# w_ppb = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
# e_ppb = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
# n_ppb = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
# omegas = (1/p3.S) * np.ones(p3.S)
# p3.omega_SS = omegas
# p3.vpoint = 0.4
# factor = 2
# Y = None
# lambdas = 1
# j_ind = 1
# p3.g_y = np.ones(p3.T) * 0.03
# pension_expected_ppb = [0, 0, 0, 0, 0.004164689, 0.004041603, 0.003922156]
# args_ps = (w_db, e_db, n_db, r, Y, lambdas, j_ind, factor)

# test_data = [(args_pb, pension_expected_db),
#              (args_ndc, pension_expected_ndc),
#              (args_ps, pension_expected_ppb)]

# @pytest.mark.parametrize('classes,args,pension_expected', test_data,
#                          ids=['DB', 'NDC', 'PS'])
# def test_pension_benefit(classes, args, pension_expected):
#     '''
#     Test of pensions.get_pension_benefit
#     '''
#     w, e, n, r, Y, lambdas, j_ind, factor = args

#     pension = pensions.pension_amount(
#         w,  n, theta, t, j, shift, method, e, p)
#     assert (np.allclose(pension, pension_expected))


############SS or complete lifetimes############
p = Specifications()
p.S = 3
p.retire = 2
per_rmn = p.S
p.g_n_SS = 0.0
p.g_y = np.ones(p.T) * 0.03
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
    (w, e, per_rmn, g_ndc_value, delta_ret_value, d_theta, p) = args

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
p.g_y = np.ones(p.T) * 0.04
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
    (surv_rates, g_ndc_value, dir_delta_s_empty, p) = args
    dir_delta = pensions.delta_ret_loop(
        p.S, p.retire, surv_rates, g_ndc_value, dir_delta_s_empty
    )

    assert np.allclose(dir_delta, dir_delta_ret_expected)
