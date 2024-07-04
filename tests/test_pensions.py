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
deriv_DB_expected1 = np.array(
    [0.352, 0.3256, 0.2904, 0.232, 0.0, 0.0, 0.0])
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
deriv_DB_expected2 = np.array(
    [0.6105, 0.5445, 0.435, 0.43935, 0.0, 0.0])
args_ddb2 = (w_ddb1, e_ddb1, per_rmn, p2)

test_data = [(args_ddb1, deriv_DB_expected1),
             (args_ddb2, deriv_DB_expected2)]


@pytest.mark.parametrize('args,deriv_DB_expected', test_data,
                         ids=['non-zero d_theta: case 1',
                              'non-zero d_theta: case 2'])
def test_deriv_DB(args, deriv_DB_expected):
    """
    Test of the pensions.deriv_DB() function.
    """
    (w, e, per_rmn, p) = args
    deriv_DB = pensions.deriv_DB(w, e, per_rmn, p)

    assert (np.allclose(deriv_DB, deriv_DB_expected))