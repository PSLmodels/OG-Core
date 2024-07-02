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
p.update_specifications({
    "S": 7,
    "retirement_age": 4,
    "last_career_yrs": 3,
    "yr_contr": 4,
    "rep_rate_py":  0.2
})
j = 1
w = np.array([1.2, 1.1, 1.21, 1.0, 1.01, 0.99, 0.8])
e = np.array([1.1, 1.11, 0.9, 0.87, 0.87, 0.7, 0.6])
n = np.array([0.4, 0.45, 0.4, 0.42, 0.3, 0.2, 0.2])
g_y = 0.03
L_inc_avg = np.zeros(0)
L_inc_avg_s = np.zeros(p.last_career_yrs)
DB_s = np.zeros(p.retirement_age)
DB = np.zeros(p.S)
DB_loop_expected1 = np.array([0, 0, 0, 0, 0.337864778, 0.327879365, 0.318189065])
args1 = w, e, n, p.retirement_age, p.S, p.g_y, L_inc_avg_s, L_inc_avg, DB_s, DB

test_data = [(args1, DB_loop_expected1)]
#             (classes2, args2, NDC_expected2)]

@pytest.mark.parametrize('args,DB_loop_expected', test_data,
                         ids=['SS/Complete'])
def test_DB_1dim_loop(args, DB_loop_expected):
    """
    Test of the pensions.DB_1dim_loop() function.
    """

    w, e, n, S_ret, S, g_y, L_inc_avg_s, L_inc_avg, DB_s, DB = args
    DB_loop = pensions.DB_1dim_loop(
        w, e, n, S_ret, S, g_y, L_inc_avg_s, L_inc_avg, DB_s, DB,
        p.last_career_yrs, p.rep_rate,
        p.rep_rate_py, p.yr_contr)
    assert (np.allclose(DB_loop, DB_loop_expected))
