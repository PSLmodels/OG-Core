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
