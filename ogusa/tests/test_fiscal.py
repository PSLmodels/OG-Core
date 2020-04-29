import numpy as np
import pandas as pd
import pytest
import os
from ogusa import fiscal
from ogusa.parameters import Specifications
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

# Read in test data from file
df = pd.read_csv(os.path.join(CUR_PATH, 'test_io_data',
                              'get_D_G_path_data.csv'))
Y = df['Y'].values
TR = df['TR'].values
Revenue = df['Revenue'].values
Gbaseline = df['Gbaseline'].values
D1 = df['D1'].values
D2 = df['D2'].values
D3 = df['D3'].values
G1 = df['G1'].values
G2 = df['G2'].values
G3 = df['G3'].values
D_d1 = df['D_d1'].values
D_d2 = df['D_d2'].values
D_d3 = df['D_d3'].values
D_f1 = df['D_f1'].values
D_f2 = df['D_f2'].values
D_f3 = df['D_f3'].values


@pytest.mark.parametrize(
    ('baseline_spending,Y,TR,Revenue,Gbaseline,D_expected,G_expected,' +
     'D_d_expected,D_f_expected,budget_balance'),
    [(False, Y, TR, Revenue, Gbaseline, D1, G1, D_d1, D_f1, False),
     (True, Y, TR, Revenue, Gbaseline, D2, G2, D_d2, D_f2, False),
     (False, Y, TR, Revenue, Gbaseline, D3, G3, D_d3, D_f3, True)],
    ids=['baseline_spending = False', 'baseline_spending = True',
         'balanced_budget = True'])
def test_D_G_path(baseline_spending, Y, TR, Revenue, Gbaseline,
                  D_expected, G_expected, D_d_expected, D_f_expected,
                  budget_balance):
    p = Specifications()
    new_param_values = {
        'T': 320,
        'S': 80,
        'debt_ratio_ss': 1.2,
        'tG1': 20,
        'tG2': 256,
        'alpha_T': [0.09],
        'alpha_G': [0.05],
        'rho_G': 0.1,
        'g_y_annual': 0.03,
        'baseline_spending': baseline_spending,
        'budget_balance': budget_balance
    }
    p.update_specifications(new_param_values, raise_errors=False)
    r_gov = np.ones(p.T + p.S) * 0.03
    p.g_n = np.ones(p.T + p.S) * 0.02
    D0 = 0.59
    G0 = 0.05
    dg_fixed_values = (Y, Revenue, TR, D0, G0)
    test_D, test_G, test_D_d, test_D_f = fiscal.D_G_path(
        r_gov, dg_fixed_values, Gbaseline, p)
    assert np.allclose(test_D[:p.T], D_expected[:p.T])
    assert np.allclose(test_G[:p.T], G_expected[:p.T])
    assert np.allclose(test_D_d[:p.T], D_d_expected[:p.T])
    assert np.allclose(test_D_f[:p.T], D_f_expected[:p.T])


p1 = Specifications()
p1.r_gov_scale = 0.5
p1.r_gov_shift = 0.0
p2 = Specifications()
p2.r_gov_scale = 0.5
p2.r_gov_shift = 0.01
p3 = Specifications()
p3.r_gov_scale = 0.5
p3.r_gov_shift = 0.03
r = 0.04
r_gov1 = 0.02
r_gov2 = 0.01
r_gov3 = 0.0


@pytest.mark.parametrize('r,p,r_gov_expected',
                         [(r, p1, r_gov1), (r, p2, r_gov2),
                          (r, p3, r_gov3),],
                         ids=['Scale only', 'Scale and shift',
                              'r_gov < 0'])
def test_get_r_gov(r, p, r_gov_expected):
    r_gov = fiscal.get_r_gov(r, p)
    assert np.allclose(r_gov, r_gov_expected)
