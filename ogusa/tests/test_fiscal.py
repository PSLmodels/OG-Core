import numpy as np
import pytest
from ogusa import fiscal


def test_D_G_path():
    T = 320
    S = 80
    debt_ratio_ss = 1.2
    tG1 = 20
    tG2 = int(T * 0.8)
    ALPHA_T = np.ones(T + S) * 0.09
    ALPHA_G = np.ones(T + S) * 0.05
    rho_G = 0.1
    r_gov = np.ones(T) * 0.04
    g_n_vector = np.ones(T) * 0.02
    g_y = 0.03
    D0 = 0.11
    G0 = 0.02
    Y = ...
    T_H = ...
    REVENUE = ...
    budget_balance = False
    baseline_spending = False
    other_dg_params = (T, r_gov, g_n_vector, g_y)
    dg_fixed_values = (Y, REVENUE, T_H, D0, G0)
    fiscal_params = (budget_balance, ALPHA_T, ALPHA_G, tG1, tG2, rho_G,
                     debt_ratio_ss)

    test_D, test_G = fiscal.D_G_path(dg_fixed_values, fiscal_params,
                                     other_dg_params, baseline_spending)
    assert np.allclose(test_D, D)
    assert np.allclose(test_G, G)
