"""
Test of steady-state module
"""

import multiprocessing
from distributed import Client, LocalCluster
import pytest
import numpy as np
import os
import pickle
from ogcore import SS, utils, aggregates, fiscal
from ogcore.parameters import Specifications
from ogcore import firm

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
NUM_WORKERS = min(multiprocessing.cpu_count(), 7)

VAR_NAME_MAPPING = {
    "Yss": "Y",
    "Bss": "B",
    "Kss": "K",
    "K_f_ss": "K_f",
    "K_d_ss": "K_d",
    "Lss": "L",
    "Css": "C",
    "Iss": "I",
    "Iss_total": "I_total",
    "I_d_ss": "I_d",
    "K_g_ss": "K_g",
    "I_g_ss": "I_g",
    "BQss": "BQ",
    "RMss": "RM",
    "Y_vec_ss": "Y_m",
    "K_vec_ss": "K_m",
    "L_vec_ss": "L_m",
    "C_vec_ss": "C_i",
    "TR_ss": "TR",
    "agg_pension_outlays": "agg_pension_outlays",
    "Gss": "G",
    "UBI_outlays_SS": "UBI",
    "total_tax_revenue": "total_tax_revenue",
    "business_tax_revenue": "business_tax_revenue",
    "iit_payroll_tax_revenue": "iit_payroll_tax_revenue",
    "iit_revenue": "iit_revenue",
    "payroll_tax_revenue": "payroll_tax_revenue",
    "bequest_tax_revenue": "bequest_tax_revenue",
    "wealth_tax_revenue": "wealth_tax_revenue",
    "cons_tax_revenue": "cons_tax_revenue",
    "Dss": "D",
    "D_f_ss": "D_f",
    "D_d_ss": "D_d",
    "new_borrowing": "new_borrowing",
    "debt_service": "debt_service",
    "new_borrowing_f": "new_borrowing_f",
    "debt_service_f": "debt_service_f",
    "rss": "r",
    "r_gov_ss": "r_gov",
    "r_p_ss": "r_p",
    "wss": "w",
    "p_m_ss": "p_m",
    "p_i_ss": "p_i",
    "p_tilde_ss": "p_tilde",
    "bssmat_splus1": "b_sp1",
    "bssmat_s": "b_s",
    "nssmat": "n",
    "cssmat": "c",
    "c_i_ss_mat": "c_i",
    "bqssmat": "bq",
    "rmssmat": "rm",
    "trssmat": "tr",
    "ubissmat": "ubi",
    "yss_before_tax_mat": "before_tax_income",
    "total_taxes_ss": "hh_taxes",
    "etr_ss": "etr",
    "mtrx_ss": "mtrx",
    "mtry_ss": "mtry",
    "theta": "theta",
    "factor_ss": "factor",
    "euler_savings": "euler_savings",
    "euler_labor_leisure": "euler_labor_leisure",
    "resource_constraint_error": "resource_constraint_error",
}


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "SS_fsolve_inputs.pkl")
)
(bssmat, nssmat, TR_ss, factor_ss) = input_tuple
# Parameterize the baseline, closed econ case
p1 = Specifications(baseline=True)
p1.update_specifications({"zeta_D": [0.0], "zeta_K": [0.0]})
guesses1 = np.array(
    [0.06, 1.1, 0.2, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051]
)
args1 = (bssmat, nssmat, None, None, None, p1, None)
expected1 = np.array(
    [
        -0.03640424626041604,
        -0.03002637958804053,
        0.2262064580426968,
        0.0,
        1.4598033016971916,
        -0.00161369,
        -0.01822709,
        -0.01675017,
        0.006676,
        0.0104632,
        -0.01955018,
        -0.00296457,
        0.13138229715274724,
        0.1237126490720427,
    ]
)
# Parameterize the reform, closed econ case
p2 = Specifications(baseline=False)
p2.update_specifications({"zeta_D": [0.0], "zeta_K": [0.0]})
guesses2 = np.array(
    [0.06, 1.1, 0.2, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07]
)
args2 = (bssmat, nssmat, None, None, 0.51, p2, None)
expected2 = np.array(
    [
        -0.0389819118896058,
        -0.03275578110093917,
        0.253354429177328,
        0.0,
        1.4764069856763156,
        -0.00165626,
        -0.01503618,
        -0.01407456,
        0.00661677,
        0.01038606,
        -0.01932943,
        -0.00294703,
        0.132876628710868,
    ]
)
# Parameterize the reform, closed econ, baseline spending case
p3 = Specifications(baseline=False)
p3.update_specifications(
    {"zeta_D": [0.0], "zeta_K": [0.0], "baseline_spending": True}
)
guesses3 = np.array(
    [0.06, 1.1, 0.2, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07]
)
args3 = (bssmat, nssmat, 0.13, 0.0, 0.51, p3, None)
expected3 = np.array(
    [
        -0.042611174492217574,
        -0.03660486260948588,
        0.2942852551844308,
        0.0,
        0.43144008183325194,
        0.0044546,
        0.00790648,
        0.01043014,
        0.00872496,
        0.01242235,
        0.00952339,
        -0.00284511,
        0.0,
    ]
)
# Parameterize the baseline, partial open economy case (default)
p4 = Specifications(baseline=True)
guesses4 = np.array(
    [0.06, 1.1, 0.2, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051]
)
args4 = (bssmat, nssmat, None, None, None, p4, None)
expected4 = np.array(
    [
        -0.04501723939772713,
        -0.039160814474571426,
        0.32336315872334676,
        0.0,
        1.5404736783359936,
        -0.00173474,
        0.00199568,
        0.00591891,
        0.00653568,
        0.01029101,
        0.0075058,
        0.00325183,
        0.13864263105023944,
        0.10922623253142945,
    ]
)
# Parameterize the baseline, small open econ case
p5 = Specifications(baseline=True)
p5.update_specifications({"zeta_D": [0.0], "zeta_K": [1.0]})
guesses5 = np.array(
    [0.06, 1.1, 0.2, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051]
)
args5 = (bssmat, nssmat, None, None, 0.51, p5, None)
expected5 = np.array(
    [
        -0.02690768327226259,
        -0.019999999999999962,
        0.1376969417785776,
        0.0,
        1.44721176202231,
        -0.00148021,
        0.00239001,
        0.00638136,
        0.00683071,
        0.01065305,
        0.00799657,
        0.00336337,
        0.1302490585820079,
        0.11156343085283874,
    ]
)
# Parameterize the baseline closed economy, delta tau = 0 case
p6 = Specifications(baseline=True)
p6.update_specifications(
    {"zeta_D": [0.0], "zeta_K": [0.0], "delta_tau_annual": [[0.0]]}
)
guesses6 = np.array(
    [0.06, 1.1, 0.2, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051]
)
args6 = (bssmat, nssmat, None, None, None, p6, None)
expected6 = np.array(
    [
        -0.051097905293268894,
        -0.047817638192649635,
        0.42739129061380643,
        0.0,
        1.5904342991581968,
        -0.00187832,
        0.00177827,
        0.00566193,
        0.00637141,
        0.01008918,
        0.00723656,
        0.00319034,
        0.1431390869242377,
        0.10614753083674845,
    ]
)
p7 = Specifications(baseline=True)
p7.update_specifications(
    {
        "M": 4,
        "I": 4,
        "io_matrix": np.eye(4),
        "alpha_c": [0.1, 0.5, 0.3, 0.1],
        "epsilon": [1.0, 1.0, 1.0, 1.0],
        "gamma": [0.3, 0.4, 0.35, 0.45],
        "gamma_g": [0.0, 0.0, 0.0, 0.0],
    }
)
guesses7 = np.array(
    [0.06, 1.1, 0.2, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051]
)
args7 = (bssmat, nssmat, None, None, None, p7, None)
expected7 = np.array(
    [
        -0.06985935377445636,
        -0.07388184648847439,
        2.596215180212739,
        3.0425352411195634,
        2.08611520332783,
        2.5993398246497392,
        0.0,
        1.6276918281128583,
        -0.0005336644680328222,
        0.003641474531794135,
        0.007892881165609,
        0.007854285496066054,
        0.011964025188377221,
        0.00905400047723115,
        0.0035962471039776792,
        0.14649226453015723,
        0.03816296076039217,
    ]
)


@pytest.mark.parametrize(
    "guesses,args,expected",
    [
        (guesses1, args1, expected1),
        (guesses2, args2, expected2),
        (guesses3, args3, expected3),
        (guesses4, args4, expected4),
        (guesses5, args5, expected5),
        (guesses6, args6, expected6),
        (guesses7, args7, expected7),
    ],
    ids=[
        "Baseline, Closed",
        "Reform, Closed",
        "Reform, Baseline spending=True, Closed",
        "Baseline, Partial Open",
        "Baseline, Small Open",
        "Baseline, Closed, delta_tau = 0",
        "Baseline, M=4",
    ],
)
def test_SS_fsolve(tmpdir, guesses, args, expected):
    """
    Test SS.SS_fsolve function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    """
    # args =
    (bssmat, nssmat, TR_ss, Ig_baseline, factor_ss, p, client) = args
    p.baseline_dir = tmpdir
    p.output_base = tmpdir

    # take old format for guesses and put in new format
    r_p = guesses[0]
    r = guesses[0]
    w = firm.get_w_from_r(r_p, p, "SS")
    p_m = np.ones(p.M)

    if p.baseline:
        BQ = guesses[3:-2]
        TR = guesses[-2]
        factor = guesses[-1]
        Y = TR / p.alpha_T[-1]
    else:
        BQ = guesses[3:-1]
        TR = guesses[-1]
        if p.baseline_spending:
            TR = TR_ss
            Y = guesses[2]
        else:
            Y = TR / p.alpha_T[-1]
    if p.baseline:
        new_guesses = [r_p, r, w] + list(p_m) + [Y] + list(BQ) + [TR, factor]
    else:
        new_guesses = [r_p, r, w] + list(p_m) + [Y] + list(BQ) + [TR]

    test_list = SS.SS_fsolve(new_guesses, *args)
    print("Test list = ", test_list)

    assert np.allclose(
        np.hstack(np.array(test_list)), np.array(expected), atol=5e-4
    )


# Parameterize baseline, partially open econ case (default)
param_updates1 = {}
filename1 = "SS_solver_outputs_baseline.pkl"
# Parameterize baseline, balanced budget case
param_updates2 = {"budget_balance": True, "alpha_G": [0.0]}
filename2 = "SS_solver_outputs_baseline_budget_balance.pkl"
# Parameterize the reform, baseline spending case
param_updates3 = {"baseline_spending": True}
filename3 = "SS_solver_outputs_reform_baseline_spending.pkl"
# Parameterize the baseline, small open econ case
param_updates4 = {"zeta_K": [1.0], "initial_guess_r_SS": 0.10}
filename4 = "SS_solver_outputs_baseline_small_open.pkl"


# Note that changing the order in which these tests are run will cause
# failures for the baseline spending=True tests which depend on the
# output of the baseline run just prior
@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [
        (True, param_updates1, filename1),
        (True, param_updates2, filename2),
        (False, param_updates3, filename3),
        (True, param_updates4, filename4),
    ],
    ids=[
        "Baseline",
        "Baseline, budget balance",
        "Reform, baseline spending=True",
        "Baseline, small open",
    ],
)
def test_SS_solver(baseline, param_updates, filename, dask_client):
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.frac_tax_payroll = np.zeros(p.frac_tax_payroll.shape)
    p.output_base = CUR_PATH
    b_guess = np.ones((p.S, p.J)) * 0.07
    n_guess = np.ones((p.S, p.J)) * 0.35 * p.ltilde
    if p.zeta_K[-1] == 1.0:
        rguess = p.world_int_rate[-1]
    else:
        rguess = 0.06483431412921253
    r_p_guess = rguess
    wguess = firm.get_w_from_r(rguess, p, "SS")
    TRguess = 0.05738932081035772
    factorguess = 139355.1547340256
    BQguess = aggregates.get_BQ(rguess, b_guess, None, p, "SS", False)
    Yguess = 0.6376591201150815
    p_m_guess = np.ones(p.M)

    # for new Ig_baseline arg
    if p.baseline_spending:
        Ig_baseline = 0.0  # tests only have zero infrastructure
    else:
        Ig_baseline = None

    test_dict = SS.SS_solver(
        b_guess,
        n_guess,
        r_p_guess,
        rguess,
        wguess,
        p_m_guess,
        Yguess,
        BQguess,
        TRguess,
        Ig_baseline,
        factorguess,
        p,
        dask_client,
        False,
    )

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", filename)
    )

    for k, v in expected_dict.items():
        print("Testing ", k)
        print("diff = ", np.abs(test_dict[VAR_NAME_MAPPING[k]] - v).max())

    for k, v in expected_dict.items():
        print("Testing ", k)
        print("diff = ", np.abs(test_dict[VAR_NAME_MAPPING[k]] - v).max())
        assert np.allclose(
            test_dict[VAR_NAME_MAPPING[k]], v, atol=1e-04, equal_nan=True
        )


param_updates5 = {"zeta_K": [1.0], "budget_balance": True, "alpha_G": [0.0]}
filename5 = "SS_solver_outputs_baseline_small_open_budget_balance.pkl"
param_updates6 = {
    "delta_tau_annual": [[0.0]],
    "zeta_K": [0.0],
    "zeta_D": [0.0],
    "initial_guess_r_SS": 0.02,
    "initial_guess_TR_SS": 0.02,
}
filename6 = "SS_solver_outputs_baseline_delta_tau0.pkl"
# Can't seem to get even close to a solution with M=4 here.
# param_updates7 = {
#     'M': 4, 'alpha_c': [0.1, 0.5, 0.3, 0.1],
#     'epsilon': [1.0, 1.0, 1.0, 1.0],
#     'gamma': [0.3, 0.4, 0.35, 0.45],
#     'gamma_g': [0.0, 0.0, 0.0, 0.0],
#     'initial_guess_r_SS': 0.15,
#     'initial_guess_TR_SS': 0.06}
# filename7 = 'SS_solver_outputs_baseline_M4.pkl'


@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [(True, param_updates5, filename5), (True, param_updates6, filename6)],
    ids=["Baseline, small open, budget balance", "Baseline, delta_tau = 0"],
)
@pytest.mark.local
def test_SS_solver_extra(baseline, param_updates, filename, dask_client):
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    b_guess = np.ones((p.S, p.J)) * 0.07
    n_guess = np.ones((p.S, p.J)) * 0.35 * p.ltilde
    if p.zeta_K[-1] == 1.0:
        rguess = p.world_int_rate[-1]
    else:
        rguess = 0.06483431412921253
    r_p_guess = rguess
    if p.baseline_spending:
        Ig_baseline = 0.0
    else:
        Ig_baseline = None
    wguess = firm.get_w_from_r(rguess, p, "SS")
    TRguess = 0.05738932081035772
    factorguess = 139355.1547340256
    BQguess = aggregates.get_BQ(rguess, b_guess, None, p, "SS", False)
    Yguess = 0.6376591201150815
    p_m_guess = np.ones(p.M)

    test_dict = SS.SS_solver(
        b_guess,
        n_guess,
        r_p_guess,
        rguess,
        wguess,
        p_m_guess,
        Yguess,
        BQguess,
        TRguess,
        Ig_baseline,
        factorguess,
        p,
        dask_client,
        False,
    )

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", filename)
    )

    for k, v in expected_dict.items():
        print("Testing ", k)
        assert np.allclose(
            test_dict[VAR_NAME_MAPPING[k]], v, atol=1e-05, equal_nan=True
        )


param_updates1 = {"zeta_K": [1.0]}
filename1 = "inner_loop_outputs_baseline_small_open.pkl"
param_updates2 = {"budget_balance": True, "alpha_G": [0.0]}
filename2 = "inner_loop_outputs_baseline_balance_budget.pkl"
param_updates3 = {}
filename3 = "inner_loop_outputs_baseline.pkl"
param_updates4 = {}
filename4 = "inner_loop_outputs_reform.pkl"
param_updates5 = {"baseline_spending": True}
filename5 = "inner_loop_outputs_reform_baselinespending.pkl"
param_updates7 = {
    "M": 4,
    "I": 4,
    "io_matrix": np.eye(4),
    "alpha_c": [0.1, 0.5, 0.3, 0.1],
    "epsilon": [1.0, 1.0, 1.0, 1.0],
    "gamma": [0.3, 0.4, 0.35, 0.45],
    "gamma_g": [0.0, 0.0, 0.0, 0.0],
}
filename7 = "inner_loop_outputs_reform_M4.pkl"
param_updates8 = {
    "M": 4,
    "I": 5,
    "io_matrix": np.array(
        [
            [0.2, 0.2, 0.2, 0.4],
            [0.3, 0.1, 0.4, 0.2],
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.7, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
    "alpha_c": [0.1, 0.4, 0.3, 0.1, 0.1],
    "epsilon": [1.0, 1.0, 1.0, 1.0],
    "gamma": [0.3, 0.4, 0.35, 0.45],
    "gamma_g": [0.0, 0.0, 0.0, 0.0],
}
filename8 = "inner_loop_outputs_reform_MneI.pkl"


@pytest.mark.parametrize(
    "baseline,r_p,param_updates,filename",
    [
        (True, 0.03309231672773741, param_updates1, filename1),
        (True, 0.05, param_updates2, filename2),
        (True, 0.04260341179572245, param_updates3, filename3),
        (False, 0.04260341179572245, param_updates4, filename4),
        (False, 0.04260341179572245, param_updates5, filename5),
        (False, 0.04759112768438152, param_updates7, filename7),
        (False, 0.04759112768438152, param_updates8, filename8),
    ],
    ids=[
        "Baseline, Small Open",
        "Baseline, Balanced Budget",
        "Baseline",
        "Reform",
        "Reform, baseline spending",
        "Reform, M>1",
        "Reform, I!=>M",
    ],
)
def test_inner_loop(baseline, r_p, param_updates, filename, dask_client):
    # Test SS.inner_loop function. Provide inputs to function and ensure that
    # output returned matches what it has been before.
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    bssmat = np.ones((p.S, p.J)) * 0.07
    nssmat = np.ones((p.S, p.J)) * 0.4 * p.ltilde
    if p.zeta_K[-1] == 1.0:
        r = p.world_int_rate[-1]
    else:
        r = 0.05
    w = firm.get_w_from_r(r, p, "SS")
    TR = 0.12
    Y = 1.3
    p_m = np.ones(p.M)
    factor = 100000
    BQ = np.ones(p.J) * 0.00019646295986015257
    if p.budget_balance:
        outer_loop_vars = (
            bssmat,
            nssmat,
            r_p,
            r,
            w,
            p_m,
            Y,
            BQ,
            TR,
            None,
            factor,
        )
    else:
        if p.baseline_spending:
            Ig_baseline = 0.0
        else:
            Ig_baseline = None
        outer_loop_vars = (
            bssmat,
            nssmat,
            r_p,
            r,
            w,
            p_m,
            Y,
            BQ,
            TR,
            Ig_baseline,
            factor,
        )
    test_tuple = SS.inner_loop(outer_loop_vars, p, dask_client)

    try:
        (
            euler_errors,
            bssmat,
            nssmat,
            new_r,
            new_r_gov,
            new_r_p,
            new_w,
            new_RM,
            new_TR,
            Y,
            new_factor,
            new_BQ,
            average_income_model,
        ) = utils.safe_read_pickle(
            os.path.join(CUR_PATH, "test_io_data", filename)
        )
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            K_vec,
            L_vec,
            Y_vec,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = test_tuple
        expected_tuple = (
            euler_errors,
            bssmat,
            nssmat,
            new_r,
            new_r_gov,
            new_r_p,
            new_w,
            1.0,
            K_vec,
            L_vec,
            Y_vec,
            new_RM,
            new_TR,
            Y,
            new_factor,
            new_BQ,
            average_income_model,
        )
    except ValueError:
        expected_tuple = utils.safe_read_pickle(
            os.path.join(CUR_PATH, "test_io_data", filename)
        )

    for i, v in enumerate(expected_tuple):
        print("Max diff = ", np.absolute(test_tuple[i] - v).max())
        print("Checking item = ", i)
        if np.absolute(test_tuple[i] - v).max() > 1.0:
            print("test_value = ", test_tuple[i])
            print("expected_value = ", v)

    for i, v in enumerate(expected_tuple):
        print("Max diff = ", np.absolute(test_tuple[i] - v).max())
        print("Checking item = ", i)
        assert np.allclose(test_tuple[i], v, atol=4e-05)


param_updates6 = {"zeta_K": [0.0], "zeta_D": [0.0]}
filename6 = "inner_loop_outputs_baseline_delta_tau0.pkl"


@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [(False, param_updates6, filename6)],
    ids=["Baseline, delta_tau = 0"],
)
@pytest.mark.local
def test_inner_loop_extra(baseline, param_updates, filename, dask_client):
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    bssmat = np.ones((p.S, p.J)) * 0.07
    nssmat = np.ones((p.S, p.J)) * 0.4 * p.ltilde
    r = 0.05
    w = firm.get_w_from_r(r, p, "SS")
    TR = 0.12
    Y = 1.3
    factor = 100000
    BQ = np.ones(p.J) * 0.00019646295986015257
    p_m = np.array([1.0])
    r_p = 0.04260341179572245
    if p.baseline_spending:
        Ig_baseline = 0.0
    else:
        Ig_baseline = None
    outer_loop_vars = (
        bssmat,
        nssmat,
        r_p,
        r,
        w,
        p_m,
        Y,
        BQ,
        TR,
        Ig_baseline,
        factor,
    )
    test_tuple = SS.inner_loop(outer_loop_vars, p, dask_client)
    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", filename)
    )
    for i, v in enumerate(expected_tuple):
        print("Max diff = ", np.absolute(test_tuple[i] - v).max())
        print("Checking item = ", i)
        assert np.allclose(test_tuple[i], v, atol=1e-05)


input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "euler_eqn_solver_inputs.pkl")
)
p1 = Specifications()
ubi_j1 = np.zeros(p1.S)
expected1 = np.array(
    [
        1.51538712e01,
        -2.56197959e00,
        -2.54265884e00,
        -3.28629424e00,
        -3.22778974e00,
        -3.41561489e00,
        -3.51017369e00,
        -3.75042314e00,
        -3.80383214e00,
        -4.14829718e00,
        -4.44514599e00,
        -4.67960243e00,
        -4.87283749e00,
        -5.18463537e00,
        -5.34423297e00,
        -5.68063373e00,
        -5.94916029e00,
        -6.38177119e00,
        -6.80812242e00,
        -7.28522062e00,
        -7.85145988e00,
        -8.66878353e00,
        -9.37562765e00,
        -1.03414011e01,
        -1.14081246e01,
        -1.25964679e01,
        -1.39480872e01,
        -1.52910868e01,
        -1.68180780e01,
        -1.83558158e01,
        -2.00575245e01,
        -2.18537125e01,
        -2.36884268e01,
        -2.55965567e01,
        -2.76079141e01,
        -2.97881573e01,
        -3.21294338e01,
        -3.44195741e01,
        -3.66247981e01,
        -3.89743765e01,
        -4.15665336e01,
        -4.44618843e01,
        -4.78531228e01,
        -5.07117561e01,
        -5.61340661e01,
        -6.11117052e01,
        -6.66779805e01,
        -7.28228235e01,
        -7.94109507e01,
        -8.66941024e01,
        -9.51612705e01,
        -1.04586761e02,
        -1.14680002e02,
        -1.25424091e02,
        -1.37538484e02,
        -1.50993053e02,
        -1.66832207e02,
        -1.84034819e02,
        -2.02926084e02,
        -2.23890548e02,
        -2.48093486e02,
        -2.75713079e02,
        -3.06078037e02,
        -3.39247892e02,
        -3.75917422e02,
        -4.17115902e02,
        -4.63605146e02,
        -5.15906513e02,
        -5.74281262e02,
        -6.38710545e02,
        -7.09090498e02,
        -7.85233553e02,
        -8.66911313e02,
        -9.53868333e02,
        -1.04585181e03,
        -1.13779879e03,
        -1.22821994e03,
        -1.31552956e03,
        -1.39805877e03,
        -4.83324702e03,
        4.00442507e00,
        7.96773877e-01,
        1.39237547e00,
        1.23660392e00,
        1.44880997e00,
        1.51531791e00,
        1.55462435e00,
        1.56389001e00,
        1.58687813e00,
        1.47274349e00,
        1.46704464e00,
        1.45168631e00,
        1.44092323e00,
        1.39732199e00,
        1.40790708e00,
        1.37934851e00,
        1.38521215e00,
        1.36048613e00,
        1.45235320e00,
        1.45320126e00,
        1.29772329e00,
        1.24589321e00,
        1.32296088e00,
        1.32456130e00,
        1.33750596e00,
        1.30805420e00,
        1.27939152e00,
        1.31109538e00,
        1.30998662e00,
        1.35165111e00,
        1.35437375e00,
        1.36173552e00,
        1.37248669e00,
        1.35943498e00,
        1.20191457e00,
        1.21529308e00,
        1.21953605e00,
        1.23370464e00,
        1.21452561e00,
        1.16058807e00,
        1.14935749e00,
        1.13913804e00,
        1.07074025e00,
        1.03428234e00,
        -7.71278789e-02,
        -2.73717009e-01,
        -3.16680626e-01,
        -4.12334380e-01,
        -4.96057463e-01,
        -4.79703674e-01,
        -9.29914867e-01,
        -8.62809736e-01,
        -9.06156963e-01,
        -6.81184476e-01,
        -8.63294517e-01,
        -8.16049710e-01,
        -9.23872676e-01,
        -9.10806284e-01,
        -9.87876289e-01,
        -1.06439023e00,
        -9.08063202e-01,
        -1.10809932e00,
        -1.15307801e00,
        -1.18854161e00,
        -1.20320964e00,
        -1.25640695e00,
        -1.39558084e00,
        -1.45512478e00,
        -1.73159290e00,
        -1.79171004e00,
        -1.98857334e00,
        -1.99474109e00,
        -2.54596323e00,
        -2.76229109e00,
        -2.91253931e00,
        -3.00742349e00,
        -3.25085876e00,
        -3.44541790e00,
        -3.59962203e00,
        -3.80400716e00,
    ]
)
p2 = Specifications()
ubi_params = {"ubi_nom_017": 1000, "ubi_nom_1864": 1500, "ubi_nom_65p": 500}
p2.update_specifications(ubi_params)
ubi_j2 = np.ones(p2.S) * 1.19073748e-06
expected2 = np.array(
    [
        1.51534392e01,
        -2.56199950e00,
        -2.54268223e00,
        -3.28630802e00,
        -3.22780451e00,
        -3.41562683e00,
        -3.51018448e00,
        -3.75043149e00,
        -3.80384094e00,
        -4.14830372e00,
        -4.44515126e00,
        -4.67960714e00,
        -4.87284201e00,
        -5.18463857e00,
        -5.34423632e00,
        -5.68063612e00,
        -5.94916278e00,
        -6.38177299e00,
        -6.80812414e00,
        -7.28522248e00,
        -7.85146169e00,
        -8.66878423e00,
        -9.37562887e00,
        -1.03414021e01,
        -1.14081257e01,
        -1.25964689e01,
        -1.39480877e01,
        -1.52910875e01,
        -1.68180784e01,
        -1.83558164e01,
        -2.00575251e01,
        -2.18537130e01,
        -2.36884274e01,
        -2.55965573e01,
        -2.76079145e01,
        -2.97881577e01,
        -3.21294342e01,
        -3.44195746e01,
        -3.66247987e01,
        -3.89743768e01,
        -4.15665337e01,
        -4.44618848e01,
        -4.78531231e01,
        -5.07117617e01,
        -5.61340662e01,
        -6.11117053e01,
        -6.66779808e01,
        -7.28228236e01,
        -7.94109510e01,
        -8.66941027e01,
        -9.51612705e01,
        -1.04586762e02,
        -1.14680002e02,
        -1.25424092e02,
        -1.37538484e02,
        -1.50993054e02,
        -1.66832208e02,
        -1.84034820e02,
        -2.02926084e02,
        -2.23890549e02,
        -2.48093486e02,
        -2.75713080e02,
        -3.06078038e02,
        -3.39247893e02,
        -3.75917423e02,
        -4.17115902e02,
        -4.63605147e02,
        -5.15906514e02,
        -5.74281263e02,
        -6.38710545e02,
        -7.09090499e02,
        -7.85233554e02,
        -8.66911314e02,
        -9.53868334e02,
        -1.04585181e03,
        -1.13779879e03,
        -1.22821994e03,
        -1.31552956e03,
        -1.39805878e03,
        -4.83324703e03,
        4.00430717e00,
        7.96743070e-01,
        1.39234573e00,
        1.23657688e00,
        1.44878368e00,
        1.51529330e00,
        1.55460112e00,
        1.56386830e00,
        1.58685749e00,
        1.47272458e00,
        1.46702689e00,
        1.45166945e00,
        1.44090726e00,
        1.39730708e00,
        1.40789273e00,
        1.37933501e00,
        1.38519906e00,
        1.36047368e00,
        1.45234104e00,
        1.45318944e00,
        1.29771202e00,
        1.24588257e00,
        1.32295006e00,
        1.32455077e00,
        1.33749565e00,
        1.30804422e00,
        1.27938187e00,
        1.31108565e00,
        1.30997702e00,
        1.35164135e00,
        1.35436405e00,
        1.36172584e00,
        1.37247700e00,
        1.35942541e00,
        1.20190512e00,
        1.21528357e00,
        1.21952648e00,
        1.23369495e00,
        1.21451603e00,
        1.16057869e00,
        1.14934809e00,
        1.13912837e00,
        1.07073058e00,
        1.03427254e00,
        -7.71324513e-02,
        -2.73721717e-01,
        -3.16685410e-01,
        -4.12339087e-01,
        -4.96062316e-01,
        -4.79708537e-01,
        -9.29919474e-01,
        -8.62814900e-01,
        -9.06162195e-01,
        -6.81189299e-01,
        -8.63298778e-01,
        -8.16054579e-01,
        -9.23877222e-01,
        -9.10811055e-01,
        -9.87880938e-01,
        -1.06439480e00,
        -9.08067734e-01,
        -1.10810385e00,
        -1.15308259e00,
        -1.18854629e00,
        -1.20321451e00,
        -1.25641184e00,
        -1.39558574e00,
        -1.45512970e00,
        -1.73159783e00,
        -1.79171499e00,
        -1.98857832e00,
        -1.99474610e00,
        -2.54596828e00,
        -2.76229618e00,
        -2.91254446e00,
        -3.00742873e00,
        -3.25086411e00,
        -3.44542340e00,
        -3.59962769e00,
        -3.80401297e00,
    ]
)


@pytest.mark.parametrize(
    "input_tuple,ubi_j,p,expected",
    [
        (input_tuple, ubi_j1, p1, expected1),
        (input_tuple, ubi_j2, p2, expected2),
    ],
    ids=["Baseline", "w/ UBI"],
)
def test_euler_equation_solver(input_tuple, ubi_j, p, expected):
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    guesses, r, w, bq, rm, tr, _, factor, j = input_tuple
    args = (r, w, 1.0, bq, rm, tr, ubi_j, factor, j, p)
    test_list = SS.euler_equation_solver(guesses, *args)
    print(repr(test_list))

    assert np.allclose(np.array(test_list), np.array(expected))


param_updates1 = {"initial_guess_r_SS": 0.035}
filename1 = "run_SS_baseline_outputs.pkl"
param_updates2 = {
    "use_zeta": True,
    "initial_guess_r_SS": 0.065,
    "initial_guess_TR_SS": 0.06,
}
filename2 = "run_SS_baseline_use_zeta.pkl"
param_updates3 = {"zeta_K": [1.0], "initial_guess_r_SS": 0.04}
filename3 = "run_SS_baseline_small_open.pkl"
param_updates4 = {
    "zeta_K": [1.0],
    "use_zeta": True,
    "initial_guess_r_SS": 0.12,
    "initial_guess_TR_SS": 0.04,
    # "initial_guess_r_SS": 0.033092316727737416,
    # "initial_guess_TR_SS": 0.06323878350496814,
    # "initial_guess_w_SS": 1.3320748594894016,
    "initial_guess_factor_SS": 111267.90426318572,
}
filename4 = "run_SS_baseline_small_open_use_zeta.pkl"
param_updates5 = {"initial_guess_r_SS": 0.035}
filename5 = "run_SS_reform.pkl"
param_updates6 = {
    "use_zeta": True,
    "initial_guess_r_SS": 0.06,
    "initial_guess_TR_SS": 0.06,
}
filename6 = "run_SS_reform_use_zeta.pkl"
param_updates7 = {"zeta_K": [1.0], "initial_guess_r_SS": 0.04}
filename7 = "run_SS_reform_small_open.pkl"
param_updates8 = {
    "zeta_K": [1.0],
    "use_zeta": True,
    "initial_guess_r_SS": 0.04,
    "initial_guess_TR_SS": 0.07,
}
filename8 = "run_SS_reform_small_open_use_zeta.pkl"
param_updates9 = {
    "baseline_spending": True,
    "initial_guess_r_SS": 0.04,
}
filename9 = "run_SS_reform_baseline_spend.pkl"
param_updates10 = {
    "baseline_spending": True,
    "use_zeta": True,
    "initial_guess_r_SS": 0.065,
    "initial_guess_TR_SS": 0.06,
}
filename10 = "run_SS_reform_baseline_spend_use_zeta.pkl"
param_updates11 = {
    "delta_tau_annual": [[0.0]],
    "zeta_K": [0.0],
    "zeta_D": [0.0],
    "initial_guess_r_SS": 0.015,
}
filename11 = "run_SS_baseline_delta_tau0.pkl"
param_updates12 = {
    "delta_g_annual": 0.02,
    "alpha_I": [0.01],
    "gamma_g": [0.07],
    "initial_guess_r_SS": 0.06,
    "initial_guess_TR_SS": 0.03,
    "initial_Kg_ratio": 0.01,
}
filename12 = "run_SS_baseline_Kg_nonzero.pkl"
param_updates13 = {
    "frisch": 0.41,
    "cit_rate": [[0.21, 0.25, 0.35]],
    "M": 3,
    "I": 3,
    "io_matrix": np.eye(3),
    "epsilon": [1.0, 1.0, 1.0],
    "gamma": [0.3, 0.35, 0.4],
    "gamma_g": [0.1, 0.05, 0.15],
    "alpha_c": [0.2, 0.4, 0.4],
    "initial_guess_r_SS": 0.11,
    "initial_guess_TR_SS": 0.07,
    "alpha_I": [0.01],
    "initial_Kg_ratio": 0.01,
    "debt_ratio_ss": 1.5,
}
filename13 = "run_SS_baseline_M3_Kg_nonzero.pkl"
param_updates14 = {
    "start_year": 2023,
    "budget_balance": True,
    "frisch": 0.41,
    "cit_rate": [[0.21, 0.25, 0.35]],
    "M": 3,
    "I": 3,
    "io_matrix": np.eye(3),
    "epsilon": [1.0, 1.0, 1.0],
    "gamma": [0.3, 0.35, 0.4],
    "gamma_g": [0.0, 0.0, 0.0],
    "alpha_c": [0.2, 0.4, 0.4],
    "initial_guess_r_SS": 0.11,
    "initial_guess_TR_SS": 0.07,
    "debt_ratio_ss": 1.5,
}
filename14 = "run_SS_baseline_M3_Kg_zero.pkl"


# Note that changing the order in which these tests are run will cause
# failures for the baseline spending=True tests which depend on the
# output of the baseline run just prior
@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [
        (True, param_updates1, filename1),
        (False, param_updates9, filename9),
        (True, param_updates2, filename2),
        (False, param_updates10, filename10),
        (True, param_updates3, filename3),
        # (True, param_updates4, filename4),
        (False, param_updates5, filename5),
        (False, param_updates6, filename6),
        (False, param_updates7, filename7),
        # (False, param_updates8, filename8),
        (False, param_updates11, filename11),
        (True, param_updates12, filename12),
        (True, param_updates13, filename13),
        (True, param_updates14, filename14),
    ],
    ids=[
        "Baseline",
        "Reform, baseline spending",
        "Baseline, use zeta",
        "Reform, baseline spending, use zeta",
        "Baseline, small open",
        # "Baseline, small open use zeta",
        "Reform",
        "Reform, use zeta",
        "Reform, small open",
        # "Reform, small open use zeta",
        "Reform, delta_tau=0",
        "Baseline, non-zero Kg",
        "Baseline, M=3, non-zero Kg",
        "Baseline, M=3, zero Kg",
    ],
)
@pytest.mark.local
def test_run_SS(tmpdir, baseline, param_updates, filename, dask_client):
    # Test SS.run_SS function. Provide inputs to function and ensure that
    # output returned matches what it has been before.
    SS.ENFORCE_SOLUTION_CHECKS = True
    # if running reform, then need to solve baseline first to get values
    baseline_dir = os.path.join(tmpdir, "OUTPUT_BASELINE")
    if baseline is False:
        p_base = Specifications(
            output_base=baseline_dir,
            baseline_dir=baseline_dir,
            baseline=True,
            num_workers=NUM_WORKERS,
        )
        param_updates_base = param_updates.copy()
        param_updates_base["baseline_spending"] = False
        p_base.update_specifications(param_updates_base)
        base_ss_outputs = SS.run_SS(p_base, client=dask_client)
        utils.mkdirs(os.path.join(baseline_dir, "SS"))
        ss_path = os.path.join(baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_path, "wb") as f:
            pickle.dump(base_ss_outputs, f)
    # now run specification for test
    p = Specifications(
        baseline=baseline, num_workers=NUM_WORKERS, baseline_dir=baseline_dir
    )
    p.update_specifications(param_updates)
    test_dict = SS.run_SS(p, client=dask_client)
    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", filename)
    )

    try:
        expected_dict["r_p_ss"] = expected_dict.pop("r_hh_ss")
    except KeyError:
        pass
    for k, v in expected_dict.items():
        print("Checking item = ", k)
        assert np.allclose(test_dict[VAR_NAME_MAPPING[k]], v, atol=5e-04)
