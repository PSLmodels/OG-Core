"""
This module contains tests of the TPI.py module of the OG-Core model. This
module contains the following tests:
    - test_get_initial_SS_values(), 3 parameterizations
    - test_firstdoughnutring(), 1 parameterization
    - test_twist_doughnut(), 2 parameterizations
    - test_inner_loop(), 1 parameterization
    - test_run_TPI_full_run(), 11 parameterizations, local only
    - test_run_TPI(), 2 parameterizations, local only
    - test_run_TPI_extra(), 8 parameterizations, local only
"""

import multiprocessing
from distributed import Client, LocalCluster
import pytest
import pickle
import numpy as np
import os
import sys
import json
from ogcore import SS, TPI, utils
import ogcore.aggregates as aggr
from ogcore.parameters import Specifications

NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

SS_VAR_NAME_MAPPING = {
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

VAR_NAME_MAPPING = {
    "Y": "Y",
    "B": "B",
    "K": "K",
    "K_f": "K_f",
    "K_d": "K_d",
    "L": "L",
    "C": "C",
    "I": "I",
    "I_total": "I_total",
    "I_d": "I_d",
    "K_g": "K_g",
    "I_g": "I_g",
    "BQ": "BQ",
    "RM": "RM",
    "Y_vec": "Y_m",
    "K_vec": "K_m",
    "L_vec": "L_m",
    "C_vec": "C_i",
    "TR": "TR",
    "agg_pension_outlays": "agg_pension_outlays",
    "G": "G",
    "UBI_path": "UBI",
    "total_tax_revenue": "total_tax_revenue",
    "business_tax_revenue": "business_tax_revenue",
    "iit_payroll_tax_revenue": "iit_payroll_tax_revenue",
    "iit_revenue": "iit_revenue",
    "payroll_tax_revenue": "payroll_tax_revenue",
    "bequest_tax_revenue": "bequest_tax_revenue",
    "wealth_tax_revenue": "wealth_tax_revenue",
    "cons_tax_revenue": "cons_tax_revenue",
    "D": "D",
    "D_f": "D_f",
    "D_d": "D_d",
    "new_borrowing": "new_borrowing",
    "debt_service": "debt_service",
    "new_borrowing_f": "new_borrowing_f",
    "debt_service_f": "debt_service_f",
    "r": "r",
    "r_gov": "r_gov",
    "r_p": "r_p",
    "w": "w",
    "p_m": "p_m",
    "p_i": "p_i",
    "p_tilde": "p_tilde",
    "bmat_splus1": "b_sp1",
    "bmat_s": "b_s",
    "n_mat": "n",
    "c_path": "c",
    "c_i_path": "c_i",
    "bq_path": "bq",
    "rm_path": "rm",
    "tr_path": "tr",
    "ubi_path": "ubi",
    "y_before_tax_mat": "before_tax_income",
    "tax_path": "hh_taxes",
    "etr_path": "etr",
    "mtrx_path": "mtrx",
    "mtry_path": "mtry",
    "theta": "theta",
    "factor": "factor",
    "euler_savings": "euler_savings",
    "euler_laborleisure": "euler_labor_leisure",
    "eul_savings": "euler_savings",
    "eul_laborleisure": "euler_labor_leisure",
    "resource_constraint_error": "resource_constraint_error",
}

TEST_PARAM_DICT = json.load(
    open(os.path.join(CUR_PATH, "testing_params.json"))
)


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


filename1 = "intial_SS_values_baseline.pkl"
filename2 = "intial_SS_values_reform.pkl"
filename3 = "intial_SS_values_reform_base_spend.pkl"


@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [
        (True, {}, filename1),
        (False, {}, filename2),
        (False, {"baseline_spending": True}, filename3),
    ],
    ids=["Baseline", "Reform", "Reform, baseline_spending"],
)
def test_get_initial_SS_values(baseline, param_updates, filename, tmpdir):
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)

    old_baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT")
    ss_vars = utils.safe_read_pickle(
        os.path.join(old_baseline_dir, "SS", "SS_vars.pkl")
    )
    ss_vars_new = {}
    for k, v in ss_vars.items():
        ss_vars_new[SS_VAR_NAME_MAPPING[k]] = v
    tpi_vars = utils.safe_read_pickle(
        os.path.join(old_baseline_dir, "TPI", "TPI_vars.pkl")
    )
    tpi_vars_new = {}
    for k, v in tpi_vars.items():
        tpi_vars_new[VAR_NAME_MAPPING[k]] = v
    baseline_dir = os.path.join(tmpdir, "baseline")
    ss_dir = os.path.join(baseline_dir, "SS")
    utils.mkdirs(ss_dir)
    tpi_dir = os.path.join(baseline_dir, "TPI")
    utils.mkdirs(tpi_dir)
    ss_path = os.path.join(baseline_dir, "SS", "SS_vars.pkl")
    with open(ss_path, "wb") as f:
        pickle.dump(ss_vars_new, f)
    tpi_path = os.path.join(baseline_dir, "TPI", "TPI_vars.pkl")
    with open(tpi_path, "wb") as f:
        pickle.dump(tpi_vars_new, f)

    p.baseline_dir = baseline_dir
    p.output_base = baseline_dir
    test_tuple = TPI.get_initial_SS_values(p)
    (
        test_initial_values,
        test_ss_vars,
        test_theta,
        test_baseline_values,
    ) = test_tuple
    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", filename)
    )
    (
        exp_initial_values,
        exp_ss_vars,
        exp_theta,
        exp_baseline_values,
    ) = expected_tuple

    for k, v in enumerate(exp_initial_values):
        assert np.allclose(test_initial_values[k], v, equal_nan=True)

    if p.baseline_spending:
        for k, v in enumerate(exp_baseline_values):
            assert np.allclose(test_baseline_values[k], v, equal_nan=True)

    assert np.allclose(test_theta, exp_theta)

    for k, v in exp_ss_vars.items():
        assert np.allclose(
            test_ss_vars[SS_VAR_NAME_MAPPING[k]], v, equal_nan=True
        )


def test_firstdoughnutring():
    # Test TPI.firstdoughnutring function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "firstdoughnutring_inputs.pkl")
    )
    guesses, r, w, bq, rm, tr, theta, factor, ubi, j, initial_b = input_tuple
    p_tilde = 1.0  # needed for multi-industry version
    p = Specifications()
    test_list = TPI.firstdoughnutring(
        guesses, r, w, p_tilde, bq, rm, tr, theta, factor, ubi, j, initial_b, p
    )

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "firstdoughnutring_outputs.pkl")
    )

    assert np.allclose(np.array(test_list), np.array(expected_list))


file_in1 = os.path.join(
    CUR_PATH, "test_io_data", "twist_doughnut_inputs_2.pkl"
)
file_in2 = os.path.join(
    CUR_PATH, "test_io_data", "twist_doughnut_inputs_S.pkl"
)
file_out1 = os.path.join(
    CUR_PATH, "test_io_data", "twist_doughnut_outputs_2.pkl"
)
file_out2 = os.path.join(
    CUR_PATH, "test_io_data", "twist_doughnut_outputs_S.pkl"
)


@pytest.mark.parametrize(
    "file_inputs,file_outputs",
    [(file_in1, file_out1), (file_in2, file_out2)],
    ids=["s<S", "s==S"],
)
def test_twist_doughnut(file_inputs, file_outputs):
    """
    Test TPI.twist_doughnut function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    """
    input_tuple = utils.safe_read_pickle(file_inputs)
    (
        guesses,
        r,
        w,
        bq,
        rm,
        tr,
        theta,
        factor,
        ubi,
        j,
        s,
        t,
        tau_c,
        etr_params,
        mtrx_params,
        mtry_params,
        initial_b,
    ) = input_tuple
    p_tilde = np.ones_like(r)  # needed for multi-industry version
    p = Specifications()
    input_tuple2 = (
        guesses,
        r,
        w,
        p_tilde,
        bq,
        rm,
        tr,
        theta,
        factor,
        ubi,
        j,
        s,
        t,
        etr_params,
        mtrx_params,
        mtry_params,
        initial_b,
        p,
    )
    test_list = TPI.twist_doughnut(*input_tuple2)
    expected_list = utils.safe_read_pickle(file_outputs)
    assert np.allclose(np.array(test_list), np.array(expected_list), atol=1e-5)


def test_inner_loop():
    # Test TPI.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "tpi_inner_loop_inputs.pkl")
    )
    guesses, outer_loop_vars_old, initial_values, ubi, j, ind = input_tuple
    p = Specifications()
    r = outer_loop_vars_old[0]
    r_p = outer_loop_vars_old[2]
    w = outer_loop_vars_old[1]
    BQ = outer_loop_vars_old[3]
    RM = outer_loop_vars_old[4]
    TR = outer_loop_vars_old[5]
    theta = outer_loop_vars_old[6]
    p_m = np.ones((p.T + p.S, p.M))
    outer_loop_vars = (r_p, r, w, p_m, BQ, RM, TR, theta)
    test_tuple = TPI.inner_loop(
        guesses, outer_loop_vars, initial_values, ubi, j, ind, p
    )
    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "tpi_inner_loop_outputs.pkl")
    )

    for i, v in enumerate(expected_tuple):
        assert np.allclose(test_tuple[i], v)


filename1 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline.pkl"
)
param_updates2 = {"budget_balance": True, "alpha_G": [0.0], "zeta_D": [0.0]}
filename2 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_balanced_budget.pkl"
)
filename3 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_reform.pkl"
)
param_updates4 = {"baseline_spending": True}
filename4 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_reform_baseline_spend.pkl"
)
param_updates5 = {"zeta_K": [1.0], "initial_guess_r_SS": 0.04}
filename5 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_small_open.pkl"
)
param_updates6 = {
    "zeta_K": [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2],
    "initial_guess_r_SS": 0.10,
}
filename6 = os.path.join(
    CUR_PATH,
    "test_io_data",
    "run_TPI_outputs_baseline_small_open_some_periods.pkl",
)
param_updates7 = {
    "delta_tau_annual": [[0.0]],
    "zeta_K": [0.0],
    "zeta_D": [0.0],
    "initial_guess_r_SS": 0.015,
}
filename7 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_delta_tau0.pkl"
)
param_updates8 = {
    "delta_g_annual": 0.02,
    "alpha_I": [0.01],
    "gamma_g": [0.07],
    "initial_Kg_ratio": 0.15,
    "initial_guess_r_SS": 0.06,
    "initial_guess_TR_SS": 0.03,
}
filename8 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_Kg_nonzero.pkl"
)
param_updates9 = {
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
filename9 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_baseline_M3_Kg_nonzero.pkl"
)
alpha_T = np.zeros(50)  # Adjusting the path of transfer spending
alpha_T[0:2] = 0.09
alpha_T[2:10] = 0.09 + 0.01
alpha_T[10:40] = 0.09 - 0.01
alpha_T[40:] = 0.09
alpha_G = np.zeros(7)  # Adjusting the path of non-transfer spending
alpha_G[0:3] = 0.05 - 0.01
alpha_G[3:6] = 0.05 - 0.005
alpha_G[6:] = 0.05
param_updates10 = {
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
    "alpha_T": alpha_T.tolist(),
    "alpha_G": alpha_G.tolist(),
}
filename10 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_baseline_M3_Kg_zero.pkl"
)
param_updates11 = {
    "start_year": 2023,
    "budget_balance": True,
    "frisch": 0.41,
    "cit_rate": [[0.21, 0.25, 0.35]],
    "M": 3,
    "I": 4,
    "io_matrix": np.array(
        [
            [0.3, 0.3, 0.4],
            [0.6, 0.1, 0.3],
            [0.25, 0.5, 0.25],
            [0.0, 1.0, 0.0],
        ]
    ),
    "epsilon": [1.0, 1.0, 1.0],
    "gamma": [0.3, 0.35, 0.4],
    "gamma_g": [0.0, 0.0, 0.0],
    "alpha_c": [0.2, 0.4, 0.3, 0.1],
    "initial_guess_r_SS": 0.11,
    "initial_guess_TR_SS": 0.07,
    "debt_ratio_ss": 1.5,
    "alpha_T": alpha_T.tolist(),
    "alpha_G": alpha_G.tolist(),
}
filename11 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_baseline_MneI.pkl"
)


@pytest.mark.local
@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [
        (True, param_updates2, filename2),
        (True, {"initial_guess_r_SS": 0.035}, filename1),
        (False, {}, filename3),
        (False, param_updates4, filename4),
        (True, param_updates5, filename5),
        (True, param_updates6, filename6),
        (True, param_updates7, filename7),
        (True, param_updates8, filename8),
        (True, param_updates9, filename9),
        (True, param_updates10, filename10),
        (True, param_updates11, filename11),
    ],
    ids=[
        "Baseline, balanced budget",
        "Baseline",
        "Reform",
        "Reform, baseline spending",
        "Baseline, small open",
        "Baseline, small open some periods",
        "Baseline, delta_tau = 0",
        "Baseline, Kg > 0",
        "Baseline, M=3 non-zero Kg",
        "Baseline, M=3 zero Kg",
        "Baseline, M!=I",
    ],
)
def test_run_TPI_full_run(
    baseline, param_updates, filename, tmpdir, dask_client
):
    """
    Test TPI.run_TPI function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    """
    if baseline:
        baseline_dir = os.path.join(tmpdir, "baseline")
        output_base = baseline_dir
    else:
        old_baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT")
        ss_vars = utils.safe_read_pickle(
            os.path.join(old_baseline_dir, "SS", "SS_vars.pkl")
        )
        ss_vars_new = {}
        for k, v in ss_vars.items():
            ss_vars_new[SS_VAR_NAME_MAPPING[k]] = v
        tpi_vars = utils.safe_read_pickle(
            os.path.join(old_baseline_dir, "TPI", "TPI_vars.pkl")
        )
        tpi_vars_new = {}
        for k, v in tpi_vars.items():
            tpi_vars_new[VAR_NAME_MAPPING[k]] = v
        baseline_dir = os.path.join(tmpdir, "baseline")
        ss_dir = os.path.join(baseline_dir, "SS")
        utils.mkdirs(ss_dir)
        tpi_dir = os.path.join(baseline_dir, "TPI")
        utils.mkdirs(tpi_dir)
        ss_path = os.path.join(baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_path, "wb") as f:
            pickle.dump(ss_vars_new, f)
        tpi_path = os.path.join(baseline_dir, "TPI", "TPI_vars.pkl")
        with open(tpi_path, "wb") as f:
            pickle.dump(tpi_vars_new, f)
        output_base = os.path.join(tmpdir, "reform")

    p = Specifications(
        baseline=baseline,
        baseline_dir=baseline_dir,
        output_base=output_base,
        num_workers=NUM_WORKERS,
    )
    p.update_specifications(param_updates)

    # Need to run SS first to get results
    SS.ENFORCE_SOLUTION_CHECKS = True
    ss_outputs = SS.run_SS(p, client=dask_client)

    if p.baseline:
        utils.mkdirs(os.path.join(p.baseline_dir, "SS"))
        ss_dir = os.path.join(p.baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)
    else:
        utils.mkdirs(os.path.join(p.output_base, "SS"))
        ss_dir = os.path.join(p.output_base, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)

    test_dict = TPI.run_TPI(p, client=dask_client)
    expected_dict = utils.safe_read_pickle(filename)
    try:
        expected_dict["r_p"] = expected_dict.pop("r_hh")
        test_dict["euler_savings"] = (
            test_dict["euler_savings"][:, :, :].max(1).max(1)
        )
        test_dict["euler_labor_leisure"] = (
            test_dict["euler_labor_leisure"][:, :, :].max(1).max(1)
        )
    except KeyError:
        pass

    for k, v in expected_dict.items():
        print("Testing, ", k)
        try:
            print(
                "Diff = ",
                np.abs(test_dict[VAR_NAME_MAPPING[k]][: p.T] - v[: p.T]).max(),
            )
        except ValueError:
            print(
                "Diff = ",
                np.abs(
                    test_dict[VAR_NAME_MAPPING[k]][: p.T, :, :]
                    - v[: p.T, :, :]
                ).max(),
            )

    for k, v in expected_dict.items():
        print("Testing, ", k)
        try:
            print(
                "Diff = ",
                np.abs(test_dict[VAR_NAME_MAPPING[k]][: p.T] - v[: p.T]).max(),
            )
            assert np.allclose(
                test_dict[VAR_NAME_MAPPING[k]][: p.T],
                v[: p.T],
                rtol=1e-04,
                atol=1e-04,
            )
        except ValueError:
            print(
                "Diff = ",
                np.abs(
                    test_dict[VAR_NAME_MAPPING[k]][: p.T, :, :]
                    - v[: p.T, :, :]
                ).max(),
            )
            assert np.allclose(
                test_dict[VAR_NAME_MAPPING[k]][: p.T, :, :],
                v[: p.T, :, :],
                rtol=1e-04,
                atol=1e-04,
            )


filename1 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_2.pkl"
)
param_updates2 = {"budget_balance": True, "alpha_G": [0.0], "zeta_D": [0.0]}
filename2 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_balanced_budget_2.pkl"
)
filename3 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_reform_2.pkl"
)
param_updates4 = {"baseline_spending": True}
filename4 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_reform_baseline_spend_2.pkl"
)


@pytest.mark.local
@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [(True, {}, filename1), (False, {}, filename3)],
    ids=["Baseline", "Reform"],
)
def test_run_TPI(baseline, param_updates, filename, tmpdir, dask_client):
    """
    Test TPI.run_TPI function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    """
    if baseline:
        baseline_dir = os.path.join(tmpdir, "baseline")
        output_base = baseline_dir
    else:
        # If running reform, used cached baseline results
        old_baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT2")
        # map new var names and save to tmpdir
        ss_vars = utils.safe_read_pickle(
            os.path.join(old_baseline_dir, "SS", "SS_vars.pkl")
        )
        ss_vars_new = {}
        for k, v in ss_vars.items():
            ss_vars_new[SS_VAR_NAME_MAPPING[k]] = v
        tpi_vars = utils.safe_read_pickle(
            os.path.join(old_baseline_dir, "TPI", "TPI_vars.pkl")
        )
        tpi_vars_new = {}
        for k, v in tpi_vars.items():
            tpi_vars_new[VAR_NAME_MAPPING[k]] = v
        baseline_dir = os.path.join(tmpdir, "baseline")
        ss_dir = os.path.join(baseline_dir, "SS")
        utils.mkdirs(ss_dir)
        tpi_dir = os.path.join(baseline_dir, "TPI")
        utils.mkdirs(tpi_dir)
        ss_path = os.path.join(baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_path, "wb") as f:
            pickle.dump(ss_vars_new, f)
        tpi_path = os.path.join(baseline_dir, "TPI", "TPI_vars.pkl")
        with open(tpi_path, "wb") as f:
            pickle.dump(tpi_vars_new, f)
        output_base = os.path.join(tmpdir, "reform")
    p = Specifications(
        baseline=baseline,
        baseline_dir=baseline_dir,
        output_base=output_base,
        num_workers=NUM_WORKERS,
    )
    test_params = TEST_PARAM_DICT.copy()
    test_params.update(param_updates)
    p.update_specifications(test_params)
    p.maxiter = 2  # this test runs through just two iterations

    # Need to run SS first to get results
    SS.ENFORCE_SOLUTION_CHECKS = False
    ss_outputs = SS.run_SS(p, client=dask_client)

    if p.baseline:
        utils.mkdirs(os.path.join(p.baseline_dir, "SS"))
        ss_dir = os.path.join(p.baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)
    else:
        utils.mkdirs(os.path.join(p.output_base, "SS"))
        ss_dir = os.path.join(p.output_base, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)

    TPI.ENFORCE_SOLUTION_CHECKS = False
    test_dict = TPI.run_TPI(p, client=dask_client)
    expected_dict = utils.safe_read_pickle(filename)

    for k, v in expected_dict.items():
        print("Max diff in ", k, " = ")
        try:
            print(
                np.absolute(
                    test_dict[VAR_NAME_MAPPING[k]][: p.T] - v[: p.T]
                ).max()
            )
        except ValueError:
            print(
                np.absolute(
                    test_dict[VAR_NAME_MAPPING[k]][: p.T, :, :]
                    - v[: p.T, :, :]
                ).max()
            )

    for k, v in expected_dict.items():
        try:
            assert np.allclose(
                test_dict[VAR_NAME_MAPPING[k]][: p.T],
                v[: p.T],
                rtol=1e-04,
                atol=1e-04,
            )
        except ValueError:
            assert np.allclose(
                test_dict[VAR_NAME_MAPPING[k]][: p.T, :, :],
                v[: p.T, :, :],
                rtol=1e-04,
                atol=1e-04,
            )


param_updates5 = {"zeta_K": [1.0]}
filename5 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_small_open_2.pkl"
)
param_updates6 = {
    "zeta_K": [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2],
    "initial_guess_r_SS": 0.10,
}
filename6 = filename = os.path.join(
    CUR_PATH,
    "test_io_data",
    "run_TPI_outputs_baseline_small_open_some_periods_2.pkl",
)
param_updates7 = {
    "delta_tau_annual": [[0.0]],
    "zeta_K": [0.0],
    "zeta_D": [0.0],
    "initial_guess_r_SS": 0.01,
}
filename7 = filename = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_delta_tau0_2.pkl"
)
param_updates8 = {
    "delta_g_annual": 0.02,
    "alpha_I": [0.01],
    "gamma_g": [0.07],
    "initial_Kg_ratio": 0.15,
    "initial_guess_r_SS": 0.06,
    "initial_guess_TR_SS": 0.03,
}
filename8 = os.path.join(
    CUR_PATH, "test_io_data", "run_TPI_outputs_baseline_Kg_nonzero_2.pkl"
)
# read in mono tax funcs (not age specific)
if sys.version_info[1] < 11:
    dict_params = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_mono_nonage.pkl")
    )
    p = Specifications()
    etr_params = [[None] * p.S] * p.T
    mtrx_params = [[None] * p.S] * p.T
    mtry_params = [[None] * p.S] * p.T
    for s in range(p.S):
        for t in range(p.T):
            if t < p.BW:
                etr_params[t][s] = dict_params["tfunc_etr_params_S"][t][s]
                mtrx_params[t][s] = dict_params["tfunc_mtrx_params_S"][t][s]
                mtry_params[t][s] = dict_params["tfunc_mtry_params_S"][t][s]
            else:
                etr_params[t][s] = dict_params["tfunc_etr_params_S"][-1][s]
                mtrx_params[t][s] = dict_params["tfunc_mtrx_params_S"][-1][s]
                mtry_params[t][s] = dict_params["tfunc_mtry_params_S"][-1][s]
    param_updates9 = {
        "tax_func_type": "mono",
        "etr_params": etr_params,
        "mtrx_params": mtrx_params,
        "mtry_params": mtry_params,
    }
    filename9 = os.path.join(
        CUR_PATH, "test_io_data", "run_TPI_outputs_mono_2.pkl"
    )

if sys.version_info[1] < 11:
    test_list = [
        (True, param_updates2, filename2),
        (True, param_updates5, filename5),
        (True, param_updates6, filename6),
        (True, param_updates7, filename7),
        (True, {}, filename1),
        (False, param_updates4, filename4),
        (True, param_updates8, filename8),
        (True, param_updates9, filename9),
    ]
    id_list = [
        "Baseline, balanced budget",
        "Baseline, small open",
        "Baseline, small open for some periods",
        "Baseline, delta_tau = 0",
        "Baseline",
        "Reform, baseline spending",
        "Baseline, Kg>0",
        "mono tax functions",
    ]
else:
    test_list = [
        (True, param_updates2, filename2),
        (True, param_updates5, filename5),
        (True, param_updates6, filename6),
        (True, param_updates7, filename7),
        (True, {}, filename1),
        (False, param_updates4, filename4),
        (True, param_updates8, filename8),
    ]
    id_list = [
        "Baseline, balanced budget",
        "Baseline, small open",
        "Baseline, small open for some periods",
        "Baseline, delta_tau = 0",
        "Baseline",
        "Reform, baseline spending",
        "Baseline, Kg>0",
    ]


@pytest.mark.local
@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    test_list,
    ids=id_list,
)
def test_run_TPI_extra(baseline, param_updates, filename, tmpdir, dask_client):
    """
    Test TPI.run_TPI function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    """
    if baseline:
        baseline_dir = os.path.join(tmpdir, "baseline")
        output_base = baseline_dir
    else:
        old_baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT2")
        ss_vars = utils.safe_read_pickle(
            os.path.join(old_baseline_dir, "SS", "SS_vars.pkl")
        )
        ss_vars_new = {}
        for k, v in ss_vars.items():
            ss_vars_new[SS_VAR_NAME_MAPPING[k]] = v
        tpi_vars = utils.safe_read_pickle(
            os.path.join(old_baseline_dir, "TPI", "TPI_vars.pkl")
        )
        tpi_vars_new = {}
        for k, v in tpi_vars.items():
            tpi_vars_new[VAR_NAME_MAPPING[k]] = v
        baseline_dir = os.path.join(tmpdir, "baseline")
        ss_dir = os.path.join(baseline_dir, "SS")
        utils.mkdirs(ss_dir)
        tpi_dir = os.path.join(baseline_dir, "TPI")
        utils.mkdirs(tpi_dir)
        ss_path = os.path.join(baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_path, "wb") as f:
            pickle.dump(ss_vars_new, f)
        tpi_path = os.path.join(baseline_dir, "TPI", "TPI_vars.pkl")
        with open(tpi_path, "wb") as f:
            pickle.dump(tpi_vars_new, f)
        output_base = os.path.join(tmpdir, "reform")
    p = Specifications(
        baseline=baseline,
        baseline_dir=baseline_dir,
        output_base=output_base,
        num_workers=NUM_WORKERS,
    )
    test_dict = TEST_PARAM_DICT.copy()
    test_dict.update(param_updates)
    p.update_specifications(test_dict)
    p.maxiter = 2  # this test runs through just two iterations

    # Need to run SS first to get results
    SS.ENFORCE_SOLUTION_CHECKS = False
    ss_outputs = SS.run_SS(p, client=dask_client)

    if p.baseline:
        utils.mkdirs(os.path.join(p.baseline_dir, "SS"))
        ss_path = os.path.join(p.baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_path, "wb") as f:
            pickle.dump(ss_outputs, f)
    else:
        utils.mkdirs(os.path.join(p.output_base, "SS"))
        ss_path = os.path.join(p.output_base, "SS", "SS_vars.pkl")
        with open(ss_path, "wb") as f:
            pickle.dump(ss_outputs, f)

    TPI.ENFORCE_SOLUTION_CHECKS = False
    test_dict = TPI.run_TPI(p, client=dask_client)
    expected_dict = utils.safe_read_pickle(filename)

    for k, v in expected_dict.items():
        print("Checking ", k)
        try:
            print(
                "Diff = ",
                np.absolute(
                    test_dict[VAR_NAME_MAPPING[k]][: p.T] - v[: p.T]
                ).max(),
            )
            assert np.allclose(
                test_dict[VAR_NAME_MAPPING[k]][: p.T],
                v[: p.T],
                rtol=1e-04,
                atol=1e-04,
            )
        except ValueError:
            print(
                "Diff = ",
                np.absolute(
                    test_dict[VAR_NAME_MAPPING[k]][: p.T, :, :]
                    - v[: p.T, :, :]
                ).max(),
            )
            assert np.allclose(
                test_dict[VAR_NAME_MAPPING[k]][: p.T, :, :],
                v[: p.T, :, :],
                rtol=1e-04,
                atol=1e-04,
            )
