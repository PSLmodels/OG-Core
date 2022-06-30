import multiprocessing
from distributed import Client, LocalCluster
import pytest
import pickle
import numpy as np
import os
import json
from ogcore import SS, TPI, utils
import ogcore.aggregates as aggr
from ogcore.parameters import Specifications

NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

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
def test_get_initial_SS_values(baseline, param_updates, filename, dask_client):
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT")
    p.output_base = os.path.join(CUR_PATH, "test_io_data", "OUTPUT")
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
    (
        B0,
        b_sinit,
        b_splus1init,
        factor,
        initial_b,
        initial_n,
    ) = exp_initial_values
    B0 = aggr.get_B(exp_ss_vars["bssmat_splus1"], p, "SS", True)
    initial_b = exp_ss_vars["bssmat_splus1"] * (exp_ss_vars["Bss"] / B0)
    B0 = aggr.get_B(initial_b, p, "SS", True)
    b_sinit = np.array(
        list(np.zeros(p.J).reshape(1, p.J)) + list(initial_b[:-1])
    )
    b_splus1init = initial_b
    exp_initial_values = (
        B0,
        b_sinit,
        b_splus1init,
        factor,
        initial_b,
        initial_n,
    )

    for i, v in enumerate(exp_initial_values):
        assert np.allclose(test_initial_values[i], v, equal_nan=True)

    if p.baseline_spending:
        for i, v in enumerate(exp_baseline_values):
            assert np.allclose(test_baseline_values[i], v, equal_nan=True)

    assert np.allclose(test_theta, exp_theta)

    for k, v in exp_ss_vars.items():
        assert np.allclose(test_ss_vars[k], v, equal_nan=True)


def test_firstdoughnutring():
    # Test TPI.firstdoughnutring function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "firstdoughnutring_inputs.pkl")
    )
    guesses, r, w, bq, tr, theta, factor, ubi, j, initial_b = input_tuple
    p_tilde = 1.0  # needed for multi-industry version
    p = Specifications()
    test_list = TPI.firstdoughnutring(
        guesses, r, w, p_tilde, bq, tr, theta, factor, ubi, j, initial_b, p
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
    input_tuple = (
        guesses,
        r,
        w,
        p_tilde,
        bq,
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
    test_list = TPI.twist_doughnut(*input_tuple)
    expected_list = utils.safe_read_pickle(file_outputs)
    assert np.allclose(np.array(test_list), np.array(expected_list))


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
    TR = outer_loop_vars_old[4]
    theta = outer_loop_vars_old[5]
    p_m = np.ones((p.T + p.S, p.M))
    outer_loop_vars = (r_p, r, w, p_m, BQ, TR, theta)
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
param_updates5 = {"zeta_K": [1.0], "initial_guess_r_SS": 0.10}
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
    "initial_guess_r_SS": 0.01,
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


@pytest.mark.local
@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [
        (True, param_updates2, filename2),
        (True, {"initial_guess_r_SS": 0.03}, filename1),
        (False, {}, filename3),
        (False, param_updates4, filename4),
        (True, param_updates5, filename5),
        (True, param_updates6, filename6),
        (True, param_updates7, filename7),
        (True, param_updates8, filename8),
        (True, param_updates9, filename9),
        (True, param_updates10, filename10),
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
        baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT")
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
        test_dict["eul_savings"] = (
            test_dict["eul_savings"][:, :, :].max(1).max(1)
        )
        test_dict["eul_laborleisure"] = (
            test_dict["eul_laborleisure"][:, :, :].max(1).max(1)
        )
    except KeyError:
        pass

    for k, v in expected_dict.items():
        print("Testing, ", k)
        try:
            print("Diff = ", np.abs(test_dict[k][: p.T] - v[: p.T]).max())
        except ValueError:
            print(
                "Diff = ",
                np.abs(test_dict[k][: p.T, :, :] - v[: p.T, :, :]).max(),
            )

    for k, v in expected_dict.items():
        print("Testing, ", k)
        try:
            print("Diff = ", np.abs(test_dict[k][: p.T] - v[: p.T]).max())
            assert np.allclose(
                test_dict[k][: p.T], v[: p.T], rtol=1e-04, atol=1e-04
            )
        except ValueError:
            print(
                "Diff = ",
                np.abs(test_dict[k][: p.T, :, :] - v[: p.T, :, :]).max(),
            )
            assert np.allclose(
                test_dict[k][: p.T, :, :],
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
        baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT2")
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
            print(np.absolute(test_dict[k][: p.T] - v[: p.T]).max())
        except ValueError:
            print(
                np.absolute(test_dict[k][: p.T, :, :] - v[: p.T, :, :]).max()
            )

    for k, v in expected_dict.items():
        try:
            assert np.allclose(
                test_dict[k][: p.T], v[: p.T], rtol=1e-04, atol=1e-04
            )
        except ValueError:
            assert np.allclose(
                test_dict[k][: p.T, :, :],
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


@pytest.mark.local
@pytest.mark.parametrize(
    "baseline,param_updates,filename",
    [
        (True, param_updates2, filename2),
        (True, param_updates5, filename5),
        (True, param_updates6, filename6),
        (True, param_updates7, filename7),
        (True, {}, filename1),
        (False, param_updates4, filename4),
        (True, param_updates8, filename8),
    ],
    ids=[
        "Baseline, balanced budget",
        "Baseline, small open",
        "Baseline, small open for some periods",
        "Baseline, delta_tau = 0",
        "Baseline",
        "Reform, baseline spending",
        "Baseline, Kg>0",
    ],
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
        baseline_dir = os.path.join(CUR_PATH, "test_io_data", "OUTPUT2")
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
        print("Checking ", k)
        try:
            print("Diff = ", np.abs(test_dict[k][: p.T] - v[: p.T]).max())
            assert np.allclose(
                test_dict[k][: p.T], v[: p.T], rtol=1e-04, atol=1e-04
            )
        except ValueError:
            print(
                "Diff = ",
                np.abs(test_dict[k][: p.T, :, :] - v[: p.T, :, :]).max(),
            )
            assert np.allclose(
                test_dict[k][: p.T, :, :],
                v[: p.T, :, :],
                rtol=1e-04,
                atol=1e-04,
            )
