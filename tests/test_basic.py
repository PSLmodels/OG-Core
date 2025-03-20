import multiprocessing
from distributed import Client, LocalCluster
import os
import json
import pytest
import pickle
import numpy as np
from ogcore import SS, TPI, utils
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


@pytest.mark.local
@pytest.mark.parametrize("time_path", [False, True], ids=["SS", "TPI"])
def test_run_small(tmpdir, time_path, dask_client):
    from ogcore.execute import runner

    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    SS.MINIMIZER_TOL = 1e-6
    TPI.MINIMIZER_TOL = 1e-6
    p = Specifications(
        baseline=True,
        num_workers=NUM_WORKERS,
        baseline_dir=tmpdir,
        output_base=tmpdir,
    )
    p.update_specifications(TEST_PARAM_DICT)
    runner(p, time_path=time_path, client=dask_client)


@pytest.mark.local
def test_constant_demographics_TPI(tmpdir, dask_client):
    """
    This tests solves the model under the assumption of constant
    demographics, a balanced budget, and tax functions that do not vary
    over time.
    In this case, given how initial guesses for the time
    path are made, the time path should be solved for on the first
    iteration and the values all along the time path should equal their
    steady-state values.
    """
    # Create output directory structure
    spec = Specifications(
        output_base=tmpdir,
        baseline_dir=tmpdir,
        baseline=True,
        num_workers=NUM_WORKERS,
    )
    og_spec = {
        "constant_demographics": True,
        "budget_balance": True,
        "maxiter": 2,
        "r_gov_shift": [0.0],
        "zeta_D": [0.0, 0.0],
        "zeta_K": [0.0, 0.0],
        "debt_ratio_ss": 1.0,
        "initial_foreign_debt_ratio": 0.0,
        "start_year": 2019,
        "cit_rate": [[0.0]],
        "PIA_rate_bkt_1": 0.0,
        "PIA_rate_bkt_2": 0.0,
        "PIA_rate_bkt_3": 0.0,
        "eta": (
            spec.omega_SS.reshape(spec.S, 1) * spec.lambdas.reshape(1, spec.J)
        ),
    }
    spec.update_specifications(og_spec)
    spec.etr_params = np.zeros(
        (spec.T + spec.S, spec.S, len(spec.etr_params[0][0]))
    ).tolist()
    spec.mtrx_params = np.zeros(
        (spec.T + spec.S, spec.S, len(spec.mtrx_params[0][0]))
    ).tolist()
    spec.mtry_params = np.zeros(
        (spec.T + spec.S, spec.S, len(spec.mtry_params[0][0]))
    ).tolist()
    # Run SS
    ss_outputs = SS.run_SS(spec, client=dask_client)
    # save SS results
    utils.mkdirs(os.path.join(tmpdir, "SS"))
    ss_dir = os.path.join(tmpdir, "SS", "SS_vars.pkl")
    with open(ss_dir, "wb") as f:
        pickle.dump(ss_outputs, f)
    # Run TPI
    tpi_output = TPI.run_TPI(spec, client=dask_client)
    assert np.allclose(
        tpi_output["b_sp1"][: spec.T, :, :], ss_outputs["b_sp1"]
    )


@pytest.mark.local
def test_constant_demographics_TPI_small_open(tmpdir, dask_client):
    """
    This tests solves the model under the assumption of constant
    demographics, a balanced budget, and tax functions that do not vary
    over time, as well as with a small open economy assumption.
    """
    # Create output directory structure
    spec = Specifications(
        output_base=tmpdir,
        baseline_dir=tmpdir,
        baseline=True,
        num_workers=NUM_WORKERS,
    )
    og_spec = {
        "constant_demographics": True,
        "budget_balance": True,
        "maxiter": 2,
        "r_gov_shift": [0.0],
        "zeta_D": [0.0, 0.0],
        "zeta_K": [1.0],
        "debt_ratio_ss": 1.0,
        "initial_foreign_debt_ratio": 0.0,
        "start_year": 2019,
        "cit_rate": [[0.0]],
        "PIA_rate_bkt_1": 0.0,
        "PIA_rate_bkt_2": 0.0,
        "PIA_rate_bkt_3": 0.0,
        "eta": (
            spec.omega_SS.reshape(spec.S, 1) * spec.lambdas.reshape(1, spec.J)
        ),
    }
    spec.update_specifications(og_spec)
    spec.etr_params = np.zeros(
        (spec.T + spec.S, spec.S, len(spec.etr_params[0][0]))
    ).tolist()
    spec.mtrx_params = np.zeros(
        (spec.T + spec.S, spec.S, len(spec.mtrx_params[0][0]))
    ).tolist()
    spec.mtry_params = np.zeros(
        (spec.T + spec.S, spec.S, len(spec.mtry_params[0][0]))
    ).tolist()
    # Run SS
    ss_outputs = SS.run_SS(spec, client=dask_client)
    # save SS results
    utils.mkdirs(os.path.join(tmpdir, "SS"))
    ss_dir = os.path.join(tmpdir, "SS", "SS_vars.pkl")
    with open(ss_dir, "wb") as f:
        pickle.dump(ss_outputs, f)
    # Run TPI
    tpi_output = TPI.run_TPI(spec, client=dask_client)
    assert np.allclose(
        tpi_output["b_sp1"][: spec.T, :, :], ss_outputs["b_sp1"]
    )
