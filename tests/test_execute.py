import multiprocessing
from distributed import Client, LocalCluster
import pytest
from ogcore import SS, TPI
from ogcore.execute import runner
from ogcore.parameters import Specifications
import os
import json

NUM_WORKERS = min(multiprocessing.cpu_count(), 7)

# Set paths
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

# Monkey patch enforcement flag since small data won't pass checks
SS.ENFORCE_SOLUTION_CHECKS = False
TPI.ENFORCE_SOLUTION_CHECKS = False

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
def test_runner_baseline_reform(tmpdir, dask_client):
    # Run baseline runner(). If errors out, test will fail
    p_b = Specifications(baseline=True, num_workers=NUM_WORKERS)
    p_b.update_specifications(TEST_PARAM_DICT)
    p_b.baseline_dir = p_b.output_base = os.path.join(
        tmpdir, "OUTPUT_BASELINE"
    )
    runner(p_b, time_path=True, client=dask_client)

    # Run reform runner(). If errors out, test will fail. These two have to be
    # run in the same test because the reform run below depends on output saved
    # in the baseline run above.
    p_r = Specifications(baseline=False, num_workers=NUM_WORKERS)
    p_r.update_specifications(TEST_PARAM_DICT)
    p_r.baseline_dir = os.path.join(tmpdir, "OUTPUT_BASELINE")
    p_r.output_base = os.path.join(tmpdir, "OUTPUT_REFORM")
    runner(p_r, time_path=False, client=dask_client)
