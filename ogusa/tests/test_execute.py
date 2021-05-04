import multiprocessing
from distributed import Client, LocalCluster
import pytest
from ogusa import SS, TPI
from ogusa.execute import runner
import os
import json

NUM_WORKERS = min(multiprocessing.cpu_count(), 7)

# Set paths
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
BASELINE_DIR = os.path.join(CUR_PATH, 'OUTPUT_BASELINE')
REFORM_DIR = os.path.join(CUR_PATH, 'OUTPUT_REFORM')

# Monkey patch enforcement flag since small data won't pass checks
SS.ENFORCE_SOLUTION_CHECKS = False
TPI.ENFORCE_SOLUTION_CHECKS = False

TEST_PARAM_DICT = json.load(open(os.path.join(CUR_PATH, 'testing_params.json')))


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


@pytest.mark.local
def test_runner_baseline(dask_client):
    runner(output_base=BASELINE_DIR, baseline_dir=BASELINE_DIR,
           time_path=True, baseline=True,
           og_spec=TEST_PARAM_DICT, client=dask_client,
           num_workers=NUM_WORKERS)


@pytest.mark.local
def test_runner_reform(dask_client):
    runner(output_base=REFORM_DIR, baseline_dir=BASELINE_DIR,
           time_path=False, baseline=False,
           og_spec=TEST_PARAM_DICT, client=dask_client,
           num_workers=NUM_WORKERS)
