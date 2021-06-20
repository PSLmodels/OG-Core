import multiprocessing
from distributed import Client, LocalCluster
import pytest
from ogusa import SS, TPI
from ogusa.execute import runner
from ogusa.parameters import Specifications
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
    p = Specifications()
    p.update_specifications(TEST_PARAM_DICT)
    p.baseline_dir = p.output_base = BASELINE_DIR
    runner(p, time_path=True, client=dask_client)


@pytest.mark.local
def test_runner_reform(dask_client):
    p = Specifications()
    p.update_specifications(TEST_PARAM_DICT)
    p.baseline_dir = BASELINE_DIR
    p.output_base = REFORM_DIR
    runner(p, time_path=False, client=dask_client)
