import multiprocessing
from distributed import Client, LocalCluster
import pytest
from ogusa import SS, TPI
from ogusa.execute import runner
import os
NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
BASELINE_DIR = os.path.join(CUR_PATH, 'OUTPUT_BASELINE')
REFORM_DIR = os.path.join(CUR_PATH, 'OUTPUT_REFORM')

# Monkey patch enforcement flag since small data won't pass checks
SS.ENFORCE_SOLUTION_CHECKS = False
TPI.ENFORCE_SOLUTION_CHECKS = False


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


@pytest.mark.full_run
def test_runner_baseline(dask_client):
    runner(output_base=BASELINE_DIR, baseline_dir=BASELINE_DIR,
           test=True, time_path=True, baseline=True, run_micro=False,
           data='cps', client=dask_client, num_workers=NUM_WORKERS)


@pytest.mark.full_run
def test_runner_reform(dask_client):
    runner(output_base=REFORM_DIR, baseline_dir=BASELINE_DIR,
           test=True, time_path=False, baseline=False, run_micro=False,
           data='cps', client=dask_client, num_workers=NUM_WORKERS)
