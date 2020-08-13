import multiprocessing
from distributed import Client, LocalCluster
import numpy as np
import psutil
import pytest
from ogusa import SS, TPI
from ogusa.execute import runner
import os

# Get named tuple for RAM stats and compute total available RAM in GB and set
# number of workers
num_workers_max_sys = multiprocessing.cpu_count()
RAM_stats = psutil.virtual_memory()
print(RAM_stats)
RAM_total_bytes = RAM_stats.total
RAM_total_GB = RAM_total_bytes /  1073741824
print(RAM_total_GB)
mem_per_wkr_tot = RAM_total_GB / num_workers_max_sys
num_workers_max = min(num_workers_max_sys, 7)
# NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
NUM_WORKERS = np.minimum(num_workers_max,
                         int(np.floor((0.6 * RAM_total_GB) /
                                      mem_per_wkr_tot)))
print('Max num worders=', num_workers_max, ', and NUM_WORKERS=', NUM_WORKERS)


# Set paths
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
BASELINE_DIR = os.path.join(CUR_PATH, 'OUTPUT_BASELINE')
REFORM_DIR = os.path.join(CUR_PATH, 'OUTPUT_REFORM')
BASE_TAX = os.path.join(CUR_PATH, 'TxFuncEst_baseline.pkl')
REFORM_TAX = os.path.join(CUR_PATH, 'TxFuncEst_policy.pkl')

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
           test=True, time_path=True, baseline=True,
           og_spec={'start_year': 2018}, run_micro=False,
           tax_func_path=BASE_TAX, data='cps', client=dask_client,
           num_workers=NUM_WORKERS)


@pytest.mark.full_run
def test_runner_reform(dask_client):
    runner(output_base=REFORM_DIR, baseline_dir=BASELINE_DIR,
           test=True, time_path=False, baseline=False,
           og_spec={'start_year': 2018}, run_micro=False,
           tax_func_path=None, data='cps', client=dask_client,
           num_workers=NUM_WORKERS)
