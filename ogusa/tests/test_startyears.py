import multiprocessing
from distributed import Client, LocalCluster
import pytest
import os
from ogusa import SS, TPI
from ogusa.execute import runner
NUM_WORKERS = 2
CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


@pytest.mark.full_run
@pytest.mark.parametrize(
        'year', [2014, 2017, 2026], ids=['2014', '2017', '2026'])
def test_diff_start_year(year, dask_client):
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    output_base = os.path.join(CUR_PATH, "STARTYEAR_OUTPUT")
    input_dir = os.path.join(CUR_PATH, "STARTYEAR_OUTPUT")
    og_spec = {'frisch': 0.41, 'debt_ratio_ss': 1.0, 'start_year': year}
    runner(output_base=output_base, baseline_dir=input_dir, test=True,
           time_path=True, baseline=True, og_spec=og_spec,
           run_micro=True, data='cps', client=dask_client,
           num_workers=NUM_WORKERS)
