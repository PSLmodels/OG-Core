import multiprocessing
from distributed import Client, LocalCluster
import pytest
from ogcore import SS, TPI
import time
import os
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle
import ogcore.output_tables as ot

SS.ENFORCE_SOLUTION_CHECKS = False
TPI.ENFORCE_SOLUTION_CHECKS = False
NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


@pytest.mark.local
def test_run_micro_macro(tmpdir, dask_client):
    og_spec = os.path.join(CUR_PATH, "testing_params.json")
    guid = ""
    start_time = time.time()

    REFORM_DIR = os.path.join(tmpdir, "OUTPUT_REFORM_" + guid)
    BASELINE_DIR = os.path.join(tmpdir, "OUTPUT_BASELINE" + guid)

    with open("log_{}.log".format(guid), "w") as f:
        f.write("guid: {}\n".format(guid))
        f.write("og_spec: {}\n".format(og_spec))

    """
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    """
    p = Specifications(
        baseline=True,
        num_workers=NUM_WORKERS,
        baseline_dir=BASELINE_DIR,
        output_base=BASELINE_DIR,
    )
    p.update_specifications(og_spec)
    runner(p, time_path=True, client=dask_client)

    """
    ------------------------------------------------------------------------
        Run reform
    ------------------------------------------------------------------------
    """
    p = Specifications(
        baseline=False,
        num_workers=NUM_WORKERS,
        baseline_dir=BASELINE_DIR,
        output_base=REFORM_DIR,
    )
    p.update_specifications(og_spec)
    runner(p, time_path=True, client=dask_client)
    time.sleep(0.5)
    base_tpi = safe_read_pickle(
        os.path.join(BASELINE_DIR, "TPI", "TPI_vars.pkl")
    )
    base_params = safe_read_pickle(
        os.path.join(BASELINE_DIR, "model_params.pkl")
    )
    reform_tpi = safe_read_pickle(
        os.path.join(REFORM_DIR, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(REFORM_DIR, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )
    print("total time was ", (time.time() - start_time))
