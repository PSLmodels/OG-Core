import multiprocessing
from distributed import Client, LocalCluster
import pytest
import os
from ogcore.execute import runner
from ogcore.parameters import Specifications

NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(CUR_PATH, "OUTPUT")


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


@pytest.mark.local
@pytest.mark.parametrize(
    "frisch",
    [0.32, 0.4, 0.62],
    ids=["Frisch 0.32", "Frisch 0.4", "Frisch 0.6"],
)
def test_frisch(tmpdir, frisch, dask_client):
    if frisch == 0.4:
        og_spec = {
            "frisch": frisch,
            "debt_ratio_ss": 1.0,
            "initial_guess_r_SS": 0.06,
        }
    else:
        og_spec = {"frisch": frisch, "debt_ratio_ss": 1.0}
    p = Specifications(
        baseline=True,
        num_workers=NUM_WORKERS,
        baseline_dir=tmpdir,
        output_base=tmpdir,
    )
    p.update_specifications(og_spec)
    runner(p, time_path=False, client=dask_client)


@pytest.mark.local
@pytest.mark.parametrize("g_y_annual", [0.0, 0.04], ids=["0.0", "0.04"])
def test_gy(tmpdir, g_y_annual, dask_client):
    if g_y_annual == 0.0:
        og_spec = {
            "frisch": 0.41,
            "debt_ratio_ss": 1.0,
            "g_y_annual": g_y_annual,
            "initial_guess_r_SS": 0.06,
        }
    else:
        og_spec = {
            "frisch": 0.41,
            "debt_ratio_ss": 1.0,
            "g_y_annual": g_y_annual,
        }
    p = Specifications(
        baseline=True,
        num_workers=NUM_WORKERS,
        baseline_dir=tmpdir,
        output_base=tmpdir,
    )
    p.update_specifications(og_spec)
    runner(p, time_path=False, client=dask_client)


@pytest.mark.local
@pytest.mark.parametrize(
    "sigma",
    [1.3, 1.5, 1.7, 1.9],
    ids=["sigma=1.3", "sigma=1.5", "sigma=1.7", "sigma=1.9"],
)
def test_sigma(tmpdir, sigma, dask_client):
    og_spec = {"frisch": 0.41, "debt_ratio_ss": 1.0, "sigma": sigma}
    p = Specifications(
        baseline=True,
        num_workers=NUM_WORKERS,
        baseline_dir=tmpdir,
        output_base=tmpdir,
    )
    p.update_specifications(og_spec)
    runner(p, time_path=False, client=dask_client)
