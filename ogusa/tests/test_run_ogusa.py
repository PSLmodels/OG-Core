import multiprocessing
from distributed import Client, LocalCluster
import pytest
from ogusa import SS, TPI
import time
import os
from ogusa.execute import runner
from ogusa.utils import safe_read_pickle
import ogusa.output_tables as ot
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


def run_micro_macro(iit_reform, og_spec, guid, client):

    guid = ''
    start_time = time.time()

    REFORM_DIR = os.path.join(CUR_PATH, "OUTPUT_REFORM_" + guid)
    BASELINE_DIR = os.path.join(CUR_PATH, "OUTPUT_BASELINE" + guid)
    tax_func_path_baseline = os.path.join(
        CUR_PATH, 'TxFuncEst_baseline.pkl')
    tax_func_path_reform = os.path.join(
        CUR_PATH, 'TxFuncEst_policy.pkl')

    # Add start year from reform to user parameters
    og_spec['start_year'] = 2018

    with open("log_{}.log".format(guid), 'w') as f:
        f.write("guid: {}\n".format(guid))
        f.write("iit_reform: {}\n".format(iit_reform))
        f.write("og_spec: {}\n".format(og_spec))

    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': True, 'time_path': True, 'baseline': True,
              'og_spec': og_spec, 'run_micro': False,
              'tax_func_path': tax_func_path_baseline, 'guid': guid,
              'client': client, 'num_workers': NUM_WORKERS}
    runner(**kwargs)

    '''
    ------------------------------------------------------------------------
        Run reform
    ------------------------------------------------------------------------
    '''

    output_base = REFORM_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': True, 'time_path': True, 'baseline': False,
              'iit_reform': iit_reform, 'og_spec': og_spec,
              'guid': guid, 'run_micro': False,
              'tax_func_path': tax_func_path_reform,
              'client': client, 'num_workers': NUM_WORKERS}
    runner(**kwargs)
    time.sleep(0.5)
    base_tpi = safe_read_pickle(
        os.path.join(BASELINE_DIR, 'TPI', 'TPI_vars.pkl'))
    base_params = safe_read_pickle(
        os.path.join(BASELINE_DIR, 'model_params.pkl'))
    reform_tpi = safe_read_pickle(
        os.path.join(REFORM_DIR, 'TPI', 'TPI_vars.pkl'))
    reform_params = safe_read_pickle(
        os.path.join(REFORM_DIR, 'model_params.pkl'))
    ans = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['Y', 'C', 'K', 'L', 'r', 'w'], output_type='pct_diff',
        num_years=10, start_year=og_spec['start_year'])
    print("total time was ", (time.time() - start_time))

    return ans


@pytest.mark.full_run
def test_run_micro_macro(dask_client):

    iit_reform = {
        "II_rt1": {2018: 0.09},
        "II_rt2": {2018: 0.135},
        "II_rt3": {2018: 0.225},
        "II_rt4": {2018: 0.252},
        "II_rt5": {2018: 0.297},
        "II_rt6": {2018: 0.315},
        "II_rt7": {2018: 0.3564}
    }
        
    run_micro_macro(iit_reform=iit_reform, og_spec={
        'frisch': 0.44, 'g_y_annual': 0.021}, guid='abc',
                    client=dask_client)
