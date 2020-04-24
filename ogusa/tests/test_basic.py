import multiprocessing
from distributed import Client, LocalCluster
import os
import pytest
import pickle
import numpy as np
from ogusa import SS, TPI, utils
from ogusa.parameters import Specifications
NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
TAX_FUNC_PATH = os.path.join(CUR_PATH, 'TxFuncEst_baseline.pkl')
OUTPUT_DIR = os.path.join(CUR_PATH, "OUTPUT")


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


@pytest.mark.full_run
@pytest.mark.parametrize('time_path', [False, True], ids=['SS', 'TPI'])
def test_run_small(time_path, dask_client):
    from ogusa.execute import runner
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    SS.MINIMIZER_TOL = 1e-6
    TPI.MINIMIZER_TOL = 1e-6
    og_spec = {'frisch': 0.41, 'debt_ratio_ss': 0.4}
    runner(output_base=OUTPUT_DIR, baseline_dir=OUTPUT_DIR, test=True,
           time_path=time_path, baseline=True, og_spec=og_spec,
           run_micro=False, tax_func_path=TAX_FUNC_PATH,
           client=dask_client, num_workers=NUM_WORKERS)


@pytest.mark.full_run
def test_constant_demographics_TPI(dask_client):
    '''
    This tests solves the model under the assumption of constant
    demographics, a balanced budget, and tax functions that do not vary
    over time.
    In this case, given how initial guesss for the time
    path are made, the time path should be solved for on the first
    iteration and the values all along the time path should equal their
    steady-state values.
    '''
    # Create output directory structure
    spec = Specifications(run_micro=False, output_base=OUTPUT_DIR,
                          baseline_dir=OUTPUT_DIR, test=False,
                          time_path=True, baseline=True, iit_reform={},
                          guid='', client=dask_client,
                          num_workers=NUM_WORKERS)
    og_spec = {'constant_demographics': True, 'budget_balance': True,
               'zero_taxes': True, 'maxiter': 2,
               'r_gov_shift': 0.0, 'zeta_D': [0.0, 0.0],
               'zeta_K': [0.0, 0.0], 'debt_ratio_ss': 1.0,
               'initial_foreign_debt_ratio': 0.0,
               'start_year': 2019, 'cit_rate': [0.0],
               'PIA_rate_bkt_1': 0.0, 'PIA_rate_bkt_2': 0.0,
               'PIA_rate_bkt_3': 0.0,
               'eta': (spec.omega_SS.reshape(spec.S, 1) *
                       spec.lambdas.reshape(1, spec.J))}
    spec.update_specifications(og_spec)
    spec.get_tax_function_parameters(None, False,
                                     tax_func_path=TAX_FUNC_PATH)
    # Run SS
    ss_outputs = SS.run_SS(spec, None)
    # save SS results
    utils.mkdirs(os.path.join(OUTPUT_DIR, "SS"))
    ss_dir = os.path.join(OUTPUT_DIR, "SS", "SS_vars.pkl")
    with open(ss_dir, "wb") as f:
        pickle.dump(ss_outputs, f)
    # Run TPI
    tpi_output = TPI.run_TPI(spec, None)
    assert(np.allclose(tpi_output['bmat_splus1'][:spec.T, :, :],
                       ss_outputs['bssmat_splus1']))


@pytest.mark.full_run
def test_constant_demographics_TPI_small_open():
    '''
    This tests solves the model under the assumption of constant
    demographics, a balanced budget, and tax functions that do not vary
    over time, as well as with a small open economy assumption.
    '''
    # Create output directory structure
    spec = Specifications(run_micro=False, output_base=OUTPUT_DIR,
                          baseline_dir=OUTPUT_DIR, test=False,
                          time_path=True, baseline=True, iit_reform={},
                          guid='')
    og_spec = {'constant_demographics': True, 'budget_balance': True,
               'zero_taxes': True, 'maxiter': 2,
               'r_gov_shift': 0.0, 'zeta_D': [0.0, 0.0],
               'zeta_K': [1.0], 'debt_ratio_ss': 1.0,
               'initial_foreign_debt_ratio': 0.0,
               'start_year': 2019, 'cit_rate': [0.0],
               'PIA_rate_bkt_1': 0.0, 'PIA_rate_bkt_2': 0.0,
               'PIA_rate_bkt_3': 0.0,
               'eta': (spec.omega_SS.reshape(spec.S, 1) *
                       spec.lambdas.reshape(1, spec.J))}
    spec.update_specifications(og_spec)
    spec.get_tax_function_parameters(None, False,
                                     tax_func_path=TAX_FUNC_PATH)
    # Run SS
    ss_outputs = SS.run_SS(spec, None)
    # save SS results
    utils.mkdirs(os.path.join(OUTPUT_DIR, "SS"))
    ss_dir = os.path.join(OUTPUT_DIR, "SS", "SS_vars.pkl")
    with open(ss_dir, "wb") as f:
        pickle.dump(ss_outputs, f)
    # Run TPI
    tpi_output = TPI.run_TPI(spec, None)
    assert(np.allclose(tpi_output['bmat_splus1'][:spec.T, :, :],
                       ss_outputs['bssmat_splus1']))
