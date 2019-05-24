from __future__ import print_function
# import modules
import multiprocessing
from dask.distributed import Client
import time
import numpy as np

from taxcalc import Calculator
from ogusa import postprocess
from ogusa.execute import runner
from ogusa.utils import REFORM_DIR, BASELINE_DIR


def run_micro_macro(user_params):
    # Grab a reform JSON file already in Tax-Calculator
    # In this example the 'reform' is a change to 2017 law (the
    # baseline policy is tax law in 2018)
    reform_url = ('https://raw.githubusercontent.com/'
                  'PSLmodels/Tax-Calculator/master/taxcalc/'
                  'reforms/2017_law.json')
    ref = Calculator.read_json_param_objects(reform_url, None)
    reform = ref['policy']

    # Define parameters to use for multiprocessing
    client = Client(processes=False)
    num_workers = 1  # multiprocessing.cpu_count()
    print('Number of workers = ', num_workers)
    run_start_time = time.time()

    # Set some model parameters
    # See parameters.py for description of these parameters
    alpha_T = np.zeros(50)
    alpha_T[0:2] = 0.09
    alpha_T[2:10] = 0.09 + 0.01
    alpha_T[10:40] = 0.09 - 0.01
    alpha_T[40:] = 0.09
    alpha_G = np.zeros(7)
    alpha_G[0:3] = 0.05 - 0.01
    alpha_G[3:6] = 0.05 - 0.005
    alpha_G[6:] = 0.05
    small_open = False
    user_params = {'frisch': 0.41, 'start_year': 2018,
                   'tau_b': [(0.21 * 0.55) * (0.017 / 0.055), (0.21 * 0.55) * (0.017 / 0.055)],
                   'debt_ratio_ss': 1.0, 'alpha_T': alpha_T.tolist(),
                   'alpha_G': alpha_G.tolist(), 'small_open': small_open}

    '''
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': True, 'baseline': True,
              'user_params': user_params, 'guid': '_example',
              'run_micro': True, 'data': 'cps', 'client': client,
              'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    '''
    ------------------------------------------------------------------------
    Run reform policy
    ------------------------------------------------------------------------
    '''
    user_params = {'frisch': 0.41, 'start_year': 2018,
                   'tau_b': [(0.35 * 0.55) * (0.017 / 0.055)],
                   'debt_ratio_ss': 1.0, 'alpha_T': alpha_T.tolist(),
                   'alpha_G': alpha_G.tolist(), 'small_open': small_open}
    output_base = REFORM_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': True, 'baseline': False,
              'user_params': user_params, 'guid': '_example',
              'reform': reform, 'run_micro': True, 'data': 'cps',
              'client': client, 'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    # return ans - the percentage changes in macro aggregates and prices
    # due to policy changes from the baseline to the reform
    ans = postprocess.create_diff(
        baseline_dir=BASELINE_DIR, policy_dir=REFORM_DIR)

    print("total time was ", (time.time() - run_start_time))
    print('Percentage changes in aggregates:', ans)


if __name__ == "__main__":
    run_micro_macro(user_params={})
