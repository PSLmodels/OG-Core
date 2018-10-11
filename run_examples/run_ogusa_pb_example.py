from __future__ import print_function
# import modules
import multiprocessing
from multiprocessing import Process
from dask.distributed import Client
import time
import numpy as np

from taxcalc import *
import ogusa
from ogusa.scripts import postprocess
from ogusa.scripts.execute import runner
from ogusa.utils import REFORM_DIR, BASELINE_DIR


def run_micro_macro(user_params):
    # Grab a reform JSON file already in Tax-Calculator
    # In this example the 'reform' is a change to 2017 law (the
    # baseline policy is tax law in 2018)
    ref = Calculator.read_json_param_objects('2017_law.json', None)
    reform = ref['policy']

    # Define parameters to use for multiprocessing
    client = Client(processes=False)
    num_workers = multiprocessing.cpu_count()
    print('Number of workers = ', num_workers)
    start_time = time.time()

    # Set some model parameters
    # See parameters.py for description of these parameters
    T_shifts = np.zeros(50)
    T_shifts[2:10] = 0.01
    T_shifts[10:40] = -0.01
    G_shifts = np.zeros(6)
    G_shifts[0:3] = -0.01
    G_shifts[3:6] = -0.005
    small_open = False
    # small_open = dict(world_int_rate=0.04)
    # Alternatively small_open can be False/None
    # if False/None then 0.04 is used
    user_params = {'frisch': 0.41, 'start_year': 2018,
                   'tau_b': (0.21 * 0.55) * (0.017 / 0.055),
                   'debt_ratio_ss': 1.0, 'T_shifts': T_shifts,
                   'G_shifts': G_shifts, 'small_open': small_open}

    '''
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': True, 'baseline': True,
              'constant_rates': False,
              'analytical_mtrs': False, 'age_specific': True,
              'user_params': user_params, 'guid': '_example',
              'run_micro': True, 'small_open': small_open,
              'budget_balance': False, 'baseline_spending': False,
              'data': 'cps', 'client': client,
              'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)
    quit()

    '''
    ------------------------------------------------------------------------
    Run reform policy
    ------------------------------------------------------------------------
    '''
    user_params = {'frisch': 0.41, 'start_year': 2018,
                   'tau_b': (0.35 * 0.55) * (0.017 / 0.055),
                   'debt_ratio_ss': 1.0, 'T_shifts': T_shifts,
                   'G_shifts': G_shifts, 'small_open': small_open}
    output_base = REFORM_DIR
    input_dir = REFORM_DIR
    guid_iter = 'reform_' + str(0)
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': True, 'baseline': False,
              'constant_rates': False, 'analytical_mtrs': False,
              'age_specific': True, 'user_params': user_params,
              'guid': '_example', 'reform': reform, 'run_micro': True,
              'small_open': small_open, 'budget_balance': False,
              'baseline_spending': False, 'data': 'cps',
              'client': client, 'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    # return ans - the percentage changes in macro aggregates and prices
    # due to policy changes from the baseline to the reform
    ans = postprocess.create_diff(baseline_dir=BASELINE_DIR,
                                  policy_dir=REFORM_DIR)

    print("total time was ", (time.time() - start_time))
    print('Percentage changes in aggregates:', ans)



if __name__ == "__main__":
    run_micro_macro(user_params={})
