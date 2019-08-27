'''
Example script for setting policy and running OG-USA.
'''

# import modules
import multiprocessing
from dask.distributed import Client
import time
import numpy as np

from taxcalc import Calculator
from ogusa import postprocess
from ogusa.execute import runner
from ogusa.utils import REFORM_DIR, BASELINE_DIR

# Define parameters to use for multiprocessing
client = Client(processes=False)
num_workers = min(multiprocessing.cpu_count(), 7)
print('Number of workers = ', num_workers)
run_start_time = time.time()

# Grab a reform JSON file already in Tax-Calculator
# In this example the 'reform' is a change to 2017 law (the
# baseline policy is tax law in 2018)
reform_url = ('https://raw.githubusercontent.com/'
              'PSLmodels/Tax-Calculator/master/taxcalc/'
              'reforms/2017_law.json')
ref = Calculator.read_json_param_objects(reform_url, None)
iit_reform = ref['policy']

# Set some OG model parameters
# See default_parameters.json for more description of these parameters
alpha_T = np.zeros(50)  # Adjusting the path of transfer spending
alpha_T[0:2] = 0.09
alpha_T[2:10] = 0.09 + 0.01
alpha_T[10:40] = 0.09 - 0.01
alpha_T[40:] = 0.09
alpha_G = np.zeros(7)  # Adjusting the path of non-transfer spending
alpha_G[0:3] = 0.05 - 0.01
alpha_G[3:6] = 0.05 - 0.005
alpha_G[6:] = 0.05
# Also adjust the Frisch elasticity, the start year, the
# effective corporate income tax rate, and the SS debt-to-GDP ratio
og_spec = {'frisch': 0.41, 'start_year': 2018, 'tau_b': [0.0357],
           'debt_ratio_ss': 1.0, 'alpha_T': alpha_T.tolist(),
           'alpha_G': alpha_G.tolist()}

'''
------------------------------------------------------------------------
Run baseline policy first
------------------------------------------------------------------------
'''
output_base = BASELINE_DIR
kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
          'test': False, 'time_path': True, 'baseline': True,
          'og_spec': og_spec, 'guid': '_example',
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
# update the effective corporate income tax rate
og_spec = {'frisch': 0.41, 'start_year': 2018,
           'tau_b': [0.0595], 'debt_ratio_ss': 1.0,
           'alpha_T': alpha_T.tolist(),
           'alpha_G': alpha_G.tolist()}
output_base = REFORM_DIR
kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
          'test': False, 'time_path': True, 'baseline': False,
          'og_spec': og_spec, 'guid': '_example',
          'iit_reform': iit_reform, 'run_micro': True, 'data': 'cps',
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
