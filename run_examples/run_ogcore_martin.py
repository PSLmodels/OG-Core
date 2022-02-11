'''
Example script for setting policy and running OG-Core.
'''

# import modules
import multiprocessing
from distributed import Client
import time
import numpy as np
import os
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.constants import REFORM_DIR, BASELINE_DIR
from ogcore.utils import safe_read_pickle
import matplotlib.pyplot as plt
style_file_url = ('https://raw.githubusercontent.com/PSLmodels/OG-Core/' +
                  'master/ogcore/OGcorePlots.mplstyle')
plt.style.use(style_file_url)


def main():
    # Define parameters to use for multiprocessing
    client = Client()
    num_workers = min(multiprocessing.cpu_count(), 7)
    print('Number of workers = ', num_workers)
    run_start_time = time.time()

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, BASELINE_DIR)
    reform_dir = os.path.join(CUR_DIR, REFORM_DIR)

    # Set some OG model parameters
    # See default_parameters.json for more description of these parameters
    # alpha_T = np.zeros(50)  # Adjusting the path of transfer spending
    # alpha_T[0:2] = 0.09
    # alpha_T[2:10] = 0.09 + 0.01
    # alpha_T[10:40] = 0.09 - 0.01
    # alpha_T[40:] = 0.09
    # alpha_G = np.zeros(7)  # Adjusting the path of non-transfer spending
    # alpha_G[0:3] = 0.05 - 0.01
    # alpha_G[3:6] = 0.05 - 0.005
    # alpha_G[6:] = 0.05
    # Set start year for baseline and reform.
    START_YEAR = 2021
    # Also adjust the Frisch elasticity, the start year, the
    # effective corporate income tax rate, and the SS debt-to-GDP ratio
    # og_spec = {
    #     'frisch': 0.41, 'start_year': START_YEAR, 'cit_rate': [0.21],
    #     'debt_ratio_ss': 1.0, 'alpha_T': alpha_T.tolist(),
    #     'alpha_G': alpha_G.tolist()}

    og_spec = {
        'start_year': 2021,
        'frisch': 0.41,
        'cit_rate': [0.21],
        'alpha_T': [0.07],
        'alpha_G': [0.03],
        'r_gov_shift': 0.03,
        # 'PIA_maxpayment': 470,
        'baseline_spending': True,
        'debt_ratio_ss': 1.3,
        'tG1': 20, 'tG2': 256,
        'initial_guess_r_SS': 0.06,
    }

    '''
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    '''
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    # Update parameters for baseline from default json file
    p.update_specifications(og_spec)

    start_time = time.time()
    runner(p, time_path=False, client=client)
    print('run time = ', time.time()-start_time)

    client.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
