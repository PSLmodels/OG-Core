from __future__ import print_function
'''
This module defines the runner() function, whic is used to run OG-USA
'''

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import numpy as np
import time


import ogusa
from ogusa import SS, TPI
# ogusa.parameters.DATASET = 'REAL'
from ogusa.pb_api import Specifications
from ogusa.utils import DEFAULT_START_YEAR, TC_LAST_YEAR


def runner(output_base, baseline_dir, test=False, time_path=True,
           baseline=True, constant_rates=True, tax_func_type='DEP',
           analytical_mtrs=False, age_specific=False, reform={},
           user_params={}, guid='', run_micro=True, small_open=False,
           budget_balance=False, baseline_spending=False, data=None,
           client=None, num_workers=1):

    from ogusa import parameters, demographics, income, utils

    tick = time.time()

    # start_year = user_params.get('start_year', DEFAULT_START_YEAR)
    # if start_year > TC_LAST_YEAR:
    #     raise RuntimeError("Start year is beyond data extrapolation.")
    #
    # # Make sure options are internally consistent
    # if baseline and baseline_spending:
    #     print("Inconsistent options. Setting <baseline_spending> to False, "
    #           "leaving <baseline> True.'")
    #     baseline_spending = False
    # if budget_balance and baseline_spending:
    #     print("Inconsistent options. Setting <baseline_spending> to False, "
    #           "leaving <budget_balance> True.")
    #     baseline_spending = False

    # Create output directory structure
    ss_dir = os.path.join(output_base, "SS")
    tpi_dir = os.path.join(output_base, "TPI")
    dirs = [ss_dir, tpi_dir]
    for _dir in dirs:
        try:
            print("making dir: ", _dir)
            os.makedirs(_dir)
        except OSError as oe:
            pass

    print('In runner, baseline is ', baseline)

    # Get parameter class
    spec = Specifications(output_base=output_base,
                          baseline_dir=baseline_dir, baseline=baseline,
                          client=client, num_workers=num_workers)
    spec.update_specifications({'age_specific': False})
    print('path for tax functions: ', spec.output_base)
    spec.get_tax_function_parameters(run_micro=False)

    # ogusa_05242018 = ogusaclass(spec1, spec2)
    # ogusa_05242018.runner()
    # pickle.dump(ogusa_05242018, open())
    #
    # ssresults = ogusa_05242018.ss_output

    '''
    ------------------------------------------------------------------------
        Run SS
    ------------------------------------------------------------------------
    '''

    ss_outputs = SS.run_SS(spec, client=client)

    '''
    ------------------------------------------------------------------------
        Pickle SS results
    ------------------------------------------------------------------------
    '''
    if baseline:
        utils.mkdirs(os.path.join(baseline_dir, "SS"))
        ss_dir = os.path.join(baseline_dir, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
        # Save pickle with parameter values for the run
        param_dir = os.path.join(baseline_dir, "model_params.pkl")
        pickle.dump(spec, open(param_dir, "wb"))
    else:
        utils.mkdirs(os.path.join(output_base, "SS"))
        ss_dir = os.path.join(output_base, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
        # Save pickle with parameter values for the run
        param_dir = os.path.join(output_base, "model_params.pkl")
        pickle.dump(spec, open(param_dir, "wb"))

    if time_path:
        '''
        ------------------------------------------------------------------------
            Run the TPI simulation
        ------------------------------------------------------------------------
        '''
        tpi_output = TPI.run_TPI(spec, client=client)

        '''
        ------------------------------------------------------------------------
            Pickle TPI results
        ------------------------------------------------------------------------
        '''
        tpi_dir = os.path.join(output_base, "TPI")
        utils.mkdirs(tpi_dir)
        tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
        pickle.dump(tpi_output, open(tpi_vars, "wb"))

        print("Time path iteration complete.")
    print("It took {0} seconds to get that part done.".format(time.time() - tick))
