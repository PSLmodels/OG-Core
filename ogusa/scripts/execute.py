from __future__ import print_function
'''
This module defines the runner() function, whic is used to run OG-USA
'''

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import time
from ogusa import SS, TPI, utils
from ogusa.pb_api import Specifications


def runner(output_base, baseline_dir, test=False, time_path=True,
           baseline=True, constant_rates=True, tax_func_type='DEP',
           analytical_mtrs=False, age_specific=False, reform={},
           user_params={}, guid='', run_micro=True, small_open=False,
           budget_balance=False, baseline_spending=False, data=None,
           client=None, num_workers=1):

    tick = time.time()
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
    spec.get_tax_function_parameters(client, run_micro)

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
    print("It took {0} seconds to get that part done.".format(
        time.time() - tick))
