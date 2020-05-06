'''
This module defines the runner() function, which is used to run OG-USA
'''

import pickle
import cloudpickle
import os
import time
from ogusa import SS, TPI, utils
from ogusa.parameters import Specifications


def runner(output_base, baseline_dir, test=False, time_path=True,
           baseline=True, iit_reform={}, og_spec={}, guid='',
           run_micro=True, tax_func_path=None, data=None, client=None,
           num_workers=1):
    '''
    This function runs the OG-USA model, solving for the steady-state
    and (optionally) the time path equilibrium.

    Args:
        output_base (str): path to save output to
        baseline_dir (str): path where baseline model results are saved
        test (bool): whether to run model in test mode (which has
            a smaller state space and higher tolerances for solution)
        time_path (bool): whether to solve for the time path equlibrium
        baseline (bool): whether the model run is the baseline run
        iit_reform (dict): Tax-Calculator policy dictionary
        og_spec (dict): dictionary with updates to default
            parameters in OG-USA
        guid (str): id for OG-USA run
        run_micro (bool): whether to estimate tax functions from micro
            data or load saved parameters from pickle file
        data (str or Pandas DataFrame): path to or data to use in
            Tax-Calculator
        client (Dask client object): client
        num_workers (int): number of workers to use for parallelization
            with Dask

    Returns:
        None

    '''

    tick = time.time()
    # Create output directory structure
    ss_dir = os.path.join(output_base, "SS")
    tpi_dir = os.path.join(output_base, "TPI")
    dirs = [ss_dir, tpi_dir]
    for _dir in dirs:
        try:
            print("making dir: ", _dir)
            os.makedirs(_dir)
        except OSError:
            pass

    print('In runner, baseline is ', baseline)

    # Get parameter class
    # Note - set run_micro false when initially load class
    # Update later with call to spec.get_tax_function_parameters()
    spec = Specifications(run_micro=False, tax_func_path=tax_func_path,
                          output_base=output_base,
                          baseline_dir=baseline_dir, test=test,
                          time_path=time_path, baseline=baseline,
                          iit_reform=iit_reform, guid=guid, data=data,
                          client=client, num_workers=num_workers)

    spec.update_specifications(og_spec)
    print('path for tax functions: ', tax_func_path)
    spec.get_tax_function_parameters(client, run_micro, tax_func_path)

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
        ss_dir = os.path.join(baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)
        print('JUST SAVED SS output to ', ss_dir)
        # Save pickle with parameter values for the run
        param_dir = os.path.join(baseline_dir, "model_params.pkl")
        with open(param_dir, "wb") as f:
            cloudpickle.dump((spec), f)
    else:
        utils.mkdirs(os.path.join(output_base, "SS"))
        ss_dir = os.path.join(output_base, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)
        # Save pickle with parameter values for the run
        param_dir = os.path.join(output_base, "model_params.pkl")
        with open(param_dir, "wb") as f:
            cloudpickle.dump((spec), f)

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
        with open(tpi_vars, "wb") as f:
            pickle.dump(tpi_output, f)

        print("Time path iteration complete.")
    print("It took {0} seconds to get that part done.".format(
        time.time() - tick))
