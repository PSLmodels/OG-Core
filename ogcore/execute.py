"""
This module defines the runner() function, which is used to run OG-Core
"""

import pickle
import cloudpickle
import os
import time
from ogcore import SS, TPI, utils


def runner(p, time_path=True, client=None):
    """
    This function runs the OG-Core model, solving for the steady-state
    and (optionally) the time path equilibrium.

    Args:
        p (Specifications object): model parameters
        time_path (bool): whether to solve for the time path equilibrium
        client (Dask client object): client

    Returns:
        None

    """

    tick = time.time()
    # Create output directory structure
    ss_dir = os.path.join(p.output_base, "SS")
    tpi_dir = os.path.join(p.output_base, "TPI")
    dirs = [ss_dir, tpi_dir]
    for _dir in dirs:
        try:
            print("making dir: ", _dir)
            os.makedirs(_dir)
        except OSError:
            pass

    print("In runner, baseline is ", p.baseline)

    """
    ------------------------------------------------------------------------
        Run SS
    ------------------------------------------------------------------------
    """
    ss_outputs = SS.run_SS(p, client=client)

    """
    ------------------------------------------------------------------------
        Pickle SS results
    ------------------------------------------------------------------------
    """
    utils.mkdirs(os.path.join(p.output_base, "SS"))
    ss_dir = os.path.join(p.output_base, "SS", "SS_vars.pkl")
    with open(ss_dir, "wb") as f:
        pickle.dump(ss_outputs, f)
    print("JUST SAVED SS output to ", ss_dir)
    # Save pickle with parameter values for the run
    param_dir = os.path.join(p.output_base, "model_params.pkl")
    with open(param_dir, "wb") as f:
        cloudpickle.dump((p), f)

    if time_path:
        """
        ------------------------------------------------------------------------
            Run the TPI simulation
        ------------------------------------------------------------------------
        """
        tpi_output = TPI.run_TPI(p, client=client)

        """
        ------------------------------------------------------------------------
            Pickle TPI results
        ------------------------------------------------------------------------
        """
        tpi_dir = os.path.join(p.output_base, "TPI")
        utils.mkdirs(tpi_dir)
        tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
        with open(tpi_vars, "wb") as f:
            pickle.dump(tpi_output, f)

        print("Time path iteration complete.")
    print(
        "It took {0} seconds to get that part done.".format(time.time() - tick)
    )
