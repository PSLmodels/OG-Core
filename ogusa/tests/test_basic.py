import os
import pytest
import pickle
import numpy as np
from ogusa import SS, TPI, utils
from ogusa.parameters import Specifications

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.full_run
@pytest.mark.parametrize('time_path', [False, True], ids=['SS', 'TPI'])
def test_run_small(time_path):
    from ogusa.execute import runner
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    SS.MINIMIZER_TOL = 1e-6
    TPI.MINIMIZER_TOL = 1e-6
    output_base = os.path.join(CUR_PATH, 'OUTPUT')
    input_dir = output_base
    og_spec = {'frisch': 0.41, 'debt_ratio_ss': 0.4}
    runner(output_base=output_base, baseline_dir=input_dir, test=True,
           time_path=time_path, baseline=True, og_spec=og_spec,
           run_micro=False)


@pytest.mark.full_run
def test_constant_demographics_TPI():
    '''
    This tests solves the model under the assumption of constant
    demographics, a balanced budget, and tax functions that do not vary
    over time.
    In this case, given how initial guesss for the time
    path are made, the time path should be solved for on the first
    iteration and the values all along the time path should equal their
    steady-state values.
    '''
    output_base = os.path.join(CUR_PATH, 'OUTPUT')
    baseline_dir = output_base
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
    spec = Specifications(run_micro=False, output_base=output_base,
                          baseline_dir=baseline_dir, test=False,
                          time_path=True, baseline=True, iit_reform={},
                          guid='')
    og_spec = {'constant_demographics': True,
               'budget_balance': True,
               'zero_taxes': True,
               'maxiter': 2,
               'eta': (spec.omega_SS.reshape(spec.S, 1) *
                       spec.lambdas.reshape(1, spec.J))}
    spec.update_specifications(og_spec)
    spec.get_tax_function_parameters(None, False)
    # Run SS
    ss_outputs = SS.run_SS(spec, None)
    # save SS results
    utils.mkdirs(os.path.join(baseline_dir, "SS"))
    ss_dir = os.path.join(baseline_dir, "SS/SS_vars.pkl")
    with open(ss_dir, "wb") as f:
        pickle.dump(ss_outputs, f)
    # Run TPI
    tpi_output = TPI.run_TPI(spec, None)
    assert(np.allclose(tpi_output['bmat_splus1'][:spec.T, :, :],
                       ss_outputs['bssmat_splus1']))
