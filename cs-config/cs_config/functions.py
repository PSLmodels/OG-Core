import ogusa
from ogusa.parameters import Specifications, revision_warnings_errors
from ogusa.utils import TC_LAST_YEAR, REFORM_DIR, BASELINE_DIR
from ogusa import output_plots as op
from ogusa import SS
import os
import paramtools
from .helpers import retrieve_puf

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")


class MetaParams(paramtools.Parameters):
    '''
    Meta parameters class for COMP.  These parameters will be in a drop
    down menu on COMP.
    '''
    array_first = True
    defaults = {
        "year": {
            "title": "Start year",
            "description": "Year for parameters.",
            "type": "int",
            "value": 2019,
            "validators": {"range": {"min": 2015, "max": TC_LAST_YEAR}}
        },
        "data_source": {
            "title": "Data source",
            "description": "Data source for Tax-Calculator to use",
            "type": "str",
            "value": "CPS",
            "validators": {"choice": {"choices": ["PUF", "CPS"]}}
        }
    }


def get_version():
    return ogusa.__version__    


def get_inputs(meta_param_dict):
    meta_params = MetaParams()
    meta_params.adjust(meta_param_dict)
    params = Specifications()
    # TODO: does OG-USA use year?
    # params.set_state(year=meta_params.year)
    return {
        "meta_parameters": meta_params.dump(),
        "model_parameters": {
            "ogusa": params.dump()
        }
    }


def validate_inputs(meta_param_dict, adjustment, errors_warnings):
    # ogusa doesn't look at meta_param_dict for validating inputs.
    params = Specifications()
    params.adjust(adjustment["ogusa"], raise_errors=False)
    # errors_warnings = revision_warnings_errors(adjustment["ogusa"])
    # return {"errors_warnings": errors_warnings}

    return {"errors_warnings": {"ogusa": {"errors": params.errors}}}

def run_model(meta_param_dict, adjustment):
    '''
    Initializes classes from OG-USA that compute the model under
    different policies.  Then calls function get output objects.
    '''
    meta_params = MetaParams()
    meta_params.adjust(meta_param_dict)
    if meta_params.data_source == "PUF":
        data = retrieve_puf(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    else:
        data = "cps"
    # Create output directory structure
    output_base = BASELINE_DIR
    base_dir = os.path.join(output_base, "SS")
    reform_dir = os.path.join(REFORM_DIR, "SS")
    dirs = [base_dir, reform_dir]
    for _dir in dirs:
        try:
            print("making dir: ", _dir)
            os.makedirs(_dir)
        except OSError:
            pass
    # Dask parmeters
    client = None
    num_workers = 1

    # whether to estimate tax functions from microdata
    run_micro = False

    # Solve baseline model
    base_spec = {'start_year': meta_param_dict['year'],
                 'debt_ratio_ss': 2.0,
                 'r_gov_scale': 1.0, 'r_gov_shift': 0.02,
                 'zeta_D': [0.4], 'zeta_K': [0.1],
                 'initial_debt_ratio': 0.78,
                 'initial_foreign_debt_ratio': 0.4}
    base_params = Specifications(
        run_micro=False, output_base=output_base,
        baseline_dir=BASELINE_DIR, test=False, time_path=False,
        baseline=True, iit_reform={}, guid='',
        data=data,
        client=client, num_workers=num_workers)
    base_params.update_specifications(base_spec)
    base_params.get_tax_function_parameters(client, run_micro)
    base_ss = SS.run_SS(base_params, client=client)

    # Solve reform model
    reform_spec = {'start_year': meta_param_dict['year'],
                   'debt_ratio_ss': 2.0,
                   'r_gov_scale': 1.0, 'r_gov_shift': 0.02,
                   'zeta_D': [0.4], 'zeta_K': [0.1],
                   'initial_debt_ratio': 0.78,
                   'initial_foreign_debt_ratio': 0.4}.update(adjustment)
    reform_params = Specifications(
        run_micro=False, output_base=REFORM_DIR,
        baseline_dir=BASELINE_DIR, test=False, time_path=False,
        baseline=False, iit_reform={}, guid='',
        data=data,
        client=client, num_workers=num_workers)
    reform_params.update_specifications(reform_spec)
    reform_params.get_tax_function_parameters(client, run_micro)
    reform_ss = SS.run_SS(reform_params, client=client)

    comp_dict = comp_output(base_ss, base_params, reform_ss,
                            reform_params)

    return comp_dict


def comp_output(base_ss, base_params, reform_ss, reform_params,
                var='nssmat'):
    '''
    Function to create output for the COMP platform
    '''
    plt = op.ss_profiles(
        base_ss, base_params, reform_ss=reform_ss,
        reform_params=reform_params, by_j=True, var=var,
        plot_title='Labor Supply in Baseline and Reform Policy')
    comp_dict = {
        "renderable": [
            {
              "media_type": "matplotlib",
              "title": plt.title,
              "data": {
                        "png": plt.savefig()
                    }
              }
            ]
        }

    return comp_dict
