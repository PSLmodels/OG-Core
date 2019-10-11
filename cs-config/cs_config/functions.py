import ogusa
from ogusa.parameters import Specifications
from ogusa.utils import TC_LAST_YEAR, REFORM_DIR, BASELINE_DIR
from ogusa import output_plots as op
from ogusa import SS, utils
import os
import io
import pickle
import json
import inspect
import paramtools
from taxcalc import Policy
from collections import OrderedDict
from .helpers import retrieve_puf, convert_adj, convert_defaults

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# Get Tax-Calculator default parameters
TCPATH = inspect.getfile(Policy)
TCDIR = os.path.dirname(TCPATH)
with open(os.path.join(TCDIR, "policy_current_law.json"), "r") as f:
    pcl = json.loads(f.read())
RES = convert_defaults(pcl)


class TCParams(paramtools.Parameters):
    defaults = RES


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
    # Set default OG-USA parameters
    ogusa_params = Specifications()
    ogusa_params.start_year = meta_params.year
    filtered_ogusa_params = OrderedDict()
    filter_list = [
        'chi_n_80', 'chi_b', 'eta', 'zeta', 'constant_demographics',
        'ltilde', 'use_zeta', 'constant_rates', 'zero_taxes',
        'analytical_mtrs', 'age_specific']
    for k, v in ogusa_params.dump().items():
        if ((k not in filter_list) and
            (v.get("section_1", False) != "Model Solution Parameters")
            and (v.get("section_2", False) != "Model Dimensions")):
            filtered_ogusa_params[k] = v
            print('filtered ogusa = ', k)
    # Set default TC params
    iit_params = TCParams()
    iit_params.set_state(year=meta_params.year.tolist())
    filtered_iit_params = OrderedDict()
    for k, v in iit_params.dump().items():
        if k == "schema" or v.get("section_1", False):
            filtered_iit_params[k] = v

    default_params = {
        "OG-USA Parameters": filtered_ogusa_params,
        "Tax-Calculator Parameters": filtered_iit_params
    }

    return {
         "meta_parameters": meta_params.dump(),
         "model_parameters": default_params
         }


def validate_inputs(meta_param_dict, adjustment, errors_warnings):
    # ogusa doesn't look at meta_param_dict for validating inputs.
    params = Specifications()
    params.adjust(adjustment["OG-USA Parameters"], raise_errors=False)
    errors_warnings["OG-USA Parameters"]["errors"].update(
        params.errors)
    # Validate TC parameter inputs
    pol_params = {}
    # drop checkbox parameters.
    for param, data in list(adjustment[
        "Tax-Calculator Parameters"].items()):
        if not param.endswith("checkbox"):
            pol_params[param] = data
    iit_params = TCParams()
    iit_params.adjust(pol_params, raise_errors=False)
    errors_warnings["Tax-Calculator Parameters"][
        "errors"].update(iit_params.errors)

    return {"errors_warnings": errors_warnings}


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
    # Get TC params adjustments
    iit_mods = convert_adj(adjustment[
        "Tax-Calculator Parameters"],
                           meta_params.year.tolist())
    # Create output directory structure
    base_dir = os.path.join(CUR_DIR, BASELINE_DIR)
    reform_dir = os.path.join(CUR_DIR, REFORM_DIR)
    dirs = [base_dir, reform_dir]
    for _dir in dirs:
        utils.mkdirs(_dir)
    # Dask parmeters
    client = None
    num_workers = 1

    # whether to estimate tax functions from microdata
    run_micro = True

    # Solve baseline model
    base_spec = {'start_year': meta_param_dict['year'],
                 'debt_ratio_ss': 2.0,
                 'r_gov_scale': 1.0, 'r_gov_shift': 0.02,
                 'zeta_D': [0.4], 'zeta_K': [0.1],
                 'initial_debt_ratio': 0.78,
                 'initial_foreign_debt_ratio': 0.4,
                 'tax_func_type': 'linear',
                 'age_specific': False}
    base_params = Specifications(
        run_micro=False, output_base=base_dir,
        baseline_dir=base_dir, test=False, time_path=False,
        baseline=True, iit_reform={}, guid='',
        data=data,
        client=client, num_workers=num_workers)
    base_params.update_specifications(base_spec)
    base_params.get_tax_function_parameters(client, run_micro)
    base_ss = SS.run_SS(base_params, client=client)
    utils.mkdirs(os.path.join(base_dir, "SS"))
    ss_dir = os.path.join(base_dir, "SS", "SS_vars.pkl")
    with open(ss_dir, "wb") as f:
        pickle.dump(base_ss, f)

    # Solve reform model
    reform_spec = base_spec
    reform_spec.update(adjustment["OG-USA Parameters"])
    reform_params = Specifications(
        run_micro=False, output_base=reform_dir,
        baseline_dir=base_dir, test=False, time_path=False,
        baseline=False, iit_reform=iit_mods, guid='',
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
    fig = op.ss_profiles(
        base_ss, base_params, reform_ss=reform_ss,
        reform_params=reform_params, by_j=True, var=var,
        plot_title='Labor Supply in Baseline and Reform Policy')
    in_memory_file = io.BytesIO()
    fig.savefig(in_memory_file, format="png")
    in_memory_file.seek(0)
    comp_dict = {
        "renderable": [
            {
              "media_type": "PNG",
              "title": 'Labor Supply in Baseline and Reform Policy',
              "data": in_memory_file.read()
              }
            ],
        "downloadable": []
        }

    return comp_dict
