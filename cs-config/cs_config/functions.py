import ogusa
from ogusa.parameters import Specifications
from ogusa.utils import TC_LAST_YEAR, REFORM_DIR, BASELINE_DIR
from ogusa import output_plots as op
from ogusa import output_tables as ot
from ogusa import SS, utils
import os
import io
import pickle
import json
import inspect
import paramtools
from distributed import Client
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
            "value": 2020,
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
        'analytical_mtrs', 'age_specific', 'gamma_s', 'epsilon_s']
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
        # set name of cached baseline file in case use below
        cached_pickle = 'TxFuncEst_baseline_PUF.pkl'
    else:
        data = "cps"
        # set name of cached baseline file in case use below
        cached_pickle = 'TxFuncEst_baseline_CPS.pkl'
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
    # Limit to one worker and one thread to satisfy celery
    # constraints on multiprocessing.
    client = Client(n_workers=1, threads_per_worker=1, processes=False)
    num_workers = 1

    # whether to estimate tax functions from microdata
    run_micro = True

    # filter out OG-USA params that will not change between baseline and
    # reform runs (these are the non-policy parameters)
    filtered_ogusa_params = {}
    constant_param_set = {
        'frisch', 'beta_annual', 'sigma', 'g_y_annual', 'gamma',
        'epsilon', 'Z', 'delta_annual', 'small_open', 'world_int_rate',
        'initial_foreign_debt_ratio', 'zeta_D', 'zeta_K', 'tG1', 'tG2',
        'rho_G', 'debt_ratio_ss', 'start_year', 'budget_balance'}
    filtered_ogusa_params = OrderedDict()
    for k, v in adjustment['OG-USA Parameters'].items():
        if k in constant_param_set:
            filtered_ogusa_params[k] = v

    # Solve baseline model
    start_year = meta_param_dict['year'][0]['value']
    if start_year == 2020:
        print('In path where use cached file')
        OGPATH = inspect.getfile(SS)
        OGDIR = os.path.dirname(OGPATH)
        tax_func_path = os.path.join(OGDIR, 'data', 'tax_functions',
                                     cached_pickle)
        run_micro_baseline = False
    else:
        print('Not in path where use cached file')
        tax_func_path = None
        run_micro_baseline = True
    base_spec = {
        **{'start_year': start_year,
           'tax_func_type': 'linear',
           'age_specific': False}, **filtered_ogusa_params}
    base_params = Specifications(
        run_micro=False, output_base=base_dir, baseline_dir=base_dir,
        test=False, time_path=False, baseline=True, iit_reform={},
        guid='', data=data, client=client, num_workers=num_workers)
    base_params.update_specifications(base_spec)
    print('Args for getting basline tax functions are: ', run_micro_baseline, tax_func_path)
    base_params.get_tax_function_parameters(
        client, run_micro_baseline, tax_func_path=tax_func_path)
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
        data=data, client=client, num_workers=num_workers)
    reform_params.update_specifications(reform_spec)
    reform_params.get_tax_function_parameters(client, run_micro)
    reform_ss = SS.run_SS(reform_params, client=client)

    comp_dict = comp_output(base_ss, base_params, reform_ss,
                            reform_params)

    # Shut down client and make sure all of its references are
    # cleaned up.
    client.close()
    del client

    return comp_dict


def comp_output(base_ss, base_params, reform_ss, reform_params,
                var='cssmat'):
    '''
    Function to create output for the COMP platform
    '''
    table_title = 'Percentage Changes in Economic Aggregates Between'
    table_title += ' Baseline and Reform Policy'
    plot_title = 'Percentage Changes in Consumption by Lifetime Income'
    plot_title += ' Percentile Group'
    out_table = ot.macro_table_SS(
        base_ss, reform_ss,
        var_list=['Yss', 'Css', 'Iss_total', 'Gss', 'total_revenue_ss',
                  'Lss', 'rss', 'wss'], table_format='csv')
    html_table = ot.macro_table_SS(
        base_ss, reform_ss,
        var_list=['Yss', 'Css', 'Iss_total', 'Gss', 'total_revenue_ss',
                  'Lss', 'rss', 'wss'], table_format='html')
    fig = op.ability_bar_ss(
        base_ss, base_params, reform_ss, reform_params, var=var)
    in_memory_file = io.BytesIO()
    fig.savefig(in_memory_file, format="png")
    in_memory_file.seek(0)
    comp_dict = {
        "renderable": [
            {
              "media_type": "PNG",
              "title": plot_title,
              "data": in_memory_file.read()
              },
            {
              "media_type": "table",
              "title":  table_title,
              "data": html_table
            }
            ],
        "downloadable": [
            {
              "media_type": "CSV",
              "title": table_title,
              "data": out_table.to_csv()
            }
        ]
        }

    return comp_dict
