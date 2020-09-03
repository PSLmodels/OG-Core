from typing import Union

from paramtools import Parameters


def convert_policy_defaults(meta_params: Parameters, policy_params: Parameters):
    """
    Convert defaults for taxcalc's Policy class to work with C/S.
    """
    policy_params.array_first = False
    policy_params.uses_extend_func = False
    init = policy_params.dump()
    defaults = convert_sections(init)
    defaults = convert_data_source(defaults, meta_params.data_source)
    defaults = convert_indexed_to_checkbox(defaults)
    return defaults


def convert_policy_adjustment(policy_adjustment: dict):
    """
    Convert adjutments for taxcalc's Policy class to work with C/S.
    """
    return convert_checkbox(policy_adjustment)


def convert_behavior_adjustment(adj):
    """
    Convert a C/S behavioral adjustment to work with the Behavioral-Responses
    package
    """
    behavior = {}
    if adj:
        for param, value in adj.items():
            behavior[param] = value[0]["value"]
    return behavior


def convert_checkbox(policy_params):
    """
    Replace param_checkbox with param-indexed.
    """
    params = {}
    # drop checkbox parameters.
    for param, data in policy_params.items():
        if param.endswith("checkbox"):
            base_param = param.split("_checkbox")[0]
            params[f"{base_param}-indexed"] = data
        else:
            params[param] = data

    return params


def convert_data_source(defaults: dict, data_source: Union["PUF", "CPS"]):
    """
    Handle parameters that are incompatible with the selected dataset.
    """
    new_defaults = {}
    for param, data in defaults.items():
        if (
            defaults.get("compatible_data") is not None
            and not defaults["compatible_data"][data_source.lower()]
        ):
            new_defaults[param] = dict(data, value=[])
        else:
            new_defaults[param] = data
    return new_defaults


def convert_indexed_to_checkbox(defaults: dict):
    """
    C/S expects there to be a checkbox attribute instead of
    indexed in the defaults.
    """
    new_defaults = {}
    for param, data in defaults.items():

        if param == "schema":
            data["additional_members"]["checkbox"] = dict(
                data["additional_members"]["indexed"]
            )
            new_defaults["schema"] = data

        elif data["indexable"] and data.get("indexed", None) is True:
            new_defaults[param] = dict(data, checkbox= True)

        elif data["indexable"] and not data.get("indexed", None) is False:
            new_defaults[param] = dict(data, checkbox=False)

        else:
            new_defaults[param] = data

    return new_defaults


def convert_sections(defaults):
    """
    Drop parameters that are missing section_1.
    """
    filtered_pol_params = {}
    for k, v in defaults.items():
        if k == "schema" or v.get("section_1", False):
            filtered_pol_params[k] = v
    return filtered_pol_params