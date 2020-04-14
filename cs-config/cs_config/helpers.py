"""
Functions used to help OG-USA configure to COMP
"""
try:
    import boto3
except ImportError:
    boto3 = None
import gzip
import pandas as pd
from taxcalc import Policy
from collections import defaultdict

TC_LAST_YEAR = Policy.LAST_BUDGET_YEAR

POLICY_SCHEMA = {
    "labels": {
        "year": {
            "type": "int",
            "validators": {
                "choice": {
                    "choices": [
                        yr for yr in range(2013, TC_LAST_YEAR + 1)
                    ]
                }
            }
        },
        "MARS": {
            "type": "str",
            "validators": {"choice": {"choices": ["single", "mjoint",
                                                  "mseparate", "headhh",
                                                  "widow"]}}
        },
        "idedtype": {
            "type": "str",
            "validators": {"choice": {"choices": ["med", "sltx", "retx", "cas",
                                                  "misc", "int", "char"]}}
        },
        "EIC": {
            "type": "str",
            "validators": {"choice": {"choices": ["0kids", "1kid",
                                                  "2kids", "3+kids"]}}
        },
        "data_source": {
            "type": "str",
            "validators": {"choice": {"choices": ["PUF", "CPS", "other"]}}
        }
    },
    "additional_members": {
        "section_1": {"type": "str"},
        "section_2": {"type": "str"},
        "start_year": {"type": "int"},
        "checkbox": {"type": "bool"}
    }
}


def retrieve_puf(aws_access_key_id, aws_secret_access_key):
    """
    Function for retrieving the PUF from the OSPC S3 bucket
    """
    has_credentials = aws_access_key_id and aws_secret_access_key
    if has_credentials and boto3 is not None:
        client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        obj = client.get_object(Bucket="ospc-data-files", Key="puf.csv.gz")
        gz = gzip.GzipFile(fileobj=obj["Body"])
        puf_df = pd.read_csv(gz)
        return puf_df
    else:
        return None


def convert_defaults(pcl):
    '''
    Function to convert default Tax-Calculator policy parameters to
    ParamTools compliant parameters.
    '''
    type_map = {
        "real": "float",
        "boolean": "bool",
        "integer": "int",
        "string": "str",
    }

    new_pcl = defaultdict(dict)
    new_pcl["schema"] = POLICY_SCHEMA
    LAST_YEAR = TC_LAST_YEAR
    pol = Policy()
    pol.set_year(TC_LAST_YEAR)
    for param, item in pcl.items():
        values = []
        pol_val = getattr(pol, f"_{param}").tolist()
        min_year = min(item["value_yrs"])
        if isinstance(pol_val[0], list):
            for year in range(len(pol_val)):
                if min_year + year > LAST_YEAR:
                    break
                for dim1 in range(len(pol_val[0])):
                    values.append({
                        "year": min_year + year,
                        item["vi_name"]: item["vi_vals"][dim1],
                        "value": pol_val[year][dim1]
                    })
        else:
            for year in range(len(pol_val)):
                if min_year + year > LAST_YEAR:
                    break
                values.append({
                    "year": min_year + year,
                    "value": pol_val[year]
                })

        new_pcl[param]['value'] = values
        new_pcl[param]['title'] = pcl[param]["long_name"]
        new_pcl[param]['type'] = type_map[pcl[param]["value_type"]]

        new_pcl[param]["validators"] = {"range": pcl[param]["valid_values"]}

        # checkbox if indexable
        if item["indexable"]:
            if item["indexed"]:
                new_pcl[param]["checkbox"] = True
            else:
                new_pcl[param]["checkbox"] = False

        to_keep = list(POLICY_SCHEMA["additional_members"].keys()) + [
            "description", "notes",
        ]
        for k in to_keep:
            if k in pcl[param]:
                new_pcl[param][k] = pcl[param][k]

    return new_pcl


def convert_adj(adj, start_year):
    '''
    Function to convert Tax-Calculator parameters into ParmTools
    compliant parameters.
    '''
    type_map = {
        "real": float,
        "boolean": bool,
        "integer": int,
        "string": str,
    }
    pol = Policy()
    new_adj = defaultdict(dict)
    for param, valobjs in adj.items():
        if param.endswith("checkbox"):
            param_name = param.split("_checkbox")[0]
            new_adj[f"{param_name}-indexed"][start_year] =\
                valobjs[0]["value"]
            pol.implement_reform({f"{param_name}-indexed":
                                 {start_year: valobjs[0]["value"]}},
                                 raise_errors=False)
            continue
    for param, valobjs in adj.items():
        if param.endswith("checkbox"):
            continue
        for valobj in valobjs:
            valobj["year"] = int(valobj["year"])
            if not (set(valobj.keys()) - set(["value", "year"])):
                new_adj[param][valobj["year"]] = valobj["value"]
            else:
                try:
                    other_label = next(k for k in valobj.keys()
                                       if k not in ("year", "value"))
                except StopIteration:
                    print(valobj)
                    raise StopIteration
                param_meta = pol._vals[f"_{param}"]
                if other_label != param_meta["vi_name"]:
                    msg = (f"Label {other_label} does not match expected"
                           f"label {param_meta['vi_name']}")
                    raise ValueError(msg)
                ol_ix = param_meta["vi_vals"].index(valobj[other_label])
                other_label_ix = ol_ix

                if valobj["year"] in new_adj[param]:
                    defaultlist = new_adj[param][valobj["year"]]
                else:
                    year_ix = valobj["year"] - min(param_meta["value_yrs"])
                    # shallow copy the list
                    type_func = type_map[param_meta["value_type"]]
                    # convert from numpy type to basic python type.
                    defaultlist = list(
                        map(type_func, getattr(pol, f"_{param}")[year_ix]))

                defaultlist[other_label_ix] = valobj["value"]

                new_adj[param][valobj["year"]] = defaultlist
            pol.implement_reform(new_adj, raise_errors=False)
    return new_adj
