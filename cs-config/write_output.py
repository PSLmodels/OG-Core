from cs_config import functions
import cs_storage
from cs_storage.screenshot import write_template
from ogusa import utils
from ogusa.parameters import Specifications

'''
This script is useful to testing the outputs from the cs_config/functions.py
script.  Those outputs will be saved in the 
/cs_config/OUTPUT_BASELINE/ and /cs_config/OUTPUT_REFORM/ folders
that are read in below.
'''

base_ss = utils.safe_read_pickle(
    './cs_config/OUTPUT_BASELINE/SS/SS_vars.pkl')
base_tpi = utils.safe_read_pickle(
    './cs_config/OUTPUT_BASELINE/TPI/TPI_vars.pkl')
reform_ss = utils.safe_read_pickle(
    './cs_config/OUTPUT_REFORM/SS/SS_vars.pkl')
reform_tpi = utils.safe_read_pickle(
    './cs_config/OUTPUT_REFORM/TPI/TPI_vars.pkl')
time_path = True
base_params = Specifications()
reform_params = Specifications()

meta_param_dict = {'year': [{'value': 2020}],
                   'data_source': [{'value': 'CPS'}],
                   'time_path': [{'value': True}]}
adjustment_dict = {'OG-USA Parameters': {'frisch': 0.41},
                   'Tax-Calculator Parameters': {}}
# outputs = functions.run_model(meta_param_dict, adjustment_dict)
outputs = functions.comp_output(
    base_params, base_ss, reform_params, reform_ss,
    time_path, base_tpi=base_tpi, reform_tpi=reform_tpi,
    var='cssmat')

for output in outputs["renderable"]:
    serializer = cs_storage.get_serializer(output["media_type"])
    ser = serializer.serialize(output["data"])
    deserialized = dict(
        output, data=serializer.deserialize(ser, json_serializable=True)
    )
    res = write_template(deserialized)
    with open(f"{output['title']}.html", "w") as f:
        f.write(res)
