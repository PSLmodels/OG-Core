from cs_config import functions
import cs_storage
from cs_storage.screenshot import write_template
from ogusa import utils
from ogusa.parameters import Specifications
import pickle
import io

'''
This script is useful to testing the outputs from the cs_config/functions.py
script.  Those outputs will be saved in the 
/cs_config/OUTPUT_BASELINE/ and /cs_config/OUTPUT_REFORM/ folders
that are read in below.
'''
def generate_plots():
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


def run_model():
    meta_param_dict = {'year': [{'value': 2020}],
                       'data_source': [{'value': 'CPS'}],
                       'time_path': [{'value': True}]}
    adjustment_dict = {'OG-USA Parameters': {
                                             'frisch': 0.39,
                                             'initial_debt_ratio': 1.1,
                                             'g_y_annual': 0.029,
                                             'tG1': 22},
                       'Tax-Calculator Parameters': {}}
    comp_dict = functions.run_model(meta_param_dict, adjustment_dict)
    pickle.dump(comp_dict, open('ogusa_cs_test_dict.pkl', 'wb'))
    s = io.StringIO(comp_dict['downloadable'][0]['data'])
    with open('ogusa_test_output.csv', 'w') as f:
        for line in s:
            f.write(line)


if __name__ == "__main__":
    generate_plots()
    run_model()
