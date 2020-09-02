from cs_kit import CoreTestFunctions
from cs_config import functions, helpers
import pytest


@pytest.mark.full_run
class TestFunctions1(CoreTestFunctions):
    get_version = functions.get_version
    get_inputs = functions.get_inputs
    validate_inputs = functions.validate_inputs
    run_model = functions.run_model
    ok_adjustment = {"OG-USA Parameters": {"frisch": 0.41},
                     "Tax-Calculator Parameters": {}}
    bad_adjustment = {"OG-USA Parameters": {"frisch": 1.5},
                      "Tax-Calculator Parameters": {"STD": -1}}


# Comment out until do tabular output
# def test_param_effect():
#     adjustment = {"ogusa": {"frisch": 0.35}}
#     comp_dict = functions.run_model({}, adjustment)
#     df = pd.read_csv(io.StringIO(comp_dict['downloadable'][0]['data']))
#     assert df.loc[0, 'Change from Baseline (pp)'] != 0


def test_convert_defaults():
    # TODO
    return
