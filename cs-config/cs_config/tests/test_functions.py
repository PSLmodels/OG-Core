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


@pytest.mark.full_run
def test_convert_adj():
    adj = {
        "STD": [
            {"MARS": "single", "year": "2019", "value": 0},
            {"MARS": "mjoint", "year": 2019, "value": 1}
        ],
        "EITC_c": [{"EIC": "0kids", "year": "2019", "value": 1000.0}],
        "BEN_ssi_repeal": [
            {"year": 2019, "value": True}
        ]
    }
    res = helpers.convert_adj(adj, 2019)
    assert res == {
        "STD": {
            2019: [0, 1, 12200.0, 18350.0, 24400.0]
        },
        "EITC_c": {
            2019: [1000.0, 3526.0, 5828.0, 6557.0]
        },
        "BEN_ssi_repeal": {
            2019: True
        }
    }
