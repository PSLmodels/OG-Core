import os
import tempfile
import pytest
import numpy as np
from ogusa.parameters import Specifications, revision_warnings_errors
from ogusa import utils

# get path to puf if puf.csv in ogusa/ directory
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

JSON_REVISION_FILE = """{
    "revision": {
        "frisch": 0.3
    }
}"""


@pytest.fixture(scope='module')
def revision_file():
    f = tempfile.NamedTemporaryFile(mode="a", delete=False)
    f.write(JSON_REVISION_FILE)
    f.close()
    # Must close and then yield for Windows platform
    yield f
    os.remove(f.name)


def test_create_specs_object():
    specs = Specifications()
    assert specs


def test_create_specs_object_test():
    specs = Specifications(test=True)
    assert specs


def test_compute_default_params():
    specs = Specifications()
    specs.alpha_G = np.ones((10, 1))
    specs.compute_default_params()
    assert specs.alpha_G[10] == 1


def test_get_tax_function_parameters():
    specs = Specifications()
    specs.get_tax_function_parameters(None, run_micro=False)
    assert specs.etr_params.shape == (specs.T, specs.S, 12)
    assert specs.mtrx_params.shape == (specs.T, specs.S, 12)
    assert specs.mtry_params.shape == (specs.T, specs.S, 12)


def test_get_tax_function_parameters_constant_rates():
    specs = Specifications()
    specs.constant_rates = True
    specs.get_tax_function_parameters(None, run_micro=False)
    assert specs.etr_params.shape == (specs.T, specs.S, 12)
    assert specs.mtrx_params.shape == (specs.T, specs.S, 12)
    assert specs.mtry_params.shape == (specs.T, specs.S, 12)
    assert np.allclose(specs.etr_params[:, :, :10],
                       np.zeros((specs.T, specs.S, 10)))
    assert np.allclose(specs.mtrx_params[:, :, :10],
                       np.zeros((specs.T, specs.S, 10)))
    assert np.allclose(specs.mtry_params[:, :, :10],
                       np.zeros((specs.T, specs.S, 10)))


def test_get_tax_function_zero_taxes():
    specs = Specifications()
    specs.zero_taxes = True
    specs.get_tax_function_parameters(None, run_micro=False)
    assert np.allclose(specs.etr_params,
                       np.zeros((specs.T, specs.S, 12)))
    assert np.allclose(specs.mtrx_params,
                       np.zeros((specs.T, specs.S, 12)))
    assert np.allclose(specs.mtry_params,
                       np.zeros((specs.T, specs.S, 12)))


def test_read_tax_func_estimate():
    specs = Specifications()
    pickle_path = os.path.join(CUR_PATH, 'test_io_data')
    pickle_file = 'TxFuncEst_baseline.pkl'
    expected_dict = utils.safe_read_pickle(os.path.join(
        pickle_path, pickle_file))
    expected_dict2 = utils.safe_read_pickle(os.path.join(
        CUR_PATH, pickle_file))
    test_dict = specs.read_tax_func_estimate(os.path.join(
        pickle_path, pickle_file), pickle_file)
    assert np.allclose(expected_dict['tfunc_avg_etr'],
                       test_dict['tfunc_avg_etr'])
    test_dict2 = specs.read_tax_func_estimate('not_a_path', pickle_file)
    assert np.allclose(expected_dict2['tfunc_avg_etr'],
                       test_dict2['tfunc_avg_etr'])



def test_update_specifications_with_dict():
    spec = Specifications()
    new_spec_dict = {
        'frisch': 0.3,
    }
    spec.update_specifications(new_spec_dict)
    assert spec.frisch == 0.3
    assert len(spec.errors) == 0


def test_update_specification_with_json():
    spec = Specifications()
    new_spec_json = """
        {
            "frisch": 0.3
        }
    """
    spec.update_specifications(new_spec_json)
    assert spec.frisch == 0.3
    assert len(spec.errors) == 0


def test_implement_reform():
    specs = Specifications()
    new_specs = {
        'tG1': 30,
        'T': 80,
        'frisch': 0.3,
        'tax_func_type': 'DEP'
    }

    specs.update_specifications(new_specs)
    assert specs.frisch == 0.3
    assert specs.tG1 == 30
    assert specs.T == 80
    assert specs.tax_func_type == 'DEP'
    assert len(specs.errors) == 0
    # assert len(specs.warnings) == 0


def test_implement_bad_reform1():
    specs = Specifications()
    # tG1 has an upper bound at T / 2
    new_specs = {
        'tG1': 50,
        'T': 80,
    }

    specs.update_specifications(new_specs, raise_errors=False)

    assert len(specs.errors) == 0 # > 0
    # assert specs.errors['tG1'] == 'ERROR: tG1 value 50 > max value 40.0\n'  # to redo when can have param valid values depend on others'
    # assert len(specs.warnings) == 0


def test_implement_bad_reform2():
    specs = Specifications()
    # tG1 has an upper bound at T / 2
    new_specs = {
        'T': 80,
        'tax_func_type': 'not_a_functional_form'
    }

    specs.update_specifications(new_specs, raise_errors=False)

    assert len(specs.errors) > 0
    assert specs.errors['tax_func_type'][0] == 'tax_func_type "not_a_functional_form" must be in list of choices DEP, DEP_totalinc, GS, linear.'
    # assert len(specs.warnings) == 0



def test_revision_warnings_errors():
    user_mods = {'frisch': 0.41}

    ew = revision_warnings_errors(user_mods)
    assert len(ew['errors']) == 0
    assert len(ew['warnings']) == 0

    user_mods = {'frisch': 0.1}

    bad_ew = revision_warnings_errors(user_mods)
    assert len(bad_ew['errors']) > 0
    assert len(bad_ew['warnings']) == 0


## Commenting out because I don't think ParamTools allows this yet
# def test_simple_eval():
#     specs = Specifications()
#     specs.T = 100
#     assert specs.simple_eval('T / 2') == 50
#     assert specs.simple_eval('T * 2') == 200
#     assert specs.simple_eval('T - 2') == 98
#     assert specs.simple_eval('T + 2') == 102
