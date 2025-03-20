import os
import tempfile
import pytest
import numpy as np
from ogcore.parameters import Specifications, revision_warnings_errors
from ogcore import utils

# get path to puf if puf.csv in ogcore/ directory
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

JSON_REVISION_FILE = """{
    "revision": {
        "frisch": 0.3
    }
}"""


@pytest.fixture(scope="module")
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


def test_compute_default_params():
    specs = Specifications()
    specs.alpha_G = np.ones((10, 1))
    specs.compute_default_params()
    assert specs.alpha_G[10] == 1


param_updates1 = {
    "T": 4,
    "S": 3,
    "rho": [[0.0, 0.0, 1.0]],
    "e": np.ones((3, 7)),
    "ubi_nom_017": 1000,
    "eta": np.ones((4, 3, 1)) / 12,
    "ubi_nom_1864": 1200,
    "ubi_nom_65p": 400,
    "ubi_growthadj": True,
}
expected1 = np.ones((7, 3, 7)) * 2180
param_updates2 = {
    "T": 4,
    "S": 3,
    "rho": [[0.0, 0.0, 1.0]],
    "e": np.ones((3, 7)),
    "ubi_nom_017": 1000,
    "eta": np.ones((4, 3, 1)) / 12,
    "ubi_nom_1864": 1200,
    "ubi_nom_max": 2000,
    "ubi_nom_65p": 400,
    "ubi_growthadj": True,
}
expected2 = np.ones((7, 3, 7)) * 2000
param_updates3 = {
    "T": 4,
    "S": 3,
    "rho": [[0.0, 0.0, 1.0]],
    "e": np.ones((3, 7)),
    "ubi_nom_017": 1000,
    "eta": np.ones((4, 3, 1)) / 12,
    "ubi_nom_1864": 1200,
    "ubi_nom_65p": 400,
    "ubi_growthadj": False,
    "g_y_annual": 0.03,
}
expected3 = np.array(
    [
        [[2180], [2180.0], [2180.0]],
        [[656.9250257], [656.9250257], [656.9250257]],
        [[197.9589401], [197.9589401], [197.9589401]],
        [[59.65329442], [59.65329442], [59.65329442]],
        [[17.97602843], [17.97602843], [17.97602843]],
        [[17.97602843], [17.97602843], [17.97602843]],
        [[17.97602843], [17.97602843], [17.97602843]],
    ]
)


@pytest.mark.parametrize(
    "param_updates, expected",
    [
        (param_updates1, expected1),
        (param_updates2, expected2),
        (param_updates3, expected3),
    ],
    ids=["UBI growth adj", "UBI hit max", "UBI no growth adj"],
)
def test_get_ubi_nom_objs(param_updates, expected):
    spec = Specifications()
    spec.update_specifications(param_updates)
    assert np.allclose(spec.ubi_nom_array, expected)


def test_update_specifications_with_dict():
    spec = Specifications()
    new_spec_dict = {
        "frisch": 0.3,
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
    new_specs = {"tG1": 30, "T": 80, "frisch": 0.3, "tax_func_type": "DEP"}

    specs.update_specifications(new_specs)
    assert specs.frisch == 0.3
    assert specs.tG1 == 30
    assert specs.T == 80
    assert specs.tax_func_type == "DEP"
    assert len(specs.errors) == 0
    # assert len(specs.warnings) == 0


def test_implement_bad_reform1():
    specs = Specifications()
    # tG1 has an upper bound at T / 2
    new_specs = {
        "tG1": 50,
        "T": 80,
    }

    specs.update_specifications(new_specs, raise_errors=False)

    assert len(specs.errors) == 0


def test_implement_bad_reform2():
    specs = Specifications()
    # tG1 has an upper bound at T / 2
    new_specs = {"T": 80, "tax_func_type": "not_a_functional_form"}

    specs.update_specifications(new_specs, raise_errors=False)

    assert len(specs.errors) > 0
    assert specs.errors["tax_func_type"][0] == (
        'tax_func_type "not_a_functional_form" must be in list of '
        + "choices DEP, DEP_totalinc, GS, HSV, linear, mono, mono2D."
    )


def test_implement_bad_reform3():
    specs = Specifications()
    with pytest.raises(ValueError):
        specs.update_specifications(None, raise_errors=False)


def test_revision_warnings_errors():
    user_mods = {"frisch": 0.41}

    ew = revision_warnings_errors(user_mods)
    assert len(ew["errors"]) == 0
    assert len(ew["warnings"]) == 0

    user_mods = {"frisch": 0.1}

    bad_ew = revision_warnings_errors(user_mods)
    assert len(bad_ew["errors"]) > 0
    assert len(bad_ew["warnings"]) == 0


def test_conditional_validator():
    specs = Specifications()
    new_specs = {"budget_balance": True, "baseline_spending": True}
    specs.update_specifications(new_specs, raise_errors=False)
    assert len(specs.errors) > 0


def test_expand_taxfunc_params():
    specs = Specifications()
    new_specs = {"etr_params": [[[0.35]]]}
    specs.update_specifications(new_specs)
    assert len(specs.etr_params) == specs.T + specs.S
    assert len(specs.etr_params[0]) == specs.S
    assert specs.etr_params[0][0][0] == 0.35
