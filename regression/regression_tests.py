from ogusa.macro_output import dump_diff_output
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pytest

REG_BASELINE = '../regression/REG_OUTPUT_BASELINE'
REG_REFORM = '../regression/REG_OUTPUT_REFORM_{ref_idx}'
REF_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#SWAP TO NEW DIRECTORY -- for test development only!!!!
BASELINE = REG_BASELINE
REFORM = REG_REFORM

@pytest.fixture(scope="module", params=REF_IDXS)
def macro_outputs(request):
    (pct_changes,
        baseline_macros,
        policy_macros) = dump_diff_output(BASELINE,
                                          REFORM.format(ref_idx=request.param))
    (reg_pct_changes,
        reg_baseline_macros,
        reg_policy_macros) = dump_diff_output(REG_BASELINE,
                                              REG_REFORM.format(ref_idx=request.param))

    return {"new":{
                   "pct_changes": pct_changes,
                   "baseline_macros": baseline_macros,
                   "policy_macros": policy_macros
                  },
            "reg": {
                    "pct_changes": reg_pct_changes,
                    "baseline_macros": reg_baseline_macros,
                    "policy_macros": reg_policy_macros
                    },
            }

OUTPUT_VARS = ["Y", "C", "I", "L", "w", "r", "Revenue"]
@pytest.mark.parametrize("output_var_idx", np.arange(len(OUTPUT_VARS)))
def test_macro_output(macro_outputs, output_var_idx):
    """
    Compare macro output

    baseline_dir: directory for baseline input to compare to regression results
    reform_dir: directory for reform input to compare to regression results
    """
    assert np.allclose(
        macro_outputs["new"]["pct_changes"][output_var_idx, :],
        macro_outputs["reg"]["pct_changes"][output_var_idx, :],
        atol=0.0, rtol=0.001
    )


@pytest.fixture(scope="module", params=REF_IDXS + ["baseline"])
def tpi_output(request):
    def get_tpi_output(path):
        with open(path + "/TPI/TPI_vars.pkl", 'rb') as f:
            return pickle.load(f)

    ref_idx = request.param
    if ref_idx == "baseline":
        return (get_tpi_output(REG_BASELINE), get_tpi_output(BASELINE))
    else:
        return (get_tpi_output(path=REG_REFORM.format(ref_idx=request.param)),
                get_tpi_output(path=REFORM.format(ref_idx=request.param)))


TPI_VARS = ['C', 'D', 'G', 'REVENUE', 'I', 'K', 'tax_path', 'L',
            'eul_laborleisure', 'T_H', 'r', 'n_mat', 'BQ', 'w', 'Y',
            'eul_savings', 'c_path', 'b_mat']


@pytest.mark.parametrize("tpi_var", TPI_VARS)
def test_tpi_vars(tpi_output, tpi_var):
    """
    Compare TPI_vars
    """
    reg = tpi_output[0][tpi_var]
    new = tpi_output[1][tpi_var]
    assert np.allclose(reg, new, atol=0.0, rtol=0.001)
