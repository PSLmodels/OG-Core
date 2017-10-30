from ogusa.macro_output import dump_diff_output
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pytest

REG_BASELINE = '../regression/REG_OUTPUT_BASELINE'
REG_REFORM = '../regression/REG_OUTPUT_REFORM_{id}'
REF_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#SWAP TO NEW DIRECTORY -- for test development only!!!!
BASELINE = REG_BASELINE
REFORM = REG_REFORM

@pytest.fixture(scope="module", params=REF_IDXS)
def macro_outputs(request):
    (pct_changes,
        baseline_macros,
        policy_macros) = dump_diff_output(BASELINE,
                                          REFORM.format(id=request.param))
    (reg_pct_changes,
        reg_baseline_macros,
        reg_policy_macros) = dump_diff_output(REG_BASELINE,
                                              REG_REFORM.format(id=request.param))

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

    # output_vars = ["Y", "C", "I", "L", "w", "r", "Revenue"]

    # for i in range(len(output_vars)):
    assert np.allclose(
        macro_outputs["new"]["pct_changes"][output_var_idx, :],
        macro_outputs["reg"]["pct_changes"][output_var_idx, :],
        atol=0.0, rtol=0.001
    )

# def get_tpi_vars(path):
#     with open(path, 'rb') as f:
#         return pickle.loads(path)
#
# def comp_TPI_vars(baseline_dir, output_dir, ref_idx=0):
#     """
#     Compare TPI_vars
#     """
#     tpi = get_tpi_vars(path=REG )



def plot_reforms(r1_ix, r2_ix=None, r3_ix=None):
    plot_vars = ["Y", "C", "I", "L", "w", "r", "Revenue"]
    plt.figure(figsize=(3, 3))
    for i in range(0, len(plot_vars)):
        t = list(range(12))
        t = [n + 2016 for n in t]
        plt.subplot(331 + i)
        plt.plot(t, pct_changes[r1_ix][i,:],'-b', label='reform {}'.format(r1_ix))
        if r2_ix is not None:
            plt.plot(t, pct_changes[r2_ix][i,:], '-r', label='reform {}'.format(r2_ix))
        if r3_ix is not None:
            plt.plot(t, pct_changes[r3_ix][i,:], '-g', label='reform {}'.format(r3_ix))
        plt.title("PCT Changes {0} from 2016 to steady state".format(plot_vars[i]))
        plt.legend()
    plt.show()
    plt.figure()


# if __name__ == '__main__':
#     pct_changes, baseline_macros, policy_macros = get_macros()
#
#     plot_reforms(9)
