from ogusa.macro_output import dump_diff_output
import matplotlib.pyplot as plt
import numpy as np

REG_BASELINE = '../regression/REG_OUTPUT_BASELINE'
REG_OUTPUT = '../regression/REG_OUTPUT_REFORM_{id}'
REF_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def get_macros(baseline_dir=REG_BASELINE, output_dir=REG_OUTPUT,
               ref_idxs=REF_IDXS):
    pct_changes = []
    baseline_macros = []
    policy_macros = []

    for i in ref_idxs:
        res = dump_diff_output(
            baseline_dir,
            output_dir.format(id=i)
        )
        pct_changes.append(res[0])
        baseline_macros.append(res[1])
        policy_macros.append(res[2])

    return pct_changes, baseline_macros, policy_macros


def comp(baseline_dir, output_dir, ref_idxs=REF_IDXS):
    """
    Currently only compares macro output but in the future will compare
    all ouput
    """
    pct_changes, baseline_macros, policy_macros = get_macros(ref_idxs=ref_idxs)
    (reg_pct_changes,
        reg_baseline_macros,
        reg_policy_macros) = get_macros(baseline_dir, output_dir,
                                        ref_idxs=ref_idxs)

    output_vars = ["Y", "C", "I", "L", "w", "r", "Revenue"]

    for ref_idx in ref_idxs:
        for i in range(len(output_vars)):
            assert np.allclose(
                pct_changes[ref_idx][i,:], reg_pct_changes[ref_idx][i,:],
                atol=0.0, rtol=0.001
            )


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


if __name__ == '__main__':
    pct_changes, baseline_macros, policy_macros = get_macros()

    plot_reforms(9)
