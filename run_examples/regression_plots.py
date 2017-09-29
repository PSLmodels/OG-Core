from ogusa.macro_output import dump_diff_output
import matplotlib.pyplot as plt

def get_macros():
    baseline_dir = '../run_examples/OUTPUT_BASELINE'
    output_dir = '../run_examples/OUTPUT_REFORM_{id}'

    pct_changes = []
    baseline_macros = []
    policy_macros = []

    for i in range(0, 10):
        res = dump_diff_output(
            baseline_dir,
            output_dir.format(id=i)
        )
        pct_changes.append(res[0])
        baseline_macros.append(res[1])
        policy_macros.append(res[2])

    return pct_changes, baseline_macros, policy_macros

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
