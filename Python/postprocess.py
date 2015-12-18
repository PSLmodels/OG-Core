import ogusa
from ogusa import macro_output
import pickle

def create_diff(baseline, policy, dump_output=False):
    pct_changes = macro_output.dump_diff_output(baseline, policy)

    if dump_output:
        pickle.dump(output, open("ogusa_output.pkl", "wb"))

    return pct_changes

if __name__ == "__main__":
    create_diff(baseline="./OUTPUT_BASELINE", policy="./OUTPUT")
