'''
------------------------------------------------------------------------
Last updated 4/8/2016

This function finds the percentage changes in macro variables that
result from the tax reform.

This py-file calls the following other files:
            macro_output.py

This py-file creates the following other file(s):
            ./ogusa_output{}.pkl
------------------------------------------------------------------------
'''

import pickle
import numpy as np

import ogusa
from ogusa import macro_output
from ogusa.utils import REFORM_DIR, BASELINE_DIR

DEFAULTS = dict(baseline_dir=BASELINE_DIR,
                policy_dir=REFORM_DIR)

def create_diff(baseline_dir, policy_dir, dump_output=False):
    '''
    --------------------------------------------------------------------
    This function finds the percentage changes in macro variables that
    result from the tax reform.
    --------------------------------------------------------------------

    INPUTS:
    baseline_dir = string, path for directory with baseline policy results
    policy_dir   = string, path for directory with reform policy results
    dump_output  = boolean, =True if want results saved to pickle

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    macro_output.dump_diff_output()

    OBJECTS CREATED WITHIN FUNCTION:
    pct_changes  = [7,12] array, numpy array with pct changes in macro variables
                    from baseline to reform for each year. Final column = steady state.
                    Macro vars: Y, C, K, L, w, r, T_H

    RETURNS:
    pct_changes

    OUTPUT:
    ./ogusa_output{}.pkl

    --------------------------------------------------------------------
    '''
    out = macro_output.dump_diff_output(baseline_dir, policy_dir)
    pct_changes, baseline_macros, policy_macros = out

    np.savetxt('ClosedEconPctChanges.csv',pct_changes,delimiter=",")
    if dump_output:
        pickle.dump(pct_changes, open("ogusa_output.pkl", "wb"))

    np.savetxt('ClosedEconBaseline.csv',baseline_macros,delimiter=",")
    np.savetxt('ClosedEconPolicy.csv',policy_macros,delimiter=",")

    return pct_changes

if __name__ == "__main__":
    create_diff(**DEFAULTS)
