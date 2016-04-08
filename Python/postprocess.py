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

import ogusa
from ogusa import macro_output
import pickle

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
                    from baseline to reform for each year 

    RETURNS:
    pct_changes

    OUTPUT: 
    ./ogusa_output{}.pkl

    --------------------------------------------------------------------
    '''
    pct_changes = macro_output.dump_diff_output(baseline_dir, policy_dir)

    if dump_output:
        pickle.dump(pct_changes, open("ogusa_output.pkl", "wb"))

    return pct_changes

if __name__ == "__main__":
    create_diff(baseline_dir="./OUTPUT_BASELINE", policy_dir="./OUTPUT_REFORM")
