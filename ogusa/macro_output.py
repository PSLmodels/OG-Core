'''
------------------------------------------------------------------------
This program reads in the output from TPI.py and creates table of
percentage changes between the baseline and policy results.

This py-file calls the following other file(s):
            /baseline_dir/TPI/TPI_macro_vars.pkl
            /policy_dir/TPI/TPI_macro_vars.pkl
            /baseline_dir/SSinit/ss_init_vars.pkl
            /policy_dir/SSinit/ss_init_vars.pkl

This py-file creates the following other file(s): None
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import os
from ogusa.utils import safe_read_pickle


def dump_diff_output(baseline_dir, policy_dir):
    '''
    --------------------------------------------------------------------
    This function reads the pickles with the SS and time path results
    from the baseline and reform and then calculates the percentage
    differences between the two for each year in the 10-year budget
    window, over the entire budget window, and in the SS.
    --------------------------------------------------------------------

    INPUTS:
    baseline_dir = string, path for directory with baseline policy results
    policy_dir   = string, path for directory with reform policy results

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    tpi_macro_vars_policy_path   = string, path to pickle with time path
                                    results for reform
    tpi_macro_vars_policy        = dictionary, dictionary with numpy arrays of
                                    results from transition path equilibrium for reform
    tpi_macro_vars_baseline_path = string, path to pickle with time path
                                    results for baseline policy
    tpi_macro_vars_baseline      = dictionary, dictionary with numpy arrays of
                                    results from transition path equilibrium for baseline policy
    baseline_macros              = [7,T] array, numpy array with time path for relevant macro
                                    variables from baseline equilibrium
    policy_macros                = [7,T] array, numpy array with time path for relevant macro
                                    variables from reform equilibrium
    pct_changes                  = [7,12] array, numpy array with pct changes in macro variables
                                    from baseline to reform for each year
                                    in the budget window (10 years), over all 10 years, and in the SS
    ss_policy_path               = string, path to pickle of SS results for reform
    ss_policy                    = dictionary, dictionary with numpy arrays of results from
                                    SS equilibrium for reform
    ss_baseline_path             = string, path to pickle of SS results for baseline
    ss_baseline                  = dictionary, dictionary with numpy arrays of results from
                                    SS equilibrium for baseline

    RETURNS: pct_changes
    --------------------------------------------------------------------
    '''

    # read macro output
    tpi_baseline_dir = os.path.join(baseline_dir, "TPI")
    tpi_policy_dir = os.path.join(policy_dir, "TPI")
    if not os.path.exists(tpi_policy_dir):
        os.mkdir(tpi_policy_dir)
    tpi_macro_vars_policy_path = os.path.join(tpi_policy_dir,
                                              "TPI_vars.pkl")
    tpi_macro_vars_policy = safe_read_pickle(tpi_macro_vars_policy_path)
    tpi_macro_vars_baseline_path = os.path.join(tpi_baseline_dir,
                                                "TPI_vars.pkl")
    tpi_macro_vars_baseline = safe_read_pickle(tpi_macro_vars_baseline_path)

    T = len(tpi_macro_vars_baseline['C'])
    baseline_macros = np.zeros((7, T))
    baseline_macros[0, :] = tpi_macro_vars_baseline['Y'][:T]
    baseline_macros[1, :] = tpi_macro_vars_baseline['C'][:T]
    baseline_macros[2, :] = tpi_macro_vars_baseline['I'][:T]
    baseline_macros[3, :] = tpi_macro_vars_baseline['L'][:T]
    baseline_macros[4, :] = tpi_macro_vars_baseline['w'][:T]
    baseline_macros[5, :] = tpi_macro_vars_baseline['r'][:T]
    baseline_macros[6, :] = tpi_macro_vars_baseline['total_revenue'][:T]

    policy_macros = np.zeros((7, T))
    policy_macros[0, :] = tpi_macro_vars_policy['Y'][:T]
    policy_macros[1, :] = tpi_macro_vars_policy['C'][:T]
    policy_macros[2, :] = tpi_macro_vars_policy['I'][:T]
    policy_macros[3, :] = tpi_macro_vars_policy['L'][:T]
    policy_macros[4, :] = tpi_macro_vars_policy['w'][:T]
    policy_macros[5, :] = tpi_macro_vars_policy['r'][:T]
    policy_macros[6, :] = tpi_macro_vars_policy['total_revenue'][:T]

    pct_changes = np.zeros((7, 12))
    # pct changes for each year in budget window
    pct_changes[:, :10] = ((policy_macros-baseline_macros) /
                           policy_macros)[:, :10]
    # pct changes over entire budget window
    pct_changes[:, 10] = ((policy_macros[:, :10].sum(axis=1) -
                           baseline_macros[:, :10].sum(axis=1)) /
                          policy_macros[:, :10].sum(axis=1))

    # Load SS results
    ss_policy_path = os.path.join(policy_dir, "SS", "SS_vars.pkl")
    ss_policy = safe_read_pickle(ss_policy_path)
    ss_baseline_path = os.path.join(baseline_dir, "SS", "SS_vars.pkl")
    ss_baseline = safe_read_pickle(ss_baseline_path)
    # pct changes in macro aggregates in SS
    pct_changes[0, 11] = ((ss_policy['Yss'] - ss_baseline['Yss']) /
                          ss_baseline['Yss'])
    pct_changes[1, 11] = ((ss_policy['Css'] - ss_baseline['Css']) /
                          ss_baseline['Css'])
    pct_changes[2, 11] = ((ss_policy['Iss'] - ss_baseline['Iss']) /
                          ss_baseline['Iss'])
    pct_changes[3, 11] = ((ss_policy['Lss'] - ss_baseline['Lss']) /
                          ss_baseline['Lss'])
    pct_changes[4, 11] = ((ss_policy['wss'] - ss_baseline['wss']) /
                          ss_baseline['wss'])
    pct_changes[5, 11] = ((ss_policy['rss'] - ss_baseline['rss']) /
                          ss_baseline['rss'])
    pct_changes[6, 11] = ((ss_policy['total_revenue_ss'] -
                           ss_baseline['total_revenue_ss']) /
                          ss_baseline['total_revenue_ss'])

    return pct_changes, baseline_macros, policy_macros
