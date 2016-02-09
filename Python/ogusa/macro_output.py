'''
------------------------------------------------------------------------
Last updated 12/02/2015

This program read in the output from TPI.py and creates table of
percentage changes between the baseline and policy results.

This py-file calls the following other file(s):
            OUTPUT/TPIinit/TPIinit_vars.pkl
            OUTPUT/TPI/TPI_vars.pkl


This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/TPI/ogusa_output.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import cPickle as pickle
import os

def dump_diff_output(baseline_dir, policy_dir):
    # read macro output
    tpi_macro_vars_policy_path = os.path.join(policy_dir, "TPI", "TPI_macro_vars.pkl")
    TPI_macro_vars_policy = pickle.load(open( tpi_macro_vars_policy_path, "rb" ))
    tpi_macro_vars_baseline_path = os.path.join(baseline_dir, "TPI", "TPI_macro_vars.pkl")
    TPI_macro_vars_baseline = pickle.load(open( tpi_macro_vars_baseline_path, "rb" ) )

    T = len(TPI_macro_vars_baseline['C_path'])
    baseline_macros = np.zeros((7,T))
    baseline_macros[0,:] = TPI_macro_vars_baseline['Y_path'][:T]
    baseline_macros[1,:] = TPI_macro_vars_baseline['C_path'][:T]
    baseline_macros[2,:] = TPI_macro_vars_baseline['I_path'][:T]
    baseline_macros[3,:] = TPI_macro_vars_baseline['Lpath_TPI'][:T]
    baseline_macros[4,:] = TPI_macro_vars_baseline['winit'][:T]
    baseline_macros[5,:] = TPI_macro_vars_baseline['rinit'][:T]
    baseline_macros[6,:] = TPI_macro_vars_baseline['T_H_init'][:T]

    policy_macros = np.zeros((7,T))
    policy_macros[0,:] = TPI_macro_vars_policy['Y_path'][:T]
    policy_macros[1,:] = TPI_macro_vars_policy['C_path'][:T]
    policy_macros[2,:] = TPI_macro_vars_policy['I_path'][:T]
    policy_macros[3,:] = TPI_macro_vars_policy['Lpath_TPI'][:T]
    policy_macros[4,:] = TPI_macro_vars_policy['winit'][:T]
    policy_macros[5,:] = TPI_macro_vars_policy['rinit'][:T]
    policy_macros[6,:] = TPI_macro_vars_policy['T_H_init'][:T]

    pct_changes = np.zeros((7,12))
    # pct changes for each year in budget window
    pct_changes[:,:10] = ((policy_macros-baseline_macros)/policy_macros)[:,:10]
    # pct changes over entire budget window
    pct_changes[:,10] = ((policy_macros[:,:10].sum(axis=1)-baseline_macros[:,:10].sum(axis=1))/policy_macros[:,:10].sum(axis=1))
    
    ## Load SS results
    ss_policy_path = os.path.join(policy_dir, "SSinit", "ss_init_vars.pkl")
    ss_policy = pickle.load(open( ss_policy_path, "rb" ))
    ss_baseline_path = os.path.join(baseline_dir, "SSinit", "ss_init_vars.pkl")
    ss_baseline = pickle.load(open( ss_baseline_path, "rb" ) )
    # pct changes in macro aggregates in SS
    pct_changes[0,11] = (ss_policy['Yss']-ss_baseline['Yss'])/ss_baseline['Yss']
    pct_changes[1,11] = (ss_policy['Css']-ss_baseline['Css'])/ss_baseline['Css']
    pct_changes[2,11] = (ss_policy['Kss']-ss_baseline['Kss'])/ss_baseline['Kss']
    pct_changes[3,11] = (ss_policy['Lss']-ss_baseline['Lss'])/ss_baseline['Lss']
    pct_changes[4,11] = (ss_policy['wss']-ss_baseline['wss'])/ss_baseline['wss']
    pct_changes[5,11] = (ss_policy['rss']-ss_baseline['rss'])/ss_baseline['rss']
    pct_changes[6,11] = (ss_policy['T_Hss']-ss_baseline['T_Hss'])/ss_baseline['T_Hss']

    #print 'pct changes: ', pct_changes


    return pct_changes


#pct_changes = dump_diff_output('/Users/jasondebacker/repos/dynamic/Python/OUTPUT_BASELINE','/Users/jasondebacker/repos/dynamic/Python/OUTPUT_REFORM')
#np.savetxt("pct_changes_02082016.csv", pct_changes, delimiter=",")

