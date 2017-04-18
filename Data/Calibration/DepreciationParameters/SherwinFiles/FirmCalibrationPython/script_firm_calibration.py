"""
Script (script_firm_calibration.py):
-------------------------------------------------------------------------------
Last updated: 6/24/2015.

This script calibrates parameters for firms on all NAICS levels.
The script primarily calls helper functions in the *parameter_calibrations*
module. This module splits up the calibration tasks into
various functions, specifically, there is a function for each set of
parameters that need to be calibrated. The script uses these functions to
generate :term:`NAICS trees<NAICS tree>` with all firm parameters calibrated for each NAICS
code. The script outputs these parameter calibrations and processed data to
csv files.
"""
# Packages:
import os.path
import sys
from pprint import pprint
import numpy as np
import pandas as pd
import ipdb
# Directories:
_CUR_DIR = os.path.dirname(__file__)
_PROC_DIR = os.path.join(_CUR_DIR, "processing")
_OUT_DIR = os.path.join(_CUR_DIR, "output")
_DATA_DIR = os.path.join(_CUR_DIR, "data")
_SOI_DIR = os.path.join(_DATA_DIR, "soi")
_DATA_STRUCT_DIR = os.path.join(_CUR_DIR, "data_structures")
_CST_DIR = os.path.join(_CUR_DIR, "constants")
# Appending directories of custom modules to list of system paths (sys.path):
sys.path.append(_PROC_DIR)
sys.path.append(_DATA_STRUCT_DIR)
sys.path.append(_CST_DIR)
sys.path.append(_SOI_DIR)

# Importing custom modules:
import naics_processing as naics
import soi_processing as soi
import constants as cst
import parameter_calibrations as clbr
import file_processing as fp
import cPickle as pickle

"""
Creating NAICS trees with all the relevant firm parameters calibrated using
helper functions from the parameter_calibrations module.
"""
#Breakpoint used for debugging
ipdb.set_trace()
#calculates the depreciation rates
depr_rates = clbr.calibrate_depr_rates(get_all = True)
#soi_tree = clbr.pull_soi_data(get_all=True, output_data=True)
#debt_tree = clbr.calibrate_debt(soi_tree=soi_tree)
#inc_tree = clbr.calibrate_incomes(output_data=True)

# NAICS codes of the relevant production industries
relevant_industries = [11, 211, 212, 213, 22, 23, 31, 32411, 336, 3391, 42, 44, 48, 51, 52, 531, 532, 533, 54, 55, 56, 61, 62, 71, 72, 81, 92]
# put econ depreciation rates in a dataframe for printing to csv
N = len(depr_rates['Econ'].enum_inds)
econ_rates_df = pd.DataFrame(index=np.arange(0,N),columns=('NAICS','All','Corp','Non-Corp'))
tax_rates_df = pd.DataFrame(index=np.arange(0,N),columns=('Type', 'All', 'Corp', 'Non-Corp'))
types = ['GDS 150%', 'ADS SL', 'GDS 200%', 'Recommended', 'GDS SL']
j = 0
for i in xrange(0,N):
    # storing the NAICS code in another variable that we can index
    code = depr_rates['Econ'].enum_inds[i].data.dfs['Codes:'].values
    # comparing the NAICS code of the relevant industries list and the NAICS code of the dataframe, it's indexed to pull out the integer value of the code
    if(relevant_industries[j] == code[0][0]): 
        # stores all the pertinent information about the industry in a dataframe
        econ_rates_df['NAICS'].iloc[j]= depr_rates['Econ'].enum_inds[i].data.dfs['Codes:'].values
        econ_rates_df['All'].iloc[j]= depr_rates['Econ'].enum_inds[i].data.dfs['Economic']['All'].iloc[0]
        econ_rates_df['Corp'].iloc[j]= depr_rates['Econ'].enum_inds[i].data.dfs['Economic']['Corp'].iloc[0]
        econ_rates_df['Non-Corp'].iloc[j]= depr_rates['Econ'].enum_inds[i].data.dfs['Economic']['Non-Corp'].iloc[0]
        
        tax_rates_df['Type'].iloc[j] = types[0]
        tax_rates_df['All'].iloc[j]= depr_rates['Tax'].enum_inds[i].data.dfs['GDS 150%']['All'].iloc[0]
        tax_rates_df['Corp'].iloc[j]= depr_rates['Tax'].enum_inds[i].data.dfs['GDS 150%']['Corp'].iloc[0]
        tax_rates_df['Non-Corp'].iloc[j]= depr_rates['Tax'].enum_inds[i].data.dfs['GDS 150%']['Non-Corp'].iloc[0]
        tax_rates_df['Corp'].iloc[j]= depr_rates['Tax'].enum_inds[i].data.dfs['ADS SL']['Corp'].iloc[0]
        tax_rates_df['Non-Corp'].iloc[j]= depr_rates['Tax'].enum_inds[i].data.dfs['GDS 200%']['Non-Corp'].iloc[0]
        
        # prevents the relevant industry list from passing its bounds
        if(j < len(relevant_industries)-1): 
            j += 1 
#writes the dataframe to a file
ipdb.set_trace()
econ_rates_df.to_csv(_CUR_DIR+'relevant_industries_output2.csv', index=False)


'''
for i in xrange(0,N):
    econ_rates_df['NAICS'].iloc[i]= depr_rates['Econ'].enum_inds[i].data.dfs['Codes:'].values
    econ_rates_df['All'].iloc[i]= depr_rates['Econ'].enum_inds[i].data.dfs['Economic']['All'].iloc[0]
    econ_rates_df['Corp'].iloc[i]= depr_rates['Econ'].enum_inds[i].data.dfs['Economic']['Corp'].iloc[0]
    econ_rates_df['Non-Corp'].iloc[i]= depr_rates['Econ'].enum_inds[i].data.dfs['Economic']['Non-Corp'].iloc[0]
    

econ_rates_df.to_csv(_CUR_DIR+'depr_rates_output.csv', index=False)

'''


