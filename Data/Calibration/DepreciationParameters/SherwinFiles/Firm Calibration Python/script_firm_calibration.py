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
#import numpy as np
#import pandas as pd
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

"""
Creating NAICS trees with all the relevant firm parameters calibrated using
helper functions from the parameter_calibrations module.
"""
depr_rates = clbr.calibrate_depr_rates(get_all=True)
#soi_tree = clbr.pull_soi_data(get_all=True, output_data=True)
#debt_tree = clbr.calibrate_debt(soi_tree=soi_tree)
#inc_tree = clbr.calibrate_incomes(output_data=True)

