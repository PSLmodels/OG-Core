"""
Depreciation Rate Calibration (depreciation_calibration.py):
-------------------------------------------------------------------------------
Last updated: 6/26/2015.

This module calibrates the firm economic and tax depreciation parameters.
"""
# Packages:
import os.path
import sys
import numpy as np
import pandas as pd
# Directories:
_CUR_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_CUR_DIR, "data")
_PROC_DIR = os.path.join(_CUR_DIR, "processing")
_OUT_DIR = os.path.join(_CUR_DIR, "output")
# Importing custom modules:
import naics_processing as naics
import constants as cst
# Importing depreciation helper custom modules:
sys.path.append(_PROC_DIR)
import calc_rates as calc_rates
import read_bea as read_bea
import read_inventories as read_inv
import read_land as read_land
# Dataframe names:
_CODE_DF_NM = cst.CODE_DF_NM
# Dataframe column names:
_CORP_TAX_SECTORS_NMS_DICT = cst.CORP_TAX_SECTORS_NMS_DICT
_CORP_NMS = _CORP_TAX_SECTORS_NMS_DICT.values()
_NON_CORP_TAX_SECTORS_NMS_DICT = cst.NON_CORP_TAX_SECTORS_NMS_DICT
_NCORP_NMS = _NON_CORP_TAX_SECTORS_NMS_DICT.values()


def init_depr_rates(asset_tree=naics.generate_tree(), get_econ=False, 
                    get_tax_est=False, get_tax_150=False,
                    get_tax_200=False, get_tax_sl=False,
                    get_tax_ads=False, soi_from_out=False,
                    output_data=False):
    """ This fun
    
    
    """
    # Calculating the fixed asset data:
    fixed_asset_tree = read_bea.read_bea(asset_tree)
    # Calculating the inventory data:
    inv_tree = read_inv.read_inventories(asset_tree)
    # Calculating the land data:
    land_tree = read_land.read_land(asset_tree)
    # Calculating the depreciation rates:
    econ_depr_tree = calc_rates.calc_depr_rates(fixed_asset_tree, inv_tree, land_tree)
    tax_depr_tree = calc_rates.calc_tax_depr_rates(fixed_asset_tree, inv_tree, land_tree)
    #naics.pop_rates(tax_depr_tree)
    
    
    return {"Econ": econ_depr_tree, "Tax": tax_depr_tree}





