"""
Depreciation Rate Calibration (depreciation_calibration.py):
-------------------------------------------------------------------------------
Last updated: 6/26/2015.

This module calibrates the firm economic and tax depreciation parameters.
"""
'''
This py-file calls the following file(s):
        Raw input files:
            data\**sbfltfile\****sb1.csv
            data\**sbfltfile\****sb3.csv
            data\SOI_Partner\pa01.csv
            data\SOI_Partner\pa03.csv
            data\SOI_Partner\pa05.csv
            
        Formatted input files:
            data\****_NAICS_Codes.csv
            data\SOI_Partner\pa01_Crosswalk.csv
            data\SOI_Partner\pa03_Crosswalk.csv
            data\SOI_Partner\pa05_Crosswalk.csv
'''
# Packages:
import os.path
import sys
import numpy as np
import pandas as pd
# Relevant directories:
_CUR_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.abspath(_CUR_DIR + "\\data")
_PROC_DIR = os.path.abspath(_CUR_DIR + "\\processing")
_OUT_DIR = os.path.abspath(_CUR_DIR + "\\output")
# Importing standard custom modules:
import naics_processing as naics
import constants as cst
import soi_processing as read_soi
# Importing depreciation helper custom modules:
sys.path.append(_PROC_DIR)
import calc_asset_types as calc_assets
import calc_rates as calc_rates
import read_bea as read_bea
import read_inventories as read_inv
import read_land as read_land
# Dataframe names:
_CODE_DF_NM = cst.CODE_DF_NM


def init_depr_rates(data_tree=naics.generate_tree(), get_econ=False, 
                    get_tax_est=False, get_tax_150=False,
                    get_tax_200=False, get_tax_sl=False,
                    get_tax_ads=False, soi_from_out=False,
                    output_data=False):
    # Reading in the SOI Tax Stats-Corporation data:
    soi_tree = naics.generate_tree()
    soi_tree = read_soi.load_corporate(soi_tree=soi_tree, 
                                       from_out=soi_from_out,
                                       output_data=(not soi_from_out))
    # Reading in the SOI Tax Stats-Partnership data:
    soi_tree = read_soi.load_partner(soi_tree=soi_tree, 
                                     from_out=soi_from_out,
                                     output_data=(not soi_from_out))
    # Reading in the SOI Tax Stats-Proprietorship data:
    soi_tree = read_soi.load_soi_proprietorship(soi_tree=soi_tree, 
                                                from_out=soi_from_out,
                                                output_data=(not soi_from_out))
    '''
    Many industries are not listed in the SOI datasets. The data for these missing
        industries are interpolated.
    '''
    # Get a list of the names of all the pd dfs besides the list of codes:
    #cur_names = soi_tree.enum_inds[0].data.dfs.keys()
    #cur_names.remove(_CODE_DF_NM)
    # Populate missing industry data backwards throught the tree:
    #naics.pop_back(data_tree, cur_names)
    # Populate the missing total corporate data forwards through the tree:
    #naics.pop_forward(data_tree, ["tot_corps"])
    # Populate other missing data using tot_corps as a "blueprint":
    #cur_names = ["c_corps", "s_corps", "PA_inc_loss", "PA_assets", "soi_prop"]
    #naics.pop_forward(data_tree, cur_names, "tot_corps")
    # Calculate c_corps data:
    #read_soi.calc_c_corp(data_tree)
    #naics.pop_back(data_tree,["c_corps"])
    #naics.pop_forward(data_tree, ["c_corps"], "tot_corps")
    # Populate pa05 using pa01:
    #naics.pop_forward(data_tree, ["PA_types"], "PA_inc_loss")
    #
    #naics.pop_back(data_tree, ["farm_prop"])
    #naics.pop_forward(data_tree, ["farm_prop"], "tot_corps")
    
    #Create an output tree containing only the final data on FA, INV, and LAND.
    output_tree = calc_assets.summary_tree(data_tree, _DATA_DIR)
    # Create a tree with all the FA's broken down by type of asset:
    asset_tree = read_bea.read_bea(output_tree, _DATA_DIR)
    naics.pop_back(asset_tree, ["All", "Corp", "Non-Corp"])
    #
    corp_types = ["C Corporations",
                  "Corporate general partners", 
                  "Corporate limited partners"]
    non_corp_types = ["S Corporations",
                      "Individual general partners",
                      "Individual limited partners",
                      "Partnership general partners",
                      "Partnership limited partners",
                      "Tax-exempt organization general partners",
                      "Tax-exempt organization limited partners",
                      "Nominee and other general partners", 
                      "Nominee and other limited partners",
                      "Sole Proprietors"]
    naics.pop_forward(asset_tree, ["All"], "FA", output_tree)
    naics.pop_forward(asset_tree, ["Corp"], "FA", output_tree, corp_types)
    naics.pop_forward(asset_tree, ["Non-Corp"], "FA", output_tree, non_corp_types)
    #
    inv_tree = read_inv.read_inventories(output_tree, _DATA_DIR)
    naics.pop_back(inv_tree, ["Inventories"])
    naics.pop_forward(inv_tree, ["Inventories"])
    #
    land_tree = read_land.read_land(output_tree, _DATA_DIR)
    naics.pop_back(land_tree, ["Land"])
    naics.pop_forward(land_tree, ["Land"], "LAND", output_tree)
    #
    econ_depr_tree = calc_rates.calc_depr_rates(asset_tree, inv_tree, land_tree, _DATA_DIR)
    tax_depr_tree = calc_rates.calc_tax_depr_rates(asset_tree, inv_tree, land_tree, _DATA_DIR)
    naics.pop_rates(tax_depr_tree)
    return {"Econ": econ_depr_tree, "Tax": tax_depr_tree}
    

#
#naics.print_tree_dfs(econ_depr_tree, _OUT_DIR)
#naics.print_tree_dfs(tax_depr_tree, _OUT_DIR)





