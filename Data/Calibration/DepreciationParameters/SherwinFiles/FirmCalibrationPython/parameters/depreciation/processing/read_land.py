'''
Allocating Land Across Firms (read_land.py):
-------------------------------------------------------------------------------
Last updated 7/1/2015

For each NAICS code, this module calculates the land held by firms
that are taxed like corporations and those that are not taxed like
corporations.
'''
# Packages:
import os.path
import numpy as np
import pandas as pd
import xlrd
# Directories:
_CUR_DIR = os.path.dirname(__file__)
_MAIN_DIR = os.path.dirname(_CUR_DIR)
_DATA_DIR = os.path.join(_MAIN_DIR, "data")
_LAND_DIR = os.path.join(_DATA_DIR, "Land")
# Importing custom modules:
import naics_processing as naics
import constants as cst
# Full file paths:
_LAND_IN_PATH = os.path.join(_LAND_DIR, "Fin_Accounts-Land.csv")
# Dataframe column names:
_CORP_TAX_SECTORS_NMS_DICT = cst.CORP_TAX_SECTORS_NMS_DICT
_CORP_NMS = _CORP_TAX_SECTORS_NMS_DICT.values()
_NON_CORP_TAX_SECTORS_NMS_DICT = cst.NON_CORP_TAX_SECTORS_NMS_DICT
_NCORP_NMS = _NON_CORP_TAX_SECTORS_NMS_DICT.values()
# Constant factors:
_LAND_IN_FILE_FCTR = 10**9


def read_land(asset_tree):
    land_data = pd.read_csv(_LAND_IN_PATH)
    land_data = _LAND_IN_FILE_FCTR * land_data
    # Initializing NAICS tree for the land data:
    df_cols = ["All", "Corp", "Non-Corp"]
    land_tree = naics.generate_tree()
    land_tree.append_all(df_nm="Land", df_cols=df_cols)
    ''' Calculate the proportion that belong in corporate and non-corporate
    tax categories:'''
    corp_sum = 0.0
    non_corp_sum = 0.0
    for i in _CORP_NMS:
        corp_sum += asset_tree.enum_inds[0].data.dfs["LAND"][i][0]
    for i in _NCORP_NMS:
        non_corp_sum += asset_tree.enum_inds[0].data.dfs["LAND"][i][0]
    if corp_sum + non_corp_sum == 0:
        return land_tree
    ''' Initialize the total industry category--corresponding to NAICS code 
    of "1": '''
    land_df = land_tree.enum_inds[0].data.dfs["Land"]
    land_df["Corp"][0] = land_data["Corporate"][0]
    land_df["Non-Corp"][0] = land_data["Non-Corporate"][0]
    land_df["All"][0] = (land_data["Corporate"][0]+
                            land_data["Non-Corporate"][0])
    # Use the asset_tree to populate the rest:
    naics.pop_back(land_tree, ["Land"])
    naics.pop_forward(land_tree, ["Land"], "LAND", asset_tree)
    return land_tree












