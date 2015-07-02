'''
Allocating Inventories Across Firms (read_land.py):
-------------------------------------------------------------------------------
Last updated 7/1/2015

For each NAICS code, this module calculates the inventories held by firms
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
_INV_DIR = os.path.join(_DATA_DIR, "Inventories")
# Importing custom modules:
import naics_processing as naics
import constants as cst
# Full file paths:
_INV_IN_PATH = os.path.join(_INV_DIR, "Inventories.xls")
_INV_IN_CROSS_PATH = os.path.join(_INV_DIR, "Inventories_Crosswalk.csv")
# Dataframe column names:
_CORP_TAX_SECTORS_NMS_DICT = cst.CORP_TAX_SECTORS_NMS_DICT
_CORP_NMS = _CORP_TAX_SECTORS_NMS_DICT.values()
_NON_CORP_TAX_SECTORS_NMS_DICT = cst.NON_CORP_TAX_SECTORS_NMS_DICT
_NCORP_NMS = _NON_CORP_TAX_SECTORS_NMS_DICT.values()
# Constant factors:
_INV_IN_FILE_FCTR = 10**9


def read_inventories(asset_tree):
    # Opening BEA's excel file on depreciable assets by industry:
    inv_book = xlrd.open_workbook(_INV_IN_PATH)
    sht0 = inv_book.sheet_by_index(0)
    num_rows = sht0.nrows
    num_cols = sht0.ncols
    #Find the starting index in worksheet.
    cur_index = naics.search_ws(sht0, 1, 25, True, [0,0], True)
    check_index = naics.search_ws(sht0, "line", 20)
    if(cur_index[1] != check_index[1]):
        print "ERROR"
    # Reading in the crosswalk:
    inv_cross = pd.read_csv(_INV_IN_CROSS_PATH)
    # Creating a tree for the inventory data:
    data_cols = ["All", "Corp", "Non-Corp"]
    inv_tree = naics.generate_tree()
    inv_tree.append_all(df_nm="Inventories", df_cols=data_cols)
    #
    inv_data = np.zeros(inv_cross.shape[0])
    #
    cross_index = 0
    for i in xrange(cur_index[0], num_rows):
        if(cross_index >= inv_cross.shape[0]):
            break
        cur_list = str(sht0.cell_value(i, cur_index[1])).strip()
        cur_name = str(sht0.cell_value(i, cur_index[1]+1)).strip()
        checks = ((str(cur_list) == str(inv_cross["List"][cross_index])) and 
                    (str(cur_name) == str(inv_cross["Industry"][cross_index])))
        if(checks):
            cross_index += 1
            try:
                cur_value = float(sht0.cell_value(i, num_cols-1))
            except ValueError:
                continue
            inv_data[cross_index-1] = cur_value
            # Data is in billions:
            inv_data[cross_index-1] = _INV_IN_FILE_FCTR * inv_data[cross_index-1]
    #
    for i in xrange(0, inv_cross.shape[0]):
        cur_codes = inv_cross["NAICS"][i].strip().split(".")
        proportions = naics.get_proportions(cur_codes, asset_tree, "INV")
        for j in xrange(0, proportions.shape[1]):
            cur_ind = inv_tree.enum_inds[int(proportions.iloc[0,j])]
            prev_ind = asset_tree.enum_inds[int(proportions.iloc[0,j])]
            prev_df = prev_ind.data.dfs["INV"]
            if(sum(prev_df.iloc[0, :]) != 0):
                cur_dfs = ((prev_df/sum(prev_df.iloc[0,:])) *
                                (inv_data[i] * proportions.iloc[1,j]))
                inv_df = cur_ind.data.dfs["Inventories"]
                inv_df["All"] += sum(cur_dfs.iloc[0,:])
                for k in _CORP_NMS:
                    inv_df["Corp"] += cur_dfs[k][0]
                for k in _NCORP_NMS:
                    inv_df["Non-Corp"] += cur_dfs[k][0]
    #
    naics.pop_back(inv_tree, ["Inventories"])
    naics.pop_forward(inv_tree, ["Inventories"], "INV", asset_tree)
    return inv_tree











