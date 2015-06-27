"""
SOI Corporate Tax Data (pull_soi_corp.py):
-------------------------------------------------------------------------------
Last updated: 6/26/2015.

This module calibrates the firm economic and tax depreciation parameters.
"""
import os.path
import numpy as np
import pandas as pd
#
import naics_processing as naics
import file_processing as fp
import constants as cst
# Directory names:
_CUR_DIR = os.path.dirname(__file__)
_OUT_DIR = os.path.join(os.path.dirname(_CUR_DIR), "output")
_DATA_DIR = os.path.join(os.path.dirname(_CUR_DIR), "data")
_CORP_DIR = os.path.join(_DATA_DIR, "soi_corporate")
# Dataframe names:
_TOT_DF_NM = cst.TOT_CORP_DF_NM
_S_DF_NM = cst.S_CORP_DF_NM
_C_DF_NM = cst.C_CORP_DF_NM
# (Optional) Hardcode the year that the partner data is from:
_YR = ""
_YR = str(_YR)
# Filenames:
_TOT_CORP_IN_FILE = fp.get_file(dirct=_CORP_DIR, contains=[_YR+"sb1.csv"])
_S_CORP_IN_FILE = fp.get_file(dirct=_CORP_DIR, contains=[_YR+"sb3.csv"])
_TOT_CORP_OUT_FILE = os.path.join(_OUT_DIR, _TOT_DF_NM+".csv")
_S_CORP_OUT_FILE = os.path.join(_OUT_DIR, _S_DF_NM+".csv")
_C_CORP_OUT_FILE = os.path.join(_OUT_DIR, _C_DF_NM+".csv")
# Full path for files:
_TOT_CORP_IN_PATH = os.path.join(_CORP_DIR, _TOT_CORP_IN_FILE)
_S_CORP_IN_PATH = os.path.join(_CORP_DIR, _S_CORP_IN_FILE)
# Constant factors:
_TOT_CORP_IN_FILE_FCTR = 1
_S_CORP_IN_FILE_FCTR = 1
# Input--default dictionaries for df-columns to input-columns.
_DFLT_TOT_CORP_COLS_DICT = cst.DFLT_TOT_CORP_COLS_DICT
_DFLT_S_CORP_COLS_DICT = cst.DFLT_S_CORP_COLS_DICT
# Input--NAICS column:
_NAICS_COL_NM = "INDY_CD"
'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Corporation Data:
(Note: SOI gives data for all corporations as well as for just s-corporations.
    The c-corporation data is inferred from this.)
-------------------------------------------------------------------------------
'''
def load_soi_tot_corp(data_tree=naics.generate_tree(),
                      cols_dict=_DFLT_TOT_CORP_COLS_DICT, 
                      blue_tree=None, blueprint=None,
                      from_out=False, output_path=_TOT_CORP_OUT_FILE):
    if from_out:
        data_tree = naics.load_tree_dfs(input_file=output_path, tree=data_tree)
        return data_tree
    #
    num_inds = len(data_tree.enum_inds)
    data_cols = cols_dict.keys()
    #
    try:
        tot_corp_data = pd.read_csv(_TOT_CORP_IN_PATH).fillna(0)
    except IOError:
        print "IOError: Tot-Corp soi data file not found."
        return None
    # Initializing data on all corporations:
    data_tree.append_all(df_nm=_TOT_DF_NM, df_cols=data_cols)
    # Loading total-corporation data:
    enum_index = 0
    for code_num in np.unique(tot_corp_data[_NAICS_COL_NM]):
        # Find the industry with a code that matches "code_num":
        ind_found = False
        for i in range(0, num_inds):
            enum_index = (enum_index + 1) % num_inds
            cur_ind = data_tree.enum_inds[enum_index]
            cur_dfs = cur_ind.data.dfs[cst.CODE_DF_NM]
            for j in range(0, cur_dfs.shape[0]):
                if(cur_dfs.iloc[j,0] == code_num):
                    # Industry with the matching code has been found:
                    ind_found = True
                    cur_dfs = cur_ind.data.dfs[_TOT_DF_NM]
                    break
            # If the matching industry has been found stop searching for it.
            if ind_found:
                break
        # If no match was found, then ignore data.
        if not ind_found:
            continue
        # Indicators for if rows in tot_corp_data match current industry code:
        indicators = (tot_corp_data[_NAICS_COL_NM] == code_num)
        # Filling in every column in the dataframe:
        for j in cols_dict:
            if cols_dict[j] == "":
                cur_dfs[j] = 0
            else:
                cur_dfs[j][0] = sum(indicators * tot_corp_data[cols_dict[j]])
    #
    naics.pop_back(tree=data_tree, df_list=[_TOT_DF_NM])
    naics.pop_forward(tree=data_tree, df_list=[_TOT_DF_NM],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree
    


def load_soi_s_corp(data_tree=naics.generate_tree(),
                    cols_dict=_DFLT_S_CORP_COLS_DICT,
                    blue_tree=None, blueprint=None,
                    from_out=False, out_path=_S_CORP_OUT_FILE):
    #
    num_inds = len(data_tree.enum_inds)
    data_cols = cols_dict.keys()
    #
    if from_out:
        data_tree = naics.load_tree_dfs(input_file=out_path, tree=data_tree)
        return data_tree
    # The aggregate 1120 filings data for all corporations:
    try:
        s_corp_data = pd.read_csv(_S_CORP_IN_PATH).fillna(0)
    except IOError:
        print "IOError: S-Corp soi data file not found."
        return None
    # Initializing data on all corporations:
    data_tree.append_all(df_nm=_S_DF_NM, df_cols=data_cols)
    # Loading s-corporation data:
    enum_index = 0
    for code_num in np.unique(s_corp_data[_NAICS_COL_NM]):
        # Find the industry with a code that matches "code_num":
        ind_found = False
        for i in range(0, len(data_tree.enum_inds)):
            enum_index = (enum_index + 1) % num_inds
            cur_ind = data_tree.enum_inds[i]
            cur_dfs = cur_ind.data.dfs[cst.CODE_DF_NM]
            for j in range(0, cur_dfs.shape[0]):
                if(cur_dfs.iloc[j,0] == code_num):
                    # Industry with the matching code has been found:
                    ind_found = True
                    cur_dfs = cur_ind.data.dfs[cst.S_CORP_DF_NM]
                    break
            # If the matching industry has been found stop searching for it.
            if ind_found:
                break
        # If no match was found, then ignore data.
        if not ind_found:
            continue
        # Indicators for if rows in s_corp_data match current industry code:
        indicators = (s_corp_data[_NAICS_COL_NM] == code_num)
        # Filling in every column in the dataframe:
        for j in cols_dict:
            # Some are not reported for S Corporations:
            if cols_dict[j] == "":
                cur_dfs[j] = 0
            else:
                cur_dfs.loc[0,j] = sum(indicators * s_corp_data[cols_dict[j]])
    # Default blueprint is tot_corps:
    if blueprint == None and _TOT_DF_NM in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = _TOT_DF_NM
    naics.pop_back(tree=data_tree, df_list=[_S_DF_NM])
    naics.pop_forward(tree=data_tree, df_list=[_S_DF_NM],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree

'''
Calculates the c-corp data from the tot and s corp data.
'''
def calc_c_corp(data_tree=naics.generate_tree(), from_out=False,
                out_path=_C_CORP_OUT_FILE):
    #
    if from_out:
        data_tree = naics.load_tree_dfs(input_file=out_path, tree=data_tree)
        return data_tree
    #
    for ind in data_tree.enum_inds:
        try:
            cur_tot = ind.data.dfs[_TOT_DF_NM]
        except KeyError:
            print "Total-Corp data not initialized when interpolating C-Corp."
        try:
            cur_s = ind.data.dfs[_S_DF_NM]
        except KeyError:
            print "S-Corp data not initialized when interpolating C-Corp."
        data_cols = cur_tot.columns.values.tolist()
        ind.append_dfs((_C_DF_NM, pd.DataFrame(np.zeros((1,len(data_cols))),
                                                columns = data_cols)))
        ind.data.dfs[_C_DF_NM] = cur_tot - cur_s
    #
    return data_tree

