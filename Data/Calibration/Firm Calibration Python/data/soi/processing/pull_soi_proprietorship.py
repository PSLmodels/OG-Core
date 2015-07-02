'''
SOI Proprietorship Tax Data (pull_soi_proprietorship.py):
-------------------------------------------------------------------------------
Last updated: 6/29/2015.

This module creates functions for pulling the proprietorship soi tax data into
NAICS trees.
'''
# Packages:
import os.path
import numpy as np
import pandas as pd
import xlrd
# Directories:
_CUR_DIR = os.path.dirname(__file__)
_OUT_DIR = os.path.join(os.path.dirname(_CUR_DIR), "output")
_DATA_DIR = os.path.join(os.path.dirname(_CUR_DIR), "data")
_PROP_DIR = os.path.join(_DATA_DIR, "soi_proprietorship")
# Importing custom packages:
import naics_processing as naics
import file_processing as fp
import constants as cst
# Dataframe names:
_FARM_DF_NM = cst.FARM_PROP_DF_NM
_NFARM_DF_NM = cst.NON_FARM_PROP_DF_NM
_CODE_DF_NM = cst.CODE_DF_NM
_TOT_CORP_DF_NM = cst.TOT_CORP_DF_NM
_AST_PRT_DF_NM = cst.AST_PRT_DF_NM
# (Optional) Hardcode the year that the partner data is from:
_YR = ""
_YR = str(_YR)
# Filenames:
_DDCT_IN_FILE = fp.get_file(dirct=_PROP_DIR, contains=[_YR+"sp01br.xls"])
_FARM_IN_FILE = fp.get_file(dirct=_PROP_DIR, contains=["farm_data.csv"])
_DDCT_IN_CROSS_FILE = fp.get_file(dirct=_PROP_DIR,
                                  contains=[_YR+"sp01br_Crosswalk.csv"])
# Full path for files:
_DDCT_IN_PATH = os.path.join(_PROP_DIR, _DDCT_IN_FILE)
_FARM_IN_PATH = os.path.join(_PROP_DIR, _FARM_IN_FILE)
_DDCT_IN_CROSS_PATH = os.path.join(_PROP_DIR, _DDCT_IN_CROSS_FILE)
_NFARM_PROP_OUT_PATH = os.path.join(_OUT_DIR, _NFARM_DF_NM+".csv")
_FARM_PROP_OUT_PATH = os.path.join(_OUT_DIR, _FARM_DF_NM+".csv")
# Constant factors:
_DDCT_FILE_FCTR = 10**3
# Dataframe columns:
_NFARM_DF_COL_NMS = cst.DFLT_PROP_NFARM_DF_COL_NMS_DICT
_NFARM_DF_COL_NM = _NFARM_DF_COL_NMS["DEPR_DDCT"]
_AST_DF_COL_NMS_DICT = cst.DFLT_PRT_AST_DF_COL_NMS_DICT
_LAND_COL_NM = _AST_DF_COL_NMS_DICT["LAND_NET"]
_DEPR_COL_NM = _AST_DF_COL_NMS_DICT["DEPR_AST_NET"]
# Input--relevant row/column names in the nonfarm prop types excel worksheet:
_IN_COL_DF_DICT = dict([
                    ("Depreciation\ndeduction", "DEPR_DDCT")
                    ])
_SECTOR_COL = "Industrial sector"
_DDCT_COL1 = "Depreciation\ndeduction"
_DDCT_COL2 = "Depreciation\ndeduction"


def load_soi_nonfarm_prop(data_tree=naics.generate_tree(), 
                          blue_tree=None, blueprint=None, 
                          from_out=False, out_path=_NFARM_PROP_OUT_PATH):
    """ This function loads the soi nonfarm proprietorship data:
    
    :param data_tree: The NAICS tree to read the data into.
    :param cols_dict: A dictionary mapping dataframe columns to the name of
           the column names in the input file
    :param blueprint: The key corresponding to a dataframe in a tree to be
           used as a "blueprint" for populating the df_list dataframes forward.
    :param blue_tree: A NAICS tree with the "blueprint" dataframe. The default
           is the original NAICS tree.
    :param from_out: Whether to read in the data from output.
    :param output_path: The path of the output file.
    """
    # If from_out, load the data tree from output:
    if from_out:
        data_tree = naics.load_tree_dfs(input_path=out_path, tree=data_tree)
        return data_tree
    # Opening nonfarm proprietor data:
    wb = xlrd.open_workbook(_DDCT_IN_PATH)
    ws = wb.sheet_by_index(0)
    cross = pd.read_csv(_DDCT_IN_CROSS_PATH)
    # Finding the relevant positions in worksheet:
    pos1 = naics.search_ws(ws, _SECTOR_COL, 20, True, [0,0], True)
    pos2 = naics.search_ws(ws, _DDCT_COL1, 20)
    pos3 = naics.search_ws(ws,_DDCT_COL2, 20,
                           True, np.array(pos2) + np.array([0,1]))
    #
    data_tree.append_all(df_nm=_NFARM_DF_NM, df_cols=[_NFARM_DF_COL_NM])
    #
    cross_index = cross.shape[0]-1
    enum_index = len(data_tree.enum_inds)-1
    for i in xrange(pos1[0],ws.nrows):
        cur_cell = str(ws.cell_value(i,pos1[1])).lower().strip()
        #
        tot_proportions = 0
        for j in xrange(0, cross.shape[0]):
            cross_index = (cross_index+1) % cross.shape[0]
            cur_ind_name = str(cross.iloc[cross_index,0]).lower().strip()
            if(cur_cell == cur_ind_name):
                if pd.isnull(cross.iloc[cross_index,1]):
                    continue
                ind_codes = str(cross.iloc[cross_index,1]).split(".")
                for k in xrange(0, len(data_tree.enum_inds)):
                    enum_index = (enum_index+1) % len(data_tree.enum_inds)
                    cur_data = data_tree.enum_inds[enum_index].data
                    cur_codes = cur_data.dfs[_CODE_DF_NM]
                    cur_proportions = naics.compare_codes(ind_codes, cur_codes.iloc[:,0])
                    if cur_proportions == 0:
                        continue
                    tot_proportions += cur_proportions
                    cur_dfs = cur_data.dfs[_NFARM_DF_NM][_NFARM_DF_COL_NM]
                    cur_dfs[0] += (_DDCT_FILE_FCTR * cur_proportions 
                                        * (ws.cell_value(i,pos2[1]) 
                                        + ws.cell_value(i,pos3[1])))
            if(tot_proportions == 1):
                break
    # Default:
    if blueprint == None and _TOT_CORP_DF_NM in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = _TOT_CORP_DF_NM
    naics.pop_back(tree=data_tree, df_list=[_NFARM_DF_NM])
    naics.pop_forward(tree=data_tree, df_list=[_NFARM_DF_NM],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree


def load_soi_farm_prop(data_tree=naics.generate_tree(),
                       blue_tree=None, blueprint=None,
                       from_out=False, out_path=_FARM_PROP_OUT_PATH):
    """ This function loads the soi nonfarm proprietorship data:
    
    :param data_tree: The NAICS tree to read the data into.
    :param cols_dict: A dictionary mapping dataframe columns to the name of
           the column names in the input file
    :param blueprint: The key corresponding to a dataframe in a tree to be
           used as a "blueprint" for populating the df_list dataframes forward.
    :param blue_tree: A NAICS tree with the "blueprint" dataframe. The default
           is the original NAICS tree.
    :param from_out: Whether to read in the data from output.
    :param output_path: The path of the output file.
    """
    # If from_out, load the data tree from output:
    if from_out:
        data_tree = naics.load_tree_dfs(input_path=out_path, tree=data_tree)
        return data_tree
    # Load Farm Proprietorship data:
    farm_data = pd.read_csv(_FARM_IN_PATH)
    new_farm_cols = ["Land", "FA"]
    #
    data_tree.append_all(df_nm=_FARM_DF_NM, df_cols=new_farm_cols)
    #
    land_mult = ((farm_data["R_sp"][0] + farm_data["Q_sp"][0]) * 
                        (float(farm_data["A_sp"][0])/farm_data["A_p"][0]))
    total = farm_data["R_p"][0] + farm_data["Q_p"][0]
    total_pa = 0
    cur_codes = [111,112]
    proportions = np.zeros(len(cur_codes))
    proportions = naics.get_proportions(cur_codes, data_tree, _AST_PRT_DF_NM, 
                                 [_LAND_COL_NM, _DEPR_COL_NM])
    #
    for ind_code in cur_codes:
        cur_ind = naics.find_naics(data_tree, ind_code)
        cur_df = cur_ind.data.dfs[_AST_PRT_DF_NM]
        total_pa += (cur_df[_LAND_COL_NM][0] + cur_df[_DEPR_COL_NM][0])
    #
    for i in xrange(0,len(cur_codes)):
        cur_ind = naics.find_naics(data_tree, cur_codes[i])
        cur_ind.data.dfs[_FARM_DF_NM]["Land"][0] = (land_mult * 
                            cur_ind.data.dfs[_AST_PRT_DF_NM][_LAND_COL_NM][0]/
                            total_pa)
        cur_ind.data.dfs[_FARM_DF_NM]["FA"][0] = ((proportions.iloc[1,i]*total)
                                    - cur_ind.data.dfs[_FARM_DF_NM]["Land"][0])
    # Default:            
    if blueprint == None and _TOT_CORP_DF_NM in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = _TOT_CORP_DF_NM
    naics.pop_back(tree=data_tree, df_list=[_FARM_DF_NM])
    naics.pop_forward(tree=data_tree, df_list=[_FARM_DF_NM],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree

