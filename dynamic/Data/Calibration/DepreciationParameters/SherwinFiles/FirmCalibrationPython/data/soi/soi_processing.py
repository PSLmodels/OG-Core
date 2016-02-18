"""
SOI Tax Data (soi_processing.py):
-------------------------------------------------------------------------------
Last updated: 6/29/2015.

This module creates functions for gathering and processing various SOI Tax
data into a NAICS tree.
"""
# Packages:
import os.path
import sys
import numpy as np
import pandas as pd
import xlrd
# Directories:
_CUR_DIR = os.path.dirname(__file__)
_PROC_DIR = os.path.join(_CUR_DIR, "processing")
_OUT_DIR = os.path.join(_CUR_DIR, "output")
_DATA_DIR = os.path.join(_CUR_DIR, "data")
# Importing custom modules:
import naics_processing as naics
import constants as cst
# Importing soi tax data helper custom modules
sys.path.append(_PROC_DIR)
import pull_soi_corp as corp
import pull_soi_partner as prt
import pull_soi_proprietorship as prop
# Dataframe names:
_TOT_CORP_DF_NM = cst.TOT_CORP_DF_NM
_S_CORP_DF_NM = cst.S_CORP_DF_NM
_C_CORP_DF_NM = cst.C_CORP_DF_NM
_INC_DF_NM = cst.INC_PRT_DF_NM
_AST_DF_NM = cst.AST_PRT_DF_NM
_TYP_DF_NM = cst.TYP_PRT_DF_NM
_NFARM_DF_NM = cst.NON_FARM_PROP_DF_NM
_FARM_DF_NM = cst.FARM_PROP_DF_NM
#
_ALL_SECTORS = cst.ALL_SECTORS_NMS_LIST
_ALL_SECTORS_DICT = cst.ALL_SECTORS_NMS_DICT


def load_corporate(soi_tree=naics.generate_tree(),
                   from_out=False, get_all=False,
                   get_tot=False, get_s=False, get_c=False,
                   output_data=False, out_path=None):
    """ Loading the corporate tax soi data into a NAICS Tree.
    
    :param soi_tree: The NAICS tree to put all of the data in.
    :param from_out: If the corporate soi data is already in an output folder,
           then it can be read in directly from the output.
    :param get_all: Get corporate soi data for all kinds of corporations.
    :param get_tot: Get the aggregate soi data for corporations.
    :param get_s: Get the soi data for s corporations.
    :param get_c: Interpolate the soi data for c corporations.
    :param output_data: Print the corporate dataframes to csv files in the
           output folder.
    :param out_path: The output_path, both for reading in output data and for
           printing to the output file
    
    .. note: Because there is only data on the aggregate and s corporations,
       the c corporations data can only be interpolated if the other two have
       been calculated.
    """
    # Initializing the output path:
    if out_path == None:
        out_path = _OUT_DIR
    # Initializing booleans based of initial input booleans:
    if get_all:
        get_tot = True
        get_s = True
        get_c = True
    if not get_tot or not get_s:
        get_c = False
    # Load the total corporate soi data into the NAICS tree:
    if get_tot:
        soi_tree = corp.load_soi_tot_corp(data_tree=soi_tree,
                                          from_out=from_out)
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[_TOT_CORP_DF_NM])
    # Load the S-corporate soi data into the NAICS tree:
    if get_s:
        soi_tree = corp.load_soi_s_corp(data_tree=soi_tree,
                                        from_out=from_out)
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[_S_CORP_DF_NM])
    # Calculate the C-corporate soi data for the NAICS tree:
    if get_c:
        soi_tree = corp.calc_c_corp(data_tree=soi_tree,
                                    from_out=from_out)
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[_C_CORP_DF_NM])
    return soi_tree
    

def load_partner(soi_tree=naics.generate_tree(),
                 from_out=False, output_data=False,
                 out_path=None):
    """ Loading the partnership tax soi data into a NAICS Tree.
    
    :param soi_tree: The NAICS tree to put all of the data in.
    :param from_out: If the corporate soi data is already in an output file,
           then it can be read in directly from the output.
    :param output_data: Print the corporate dataframes to csv files in the
           output folder.
    :param out_path: The output_path, both for reading in output data and for
           printing to the output file
    """
    # Initializing the output path:
    if out_path == None:
        out_path = _OUT_DIR
    # Load the soi income data into the NAICS tree:
    soi_tree = prt.load_income(data_tree=soi_tree, from_out=from_out)
    # Load the soi asset data into the NAICS tree:
    soi_tree = prt.load_asset(data_tree=soi_tree, from_out=from_out)
    # Load the soi partnership types data into the NAICS tree:
    soi_tree = prt.load_type(data_tree=soi_tree, from_out=from_out)
    # Output the data to csv files in the output folder:
    if output_data:
        naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                             data_types=[_INC_DF_NM, _AST_DF_NM, _TYP_DF_NM])
    return soi_tree


def load_proprietorship(soi_tree=naics.generate_tree(),
                       from_out=False, get_all=False,
                       get_nonfarm=False, get_farm=False,
                       output_data=False, out_path=None):
    """ Loading the proprietorship tax soi data into a NAICS Tree.
    
    :param soi_tree: The NAICS tree to put all of the data in.
    :param from_out: If the corporate soi data is already in an output file,
           then it can be read in directly from the output.
    :param output_data: Print the corporate dataframes to csv files in the
           output folder.
    :param out_path: The output_path, both for reading in output data and for
           printing to the output file
    """
    # Initializing the output path:
    if out_path == None:
        out_path = _OUT_DIR
    # Load the soi nonfarm data into the NAICS tree:
    if get_nonfarm:
        soi_tree = prop.load_soi_nonfarm_prop(
                                    data_tree=soi_tree, from_out=from_out
                                    )
    # Load the farm data into to the NAICS tree:
    if get_farm:
        soi_tree = prop.load_soi_farm_prop(
                                    data_tree=soi_tree, from_out=from_out
                                    )
    # Output the data to csv files in the output folder:
    if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[_NFARM_DF_NM, _FARM_DF_NM])
    return soi_tree


def calc_assets(soi_tree, asset_tree=naics.generate_tree()):
    """ Calculating a breakdown of the various sector type's assets
    into fixed assets, inventories, and land. 
    
    :param asset_tree: The NAICS tree to put all of the data in.
    :param soi_tree: A NAICS tree containing all the pertinent soi data.
    """
    # Initializing dataframes for all NAICS industries:
    asset_tree.append_all(df_nm="FA", df_cols=_ALL_SECTORS)
    asset_tree.append_all(df_nm="INV", df_cols=_ALL_SECTORS)
    asset_tree.append_all(df_nm="LAND", df_cols=_ALL_SECTORS)
    # Calculate fixed assets, inventories, and land for each industry/sector
    for i in range(0, len(asset_tree.enum_inds)):
        cur_dfs = soi_tree.enum_inds[i].data.dfs
        out_dfs = asset_tree.enum_inds[i].data.dfs
        # Total of all the partner data for the current industry:
        partner_sum = sum(cur_dfs[_TYP_DF_NM].iloc[0,:]) 
        # C-Corporations:
        sector = _ALL_SECTORS_DICT["C_CORP"]
        cur_df = cur_dfs[_C_CORP_DF_NM]
        out_dfs["FA"][sector][0] = cur_df["depreciable_assets"][0]
        out_dfs["INV"][sector][0] = cur_df["inventories"][0]
        out_dfs["LAND"][sector][0] = cur_df["land"][0]
        # S-Corporations:
        sector = _ALL_SECTORS_DICT["S_CORP"]
        cur_df = cur_dfs[_S_CORP_DF_NM]
        out_dfs["FA"][sector][0] = cur_df["depreciable_assets"][0]
        out_dfs["INV"][sector][0] = cur_df["inventories"][0]
        out_dfs["LAND"][sector][0] = cur_df["land"][0]
        # Partnership sectors:
        for sector in cst.DFLT_PRT_TYP_DF_COL_NMS_DICT.values():
            if partner_sum != 0:
                ratio = abs(float(cur_dfs[_TYP_DF_NM][sector][0]))/partner_sum
            else:
                ratio = abs(1.0/float(cur_dfs[_TYP_DF_NM].shape[0]))
            cur_df = cur_dfs[_AST_DF_NM]
            out_dfs["FA"][sector][0] = abs(
                                ratio*cur_df["depreciable_assets_net"][0]
                                )
            out_dfs["INV"][sector][0] = abs(
                                    ratio*cur_df["inventories_net"][0]
                                    )
            out_dfs["LAND"][sector][0] = abs(
                                            ratio*cur_df["land_net"][0]
                                            )
        # Sole Proprietorships:
        sector = _ALL_SECTORS_DICT["SOLE_PROP"]
        if cur_dfs[_INC_DF_NM]["depreciation"][0] != 0:
            ratio = abs(float(cur_dfs[_NFARM_DF_NM]["depreciation_deductions"][0])/
                        cur_dfs[_INC_DF_NM]["depreciation"][0])
        else:
            ratio = 0.0
        cur_df = cur_dfs[_AST_DF_NM]
        out_dfs["FA"][sector][0] = abs(
                                (ratio*
                                cur_df["depreciable_assets_net"][0])+
                                cur_dfs[_FARM_DF_NM]["FA"][0]
                                )
        out_dfs["INV"][sector][0] = abs(
                                (ratio*cur_df["inventories_net"][0])+
                                cur_dfs[_FARM_DF_NM]["Land"][0]
                                )
        out_dfs["LAND"][sector][0] = abs(ratio*cur_df["land_net"][0])
    return asset_tree

