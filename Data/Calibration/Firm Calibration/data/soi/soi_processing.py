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


def load_corporate(soi_tree=naics.generate_tree(),
                   from_out=False, get_all=False,
                   get_tot=False, get_s=False, get_c=False,
                   output_data=False, out_path=_OUT_DIR):
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
                 out_path=_OUT_DIR):
    """ Loading the partnership tax soi data into a NAICS Tree.
    
    :param soi_tree: The NAICS tree to put all of the data in.
    :param from_out: If the corporate soi data is already in an output file,
           then it can be read in directly from the output.
    :param output-data: Print the corporate dataframes to csv files in the
           output folder.
    :param out_path: The output_path, both for reading in output data and for
           printing to the output file
    """
    # Load the soi income data into the NAICS tree:
    soi_tree = prt.load_income(data_tree=soi_tree, from_out=from_out)
    # Load the soi asset data into the NAICS tree:
    soi_tree = prt.load_asset(data_tree=soi_tree, from_out=from_out)
    # Load the soi partnership types data into the NAICS tree:
    soi_tree = prt.load_type(data_tree=soi_tree, from_out=from_out)
    # Outputting the data to csv files in the output folder:
    if output_data:
        naics.print_tree_dfs(tree=soi_tree, out_path=_OUT_DIR,
                             data_types=[_INC_DF_NM, _AST_DF_NM, _TYP_DF_NM])
    return soi_tree


def load_proprietorship(soi_tree=naics.generate_tree(),
                       from_out=False, get_all=False,
                       get_nonfarm=False, get_farm=False,
                       output_data=False, out_path=_OUT_DIR):
    # Get the nonfar
    if get_nonfarm:
        soi_tree = prop.load_soi_nonfarm_prop()
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[])
    if get_farm:
        soi_tree = prop.load_soi_farm_prop(data_tree=soi_tree,)


def get_assets(asset_tree=naics.generate_tree(), soi_tree=None):
    all_sectors = ["C Corporations", 
                   "S Corporations",
                   "Corporate general partners", 
                   "Corporate limited partners",
                   "Individual general partners",
                   "Individual limited partners",
                   "Partnership general partners",
                   "Partnership limited partners",
                   "Tax-exempt organization general partners",
                   "Tax-exempt organization limited partners",
                   "Nominee and other general partners", 
                   "Nominee and other limited partners", 
                   "Sole Proprietors"]
    #
    pa_types = soi_tree.enum_inds[0].data.dfs["PA_types"].columns
    pa_types = pa_types.values.tolist()
    #
    asset_tree = naics.generate_tree()
    #
    for i in asset_tree.enum_inds:
        i.append_dfs(("FA",pd.DataFrame(np.zeros((1, len(all_sectors))),
                                        columns = all_sectors)))
        i.append_dfs(("INV",pd.DataFrame(np.zeros((1, len(all_sectors))),
                                         columns = all_sectors)))
        i.append_dfs(("LAND",pd.DataFrame(np.zeros((1, len(all_sectors))),
                                          columns = all_sectors)))
    #
    for i in range(0, len(asset_tree.enum_inds)):
        #
        #cur_data = soi_tree.enum_inds[i].data
        #out_data = asset_tree.enum_inds[i].data
        cur_dfs = soi_tree.enum_inds[i].data.dfs
        out_dfs = asset_tree.enum_inds[i].data.dfs
        partner_sum = sum(cur_dfs["PA_types"].iloc[0,:])
        #
        for j in range(0, len(all_sectors)):
            sector = all_sectors[j]
            #
            if sector == "C Corporations":
                cur_df = cur_dfs["c_corps"]
                out_dfs["FA"][sector][0] = cur_df["Depreciable Assets"][0]
                out_dfs["INV"][sector][0] = cur_df["Inventories"][0]
                out_dfs["LAND"][sector][0] = cur_df["Land"][0]
            elif sector == "S Corporations":
                cur_df = cur_dfs["s_corps"]
                out_dfs["FA"][sector][0] = cur_df["Depreciable Assets"][0]
                out_dfs["INV"][sector][0] = cur_df["Inventories"][0]
                out_dfs["LAND"][sector][0] = cur_df["Land"][0]
            elif sector in pa_types:
                if partner_sum != 0:
                    ratio = abs(cur_dfs["PA_types"][sector][0])/partner_sum
                else:
                    ratio = abs(1.0/cur_dfs["PA_types"].shape[0])
                cur_df = cur_dfs["PA_assets"]
                out_dfs["FA"][sector][0] = abs(
                                    ratio*cur_df["Depreciable assets (Net)"][0]
                                    )
                out_dfs["INV"][sector][0] = abs(
                                        ratio*cur_df["Inventories (Net)"][0]
                                        )
                out_dfs["LAND"][sector][0] = abs(
                                                ratio*cur_df["Land (Net)"][0]
                                                )
            elif sector == "Sole Proprietors":
                if cur_dfs["PA_inc_loss"]["Depreciation"][0] != 0:
                    ratio = abs(cur_dfs["soi_prop"]["Depr Deductions"][0]/
                                cur_dfs["PA_inc_loss"]["Depreciation"][0])
                else:
                    ratio = 0.0
                cur_df = cur_dfs["PA_assets"]
                out_dfs["FA"][sector][0] = abs(
                                        (ratio*
                                        cur_df["Depreciable assets (Net)"][0])+
                                        cur_dfs["farm_prop"]["FA"][0]
                                        )
                out_dfs["INV"][sector][0] = abs(
                                        (ratio*cur_df["Inventories (Net)"][0])+
                                        cur_dfs["farm_prop"]["Land"][0]
                                        )
                out_dfs["LAND"][sector][0] = abs(ratio*cur_df["Land (Net)"][0])
    return asset_tree
