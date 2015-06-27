"""
SOI Tax Data (soi_processing.py):
-------------------------------------------------------------------------------
Last updated: 6/26/2015.

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
    :param from_out: If the corporate soi data is already in an output file,
           then it can be read in directly from the output.
    :param get_all: Get corporate soi data for all kinds of corporations.
    :param get_tot: Get the aggregate soi data for corporations.
    :param get_s: Get the soi data for s corporations.
    :param get_c: Interpolate the soi data for c corporations.
    
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
    #
    if get_tot:
        soi_tree = corp.load_soi_tot_corp(data_tree=soi_tree,
                                          from_out=from_out)
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[_TOT_CORP_DF_NM])
    if get_s:
        soi_tree = corp.load_soi_s_corp(data_tree=soi_tree,
                                        from_out=from_out)
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[_S_CORP_DF_NM])
    if get_c:
        soi_tree = corp.calc_c_corp(data_tree=soi_tree,
                                    from_out=from_out)
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[_C_CORP_DF_NM])
    #
    return soi_tree
    


def load_partner(soi_tree=naics.generate_tree(),
                 from_out=False, output_data=False,
                 out_path=_OUT_DIR):
    #
    soi_tree = prt.load_income(data_tree=soi_tree, from_out=from_out)
    soi_tree = prt.load_asset(data_tree=soi_tree, from_out=from_out)
    #soi_tree = prt.load_type(data_tree=soi_tree, from_out=from_out)
    
    #if output_data:
    #    naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
    #                         data_types=[_INC_DF_NM, _AST_DF_NM, _TYP_DF_NM])
    
    return soi_tree


def load_proprietorship(soi_tree=naics.generate_tree(),
                       from_out=False, get_all=False,
                       get_nonfarm=False, get_farm=False,
                       output_data=False, out_path=_OUT_DIR):
    #
    if get_nonfarm:
        soi_tree = prop.load_soi_nonfarm_prop()
        if output_data:
            naics.print_tree_dfs(tree=soi_tree, out_path=out_path,
                                 data_types=[])
    if get_farm:
        soi_tree = prop.load_soi_farm_prop(data_tree=soi_tree,)


