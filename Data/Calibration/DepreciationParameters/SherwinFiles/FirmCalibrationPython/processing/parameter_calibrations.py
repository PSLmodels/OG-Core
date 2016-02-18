"""
Parameter Calibrations (parameter_calibrations.py):
-------------------------------------------------------------------------------
Last updated: 6/26/2015.

This module creates functions that carry out various parameter calibrations.
This is the most important firm calibrations module in that it brings all
the various firm calibrations together by calling all the various helper
modules.
"""
# Packages:
import os.path
import sys
import numpy as np
import pandas as pd
import ipdb
# Relevant directories:
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(_CUR_DIR)
_DATA_DIR = os.path.abspath(_MAIN_DIR + "//data")
_DATA_STRCT_DIR = os.path.abspath(_MAIN_DIR + "//data_structures")
_PARAM_DIR = os.path.abspath(_MAIN_DIR + "//parameters")
_OUT_DIR = os.path.abspath(_MAIN_DIR + "//output")
# Importing custom modules:
sys.path.append(_DATA_STRCT_DIR)
import naics_processing as naics
import file_processing as fp


def calibrate_incomes(output_data=True):
    """ This calibrates a tree of all the income data parameters.
    
    :param out: Whether to output the dataframes in the final tree to the
           output file.
    """
    # The income directory:
    inc_dir = os.path.abspath(_PARAM_DIR + "//national_income")
    # Importing the module for gathering and processing the income data:
    sys.path.append(inc_dir)
    import national_income as inc
    # Get all the income data in an income tree:
    inc_tree = inc.get_incs()
    # Output the data to the income folder inside the output folder:
    if output_data:
        inc_out_dir = os.path.abspath(_OUT_DIR + "//income")
        # Make income folder if there isn't one:
        if not os.path.isdir(inc_out_dir):
            os.mkdir(inc_out_dir)
        # Print the data in the tree:
        naics.print_tree_dfs(inc_tree, inc_out_dir)
    return inc_tree


def calibrate_depr_rates(data_tree=naics.generate_tree(), get_all=False,
                         get_econ=False, get_tax=False,
                         get_tax_est=False, get_tax_150=False,
                         get_tax_200=False, get_tax_sl=False,
                         get_tax_ads=False, soi_from_out=False,
                         output_data=False):
    """ This calibrates a tree with all the depreciation rate parameters.
    
    :param data_tree: The NAICS tree to append the calibrated depreciation
           parameters to. Default is a newly generated tree.
    :param get_all: Whether to get all the depreciation parameters or not.
    :param get_econ: Whether to get all the economic depreciation rates.
    :param get_tax: Whether to get all of the tax data.
    :param get_tax_est: Whether to get all of the estimated tax data. This is
           the the most accurate estimate for each industry's depreciation
           rate. It is uses IRS tax documents to decide which assets fall
           under which tax depreciation methods.
    :param get_tax_150: Get the depreciation rates under the assumption that
           assets are depreciated under the GDS 150% declining balance method.
    :param get_tax_200: Get the depreciation rates under the assumption that
           assets are depreciated under the GDS 200% declining balance method.
    :param get_tax_sl: Get the depreciation rates under the assumption that
           assets are depreciated under the straight line method.
    :param get_tax_ads: Get the depreciation rates under the assumption that
           assets are depreciated under the ADS method.
    :param soi_from_out: Whether to recalibrate the relevant soi tax data.
    :param output_data: Whether to output the depreciation rates.
    """
    # The depreciation rate directory:
    depr_dir = os.path.abspath(_PARAM_DIR +"//depreciation")
    ''' Importing the module for gathering and processing the depreciation
    rate data: '''
    sys.path.append(depr_dir)
    import depreciation_calibration as depr
    import cPickle as pickle
    # If get_all, set all booleans to true:
    if get_all:
        get_econ = True
        get_tax = True
    # If get_tax, set all tax booleans to true:
    if get_tax:
        get_tax_150 = True
        get_tax_200 = True
        get_tax_sl = True
        get_tax_ads = True
        get_tax_est = True
    # Initialize NAICS tree with all the soi tax data:
    '''
    Lines 104-105 and 108 are commented out and replaced by lines 110 and 111, which load the same information from memory
    '''
    #soi_tree = pull_soi_data(get_all=True, from_out=soi_from_out,
                      #       output_data=(not soi_from_out))
    ''' Initialize NAICS tree with all assets--fixed assets, inventories, 
    and land--by sector:'''
    #asset_tree = calc_soi_assets(soi_tree=soi_tree)
    # Use the asset_tree to initialize all the depreciation rates:
    input_file = open('myfile.pkl', 'rb')
    asset_tree = pickle.load(input_file)
    input_file.close()
    depr_tree = depr.init_depr_rates(asset_tree=asset_tree, get_econ=get_econ,
                            get_tax_est=get_tax_est, get_tax_150=get_tax_150,
                            get_tax_200=get_tax_200, get_tax_sl=get_tax_sl,
                            get_tax_ads=get_tax_ads, output_data=output_data)
    
    #
    return depr_tree


def calibrate_debt(debt_tree=naics.generate_tree(), soi_tree=None,
                   from_out=False, soi_from_out=False):
    """ This function is incomplete. This is supposed to do the debt
    calibrations.
    
    :param debt_tree: The NAICS tree to append the calibrated debt
           parameters to. Default is a newly generated tree.
    :param soi_tree: A tree with all of the relevant soi data.
    :
    """
    if soi_tree == None:
        soi_tree = pull_soi_data(get_corp=True, from_out=soi_from_out)
    #
    debt_dir = os.path.abspath(_PARAM_DIR + "//debt")
    debt_data_dir = os.path.abspath(debt_dir + "//data")
    sys.path.append(debt_dir)
    import debt_calibration as debt
    #
    lblty_file = os.path.abspath(debt_data_dir + "//liabilities.csv")
    print lblty_file
    lblty_df = pd.read_csv(lblty_file)
    eqty_file = os.path.abspath(debt_data_dir + "//equity.csv")
    eqty_df = pd.read_csv(eqty_file)
    debt_tree = naics.load_tree_dfs(input_file=lblty_file, dfs_name="liabilities", tree=debt_tree)
    debt_tree = naics.load_tree_dfs(input_file=eqty_file, dfs_name="equity", tree=debt_tree)
    #
    naics.pop_forward(tree=debt_tree, df_list=["liabilities"],
                      blue_tree=soi_tree, blueprint="tot_corps",
                      sub_print = ["Interest Paid"])
    #
    return debt_tree
    

def pull_soi_data(soi_tree=naics.generate_tree(), from_out=False,
                  get_all=False, get_corp=False,
                  get_tot=False, get_s=False,
                  get_c=False, get_prt=False,
                  get_prop=False, get_farm_prop=False,
                  output_data=False, out_path=None):
    # If get_all, set all booleans to true:
    if get_all:
        get_corp = True
        get_tot = True
        get_s = True
        get_c = True
        get_prt = True
        get_prop = True
        get_farm_prop = True
    # Import the soi_processing custom module:
    soi_dir = os.path.join(_DATA_DIR, "soi")
    sys.path.append(soi_dir)
    import soi_processing as soi
    # Loading the soi corporate data into the NAICS tree:
    if get_corp or get_tot or get_s or get_c:
        soi_tree = soi.load_corporate(
                                soi_tree=soi_tree, from_out=from_out,
                                get_all=get_corp, get_tot=get_tot,
                                get_s=get_s, get_c=get_c,
                                output_data=output_data, out_path=out_path
                                )
    # Loading the soi partnership data into the NAICS tree:
    if get_prt:
        soi_tree = soi.load_partner(soi_tree=soi_tree, from_out=from_out,
                                    output_data=output_data, out_path=out_path)
    # Loading the soi proprietorship data into the NAICS tree:
    if get_prop or get_farm_prop:
        soi_tree = soi.load_proprietorship(
                            soi_tree=soi_tree, from_out=from_out,
                            get_nonfarm=get_prop, get_farm=get_farm_prop,
                            output_data=output_data, out_path=out_path
                            )
    return soi_tree


def calc_soi_assets(soi_tree, asset_tree=naics.generate_tree()):
    """ Calculating a breakdown of the various sector type's assets
    into fixed assets, inventories, and land. 
    
    :param asset_tree: The NAICS tree to put all of the data in.
    :param soi_tree: A NAICS tree containing all the pertinent soi data.
    """
    # Import the soi_processing custom module:
    soi_dir = os.path.join(_DATA_DIR, "soi")
    sys.path.append(soi_dir)
    import soi_processing as soi
    # Use soi processing helper function to do all the work:
    return soi.calc_assets(asset_tree=asset_tree, soi_tree=soi_tree)

