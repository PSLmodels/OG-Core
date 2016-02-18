"""
Calculate Depreciation Rates (calc_rates):
-------------------------------------------------------------------------------
Last updated: 5/26/2015.

This module provides functions for calculating economic and tax depreciation
rates. All of the data is processed into NAICS trees by other modules, and
taken in as input of these functions. 
"""
# Packages:
import os.path
import numpy as np
import pandas as pd
import xlrd
import ipdb
# Directories:
_CUR_DIR = os.path.dirname(__file__)
_MAIN_DIR = os.path.dirname(_CUR_DIR)
_DATA_DIR = os.path.join(_MAIN_DIR, "data")
_DEPR_DIR = os.path.join(_DATA_DIR, "Depreciation Rates")
# Importing custom modules:
import naics_processing as naics
import constants as cst
# Full file paths:
_ECON_DEPR_IN_PATH = os.path.join(_DEPR_DIR, "Economic Depreciation Rates.csv")
_TAX_DEPR_IN_PATH = os.path.join(_DEPR_DIR, "BEA_IRS_Crosswalk.csv")



def calc_depr_rates(asset_tree, inv_tree, land_tree):
    # Opening file containing depreciation rates by asset type:
    depr_econ = pd.read_csv(_ECON_DEPR_IN_PATH)
    depr_econ = depr_econ.fillna(0)
    econ_assets = depr_econ["Asset"]
    econ_rates = depr_econ["Economic Depreciation Rate"]
    #
    types = ["All", "Corp", "Non-Corp"]
    # Initialize tree for depreciation rates:
    depr_tree = naics.generate_tree()
    depr_tree.append_all(df_nm="Economic", df_cols=types)
    #Runs three times, once for all the assets, once for the corporate assets, and once for non-corporate assets
    for i in types:
        asset_list = asset_tree.enum_inds[0].data.dfs[i].columns
        asset_list = asset_list.values.tolist()
        
        '''
        match = np.array([-1] * len(asset_list))
        for j in xrange(0, asset_tree.enum_inds[0].data.dfs[i].shape[1]):
            for k in xrange(0, len(econ_assets)):
                if str(asset_list[j]).strip() == str(econ_assets[k]).strip():
                    match[j] = k
        '''
        #Iterates over every industry in the tree       
        for j in xrange(0, len(depr_tree.enum_inds)):
            total_depreciation = 0
            asset_df = asset_tree.enum_inds[j].data.dfs[i]
            depr_df = depr_tree.enum_inds[j].data.dfs['Economic']
            asset_depreciation = 0
            #Iterates over each asset in the industry
            for k in xrange(0, len(asset_list)):
                '''
               if(match[k] == -1):
                    print k
                    continue   
                '''       
                #calculates the sum of all the depreciation in the industry, multiplying the amount of each asset by its corresponding depreciation rate
                asset_depreciation += (asset_df.iloc[0,k] * econ_rates[k])
            '''
            if(sum(asset_df.iloc[0,:]) != 0):
                depr_df[i][0] = total_depreciation/sum(asset_df.iloc[0,:])
            else:
                depr_df[i][0] = 0
            '''
            tot_assets = sum(asset_tree.enum_inds[j].data.dfs[i].iloc[0,:])
            tot_inv = inv_tree.enum_inds[j].data.dfs["Inventories"][i][0]
            tot_land = land_tree.enum_inds[j].data.dfs["Land"][i][0]
            total_capital_stock = tot_assets + tot_inv + tot_land
            #
            #cur_df = depr_tree.enum_inds[j].data.dfs["Economic"]
           
            if(total_capital_stock != 0):
                #calculates the weighted average depreciation rate
                depr_df[i][0] = asset_depreciation / tot_assets
            else:
                depr_df[i][0] = 0
            
            
        
       #Inventories and land have an approximately zero depreciation rate:
        '''
        for j in xrange(0, len(depr_tree.enum_inds)):
            tot_assets = sum(asset_tree.enum_inds[j].data.dfs["All"].iloc[0,:])
            tot_inv = inv_tree.enum_inds[j].data.dfs["Inventories"]["All"][0]
            tot_land = land_tree.enum_inds[j].data.dfs["Land"]["All"][0]
            if(tot_assets+tot_inv+tot_land == 0):
                continue
            ratio = tot_assets / (tot_assets + tot_inv + tot_land)
            #
            cur_df = depr_tree.enum_inds[j].data.dfs["Economic"]
            cur_df[i][0] = ratio * cur_df[i][0]
        '''

    return depr_tree


'''

'''
def calc_tax_depr_rates(asset_tree, inv_tree, land_tree):
    #
    tax_data = pd.read_csv(_TAX_DEPR_IN_PATH).fillna(0)
    tax_assets = tax_data["Asset Type"]
    for i in xrange(0, len(tax_assets)):
        tax_assets[i] = str(tax_assets[i]).replace("\xa0", " ").strip()
    # Real Interest Rate:
    r = .05  # NEEDS TO BE AUTOMATED!!!!
    #
    tax_gds_mthds = {"GDS 200%": 2.0, "GDS 150%": 1.5, "GDS SL": 1.0}
    tax_ads_mthds = {"ADS SL": 1.0}
    tax_cols = tax_gds_mthds.keys() + tax_ads_mthds.keys()
    tax_systems = {"GDS": tax_gds_mthds, "ADS": tax_ads_mthds}
    tax_rates = pd.DataFrame(np.zeros((len(tax_assets),len(tax_cols))), 
                             columns = tax_cols)
    tax_rates["Asset"] = tax_assets
    # Compute the tax rates:
    for i in tax_systems:
        tax_yrs = tax_data[i]
        for j in tax_systems[i]:
            tax_b = tax_systems[i][j]
            tax_beta = tax_b/tax_yrs
            tax_star = tax_yrs * (1 - (1/tax_b))
            tax_z = (((tax_beta/(tax_beta+r))*
                      (1-np.exp(-1*(tax_beta+r)*tax_star))) 
                      + ((np.exp(-1*tax_beta*tax_star)/
                      ((tax_yrs-tax_star)*r))*
                      (np.exp(-1*r*tax_star)-np.exp(-1*r*tax_yrs))))
            tax_rates[j] = r/((1/tax_z)-1)
    tax_rates = tax_rates.fillna(0)
    #
    types = ["All", "Corp", "Non-Corp"]
    # Initialize tax depreciation rates tree:
    depr_tree = naics.generate_tree()
    for j in tax_systems:
        for k in tax_systems[j]:
            depr_tree.append_all(df_nm=k, df_cols=types)
    depr_tree.append_all(df_nm="Recommended", df_cols=types)
    # 
    for i in types:
        asset_list = asset_tree.enum_inds[0].data.dfs[i].columns
        asset_list = asset_list.values.tolist()
        match = np.array([-1] * len(asset_list))
        for j in xrange(0, asset_tree.enum_inds[0].data.dfs[i].shape[1]):
            for k in xrange(0, len(tax_assets)):
                if str(asset_list[j]).strip() == str(tax_assets[k]).strip():
                    match[j] = k
        for j in xrange(0, len(depr_tree.enum_inds)):
            cur_ind = depr_tree.enum_inds[j]
            asset_df = asset_tree.enum_inds[j].data.dfs[i]
            #
            tot_assets = sum(asset_tree.enum_inds[j].data.dfs[i].iloc[0,:])
            tot_inv = inv_tree.enum_inds[j].data.dfs["Inventories"][i][0]
            tot_land = land_tree.enum_inds[j].data.dfs["Land"][i][0]
            if(tot_assets+tot_inv+tot_land == 0):
                continue
            ratio = tot_assets / (tot_assets + tot_inv + tot_land)
            #
            for k in tax_cols:
                cur_tax = cur_ind.data.dfs[k][i]
                cur_sum = 0.0
                for l in xrange(0, len(asset_list)):
                    if(match[l] == -1):
                        continue
                    cur_sum += (asset_df.iloc[0,l] * tax_rates[k][match[l]])
                cur_tax[0] = ratio * (cur_sum/sum(asset_df.iloc[0,:]))
            #
            cur_tax = cur_ind.data.dfs["Recommended"][i]
            cur_sum = 0
            for l in xrange(0, len(asset_list)):
                if(match[l] == -1):
                    continue
                cur_rate = tax_rates[tax_data["Method"][match[l]]][match[l]]
                cur_sum += asset_df.iloc[0,l] * cur_rate
            cur_tax[0] = ratio * (cur_sum/sum(asset_df.iloc[0,:]))
    return depr_tree
