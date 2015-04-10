'''
-------------------------------------------------------------------------------
Last updated 3/24/2015
-------------------------------------------------------------------------------
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
-------------------------------------------------------------------------------
    Packages:
-------------------------------------------------------------------------------
'''
import os.path
import numpy as np
import pandas as pd
# 
import naics_processing as naics

'''
-------------------------------------------------------------------------------
The main script of the program:
    --Loading the SOI Tax Stats-Corporation Data.
    --Loading the SOI Tax Stats-Partnership Data.
    --Loading tax data for Proprietorships.
    --Creating "output_tree" stating FA, INV, and LAND for various sectors.
-------------------------------------------------------------------------------
'''
# Working directory:
path = os.getcwd()
# Relevant path and file names:
data_folder = os.path.abspath(path + "\\data")
output_folder = os.path.abspath(path + "\\OUTPUT")

# Create a tree based off NAICS Codes:
data_tree = naics.load_naics(data_folder + "\\2012_NAICS_Codes.csv")
# Reading in the SOI Tax Stats-Corporation Data:
naics.load_soi_corporate_data(data_tree, data_folder)
# Reading in the SOI Tax Stats-Partnership Data:
naics.load_soi_partner_data(data_tree, data_folder)
# Reading in the SOI Tax Stats-Proprietorship Data:
naics.load_soi_proprietor_data(data_tree, data_folder)

'''
Many industries are not listed in the SOI datasets. The data for these missing
    industries are interpolated.
'''
# Get a list of the names of all the pd dfs besides the list of codes:
cur_names = data_tree.enum_inds[0].data.dfs.keys()
cur_names.remove("Codes:")
# Populate missing industry data backwards throught the tree:
naics.pop_back(data_tree, cur_names)
# Populate the missing total corporate data forwards through the tree:
naics.pop_forward(data_tree, ["tot_corps"])
# Populate other missing data using tot_corps as a "blueprint":
cur_names = ["c_corps", "s_corps", "PA_inc/loss", "PA_assets", "soi_prop"]
naics.pop_forward(data_tree, cur_names, "tot_corps")
# Populate pa05 using pa01:
naics.pop_forward(data_tree, ["PA_types"], "PA_inc/loss")
#
naics.pop_back(data_tree, ["farm_prop"])
naics.pop_forward(data_tree, ["farm_prop"], "tot_corps")

#Create an output tree containing only the final data on FA, INV, and LAND.
output_tree = naics.summary_tree(data_tree, data_folder)

# Create a tree with all the FA's broken down by type of asset:
asset_tree = naics.read_bea(output_tree, data_folder)
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
inv_tree = naics.read_inventories(output_tree, data_folder)
naics.pop_back(inv_tree, ["Inventories"])
naics.pop_forward(inv_tree, ["Inventories"])
#
land_tree = naics.read_land(output_tree, data_folder)
naics.pop_forward(land_tree, ["Land"], "LAND", output_tree)
#
econ_depr_tree = naics.calc_depr_rates(asset_tree, inv_tree, land_tree, data_folder)
tax_depr_tree = naics.calc_tax_depr_rates(asset_tree, inv_tree, land_tree, data_folder)







