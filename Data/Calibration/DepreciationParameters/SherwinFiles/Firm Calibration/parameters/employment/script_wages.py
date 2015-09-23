'''
-------------------------------------------------------------------------------
Date created: 5/22/2015
Last updated 5/22/2015
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
    Packages:
-------------------------------------------------------------------------------
'''
import os.path
import sys
import numpy as np
import pandas as pd
# Find the directory of this file:
cur_dir = os.path.dirname(__file__)
# Import naics processing file:
try:
    import naics_processing as naics
except ImportError:
    data_struct_dir = os.path.dirname(os.path.dirname(cur_dir))
    data_struct_dir += "\\data_structures"
    data_struct_dir = os.path.abspath(data_struct_dir)
    sys.path.append(data_struct_dir)
    try:
        import naics_processing as naics
    except ImportError:
        print "\n\n ImportError: Failed to import naics_processing \n\n"
# Import the helper functions to read in the national income data:


import read_wages_data as read_wages
'''
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
'''
data_folder = os.path.abspath(cur_dir + "\\data")
naics_codes_file = os.path.abspath(data_folder + "\\NAICS_Codes.csv")
output_folder = os.path.abspath(cur_dir + "\\output")

def main():
    #
    naics_tree = naics.load_naics(naics_codes_file)
    #
    read_wages.load_nipa_wages_ind(data_folder, naics_tree)
    #
    parameters = [read_wages.WAGES]
    #
    naics.pop_back(naics_tree, parameters)
    naics.pop_forward(naics_tree, parameters, None, None, None, True)
    #
    naics.print_tree_dfs(naics_tree, output_folder)

if __name__ == "script_wages":
    main()







