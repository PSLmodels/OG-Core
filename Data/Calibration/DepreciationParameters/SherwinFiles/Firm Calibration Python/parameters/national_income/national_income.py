'''
-------------------------------------------------------------------------------
Date created: 5/12/2015
Last updated 5/12/2015
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
import naics_processing as naics
sys.path.append(cur_dir + "\\processing")
import read_income_data as read_inc
'''
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
'''
data_folder = os.path.abspath(cur_dir + "\\data")
naics_codes_file = os.path.abspath(data_folder + "\\NAICS_Codes.csv")
output_folder = os.path.abspath(cur_dir + "\\output")

def get_incs():
    #
    naics_tree = naics.generate_tree()
    #
    read_inc.load_nipa_inc_ind(data_folder, naics_tree)
    read_inc.load_nipa_int_ind(data_folder, naics_tree)
    read_inc.calc_bus_inc(naics_tree)
    #
    parameters = [read_inc.BUS_INC, read_inc.INT_INC, read_inc.FIN_INC]
    #
    naics.pop_back(naics_tree, parameters)
    naics.pop_forward(naics_tree, parameters)
    #
    naics.print_tree_dfs(naics_tree, output_folder)
    return naics_tree

#if __name__ == "national_income":
#    main()



