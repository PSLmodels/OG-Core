'''
-------------------------------------------------------------------------------
Date created: 4/15/2015
Last updated 4/15/2015
-------------------------------------------------------------------------------
This py-file tests the naics_processing.py program.
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
-------------------------------------------------------------------------------
Testing the "load_naics" function.
Checks:
    1)  Recreate the list of naics codes using the tree. Check to see that this
        matches the input.
    2)  
-------------------------------------------------------------------------------
'''
def test_load_naics(path = None, messages = True):
    # Default path if none is specified:
    if path == None:
        path = os.getcwd()
        path = os.path.abspath(path + "\\data\\2012_NAICS_Codes.csv")
    # Using the function being tested to create a tree:
    cur_tree = naics.load_naics(path)
    # Replicating the codes in the input file:
    rep_codes = np.zeros(0)
    for ind in cur_tree.enum_inds:
        cur_codes = ind.data.dfs["Codes:"].iloc[:,0]
        rep_codes = np.append(rep_codes, cur_codes)
    rep_codes = rep_codes.astype(int)
    rep_codes = np.unique(rep_codes)
    rep_codes = np.sort(rep_codes)
    #
    orig_data = pd.read_csv(path).iloc[:,0]
    orig_codes = np.zeros(0)
    for i in xrange(0, len(orig_data)):
        cur_codes = str(orig_data[i]).split("-")
        orig_codes = np.append(orig_codes, cur_codes)
    orig_codes = orig_codes.astype(int)
    orig_codes = np.unique(orig_codes)
    orig_codes = np.sort(orig_codes)
    #
    rep_index = 0
    orig_index = 0
    matches = 0
    while((rep_index < len(rep_codes)) and (orig_index < len(orig_codes))):
        if(rep_codes[rep_index] == int(orig_codes[orig_index])):
            rep_index += 1
            orig_index += 1
            matches += 1
        elif(rep_codes[rep_index] <= orig_codes[orig_index]):
            rep_index += 1
        elif(rep_codes[rep_index] >= orig_codes[orig_index]):
            orig_index += 1
    if matches == len(orig_codes):
        if messages:
            print "\"load_naics\" passed test 1."
        return None
    else:
        mismatch = str(len(orig_codes) - matches)
        if messages:
            print "\"load_naics\" failed test 1. Mismatches:" + mismatch + "."
        return int(mismatch)


'''
-------------------------------------------------------------------------------
Prints out the contents of a tree.  Creates a csv file for each dataframe key.
Each line in the csv file has the contents of the df for a specific industry.
This allows the data to be manually checked in excel.
-------------------------------------------------------------------------------
'''
def print_tree_dfs(data_tree, out_path = None, data_types = None):
    if out_path == None:
        out_path = os.getcwd()
        out_path = os.path.abspath(out_path + "\\OUTPUT\\tests\\tree_data")
    #
    if data_types == None:
        data_types = data_tree.enum_inds[0].data.dfs.keys()
        data_types.remove("Codes:")
    #
    for i in data_types:
        cur_cols = data_tree.enum_inds[0].data.dfs[i].columns.values.tolist()
        cur_cols = ["Codes:"] + cur_cols
        cur_pd = np.zeros((0,len(cur_cols)))
        for j in xrange(0,len(data_tree.enum_inds)):
            cur_data = data_tree.enum_inds[j].data.dfs[i].iloc[0,:]
            if(np.sum((cur_data != np.zeros(len(cur_cols)-1))) == 0):
                continue
            cur_code = data_tree.enum_inds[j].data.dfs["Codes:"].iloc[0,0]
            cur_data = np.array([cur_code] + cur_data.tolist())
            
            cur_pd = np.vstack((cur_pd, cur_data))
        cur_pd = pd.DataFrame(cur_pd, columns = cur_cols)
        cur_pd.to_csv(out_path + "\\" + i + ".csv")
        
    


#'''
#-------------------------------------------------------------------------------
#Testing the "load_soi_corporate_data" function.
#Checks:
#    1)  
#    2)  
#-------------------------------------------------------------------------------
#'''
#def test_load_soi_corporate_data(data_tree, loaded = False, path = None, out_path = None):
#    # Default path if none is specified:
#    if path == None:
#        path = os.getcwd()
#        path = os.path.abspath(path + "\\data")
#    #
#    if out_path == None:
#        out_path = os.getcwd()
#        out_path = os.path.abspath(out_path + "\\OUTPUT\\tests")
#    #
#    if(not loaded):
#        naics.load_soi_corporate_data(data_tree, path)
#    #
#    corp_types = ["tot_corps", "s_corps", "c_corps"]
#    #
#    for i in corp_types:
#        cur_cols = data_tree.enum_inds[0].data.dfs[i].columns.values.tolist()
#        cur_cols = ["Codes:"] + cur_cols
#        cur_pd = np.zeros((0,len(cur_cols)))
#        for j in xrange(0,len(data_tree.enum_inds)):
#            cur_data = data_tree.enum_inds[j].data.dfs[i].iloc[0,:]
#            if(np.sum((cur_data != np.zeros(len(cur_cols)-1))) == 0):
#                continue
#            cur_code = data_tree.enum_inds[j].data.dfs["Codes:"].iloc[0,0]
#            cur_data = np.array([cur_code] + cur_data.tolist())
#            
#            cur_pd = np.vstack((cur_pd, cur_data))
#        cur_pd = pd.DataFrame(cur_pd, columns = cur_cols)
#        cur_pd.to_csv(out_path + "\\" + i + ".csv")
#    
#'''
#-------------------------------------------------------------------------------
#Testing the "load_soi_corporate_data" function.
#Checks:
#    1)  
#    2)  
#-------------------------------------------------------------------------------
#'''
#def test_load_soi_partner_data(data_tree, loaded = False, path = None, out_path = None):
#    # Default path if none is specified:
#    if path == None:
#        path = os.getcwd()
#        path = os.path.abspath(path + "\\data")
#    #
#    if out_path == None:
#        out_path = os.getcwd()
#        out_path = os.path.abspath(out_path + "\\OUTPUT\\tests")
#    #
#    if(not loaded):
#        naics.load_soi_partner_data(data_tree, path)
#    #
#    #corp_types = ["tot_corps", "s_corps", "c_corps"]
#    asset_types = ['PA_inc_loss', 'PA_assets', 'PA_types']
#    #
#    for i in asset_types:
#        cur_cols = data_tree.enum_inds[0].data.dfs[i].columns.values.tolist()
#        cur_cols = ["Codes:"] + cur_cols
#        cur_pd = np.zeros((0,len(cur_cols)))
#        for j in xrange(0,len(data_tree.enum_inds)):
#            cur_data = data_tree.enum_inds[j].data.dfs[i].iloc[0,:]
#            if(np.sum((cur_data != np.zeros(len(cur_cols)-1))) == 0):
#                continue
#            cur_code = data_tree.enum_inds[j].data.dfs["Codes:"].iloc[0,0]
#            cur_data = np.array([cur_code] + cur_data.tolist())
#            
#            cur_pd = np.vstack((cur_pd, cur_data))
#        cur_pd = pd.DataFrame(cur_pd, columns = cur_cols)
#        print cur_pd
#        cur_pd.to_csv(os.path.abspath(out_path + "\\" + i + ".csv"))


