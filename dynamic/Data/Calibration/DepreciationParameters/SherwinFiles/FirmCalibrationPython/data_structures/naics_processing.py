"""
Processing NAICS Codes (naics_processing.py):
-------------------------------------------------------------------------------
Last updated: 6/24/2015.

This module defines functions that process a *NAICS tree*.
These functions initialize, load data, interpolate data to the various NAICS
levels, print data from trees to csv files, etc.
"""
# Packages:
import os.path
import numpy as np
import pandas as pd
import sys
import xlrd
# Relevant directories:
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.dirname(_CUR_DIR)
_DATA_DIR = os.path.abspath(_MAIN_DIR + "//data")
_NAICS_DIR = os.path.abspath(_DATA_DIR + "//naics")
_CONSTANTS_DIR = os.path.abspath(_MAIN_DIR + "//constants")
_NAICS_CODE_PATH = os.path.abspath(_NAICS_DIR + "//naics_codes.csv")
# Importing custom modules:
sys.path.append(_CONSTANTS_DIR)
import data_class as dc
import constants as cst
# Dataframe names:
_CODE_DF_NM = cst.CODE_DF_NM


def generate_tree():
    """ Using a default listing of the NAICS Codes to create a NAICS tree."""
    return dc.tree(path=_NAICS_CODE_PATH)


def find_naics(tree, code):
    """ Finds an industry with specific code in a NAICS tree.
    
    :param tree: A NAICS tree.
    :param code: The code of the industry being searched for.
    """
    for i in tree.enum_inds:
        for j in xrange(0, i.data.dfs[_CODE_DF_NM].shape[0]):
            if(code == i.data.dfs[_CODE_DF_NM].iloc[j,0]):
                return i
    return None


def pop_back(tree, df_list):
    """ Data is often collected for various levels of NAICS codes. 
    However, it is not explicitly collected for higher or lower levels of
    specificity. This function automatically interpolates the data for the
    less specific NAICS codes using the data for the more specific codes.
    This function "populates" the data "backwards" in the tree.
    
    If data has been entered for an industry and some of its sub-industries,
    then the data for the more general industry will not be altered by
    this function.
    
    :param tree: The NAICS tree.
    :param df_list: A list of the dataframes to be populated backwards.
    """
    for corps in df_list:
        cur_dfs = None
        was_empty = [False]*len(tree.enum_inds)
        count = len(tree.enum_inds)-1
        header = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
        # Working backwards through the tree
        for i in range(1, len(tree.enum_inds)):
            cur_dfs = tree.enum_inds[count].data.dfs[corps]
            par_dfs = tree.enum_inds[tree.par[count]].data.dfs[corps]
            cur_dfs_filled = False
            if sum((cur_dfs != pd.DataFrame(np.zeros((1,len(header))), 
                                            columns = header)).iloc[0]) == 0:
                cur_dfs_filled = False
            else:
                cur_dfs_filled = True
            if sum((par_dfs != pd.DataFrame(np.zeros((1,len(header))), 
                                            columns = header)).iloc[0]) == 0:
                was_empty[tree.par[count]] = True
            if cur_dfs_filled and was_empty[tree.par[count]]:
                tree.enum_inds[tree.par[count]].data.dfs[corps] += cur_dfs
            count = count - 1


def pop_forward(tree, df_list, blueprint = None, blue_tree = None,
                sub_print = None, flat = False):
    """ Data is often collected for various levels of NAICS codes. 
    However, it is not explicitly collected for higher or lower levels of
    specificity. This function automatically estimates the data for the
    more specific NAICS codes using the data for the less specific codes.
    This function "populates" the data "forwards" in the tree.
    
    In order to estimate the more specific data, assumptions have to be made
    about what proportion of the aggregate data goes into each of the more
    specific industry categories. This function allows for these assumptions
    by having "blueprint" inputs.
    
    If the data are say rates or percentages, then the sum of the data for the
    sub-industries need not equal the data for the.  Instead the same rate or
    percentage will be the default value for all of its sub-industries that
    do not already have data entered. The data is then populated forward in
    a "flat" sense.
    
    If no blueprints are entered, then the default is that all of the data is
    allocated in porportion to any data already entered in the sub-industries.
    If there are no blueprints, and no data has been entered in the
    sub-industries, then the data is evenly divided among them.
    
    :param tree: The NAICS tree.
    :param df_list: A list of the dataframes to be populated backwards.
    :param blueprint: The key corresponding to a dataframe in a tree to be
           used as a "blueprint" for populating the df_list dataframes forward.
    :param blue_tree: A NAICS tree with the "blueprint" dataframe. The default
           is the original NAICS tree.
    :param sub_print: A subset of the data columns in blueprint dataframe to be
           used.
    :param flat: Whether the data should populated forward to all the
           sub-industries as is. That is the data is copied to sub-industries,
           not divided up among them.
    """
    if flat:
        for corps in df_list:
            for i in range(0, len(tree.enum_inds)):
                if tree.enum_inds[i].sub_ind != []:
                    cur_ind = tree.enum_inds[i]
                    for sub_ind in cur_ind.sub_ind:
                        if(sum(sub_ind.data.dfs[corps].sum()) == 0):
                            sub_ind.data.dfs[corps] = cur_ind.data.dfs[corps].copy()
        return None
                        
    if blueprint == None:
        for corps in df_list:
            header = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
            for i in range(0, len(tree.enum_inds)):
                if tree.enum_inds[i].sub_ind != []:
                    cur_ind = tree.enum_inds[i]
                    cur_dfs = cur_ind.data.dfs[corps]
                    sum_dfs = pd.DataFrame(np.zeros((1,len(header))), 
                                           columns = header)
                    proportion = 1
                    for j in cur_ind.sub_ind:
                        sum_dfs += j.data.dfs[corps]
                    for j in range(0, len(header)):
                        if sum_dfs.iloc[0,j] == 0:
                            for k in cur_ind.sub_ind:
                                k.data.dfs[corps].iloc[0,j] = cur_dfs.iloc[0,j]
                                k.data.dfs[corps].iloc[0,j] /= len(cur_ind.sub_ind)
                        else:
                            proportion = cur_dfs.iloc[0,j]/sum_dfs.iloc[0,j]
                            for k in cur_ind.sub_ind:
                                k.data.dfs[corps].iloc[0,j] *= proportion
    else:
        if(blue_tree == None):
            blue_tree = tree
        for corps in df_list:
            header1 = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
            if sub_print == None:
                header2 = blue_tree.enum_inds[0].data.dfs[blueprint].columns
                header2 = header2.values.tolist()
            else:
                header2 = sub_print
            for i in range(0, len(tree.enum_inds)):
                if tree.enum_inds[i].sub_ind != []:
                    cur_ind = tree.enum_inds[i]
                    cur_ind_blue = blue_tree.enum_inds[i]
                    cur_dfs = cur_ind.data.dfs[corps]
                    sum_dfs = pd.DataFrame(np.zeros((1,len(header1))), 
                                           columns = header1)
                    proportions = np.zeros(len(tree.enum_inds[i].sub_ind))
                    for j in range(0, len(cur_ind.sub_ind)):
                        sum_dfs += cur_ind.sub_ind[j].data.dfs[corps]
                        for k in header2:
                            blue_dfs = cur_ind_blue.sub_ind[j].data.dfs
                            proportions[j] += blue_dfs[blueprint][k][0]
                    if sum(proportions) != 0:
                        proportions = proportions/sum(proportions)
                    else:
                        for k in range(0, len(proportions)):
                            proportions[k] = 1.0/len(proportions)
                        
                    for j in range(0, len(cur_ind.sub_ind)):
                        for k in range(0, len(header1)):
                            change = proportions[j] * (cur_dfs.iloc[0,k]-
                                                        sum_dfs.iloc[0,k])
                            cur_ind.sub_ind[j].data.dfs[corps].iloc[0,k] += change



def pop_rates(tree):
    """ Copies the dataframes down the tree.
    
    :param tree: The NAICS tree.
    """
    for i in xrange(1, len(tree.enum_inds)):
        par_ind = tree.enum_inds[tree.par[i]]
        cur_ind = tree.enum_inds[i]
        df_names = cur_ind.data.dfs.keys()
        df_names.remove(_CODE_DF_NM)
        for j in df_names:
            if(np.sum(1 - tree.enum_inds[i].data.dfs[j].isnull().iloc[0,:]) == 0):
                cur_ind.data.dfs[j] = par_ind.data.dfs[j].copy()
            elif(np.sum(tree.enum_inds[i].data.dfs[j].iloc[0,:] != 0) == 0):
                cur_ind.data.dfs[j] = par_ind.data.dfs[j].copy()


def get_proportions(codes, tree, df_name, sub_df = None):
    """ Given a list of codes, this function finds all the industries
    containing a code in the list and calculates the relative proportion
    of the data contained in specified dataframe relative to all the other
    similar dataframes in the other industries.
    
    :param codes: An array of codes to find the proportions for.
    :param tree: The NAICS tree.
    :param df_nm: The key corresponding to the dataframe.
    :param sub_df: A subset of the datacolumns in the dataframe.
    """
    proportions = np.zeros(len(codes))
    indexes = np.zeros(len(codes))
    enum_index = len(tree.enum_inds) - 1
    for i in xrange(0, len(codes)):
        cur_code = codes[i]
        for j in xrange(0, len(tree.enum_inds)):
            enum_index = (enum_index+1) % len(tree.enum_inds)
            cur_ind = tree.enum_inds[enum_index]
            if(compare_codes([cur_code], 
                             cur_ind.data.dfs[_CODE_DF_NM].iloc[:,0]) != 0):
                cur_sum = 0
                if sub_df != None:
                    for k in sub_df:
                        cur_sum += cur_ind.data.dfs[df_name][k][0]
                else:
                    cur_sum = sum(cur_ind.data.dfs[df_name].iloc[0,:])
                proportions[i] = cur_sum
                indexes[i] = enum_index
                break
    #
    if(sum(proportions) == 0):
        proportions = np.array([1.0]*len(proportions))/len(proportions)
        return pd.DataFrame(np.vstack((indexes, proportions)))
    #
    proportions = proportions/sum(proportions)
    return pd.DataFrame(np.vstack((indexes, proportions)))

def compare_codes(codes1, codes2):
    if(len(codes2) == 0):
        return 0
    num_match = 0.0
    for i in xrange(0, len(codes2)):
        for j in xrange(0, len(codes1)):
            if(str(codes2[i]) == str(codes1[j])):
                num_match += 1.0
    return float(num_match)/len(codes1)


'''
-------------------------------------------------------------------------------
Prints out the contents of a tree.  Creates a csv file for each dataframe key.
Each line in the csv file has the contents of the df for a specific industry.
-------------------------------------------------------------------------------
'''
def print_tree_dfs(tree, out_path, file_name = None,
                   data_types=None, naics_codes=None):
    """ Prints out the contents of a tree.  Creates a csv file for each
    dataframe key. Each line in the csv file has the contents of the dataframe 
    for a specific industry.
    
    :param tree: The NAICS tree.
    :param out_path: The path of the csv files to be outputted
    :param file_name: The name of the csv file to be outputted. The default is
           the key corresponding to the dataframe
    :param data_types: The keys of the dataframes to be outputted.
    :param naics_codes: Specifying which NAICS codes to be outputted. The
           default is all of the NAICS codes.
    """
    if out_path == None:
        return None
    #
    if data_types == None:
        data_types = tree.enum_inds[0].data.dfs.keys()
        data_types.remove(_CODE_DF_NM)
    #
    for i in data_types:
        cur_cols = tree.enum_inds[0].data.dfs[i].columns.values.tolist()
        cur_cols = [_CODE_DF_NM] + cur_cols
        cur_pd = np.zeros((0,len(cur_cols)))
        cur_data = tree.enum_inds[0].data.dfs[i].iloc[0,:]
        for j in range(0,len(tree.enum_inds)):
            try:
                cur_data = tree.enum_inds[j].data.dfs[i].iloc[0,:]
            except KeyError:
                continue
            if(np.sum((cur_data != np.zeros(len(cur_cols)-1))) == 0):
                continue
            cur_code = str(tree.enum_inds[j].data.dfs[_CODE_DF_NM].iloc[0,0])
            for k in xrange(1, tree.enum_inds[j].data.dfs[_CODE_DF_NM].shape[0]):
                cur_code += "." + str(tree.enum_inds[j].data.dfs[_CODE_DF_NM].iloc[k,0])
            if naics_codes != None:
                if cur_code not in naics_codes:
                    continue
            #cur_code = tree.enum_inds[j].data.dfs[_CODE_DF_NM].iloc[0,0]
            cur_data = np.array([cur_code] + cur_data.tolist())
            
            cur_pd = np.vstack((cur_pd, cur_data))
        cur_pd = pd.DataFrame(cur_pd, columns = cur_cols)
        if file_name == None:
            cur_pd.to_csv(out_path + "\\" + i + ".csv", index = False)
        else:
            cur_pd.to_csv(out_path + "\\" + file_name + ".csv", index = False)


def load_tree_dfs(input_path, dfs_name=None, tree=generate_tree()):
    """ This takes in an input csv file that describes a dataframe to be added
    to each industry in the tree. The header in the input file describes the 
    columns in the dataframe, and each row corresponds to a NAICS industry.
    
    :param input_path: The path of the csv file to be uploaded as dataframe to
           each industry of the tree
    :param dfs_name: The key to use for the dataframe. The default is the
           filename (excluding the '.csv').
    :param tree: The NAICS tree.
    """
    #
    if dfs_name == None:
        dfs_name = os.path.basename(input_path)[:-4]
    #
    input_df = pd.read_csv(input_path)
    #
    data_types = input_df.columns.tolist()
    data_types.remove(_CODE_DF_NM)
    #
    for ind in tree.enum_inds:
        ind.append_dfs((dfs_name, pd.DataFrame(np.zeros((1,len(data_types))), 
                                               columns = data_types)))
    #
    enum_index = len(tree.enum_inds) - 1
    for i in xrange(0, input_df.shape[0]):
        ind_code = str(input_df[_CODE_DF_NM][i])
        #print ind_code
        for j in xrange(0, len(tree.enum_inds)):
            enum_index = (enum_index+1) % len(tree.enum_inds)
            cur_ind = tree.enum_inds[enum_index]
            cur_code = str(cur_ind.data.dfs[_CODE_DF_NM].iloc[0,0])
            for k in xrange(1, cur_ind.data.dfs[_CODE_DF_NM].shape[0]):
                cur_code += "." + str(cur_ind.data.dfs[_CODE_DF_NM].iloc[k,0])
            if cur_code == ind_code:
                cur_ind.data.dfs[dfs_name].iloc[0,:] = input_df.iloc[i,1:]
                break
    return tree


def load_data_with_cross(data_df, cross_df,
                         data_tree=generate_tree(),
                         df_nm="", df_cols=None, 
                         bluetree=None, blueprint=None):
    """ Given a a dataset and a mapping of rows in the data set to NAICS codes,
    the function reads all the data into the tree.
    
    :param data_df: The dataframe containing all the data to be uploaded to
           the tree.
    :param cross_df: A crosswalk  between the rows of the dataset and the
           NAICS codes.
    :param data_tree: The NAICS tree.
    :param df_nm: Specifying the key for the new dataframes.
    :param df_cols: The data in each of the columns if different from what is
           specified in the input file.
    :param bluetree: Additional functionality to be added.
    :param blueprint: Additional functionality to be added.
    """
    # The default value of the columns in the dataframe are what is specified
    # in the data set.
    if df_cols == None:
        df_cols = data_df.columns.values.tolist()
    # Initialize dataframes for each in industry in the tree:
    data_tree.append_all(df_nm=df_nm, df_cols=df_cols)
    # Add data to industries in the tree based on fraction of codes shared:
    enum_index = 0
    for i in xrange(0, len(cross_df[_CODE_DF_NM])):
        if pd.isnull(cross_df[_CODE_DF_NM][i]):
            continue
        #
        cur_codes = str(cross_df[_CODE_DF_NM][i]).split(".")
        tot_prop = 0.0
        for j in xrange(0, len(data_tree.enum_inds)):
            cur_ind = data_tree.enum_inds[enum_index]
            cur_prop = compare_codes(cur_codes, cur_ind.data.dfs[_CODE_DF_NM][0])
            if cur_prop == 0:
                enum_index = (enum_index+1) % len(data_tree.enum_inds)
                continue
            cur_ind.data.dfs[df_nm] += cur_prop * np.array(data_df.iloc[i,:])
            #
            enum_index = (enum_index+1) % len(data_tree.enum_inds)
            tot_prop += cur_prop
            if(tot_prop >= 1):
                break
    return data_tree





def search_ws(sheet, search_term, distance=20, warnings=True, origin=[0,0], 
              exact = False):
    """ Searches through an excel sheet for a specified term.
    The function searches along the bottom left to top right diagonals.
    The function starts at the "origin" and only looks for values below or to
    the right of it up to the number of diagonals away specified by the
    "distance."
        
    :param sheet: The worksheet to be searched.
    :param search_term: The term to be searched for in the worksheet.
    :param distance: The number of diagonals to look through until stopping.
    :param warnings: Whether to print warning if the search_term wasn't found.
    :param origin: Where to start in the excel sheet.
    :param exact: Whether the cell must match the search term exactly.
    """
    final_search = ((distance+1)*distance)/2
    current_diagonal = 1
    total_columns  = sheet.ncols
    total_rows  = sheet.nrows
    for n in xrange(0, final_search):
        if ((current_diagonal+1)*current_diagonal)/2 < n+1:
            current_diagonal += 1
        
        i = ((current_diagonal+1)*current_diagonal)/2 - (n+1)
        j = current_diagonal - i - 1
        
        if j + origin[1] >= total_columns:
            continue
        if i + origin[0] >= total_rows:
            continue
        cur_cell = str(unicode(sheet.cell_value(i+origin[0],j+origin[1])).encode('utf8')).lower()
        if(exact):
            if str(search_term).lower() == cur_cell:
                return [i+origin[0],j+origin[1]]
        elif(not exact):
            if str(search_term).lower() in cur_cell:
                return [i+origin[0],j+origin[1]]
    # Failed to find search term:
    if warnings:
        print "Warning: Search entry not found in the specified search space."
        print "Check sample worksheet and consider changing distance input."
    return [-1,-1]


"""
def find_first_match(tree, codes):
    for i in tree.enum_inds:
        ind_codes = i.data.dfs[_CODE_DF_NM]
        for j in xrange(0, len(codes)):
            for k in xrange(0, ind_codes.shape[0]):
                if(str(codes[j]) == str(ind_codes.iloc[k,0])):
                    return i
    return None
"""


"""
def find_matches(tree, codes):
    matches = []
    for i in tree.enum_inds:
        ind_codes = i.data.dfs[_CODE_DF_NM]
        is_match = False
        for j in xrange(0, len(codes)):
            for k in xrange(0, ind_codes.shape[0]):
                if(str(codes[j]) == str(ind_codes.iloc[k,0])):
                    matches.append[i]
                    is_match = True
                    break
            if(is_match):
                break
"""


"""
def load_specific_codes(input_path, naics_codes):
    tree = load_tree_dfs(input_path)
    
    out_df = pd.DataFrame(np.zeros((len(naics_codes), len)))
    for cur_ind in tree.enum_inds:
"""


"""
def load_data_with_cross(data_df, cross_df, data_tree = generate_tree(),
                         data_cols = None, df_name = ""):
    #
    if data_tree == None:
        data_tree = generate_tree()
    if data_cols == None:
        data_cols = data_df.columns.values.tolist()
    #
    enum_index = 0
    # Append empty dataframes to the tree:
    for i in data_tree.enum_inds:
        i.append_dfs((df_name, pd.DataFrame(np.zeros((1,len(data_cols))),
                                             columns = data_cols)))
    # Add data to industries in the tree based on fraction of codes shared:
    for i in xrange(0, len(cross_df[_CODE_DF_NM])):
        if pd.isnull(cross_df[_CODE_DF_NM][i]):
            continue
        cur_codes = str(cross_df[_CODE_DF_NM][i]).split(".")
        num_found = 0
        for k in xrange(0, len(cur_codes)):
            cur_codes[k] = int(cur_codes[k])
        for j in xrange(0, len(data_tree.enum_inds)):
            cur_ind = data_tree.enum_inds[enum_index]
            for k in cur_codes:
                for l in cur_ind.data.dfs[_CODE_DF_NM][0]:
                    if(k == l):
                        cur_ind.data.dfs[df_name] += np.array(
                                        data_df.iloc[i,:])/len(cur_codes
                                        )
                        num_found += 1
            enum_index = (enum_index+1) % len(data_tree.enum_inds)
            if(num_found == len(cur_codes)):
                        break
    return data_tree
"""

