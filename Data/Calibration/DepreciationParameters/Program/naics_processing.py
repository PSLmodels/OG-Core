'''
-------------------------------------------------------------------------------
Last updated 3/19/2015
-------------------------------------------------------------------------------
This py-file defines a functions that processes a list NAICS codes by creating
    a tree of NAICS industries and ancillary functions.
-------------------------------------------------------------------------------
    Packages
-------------------------------------------------------------------------------
'''
import pandas as pd
import numpy as np
#
import data_class as dc
'''
-------------------------------------------------------------------------------
Functions created:
    load_naics
    find_naics
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
def load_naics: Reads a csv file list of NAICS codes and creates a tree.
    path: the path + /filename 
-------------------------------------------------------------------------------
'''
def load_naics(path):
    # Reading table describing the tree:
    enum = pd.read_csv(path).fillna(0)
    rows = enum.shape[0]
    # Initializing the corresponding naics tree:
    naics = dc.tree()
    naics.enum_codes = enum
    naics.enum_inds = [dc.industry([]) for i in xrange(0,rows)]
    naics.root = naics.enum_inds[0]
    naics.par = [0]*rows
    #
    for i in xrange(0, rows):
        cur_codes = pd.DataFrame(enum.iloc[i,0].split("-"))
        if(cur_codes.shape[0] == 2):
            cur_codes = pd.DataFrame(range(int(cur_codes.iloc[0,0]),int(cur_codes.iloc[1,0])+1))
        naics.enum_inds[i].append_dfs(("Codes:", cur_codes))
        for j in xrange(0, naics.enum_inds[i].data.dfs["Codes:"].shape[0]):
            naics.enum_inds[i].data.dfs["Codes:"].iloc[j,0] = int(naics.enum_inds[i].data.dfs["Codes:"].iloc[j,0])
    # Keeps track of the path from the root to the current industry.
    levels = [None]
    levels[0] = naics.enum_inds[0]
    levels_index = [0]
    cur_lvl = 0
    # Go through every code and find the parent industry
    for i in xrange(1,rows):
        cur_ind = naics.enum_inds[i]
        cur_codes = cur_ind.data.dfs["Codes:"]
        cur_rows = cur_codes.shape[0]
        par_found = False
        while not par_found:
            prev_ind = levels[cur_lvl]
            prev_codes = prev_ind.data.dfs["Codes:"]
            prev_rows = prev_codes.shape[0]
            
            for j in xrange(0, cur_rows):
                for k in xrange(0, prev_rows):
                    
                    if cur_lvl == 0:
                        par_found = True
                        cur_lvl += 1
                        levels.append(cur_ind)
                        levels_index.append(i)
                        levels[0].sub_ind.append(cur_ind)
                        naics.par[i] = levels_index[cur_lvl-1]
                        break
                    elif (str(prev_codes.iloc[k,0]) in str(cur_codes.iloc[j,0])):
                        par_found = True
                        cur_lvl += 1
                        levels.append(cur_ind)
                        levels_index.append(i)
                        prev_ind.sub_ind.append(cur_ind)
                        naics.par[i] = levels_index[cur_lvl-1]
                        break
                if(par_found):
                    break
            if not par_found:
                del levels[cur_lvl]
                del levels_index[cur_lvl]
                cur_lvl -= 1
                
    return naics
    

'''
-------------------------------------------------------------------------------
def find_naics: Defines a function that finds a naics code in a tree.
-------------------------------------------------------------------------------
'''
def find_naics(tree, term):
    for i in tree.enum_inds:
        for j in xrange(0, i.data.dfs["Codes:"].shape[0]):
            if(term == i.data.dfs["Codes:"].iloc[j,0]):
                return i
    return None
    
'''
-------------------------------------------------------------------------------
def find_naics: Defines a function that searches through an excel file for
    a specified term.
-------------------------------------------------------------------------------
'''
def search_ws(sheet, search_term, distance, warnings = True, origin = [0,0], exact = False):
    '''
    Parameters: sheet - The worksheet to be searched through.
                entry - What is being searched for in the worksheet.
                        Numbers must be written with at least one decimal 
                        place, e.g. 15.0, 0.0, 21.74.
                distance - Search up to and through this diagonal.

    Returns:    A vector of the position of the first entry found.
                If not found then [-1,-1] is returned.
    '''
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
        if(exact):
            if str(search_term).lower() == str(sheet.cell_value(i+origin[0],j+origin[1])).lower():
                return [i+origin[0],j+origin[1]]
        elif(not exact):
            if str(search_term).lower() in str(sheet.cell_value(i+origin[0],j+origin[1])).lower():
                return [i+origin[0],j+origin[1]]
    if warnings:
        print "Warning: No such search entry found in the specified search space."
        print "Check sample worksheet and consider changing distance input."
    
    return [-1,-1]

def pop_back(tree, pd_list):
    for corps in pd_list:
        cur_dfs = None
        was_empty = [False]*len(tree.enum_inds)
        count = len(tree.enum_inds)-1
        header = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
        # Working backwards through the tree
        for i in range(1, len(tree.enum_inds)):
            cur_dfs = tree.enum_inds[count].data.dfs[corps]
            par_dfs = tree.enum_inds[tree.par[count]].data.dfs[corps]
            cur_dfs_filled = False
            if sum((cur_dfs != pd.DataFrame(np.zeros((1,len(header))), columns = header)).iloc[0]) == 0:
                cur_dfs_filled = False
            else:
                cur_dfs_filled = True
            if sum((par_dfs != pd.DataFrame(np.zeros((1,len(header))), columns = header)).iloc[0]) == 0:
                was_empty[tree.par[count]] = True
            if cur_dfs_filled and was_empty[tree.par[count]]:
                tree.enum_inds[tree.par[count]].data.dfs[corps] += cur_dfs
            count = count - 1


def pop_forward(tree, pd_list, blueprint = None):
    if blueprint == None:
        for corps in pd_list:
            cur_dfs = None
            header = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
            for i in range(0, len(tree.enum_inds)):
                if tree.enum_inds[i].sub_ind != []:
                    cur_ind = tree.enum_inds[i]
                    cur_dfs = cur_ind.data.dfs[corps]
                    sum_dfs = pd.DataFrame(np.zeros((1,len(header))), columns = header)
                    proportion = 1
                    for j in cur_ind.sub_ind:
                        sum_dfs += j.data.dfs[corps]
                    for j in range(0, len(header)):
                        if sum_dfs.iloc[0,j] == 0:
                            for k in cur_ind.sub_ind:
                                k.data.dfs[corps].iloc[0,j] = cur_dfs.iloc[0,j]/len(cur_ind.sub_ind)
                        else:
                            proportion = cur_dfs.iloc[0,j]/sum_dfs.iloc[0,j]
                            for k in cur_ind.sub_ind:
                                k.data.dfs[corps].iloc[0,j] *= proportion
    else:
        for corps in pd_list:
            cur_dfs = None
            header1 = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
            header2 = tree.enum_inds[0].data.dfs[blueprint].columns.values.tolist()
            for i in range(0, len(tree.enum_inds)):
                if tree.enum_inds[i].sub_ind != []:
                    cur_ind = tree.enum_inds[i]
                    cur_dfs = cur_ind.data.dfs[corps]
                    sum_dfs = pd.DataFrame(np.zeros((1,len(header1))), columns = header1)
                    proportions = np.zeros(len(tree.enum_inds[i].sub_ind))
                    for j in range(0, len(cur_ind.sub_ind)):
                        sum_dfs += cur_ind.sub_ind[j].data.dfs[corps]
                        for k in range(0, len(header2)):
                            proportions[j] += cur_ind.sub_ind[j].data.dfs[blueprint].iloc[0,k]
                    if sum(proportions) != 0:
                        proportions = proportions/sum(proportions)
                    else:
                        for k in range(0, len(proportions)):
                            proportions[k] = 1/len(proportions)
                        
                    for j in range(0, len(cur_ind.sub_ind)):
                        for k in range(0, len(header1)):
                            change = proportions[j] * (cur_dfs.iloc[0,k]-sum_dfs.iloc[0,k])
                            cur_ind.sub_ind[j].data.dfs[corps].iloc[0,k] += change



