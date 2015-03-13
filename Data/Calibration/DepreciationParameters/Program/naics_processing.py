'''
-------------------------------------------------------------------------------
Last updated 3/6/2015
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
def find_naics: Defines a function that finds 
-------------------------------------------------------------------------------
'''
def find_naics(tree, term):
    for i in tree.enum_inds:
        for j in xrange(0, i.data.dfs["Codes:"].shape[0]):
            if(term == i.data.dfs["Codes:"].iloc[j,0]):
                return i
    return None
    





