'''
Data Structures (data_class.py):
-------------------------------------------------------------------------------
Last updated 6/24/2015

This module defines data structures in order to keep track of firm data that
is categorized by NAICS codes.
Dealing with this data is made particularly difficult by the levels of
detail that firms can be differentiated on.
Different data sources consequently have varying levels of specificity.
In order to deal with this, the module creates a *NAICS tree* data structure.

The :term:`NAICS tree` is a standard tree data structure with each node
corresponding to an NAICS industry.
The nodes are coded in as custom "industry" objects.
The industry object has a list of sub-industries and a custom pandas dataframes
object. The pandas dataframes object has a dictionary of pandas dataframes as
well as custom functions for maintaining it.
'''
# Packages:
import pandas as pd
import numpy as np
'''
-------------------------------------------------------------------------------
Objects defined here:
    pd_dfs (pandas dataframes): A dictionary of pandas dataframes.
    industry: A list of sub-industries, as well as a pd_dfs for pertinent data.
    tree: A tree of industry objects.  Has a root that aggregates all the
        industries, as well as a list of all industries in the tree.
-------------------------------------------------------------------------------
class pd_dfs: This Defines an object that contains a list of pandas dataframes.
    dfs:    A dictionary of panda dataframes.
    n:      The number of pandas dataframes in the list.
-------------------------------------------------------------------------------
'''
class pd_dfs:
    """ 
    This "pandas dataframes" object has one member: a dictionary of pandas
    dataframes. The class has functions for reading in and maintaining this.
    
    :param args: Data to initialize the dictionary with. This is either a
           dictionary of pandas dataframes, or tuple/list of keys
           alternated with pandas dataframes.
    """
    def __init__(self, *args):
        # Initialize the dictionary:
        self.dfs = {}
        self.append(args)
    
    def append(self, *args):
        """ Appending to the dictionary of pandas dataframe.
        
        :param args: Data to be appendend. This is either a dictionary of
               pandas dataframes, or tuple/list of keys alternated with
               pandas dataframes.
        """
        # *args may be nested in tuples as it goes through multiple functions:
        while len(args) == 1 and isinstance(args[0], (list,tuple)):
            args = args[0]
        # If the input is a dictionary:
        if len(args) > 0 and isinstance(args[0], dict):
            for key in args[0]:
                self.dfs[key] = args[0][key]
            return None
        # If the input is a list or tuple alternating between keys and pd_dfs:
        for i in xrange(len(args)):
            if isinstance(args[i], (list,tuple)):
                self.dfs[args[i][0]] = args[i][1]
            else:
                if i%2 == 0:
                    self.dfs[args[i]] = args[i+1]
    
    def delete(self, keys=None):
        """ Deleting elements in dictionary of pandas dataframe.
        
        :param keys: A list of keys to be deleted."""
        for key in keys:
            try:
                del self.dfs[key]
            except KeyError:
                pass


class industry:
    '''
    This object represents an industry. It has a list of the NAICS codes of
    the sub-industries as well as a pandas dataframes object.
    
    :param sub_ind: A list of sub-industries of this industry.
    :param args: Data to initialize the industry with. This is either a
           dictionary of pandas dataframes, or tuple/list of keys
           alternated with pandas dataframes.
    '''
    def __init__(self, sub_ind, *args):
        self.sub_ind = sub_ind
        # Initialize the data:
        self.data = pd_dfs(args)
    
    def append_dfs(self, *args):
        ''' Append data.
        
        :param args: Data to append the industry with. This is either a
               dictionary of pandas dataframes, or tuple/list of keys
               alternated with pandas dataframes.
        '''
        self.data.append(args)
    
    def delete_df(self, keys):
        ''' Delete data.
        
        :param args: Keys corresponding to the dataframes to be deleted.
        '''
        self.data.delete(keys)


class tree:
    """
    Defines a tree where each node is an industry. The tree has a root,
    a list of all the industries, and a matching from each index of an industry
    to the index of the corresponding parent.
    
    :param path: The path of a csv file that has one column of NAICS codes.
    
    .. note:: In the input csv file, industries with multiple NAICS codes
       **must** separate the codes using periods (".").
       Anything besides digits and periods will make the function crash.
    
    .. note:: The input csv file must have "Codes:" as a header on the
       first row of the first column.
           
    :param root: An industry object corresponding to the aggregate of all the
           industries. This should have a NAICS code of '1'.
    :param enumeration: An enumeration of all the industries.
    :param par: A matching from each index of an industry to the index of the
           corresponding parent.
    """
    def __init__(self, path="", root=None, 
                 enum_inds=None, par=None):
        if path != "":
            self = self.load_naics(path)
        else:
            self.root = root
            if enum_inds == None:
                enum_inds = pd.DataFrame(np.zeros((0,0)))
            self.enum_inds = [industry([]) for i in xrange(0,len(enum_inds))]
            self.par = par    
    
    
    def append_all(self, df_nm, df_cols):
        ''' Appends pandas dataframe to all industries in a tree.
        This dataframe has dimensions 1xlen(df_cols), and corresponds to key
        df_nm in the dataframes dictionary.
        
        :param root: An industry object that aggregates over all the industries.
        :param enumeration: An enumeration of all the industries.
        '''
        for i in self.enum_inds:
            i.data.append((df_nm, pd.DataFrame(np.zeros((1, len(df_cols))),
                                                 columns=df_cols)))
    
    
    def load_naics(self, path):
        '''
        This function takes a csv file that is a column of NAICS codes and
        generates a *NAICS tree*.
        
        :param path: The path of a csv file that has one column of NAICS codes.
        
        .. note:: In the input csv file, industries with multiple NAICS codes
           **must** separate the codes using periods (".") in the csv file.
           Anything besides digits and periods will make the function crash.
        
        .. note:: The input csv file must have "Codes:" as a header on the
           first row of the first column.
        '''
        # Reading in a list of naics codes:
        naics_codes = pd.read_csv(path).fillna(0)
        rows = naics_codes.shape[0]
        # Initializing the naics tree:
        self.enum_inds = [industry([]) for i in xrange(0,rows)]
        self.root = self.enum_inds[0]
        self.par = [0]*rows
        # Read the naics codes into the tree:
        for i in xrange(0, rows):
            cur_codes = pd.DataFrame(naics_codes.iloc[i,0].split("-"))
            if(cur_codes.shape[0] == 2):
                cur_codes = pd.DataFrame(range(int(cur_codes.iloc[0,0]),
                                               int(cur_codes.iloc[1,0])+1))
            self.enum_inds[i].append_dfs(("Codes:", cur_codes))
            cur_rows = self.enum_inds[i].data.dfs["Codes:"].shape[0]
            for j in xrange(0, cur_rows):
                code = int(self.enum_inds[i].data.dfs["Codes:"].iloc[j,0])
                self.enum_inds[i].data.dfs["Codes:"].iloc[j,0] = code
        # Creating the tree structure:
        # "levels" keeps track of the path from the root to the current industry.
        levels = [None]
        levels[0] = self.enum_inds[0]
        levels_index = [0]
        cur_lvl = 0
        # Going through every industry in the tree and finding the parent/children:
        for i in xrange(1,rows):
            cur_ind = self.enum_inds[i]
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
                            # Then the industry's parent is the root.
                            par_found = True
                            cur_lvl += 1
                            levels.append(cur_ind)
                            levels_index.append(i)
                            levels[0].sub_ind.append(cur_ind)
                            self.par[i] = levels_index[cur_lvl-1]
                            break
                        elif str(prev_codes.iloc[k,0]) in str(cur_codes.iloc[j,0]):
                            # Then "levels[cur_lvl]" is the parent of "cur_ind":
                            par_found = True
                            cur_lvl += 1
                            levels.append(cur_ind)
                            levels_index.append(i)
                            prev_ind.sub_ind.append(cur_ind)
                            self.par[i] = levels_index[cur_lvl-1]
                            break
                    if(par_found):
                        break
                if not par_found:
                    del levels[cur_lvl]
                    del levels_index[cur_lvl]
                    cur_lvl -= 1
        return self
            

