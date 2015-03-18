'''
-------------------------------------------------------------------------------
Last updated 3/6/2015
-------------------------------------------------------------------------------
This py-file defines objects that will be used to keep track of all the data
    pertinent to depreciation rates. Specifically, these objects will address
    the problem of breaking down the data by industry.
-------------------------------------------------------------------------------
    Packages
-------------------------------------------------------------------------------
'''

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

-------------------------------------------------------------------------------
class pd_dfs: This Defines an object that contains a list of pandas dataframes.
    dfs:    A dictionary of panda dataframes.
    n:      The number of pandas dataframes in the list.
-------------------------------------------------------------------------------
'''
class pd_dfs:
    '''
    Constructor function.
    Input: A sequence of pandas dataframes.
    '''
    def __init__(self, *args):
        #
        #while len(args) == 1 and isinstance(args[0],tuple) and isinstance(args[0][0],tuple):
        #    args = args[0]
        '''
        #Handling invalid inputs:
        for i in xrange(0, len(args)):
            if isinstance(args[i], pd.DataFrame):
                self.n += 1
            else:
                print "Invalid input"
        '''
        self.dfs = {}
        self.append(args)
    
    def append(self, *args):
        #print args
        while len(args) == 1 and isinstance(args[0], (list,tuple)):
            args = args[0]
        #print args
        for i in xrange(len(args)):
            if isinstance(args[i], (list,tuple)):
                self.dfs[args[i][0]] = args[i][1]
            else:
                if i%2 == 0:
                    self.dfs[args[i]] = args[i+1]
    
    def delete(self, index=None):
        del self.dfs[index]
    

'''
-------------------------------------------------------------------------------
class industry: This defines an object that represents an industry.
    sub_ind:    A list of all sub-industries.
    data:       A list of pd dataframes of relevant data on the industry.
-------------------------------------------------------------------------------
'''
class industry:
    '''
    Class constructor, the optional inputs initialize the instance variables.
        sub_ind:    default value is the empty list.
        args:       List of pd dataframes.
    '''
    def __init__(self, sub_ind, *args):
        # This industry can be broken down into the following sub-industries:
        self.sub_ind = sub_ind
        # Relevant industry data:
        self.data = pd_dfs(args)
    
    def replace_dfs(self, *args):
        self.data = pd_dfs(args)
    
    def append_dfs(self, *args):
        self.data.append(args)
    
    def delete_df(self, index):
        self.data.delete(index)
        
    

'''
-------------------------------------------------------------------------------
class tree: Defines an object that represents a tree of all the industries. 
        Each node of the tree is an industry object, and its children are its
        sub-industries. The tree has a root and a list of all the industries.
    root: An industry object that aggregates over all the industries.
    enum_inds: An enumeration of all the industries.
-------------------------------------------------------------------------------
'''
class tree:
    #
    def __init__(self, root = None, enumeration = pd.DataFrame(np.zeros((0,0)))):
        self.root = root
        self.enum_codes = enumeration
        self.enum_inds = [industry([]) for i in xrange(0,len(enumeration))]
        self.par = None
        
    def append_all(self, *args):
        for i in self.enum_inds:
            i.append_dfs(args)
            
            

