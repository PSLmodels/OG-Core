'''
------------------------------------------------------------------------
Last updated 2/6/2015




This py-file calls the following other file(s):
            
            
This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT\
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import pandas as pd
import numpy as np

'''
------------------------------------------------------------------------
    
------------------------------------------------------------------------
    
------------------------------------------------------------------------
'''


'''
This defines an industry object. The object has two instance variables:
    sub_ind:    A list of all sub-industries.
    data:       A pd dataframe of all relevant data on the industry, e.g. name.
'''
class industry:
    
    '''
    Class constructor, the optional inputs initialize the instance variables.
        sub_ind:    default value is the empty list.
        data:       default value is a one element pandas dataframe.
    '''
    def __init__(self, sub_ind, data = pd.DataFrame(np.zeros((1,1)))):
        # This industry can be broken down into the following sub-industries:
        self.sub_ind = sub_ind
        # Relevant industry data:
        self.data = data
        
    def make_industry(self):
        industries = industry()
        return industries
    
    def update_data(self, data):
        self.data = data
    
    '''
    Initializes a tree of industry objects using a csv table of industries as
        input. The column of an industry in the table indicates what level of
        the tree it is in.  If input are strings
    '''
    def load_ind(self, path, data = []):
        # Reading table describing the tree:
        tree = pd.read_csv(path).fillna(0)
        # Getting dimensions of the table
        rows = len(tree.iloc[:,0])
        columns = len(tree.iloc[0,:])        
        # Keeps track of the path from the root to the current industry.
        # The index of the levels variables is how far from the root the
        #   industry is.
        levels = [None]*columns
        
        if(isinstance(data, pd.DataFrame) and isinstance(data.iloc[0,0], str)):
            self.data = pd.DataFrame(np.concatenate((np.array([""]),np.zeros(len(data.iloc[0,:])-1))),data.columns)
        elif(isinstance(data, pd.DataFrame)):
            self.data = pd.DataFrame((np.zeros(len(data.iloc[0,:]))), data.columns)
        else:
            self.data = pd.DataFrame(np.zeros(1))
        # Initializing the tree:
        for i in xrange(0, rows):
            for j in xrange(0, columns):
                # If the (i,j)th element of the table contains an industry:
                if(tree.iloc[i,j] != 0):
                    # Construct an new industry object:
                    new_ind = industry([])
                    new_ind.sub_ind = []
                    if(len(data) == 0):
                        if isinstance(tree.iloc[i,j], str):
                            new_ind.update_data(pd.DataFrame([tree.iloc[i,j].replace('\xa0', ' ').strip()]))                    
                        else:
                            new_ind.update_data(pd.DataFrame([tree.iloc[i,j]]))
                    elif(len(data) != 0):
                        if isinstance(tree.iloc[i,j], str):
                            data.iloc[i,0] = tree.iloc[i,j].replace('\xa0', ' ').strip()
                            new_ind.update_data(data.iloc[i,:])
                        else:
                            data.iloc[i,0] = tree.iloc[i,j]
                            new_ind.update_data(data[i,:])
                    # Update the current path from root:
                    levels[j] = new_ind
                    for k in xrange(j+1, columns):
                        levels[k] = None
                        
                    # Update the parent industry:
                    if(j > 0):
                        levels[j-1].sub_ind.append(new_ind)
                    if(j == 0):
                        self.sub_ind.append(new_ind)
                    # There is only one industry per row:
                    break
    '''
    Finds an industry with a specific name or code in an industry tree.
    '''
    def find(self, term):
        for i in self.sub_ind:
            if(i.data.iloc[0,0] == term):
                return i
            dummy = i.find(term)
            if(dummy != None):
                return dummy
        return None
        
    
    '''
    This populates the data of sub-industries that are the only sub-industry of
        the parent industry. The data files usually do not explicitly have data
        for these sub-industries. However, they may be listed in the files used
        to initialize the tree.
    '''
    def populate_singles(self):
        a = self.sub_ind
        prev_a = [None]
        counts = [0]
        current = 0   
        
        while(counts[0] < len(a)):
            if(current == 0):
                prev_a[current] = a[counts[0]]
                current = 1
                continue
            
            if(current >= len(counts)):
                counts.append(0)
                prev_a.append(None)
            
            if(counts[current] >= len(prev_a[current-1].sub_ind)):
                for i in xrange(current,len(prev_a)):
                    counts[i] = 0
                    prev_a[i] = None
                    
                current -=1
                counts[current] += 1
                continue
            
            prev_a[current] = prev_a[current-1].sub_ind[counts[current]]
            
            if(len(prev_a[current-1].sub_ind) == 1):
                element = prev_a[current].data.iloc[0,0]
                prev_a[current].data = pd.DataFrame(np.zeros((1,len(self.data.columns))), columns = self.data.columns)
                prev_a[current].data.iloc[0,0] = element
                for i in xrange(1, prev_a[current-1].data.shape[1]):
                    prev_a[current].data.iloc[0,i] = prev_a[current-1].data.iloc[0,i]
                    
            current += 1
    
    
    '''
    Purpose: populate one industry tree with incomplete data using a similar
        industry tree with complete data. The latter is used to estimate the
        proportion that sub-categories makeup of their larger categories.
    Inputs:
    This function takes in two industry objects representing industry trees.
        Both must have the same tree structure: sub-industries must be the same
        as well as the names/code of first element of the data pandas DF.
            "self":     Industry tree with data for all the broadest industry
                        categories filled out.
            "other":    Industry tree with data for all industries filled out.
    '''
    def populate_down(self, other):
        a = self.sub_ind
        b = other.sub_ind
        # Do some sort of checks!!
        
        
        prev_a = [None]
        prev_b = [None]
        counts = [0]
        current = 0            
        
        while(counts[0] < len(a)):
            if(current == 0):
                prev_a[current] = a[counts[0]]
                prev_b[current] = b[counts[0]]
                current = 1
                continue
            
            if(current >= len(counts)):
                counts.append(0)
                prev_a.append(None)
                prev_b.append(None)
            
            if(counts[current] >= len(prev_a[current-1].sub_ind)):
                for i in xrange(current,len(prev_a)):
                    counts[i] = 0
                    prev_a[i] = None
                    prev_b[i] = None
                    
                current -=1
                counts[current] += 1
                continue
            
            prev_a[current] = prev_a[current-1].sub_ind[counts[current]]
            prev_b[current] = prev_b[current-1].sub_ind[counts[current]]
            
            
            if(prev_a[current].data.shape[1] == 1):
                element = prev_a[current].data.iloc[0,0]
                prev_a[current].data = pd.DataFrame(np.zeros((1,len(self.data.columns))), columns = self.data.columns)
                prev_a[current].data.iloc[0,0] = element
                for i in xrange(1, prev_b[current-1].data.shape[1]):
                    if(prev_b[current-1].data.iloc[0,i]==0):
                        prev_a[current].data.iloc[0,i] = 0
                        continue
                    prev_a[current].data.iloc[0,i] = prev_a[current-1].data.iloc[0,i] * (prev_b[current].data.iloc[0,i]/prev_b[current-1].data.iloc[0,i])
                    
            current += 1
        
        
    #def populate_up(self):
    
    '''
    Subtracts two industry trees from one another.
    '''
    def subtract(self, other, result):
        a = self.sub_ind
        b = other.sub_ind
        c = result.sub_ind
        # Do some sort of checks!!
        
        prev_a = [None]
        prev_b = [None]
        prev_c = [None]
        counts = [0]
        current = 0            
        
        while(counts[0] < len(a)):
            if(current == 0):
                prev_a[current] = a[counts[0]]
                prev_b[current] = b[counts[0]]
                prev_c[current] = c[counts[0]]
                
                element = prev_a[current].data.iloc[0,0]
                prev_c[current].data = pd.DataFrame(np.zeros((1,len(self.data.columns))), columns = self.data.columns)
                prev_c[current].data.iloc[0,0] = element
                for i in xrange(1, prev_b[current].data.shape[1]):
                    prev_c[current].data.iloc[0,i] = prev_a[current].data.iloc[0,i] - prev_b[current].data.iloc[0,i]
                current = 1
                continue
            
            if(current >= len(counts)):
                counts.append(0)
                prev_a.append(None)
                prev_b.append(None)
                prev_c.append(None)
            
            if(counts[current] >= len(prev_a[current-1].sub_ind)):
                for i in xrange(current,len(prev_a)):
                    counts[i] = 0
                    prev_a[i] = None
                    prev_b[i] = None
                    prev_c[i] = None
                    
                current -=1
                counts[current] += 1
                continue
            
            prev_a[current] = prev_a[current-1].sub_ind[counts[current]]
            prev_b[current] = prev_b[current-1].sub_ind[counts[current]]
            prev_c[current] = prev_c[current-1].sub_ind[counts[current]]
            
            element = prev_a[current].data.iloc[0,0]
            prev_c[current].data = pd.DataFrame(np.zeros((1,len(self.data.columns))), columns = self.data.columns)
            prev_c[current].data.iloc[0,0] = element
            
            for i in xrange(1, prev_b[current-1].data.shape[1]):
                prev_c[current].data.iloc[0,i] = prev_a[current].data.iloc[0,i] - prev_b[current].data.iloc[0,i]
                    
            current += 1
        
        
        
        
        
