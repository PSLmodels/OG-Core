'''
------------------------------------------------------------------------
Last updated 1/30/2015




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

import os.path
import numpy as np
import pandas as pd

'''
------------------------------------------------------------------------
    
------------------------------------------------------------------------
    
------------------------------------------------------------------------
'''

#a = industry("N:\Lott, Sherwin\OLG Dynamic Scoring Model\Calibration\Draft\Program\data\Industry_List.csv",True)

# Defines an industry object.  It contains a list of sub_industries as well as
#   a pandas df of relevant data.
class industry:
    # Class constructor
    def __init__(self, sub_ind = []):
        # This industry can be broken down into the following sub-industries:
        self.sub_ind = sub_ind
        # Pandas df of relevant data:
        self.data = [None]*10
    
    # Given a csv file, initializes an 
    def load_ind(self, path):
        dummy = pd.read_csv(path).fillna(0)
        rows = len(dummy.iloc[:,0])
        columns = len(dummy.iloc[0,:])
#        counter = [0]*columns
        levels = [None]*columns
        self.sub_ind = [] #[None]*len(dummy.iloc[:,0].value_counts())
        for i in xrange(0, rows):
            for j in xrange(0, columns):
                #print(str(i) + ", " + str(j))
                if(dummy.iloc[i,j] != 0):
#                    print(str(i) + ", " + str(j))
                    levels[j] = industry()
                    levels[j].sub_ind = []
                    #print dummy.iloc[i,j]
                    levels[j].data[0] = dummy.iloc[i,j].replace('\xa0', ' ').strip()
#                    print levels[max(j-1,0)].data[0]
                    if(j > 0):
                        #print levels[j-1].data[0]
                        #print levels[j].data[0]
                        levels[j-1].sub_ind.append(levels[j])
                    if(j == 0):
                        self.sub_ind.append(levels[j])
                    for k in xrange(j+1, columns):
#                        counter[k] = 0
                        levels[k] = None
                    #print levels
                    break
#                        counter[j] += 1

            
        

