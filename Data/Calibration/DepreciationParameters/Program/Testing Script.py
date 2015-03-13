import pandas as pd
import numpy as np
#
import data_class as dc

a1 = pd.DataFrame(np.zeros((1,1)))
a2 = pd.DataFrame(np.zeros((2,2)))
a3 = pd.DataFrame(np.zeros((3,3)))
a4 = pd.DataFrame(np.zeros((4,4)))
a5 = 5
a6 = pd.DataFrame(np.zeros((6,6)))

a = dc.pd_dfs(a1,a2,a3,a4,a5,a6)

'''
------------------------------------------------------------------------
    
------------------------------------------------------------------------
    
------------------------------------------------------------------------
'''

a = dc.industry([],a1,a2,a3,a4,a5,a6)

b = dc.tree(a)


path = "N:\Lott, Sherwin\OLG Dynamic Scoring Model\Calibration\Draft 2\data\NAICS Codes\Listing of all 2012 NAICS Codes.csv"




