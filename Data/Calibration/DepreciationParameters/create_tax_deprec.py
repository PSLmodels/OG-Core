''' Work in Progress
------------------------------------------------------------------------
Last updated 02/02/2015

Create a geometric rate for tax depreciation.

This py-file calls the following other file(s):
            data/detailnonres_stk1.xlsx

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/
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
import excel_toolbox as tb

'''
------------------------------------------------------------------------
------------------------------------------------------------------------
------------------------------------------------------------------------
'''

# Working directory and input file name
path = '/Users/jasondebacker/repos/aei-byu-dynamic-tax-scoring/Data/Calibration/DepreciationParameters/'
file_name = 'IRS_depr_rates_test.csv' ;

# read in the csv file with the tax deprec methods/asset lives
tax_depr_df = pd.read_csv(path + file_name)

# depreciation parameters helpful for calculation
# rate of acceleration over straight-line method
method_map = {'200%': 2.0, '150%': 1.5, 'sl': 1.0}
tax_depr_df['b'] = tax_depr_df.method.map(method_map)

# Y = depreciable life
tax_depr_df['Y']=1*tax_depr_df['GDS'] # who uses ADS?
# Y-star if time to switch to SL if method DB
tax_depr_df['Y_star']= np.where(tax_depr_df['method']=='sl',0,tax_depr_df.Y*(1-(1/tax_depr_df.b)))
tax_depr_df['beta']= tax_depr_df.b/tax_depr_df.Y
r = 0.05 # real interest rate
tax_depr_df['Z']= (((tax_depr_df.beta/(tax_depr_df.beta+r))*
                  (1-np.exp(-1*(tax_depr_df.beta+r)*tax_depr_df.Y_star))) 
                  + ((np.exp(-1*tax_depr_df.beta*tax_depr_df.Y_star)/
                  ((tax_depr_df.Y-tax_depr_df.Y_star)*r))*
                  (np.exp(-1*r*tax_depr_df.Y_star)-np.exp(-1*r*tax_depr_df.Y))))

# geometric tax depreciation rate
tax_depr_df['delta_tau'] = r/((1/tax_depr_df.Z)-1)

# save file to csv
tax_depr_df.to_csv(path+'IRS_depr_rates_out_test.csv', index=False)