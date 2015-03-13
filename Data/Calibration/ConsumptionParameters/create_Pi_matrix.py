''' 
------------------------------------------------------------------------
Last updated 03/12/2015

Read in BEA's 2007 PCE Bridge data, use these data to
create matrix relating production industry output to 
consumption good categories.  This matrix is called 
'Pi'.

This py-file calls the following other file(s):
            PCEBridge_2007_Detail.xlsx
            nipa_cons_category_crosswalk.xlsx
            prod_sector_crosswalk.xlsx

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Pi_mat.pkl
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
import xlrd


'''
------------------------------------------------------------------------
------------------------------------------------------------------------
------------------------------------------------------------------------
'''

# Working directory and input file name
path = '/Users/jasondebacker/repos/dynamic/Data/Calibration/ConsumptionParameters/'
pce_bridge_name = 'PCEBridge_2007_Detail.xlsx' 
cons_xwalk_name = 'nipa_cons_category_crosswalk.xlsx' 
prod_xwalk_name = 'prod_industry_crosswalk.xlsx' 
sheet_name = '2007'

# read in the BEA's PCE Bridge File
pce_bridge_df = pd.read_excel(path + pce_bridge_name,
                              sheetname=sheet_name,header=5)

# read in the NIPA consumption crosswalk
cons_cat_df = pd.read_excel(path + cons_xwalk_name,header=0)
# read in the BEA/NAICS production industry crosswalk
prod_ind_df = pd.read_excel(path + prod_xwalk_name,header=0)


# rename columns (since problem reading headers with apostrophes)
pce_bridge_df.rename(columns={'Unnamed: 4': 'Producer Value', 
                              'Unnamed: 5': 'Transportation Value',
                              'Unnamed: 8': 'Purchasers Value'}, inplace=True)
# rename to shorter variable names 
pce_bridge_df.rename(columns={'NIPA Line': 'nipa_line', 'PCE Category': 
                              'pce_category','Commodity Code': 'comm_code', 
                              'Commodity Description': 'comm_descrip', 
                              'Producer Value': 'prod_value', 
                              'Transportation Value': 'trans_value', 
                              'Wholesale': 'whole_value', 
                              'Retail': 'retail_value', 
                              'Purchasers Value': 'purch_value', 
                              'Year': 'year'}, inplace=True)

# merge w/ x-walk to get cons categories for model
pce_bridge_df = pd.merge(pce_bridge_df,cons_cat_df,how='left',on='nipa_line')

# merge w/ x-walk to get production industries for model
pce_bridge_df = pd.merge(pce_bridge_df,prod_ind_df,how='left',on='comm_code')

# collapse data by model consumption category and production industry
pce_bridge_sums_df = pce_bridge_df.groupby(['cons_cat','production_industry']
                                           ,as_index=False)['prod_value',
                                           'trans_value','whole_value',
                                           'retail_value','purch_value'].sum()
prod_sum_df = pce_bridge_sums_df[['cons_cat','production_industry',
                                  'prod_value']]
prod_sum_df.rename(columns={'prod_value': 'value'}, inplace=True)
# Put zeros for categories with no cons goods
# The fills values to make matrix IxM
prod_sum_df.loc[len(prod_sum_df)+1]=[0,2,0] 
prod_sum_df.loc[len(prod_sum_df)+1]=[0,5,0] 
prod_sum_df.loc[len(prod_sum_df)+1]=[0,17,0] 
prod_sum_df.to_csv(path+'test_prod_sum.csv', index=False)

# Create totals for transport, whole sale, retail
trans_sum_df = pce_bridge_df.groupby(['cons_cat'],
                                     as_index=False)['trans_value'].sum()
trans_sum_df.rename(columns={'trans_value': 'value'}, inplace=True)
trans_sum_df['production_industry'] = 12
retail_sum_df = pce_bridge_df.groupby(['cons_cat'],
                                      as_index=False)['retail_value'].sum()
retail_sum_df.rename(columns={'retail_value': 'value'}, inplace=True)
retail_sum_df['production_industry'] = 11
whole_sum_df = pce_bridge_df.groupby(['cons_cat'],
                                     as_index=False)['whole_value'].sum()
whole_sum_df.rename(columns={'whole_value': 'value'}, inplace=True)
whole_sum_df['production_industry'] = 10


# append data so have each prod industry together
pi_long_df=whole_sum_df.append([retail_sum_df,trans_sum_df,prod_sum_df])
# collaspe again, because some transport, etc in prod industries 
pi_long_df = pi_long_df.groupby(['cons_cat','production_industry']
                                           ,as_index=False)['value'].sum()

# pivot table to create matrix
pi_mat_df=pd.pivot_table(pi_long_df,index='production_industry',columns='cons_cat', values='value',fill_value=0)

# put in percentages instead of dollars (sum percent over cons_cat =1)
# I'm sure there is  more elegant way than using a merge...
con_tot_df = pi_long_df.groupby(["cons_cat"],as_index=False)['value'].sum()
con_tot_df.rename(columns={'value': 'cons_total'}, inplace=True)
pi_long_pct_df = pd.merge(pi_long_df,con_tot_df,how='left',on='cons_cat')
pi_long_pct_df['fraction'] = pi_long_pct_df['value']/pi_long_pct_df['cons_total']

# pivot table to create matrix
pi_pct_mat_df=pd.pivot_table(pi_long_pct_df,index='production_industry', 
                             columns='cons_cat', values='fraction',fill_value=0)


# save files to csv
pi_long_pct_df.to_csv(path+'test_Pi_long.csv', index=False)
pi_pct_mat_df.to_csv(path+'test_Pi_mat_pct.csv', index=False)
pi_mat_df.to_csv(path+'test_Pi_mat.csv', index=False)
