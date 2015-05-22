''' 
------------------------------------------------------------------------
Last updated 05/20/2015

Read in CEX data from 2012-2014 (interview survey), use these data to
create summary tables and estimate parameters of cons subutility function

This py-file calls the following other file(s):
            

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/cons_params.pkl
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
CEXpath = '/Users/jasondebacker/Econ/AEI_TaxModel_Project/Calibration/Data/'
CEX_file_name = 'categorizedcex.dta'
#Notes:
#Download the Consumer Expenditure Survey (CEX), years 2000-2013. 
#You can find the data here: http://www.bls.gov/cex/pumdhome.htm
 
#We download the Stata data 
#files including the codebook(s) that accompany the data.*/

#Use the monthly expenditure and income files (MTBI & ITBI) of the Interview Survey.
#Also, use the FMLI interview files for demographic data.*/
 
  
   
## Format the data
#Once each year's file is downloaded, we create a loop to append them together into one pooled 
#cross-sectional dataset. Total Household Income (UCC 980000)is available in the ITBI files.
# Since the consumption, income and demographic data is found in different files 
# we will have to merge the 3 appended files to create one comprehensive dataset. 


year_list = ('12','13')
qtr_list = ('1x','2','3','4','1')
ucc_inc_list = ('980000','900150','980230','980240','980260')
#ucc_inc 980000 is income before tax and 900150 is annual value of food stamps
	#ucc_inc 980230 , 980240 & 980260 are Homeowner variables  distinguishing whether 
	#the consumer unit owns (with or without mortgage) or rents the home.
	
for t in year_list:
    for q in qtr_list:
	# Clean up income data files			
	temp_df = pd.io.stata.read_stata(CEXpath+'intrvw'+t+'/'+'itbi'+t+q+'.dta')
	temp_df = temp_df.rename(columns=lambda x: x+'_inc')
	temp_df = temp_df.rename(columns={'refmo_inc': 'ref_mo', 'refyr_inc': 'ref_yr', 'newid_inc': 'newid'})
        temp_df['newid'] = temp_df['newid'].astype(str)
        temp_df['cu'] = temp_df['newid'].str[:-1].astype(str) # remove last character to get consumer unit
        temp_df['no_intrvw'] = temp_df['newid'].str[-1:].astype(str) # get number of interview
        # keep just certain ucc codes
        temp_df = temp_df[temp_df['ucc_inc'].isin(ucc_inc_list)]
        # drop value__inc
        temp_df = temp_df.drop('value__inc', axis=1, inplace=True)
        reshape wide value_inc, i(cu ref_mo) j(ucc_inc) string
        
        temp_df['value_ucc'] = 'value_inc' + temp_df.ucc_inc.astype(str)
        
        value_ucc = temp_df.pivot(index='cu',columns='value_ucc',values='value_inc')

        reshape = pd.concat([product,prc],axis=1)
        value_ucc['ref_mo'] = temp_df.set_index('cu')['ref_mo'].drop_duplicates()
        print reshape
        
        data_wide = pd.pivot_table(temp_df, values=['variable', 'species', 'value'],
                           rows=['id', 'variable']).unstack()
        
        
        
        
        idx = []
        data = []
df_grpd = temp_df.groupby(['cu', 'ref_mo'])
for name, group in df_grpd:
    idx.append(name)
    data.append(np.column_stack(([group['value'],group['lsc'],
                                 group['pop']])).flatten().tolist())

ages = map(str, group["agefrom"])
cols = ",".join([",".join(['ls' + age] + ['lsc' + age] + ['pop' + age]) for age in ages]).split(",")

new_df = pandas.DataFrame(data, index=idx, columns=cols)
        
        
        
        
        
        pivoted = temp_df.pivot('salesman', 'product', 'price')
        if `counter' == 0 {
						save "${path}/Intermediate/income.dta", replace
						}
					else {
						append using "${path}/Intermediate/income.dta"
						save "${path}/Intermediate/income.dta", replace
						}
						
						
			tostring newid, replace
				gen cu = substr(newid,1,5) if length(newid) == 6
				replace cu = substr(newid,1,6) if length(newid) == 7
				gen no_intrvw = substr(newid,-1,1)
				keep if ucc_inc == "980000" | ucc_inc== "900150" | ucc_inc== "980230" | ucc_inc== "980240" | ucc_inc== "980260" 
				*ucc_inc 980000 is income before tax and 900150 is annual value of food stamps
				*ucc_inc 980230 , 980240 & 980260 are Homeowner variables  distinguishing whether the consumer unit owns (with or without mortgage) or rents the home.
				drop value__inc
				reshape wide value_inc, i(cu ref_mo) j(ucc_inc) string
				if `counter' == 0 {
						save "${path}/Intermediate/income.dta", replace
						}
					else {
						append using "${path}/Intermediate/income.dta"
						save "${path}/Intermediate/income.dta", replace
						}
			
		/* Clean up demographic data files*/				
				local dir = "${path}/Raw/20`yr'/intrvw`yr'/fmli`yr2'"+"`x'"+".dta"
				use `dir', clear
				di "`dir'"
				tostring newid, replace
				gen cu = substr(newid,1,5) if length(newid) == 6
				replace cu = substr(newid,1,6) if length(newid) == 7
				gen no_intrvw = substr(newid,-1,1)
				keep cu no_intrvw finlwt21 age_ref age_ref_ no_earnr fam_size fam__ize roomsq roomsq_ bathrmq bathrmq_ hlfbathq
				* Age of Reference Person (AGE_REF), Household size (FAM_SIZE), dwelling size (ROOMSQ+BATHRMQ+HLFBATHQ), number of earners in household (NO_EARNR) and FINLWT21 is the weighting variable for the full sample
					if `counter' == 0 {
						save "${path}/Intermediate/demo.dta", replace
						}
					else {
						append using "${path}/Intermediate/demo.dta"
						save "${path}/Intermediate/demo.dta", replace
						}
						
		/* Clean up consumption data files*/
				local dir = "${path}/Raw/20`yr'/intrvw`yr'/mtbi`yr2'"+"`x'"+".dta"
				use `dir', clear
				di "`dir'"
				tostring newid, replace
				gen cu = substr(newid,1,5) if length(newid) == 6
				replace cu = substr(newid,1,6) if length(newid) == 7
				gen no_intrvw = substr(newid,-1,1)
					if `counter' == 0 {
						save "${path}/Intermediate/consumption.dta", replace
						}
					else {
						append using "${path}/Intermediate/consumption.dta"
						save "${path}/Intermediate/consumption.dta", replace
						}
				local counter = `counter'+1
				}
		local counter2 = `counter2'+1
	}
}
    
      
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
