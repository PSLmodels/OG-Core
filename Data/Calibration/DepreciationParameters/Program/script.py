'''
------------------------------------------------------------------------
Last updated 3/2/2015




This py-file calls the following other file(s):
            data\detailnonres_stk1.xlsx
            
This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT\Industry_Depreciation_Rates.csv
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

#
import industry_class

'''
------------------------------------------------------------------------
    
------------------------------------------------------------------------
    
------------------------------------------------------------------------
'''

# Working directory, note that there should already be an "OUTPUT" file as well
#   as a "data" file with all relevant data files.
path = os.getcwd()
# Relevant path and file names:
data_folder = path + "\data"
output_folder = path + "\OUTPUT"


'''
-------------------------------------------------------------------------------
Getting the SOI Tax Stats-Corporation Source Book Data:
    This data summarizes the 1120 tax filings by industry for corporations as a 
    whole as well as s corporations. By subtracting the s corporations from the
    totals give, this data indirectly gives data for both c and s corporations.
------------------------------------------------------------------------------
'''
# Finding the "20__sbfltfile" file in the data_folder that contains Tax-Stats-
#   -Corporation Data:
for i in os.listdir(data_folder):
    if(i[2:] == "sbfltfile"):
        sbflt_year = "20" + i[:2]
        sbflt_folder = data_folder + "\\" + sbflt_year[2:] + "sbfltfile"

# The aggregate 1120 filings data for all corporations:
tot_corp_file = "\\" + sbflt_year + "sb1.csv"
tot_corp_data = pd.read_csv(sbflt_folder + tot_corp_file).fillna(0)
# The aggregate 1120 filings data for all S corporations:
s_corp_file = "\\" + sbflt_year + "sb3.csv"
s_corp_data = pd.read_csv(sbflt_folder + s_corp_file).fillna(0)

'''
-------------------------------------------------------------------------------
Relevant column headers in the data sets:
    --"INDY_CD":                SOI industry code.
    --"AC":                     Asset class.
    --"DPRCBL_ASSTS":           Depreciable assets.
    --"ACCUM_DPR":              Accumulated Depreciation.
    --"LAND":                   Land.
    --"INVNTRY":                Inventories.
    --"INTRST_PD":              Interest paid.
    --"CAP_STCK":               Capital stock.
    --"PD_CAP_SRPLS":           Additional paid-in Capital.
    --"RTND_ERNGS_APPR":        Retained earnings, appropiated.
    --"COMP_RTND_ERNGS_UNAPPR": Retained earnings, unappropiated.
    --"CST_TRSRY_STCK":         Cost of treasury stock.
-------------------------------------------------------------------------------
'''
# A listing of all unique industry categories that the IRS breaks down the
#   total corporations data into for the 1120 forms
ind_codes = np.unique(tot_corp_data["INDY_CD"])
ind_num_codes = len(np.unique(tot_corp_data["INDY_CD"]))

# Listing out the relevant columns that are being extracted from the dataset.
data_columns = np.array(
                ["Code","Depreciable Assets",
                "Accumulated Depreciation", "Land", "Inventories",
                "Interest Paid", "Capital Stock", 
                "Additional paid-in Capital",
                "Retained Earnings (appropiated)",
                "Retained Earnings (unappropiated)",
                "Cost of Treasury Stock"]
                )

'''
Initalizes a pandas dataframe with a specific setup.
The structure of the dataset can be directly changed from here.
'''
def __init_data(the_data):
    return pd.DataFrame(np.zeros((1,len(the_data))), columns = the_data)

# Initializing some of the 
#tot_ind = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)
#s_ind = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)
#c_ind = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)

# Initializes a tree of  SOI industry categories, for data on all corporations:
tot_industries = industry_class.industry([])
tot_industries.load_ind(data_folder + "\SOI_Corporation_Industry_List.csv")
tot_industries.data = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)


# Initializes a tree of  SOI industry categories, for data on s corporations:
s_industries = industry_class.industry([])
s_industries.load_ind(data_folder + "\SOI_Corporation_Industry_List.csv")
s_industries.data = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)

# Initializes a tree of  SOI industry categories, for data on c corporations:
c_industries = industry_class.industry([])
c_industries.load_ind(data_folder + "\SOI_Corporation_Industry_List.csv")
c_industries.data = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)

# Aggregates each industry across asset classes using total corporation data:
for code_num in np.unique(tot_corp_data["INDY_CD"]):
    # Initializing a the dataframe to hold the data for one industry:
    tot_data_dummy = pd.DataFrame(np.zeros((1,len(data_columns))), 
                                  columns = data_columns)                              
    # Indicators for rows in tot_corp_data with the current industry number:
    indicators = (tot_corp_data["INDY_CD"] == code_num)
    # Filling in every column in the dataframe:
    tot_data_dummy["Code"][0] = code_num
    # See 'relevant column headers' comment to see what the tot_corp_data
    #   column headers correspond to (search this file).
    tot_data_dummy["Depreciable Assets"][0] = sum(
            indicators * tot_corp_data["DPRCBL_ASSTS"]
            )
    tot_data_dummy["Accumulated Depreciation"][0] = sum(
            indicators * tot_corp_data["ACCUM_DPR"]
            )
    tot_data_dummy["Land"][0] = sum(
            indicators * tot_corp_data["LAND"]
            )
    tot_data_dummy["Inventories"][0] = sum(
            indicators * tot_corp_data["INVNTRY"]
            )
    tot_data_dummy["Interest Paid"][0] = sum(
            indicators * tot_corp_data["INTRST_PD"]
            )
    tot_data_dummy["Capital Stock"][0] = sum(
            indicators * tot_corp_data["CAP_STCK"]
            )
    tot_data_dummy["Additional paid-in Capital"][0] = sum(
            indicators * tot_corp_data["PD_CAP_SRPLS"]
            )
    tot_data_dummy["Retained Earnings (appropiated)"][0] = sum(
            indicators * tot_corp_data["RTND_ERNGS_APPR"]
            )
    tot_data_dummy["Retained Earnings (unappropiated)"][0] = sum(
            indicators * tot_corp_data["COMP_RTND_ERNGS_UNAPPR"]
            )
    tot_data_dummy["Cost of Treasury Stock"][0] = sum(
            indicators * tot_corp_data["CST_TRSRY_STCK"]
            )
    
    # Finding the pointer for num_code industry in the 'tot_industries' tree.
    current_ind = tot_industries.find(code_num)
    # Updating it with the aggregated data:
    current_ind.update_data(tot_data_dummy)

'''
There are some industries listed in the IRS sbflt file documentation that have
    only one sub-category. Often, the IRS files will not explicitly give data
    on these sub-categories. For robustness, and ease of setting up the csv
    file with the SOI IRS industries, we will automatically populate these
    industries with the data from the parent industry:
'''
tot_industries.populate_singles()

# Aggregates each industry across asset classes using total corporation data:
for code_num in np.unique(s_corp_data["INDY_CD"]):
    # Initializing a the dataframe to hold the data for one industry:
    s_data_dummy = pd.DataFrame(np.zeros((1,len(data_columns))), 
                                columns = data_columns)
    # Indicators for rows in tot_corp_data with the current industry number:
    indicators = (s_corp_data["INDY_CD"] == code_num)
    # Filling in every column in the dataframe:
    s_data_dummy["Code"][0] = code_num
    # See 'relevant column headers' comment to see what the tot_corp_data
    #   column headers correspond to (search this file).
    s_data_dummy["Depreciable Assets"][0] = sum(
            indicators * s_corp_data["DPRCBL_ASSTS"]
            )
    s_data_dummy["Accumulated Depreciation"][0] = sum(
            indicators * s_corp_data["ACCUM_DPR"]
            )
    s_data_dummy["Land"][0] = sum(
            indicators * s_corp_data["LAND"]
            )
    s_data_dummy["Inventories"][0] = sum(
            indicators * s_corp_data["INVNTRY"]
            )
    s_data_dummy["Interest Paid"][0] = sum(
            indicators * s_corp_data["INTRST_PD"]
            )
    s_data_dummy["Capital Stock"][0] = sum(
            indicators * s_corp_data["CAP_STCK"]
            )
    s_data_dummy["Additional paid-in Capital"][0] = sum(
            indicators * s_corp_data["PD_CAP_SRPLS"]
            )
    '''
    Retained earnings (appropiated), are not reported for S Corporations.
    '''
    s_data_dummy["Retained Earnings (appropiated)"] = 0
    
    s_data_dummy["Retained Earnings (unappropiated)"][0] = sum(
            indicators * s_corp_data["COMP_RTND_ERNGS_UNAPPR"]
            )
    s_data_dummy["Cost of Treasury Stock"][0] = sum(
            indicators * s_corp_data["CST_TRSRY_STCK"]
            )
    
    # Finding the pointer for num_code industry in the 'tot_industries' tree.
    current_ind = s_industries.find(code_num)
    # Updating it with the aggregated data:
    current_ind.update_data(s_data_dummy)
    
# Call the industries helpe function to populate the data down the tree.
s_industries.populate_down(tot_industries)
tot_industries.subtract(s_industries, c_industries)

corp_columns = ["Code","C_FA","C_INV","C_LAND","S_FA","S_INV","S_LAND"]
corp_industries = industry_class.industry([])
corp_industries.load_ind(data_folder + "\SOI_Corporation_Industry_List.csv")
corp_industries.data = pd.DataFrame(np.zeros((1,len(corp_columns))), columns = corp_columns)

for code_num in np.unique(tot_corp_data["INDY_CD"]):
    c_data_dummy = c_industries.find(code_num).data
    s_data_dummy = s_industries.find(code_num).data
    # Initializing a the dataframe to hold the data for one industry:
    corp_data_dummy = pd.DataFrame(np.zeros((1,len(corp_columns))), 
                                   columns = corp_columns)
    # Filling in every column in the dataframe:
    corp_data_dummy["Code"] = code_num
    corp_data_dummy["C_FA"] = c_data_dummy["Depreciable Assets"] - c_data_dummy["Accumulated Depreciation"]
    corp_data_dummy["C_INV"] = c_data_dummy["Inventories"]
    corp_data_dummy["C_LAND"] = c_data_dummy["Land"]
    try:
        corp_data_dummy["S_FA"] = s_data_dummy["Depreciable Assets"] - s_data_dummy["Accumulated Depreciation"]
        corp_data_dummy["S_INV"] = s_data_dummy["Inventories"]
        corp_data_dummy["S_LAND"] = s_data_dummy["Land"]
    except KeyError:
        pass
    
    current_ind = corp_industries.find(code_num)
    current_ind.update_data(corp_data_dummy)






