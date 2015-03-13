'''
-------------------------------------------------------------------------------
Last updated 3/10/2015
-------------------------------------------------------------------------------
This py-file calls the following file(s):
            data\**sbfltfile\****sb1
            data\**sbfltfile\****sb3
-------------------------------------------------------------------------------
    Packages
-------------------------------------------------------------------------------
'''
import os.path
import numpy as np
import pandas as pd
import xlrd
# Predefined classes and functions for processing the data:
import data_class as dc
import naics_processing as naics

'''
-------------------------------------------------------------------------------
The main script of the program:
-------------------------------------------------------------------------------
'''
# Working directory:
path = os.getcwd()
# Relevant path and file names:
data_folder = path + "\data"
output_folder = path + "\OUTPUT"

'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Corporation Source Book Data:
(Note: SOI gives data for all corporations as well as for just s-corporations.
    The c-corporation is inferred from this.)
-------------------------------------------------------------------------------
'''
# Finding the "\**sbfltfile" file in the data_folder that contains Tax-Stats-
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
Note on the column names used in the SOI files:
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
'''
# Listing out the relevant columns that are being extracted from the dataset.
data_columns = np.array(
                ["Depreciable Assets",
                "Accumulated Depreciation", "Land", "Inventories",
                "Interest Paid", "Capital Stock", 
                "Additional paid-in Capital",
                "Retained Earnings (appropiated)",
                "Retained Earnings (unappropiated)",
                "Cost of Treasury Stock"]
                )
                
# Initializes a tree of  SOI industry categories, for data on all corporations:
data_tree = naics.load_naics(data_folder + "\\2012_NAICS_Codes.csv")
for i in data_tree.enum_inds:
    i.append_dfs(("tot_corps", pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)))
    i.append_dfs(("s_corps", pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)))
    i.append_dfs(("c_corps", pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)))

'''
-------------------------------------------------------------------------------
Processing the SOI data by putting it in tree objects (from data_class):
-------------------------------------------------------------------------------
'''

# Aggregates each industry across asset classes using total-corporation data:
count = 0
for code_num in np.unique(tot_corp_data["INDY_CD"]):
    found = False
    for i in range(0, len(data_tree.enum_inds)):
        count = (count + 1) % len(data_tree.enum_inds)
        cur_dfs0 = data_tree.enum_inds[i].data.dfs["Codes:"]
        for j in range(0, cur_dfs0.shape[0]):
            if(cur_dfs0.iloc[j,0] == code_num):
                found = True
                cur_dfs1 = data_tree.enum_inds[i].data.dfs["tot_corps"]
                break
        if found:
            break
    if not found:
        continue
    # Indicators for rows in tot_corp_data with the current industry number:
    indicators = (tot_corp_data["INDY_CD"] == code_num)
    # Filling in every column in the dataframe:
    cur_dfs1["Accumulated Depreciation"][0] = sum(
            indicators * tot_corp_data["ACCUM_DPR"]
            )
    cur_dfs1["Depreciable Assets"][0] = sum(
            indicators * tot_corp_data["DPRCBL_ASSTS"]
            )
    cur_dfs1["Land"][0] = sum(
            indicators * tot_corp_data["LAND"]
            )
    cur_dfs1["Inventories"][0] = sum(
            indicators * tot_corp_data["INVNTRY"]
            )
    cur_dfs1["Interest Paid"][0] = sum(
            indicators * tot_corp_data["INTRST_PD"]
            )
    cur_dfs1["Capital Stock"][0] = sum(
            indicators * tot_corp_data["CAP_STCK"]
            )
    cur_dfs1["Additional paid-in Capital"][0] = sum(
            indicators * tot_corp_data["PD_CAP_SRPLS"]
            )
    cur_dfs1["Retained Earnings (appropiated)"][0] = sum(
            indicators * tot_corp_data["RTND_ERNGS_APPR"]
            )
    cur_dfs1["Retained Earnings (unappropiated)"][0] = sum(
            indicators * tot_corp_data["COMP_RTND_ERNGS_UNAPPR"]
            )
    cur_dfs1["Cost of Treasury Stock"][0] = sum(
            indicators * tot_corp_data["CST_TRSRY_STCK"]
            )

# Aggregates each industry across asset classes using s-corporation data:            
count = 0
for code_num in np.unique(s_corp_data["INDY_CD"]):
    found = False
    for i in range(0, len(data_tree.enum_inds)):
        count = (count + 1) % len(data_tree.enum_inds)
        cur_dfs0 = data_tree.enum_inds[i].data.dfs["Codes:"]
        for j in range(0, cur_dfs0.shape[0]):
            if(cur_dfs0.iloc[j,0] == code_num):
                found = True
                cur_dfs1 = data_tree.enum_inds[i].data.dfs["s_corps"]
                break
        if found:
            break
    if not found:
        continue
    # Indicators for rows in s_corp_data with the current industry number:
    indicators = (s_corp_data["INDY_CD"] == code_num)
    # Filling in every column in the dataframe:
    cur_dfs1["Depreciable Assets"][0] = sum(
            indicators * s_corp_data["DPRCBL_ASSTS"]
            )
    cur_dfs1["Accumulated Depreciation"][0] = sum(
            indicators * s_corp_data["ACCUM_DPR"]
            )
    cur_dfs1["Land"][0] = sum(
            indicators * s_corp_data["LAND"]
            )
    cur_dfs1["Inventories"][0] = sum(
            indicators * s_corp_data["INVNTRY"]
            )
    cur_dfs1["Interest Paid"][0] = sum(
            indicators * s_corp_data["INTRST_PD"]
            )
    cur_dfs1["Capital Stock"][0] = sum(
            indicators * s_corp_data["CAP_STCK"]
            )
    cur_dfs1["Additional paid-in Capital"][0] = sum(
            indicators * s_corp_data["PD_CAP_SRPLS"]
            )
    '''
    Retained earnings (appropiated), are not reported for S Corporations.
    '''
    cur_dfs1["Retained Earnings (appropiated)"] = 0
    
    cur_dfs1["Retained Earnings (unappropiated)"][0] = sum(
            indicators * s_corp_data["COMP_RTND_ERNGS_UNAPPR"]
            )
    cur_dfs1["Cost of Treasury Stock"][0] = sum(
            indicators * s_corp_data["CST_TRSRY_STCK"]
            )

'''
Many industries are not listed in the SOI datasets. The data for these missing
    industries are interpolated.
'''

for corps in ["tot_corps", "s_corps"]:
    cur_dfs0 = None
    was_empty = [False]*len(data_tree.enum_inds)
    count = len(data_tree.enum_inds)-1
    # Working backwards through the tree
    for i in range(1, len(data_tree.enum_inds)):
        cur_dfs0 = data_tree.enum_inds[count].data.dfs[corps]
        par_dfs = data_tree.enum_inds[data_tree.par[count]].data.dfs[corps]
        cur_dfs_filled = False
        if sum((cur_dfs0 != pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)).iloc[0]) == 0:
            cur_dfs_filled = False
        else:
            cur_dfs_filled = True
        if not sum((par_dfs != pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)).iloc[0]) == 0:
            was_empty[data_tree.par[i]] = True            
        if cur_dfs_filled and was_empty[data_tree.par[i]]:
            par_dfs += cur_dfs0
        count = count - 1
        
    # Working forwards through the tree:
    for i in range(0, len(data_tree.enum_inds)):
        if data_tree.enum_inds[i].sub_ind != []:
            cur_ind = data_tree.enum_inds[i]
            cur_dfs0 = cur_ind.data.dfs[corps]
            sum_dfs = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)
            proportions = pd.DataFrame(np.zeros((1,len(data_columns))), columns = data_columns)
            for j in cur_ind.sub_ind:
                sum_dfs += j.data.dfs[corps]
            # j's flip!
            for j in range(0, len(data_columns)):
                if sum_dfs.iloc[0,j] == 0:
                    for k in cur_ind.sub_ind:
                        k.data.dfs[corps].iloc[0,j] = cur_dfs0.iloc[0,j]/len(cur_ind.sub_ind)
                else:
                    proportions.iloc[0,j] = cur_dfs0.iloc[0,j]/sum_dfs.iloc[0,j]
                    for k in cur_ind.sub_ind:
                        k.data.dfs[corps].iloc[0,j] = proportions.iloc[0,j]*k.data.dfs[corps].iloc[0,j]


# Inferring the c-corporation data from the 
for i in range(0, len(data_tree.enum_inds)):
    data_tree.enum_inds[i].data.dfs["c_corps"] = data_tree.enum_inds[i].data.dfs["tot_corps"] - data_tree.enum_inds[i].data.dfs["s_corps"]


'''
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
'''

soi_pa_folder = data_folder + "\SOI_Partner"
# Find the year corresponding to the 'partnership' data files:
for i in os.listdir(soi_pa_folder):
    if(i[2:] == "pa01.xls"):
        pa_year = "20" + i[:2]
# Names of the files with the partnership data:
pa_01_file = "\\" + pa_year[2:] + "pa01.xls"
pa_03_file = "\\" + pa_year[2:] + "pa03.xlsx"  #Noteb this is .xlsx
pa_05_file = "\\" + pa_year[2:] + "pa05.xls"

# Names of the files mapping the data to NAICS Codes:
pa_01_cross_file = "\\" + pa_year[2:] + "pa01_Crosswalk.csv"
pa_03_cross_file = "\\" + pa_year[2:] + "pa03_Crosswalk.csv"
pa_05_cross_file = "\\" + pa_year[2:] + "pa05_Crosswalk.csv"

'''
Collecting the data on net income and losses by industry from the partnership
    data set "**pa01.xls":
'''
# Opening the workbook and worksheet with this data:
book_01 = xlrd.open_workbook(soi_pa_folder + pa_01_file)
sheet_01 = book_01.sheet_by_index(0)
# Finding the relevant details about the table, e.g. dimensions:
cur_rows = sheet_01.nrows
# The data to be extracted:
cols_01 = ["Total net income", "Total net loss"]
data_01 = [None]*2
# Extracting the data:
for i in xrange(0, cur_rows):
    if("total net income".lower() in str(sheet_01.cell_value(i,0)).lower()):
        data_01[0] = sheet_01.row_values(i+1,2)
        data_01[1] = sheet_01.row_values(i+2,2)
        break
# Reformatting the data:
data_01 = pd.DataFrame(data_01).T
'''
Collecting the data on depreciable fixed assets, inventories, and land. This 
    data is taken from the SOI partnership data set "**pa03.xls":
'''
# Opening the workbook and worksheet with this data:
book_03 = xlrd.open_workbook(soi_pa_folder + pa_03_file)
sheet_03 = book_03.sheet_by_index(0)
# Finding the relevant details about the table, e.g. dimensions:
cur_rows = sheet_03.nrows
# The following categories of data to be extracted:
cols_03 = ["Depreciable assets (Net)", "Accumulated depreciation (Net)", 
                "Inventories (Net)", "Land (Net)", 
                "Depreciable assets (Income)", 
                "Accumulated depreciation (Income)", "Inventories (Income)",
                "Land (Income)"]
# The following general categories of data will be extracted seperately for
#   all partnerships as well as all partnerships with income:
gen_cols_03 = ["Depreciable assets", "Accumulated depreciation", 
                "Inventories", "Land"]

# The data to be extracted on partnerships as a whole:
tot_data_03 = [None]*len(gen_cols_03)
# The data to be extracted on partnerships with income:
inc_data_03 = [None]*len(gen_cols_03)
# Extracting the data (note that the rows with total data appear first):
for i in xrange(0, len(gen_cols_03)):
    for row1 in xrange(0, cur_rows):
        if(gen_cols_03[i].lower() in str(sheet_03.cell_value(row1,0)).lower()):
            tot_data_03[i] = sheet_03.row_values(row1,2)
            for row2 in xrange(row1+1, cur_rows):
                if(gen_cols_03[i].lower() in str(sheet_03.cell_value(row2,0)).lower()):
                    inc_data_03[i] = sheet_03.row_values(row2,2)
                    break
            break
# Reformatting the data:
data_03 = pd.concat([pd.DataFrame(tot_data_03).T,pd.DataFrame(inc_data_03).T], axis = 1)

'''
Collecting the data on income/loss by industry and sector. This data is taken
    from the SOI partnership data set "**pa05.xls":
'''
# Opening the workbook and worksheet with this data:
book_05 = xlrd.open_workbook(soi_pa_folder + pa_05_file)
sheet_05 = book_05.sheet_by_index(0)
# Finding the relevant details about the table, e.g. dimensions:
cur_rows = sheet_05.nrows
# The following categories of data to be extracted:
'''
Sectors:
Corporate general partners                  CG
Corporate limited partners                  CL
Individual general partners                 IG
Individual limited partners                 IL
Partnership general partners                PG
Partnership limited partners                PL
Tax-exempt organization general partners    TOG
Tax-exempt organization limited partners    TOL
Nominee and other general partners          NOG
Nominee and other limited partners          NOL
'''
ptner_types = ["CG", "CL", "IG", "IL", "PG", "PL", "TOG", "TOL", "NOG", "NOL"]
cols_05 = ["Corporate general partners", "Corporate limited partners",
                "Individual general partners", "Individual limited partners",
                "Partnership general partners", "Partnership limited partners",
                "Tax-exempt organization general partners",
                "Tax-exempt organization limited partners",
                "Nominee and other general partners", 
                "Nominee and other limited partners"]
# The data to be extracted
data_05 = [None]*len(cols_05)
# Extracting the data:
for i in xrange(0, len(cols_05)):
    for row in xrange(0, cur_rows):
        if(cols_05[i].lower() in str(sheet_05.cell_value(row,0)).lower()):
            data_05[i] = sheet_05.row_values(row,2)
            break
# Reformat data:
data_05 = pd.DataFrame(data_05).T
# Removing no longer relevant workbook/worksheet variables:
book_01 = None
sheet_01 = None
book_03 = None
sheet_03 = None
book_05 = None
sheet_05 = None
# Reading in the crosswalks between the columns and the NAICS codes:
pa01cross = pd.read_csv(soi_pa_folder + pa_01_cross_file)
pa03cross = pd.read_csv(soi_pa_folder + pa_03_cross_file)
pa05cross = pd.read_csv(soi_pa_folder + pa_05_cross_file)
'''
Processing the partnership data into to the tree.
'''
for index in xrange(0,3):
    #
    cur_name = ["PA_inc/loss","PA_assets","PA_types"][index]
    cur_data = [data_01, data_03, data_05][index]
    cur_cols = [cols_01, cols_03, cols_05][index]
    cur_cross = [pa01cross, pa03cross, pa05cross][index]
    cur_count = 0
    #
    for i in data_tree.enum_inds:
        i.append_dfs((cur_name, pd.DataFrame(np.zeros((1,len(cur_cols))), columns = cur_cols)))
    #
    for i in xrange(0, len(cur_cross["NAICS Code:"])):
        if pd.isnull(cur_cross["NAICS Code:"][i]):
            continue
        cur_df = pd.DataFrame(np.zeros((1,len(cur_cols))), columns = cur_cols)
        cur_codes = cur_cross["NAICS Code:"][i].split(".")
        num_found = 0
        for k in xrange(0, len(cur_codes)):
            cur_codes[k] = int(cur_codes[k])
        for j in xrange(0, len(data_tree.enum_inds)):
            cur_ind = data_tree.enum_inds[cur_count]
            for k in cur_codes:
                for l in cur_ind.data.dfs["Codes:"][0]:
                    if(k == l):
                        cur_ind.data.dfs[cur_name] += np.array(cur_data.iloc[i,:])/len(cur_codes) #[data_01[0][i]/len(cur_codes),data_01[1][i]/len(cur_codes)]
                        num_found += 1
            cur_count = (cur_count+1) % len(data_tree.enum_inds)
            if(num_found == len(cur_codes)):
                    break
                
                
                
                
                
                