'''
-------------------------------------------------------------------------------
Last updated 3/19/2015
-------------------------------------------------------------------------------
This py-file calls the following file(s):
        Raw input files:
            data\**sbfltfile\****sb1.csv
            data\**sbfltfile\****sb3.csv
            data\SOI_Partner\pa01.csv
            data\SOI_Partner\pa03.csv
            data\SOI_Partner\pa05.csv
            
        Formatted input files:
            data\****_NAICS_Codes.csv
            data\SOI_Partner\pa01_Crosswalk.csv
            data\SOI_Partner\pa03_Crosswalk.csv
            data\SOI_Partner\pa05_Crosswalk.csv
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
    --Loading the SOI Tax Stats-Corporation Data.
    --Loading the SOI Tax Stats-Partnership Data.
-------------------------------------------------------------------------------
'''
# Working directory:
path = os.getcwd()
# Relevant path and file names:
data_folder = path + "/data"
output_folder = path + "/OUTPUT"

'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Corporation Data:
(Note: SOI gives data for all corporations as well as for just s-corporations.
    The c-corporation data is inferred from this.)
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
Note on the column names used in the SOI-corporation files:
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
    i.append_dfs(("tot_corps", pd.DataFrame(np.zeros((1,len(data_columns))),
                                            columns = data_columns)))
    i.append_dfs(("s_corps", pd.DataFrame(np.zeros((1,len(data_columns))),
                                            columns = data_columns)))
    i.append_dfs(("c_corps", pd.DataFrame(np.zeros((1,len(data_columns))),
                                            columns = data_columns)))
# Loading total-corporation data:
enum_index = 0
for code_num in np.unique(tot_corp_data["INDY_CD"]):
    # Search through all industries to find one with a matching code:
    ind_found = False
    for i in range(0, len(data_tree.enum_inds)):
        enum_index = (enum_index + 1) % len(data_tree.enum_inds)
        cur_dfs = data_tree.enum_inds[i].data.dfs["Codes:"]
        for j in range(0, cur_dfs.shape[0]):
            if(cur_dfs.iloc[j,0] == code_num):
                # Industry with the matching code has been found:
                ind_found = True
                cur_dfs = data_tree.enum_inds[i].data.dfs["tot_corps"]
                break
        # If the matching industry has been found stop searching for it.
        if ind_found:
            break
    # If no match was found, then ignore data.(Perhaps should issue a warning.)
    if not ind_found:
        continue
    # Indicators for rows in tot_corp_data with the current industry number:
    indicators = (tot_corp_data["INDY_CD"] == code_num)
    # Filling in every column in the dataframe:
    cur_dfs["Accumulated Depreciation"][0] = sum(
            indicators * tot_corp_data["ACCUM_DPR"]
            )
    cur_dfs["Depreciable Assets"][0] = sum(
            indicators * tot_corp_data["DPRCBL_ASSTS"]
            )
    cur_dfs["Land"][0] = sum(
            indicators * tot_corp_data["LAND"]
            )
    cur_dfs["Inventories"][0] = sum(
            indicators * tot_corp_data["INVNTRY"]
            )
    cur_dfs["Interest Paid"][0] = sum(
            indicators * tot_corp_data["INTRST_PD"]
            )
    cur_dfs["Capital Stock"][0] = sum(
            indicators * tot_corp_data["CAP_STCK"]
            )
    cur_dfs["Additional paid-in Capital"][0] = sum(
            indicators * tot_corp_data["PD_CAP_SRPLS"]
            )
    cur_dfs["Retained Earnings (appropiated)"][0] = sum(
            indicators * tot_corp_data["RTND_ERNGS_APPR"]
            )
    cur_dfs["Retained Earnings (unappropiated)"][0] = sum(
            indicators * tot_corp_data["COMP_RTND_ERNGS_UNAPPR"]
            )
    cur_dfs["Cost of Treasury Stock"][0] = sum(
            indicators * tot_corp_data["CST_TRSRY_STCK"]
            )
            

# Loading s-corporation data:
enum_index = 0
for code_num in np.unique(s_corp_data["INDY_CD"]):
    # Search through all industries to find one with a matching code:
    ind_found = False
    for i in range(0, len(data_tree.enum_inds)):
        enum_index = (enum_index + 1) % len(data_tree.enum_inds)
        cur_dfs = data_tree.enum_inds[i].data.dfs["Codes:"]
        for j in range(0, cur_dfs.shape[0]):
            if(cur_dfs.iloc[j,0] == code_num):
                # Industry with the matching code has been found:
                ind_found = True
                cur_dfs = data_tree.enum_inds[i].data.dfs["s_corps"]
                break
        if ind_found:
            break
    if not ind_found:
        continue
    # Indicators for rows in s_corp_data with the current industry number:
    indicators = (s_corp_data["INDY_CD"] == code_num)
    # Filling in every column in the dataframe:
    cur_dfs["Depreciable Assets"][0] = sum(
            indicators * s_corp_data["DPRCBL_ASSTS"]
            )
    cur_dfs["Accumulated Depreciation"][0] = sum(
            indicators * s_corp_data["ACCUM_DPR"]
            )
    cur_dfs["Land"][0] = sum(
            indicators * s_corp_data["LAND"]
            )
    cur_dfs["Inventories"][0] = sum(
            indicators * s_corp_data["INVNTRY"]
            )
    cur_dfs["Interest Paid"][0] = sum(
            indicators * s_corp_data["INTRST_PD"]
            )
    cur_dfs["Capital Stock"][0] = sum(
            indicators * s_corp_data["CAP_STCK"]
            )
    cur_dfs["Additional paid-in Capital"][0] = sum(
            indicators * s_corp_data["PD_CAP_SRPLS"]
            )
    '''
    Retained earnings (appropiated), are not reported for S Corporations.
    '''
    cur_dfs["Retained Earnings (appropiated)"] = 0
    
    cur_dfs["Retained Earnings (unappropiated)"][0] = sum(
            indicators * s_corp_data["COMP_RTND_ERNGS_UNAPPR"]
            )
    cur_dfs["Cost of Treasury Stock"][0] = sum(
            indicators * s_corp_data["CST_TRSRY_STCK"]
            )

# Inferring the c-corporation data from the 
for i in range(0, len(data_tree.enum_inds)):
    data_tree.enum_inds[i].data.dfs["c_corps"] = data_tree.enum_inds[i].data.dfs["tot_corps"] - data_tree.enum_inds[i].data.dfs["s_corps"]

# Deleting variables:
sbflt_year = None
sbflt_folder = None
tot_corp_file = None
tot_corp_data = None
s_corp_file = None
s_corp_data = None
data_columns = None
enum_index = None
ind_found = None
cur_dfs = None
code_num = None
indicators = None

'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Partnership Data:
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
                cur_cell = str(sheet_03.cell_value(row2,0)).lower()
                if(gen_cols_03[i].lower() in cur_cell):
                    inc_data_03[i] = sheet_03.row_values(row2,2)
                    break
            break
# Reformatting the data:
data_03 = pd.concat([pd.DataFrame(tot_data_03).T,pd.DataFrame(inc_data_03).T],
                         axis = 1)

'''
Collecting the data on income/loss by industry and sector. This data is taken
    from the SOI partnership data set "**pa05.xls":
'''
book_05 = xlrd.open_workbook(soi_pa_folder + pa_05_file)
sheet_05 = book_05.sheet_by_index(0)
cur_rows = sheet_05.nrows
'''
Note on the names of the sectors used in this file:
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
# Extracting the relevant data:
data_05 = [None]*len(cols_05)
for i in xrange(0, len(cols_05)):
    for row in xrange(0, cur_rows):
        if(cols_05[i].lower() in str(sheet_05.cell_value(row,0)).lower()):
            data_05[i] = sheet_05.row_values(row,2)
            break
# Reformatting the data:
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
# Processing the three dataframes into the tree
for index in xrange(0,3):
    cur_name = ["PA_inc/loss","PA_assets","PA_types"][index]
    cur_data = [data_01, data_03, data_05][index]
    cur_cols = [cols_01, cols_03, cols_05][index]
    cur_cross = [pa01cross, pa03cross, pa05cross][index]
    enum_index = 0
    # Append empty dataframes to the tree:
    for i in data_tree.enum_inds:
        i.append_dfs((cur_name, pd.DataFrame(np.zeros((1,len(cur_cols))),
                                             columns = cur_cols)))
    # Add the data to industries in the tree based on fraction of codes shared:
    for i in xrange(0, len(cur_cross["NAICS Code:"])):
        if pd.isnull(cur_cross["NAICS Code:"][i]):
            continue
        cur_df = pd.DataFrame(np.zeros((1,len(cur_cols))), columns = cur_cols)
        cur_codes = cur_cross["NAICS Code:"][i].split(".")
        num_found = 0
        for k in xrange(0, len(cur_codes)):
            cur_codes[k] = int(cur_codes[k])
        for j in xrange(0, len(data_tree.enum_inds)):
            cur_ind = data_tree.enum_inds[enum_index]
            for k in cur_codes:
                for l in cur_ind.data.dfs["Codes:"][0]:
                    if(k == l):
                        cur_ind.data.dfs[cur_name] += np.array(
                                            cur_data.iloc[i,:])/len(cur_codes)
                        num_found += 1
            enum_index = (enum_index+1) % len(data_tree.enum_inds)
            if(num_found == len(cur_codes)):
                    break

'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Partnership Data:
-------------------------------------------------------------------------------
'''
prop_folder = data_folder + "\\SOI_Proprietorships"
# Finding the "\**sp01br" file in the proprietorships folder:
for i in os.listdir(prop_folder):
    if(i[2:] == "sp01br.xls"):
        prop_year = "20" + i[:2]
        sp01brfile = prop_folder + "\\" + prop_year[2:] + "sp01br.xls"
        sp01brfile_cross = prop_folder + "\\" + prop_year[2:] + "sp01br_Crosswalk.csv"

cur_wb = xlrd.open_workbook(sp01brfile)
cur_ws = cur_wb.sheet_by_index(0)
cur_cross = pd.read_csv(sp01brfile_cross)

pos1 = naics.search_ws(cur_ws,"Industrial sector",20, True, [0,0], True)
pos2 = naics.search_ws(cur_ws,"Depreciation\ndeduction",20)
pos3 = naics.search_ws(cur_ws,"Depreciation\ndeduction",20, True, np.array(pos2) + np.array([0,1]))

for i in data_tree.enum_inds:
    i.append_dfs(("soi_prop", pd.DataFrame(np.zeros((1,1)), columns = ["Depreciation Deductions"])))

count1 = 0
count2 = 0
for i in xrange(pos1[0],cur_ws.nrows):
    cur_cell = str(cur_ws.cell_value(i,pos1[1]))
    for j in xrange(0, cur_cross.shape[0]):
        if(cur_cell.lower().strip() == str(cur_cross.iloc[count1,0]).lower().strip()):
            cur_codes = str(cur_cross.iloc[count1,1]).split(".")
            for k in xrange(0, len(data_tree.enum_inds)):
                for l in data_tree.enum_inds[count2].data.dfs["Codes:"].iloc[:,0]:
                    for m in cur_codes:
                        if str(l) == str(m):
                            data_tree.enum_inds[count2].data.dfs["soi_prop"]["Depreciation Deductions"][0] += (cur_ws.cell_value(i,pos2[1]) + cur_ws.cell_value(i,pos3[1]))/len(cur_codes)
                count2 = (count2+1) % len(data_tree.enum_inds)
            break
        count1 = (count1+1) % cur_cross.shape[0]
        
'''

'''
'''
farm_cols = ["Land","Fixed Assets"]
for i in data_tree.enum_inds:
    i.append_dfs(("farm_prop", pd.DataFrame(np.zeros((1,len(farm_cols))), columns=farm_cols)))

farm_data = pd.read_csv(prop_folder + "\\Farm_Data.csv")
land_mult = (farm_data.iloc[0,1] + farm_data.iloc[0,3]) * (farm_data.iloc[0,5]/farm_data.iloc[0,4])
for i in data_tree.enum_inds:
     = land_mult * i.data.dfs["PA_assets"]["Land (Net)"][0]/(i.data.dfs["PA_assets"]["Land (Net)"][0] + i.data.dfs["PA_assets"]["Depreciable assets (Net)"][0])
'''


'''
Many industries are not listed in the SOI datasets. The data for these missing
    industries are interpolated.
'''
# Get a list of the names of all the pd dfs besides the list of codes:
a = data_tree.enum_inds[0].data.dfs.keys()
a.remove("Codes:")
# Populate missing industry data backwards throught the tree:
naics.pop_back(data_tree, a)
# Populate the missing total corporate data forwards through the tree:
naics.pop_forward(data_tree, ["tot_corps"])
# Populate all other missing data using tot_corps as a "blueprint":
a.remove("tot_corps")
naics.pop_forward(data_tree, a, "tot_corps")





    
