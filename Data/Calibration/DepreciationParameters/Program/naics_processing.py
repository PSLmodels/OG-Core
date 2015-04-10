'''
-------------------------------------------------------------------------------
Last updated 4/7/2015
-------------------------------------------------------------------------------
This py-file defines functions that process an NAICS tree. The tree contains
    data related to depreciation rates for each industry in the tree. These
    functions initialize and load data into the tree as well as interpolate
    missing values.
-------------------------------------------------------------------------------
    Packages
-------------------------------------------------------------------------------
'''
import os.path
import numpy as np
import pandas as pd
import xlrd
# A class containing predefined data structures such as trees:
import data_class as dc
'''
-------------------------------------------------------------------------------
Functions created:
    load_naics
    load_soi_corporate_data
    load_soi_partner_data
    load_soi_proprietor_data
    summary_tree
    pop_back
    pop_forward
    find_naics
    search_ws
-------------------------------------------------------------------------------
def load_naics: This reads in a csv list of naics codes and creates a tree.
-------------------------------------------------------------------------------
'''
def load_naics(path):
    # Reading in a list of naics codes:
    naics_codes = pd.read_csv(path).fillna(0)
    rows = naics_codes.shape[0]
    # Initializing the corresponding naics tree:
    naics_tree = dc.tree()
    naics_tree.enum_inds = [dc.industry([]) for i in xrange(0,rows)]
    naics_tree.root = naics_tree.enum_inds[0]
    naics_tree.par = [0]*rows
    # Read the naics codes into the tree:
    for i in xrange(0, rows):
        cur_codes = pd.DataFrame(naics_codes.iloc[i,0].split("-"))
        if(cur_codes.shape[0] == 2):
            cur_codes = pd.DataFrame(range(int(cur_codes.iloc[0,0]),
                                           int(cur_codes.iloc[1,0])+1))
        naics_tree.enum_inds[i].append_dfs(("Codes:", cur_codes))
        cur_rows = naics_tree.enum_inds[i].data.dfs["Codes:"].shape[0]
        for j in xrange(0, cur_rows):
            code = int(naics_tree.enum_inds[i].data.dfs["Codes:"].iloc[j,0])
            naics_tree.enum_inds[i].data.dfs["Codes:"].iloc[j,0] = code
    # Creating the tree structure:
    # "levels" keeps track of the path from the root to the current industry.
    levels = [None]
    levels[0] = naics_tree.enum_inds[0]
    levels_index = [0]
    cur_lvl = 0
    # Going through every industry in the tree and finding the parent/children:
    for i in xrange(1,rows):
        cur_ind = naics_tree.enum_inds[i]
        cur_codes = cur_ind.data.dfs["Codes:"]
        cur_rows = cur_codes.shape[0]
        par_found = False
        while not par_found:
            prev_ind = levels[cur_lvl]
            prev_codes = prev_ind.data.dfs["Codes:"]
            prev_rows = prev_codes.shape[0]
            for j in xrange(0, cur_rows):
                for k in xrange(0, prev_rows):
                    if cur_lvl == 0:
                        # The industry's parent is the root.
                        par_found = True
                        cur_lvl += 1
                        levels.append(cur_ind)
                        levels_index.append(i)
                        levels[0].sub_ind.append(cur_ind)
                        naics_tree.par[i] = levels_index[cur_lvl-1]
                        break
                    elif str(prev_codes.iloc[k,0]) in str(cur_codes.iloc[j,0]):
                        # "levels[cur_lvl]" is the parent of "cur_ind":
                        par_found = True
                        cur_lvl += 1
                        levels.append(cur_ind)
                        levels_index.append(i)
                        prev_ind.sub_ind.append(cur_ind)
                        naics_tree.par[i] = levels_index[cur_lvl-1]
                        break
                if(par_found):
                    break
            if not par_found:
                del levels[cur_lvl]
                del levels_index[cur_lvl]
                cur_lvl -= 1
    return naics_tree


'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Corporation Data:
(Note: SOI gives data for all corporations as well as for just s-corporations.
    The c-corporation data is inferred from this.)
-------------------------------------------------------------------------------
'''
def load_soi_corporate_data(data_tree, data_folder):
    # Finding the "\**sbfltfile" file in the data_folder (contains corp data):
    for i in os.listdir(data_folder):
        if(i[2:] == "sbfltfile"):
            sbflt_year = "20" + i[:2]
            sbflt_folder = os.path.abspath(
                              data_folder + "\\" + sbflt_year[2:] + "sbfltfile"
                              )
    # The aggregate 1120 filings data for all corporations:
    tot_corp_file = sbflt_folder + "\\" + sbflt_year + "sb1.csv"
    tot_corp_file = os.path.abspath(tot_corp_file)
    tot_corp_data = pd.read_csv(tot_corp_file).fillna(0)
    # The aggregate 1120 filings data for all S corporations:
    s_corp_file = sbflt_folder + "\\" + sbflt_year + "sb3.csv"
    s_corp_file = os.path.abspath(s_corp_file)
    s_corp_data = pd.read_csv(s_corp_file).fillna(0)
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
    # Listing the relevant columns that are being extracted from the dataset:
    cols_dict = dict([("Depreciable Assets","DPRCBL_ASSTS"),
                      ("Accumulated Depreciation", "ACCUM_DPR"),
                      ("Land", "LAND"),
                      ("Inventories", "INVNTRY"),
                      ("Interest Paid", "INTRST_PD"), 
                      ("Capital Stock", "CAP_STCK"),
                      ("Additional paid-in Capital", "PD_CAP_SRPLS"),
                      ("Earnings (rtnd appr)", "RTND_ERNGS_APPR"),
                      ("Earnings (rtnd unappr.)", "COMP_RTND_ERNGS_UNAPPR"),
                      ("Cost of Treasury Stock", "CST_TRSRY_STCK")])
    data_cols = cols_dict.keys()
    # Initializing data on all corporations:
    for i in data_tree.enum_inds:
        i.append_dfs(("tot_corps", pd.DataFrame(np.zeros((1,len(data_cols))),
                                                columns = data_cols)))
        i.append_dfs(("s_corps", pd.DataFrame(np.zeros((1,len(data_cols))),
                                                columns = data_cols)))
        i.append_dfs(("c_corps", pd.DataFrame(np.zeros((1,len(data_cols))),
                                                columns = data_cols)))
    # Loading total-corporation data:
    enum_index = 0
    for code_num in np.unique(tot_corp_data["INDY_CD"]):
        # Find the industry with a code that matches "code_num":
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
        # If no match was found, then ignore data.
        if not ind_found:
            continue
        # Indicators for if rows in tot_corp_data match current industry code:
        indicators = (tot_corp_data["INDY_CD"] == code_num)
        # Filling in every column in the dataframe:
        for j in cols_dict:
            cur_dfs[j][0] = sum(indicators * tot_corp_data[cols_dict[j]])
    # Loading s-corporation data:
    enum_index = 0
    for code_num in np.unique(s_corp_data["INDY_CD"]):
        # Find the industry with a code that matches "code_num":
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
            # If the matching industry has been found stop searching for it.
            if ind_found:
                break
        # If no match was found, then ignore data.
        if not ind_found:
            continue
        # Indicators for if rows in s_corp_data match current industry code:
        indicators = (s_corp_data["INDY_CD"] == code_num)
        # Filling in every column in the dataframe:
        for j in cols_dict:
            # "Earnings (rtnd appr)" are not reported for S Corporations.
            if j == "Earnings (rtnd appr)":
                cur_dfs[j] = 0
                continue
            cur_dfs[j][0] = sum(indicators * s_corp_data[cols_dict[j]])
    # Inferring the c-corporation data from the total and S corporation data:
    for i in range(0, len(data_tree.enum_inds)):
        cur_tot = data_tree.enum_inds[i].data.dfs["tot_corps"]
        cur_s = data_tree.enum_inds[i].data.dfs["s_corps"]
        data_tree.enum_inds[i].data.dfs["c_corps"] = cur_tot - cur_s


'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Partnership Data:
-------------------------------------------------------------------------------
'''
def load_soi_partner_data(data_tree, data_folder):
    soi_pa_folder = os.path.abspath(data_folder + "\\SOI_Partner")
    # Find the year corresponding to the 'partnership' data files:
    for i in os.listdir(soi_pa_folder):
        if(i[2:] == "pa01.xls"):
            pa_year = "20" + i[:2]
    # Names of the files with the partnership data:
    pa_01_file = "\\" + pa_year[2:] + "pa01.xls"
    pa_01_file = os.path.abspath(soi_pa_folder + pa_01_file)
    pa_03_file = "\\" + pa_year[2:] + "pa03.xlsx"
    pa_03_file = os.path.abspath(soi_pa_folder + pa_03_file)
    pa_05_file = "\\" + pa_year[2:] + "pa05.xls"
    pa_05_file = os.path.abspath(soi_pa_folder + pa_05_file)
    # Names of the files mapping the data to NAICS Codes:
    pa_01_cross_file = "\\" + pa_year[2:] + "pa01_Crosswalk.csv"
    pa_01_cross_file = os.path.abspath(soi_pa_folder + pa_01_cross_file)
    pa_03_cross_file = "\\" + pa_year[2:] + "pa03_Crosswalk.csv"
    pa_03_cross_file = os.path.abspath(soi_pa_folder + pa_03_cross_file)
    pa_05_cross_file = "\\" + pa_year[2:] + "pa05_Crosswalk.csv"
    pa_05_cross_file = os.path.abspath(soi_pa_folder + pa_05_cross_file)
    # Inputting data on net income/loss by industry from "**pa01.xls":
    book_01 = xlrd.open_workbook(pa_01_file)
    sheet_01 = book_01.sheet_by_index(0)
    num_rows = sheet_01.nrows
    # The data to be extracted:
    cols_01 = ["Total net income", "Total net loss", "Depreciation"]
    num_cols = sheet_01.ncols
    start_col = search_ws(sheet_01, "All\nindustries", 20)[1]
    data_01 = pd.DataFrame(np.zeros((num_cols-start_col,3)), columns = cols_01)
    # Extracting the data:
    for i in xrange(0, num_rows):
        if("total net income" in str(sheet_01.cell_value(i,0)).lower()):
            data_01["Total net income"] = sheet_01.row_values(i+1,start_col)
            data_01["Total net loss"] = sheet_01.row_values(i+2,start_col)
            break
        if("depreciation" in str(sheet_01.cell_value(i,0)).lower()):
            data_01["Depreciation"] = sheet_01.row_values(i,start_col)
    # Data is in thousands of dollars:
    data_01 = data_01 * 1000
    # Inputting data on depreciable fixed assets, inventories, and land:
    book_03 = xlrd.open_workbook(pa_03_file)
    sheet_03 = book_03.sheet_by_index(0)
    # Finding the relevant details about the table, e.g. dimensions:
    cur_rows = sheet_03.nrows
    # The following categories of data to be extracted:
    cols_03 = ["Depreciable assets (Net)", "Accumulated depreciation (Net)", 
               "Inventories (Net)", "Land (Net)", 
               "Depreciable assets (Income)", 
               "Accumulated depreciation (Income)", "Inventories (Income)",
               "Land (Income)"]
    # The more general column names that are used in the input file:
    gen_03 = ["Depreciable assets", "Accumulated depreciation", 
                    "Inventories", "Land"]
    # The data to be extracted on partnerships as a whole:
    tot_data_03 = [None]*len(gen_03)
    # The data to be extracted on partnerships with income:
    inc_data_03 = [None]*len(gen_03)
    # Extracting the data (note that the rows with total data appear first):
    for i in xrange(0, len(gen_03)):
        for row1 in xrange(0, cur_rows):
            if(gen_03[i].lower() in str(sheet_03.cell_value(row1,0)).lower()):
                tot_data_03[i] = sheet_03.row_values(row1,1)
                for row2 in xrange(row1+1, cur_rows):
                    cur_cell = str(sheet_03.cell_value(row2,0)).lower()
                    if(gen_03[i].lower() in cur_cell):
                        inc_data_03[i] = sheet_03.row_values(row2,1)
                        break
                break
    # Reformatting the data:
    data_03 = pd.concat([pd.DataFrame(tot_data_03).T,
                         pd.DataFrame(inc_data_03).T], axis = 1)
    # Inputting data on income/loss by industry and sector.
    book_05 = xlrd.open_workbook(pa_05_file)
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
    cols_05 = ["Corporate general partners", 
               "Corporate limited partners",
               "Individual general partners",
               "Individual limited partners",
               "Partnership general partners",
               "Partnership limited partners",
               "Tax-exempt organization general partners",
               "Tax-exempt organization limited partners",
               "Nominee and other general partners", 
               "Nominee and other limited partners"]
    # Extracting the relevant data:
    data_05 = [None]*len(cols_05)
    for i in xrange(0, len(cols_05)):
        for row in xrange(0, cur_rows):
            if(cols_05[i].lower() in str(sheet_05.cell_value(row,0)).lower()):
                data_05[i] = sheet_05.row_values(row,1)
                break
    # Reformatting the data:
    data_05 = pd.DataFrame(data_05).T
    # Data is in thousands of dollars:
    data_01 = data_01 * 1000
    data_03 = data_03 * 1000
    data_05 = data_05 * 1000
    # Reading in the crosswalks between the columns and the NAICS codes:
    pa01cross = pd.read_csv(pa_01_cross_file)
    pa03cross = pd.read_csv(pa_03_cross_file)
    pa05cross = pd.read_csv(pa_05_cross_file)
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
        # Add data to industries in the tree based on fraction of codes shared:
        for i in xrange(0, len(cur_cross["NAICS Code:"])):
            if pd.isnull(cur_cross["NAICS Code:"][i]):
                continue
            cur_codes = str(cur_cross["NAICS Code:"][i]).split(".")
            num_found = 0
            for k in xrange(0, len(cur_codes)):
                cur_codes[k] = int(cur_codes[k])
            for j in xrange(0, len(data_tree.enum_inds)):
                cur_ind = data_tree.enum_inds[enum_index]
                for k in cur_codes:
                    for l in cur_ind.data.dfs["Codes:"][0]:
                        if(k == l):
                            cur_ind.data.dfs[cur_name] += np.array(
                                            cur_data.iloc[i,:])/len(cur_codes
                                            )
                            num_found += 1
                enum_index = (enum_index+1) % len(data_tree.enum_inds)
                if(num_found == len(cur_codes)):
                        break

'''
Reading in the SOI Tax Stats-Proprietorship Data:
'''
def load_soi_proprietor_data(data_tree, data_folder):
    prop_folder = os.path.abspath(data_folder + "\\SOI_Proprietorships")
    # Finding the "\**sp01br" file in the proprietorships folder:
    for i in os.listdir(prop_folder):
        if(i[2:] == "sp01br.xls"):
            prop_year = "20" + i[:2]
            sp01brfile = "\\" + prop_year[2:] + "sp01br.xls"
            sp01brfile = os.path.abspath(prop_folder + sp01brfile)
            sp01brfile_cross = "\\" + prop_year[2:] + "sp01br_Crosswalk.csv"
            sp01brfile_cross = os.path.abspath(prop_folder + sp01brfile_cross)
    # Opening nonfarm proprietor data:
    cur_wb = xlrd.open_workbook(sp01brfile)
    cur_ws = cur_wb.sheet_by_index(0)
    cur_cross = pd.read_csv(sp01brfile_cross)
    # Finding the relevant positions in worksheet:
    pos1 = search_ws(cur_ws,"Industrial sector",20, True, [0,0], True)
    pos2 = search_ws(cur_ws,"Depreciation\ndeduction",20)
    pos3 = search_ws(cur_ws,"Depreciation\ndeduction",20, True, np.array(pos2) + np.array([0,1]))
    #
    for i in data_tree.enum_inds:
        i.append_dfs(("soi_prop", pd.DataFrame(np.zeros((1,1)),
                                    columns = ["Depreciation Deductions"])))
    #
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
                                data_tree.enum_inds[count2].data.dfs["soi_prop"]["Depreciation Deductions"][0] += 1000 * (cur_ws.cell_value(i,pos2[1]) + cur_ws.cell_value(i,pos3[1]))/len(cur_codes)
                    count2 = (count2+1) % len(data_tree.enum_inds)
                break
            count1 = (count1+1) % cur_cross.shape[0]
    
    #Load Farm Proprietorship data:
    #farm_cols = ["R_p", "R_sp", "Q_p", "Q_sp", "A_p", "A_sp"]
    farm_data = pd.read_csv(os.path.abspath(prop_folder + "\\Farm_Data.csv"))
    new_farm_cols = ["Land", "FA"]
    #
    for i in data_tree.enum_inds:
        i.append_dfs(("farm_prop", pd.DataFrame(np.zeros((1,len(new_farm_cols))), columns=new_farm_cols)))
    #
    land_mult = (farm_data["R_sp"][0] + farm_data["Q_sp"][0]) * (float(farm_data["A_sp"][0])/farm_data["A_p"][0])
    total = farm_data.iloc[0,0] + farm_data.iloc[0,2]
    total_pa_land = 0
    total_pa = 0
    cur_codes = [111,112]
    proportions = np.zeros(len(cur_codes))
    for i in xrange(0, len(cur_codes)):
        cur_ind = find_naics(data_tree, cur_codes[i])
        total_pa_land += cur_ind.data.dfs["PA_assets"]["Land (Net)"][0]
        total_pa += cur_ind.data.dfs["PA_assets"]["Land (Net)"][0] + cur_ind.data.dfs["PA_assets"]["Depreciable assets (Net)"][0]
        proportions[i] = cur_ind.data.dfs["PA_assets"]["Land (Net)"][0] + cur_ind.data.dfs["PA_assets"]["Depreciable assets (Net)"][0]
    #
    if sum(proportions) != 0:
        proportions = proportions/sum(proportions)
    else:
        for i in len(proportions):
            proportions[i] = 1.0/len(proportions)
    #
    for i in xrange(0,len(cur_codes)):
        cur_ind = find_naics(data_tree, cur_codes[i])
        cur_ind.data.dfs["farm_prop"]["Land"][0] = land_mult * cur_ind.data.dfs["PA_assets"]["Land (Net)"][0]/total_pa
        cur_ind.data.dfs["farm_prop"]["FA"][0] = (proportions[i]*total) - cur_ind.data.dfs["farm_prop"]["Land"][0]


'''
Create an output tree containing only the final data on FA, INV, and LAND.
'''
def summary_tree(data_tree, data_folder):
    all_sectors = ["C Corporations", "S Corporations", "Corporate general partners", 
                   "Corporate limited partners",
                   "Individual general partners", "Individual limited partners",
                   "Partnership general partners", "Partnership limited partners",
                   "Tax-exempt organization general partners",
                   "Tax-exempt organization limited partners",
                   "Nominee and other general partners", 
                   "Nominee and other limited partners", "Sole Proprietors"]
    
    pa_types = data_tree.enum_inds[0].data.dfs["PA_types"].columns.values.tolist()
    
    #
    output_tree = load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    
    for i in output_tree.enum_inds:
        i.append_dfs(("FA",pd.DataFrame(np.zeros((1, len(all_sectors))),columns = all_sectors)))
        i.append_dfs(("INV",pd.DataFrame(np.zeros((1, len(all_sectors))),columns = all_sectors)))
        i.append_dfs(("LAND",pd.DataFrame(np.zeros((1, len(all_sectors))),columns = all_sectors)))
    
    for i in range(0, len(output_tree.enum_inds)):
        #
        cur_data = data_tree.enum_inds[i].data
        out_data = output_tree.enum_inds[i].data
        #
        partner_sum = sum(data_tree.enum_inds[i].data.dfs["PA_types"].iloc[0,:])
        #
        for j in range(0, len(all_sectors)):
            sector = all_sectors[j]
            #
            if sector == "C Corporations":
                out_data.dfs["FA"][sector][0] = data_tree.enum_inds[i].data.dfs["c_corps"]["Depreciable Assets"][0]
                out_data.dfs["INV"][sector][0] = data_tree.enum_inds[i].data.dfs["c_corps"]["Inventories"][0]
                out_data.dfs["LAND"][sector][0] = data_tree.enum_inds[i].data.dfs["c_corps"]["Land"][0]
            elif sector == "S Corporations":
                out_data.dfs["FA"][sector][0] = data_tree.enum_inds[i].data.dfs["s_corps"]["Depreciable Assets"][0]
                out_data.dfs["INV"][sector][0] = data_tree.enum_inds[i].data.dfs["s_corps"]["Inventories"][0]
                out_data.dfs["LAND"][sector][0] = data_tree.enum_inds[i].data.dfs["s_corps"]["Land"][0]
            elif sector in pa_types:
                if partner_sum != 0:
                    ratio = abs(cur_data.dfs["PA_types"][sector][0])/partner_sum
                else:
                    ratio = abs(1.0/cur_data.dfs["PA_types"].shape[0])
                out_data.dfs["FA"][sector][0] = abs(ratio*cur_data.dfs["PA_assets"]["Depreciable assets (Net)"][0])
                out_data.dfs["INV"][sector][0] = abs(ratio*cur_data.dfs["PA_assets"]["Inventories (Net)"][0])
                out_data.dfs["LAND"][sector][0] = abs(ratio*cur_data.dfs["PA_assets"]["Land (Net)"][0])
            elif sector == "Sole Proprietors":
                if cur_data.dfs["PA_inc/loss"]["Depreciation"][0] != 0:
                    ratio = abs(cur_data.dfs["soi_prop"]["Depreciation Deductions"][0]/cur_data.dfs["PA_inc/loss"]["Depreciation"][0])
                else:
                    ratio = 0.0
                out_data.dfs["FA"][sector][0] = abs(ratio*cur_data.dfs["PA_assets"]["Depreciable assets (Net)"][0] + cur_data.dfs["farm_prop"]["FA"][0])
                out_data.dfs["INV"][sector][0] = abs(ratio*cur_data.dfs["PA_assets"]["Inventories (Net)"][0] + cur_data.dfs["farm_prop"]["Land"][0])
                out_data.dfs["LAND"][sector][0] = abs(ratio*cur_data.dfs["PA_assets"]["Land (Net)"][0])
    return output_tree

'''
-------------------------------------------------------------------------------
def find_naics: Defines a function that finds a naics code in a tree.
-------------------------------------------------------------------------------
'''
def find_naics(tree, term):
    for i in tree.enum_inds:
        for j in xrange(0, i.data.dfs["Codes:"].shape[0]):
            if(term == i.data.dfs["Codes:"].iloc[j,0]):
                return i
    return None
    
'''
-------------------------------------------------------------------------------
def find_naics: Defines a function that searches through an excel file for
    a specified term.
-------------------------------------------------------------------------------
'''
def search_ws(sheet, search_term, distance, warnings = True, origin = [0,0], exact = False):
    '''
    Parameters: sheet - The worksheet to be searched through.
                entry - What is being searched for in the worksheet.
                        Numbers must be written with at least one decimal 
                        place, e.g. 15.0, 0.0, 21.74.
                distance - Search up to and through this diagonal.

    Returns:    A vector of the position of the first entry found.
                If not found then [-1,-1] is returned.
    '''
    final_search = ((distance+1)*distance)/2
    current_diagonal = 1
    total_columns  = sheet.ncols
    total_rows  = sheet.nrows
    for n in xrange(0, final_search):
        if ((current_diagonal+1)*current_diagonal)/2 < n+1:
            current_diagonal += 1
        
        i = ((current_diagonal+1)*current_diagonal)/2 - (n+1)
        j = current_diagonal - i - 1
        
        if j + origin[1] >= total_columns:
            continue
        if i + origin[0] >= total_rows:
            continue
        if(exact):
            if str(search_term).lower() == str(sheet.cell_value(i+origin[0],j+origin[1])).lower():
                return [i+origin[0],j+origin[1]]
        elif(not exact):
            if str(search_term).lower() in str(sheet.cell_value(i+origin[0],j+origin[1])).lower():
                return [i+origin[0],j+origin[1]]
    if warnings:
        print "Warning: No such search entry found in the specified search space."
        print "Check sample worksheet and consider changing distance input."
    
    return [-1,-1]

def pop_back(tree, pd_list):
    for corps in pd_list:
        cur_dfs = None
        was_empty = [False]*len(tree.enum_inds)
        count = len(tree.enum_inds)-1
        header = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
        # Working backwards through the tree
        for i in range(1, len(tree.enum_inds)):
            cur_dfs = tree.enum_inds[count].data.dfs[corps]
            par_dfs = tree.enum_inds[tree.par[count]].data.dfs[corps]
            cur_dfs_filled = False
            if sum((cur_dfs != pd.DataFrame(np.zeros((1,len(header))), columns = header)).iloc[0]) == 0:
                cur_dfs_filled = False
            else:
                cur_dfs_filled = True
            if sum((par_dfs != pd.DataFrame(np.zeros((1,len(header))), columns = header)).iloc[0]) == 0:
                was_empty[tree.par[count]] = True
            if cur_dfs_filled and was_empty[tree.par[count]]:
                tree.enum_inds[tree.par[count]].data.dfs[corps] += cur_dfs
            count = count - 1


def pop_forward(tree, pd_list, blueprint = None, blue_tree = None, sub_print = None):
    if blueprint == None:
        for corps in pd_list:
            cur_dfs = None
            header = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
            for i in range(0, len(tree.enum_inds)):
                if tree.enum_inds[i].sub_ind != []:
                    cur_ind = tree.enum_inds[i]
                    cur_dfs = cur_ind.data.dfs[corps]
                    sum_dfs = pd.DataFrame(np.zeros((1,len(header))), columns = header)
                    proportion = 1
                    for j in cur_ind.sub_ind:
                        sum_dfs += j.data.dfs[corps]
                    for j in range(0, len(header)):
                        if sum_dfs.iloc[0,j] == 0:
                            for k in cur_ind.sub_ind:
                                k.data.dfs[corps].iloc[0,j] = cur_dfs.iloc[0,j]/len(cur_ind.sub_ind)
                        else:
                            proportion = cur_dfs.iloc[0,j]/sum_dfs.iloc[0,j]
                            for k in cur_ind.sub_ind:
                                k.data.dfs[corps].iloc[0,j] *= proportion
    else:
        if(blue_tree == None):
            blue_tree = tree
        for corps in pd_list:
            cur_dfs = None
            header1 = tree.enum_inds[0].data.dfs[corps].columns.values.tolist()
            if sub_print == None:
                header2 = blue_tree.enum_inds[0].data.dfs[blueprint].columns.values.tolist()
            else:
                header2 = sub_print
            for i in range(0, len(tree.enum_inds)):
                if tree.enum_inds[i].sub_ind != []:
                    cur_ind = tree.enum_inds[i]
                    cur_ind_blue = blue_tree.enum_inds[i]
                    cur_dfs = cur_ind.data.dfs[corps]
                    sum_dfs = pd.DataFrame(np.zeros((1,len(header1))), columns = header1)
                    proportions = np.zeros(len(tree.enum_inds[i].sub_ind))
                    for j in range(0, len(cur_ind.sub_ind)):
                        sum_dfs += cur_ind.sub_ind[j].data.dfs[corps]
                        for k in header2: #range(0, len(header2)):
                            proportions[j] += cur_ind_blue.sub_ind[j].data.dfs[blueprint][k][0] #
                    if sum(proportions) != 0:
                        proportions = proportions/sum(proportions)
                    else:
                        for k in range(0, len(proportions)):
                            proportions[k] = 1.0/len(proportions)
                        
                    for j in range(0, len(cur_ind.sub_ind)):
                        for k in range(0, len(header1)):
                            change = proportions[j] * (cur_dfs.iloc[0,k]-sum_dfs.iloc[0,k])
                            cur_ind.sub_ind[j].data.dfs[corps].iloc[0,k] += change

def get_proportions(codes, tree, df_name, sub_df = None):
    proportions = np.zeros(len(codes))
    indexes = np.zeros(len(codes))
    enum_index = len(tree.enum_inds) - 1
    for i in xrange(0, len(codes)):
        cur_code = codes[i]
        for j in xrange(0, len(tree.enum_inds)):
            enum_index = (enum_index+1) % len(tree.enum_inds)
            cur_ind = tree.enum_inds[enum_index]
            if(compare_codes([cur_code], cur_ind.data.dfs["Codes:"].iloc[:,0]) != 0):
                cur_sum = 0
                if sub_df != None:
                    for k in sub_df:
                        cur_sum += cur_ind.data.dfs[df_name][k][0]
                else:
                    cur_sum = sum(cur_ind.data.dfs[df_name].iloc[0,:])
                proportions[i] = cur_sum
                indexes[i] = enum_index
                break
    #
    if(sum(proportions) == 0):
        proportions = np.array([1.0]*len(proportions))/len(proportions)
        print "BOO"
        return pd.DataFrame(np.vstack((indexes, proportions)))
    #
    proportions = proportions/sum(proportions)
    return pd.DataFrame(np.vstack((indexes, proportions)))


'''
Reads in the detailnonres_stk1.xlsx BEA file:
'''
def read_bea(output_tree, data_folder):
    # The directory with BEA data:
    bea_folder = os.path.abspath(data_folder + "\\BEA")
    # Opening BEA's excel file on depreciable assets by industry:
    bea_book = xlrd.open_workbook(os.path.abspath(
                                    bea_folder + "\\detailnonres_stk1.xlsx"))
    sht_names = bea_book.sheet_names()
    num_shts = bea_book.nsheets
    # Opening "readme" sheet:
    try:
        bea_readme = bea_book.sheet_by_name("readme")
    except xlrd.XLRDError:
        bea_readme = bea_book.sheet_by_index(0)
    # Finding relevant positions in the readme sheet:
    sht_pos = search_ws(bea_readme, "Industry Title", 25, False)
    if(sht_pos == [-1,-1]):
        sht_pos = search_ws(bea_readme, "bea code", 25, False, [0,0], True)
        sht_pos[1] = sht_pos[1] - 1
    if(sht_pos == [-1,-1]):
        print "Error in reading BEA fixed asset \"readme\" sheet."
        return None
    cur_row = sht_pos[0] + 1
    cur_col = sht_pos[1]
    # Finding the number of industries listed (includes ones without bea codes):
    number_of_industries = 0
    while cur_row < bea_readme.nrows:
        if(str(bea_readme.cell_value(cur_row, cur_col)) != ""):
            number_of_industries += 1
        cur_row += 1
    # Making a list of BEA codes based on the names of the worksheets:
    bea_codes1 = np.zeros(num_shts-1, dtype=object)
    for index in xrange(1, num_shts):
        bea_codes1[index-1] = str(sht_names[index])
    # Making a list of BEA codes based on info in the readme sheet:
    array_index = 0
    cur_row = sht_pos[0] + 1
    cur_col = sht_pos[1]
    bea_codes2 = np.zeros(number_of_industries, dtype=object)
    while cur_row < bea_readme.nrows:
        if(str(bea_readme.cell_value(cur_row, cur_col)) != ""):
            bea_codes2[array_index] = str(bea_readme.cell_value(cur_row, cur_col+1)).replace("\xa0", " ").strip()
            array_index += 1
        cur_row += 1
    # Reading in a list of the assets in the BEA file:
    list_file = os.path.abspath(bea_folder + "\\detailnonres_list.csv")
    asset_list = pd.read_csv(list_file)
    for i in xrange(0, asset_list.shape[0]):
        asset_list.iloc[i,0] = asset_list.iloc[i,0].replace("\xa0", " ").strip()
    
    # Reading in the corresponding naics codes:
    naics_file = os.path.abspath(bea_folder + "\\detailnonres_naics.csv")
    naics_cross = pd.read_csv(naics_file).replace("\xa0", " ")
    for i in xrange(0, naics_cross.shape[0]):
        naics_cross["Industry"][i] = naics_cross["Industry"][i].replace("\xa0", " ").strip()
    # Creating a chart cross-referencing industry names, BEA and NAICS codes.
    chart_cols = ["Industry","BEA Code","NAICS Code"]
    bea_chart = pd.DataFrame(np.zeros(shape=(num_shts-2,3), dtype=object), columns = chart_cols)
    cur_row = sht_pos[0] + 1
    cur_col = sht_pos[1]
    num_naics = naics_cross.shape[0]
    # Filling chart with naics codes that are in both lists and the crosswalk:
    naics_counter = 0
    for i in range(0, num_shts-2):
        for cur_row in range(sht_pos[0]+1, bea_readme.nrows):
            if(str(bea_codes1[i]) == str(bea_readme.cell_value(cur_row,cur_col+1))):
                bea_chart["Industry"][i] = str(bea_readme.cell_value(cur_row,cur_col)).replace('\xa0', ' ').strip()
                bea_chart["BEA Code"][i] = str(bea_readme.cell_value(cur_row,cur_col+1))
                for k in xrange(0, num_naics):
                    naics_counter = (naics_counter+1) % num_naics
                    if(naics_cross["Industry"][naics_counter] == bea_chart["Industry"][i]):
                        bea_chart["NAICS Code"][i] = naics_cross["NAICS"][naics_counter]
                        break
                break
            # If they match except one has ".0" at the end:
            elif(str(bea_codes1[i]) == str(bea_readme.cell_value(cur_row, cur_col+1))[:-2]):
                bea_chart["Industry"][i] = str(bea_readme.cell_value(cur_row, cur_col)).replace('\xa0', ' ').strip()
                bea_chart["BEA Code"][i] = str(bea_readme.cell_value(cur_row, cur_col+1))[:-2]
                for k in xrange(0, num_naics):
                    naics_counter = (naics_counter+1) % num_naics
                    if(naics_cross["Industry"][naics_counter] == bea_chart["Industry"][i]):
                        bea_chart["NAICS Code"][i] = naics_cross["NAICS"][naics_counter]
                        break
                break
    # Initializing the table of assets:
    #cur_sht = bea_book.sheet_by_name(bea_chart["BEA Code"][0])
    #sht_pos = search_ws(cur_sht, "asset codes", 25, False)
    bea_table = pd.DataFrame(np.zeros((asset_list.shape[0],bea_chart.shape[0])), columns = bea_chart["BEA Code"])
    # For each industry, calculating 
    for i in bea_chart["BEA Code"]:
        cur_sht = bea_book.sheet_by_name(i)
        sht_pos = search_ws(cur_sht, "asset codes", 25, False)
        for j in xrange(0, len(asset_list)): #xrange(sht_pos[0]+2, cur_sht.nrows):
            cur_asset = asset_list.iloc[j,0]
            for k in xrange(sht_pos[0]+2, cur_sht.nrows):
                cur_cell = str(cur_sht.cell_value(k, sht_pos[1]+1)).replace("\xa0", " ").strip()
                if(cur_asset == cur_cell):
                    bea_table[i][j] = float(cur_sht.cell_value(k, cur_sht.ncols-1))
        #bea_table[i] = np.array(cur_sht.col_values(cur_sht.ncols-1, sht_pos[0]+2, cur_sht.nrows))
    # The dollar amounts are in millions:
    bea_table = bea_table.convert_objects(convert_numeric=True).fillna(0)
    bea_table = bea_table * 1000000
    # Breaking down by corporate tax status:
    corp_types = ["C Corporations",
                  "Corporate general partners", 
                  "Corporate limited partners"]
    non_corp_types = ["S Corporations",
                      "Individual general partners",
                      "Individual limited partners",
                      "Partnership general partners",
                      "Partnership limited partners",
                      "Tax-exempt organization general partners",
                      "Tax-exempt organization limited partners",
                      "Nominee and other general partners", 
                      "Nominee and other limited partners",
                      "Sole Proprietors"]
    # Initialize tree for assets data:
    asset_tree = load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    for i in xrange(0, len(asset_tree.enum_inds)):
        asset_tree.enum_inds[i].data.append(("All", pd.DataFrame(np.zeros((1, asset_list.shape[0])), columns = asset_list.iloc[:,0])))
        asset_tree.enum_inds[i].data.append(("Corp", pd.DataFrame(np.zeros((1, asset_list.shape[0])), columns = asset_list.iloc[:,0])))
        asset_tree.enum_inds[i].data.append(("Non-Corp", pd.DataFrame(np.zeros((1, asset_list.shape[0])), columns = asset_list.iloc[:,0])))
    # Fill in data from BEA's fixed asset table:
    enum_index = len(output_tree.enum_inds) - 1
    for i in xrange(0, bea_table.shape[1]):
        cur_codes = bea_chart["NAICS Code"][i].split(".")
        tot_share = 0
        all_proportions = get_proportions(cur_codes, output_tree, "FA").iloc[1,:]
        corp_proportions = get_proportions(cur_codes, output_tree, "FA", corp_types).iloc[1,:]
        non_corp_proportions = get_proportions(cur_codes, output_tree, "FA", non_corp_types).iloc[1,:]
        for code_index in xrange(0, len(cur_codes)):
            for j in xrange(0, len(asset_tree.enum_inds)):
                enum_index = (enum_index+1) % len(asset_tree.enum_inds)
                if(sum(output_tree.enum_inds[enum_index].data.dfs["FA"].iloc[0,:]) == 0):
                    continue
                all_ratio = 1.0 #sum(output_tree.enum_inds[enum_index].data.dfs["FA"].iloc[0,:])/sum(bea_table.iloc[:,i])
                corp_ratio = 0.0
                non_corp_ratio = 0.0
                for category in corp_types:
                    corp_ratio += output_tree.enum_inds[enum_index].data.dfs["FA"][category][0]/sum(output_tree.enum_inds[enum_index].data.dfs["FA"].iloc[0,:])
                for category in non_corp_types:
                    non_corp_ratio += output_tree.enum_inds[enum_index].data.dfs["FA"][category][0]/sum(output_tree.enum_inds[enum_index].data.dfs["FA"].iloc[0,:])
                cur_data = asset_tree.enum_inds[enum_index].data
                ind_codes = cur_data.dfs["Codes:"].iloc[:,0]
                share = compare_codes(cur_codes, ind_codes)
                tot_share += share
                if(share == 0):
                    continue
                for k in xrange(0, asset_tree.enum_inds[0].data.dfs["All"].shape[1]):
                    cur_data.dfs["All"].iloc[0,k] = bea_table.iloc[k,i] * all_ratio * all_proportions[code_index]
                    cur_data.dfs["Corp"].iloc[0,k] = bea_table.iloc[k,i] * corp_ratio * corp_proportions[code_index]
                    cur_data.dfs["Non-Corp"].iloc[0,k] = bea_table.iloc[k,i] * non_corp_ratio * non_corp_proportions[code_index]
                break
            if(tot_share == 1):
                break
    return asset_tree


def read_inventories(output_tree, data_folder):
    # The directory with inventory data:
    inv_folder = os.path.abspath(data_folder + "\\Inventories")
    # Opening BEA's excel file on depreciable assets by industry:
    inv_book = xlrd.open_workbook(os.path.abspath(
                                    inv_folder + "\\Inventories.xls"))
    sht0 = inv_book.sheet_by_index(0)
    num_rows = sht0.nrows
    num_cols = sht0.ncols
    #Find the starting index in worksheet.
    cur_index = search_ws(sht0, 1, 25, True, [0,0], True)
    check_index = search_ws(sht0, "line", 20)
    if(cur_index[1] != check_index[1]):
        print "ERROR"
    # Breaking down by corporate tax status:
    corp_types = ["C Corporations",
                  "Corporate general partners", 
                  "Corporate limited partners"]
    non_corp_types = ["S Corporations",
                      "Individual general partners",
                      "Individual limited partners",
                      "Partnership general partners",
                      "Partnership limited partners",
                      "Tax-exempt organization general partners",
                      "Tax-exempt organization limited partners",
                      "Nominee and other general partners", 
                      "Nominee and other limited partners",
                      "Sole Proprietors"]
    # Reading in the crosswalk:
    inv_cross = pd.read_csv(os.path.abspath(
                                inv_folder + "\\Inventories_Crosswalk.csv"))
    # Creating a tree for the inventory data:
    inv_tree = load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    #
    data_cols = ["All", "Corp", "Non-Corp"]
    for i in inv_tree.enum_inds:
        i.data.append(("Inventories", pd.DataFrame(np.zeros((1, len(data_cols))), columns = data_cols)))
    #
    inv_data = np.zeros(inv_cross.shape[0])
    #
    cross_index = 0
    for i in xrange(cur_index[0], num_rows):
        if(cross_index >= inv_cross.shape[0]):
            break
        cur_list = str(sht0.cell_value(i, cur_index[1])).strip()
        cur_name = str(sht0.cell_value(i, cur_index[1]+1)).strip()
        checks = (str(cur_list) == str(inv_cross["List"][cross_index])) and (str(cur_name) == str(inv_cross["Industry"][cross_index]))
        if(checks):
            cross_index += 1
            try:
                cur_value = float(sht0.cell_value(i, num_cols-1))
            except ValueError:
                continue
            inv_data[cross_index-1] = cur_value
            # Data is in billions:
            inv_data[cross_index-1] = (10**9) * inv_data[cross_index-1]
    #
    for i in xrange(0, inv_cross.shape[0]):
        cur_codes = inv_cross["NAICS"][i].strip().split(".")
        proportions = get_proportions(cur_codes, output_tree, "INV")
        for j in xrange(0, proportions.shape[1]):
            cur_ind = inv_tree.enum_inds[int(proportions.iloc[0,j])]
            prev_data = output_tree.enum_inds[int(proportions.iloc[0,j])].data.dfs["INV"]
            if(sum(prev_data.iloc[0, :]) != 0):
                cur_dfs = (prev_data/sum(prev_data.iloc[0,:])) * (inv_data[i] * proportions.iloc[1,j])
                cur_ind.data.dfs["Inventories"]["All"] += sum(cur_dfs.iloc[0,:])
                for k in corp_types:
                    cur_ind.data.dfs["Inventories"]["Corp"] += cur_dfs[k][0]
                for k in non_corp_types:
                    cur_ind.data.dfs["Inventories"]["Non-Corp"] += cur_dfs[k][0]
    #
    return inv_tree


def read_land(output_tree, data_folder):
    land_folder = os.path.abspath(data_folder + "\\Land")
    land_file = os.path.abspath(land_folder + "\\Fin_Accounts-Land.csv")
    land_data = pd.read_csv(land_file)
    # Data is in billions:
    land_data = (10**9) * land_data
    corp_types = ["C Corporations",
                  "Corporate general partners", 
                  "Corporate limited partners"]
    non_corp_types = ["S Corporations",
                      "Individual general partners",
                      "Individual limited partners",
                      "Partnership general partners",
                      "Partnership limited partners",
                      "Tax-exempt organization general partners",
                      "Tax-exempt organization limited partners",
                      "Nominee and other general partners", 
                      "Nominee and other limited partners",
                      "Sole Proprietors"]
    land_tree = load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    df_cols = ["All", "Corp", "Non-Corp"]
    for i in land_tree.enum_inds:
        i.data.append(("Land", pd.DataFrame(np.zeros((1,len(df_cols))), columns = df_cols)))
    corp_sum = 0.0
    non_corp_sum = 0.0
    for i in corp_types:
        corp_sum += output_tree.enum_inds[0].data.dfs["LAND"][i][0]
    for i in non_corp_types:
        non_corp_sum += output_tree.enum_inds[0].data.dfs["LAND"][i][0]
    if corp_sum + non_corp_sum == 0:
        return land_tree
    #corp_proportion = corp_sum / (corp_sum + non_corp_sum)
    #non_corp_proportion = non_corp_sum / (corp_sum + non_corp_sum)
    land_tree.enum_inds[0].data.dfs["Land"]["Corp"][0] = land_data["Corporate"][0]
    land_tree.enum_inds[0].data.dfs["Land"]["Non-Corp"][0] = land_data["Non-Corporate"][0]
    land_tree.enum_inds[0].data.dfs["Land"]["All"][0] = land_data["Corporate"][0] + land_data["Non-Corporate"][0]
    return land_tree

def calc_depr_rates(asset_tree, inv_tree, land_tree, data_folder):
    # The directory with depreciation rates data:
    depr_folder = os.path.abspath(data_folder + "\\Depreciation Rates")
    # Opening file containing depreciation rates by asset type:
    depr_econ = pd.read_csv(os.path.abspath(depr_folder + "\\Economic Depreciation Rates.csv"))
    depr_econ = depr_econ.fillna(1)
    econ_assets = depr_econ["Asset"]
    econ_rates = depr_econ["Economic Depreciation Rate"]
    #
    types = ["All", "Corp", "Non-Corp"]
    # Initialize tree for depreciation rates:
    depr_tree = load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    for i in depr_tree.enum_inds:
        i.data.append(("Economic", pd.DataFrame(np.zeros((1,3)), columns = types)))
        #i.data.append(("Tax", pd.DataFrame(np.zeros((1,3)), columns = types)))
    
    for i in types:
        asset_list = asset_tree.enum_inds[0].data.dfs[i].columns.values.tolist()
        match = np.array([-1] * len(asset_list))
        for j in xrange(0, asset_tree.enum_inds[0].data.dfs[i].shape[1]):
            for k in xrange(0, len(econ_assets)):
                if str(asset_list[j]).strip() == str(econ_assets[k]).strip():
                    match[j] = k
        for j in xrange(0, len(depr_tree.enum_inds)):
            cur_sum = 0
            for k in xrange(0, len(asset_list)):
                if(match[k] == -1):
                    print k
                    continue
                cur_sum += asset_tree.enum_inds[j].data.dfs[i].iloc[0,k] * econ_rates[match[k]]
            if(sum(asset_tree.enum_inds[j].data.dfs[i].iloc[0,:]) != 0):
                depr_tree.enum_inds[j].data.dfs["Economic"][i][0] = cur_sum/sum(asset_tree.enum_inds[j].data.dfs[i].iloc[0,:])
            else:
                depr_tree.enum_inds[j].data.dfs["Economic"][i][0] = 0
        # Inventories and land have an approximately zero depreciation rate:
        for j in xrange(0, len(depr_tree.enum_inds)):
            tot_assets = sum(asset_tree.enum_inds[j].data.dfs["All"].iloc[0,:])
            tot_inv = inv_tree.enum_inds[j].data.dfs["Inventories"]["All"][0]
            tot_land = land_tree.enum_inds[j].data.dfs["Land"]["All"][0]
            if(tot_assets+tot_inv+tot_land == 0):
                continue
            ratio = tot_assets / (tot_assets + tot_inv + tot_land)
            #
            cur_df = depr_tree.enum_inds[j].data.dfs["Economic"]
            for k in cur_df:
                cur_df[k][0] = ratio * cur_df[k][0]
    return depr_tree


def calc_tax_depr_rates(asset_tree, inv_tree, land_tree, data_folder):
    # The directory with depreciation rates data:
    depr_folder = os.path.abspath(data_folder + "\\Depreciation Rates")
    #
    tax_file = os.path.abspath(depr_folder + "\\BEA_IRS_Crosswalk.csv")
    tax_data = pd.read_csv(tax_file).fillna(0)
    tax_assets = tax_data["Asset Type"]
    for i in xrange(0, len(tax_assets)):
        tax_assets[i] = str(tax_assets[i]).replace("\xa0", " ").strip()
    #
    r = .05
    #
    #tax_cols = {"GDS 200%": 2, "GDS 150%": 1.5, "GDS SL": 1.0, "ADS SL": 1.0}
    tax_gds_mthds = {"GDS 200%": 2.0, "GDS 150%": 1.5, "GDS SL": 1.0}
    tax_ads_mthds = {"ADS SL": 1.0}
    tax_cols = tax_gds_mthds.keys() + tax_ads_mthds.keys()
    tax_systems = {"GDS": tax_gds_mthds, "ADS": tax_ads_mthds}
    tax_rates = pd.DataFrame(np.zeros((len(tax_assets),len(tax_cols))), columns = tax_cols)
    tax_rates["Asset"] = tax_assets
    # Compute the tax rates:
    for i in tax_systems:
        tax_yrs = tax_data[i]
        for j in tax_systems[i]:
            tax_b = tax_systems[i][j]
            tax_beta = tax_b/tax_yrs
            tax_star = tax_yrs * (1 - (1/tax_b))
            tax_z = ((tax_beta/(tax_beta+r))* (1-np.exp(-1*(tax_beta+r)*tax_star)))+ ((np.exp(-1*tax_beta*tax_star)* np.exp(-1*r*tax_star)-np.exp(-1*r*tax_yrs))/ ((tax_yrs-tax_star)*r))
            tax_z = (((tax_beta/(tax_beta+r))*
                      (1-np.exp(-1*(tax_beta+r)*tax_star))) 
                      + ((np.exp(-1*tax_beta*tax_star)/
                      ((tax_yrs-tax_star)*r))*
                      (np.exp(-1*r*tax_star)-np.exp(-1*r*tax_yrs))))
            tax_rates[j] = r/((1/tax_z)-1)
    tax_rates = tax_rates.fillna(0)
    #
    types = ["All", "Corp", "Non-Corp"]
    # Initialize tree for depreciation rates:
    depr_tree = load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    for i in depr_tree.enum_inds:
        for j in tax_systems:
            for k in tax_systems[j]:
                i.data.append((k, pd.DataFrame(np.zeros((1,3)), columns = types)))
    for i in depr_tree.enum_inds:
        i.data.append(("True Values", pd.DataFrame(np.zeros((1,3)), columns = types)))
    #
    for i in types:
        asset_list = asset_tree.enum_inds[0].data.dfs[i].columns.values.tolist()
        match = np.array([-1] * len(asset_list))
        for j in xrange(0, asset_tree.enum_inds[0].data.dfs[i].shape[1]):
            for k in xrange(0, len(tax_assets)):
                if str(asset_list[j]).strip() == str(tax_assets[k]).strip():
                    match[j] = k
        for j in xrange(0, len(depr_tree.enum_inds)):
            cur_ind = depr_tree.enum_inds[j]
            for k in tax_cols:
                cur_tax = cur_ind.data.dfs[k][i]
                cur_sum = 0.0
                for l in xrange(0, len(asset_list)):
                    if(match[l] == -1):
                        continue
                    cur_sum += asset_tree.enum_inds[j].data.dfs[i].iloc[0,l] * tax_rates[k][match[l]]        
                cur_tax[0] = cur_sum/sum(asset_tree.enum_inds[j].data.dfs[i].iloc[0,:])
                
                tot_assets = sum(asset_tree.enum_inds[j].data.dfs["All"].iloc[0,:])
                tot_inv = inv_tree.enum_inds[j].data.dfs["Inventories"]["All"][0]
                tot_land = land_tree.enum_inds[j].data.dfs["Land"]["All"][0]
                if(tot_assets+tot_inv+tot_land == 0):
                    continue
                ratio = tot_assets / (tot_assets + tot_inv + tot_land)
                cur_tax[0] = cur_tax[0] * ratio
    return depr_tree
    

def compare_codes(codes1, codes2):
    if(len(codes2) == 0):
        return 0
    num_match = 0.0
    for i in xrange(0, len(codes2)):
        for j in xrange(0, len(codes1)):
            if(str(codes2[i]) == str(codes1[j])):
                num_match += 1.0
    return float(num_match)/len(codes1)

def find_first_match(tree, codes):
    for i in tree.enum_inds:
        ind_codes = i.data.dfs["Codes:"]
        for j in xrange(0, len(codes)):
            for k in xrange(0, ind_codes.shape[0]):
                if(str(codes[j]) == str(ind_codes.iloc[k,0])):
                    return i
    return None
                    
def find_matches(tree, codes):
    matches = []
    for i in tree.enum_inds:
        ind_codes = i.data.dfs["Codes:"]
        is_match = False
        for j in xrange(0, len(codes)):
            for k in xrange(0, ind_codes.shape[0]):
                if(str(codes[j]) == str(ind_codes.iloc[k,0])):
                    matches.append[i]
                    is_match = True
                    break
            if(is_match):
                break










