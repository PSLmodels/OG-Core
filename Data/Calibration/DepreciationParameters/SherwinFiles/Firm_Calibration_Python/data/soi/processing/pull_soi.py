'''
-------------------------------------------------------------------------------
Last updated 5/26/2015
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
    Packages
-------------------------------------------------------------------------------
'''
import os.path
import numpy as np
import pandas as pd
import xlrd
#
import naics_processing as naics
# Directory names:
cur_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.dirname(cur_dir) + "\\data")
corp_dir = os.path.abspath(data_dir + "\\soi_corporate")
prop_dir = os.path.abspath(data_dir + "\\soi_proprietorships")
prt_dir = os.path.abspath(data_dir + "\\soi_partner")
# Names of dataframes created:
'''
"tot_corps"
"s_corps"
"c_corps"
"PA_inc_loss"
"PA_assets"
"PA_types"
'''
#

#
'''
-------------------------------------------------------------------------------
Reading in the SOI Tax Stats-Corporation Data:
(Note: SOI gives data for all corporations as well as for just s-corporations.
    The c-corporation data is inferred from this.)
-------------------------------------------------------------------------------
'''
def load_soi_tot_corp(data_tree = None, cols_dict = None, 
                      blue_tree = None, blueprint = None):
    """This function pulls SOI total corporate data.

    :param data_tree: A string to be converted?
    :returns: A bar formatted string?huh
    """
    if data_tree == None:
        data_tree = naics.generate_tree()
    # The aggregate 1120 filings data for all corporations:
    tot_corp_file = ""
    for i in os.listdir(corp_dir):
        if(i[4:] == "sb1.csv"):
            tot_corp_file = os.path.abspath(corp_dir + "\\" + i)
            break
    try:
        tot_corp_data = pd.read_csv(tot_corp_file).fillna(0)
    except IOError:
        print "IOError: Could not find tot-corp soi data file."
        return None
    # Listing the relevant columns that are being extracted from the dataset:
    if cols_dict == None:
        # Default:
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
    #
    naics.pop_back(tree=data_tree, df_list=["tot_corps"])
    naics.pop_forward(tree=data_tree, df_list=["tot_corps"],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree
    


def load_soi_s_corp(data_tree = None, cols_dict = None,
                    blue_tree = None, blueprint = None):
    if data_tree == None:
        data_tree = naics.generate_tree()
    # The aggregate 1120 filings data for all corporations:
    s_corp_file = ""
    for i in os.listdir(corp_dir):
        if(i[4:] == "sb3.csv"):
            s_corp_file = os.path.abspath(corp_dir + "\\" + i)
            break
    try:
        s_corp_data = pd.read_csv(s_corp_file).fillna(0)
    except IOError:
        print "IOError: Could not find s-corp soi data file."
        return None
    # Listing the relevant columns that are being extracted from the dataset:
    if cols_dict == None:
        # Default:
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
        i.append_dfs(("s_corps", pd.DataFrame(np.zeros((1,len(data_cols))),
                                                columns = data_cols)))
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
            cur_dfs.loc[0,j] = sum(indicators * s_corp_data[cols_dict[j]])
    # Default blueprint is tot_corps:
    if blueprint == None and "tot_corps" in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = "tot_corps"
    naics.pop_back(tree=data_tree, df_list=["s_corps"])
    naics.pop_forward(tree=data_tree, df_list=["s_corps"],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree

'''
Calculates the c-corp data from the tot and s corp data.
'''
def calc_c_corp(data_tree):
    for ind in data_tree.enum_inds:
        cur_tot = ind.data.dfs["tot_corps"]
        cur_s = ind.data.dfs["s_corps"]
        data_cols = cur_tot.columns.values.tolist()
        ind.append_dfs(("c_corps", pd.DataFrame(np.zeros((1,len(data_cols))),
                                                columns = data_cols)))
        ind.data.dfs["c_corps"] = cur_tot - cur_s
    


def load_pa_01_data(data_tree = None, blue_tree = None, blueprint = None):
    # Defining constants:
    pa_01_fctr = 10 ** 3
    #
    #if data_tree == None:
    #    data_tree = naics.generate_tree()
    # Names of the files with the partnership data:
    for i in os.listdir(prt_dir):
        if("pa01.xls" in i):
            pa_01_file = os.path.abspath(prt_dir + "\\" + i)
        elif("pa01_Crosswalk.csv" in i):
            pa_01_cross_file = os.path.abspath(prt_dir + "\\" + i)
    # Inputting data on net income/loss by industry from "**pa01.xls":
    book_01 = xlrd.open_workbook(pa_01_file)
    sheet_01 = book_01.sheet_by_index(0)
    num_rows = sheet_01.nrows
    # The data to be extracted:
    cols_01 = ["Total net income", "Total net loss", "Depreciation"]
    num_cols = sheet_01.ncols
    start_col = naics.search_ws(sheet_01, "All\nindustries", 20)[1]
    data_01 = pd.DataFrame(np.zeros((num_cols-start_col,3)), columns = cols_01)
    # Extracting the data:
    for i in xrange(0, num_rows):
        if("total net income" in str(sheet_01.cell_value(i,0)).lower()):
            data_01["Total net income"] = sheet_01.row_values(i+1,start_col)
            data_01["Total net loss"] = sheet_01.row_values(i+2,start_col)
            break
        if("depreciation" in str(sheet_01.cell_value(i,0)).lower()):
            data_01["Depreciation"] = sheet_01.row_values(i,start_col)
    #
    data_01 = data_01 * pa_01_fctr
    # Reading in the crosswalks between the columns and the NAICS codes:
    pa01cross = pd.read_csv(pa_01_cross_file)
    #
    data_tree = naics.load_data_with_cross(
                    data_tree = data_tree, data_df = data_01,
                    cross_df = pa01cross, df_name = "PA_inc_loss"
                    )
    #
    if blueprint == None and "tot_corps" in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = "tot_corps"
    naics.pop_back(tree=data_tree, df_list=["PA_inc_loss"])
    naics.pop_forward(tree=data_tree, df_list=["PA_inc_loss"],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree
    
    
def load_pa_03_data(data_tree = None, blue_tree = None, blueprint = None):
    # Defining constants:
    pa_03_fctr = 10 ** 3
    #
    if data_tree == None:
        data_tree = naics.generate_tree()
    #
    for i in os.listdir(prt_dir):
        if("pa03.xls" in i):
            pa_03_file = os.path.abspath(prt_dir + "\\" + i)
        elif("pa03_Crosswalk.csv" in i):
            pa_03_cross_file = os.path.abspath(prt_dir + "\\" + i)
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
    # Data is in the thousands:
    data_03 = data_03 * pa_03_fctr
    # Reading in the crosswalks between the columns and the NAICS codes:
    pa03cross = pd.read_csv(pa_03_cross_file)
    #
    data_tree = naics.load_data_with_cross(
                    data_tree = data_tree, data_df = data_03,
                    cross_df = pa03cross, data_cols = cols_03,
                    df_name = "PA_assets"
                    )
    #
    if blueprint == None and "tot_corps" in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = "tot_corps"
    naics.pop_back(tree=data_tree, df_list=["PA_assets"])
    naics.pop_forward(tree=data_tree, df_list=["PA_assets"],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree

def load_pa_05_data(data_tree = None, blue_tree = None, blueprint = None):
    # Defining constant factor (e.g. data is in thousands):
    pa_05_fctr = 10 ** 3
    # Defining constant list of types of partners:
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
    if data_tree == None:
        data_tree = naics.generate_tree()
    #
    for i in os.listdir(prt_dir):
        if("pa05.xls" in i):
            pa_05_file = os.path.abspath(prt_dir + "\\" + i)
        elif("pa05_Crosswalk.csv" in i):
            pa_05_cross_file = os.path.abspath(prt_dir + "\\" + i)
    #
    book_05 = xlrd.open_workbook(pa_05_file)
    sheet_05 = book_05.sheet_by_index(0)
    cur_rows = sheet_05.nrows
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
    data_05 = data_05 * pa_05_fctr
    # Reading in the crosswalks between the columns and the NAICS codes:
    pa05cross = pd.read_csv(pa_05_cross_file)
    #
    data_tree = naics.load_data_with_cross(
                    data_tree = data_tree, data_df = data_05,
                    cross_df = pa05cross, data_cols = cols_05,
                    df_name = "PA_types"
                    )
    # Defaults:
    if blueprint == None and "PA_inc_loss" in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = "PA_inc_loss"
    elif blueprint == None and "tot_corps" in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = "tot_corps"
    naics.pop_back(tree=data_tree, df_list=["PA_types"])
    naics.pop_forward(tree=data_tree, df_list=["PA_types"],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree


def load_soi_partner_data(data_tree = None, get_pa01 = True, get_pa03 = True,
                          get_pa05 = True):
    #
    if data_tree == None:
        data_tree = naics.generate_tree()
    #
    if get_pa01:
        data_tree = load_pa_01_data(data_tree)
    if get_pa03:
        data_tree = load_pa_03_data(data_tree)
    if get_pa05:
        data_tree = load_pa_05_data(data_tree)
    #
    return data_tree


def load_soi_prop_data(data_tree = None, blue_tree = None, blueprint = None):
    #
    prop_fctr = 10**3
    #
    if data_tree == None:
        data_tree = naics.generate_tree()
    # Finding the "\**sp01br" file in the proprietorships folder:
    for i in os.listdir(prop_dir):
        if(i[2:] == "sp01br.xls"):
            sp01br_file = os.path.abspath(prop_dir + "\\" + i)
        if(i[2:] == "sp01br_Crosswalk.csv"):
            sp01br_cross_file = os.path.abspath(prop_dir + "\\" + i)
    # Opening nonfarm proprietor data:
    cur_wb = xlrd.open_workbook(sp01br_file)
    cur_ws = cur_wb.sheet_by_index(0)
    cur_cross = pd.read_csv(sp01br_cross_file)
    # Finding the relevant positions in worksheet:
    pos1 = naics.search_ws(cur_ws,"Industrial sector",20, True, [0,0], True)
    pos2 = naics.search_ws(cur_ws,"Depreciation\ndeduction",20)
    pos3 = naics.search_ws(cur_ws,"Depreciation\ndeduction",20,
                         True, np.array(pos2) + np.array([0,1]))
    #
    for i in data_tree.enum_inds:
        i.append_dfs(("soi_prop", pd.DataFrame(np.zeros((1,1)),
                                    columns = ["Depr Deductions"])))
    #
    cross_index = cur_cross.shape[0]-1
    enum_index = len(data_tree.enum_inds)-1
    for i in xrange(pos1[0],cur_ws.nrows):
        cur_cell = str(cur_ws.cell_value(i,pos1[1])).lower().strip()
        #
        tot_proportions = 0
        for j in xrange(0, cur_cross.shape[0]):
            cross_index = (cross_index+1) % cur_cross.shape[0]
            cur_ind_name = str(cur_cross.iloc[cross_index,0]).lower().strip()
            if(cur_cell == cur_ind_name):
                if pd.isnull(cur_cross.iloc[cross_index,1]):
                    continue
                ind_codes = str(cur_cross.iloc[cross_index,1]).split(".")
                for k in xrange(0, len(data_tree.enum_inds)):
                    enum_index = (enum_index+1) % len(data_tree.enum_inds)
                    cur_data = data_tree.enum_inds[enum_index].data
                    cur_codes = cur_data.dfs["Codes:"]
                    #
                    #print ind_codes
                    #print cur_codes
                    cur_proportions = naics.compare_codes(ind_codes, cur_codes.iloc[:,0])
                    if cur_proportions == 0:
                        continue
                    tot_proportions += cur_proportions
                    cur_dfs = cur_data.dfs["soi_prop"]["Depr Deductions"]
                    cur_dfs[0] += (prop_fctr * cur_proportions 
                                        * (cur_ws.cell_value(i,pos2[1]) 
                                        + cur_ws.cell_value(i,pos3[1])))
            if(tot_proportions == 1):
                break
    # Default:
    if blueprint == None and "tot_corps" in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = "tot_corps"
    naics.pop_back(tree=data_tree, df_list=["soi_prop"])
    naics.pop_forward(tree=data_tree, df_list=["soi_prop"],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree


def load_soi_farm_prop(data_tree = None, blue_tree = None, blueprint = None):
    #
    if data_tree == None:
        data_tree = naics.generate_tree()
    #Load Farm Proprietorship data:
    farm_data = pd.read_csv(os.path.abspath(prop_dir + "\\Farm_Data.csv"))
    new_farm_cols = ["Land", "FA"]
    #
    for i in data_tree.enum_inds:
        i.append_dfs(("farm_prop",
                      pd.DataFrame(np.zeros((1,len(new_farm_cols))), 
                                   columns=new_farm_cols)))
    #
    land_mult = ((farm_data["R_sp"][0] + farm_data["Q_sp"][0]) * 
                        (float(farm_data["A_sp"][0])/farm_data["A_p"][0]))
    total = farm_data.iloc[0,0] + farm_data.iloc[0,2]
    total_pa = 0
    cur_codes = [111,112]
    proportions = np.zeros(len(cur_codes))
    proportions = naics.get_proportions(cur_codes, data_tree, "PA_assets", 
                                 ["Land (Net)","Depreciable assets (Net)"])
    #
    for i in xrange(0, len(cur_codes)):
        cur_ind = naics.find_naics(data_tree, cur_codes[i])
        cur_df = cur_ind.data.dfs["PA_assets"]
        total_pa += (cur_df["Land (Net)"][0] + 
                                cur_df["Depreciable assets (Net)"][0])
    #
    for i in xrange(0,len(cur_codes)):
        cur_ind = naics.find_naics(data_tree, cur_codes[i])
        cur_ind.data.dfs["farm_prop"]["Land"][0] = (land_mult * 
                            cur_ind.data.dfs["PA_assets"]["Land (Net)"][0]/
                            total_pa)
        cur_ind.data.dfs["farm_prop"]["FA"][0] = ((proportions.iloc[1,i]*total)
                                    - cur_ind.data.dfs["farm_prop"]["Land"][0])
    # Default:            
    if blueprint == None and "tot_corps" in data_tree.enum_inds[0].data.dfs.keys():
        blueprint = "tot_corps"
    naics.pop_back(tree=data_tree, df_list=["farm_prop"])
    naics.pop_forward(tree=data_tree, df_list=["farm_prop"],
                      blueprint=blueprint, blue_tree=blue_tree)
    #
    return data_tree


