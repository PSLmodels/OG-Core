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
    sht_pos = naics.search_ws(bea_readme, "Industry Title", 25, False)
    if(sht_pos == [-1,-1]):
        sht_pos = naics.search_ws(bea_readme, "bea code", 25, False, [0,0], True)
        sht_pos[1] = sht_pos[1] - 1
    if(sht_pos == [-1,-1]):
        print "Error in reading BEA fixed asset \"readme\" sheet."
        return None
    cur_row = sht_pos[0] + 1
    cur_col = sht_pos[1]
    # Finding the number of industries (includes those without bea codes):
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
    code_index = 0
    cur_row = sht_pos[0] + 1
    cur_col = sht_pos[1]
    bea_codes2 = np.zeros(number_of_industries, dtype=object)
    while cur_row < bea_readme.nrows:
        if(str(bea_readme.cell_value(cur_row, cur_col)) != ""):
            cur_code = str(bea_readme.cell_value(cur_row, cur_col+1))
            cur_code = cur_code.replace("\xa0", " ").strip()
            bea_codes2[code_index] = cur_code
            code_index += 1
        cur_row += 1
    # Reading in a list of the assets in the BEA file:
    list_file = os.path.abspath(bea_folder + "\\detailnonres_list.csv")
    asset_list = pd.read_csv(list_file)
    for i in xrange(0, asset_list.shape[0]):
        asset_list.iloc[i,0] = asset_list.iloc[i,0].replace("\xa0", " ")
        asset_list.iloc[i,0] = asset_list.iloc[i,0].strip()
    
    # Reading in the corresponding naics codes:
    naics_file = os.path.abspath(bea_folder + "\\detailnonres_naics.csv")
    naics_cross = pd.read_csv(naics_file).replace("\xa0", " ")
    naics_inds = naics_cross["Industry"]
    for i in xrange(0, naics_cross.shape[0]):
        naics_inds[i] = naics_inds[i].replace("\xa0", " ").strip()
    # Creating a chart cross-referencing industry names, BEA and NAICS codes.
    chart_cols = ["Industry","BEA Code","NAICS Code"]
    bea_chart = pd.DataFrame(np.zeros(shape=(num_shts-2,3), dtype=object),
                             columns = chart_cols)
    bea_inds = bea_chart["Industry"]
    bea_naics = bea_chart["NAICS Code"]
    cur_row = sht_pos[0] + 1
    cur_col = sht_pos[1]
    num_naics = naics_cross.shape[0]
    # Filling chart with naics codes that are in both lists and the crosswalk:
    naics_counter = 0
    for i in range(0, num_shts-2):
        for cur_row in range(sht_pos[0]+1, bea_readme.nrows):
            bea_code = str(bea_readme.cell_value(cur_row,cur_col+1))
            if(str(bea_codes1[i]) == bea_code):
                bea_ind = str(bea_readme.cell_value(cur_row,cur_col))
                bea_ind = bea_ind.replace('\xa0', ' ').strip()
                bea_inds[i] = bea_ind
                bea_chart["BEA Code"][i] = bea_code
                for k in xrange(0, num_naics):
                    naics_counter = (naics_counter+1) % num_naics
                    if(naics_inds[naics_counter] == bea_chart["Industry"][i]):
                       bea_naics[i] = naics_cross["NAICS"][naics_counter]
                       break
                break
            # If they match except one has ".0" at the end:
            elif(str(bea_codes1[i]) == 
                    str(bea_readme.cell_value(cur_row, cur_col+1))[:-2]):
                bea_ind = str(bea_readme.cell_value(cur_row, cur_col))
                bea_ind = bea_ind.replace('\xa0', ' ').strip()
                bea_chart["Industry"][i] = bea_ind
                cur_code = str(bea_readme.cell_value(cur_row, cur_col+1))[:-2]
                bea_chart["BEA Code"][i] = cur_code
                for k in xrange(0, num_naics):
                    naics_counter = (naics_counter+1) % num_naics
                    if(naics_inds[naics_counter] == bea_inds[i]):
                        bea_naics[i] = naics_cross["NAICS"][naics_counter]
                        break
                break
    # Initializing the table of assets:
    #cur_sht = bea_book.sheet_by_name(bea_chart["BEA Code"][0])
    #sht_pos = naics.search_ws(cur_sht, "asset codes", 25, False)
    bea_table = pd.DataFrame(np.zeros((asset_list.shape[0],
                                       bea_chart.shape[0])), 
                             columns = bea_chart["BEA Code"])
    # For each industry, calculating 
    for i in bea_chart["BEA Code"]:
        cur_sht = bea_book.sheet_by_name(i)
        sht_pos = naics.search_ws(cur_sht, "asset codes", 25, False)
        for j in xrange(0, len(asset_list)): #xrange(sht_pos[0]+2, cur_sht.nrows):
            cur_asset = asset_list.iloc[j,0]
            for k in xrange(sht_pos[0]+2, cur_sht.nrows):
                cur_cell = str(cur_sht.cell_value(k, sht_pos[1]+1))
                cur_cell = cur_cell.replace("\xa0", " ").strip()
                if(cur_asset == cur_cell):
                    bea_table[i][j] = float(
                                        cur_sht.cell_value(k, cur_sht.ncols-1)
                                        )
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
    asset_tree = naics.load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    for i in xrange(0, len(asset_tree.enum_inds)):
        asset_tree.enum_inds[i].data.append(("All", 
                pd.DataFrame(np.zeros((1, asset_list.shape[0])), 
                             columns = asset_list.iloc[:,0])))
        asset_tree.enum_inds[i].data.append(("Corp", 
                pd.DataFrame(np.zeros((1, asset_list.shape[0])),
                             columns = asset_list.iloc[:,0])))
        asset_tree.enum_inds[i].data.append(("Non-Corp", 
                pd.DataFrame(np.zeros((1, asset_list.shape[0])),
                             columns = asset_list.iloc[:,0])))
    # Fill in data from BEA's fixed asset table:
    enum_index = len(output_tree.enum_inds) - 1
    for i in xrange(0, bea_table.shape[1]):
        cur_codes = str(bea_chart["NAICS Code"][i]).split(".")
        tot_share = 0
        all_proportions = naics.get_proportions(cur_codes, output_tree, 
                                          "FA").iloc[1,:]
        corp_proportions = naics.get_proportions(cur_codes, output_tree, "FA", 
                                           corp_types).iloc[1,:]
        non_corp_proportions = naics.get_proportions(cur_codes, output_tree, 
                                               "FA", non_corp_types).iloc[1,:]
        for code_index in xrange(0, len(cur_codes)):
            for j in xrange(0, len(asset_tree.enum_inds)):
                enum_index = (enum_index+1) % len(asset_tree.enum_inds)
                out_dfs = output_tree.enum_inds[enum_index].data.dfs
                if(sum(out_dfs["FA"].iloc[0,:]) == 0):
                    continue
                all_ratio = 1.0
                corp_ratio = 0.0
                non_corp_ratio = 0.0
                for category in corp_types:
                    corp_ratio += (out_dfs["FA"][category][0]/
                                        sum(out_dfs["FA"].iloc[0,:]))
                for category in non_corp_types:
                    non_corp_ratio += (out_dfs["FA"][category][0]/
                                            sum(out_dfs["FA"].iloc[0,:]))
                cur_data = asset_tree.enum_inds[enum_index].data
                ind_codes = cur_data.dfs["Codes:"].iloc[:,0]
                share = naics.compare_codes(cur_codes, ind_codes)
                tot_share += share
                if(share == 0):
                    continue
                num_assets = asset_tree.enum_inds[0].data.dfs["All"].shape[1]
                for k in xrange(0, num_assets):
                    cur_data.dfs["All"].iloc[0,k] = (bea_table.iloc[k,i]*
                                            all_ratio*
                                            all_proportions[code_index])
                    cur_data.dfs["Corp"].iloc[0,k] = (bea_table.iloc[k,i]*
                                            corp_ratio*
                                            corp_proportions[code_index])
                    cur_data.dfs["Non-Corp"].iloc[0,k] = (bea_table.iloc[k,i]*
                                            non_corp_ratio*
                                            non_corp_proportions[code_index])
                break
            if(tot_share == 1):
                break
    return asset_tree












