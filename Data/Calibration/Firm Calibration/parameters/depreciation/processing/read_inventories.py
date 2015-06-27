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

'''
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
    cur_index = naics.search_ws(sht0, 1, 25, True, [0,0], True)
    check_index = naics.search_ws(sht0, "line", 20)
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
    inv_tree = naics.load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    #
    data_cols = ["All", "Corp", "Non-Corp"]
    for i in inv_tree.enum_inds:
        i.data.append(("Inventories",
                       pd.DataFrame(np.zeros((1, len(data_cols))), 
                                    columns = data_cols)))
    #
    inv_data = np.zeros(inv_cross.shape[0])
    #
    cross_index = 0
    for i in xrange(cur_index[0], num_rows):
        if(cross_index >= inv_cross.shape[0]):
            break
        cur_list = str(sht0.cell_value(i, cur_index[1])).strip()
        cur_name = str(sht0.cell_value(i, cur_index[1]+1)).strip()
        checks = ((str(cur_list) == str(inv_cross["List"][cross_index])) and 
                    (str(cur_name) == str(inv_cross["Industry"][cross_index])))
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
        proportions = naics.get_proportions(cur_codes, output_tree, "INV")
        for j in xrange(0, proportions.shape[1]):
            cur_ind = inv_tree.enum_inds[int(proportions.iloc[0,j])]
            prev_ind = output_tree.enum_inds[int(proportions.iloc[0,j])]
            prev_df = prev_ind.data.dfs["INV"]
            if(sum(prev_df.iloc[0, :]) != 0):
                cur_dfs = ((prev_df/sum(prev_df.iloc[0,:])) *
                                (inv_data[i] * proportions.iloc[1,j]))
                inv_df = cur_ind.data.dfs["Inventories"]
                inv_df["All"] += sum(cur_dfs.iloc[0,:])
                for k in corp_types:
                    inv_df["Corp"] += cur_dfs[k][0]
                for k in non_corp_types:
                    inv_df["Non-Corp"] += cur_dfs[k][0]
    #
    return inv_tree











