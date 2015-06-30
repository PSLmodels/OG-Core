'''
-------------------------------------------------------------------------------
Date created: 5/22/2015
Last updated 5/22/2015
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
    Packages:
-------------------------------------------------------------------------------
'''
import os.path
import sys
sys.path.append(os.path.abspath("N:\Lott, Sherwin\Other Calibration\Program"))
import numpy as np
import pandas as pd
import xlrd
#
import naics_processing as naics
'''
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
'''
# Defining constant names:
WAGES = "WAGES"

#
def load_nipa_wages_ind(data_folder,tree = None):
    wages_ind_file = os.path.abspath(data_folder + "\\Wages--Industry.xls")
    wages_ind_cross_file = os.path.abspath(data_folder + "\\Wages--Industry_Crosswalk.csv")
    #
    data = load_nipa_ind(wages_ind_file, wages_ind_cross_file)
    data.columns = ["NAICS_Code", WAGES]
    #
    conversion_factor = 1.0
    for i in xrange(0, data.shape[0]):
        data[WAGES][i] *= conversion_factor
    if tree == None:
        return data
    naics_data_to_tree(tree, data, WAGES)
    

def load_nipa_ind(data_file, cross_file):
    #data_folder = "N:\\Lott, Sherwin\\Other Calibration\\Program\\national_income\\data"
    data_book = xlrd.open_workbook(data_file)
    data_sht = data_book.sheet_by_index(0)
    #
    data_cross = pd.read_csv(cross_file)
    #data_cross = data_cross.fillna(-1)
    #data_cross = pd.DataFrame(data_cross[data_cross["NAICS Code:"] != -1])
    output = np.zeros(data_cross.shape[0])
    
    start_pos = naics.search_ws(data_sht, "Line", 25, True, [0,0], True)
    for i in xrange(start_pos[0]+1, data_sht.nrows):
        if(str(data_sht.cell_value(i,start_pos[1])) == "1"):
            start_pos[0] = i
            break
    
    cur_row = start_pos[0]
    ind_col = start_pos[1] + 1
    data_col = data_sht.ncols - 1
    
    for i in xrange(0, data_sht.ncols):
        try:
            float(data_sht.cell_value(cur_row, data_col))
            break
        except ValueError:
            data_col -= 1
    
    for i in xrange(0, data_cross.shape[0]):
        
        for j in xrange(start_pos[0], data_sht.nrows):
            try:
                if(data_cross["Industry"][i] in data_sht.cell_value(cur_row, ind_col)):
                    output[i] = data_sht.cell_value(cur_row, data_col)
                    cur_row = start_pos[0] + ((cur_row+1-start_pos[0]) % (data_sht.nrows-start_pos[0]))
                    break
                cur_row = start_pos[0] + ((cur_row+1-start_pos[0]) % (data_sht.nrows-start_pos[0]))
            except ValueError:
                cur_row = start_pos[0] + ((cur_row+1-start_pos[0]) % (data_sht.nrows-start_pos[0]))
    
    return pd.DataFrame(np.column_stack((data_cross["NAICS_Code"], output)), columns = ["NAICS Codes:", ""])


def naics_data_to_tree(tree, df, df_name = "", bp_tree = None, bp_df = None):
    #
    for i in tree.enum_inds:
        i.append_dfs((df_name, pd.DataFrame(np.zeros((1,len(df.columns[1:]))),
                                                 columns = df.columns[1:])))
    #
    enum_index = 0
    #
    for i in xrange(0, len(tree.enum_inds)):
        cur_ind = tree.enum_inds[i]
        cur_dfs = cur_ind.data.dfs
        tot_share = 0
        for j in xrange(0, df.shape[0]):
            if df["NAICS_Code"][j] != df["NAICS_Code"][j]:
                continue
            df_code = df["NAICS_Code"][j]
            df_code = df_code.split(".")
            
            cur_share = naics.compare_codes(df_code, cur_dfs["Codes:"].iloc[:,0])
            if cur_share == 0:
                continue
            tot_share += cur_share
            #
            for k in xrange(1, df.shape[1]):
                cur_dfs[df_name].iloc[0,k-1] = df.iloc[j,k] #Removed cur_share
            #
            if tot_share == 1:
                break
            
        enum_index = (enum_index+1) % len(tree.enum_inds)





