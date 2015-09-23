'''
-------------------------------------------------------------------------------
Date created: 5/12/2015
Last updated 5/12/2015
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
BUS_INC = "Business_Income"
FIN_INC = "Financial_Income"
INT_INC = "Interest_Income"

#
def calc_bus_inc(tree):
    for i in tree.enum_inds:
        i.append_dfs((BUS_INC, pd.DataFrame(np.zeros((1,1)), columns = [BUS_INC])))
    #
    for i in tree.enum_inds:
        fin_inc = i.data.dfs[FIN_INC][FIN_INC][0]
        int_inc = i.data.dfs[INT_INC][INT_INC][0]
        i.data.dfs[BUS_INC][BUS_INC][0] = fin_inc - int_inc 
        

#tree = naics.load_naics("N:\\Lott, Sherwin\\Other Calibration\\Program\\depreciation\\data\\2012_NAICS_Codes.csv")
def load_nipa_inc_ind(data_folder, tree = None):
    inc_ind_file = os.path.abspath(data_folder + "\\National_Income--Industry.xls")
    inc_ind_cross_file = os.path.abspath(data_folder + "\\National_Income--Industry_Crosswalk.csv")
    #
    data = load_nipa_ind(inc_ind_file, inc_ind_cross_file)
    data.columns = ["NAICS_Code", FIN_INC]
    #
    conversion_factor = 10 ** 9
    for i in xrange(0, data.shape[0]):
        data[FIN_INC][i] *= conversion_factor
    #
    if tree == None:
        return data
    naics_data_to_tree(tree, data, FIN_INC)

    
def load_nipa_int_ind(data_folder, tree = None):
    int_ind_file = os.path.abspath(data_folder + "\\Interest--Industry.xls")
    int_ind_cross_file = os.path.abspath(data_folder + "\\Interest--Industry_Crosswalk.csv")
    #
    data = load_nipa_ind(int_ind_file, int_ind_cross_file)
    data.columns = ["NAICS_Code", INT_INC]
    #
    conversion_factor = 10.0 ** 6
    for i in xrange(0, data.shape[0]):
        data[INT_INC][i] *= conversion_factor
    #
    if tree == None:
        return data
    naics_data_to_tree(tree, data, INT_INC)

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
                cur_dfs[df_name].iloc[0,k-1] = cur_share *  df.iloc[j,k]
            #
            if tot_share == 1:
                break
            
        enum_index = (enum_index+1) % len(tree.enum_inds)
    
    


