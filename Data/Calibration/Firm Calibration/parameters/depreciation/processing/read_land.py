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
# A class that defines data structures such as trees:
import naics_processing as naics


'''

'''
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
    land_tree = naics.load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    df_cols = ["All", "Corp", "Non-Corp"]
    for i in land_tree.enum_inds:
        i.data.append(("Land", 
                       pd.DataFrame(np.zeros((1,len(df_cols))), 
                                    columns = df_cols)))
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
    land_df = land_tree.enum_inds[0].data.dfs["Land"]
    land_df["Corp"][0] = land_data["Corporate"][0]
    land_df["Non-Corp"][0] = land_data["Non-Corporate"][0]
    land_df["All"][0] = (land_data["Corporate"][0]+
                            land_data["Non-Corporate"][0])
    return land_tree












