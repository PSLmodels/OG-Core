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
Create an output tree containing only the final data on FA, INV, and LAND.
'''
def summary_tree(data_tree, data_folder):
    all_sectors = ["C Corporations", 
                   "S Corporations",
                   "Corporate general partners", 
                   "Corporate limited partners",
                   "Individual general partners",
                   "Individual limited partners",
                   "Partnership general partners",
                   "Partnership limited partners",
                   "Tax-exempt organization general partners",
                   "Tax-exempt organization limited partners",
                   "Nominee and other general partners", 
                   "Nominee and other limited partners", 
                   "Sole Proprietors"]
    
    pa_types = data_tree.enum_inds[0].data.dfs["PA_types"].columns
    pa_types = pa_types.values.tolist()
    #
    output_tree = naics.load_naics(data_folder + "\\2012_NAICS_Codes.csv")
    #
    for i in output_tree.enum_inds:
        i.append_dfs(("FA",pd.DataFrame(np.zeros((1, len(all_sectors))),
                                        columns = all_sectors)))
        i.append_dfs(("INV",pd.DataFrame(np.zeros((1, len(all_sectors))),
                                         columns = all_sectors)))
        i.append_dfs(("LAND",pd.DataFrame(np.zeros((1, len(all_sectors))),
                                          columns = all_sectors)))
    #
    for i in range(0, len(output_tree.enum_inds)):
        #
        #cur_data = data_tree.enum_inds[i].data
        #out_data = output_tree.enum_inds[i].data
        cur_dfs = data_tree.enum_inds[i].data.dfs
        out_dfs = output_tree.enum_inds[i].data.dfs
        partner_sum = sum(cur_dfs["PA_types"].iloc[0,:])
        #
        for j in range(0, len(all_sectors)):
            sector = all_sectors[j]
            #
            if sector == "C Corporations":
                cur_df = cur_dfs["c_corps"]
                out_dfs["FA"][sector][0] = cur_df["Depreciable Assets"][0]
                out_dfs["INV"][sector][0] = cur_df["Inventories"][0]
                out_dfs["LAND"][sector][0] = cur_df["Land"][0]
            elif sector == "S Corporations":
                cur_df = cur_dfs["s_corps"]
                out_dfs["FA"][sector][0] = cur_df["Depreciable Assets"][0]
                out_dfs["INV"][sector][0] = cur_df["Inventories"][0]
                out_dfs["LAND"][sector][0] = cur_df["Land"][0]
            elif sector in pa_types:
                if partner_sum != 0:
                    ratio = abs(cur_dfs["PA_types"][sector][0])/partner_sum
                else:
                    ratio = abs(1.0/cur_dfs["PA_types"].shape[0])
                cur_df = cur_dfs["PA_assets"]
                out_dfs["FA"][sector][0] = abs(
                                    ratio*cur_df["Depreciable assets (Net)"][0]
                                    )
                out_dfs["INV"][sector][0] = abs(
                                        ratio*cur_df["Inventories (Net)"][0]
                                        )
                out_dfs["LAND"][sector][0] = abs(
                                                ratio*cur_df["Land (Net)"][0]
                                                )
            elif sector == "Sole Proprietors":
                if cur_dfs["PA_inc_loss"]["Depreciation"][0] != 0:
                    ratio = abs(cur_dfs["soi_prop"]["Depr Deductions"][0]/
                                cur_dfs["PA_inc_loss"]["Depreciation"][0])
                else:
                    ratio = 0.0
                cur_df = cur_dfs["PA_assets"]
                out_dfs["FA"][sector][0] = abs(
                                        (ratio*
                                        cur_df["Depreciable assets (Net)"][0])+
                                        cur_dfs["farm_prop"]["FA"][0]
                                        )
                out_dfs["INV"][sector][0] = abs(
                                        (ratio*cur_df["Inventories (Net)"][0])+
                                        cur_dfs["farm_prop"]["Land"][0]
                                        )
                out_dfs["LAND"][sector][0] = abs(ratio*cur_df["Land (Net)"][0])
    return output_tree
















