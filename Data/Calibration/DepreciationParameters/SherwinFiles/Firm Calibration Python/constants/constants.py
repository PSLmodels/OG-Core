"""
Frequently Used Constants (constants.py):
-------------------------------------------------------------------------------
Last updated: 6/26/2015.

This module initializes constants and names that are frequently used
throughout this program by in various modules. The purpose is to have one
place where names can be readily altered without having to comb through every
file. While the name of the variables in a sense can only be changed by doing
that, there is less reason to change the variable names since they do not
effect any output files.
"""
# Default dataframe names:
CODE_DF_NM = "Codes:"
INC_PRT_DF_NM = "prt_inc_loss"
AST_PRT_DF_NM = "prt_asset"
TYP_PRT_DF_NM = "prt_types"
TOT_CORP_DF_NM = "tot_corps"
S_CORP_DF_NM = "s_corps"
C_CORP_DF_NM = "c_corps"
FARM_PROP_DF_NM = "farm_prop"
NON_FARM_PROP_DF_NM = "soi_prop"
# Default total-corporations data columns dictionary:
DFLT_TOT_CORP_COLS_DICT = dict([
                    ("depreciable_assets","DPRCBL_ASSTS"),
                    ("accumulated_depreciation", "ACCUM_DPR"),
                    ("land", "LAND"),
                    ("inventories", "INVNTRY"),
                    ("interest_paid", "INTRST_PD"), 
                    ("Capital_stock", "CAP_STCK"),
                    ("additional_paid-in_capital", "PD_CAP_SRPLS"),
                    ("earnings_(rtnd_appr.)", "RTND_ERNGS_APPR"),
                    ("earnings_(rtnd_unappr.)", "COMP_RTND_ERNGS_UNAPPR"),
                    ("cost_of_treasury_stock", "CST_TRSRY_STCK")
                    ])
# Default s-corporations data columns dictionary:
DFLT_S_CORP_COLS_DICT = dict([
                    ("depreciable_assets","DPRCBL_ASSTS"),
                    ("accumulated_depreciation", "ACCUM_DPR"),
                    ("land", "LAND"),
                    ("inventories", "INVNTRY"),
                    ("interest_paid", "INTRST_PD"), 
                    ("Capital_stock", "CAP_STCK"),
                    ("additional_paid-in_capital", "PD_CAP_SRPLS"),
                    ("earnings_(rtnd_appr.)", ""),
                    ("earnings_(rtnd_unappr.)", "COMP_RTND_ERNGS_UNAPPR"),
                    ("cost_of_treasury_stock", "CST_TRSRY_STCK")
                    ])
# Default partnership income df columns:
DFLT_PRT_INC_DF_COL_NMS_DICT = dict([
                    ("NET_INC", "total_net_income"),
                    ("NET_LOSS", "total_net_loss"),
                    ("DEPR", "depreciation")
                    ])
DFLT_PRT_INC_DF_COL_NMS = DFLT_PRT_INC_DF_COL_NMS_DICT.values()
DFLT_PRT_INC_DF_COL_NMS.sort()
# Default partnership assets data columns:
DFLT_PRT_AST_DF_COL_NMS_DICT = dict([
                    ("DEPR_AST_NET", "depreciable_assets_net"),
                    ("ACC_DEPR_NET", "accumulated_depreciation_net"), 
                    ("INV_NET", "inventories_net"),
                    ("LAND_NET", "land_net"),
                    ("DEPR_AST_INC", "depreciable_assets_income"),
                    ("ACC_DEPR_INC", "accumulated_depreciation_income"),
                    ("INV_INC", "inventories_income"),
                    ("LAND_INC", "land_income")
                    ])
# Default partnership types data columns
DFLT_PRT_TYP_DF_COL_NMS_DICT = dict([
                    ("CORP_GEN_PRT", "corporate_general_partners"),
                    ("CORP_LMT_PRT", "corporate_limited_partners"),
                    ("INDV_GEN_PRT", "individual_general_partners"),
                    ("INDV_LMT_PRT", "individual_limited_partners"),
                    ("PRT_GEN_PRT", "partnership_general_partners"),
                    ("PRT_LMT_PRT", "partnership_limited_partners"),
                    ("EXMP_GEN_PRT", "tax_exempt_general_partners"),
                    ("EXMP_LMT_PRT", "tax_exempt_limited_partners"),
                    ("OTHER_GEN_PRT", "other_general_partners"),
                    ("OTHER_LMT_PRT", "other_limited_partners")
                    ])
# Default proprietorship nonfarm data columns:
DFLT_PROP_NFARM_DF_COL_NMS_DICT = dict([
                    ("DEPR_DDCT", "depreciation_deductions")
                    ])
# All sectors:
ALL_SECTORS_NMS_DICT = dict([
                    ("C_CORP", "c_corporations"),
                    ("S_CORP", "s_corporations"),
                    ("CORP_GEN_PRT", "corporate_general_partners"),
                    ("CORP_LMT_PRT", "corporate_limited_partners"),
                    ("INDV_GEN_PRT", "individual_general_partners"),
                    ("INDV_LMT_PRT", "individual_limited_partners"),
                    ("PRT_GEN_PRT", "partnership_general_partners"),
                    ("PRT_LMT_PRT", "partnership_limited_partners"),
                    ("EXMP_GEN_PRT", "tax_exempt_general_partners"),
                    ("EXMP_LMT_PRT", "tax_exempt_limited_partners"),
                    ("OTHER_GEN_PRT", "other_general_partners"),
                    ("OTHER_LMT_PRT", "other_limited_partners"),
                    ("SOLE_PROP", "sole_proprietorships")
                    ])
CORP_TAX_SECTORS_NMS_DICT = dict([
                    ("C_CORP", "c_corporations"),
                    ("CORP_GEN_PRT", "corporate_general_partners"),
                    ("CORP_LMT_PRT", "corporate_limited_partners"),
                    ])
NON_CORP_TAX_SECTORS_NMS_DICT = dict([
                    ("S_CORP", "s_corporations"),
                    ("INDV_GEN_PRT", "individual_general_partners"),
                    ("INDV_LMT_PRT", "individual_limited_partners"),
                    ("PRT_GEN_PRT", "partnership_general_partners"),
                    ("PRT_LMT_PRT", "partnership_limited_partners"),
                    ("EXMP_GEN_PRT", "tax_exempt_general_partners"),
                    ("EXMP_LMT_PRT", "tax_exempt_limited_partners"),
                    ("OTHER_GEN_PRT", "other_general_partners"),
                    ("OTHER_LMT_PRT", "other_limited_partners"),
                    ("SOLE_PROP", "sole_proprietorships")
                    ])
                    
ALL_SECTORS_NMS_LIST = ALL_SECTORS_NMS_DICT.values()
#



