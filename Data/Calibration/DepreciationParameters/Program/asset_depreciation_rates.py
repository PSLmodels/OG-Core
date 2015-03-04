'''
------------------------------------------------------------------------
Last updated 1/23/2015

This Python file defines the object "asset_depreciation".
The only public variable in this class "rates" is a pandas data frame
    that has the economic and tax depreciation rates for BEA asset types. \
    There are four columns in "rates":
        --"Code":   (xxxxxx)
        --"Asset":  NIPA's Asset Type
        --"Economic Depreciation Rate"
        --"Tax Depreciation Rate"


This py-file calls the following other file(s):
            data\Generic_Economic_Data.csv
            data\Economic_Depreciation_Rates.csv
            data\Economic_Depreciation_Schedules.csv

This py-file creates no output.
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import os.path
import numpy as np
import pandas as pd

'''
------------------------------------------------------------------------
    
------------------------------------------------------------------------
    
------------------------------------------------------------------------
'''

# Working directory, note that there should already be an "OUTPUT" file as well
#   as a "data" file with all relevant data files.
path = os.getcwd()
# Relevant path and file names:
data_folder = os.getcwd() + "\data"
file_EDR = "\Economic_Depreciation_Rates.csv"
file_GED = "\Generic_Economic_Data.csv"
file_EDS = "\Economic_Depreciation_Schedules.csv"

# Column names in both "rates" and "Economic_Depreciation_Rates.csv":
# (Draft note: the purpose of this is to make this program column invariant.
#   If columns are added or moved in the csv file, then the program will still
#   work. If the names of the columns are changed, then all that needs to be
#   updated here are the following names.)
col_code = "Code"
col_asset = "Asset"
col_EDR = "Economic Depreciation Rate"
col_TDR = "Tax Depreciation Rate"




class asset_depreciation:
    
    def __init__(self, init_sched=True):
        # Initialize the table of depreciation rates for BEA assets:
        self.rates = pd.read_csv(data_folder + file_EDR)
        
        # Get the interest rate:
        #   (Needs more justification/documentation)
        #   (Could be done automatically from a URL)
        generic_data = pd.read_csv(data_folder + file_GED)
        self.__real_interest = generic_data["Real Interest Rate"][0]        
        
        # Initialize assets economic depreciation rates with schedules:
        if (init_sched):
            self.__initialize_schedules()
            
        # Handling missing rates by filling them in with a one.
        # (Draft note: for any final version, there ought not be any assets
        #       without either a explicit depreciation rate or a schedule.
        #       There ought to, and will be some sort of warning that the
        #       program throws if this occurs.)
        self.rates = self.rates.fillna(1)
        
    
    
    def __initialize_schedules(self):
        # Get the table of economic depreciation schedules:
        schedules = pd.read_csv(data_folder + file_EDS)
        # Get pertinent information about the table (e.g. dimensions, heading):
        columns = len(schedules.columns)
        rows = len(schedules.iloc[:,0])
        heading = list(schedules)
        years = schedules["Year"]
        
        # Calculate the discount rate:
        disc_rate = pow(1+self.__real_interest, -1)
        
        # For each asset with a schedule, calculate the depreciatin rate:
        for i in range(0, columns-1):
            # The percentage of remaining asset value shifted by a year:
            asset_values = np.concatenate([[1],schedules.iloc[:,i+1]])
            asset_values = np.delete(asset_values,rows)
            # Change in the percentage of a remaining assets value by year:
            change_in_values = asset_values-schedules.iloc[:,i+1]
            # Present value of all the changes in the asset's value:
            pv_deprec = sum((change_in_values*pow(disc_rate, years)))
            # The geometric depreciation rate that has the same present value:
            geo_rate = pv_deprec*(1-disc_rate)/(1 - pv_deprec*disc_rate)
            # The row of the asset in "rates" pandas DF:
            indices = self.rates[col_asset] == heading[i+1]
            index = self.rates[indices].index[0]
            # Correct the economic depreciation rate in "rates" panda DF:
            self.rates[col_EDR][index] = geo_rate
        
        
        
        







