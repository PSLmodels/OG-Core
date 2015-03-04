'''
------------------------------------------------------------------------
Last updated 1/26/2015

This Python file defines the object "industry_depreciation_rates".
The only public variable in this class "industry_rates" is a pandas DF that has
    the calculated depreciation rates for all the industries listed by the BEA


This py-file calls the following other file(s):
            data\detailnonres_stk1.xlsx
            
This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT\Industry_Depreciation_Rates.csv
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
import xlrd

'''
Contains the "asset_depreciation class" that initializes a pandas DF with all
    the asset depreciation rates "rates".
'''
import asset_depreciation_rates

'''
------------------------------------------------------------------------
    
------------------------------------------------------------------------
    
------------------------------------------------------------------------
'''

# Working directory, note that there should already be an "OUTPUT" file as well
#   as a "data" file with all relevant data files.
path = os.getcwd()
# Relevant path and file names:
data_folder = path + "\data"
output_folder = path + "\OUTPUT"
# The BEA industries input file:
bea_ind_file = "\detailnonres_stk1.xlsx"
# The output file of the pandas DF containing industry depreciation rates.
rates_out_file = "\Industry_Depreciation_Rates.csv"


'''
This class defines an object containing a pandas DF of all the depreciation
    rates by industry--"rates".
'''
class industry_depreciation_rates:
    
    def __init__(self):
        # Create the table of depreciation rates for BEA assets:
        self.__assets = asset_depreciation_rates.asset_depreciation()
        
        # Open the BEA workbook with all the industry inputs:
        bea_wb = xlrd.open_workbook(data_folder + bea_ind_file)
        bea_wb_sht_names = bea_wb.sheet_names()  #For efficiency
        bea_wb_nsht = bea_wb.nsheets
        bea_wb_sht0 = bea_wb.sheet_by_index(0)

        # Finding the position of "Industry Title" on first worksheet.
        # "Starting Position":
        st_pos = self.__search_worksheet(bea_wb_sht0, "bea code", 25, False)
        # Indices:
        row = st_pos[0] + 1
        col = st_pos[1] - 1

        # Finding total number of industries listed--including those without
        #   bea codes.
        number_of_industries = 0
        while row < bea_wb_sht0.nrows:
            if(str(bea_wb_sht0.cell_value(row, col)) != ""):
                number_of_industries += 1
            row += 1

        # Making a list of BEA codes based off the names of the worksheets.
        codes1 = np.zeros(bea_wb_nsht-1, dtype=object)
        for index in xrange(1, bea_wb_nsht):
            codes1[index-1] = str(bea_wb_sht_names[index])

        # Making a list of BEA codes codes based off the first BEA worksheet.
        array_index = 0
        row = st_pos[0] + 1
        col = st_pos[1] - 1
        codes2 = np.zeros(number_of_industries, dtype=object)
        
        # Going through the "Industry Title" column in BEA worksheet.
        while row < bea_wb_sht0.nrows:
            if(str(bea_wb_sht0.cell_value(row, col)) != ""):
                codes2[array_index] = str(bea_wb_sht0.cell_value(row, col+1))
                array_index += 1
            row += 1

        # Creating a table of codes and industries, this is listed in order
        #   based off the order of the worksheets.  This list is created by
        #   matching up the two lists created previously. "Industry names"
        ind_nms = np.zeros(shape=(bea_wb_nsht-2,2), dtype=object)
        row = st_pos[0] + 1
        col = st_pos[1] - 1

        # Checking that the two lists match up one-to-one.
        for i in range(0, bea_wb_nsht-2):
            for row in range(st_pos[0]+1, bea_wb_sht0.nrows):
                if(str(codes1[i]) == str(bea_wb_sht0.cell_value(row,col+1))):
                    ind_nms[i,0] = str(bea_wb_sht0.cell_value(row,col))
                    ind_nms[i,0] = ind_nms[i,0].replace('\xa0', ' ').strip()
                    ind_nms[i,1] = str(bea_wb_sht0.cell_value(row,col+1))
                    break
                elif codes1[i] == str(bea_wb_sht0.cell_value(row,col+1))[:-2]:
                    ind_nms[i,0] = str(bea_wb_sht0.cell_value(row,col))
                    ind_nms[i,0] = ind_nms[i,0].replace('\xa0', ' ').strip()
                    ind_nms[i,1] = str(bea_wb_sht0.cell_value(row,col+1))[:-2]
                    break
        
        self.rates = pd.DataFrame(
                        {"Code" : ind_nms[:,1],
                         "Asset" : ind_nms[:,0],
                         "Economic Depreciation Rate" : np.ones(bea_wb_nsht-2), 
                         "Tax Depreciation Rate" : np.ones(bea_wb_nsht-2)}, 
                         columns=["Code","Asset","Economic Depreciation Rate",
                                  "Tax Depreciation Rate"]
                        )
        
        # Calculating the depreciation rates for each industry:
        for i in range(0, bea_wb_nsht-2):
            # The current industry sheet that is being used from the BEA input. 
            ind_sht = bea_wb.sheet_by_index(i+1)
            start_position = self.__search_worksheet(ind_sht, "asset codes",
                                                     25, False)

            self.dummy1 = np.array(ind_sht.col_values(
                    ind_sht.ncols-1, start_position[0]+3,
                    ind_sht.nrows-1))
            self.rates["Economic Depreciation Rate"][i] = sum(
                    self.dummy1*self.__assets.rates.iloc[:,2]/sum(self.dummy1)
                    )
            
        self.rates.to_csv(output_folder + rates_out_file, index = False)
    
    
    '''
    This function searches through a worksheet along the up-right diagonals for
        a given search term.  The function searches up through the diagonal
        defined by the "distance" parameter.
    '''
    def __search_worksheet(self, sheet, search_term, distance, warnings = True):
        '''
        Parameters: sheet - The worksheet to be searched through.
                    entry - What is being searched for in the worksheet.
                            Numbers must be written with at least one decimal 
                                place, e.g. 15.0, 0.0, 21.74.
                    distance - Search up to and through this diagonal.

        Returns:    A vector of the position of the first entry found.
                    If not found then [-1,-1] is returned.
        '''
    
        # Calculating the total number of cells that need to be searched.
        final_search = ((distance+1)*distance)/2
        # Keeping track of the current diagonal being searched.
        current_diagonal = 1
        # The total number of columns in the sheet being searched.
        total_columns  = sheet.ncols
        total_rows = sheet.nrows
    
        # Search through the cells in the search area until the item is found:
        for n in xrange(0, final_search):
            # Incrementing the current diagonal being searched:
            if ((current_diagonal+1)*current_diagonal)/2 < n+1:
                current_diagonal += 1
            # Calculating the indices of the current cell being looked at:
            i = ((current_diagonal+1)*current_diagonal)/2 - (n+1)
            j = current_diagonal - i - 1
            # Making sure the program doesn't read past the end of file:
            if i >= total_rows and j >= total_columns:
                break
            elif i >= total_rows:
                continue
            elif j >= total_columns:
                continue
            # If the search term is found, return indices of the current cell:
            if str(sheet.cell_value(i,j)).lower() == str(search_term).lower():
                return [i,j]
    
        # If warnings are on (default) and item was not found:
        if warnings:
            print "Warning: No such search entry found in the specified search space."
            print "Check sample worksheet and consider changing distance input."
        # Default return value if the search item was not found.
        return [-1,-1]
        
    
    






