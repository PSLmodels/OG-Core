''' Work in Progress
------------------------------------------------------------------------
Last updated 01/13/2015

Initialize the depriciation rate for every BEA code.

This py-file calls the following other file(s):
            data/detailnonres_stk1.xlsx

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import xlrd
import os.path
import numpy as np
import pandas as pd

'''
------------------------------------------------------------------------
    Read in all the industry titles, as well as the corresponding BEA and NAICS
    codes.
------------------------------------------------------------------------
The data comes from the U.S. Bureau of Economic Analysis:
http://www.bea.gov/national/FA2004/Details/Index.html

------------------------------------------------------------------------
'''

def search_worksheet(sheet, search_term, distance, warnings = True):
    '''
    Parameters: sheet - The worksheet to be searched through.
                entry - What is being searched for in the worksheet.
                        Numbers must be written with at least one decimal 
                        place, e.g. 15.0, 0.0, 21.74.
                distance - Search up to and through this diagonal.

    Returns:    A vector of the position of the first entry found.
                If not found then [-1,-1] is returned.
    '''
    final_search = ((distance+1)*distance)/2
    current_diagonal = 1
    total_columns  = sheet.ncols
    for n in xrange(0, final_search):
        if ((current_diagonal+1)*current_diagonal)/2 < n+1:
            current_diagonal += 1
        
        i = ((current_diagonal+1)*current_diagonal)/2 - (n+1)
        j = current_diagonal - i - 1
        
        if j >= total_columns:
            continue
        
        if str(sheet.cell_value(i,j)).lower() == str(search_term).lower():
            return [i,j]
    
    if warnings:
        print "Warning: No such search entry found in the specified search space."
        print "Check sample worksheet and consider changing distance input."
    
    return [-1,-1]


# Working directory, note that there should already be an "OUTPUT" file here.
path = os.getcwd()

# Opening up the BEA's depreciation rates by industry data file.
bea_book = xlrd.open_workbook(path + "\data\detailnonres_stk1.xlsx")
bea_book_sheet_names = bea_book.sheet_names()  #For efficiency
bea_book_nsheets = bea_book.nsheets
bea_sheet0 = bea_book.sheet_by_index(0)

# Finding the position of "Industry Title" on first worksheet.
start_position = search_worksheet(bea_sheet0, "bea code", 25, False)
row_index = start_position[0] + 1
col_index = start_position[1] - 1

# Finding total number of industries listed--including those without bea codes.
number_of_industries = 0
while row_index < bea_sheet0.nrows:
    if(str(bea_sheet0.cell_value(row_index, col_index)) != ""):
        number_of_industries += 1
    row_index += 1

# Making a list of BEA codes based off the names of the worksheets.
bea_codes_text1 = np.zeros(bea_book_nsheets-1, dtype=object)
for index in xrange(1, bea_book_nsheets):
    bea_codes_text1[index-1] = str(bea_book_sheet_names[index])

# Making a list of BEA codes codes based off the first worksheet.
array_index = 0
row_index = start_position[0] + 1
col_index = start_position[1] - 1
bea_codes_text2 = np.zeros(number_of_industries, dtype=object)

while row_index < bea_sheet0.nrows:
    if(str(bea_sheet0.cell_value(row_index, col_index)) != ""):
        bea_codes_text2[array_index] = str(bea_sheet0.cell_value(row_index, 
                                            col_index+1))
        array_index += 1
    row_index += 1

    
# Checking that the two lists match up one-to-one.
# Creating a matrix listed in order based off the order of the worksheets
M = np.zeros(shape=(bea_book_nsheets-2,2), dtype=object)
row_index = start_position[0] + 1
col_index = start_position[1] - 1

for i in range(0, bea_book_nsheets-2):
    for row_index in range(start_position[0]+1, bea_sheet0.nrows):
        if(str(bea_codes_text1[i]) == str(bea_sheet0.cell_value(row_index,col_index+1))):
            M[i,0] = str(bea_sheet0.cell_value(row_index,col_index)).replace('\xa0', ' ').strip()
            M[i,1] = str(bea_sheet0.cell_value(row_index,col_index+1))
            break
        elif(str(bea_codes_text1[i]) == str(bea_sheet0.cell_value(row_index,col_index+1))[:-2]):
            M[i,0] = str(bea_sheet0.cell_value(row_index,col_index)).replace('\xa0', ' ').strip()
            M[i,1] = str(bea_sheet0.cell_value(row_index,col_index+1))[:-2]
            break


# For each industry, calculating 
for i in range(0, bea_book_nsheets-2):  # change 1 to bea_book_nsheets-1 for final program
                       ##########
    if(M[i,0] == 0):
        continue
    bea_current_sheet = bea_book.sheet_by_index(i+1)
    start_position = search_worksheet(bea_current_sheet, "asset codes", 25, False)
    year = int(bea_current_sheet.cell_value(start_position[0],bea_current_sheet.ncols-1))
    
    a = np.array(bea_current_sheet.col_values(bea_current_sheet.ncols-1, start_position[0]+2, bea_current_sheet.nrows-1))
    b = 5
    