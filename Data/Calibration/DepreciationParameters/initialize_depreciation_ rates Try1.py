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

import excel_toolbox as tb

'''
------------------------------------------------------------------------
    Read in all the industry titles, as well as the corresponding BEA and NAICS
    codes.
------------------------------------------------------------------------
The data comes from the U.S. Bureau of Economic Analysis:
http://www.bea.gov/national/FA2004/Details/Index.html

------------------------------------------------------------------------
'''

# Working directory, note that there should already be an "OUTPUT" file here.
path = os.getcwd()

# Opening up the BEA's depreciation rates by industry data file.
bea_book = xlrd.open_workbook(path + "\data\detailnonres_stk1.xlsx")
bea_book_sheet_names = bea_book.sheet_names()  #For efficiency
bea_book_nsheets = bea_book.nsheets
bea_sheet0 = bea_book.sheet_by_index(0)

# Finding the position of "Industry Title" on first worksheet.
start_position = tb.search_worksheet(bea_sheet0, "bea code", 25, False)
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
M = np.zeros(shape=(bea_book_nsheets-1,2), dtype=object)
row_index = start_position[0] + 1
col_index = start_position[1] - 1

for i in range(0, bea_book_nsheets-1):
    for row_index in range(start_position[0]+1, bea_sheet0.nrows):
        #if(row_index >= bea_sheet0.nrows):
        #    break
        if(str(bea_codes_text1[i]) == str(bea_sheet0.cell_value(row_index,col_index+1))):
            M[i,0] = str(bea_sheet0.cell_value(row_index,col_index)).replace('\xa0', ' ').lstrip()
            M[i,1] = str(bea_sheet0.cell_value(row_index,col_index+1))
            break
        elif(str(bea_codes_text1[i]) == str(bea_sheet0.cell_value(row_index,col_index+1))[:-2]):
            M[i,0] = str(bea_sheet0.cell_value(row_index,col_index)).replace('\xa0', ' ').lstrip()
            M[i,1] = str(bea_sheet0.cell_value(row_index,col_index+1))[:-2]
            break

# For each industry, calculating 
# "\xa0".replace("\xa0", " ")



    