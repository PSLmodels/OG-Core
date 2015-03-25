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
    Packages
------------------------------------------------------------------------
'''
import xlrd
import os.path
import numpy as np
import pandas as pd
#
import naics_processing as naics

'''
------------------------------------------------------------------------
    Read in all the industry titles, as well as the corresponding BEA and NAICS
    codes.
------------------------------------------------------------------------
The data comes from the U.S. Bureau of Economic Analysis:
http://www.bea.gov/national/FA2004/Details/Index.html

------------------------------------------------------------------------
'''

# Working directory:
path = os.getcwd()
# Relevant path and file names:
data_folder = os.path.abspath(path + "\\data")
output_folder = os.path.abspath(path + "\\OUTPUT")
# Opening up the BEA's depreciable assets by industry data file.
bea_book = xlrd.open_workbook(os.path.abspath(data_folder + "\\detailnonres_stk1.xlsx"))
bea_book_sheet_names = bea_book.sheet_names()
bea_book_nsheets = bea_book.nsheets
bea_sheet0 = bea_book.sheet_by_index(0)
# Finding the position of "Industry Title" on first worksheet.
start_position = naics.search_ws(bea_sheet0, "bea code", 25, False, [0,0], True)
row_index = start_position[0] + 1
col_index = start_position[1] - 1
# Find the corresponding NAICS Codes
naics_position = start_position
cur_search = naics.search_ws(bea_sheet0, "naics codes", 25, False, np.array(naics_position) + [0,1])
while cur_search != [-1,-1]:
    naics_position = np.array(cur_search)
    cur_search = naics.search_ws(bea_sheet0, "naics codes", 25, False, np.array(naics_position) + [0,1])
naics_col_index = naics_position[1]
# Finding total number of industries listed--including those without bea codes.
number_of_industries = 0
while row_index < bea_sheet0.nrows:
    if(str(bea_sheet0.cell_value(row_index, col_index)) != ""):
        number_of_industries += 1
    row_index += 1
# Making a list of BEA codes based off the names of the worksheets.
bea_codes1 = np.zeros(bea_book_nsheets-1, dtype=object)
for index in xrange(1, bea_book_nsheets):
    bea_codes1[index-1] = str(bea_book_sheet_names[index])
# Making a list of BEA codes codes based off the first worksheet.
array_index = 0
row_index = start_position[0] + 1
col_index = start_position[1] - 1
bea_codes2 = np.zeros(number_of_industries, dtype=object)
naics_codes = np.zeros(number_of_industries, dtype=object)
while row_index < bea_sheet0.nrows:
    if(str(bea_sheet0.cell_value(row_index, col_index)) != ""):
        bea_codes2[array_index] = str(bea_sheet0.cell_value(row_index, col_index+1)).replace("\xa0", " ").strip()
        naics_codes[array_index] =  str(bea_sheet0.cell_value(row_index, naics_col_index)).replace(".0","")
        array_index += 1
    row_index += 1
# Checking that the two lists match up one-to-one.
# Creating a matrix listed in order based off the order of the worksheets
chart_cols = ["Industry","BEA Code","NAICS Code"]
bea_chart = pd.DataFrame(np.zeros(shape=(bea_book_nsheets-2,3), dtype=object), columns = chart_cols)
row_index = start_position[0] + 1
col_index = start_position[1] - 1

for i in range(0, bea_book_nsheets-2):
    for row_index in range(start_position[0]+1, bea_sheet0.nrows):
        if(str(bea_codes1[i]) == str(bea_sheet0.cell_value(row_index,col_index+1))):
            bea_chart["Industry"][i] = str(bea_sheet0.cell_value(row_index,col_index)).replace('\xa0', ' ').strip()
            bea_chart["BEA Code"][i] = str(bea_sheet0.cell_value(row_index,col_index+1))
            bea_chart["NAICS Code"][i] = str(bea_sheet0.cell_value(row_index,naics_col_index))
            break
        elif(str(bea_codes1[i]) == str(bea_sheet0.cell_value(row_index,col_index+1))[:-2]):
            bea_chart["Industry"][i] = str(bea_sheet0.cell_value(row_index,col_index)).replace('\xa0', ' ').strip()
            bea_chart["BEA Code"][i] = str(bea_sheet0.cell_value(row_index,col_index+1))[:-2]
            bea_chart["NAICS Code"][i] = str(bea_sheet0.cell_value(row_index,naics_col_index))[:-2]
            break

bea_current_sheet = bea_book.sheet_by_name(bea_chart["BEA Code"][0])
start_position = naics.search_ws(bea_current_sheet, "asset codes", 25, False)
bea_table = pd.DataFrame(np.zeros((bea_current_sheet.nrows-(start_position[0]+1)-1,len(bea_chart["BEA Code"]))), columns = bea_chart["BEA Code"])
# For each industry, calculating 
for i in bea_chart["BEA Code"]:
    bea_current_sheet = bea_book.sheet_by_name(i)
    start_position = naics.search_ws(bea_current_sheet, "asset codes", 25, False)
    year = int(bea_current_sheet.cell_value(start_position[0],bea_current_sheet.ncols-1))
    bea_table[i] = np.array(bea_current_sheet.col_values(bea_current_sheet.ncols-1, start_position[0]+1, bea_current_sheet.nrows-1))
    #print np.array(bea_current_sheet.col_values(bea_current_sheet.ncols-1, start_position[0]+1, bea_current_sheet.nrows-1))
    #print 10 * np.array(bea_current_sheet.col_values(bea_current_sheet.ncols-1, start_position[0]+1, bea_current_sheet.nrows-1))
# The dollar amounts are in millions:

bea_table = bea_table.convert_objects(convert_numeric=True).fillna(0)
bea_table = bea_table * 1000000


#bea_table.iloc[:,0]



