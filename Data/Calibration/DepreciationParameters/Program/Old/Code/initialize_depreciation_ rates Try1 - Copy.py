''' Work in Progress
------------------------------------------------------------------------
Last updated 01/09/2015

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
#import numpy as np

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

#def find_entry(sheet, entry, distance):
#    '''
#    Parameters: sheet - The worksheet to be searched through.
#                entry - What is being searched for in the worksheet.
#                        Numbers must be written with at least one decimal 
#                        place, e.g. 15.0, 0.0, 21.74.
#                distance - Search up to and through this diagonal.
#
#    Returns:    A vector of the position of the first entry found.
#                If not found then ****
#    '''
#    c = ((distance+1)*distance)/2
#    m = 1
#    columns  = sheet0.ncols
#    for n in xrange(0, c):
#        if ((m+1)*m)/2 < n+1:
#            m = m+1
#        i = ((m+1)*m)/2 - (n+1)
#        j = m - i - 1
#        
#        #Note, would
#        if j >= columns:
#            continue
#        if str(sheet0.cell_value(i,j)).lower() == str(entry).lower():
#            #print("(" + str(i) + "," + str(j) +")" + "\n1\n2\n3\n")
#            return [i,j]
#            #break



path = os.getcwd()

book = xlrd.open_workbook(path + "\data\detailnonres_stk1.xlsx")

sheet0 = book.sheet_by_index(0)

#a = find_entry(sheet0, "bea code", 25)
#print a
#print "{}".format(a[0])

b = tb.search_worksheet(sheet0, 0, 5, False)
print b



#if sheet0.cell_value(18,1) == "Electro medical instruments":
#    print 1
#else:
#    print 0

#find_entry(sheet0, 20)
#distance = 20
#c = ((distance+1)*distance)/2
#m = 1
#columns = sheet0.ncols
#for n in range(0, c):
#    if ((m+1)*m)/2 < n+1:
#        m = m+1
#    i = ((m+1)*m)/2 - (n+1)
#    j = m - i - 1
#    if j >= columns:
#        continue
#    #print "(" + str(i) +"," + str(j) + ")
#    l = str(sheet0.cell_value(i,j)).lower()
#    if i == 14:
#        dummy1 = 0
#    if str(sheet0.cell_value(i,j)).lower() == "bea code": #######
#        print("(" + str(i) + "," + str(j) +")" + "\n1\n2\n3\n")
#        break





#print(sheet1.cell_value(18,1))
#total_sheets = book.nsheets



#print(os.curdir)
#os.chdir('N:\Lott, Sherwin\OLG Dynamic Scoring Model\Calibration\Firm " 
#            "Depreciation Parameters\Code')
#print(os.curdir)
#os.


#sheet = w.get_sheet(1)



