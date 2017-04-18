''' Work in Progress
------------------------------------------------------------------------
Last updated 01/12/2015

This file contains custom functions for processing excel files with the
    imported xlrd module.

This py-file call no others files.


This py-file may contain functions that create file(s).  In which case, make
    sure that an OUTPUT folder exists before using these functions.
------------------------------------------------------------------------
'''
'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''


'''
------------------------------------------------------------------------

------------------------------------------------------------------------
'''

# The xlrd package must be imported in the calling python program.
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
        






