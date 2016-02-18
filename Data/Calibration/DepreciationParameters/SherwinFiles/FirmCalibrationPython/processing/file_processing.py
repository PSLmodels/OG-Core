"""
File Processing (file_processing.py):
-------------------------------------------------------------------------------
Last updated 7/2/2015

This module creates functions for generic file operations.
"""
# Packages
import os
import sys


def get_file(dirct, contains=[""]):
    """ Given a directory, this function finds a file that contains each
    string in an array of strings. It returns the string 
    
    :param dirct: The directory to be searched in.
    :param contains: An array of strings, all of which must be contained in
           the filename
    """
    for filename in os.listdir(dirct):
        if all(item in filename for item in contains):
            return filename

# Note: move the naics_processing search_ws function here.
# Will require changing every instance it is called in.

