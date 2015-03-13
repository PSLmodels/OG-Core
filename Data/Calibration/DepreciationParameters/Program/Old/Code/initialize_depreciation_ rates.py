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

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import scipy.optimize as opt
import os.path

'''
------------------------------------------------------------------------

------------------------------------------------------------------------
'''

path = os.getcwd()

a = pd.read_excel(path + "\data\detailnonres_stk1.xlsx", 1, header=None)
pd.read_excel()
b = 5
c = 5

