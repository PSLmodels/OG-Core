############################################################
# Get tax data from TaxCalculator                          #
# Author: Kerk Phillips                                    #
# Last Update: 7/30/12                                     #
############################################################

# Import Statements
from get_micro_data import get_data
import pickle
import numpy as np
import csv

# get_data(False)

datafile = open('micro_data_baseline.pkl', 'r')
data = pickle.load(datafile)
datafile.close()

data2016 = {'2016': data['2016']}
data2016 = data2016.values()


        