'''
------------------------------------------------------------------------
This script reads in data generated from the OSPC Tax Calcultor and
the 2009 IRS PUF.  It then plots these data to help in visualizing
the relationshiop between average effective tax rates and marginal
effective rates rates and income and age.

This Python script calls the following functions:


This Python script outputs the following:

------------------------------------------------------------------------
'''

# Import packages
import numpy as np
import os.path
import numpy.random as rnd
import scipy.optimize as opt
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


'''
------------------------------------------------------------------------
Load OSPC Tax Calculator Data into a Dataframe
------------------------------------------------------------------------
'''
data_path = '/Users/rwe2/Documents/OSPC/Data/micro-dynamic/2015_tau_n.csv'
# data_path = '/Users/jasondebacker/repos/microsimint/Data/2015_tau_n.csv'
A = pd.read_csv(data_path)
A['Total Labor Income'] = A['Wage and Salaries'] + A['Self-Employed Income']
A["Effective Tax Rate"] = A['Total Tax Liability']/A["Adjusted Total income"]
A["Total Capital Income"] = A['Adjusted Total income'] - A['Total Labor Income']
B = A[['Age', 'Total Labor Income', 'Total Capital Income',
             'Adjusted Total income', 'Effective Tax Rate', 'Weights']]

# Clean up the data
# drop all obs with AETR > 0.5
B = B.drop(B[B['Effective Tax Rate'] >0.5].index)
# drop all obs with AETR < -0.15
B = B.drop(B[B['Effective Tax Rate'] < -0.15].index)
# drop all obs with ATI < $5
B = B.drop(B[B['Adjusted Total income'] <5].index)

# Create an array of the different ages in the data
min_age = int(np.maximum(B['Age'].min(),21))
max_age = int(B['Age'].max())
age_groups = np.arange(min_age, max_age+1)

for i in age_groups:
    df = B[B['Age'] == i]
    # print some desciptive stats
    print 'Descriptive Statistics for age == ', i
    print df.describe()

    # # plot effective tax rate as a function of adjusted total income
    # # without truncating income
    # B.plot(kind='scatter', x='Adjusted Total income',
    #        y='Effective Tax Rate', color='DarkGreen')
    # plt.title('Effective tax rate and ATI, age ' + str(i) + ', non-truncated income')
    # plt.show()

    # plot effective tax rate as a function of adjusted total income
    # with truncated income
    C = B.drop(B[B['Adjusted Total income'] >500000].index)
    C.plot(kind='scatter', x='Adjusted Total income',
           y='Effective Tax Rate', color='DarkGreen')
    # plt.title('Effective tax rate and ATI, age ' + str(i) + ', truncated income')
    plt.show()
