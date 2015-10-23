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
A = pd.read_csv('/Users/jasondebacker/repos/microsimint/Data/2015_tau_n.csv')
A['Wage + Self-Employed Income'] = A['Wage and Salaries'] + A['Self-Employed Income']
A["Effective Tax Rate"] = A['Total Tax Liability']/A["Adjusted Total income"]
A["Total Capital Income"] = A['Adjusted Total income'] - A['Wage + Self-Employed Income']

# print some descriptive stats
print 'Descriptive statistics: ', A['Effective Tax Rate'].describe()


'''
------------------------------------------------------------------------
Plot tax rates over income 
------------------------------------------------------------------------
'''

# drop all obs with AETR > 1
A = A.drop(A[A['Effective Tax Rate'] >1].index)
# drop all obs with AETR < 0
A = A.drop(A[A['Effective Tax Rate'] <0].index)
# drop all obs with ATI < $5
A = A.drop(A[A['Adjusted Total income'] <5].index)
#  Keep just those aged 45-55
A = A.drop(A[A['Age'] <45].index)
A = A.drop(A[A['Age'] >55].index)
A.plot(kind='scatter', x='Adjusted Total income', y='Effective Tax Rate', color='DarkGreen')
plt.show()



