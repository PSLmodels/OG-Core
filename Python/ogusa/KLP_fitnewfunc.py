############################################################
# Get tax data from TaxCalculator                          #
# Author: Kerk Phillips                                    #
# Last Update: 7/30/12                                     #
############################################################

# Import Statements
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
from KLP_TaxFuncs import TaxFuncCES
import scipy.optimize as opt

def TaxSSD(x, y, par, nobs):

    # initialize sum of squared deviations
    SSD = 0.0

    # caclulate squared deviations over all observations and sum
    for i in range(0, nobs):
        pred = TaxFuncCES(x[i], y[i], par)
        SSD = SSD + (ATR[i] - pred)**2

    # print(par, SSD)

    return SSD


# read in csv file
datafile = open('KLP_2016incomedata35.csv', 'r')    
data = pandas.read_csv(datafile)
data = np.array(data)

# unpack variables
MTRx = data[:,0]
MTRy = data[:,1]
ATR = data[:,2]
x = data[:,3]
y = data[:,4]

# find number of observations
dimen = data.shape
nobs = dimen[0]

# choose guesses for parameter starting values
parguess = np.array([1.0E-10, .0001, 1.0E-10, .0001,
                     -.1, 0.0, .35, .25,
                     .5, 2.0])

# choose min and max values for each parameter
eps = .000000001
parbounds = ((eps, None), (eps, None), (eps, None), (eps, None),
          (eps - 1.0, 1.0 - eps), (eps - 1.0, 1.0 - eps), (eps - 1.0, 1.0 - eps), (eps - 1.0, 1.0 - eps),
          (eps, 1.0 - eps), (eps, None))

# Lambda Function for Finding par using fmincon
SSD = lambda par:TaxSSD(x, y, par, nobs)
look = TaxSSD(x, y, parguess, nobs)
print(look)

# find best fitting parameters
parfit = opt.minimize(SSD, parguess, bounds = parbounds, method = 'L-BFGS-B')
print(parfit)