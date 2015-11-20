'''
------------------------------------------------------------------------
This program extracts tax rate and income data from the microsimulation
model (tax-calculator) and saves it as csv files.
------------------------------------------------------------------------
'''
from taxcalc import *
import pandas as pd
from pandas import DataFrame
import numpy as np
import copy


# create a calculator
policy1 = Policy()
records1 = Records()

reform = {
    2015: {
        '_II_rt1': [.17],
        '_II_rt2': [.17],
        '_II_rt3': [.17],
        '_II_rt4': [.17],
        '_II_rt5': [.17],
        '_II_rt6': [.17],
        '_II_rt7': [.17],
        '_CG_rt3': [.25],
        '_AMT_CG_rt3': [.25],
        '_EITC_c': [[1003, 3859, 6048, 6742]],
        '_EITC_rt': [[0.1265, 0.3900, 0.4500, 0.5000]],
        '_EITC_ps': [[18240, 28110, 28110, 28110]]
    }, }

policy1.implement_reform(reform)

# the default set up increments year to 2013
calc1 = Calculator(records=records1, policy=policy1)

# this increment_year function extrapolates all PUF variables to the next year
# so this step takes the calculator to 2015
calc1.increment_year()
calc1.increment_year()

# running all the functions and calculates taxes
calc1.calc_all()

# running marginal tax rate function for wage and salaries of primary
# three results returned, but we only need mtr_iit for now
# mtr_iit: marginal tax rate of individual income tax
[mtr_fica, mtr_iit, mtr_combined] = calc1.mtr('e00200p')

# the sum of the two e-variables here are self-employed income
[mtr_fica_sey, mtr_iit_sey, mtr_combined_sey] = calc1.mtr('e00900p')

# create a temporary array to save all variables we need
length = len(calc1.records.s006)
temp = np.empty([length, 10])

# most variables can be retrieved from calculator's Record class
# by add the variable name after (calc.records._____)
# most e-variable definition can be found here https://docs.google.com/spreadsheets/d/1WlgbgEAMwhjMI8s9eG117bBEKFioXUY0aUTfKwHwXdA/edit#gid=1029315862
# e00200 - wage and salaries, _sey - self-employed income
temp[:,0] = mtr_iit
temp[:,1] = mtr_iit_sey
temp[:,2] = calc1.records.age
temp[:,3] = calc1.records.e00200
temp[:,4] = calc1.records._sey
temp[:,4] = calc1.records._sey + calc1.records.e00200
temp[:,6] = calc1.records._expanded_income
temp[:,7] = calc1.records._iitax
temp[:,8] = calc1.current_year * np.ones(length)
temp[:,9] = calc1.records.s006

# convert the array to DataFrame and export
tau_n = DataFrame(data = temp,
                  columns = ['MTR wage', 'MTR self-employed Wage','Age',
                             'Wage and Salaries', 'Self-Employed Income','Wage + Self-Employed Income',
                             'Adjusted Total income', 'Total Tax Liability', 'Year', 'Weights'])
tau_n.to_csv('2015_tau_n.csv')


# repeat the process for each year
for i in range(1,10):
    calc1.increment_year()

    [mtr_fica, mtr_iit, mtr_combined] = calc1.mtr('e00200p')
    [mtr_fica_sey, mtr_iit_sey, mtr_combined_sey] = calc1.mtr('e00900p')

    temp = np.empty([length, 10])
    temp[:,0] = mtr_iit
    temp[:,1] = mtr_iit_sey
    temp[:,2] = calc1.records.age
    temp[:,3] = calc1.records.e00200
    temp[:,4] = calc1.records._sey
    temp[:,4] = calc1.records._sey + calc1.records.e00200
    temp[:,6] = calc1.records._expanded_income
    temp[:,7] = calc1.records._iitax
    temp[:,8] = calc1.current_year * np.ones(length)
    temp[:,9] = calc1.records.s006

    df = DataFrame(data = temp,
                   columns = ['MTR wage', 'MTR self-employed Wage','Age',
                              'Wage and Salaries', 'Self-Employed Income','Wage + Self-Employed Income',
                              'Adjusted Total income','Total Tax Liability','Year', 'Weights'])
    df.to_csv(str(calc1.current_year) + '_tau_n.csv')
    tau_n.append(df, ignore_index = True)
