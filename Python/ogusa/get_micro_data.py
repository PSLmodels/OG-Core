'''
------------------------------------------------------------------------
This program extracts tax rate and income data from the microsimulation
model (tax-calculator) and saves it as csv files.
------------------------------------------------------------------------
'''

import sys
sys.path.insert(0, '/Users/jasondebacker/repos/tax-calculator')

from taxcalc import *
import pandas as pd
from pandas import DataFrame
import numpy as np
import copy
import numba
import pickle


def get_data(baseline=False, reform={}):
    '''
    --------------------------------------------------------------------
    This function creates dataframes of micro data from the 
    tax calculator
    --------------------------------------------------------------------

    --------------------------------------------------------------------
    '''

    # create a calculator
    policy1 = Policy()
    records1 = Records()

    """reform = {
    2015: {
        '_II_rt1': [.09],
        '_II_rt2': [.135],
        '_II_rt3': [.225],
        '_II_rt4': [.252],
        '_II_rt5': [.297],
        '_II_rt6': [.315],
        '_II_rt7': [0.3564],
    }, }"""

    if not baseline:
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

    # find mtr on capital income
    capital_income_sources_taxed = ('e00300', 'e00400', 'e00600',
                                'e00650', 'e01400',
                                'e01700', 'e02000',
                                'e22250','e23250')

    # note that use total pension income (e01500) since don't have both the 
    # taxable (e01700) and non-taxable pension income separately
    # don't appear to have variable for non-taxable IRS distributions
    capital_income_sources = ('e00300', 'e00400', 'e00600',
                                'e00650', 'e01400',
                                'e01500', 'e02000',
                                'e22250','e23250')

    # calculating MTRs separately - can skip items with zero tax
    all_mtrs = {income_source: calc1.mtr(income_source) for income_source in capital_income_sources_taxed}
    # Get each column of income sources - need to include non-taxable capital income
    record_columns = [getattr(calc1.records, income_source) for income_source in capital_income_sources]
    # weighted average of all those MTRs
    total = sum(record_columns)
    # i.e., capital_gain_mtr = (e00300 * mtr_iit_300 + e00400 * mtr_iit_400 + ... + e23250 * mtr_iit_23250) /
    #                           sum_of_all_ten_variables
    # Note that all_mtrs gives fica (0), iit (1), and combined (2) mtrs  - we'll use the combined - hence all_mtrs[source][2]
    capital_gain_mtr = [ col * all_mtrs[source][2] for col, source in zip(record_columns, capital_income_sources_taxed)]
    mtr_combined_capinc = sum(capital_gain_mtr) / total

    #Get the total of every capital income source

    #if every item in capital_income_sources == 0: # no capital income taxpayers
    if np.all(total == 0): # no capital income taxpayers
        mtr_combined_capinc = all_mtrs['e00300'][2] # give all the weight to interest income

    # create a temporary array to save all variables we need
    length = len(calc1.records.s006)
    temp = np.empty([length, 11])

    # most variables can be retrieved from calculator's Record class
    # by add the variable name after (calc.records._____)
    # most e-variable definition can be found here https://docs.google.com/spreadsheets/d/1WlgbgEAMwhjMI8s9eG117bBEKFioXUY0aUTfKwHwXdA/edit#gid=1029315862
    # e00200 - wage and salaries, _sey - self-employed income
    temp[:,0] = mtr_combined
    temp[:,1] = mtr_combined_sey
    temp[:,2] = mtr_combined_capinc
    temp[:,3] = calc1.records.age
    temp[:,4] = calc1.records.e00200
    temp[:,5] = calc1.records._sey
    temp[:,6] = calc1.records._sey + calc1.records.e00200
    temp[:,7] = calc1.records._expanded_income
    temp[:,8] = calc1.records._combined
    temp[:,9] = calc1.current_year * np.ones(length)
    temp[:,10] = calc1.records.s006

    # convert the array to DataFrame and export

    # dictionary of data frames to return
    micro_data_dict = {}

    micro_data_dict['2015'] = DataFrame(data = temp,
                      columns = ['MTR wage', 'MTR self-employed Wage', 'MTR capital income','Age',
                                 'Wage and Salaries', 'Self-Employed Income','Wage + Self-Employed Income',
                                 'Adjusted Total income', 'Total Tax Liability', 'Year', 'Weights'])


    # repeat the process for each year
    for i in range(1,10):
        calc1.increment_year()

        [mtr_fica, mtr_iit, mtr_combined] = calc1.mtr('e00200p')
        [mtr_fica_sey, mtr_iit_sey, mtr_combined_sey] = calc1.mtr('e00900p')

        temp = np.empty([length, 11])
        temp[:,0] = mtr_combined
        temp[:,1] = mtr_combined_sey
        temp[:,2] = mtr_combined_capinc
        temp[:,3] = calc1.records.age
        temp[:,4] = calc1.records.e00200
        temp[:,5] = calc1.records._sey
        temp[:,6] = calc1.records._sey + calc1.records.e00200
        temp[:,7] = calc1.records._expanded_income
        temp[:,8] = calc1.records._combined
        temp[:,9] = calc1.current_year * np.ones(length)
        temp[:,10] = calc1.records.s006

        micro_data_dict[str(calc1.current_year)] = DataFrame(data = temp,
                       columns = ['MTR wage', 'MTR self-employed Wage','MTR capital income','Age',
                                  'Wage and Salaries', 'Self-Employed Income','Wage + Self-Employed Income',
                                  'Adjusted Total income','Total Tax Liability','Year', 'Weights'])
        print 'year: ', i
    
    pkl_path = "micro_data_w_capmtr_policy_12142015.pkl"
    pickle.dump(micro_data_dict, open(pkl_path, "wb"))

    return micro_data_dict
