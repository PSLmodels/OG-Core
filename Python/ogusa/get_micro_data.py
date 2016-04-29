'''
------------------------------------------------------------------------
Late updated 4/7/2016

This program extracts tax rate and income data from the microsimulation
model (tax-calculator) and saves it in pickle files.

This module defines the following functions:
    get_data()


This Python script calls the following functions:
    get_micro_data.py
    taxcalc
    
This py-file creates the following other file(s):
    ./TAX_ESTIMATE_PATH/TxFuncEst_baseline{}.pkl
    ./TAX_ESTIMATE_PATH/TxFuncEst_policy{}.pkl

------------------------------------------------------------------------
'''

import sys
import taxcalc
from taxcalc import *
import pandas as pd
from pandas import DataFrame
import numpy as np
import copy
import numba
import pickle

def only_growth_assumptions(user_mods, start_year):
    """
    Extract any reform parameters that are pertinent to growth
    assumptions
    """
    growth_dd = taxcalc.growth.Growth.default_data(start_year=start_year)
    ga = {}
    for year, reforms in user_mods.items():
        overlap = set(growth_dd.keys()) & set(reforms.keys())
        if overlap:
            ga[year] = {param:reforms[param] for param in overlap}
    return ga


def only_reform_mods(user_mods, start_year):
    """
    Extract parameters that are just for policy reforms
    """
    pol_refs = {}
    beh_dd = Behavior.default_data(start_year=start_year)
    growth_dd = taxcalc.growth.Growth.default_data(start_year=start_year)
    policy_dd = taxcalc.policy.Policy.default_data(start_year=start_year)
    for year, reforms in user_mods.items():
        all_cpis = {p for p in reforms.keys() if p.endswith("_cpi") and
                    p[:-4] in policy_dd.keys()}
        pols = set(reforms.keys()) - set(beh_dd.keys()) - set(growth_dd.keys())
        pols &= set(policy_dd.keys())
        pols ^= all_cpis
        if pols:
            pol_refs[year] = {param:reforms[param] for param in pols}
    return pol_refs

def get_calculator(baseline, calculator_start_year, reform=None, data=None, weights=None, records_start_year=None):
    '''
    --------------------------------------------------------------------
    This function creates the tax calculator object for the microsim
    --------------------------------------------------------------------
    INPUTS:
    baseline                 = boolean, True if baseline tax policy
    calculator_start_year    = integer, first year of budget window
    reform                   = dictionary, reform parameters
    data                     = DataFrame for Records object (opt.)
    weights                  = weights DataFrame for Records object (opt.)
    records_start_year       = the start year for the data and weights dfs

    RETURNS: Calculator object with a current_year equal to
             calculator_start_year
    --------------------------------------------------------------------

    '''
    # create a calculator
    policy1 = Policy()
    if data is not None:
        records1 = Records(data=data, weights=weights, start_year=records_start_year)
    else:
        records1 = Records()

    if baseline:
        #Should not be a reform if baseline is True
        assert not reform

    growth_assumptions = only_growth_assumptions(reform, calculator_start_year)
    reform_mods = only_reform_mods(reform, calculator_start_year)

    if not baseline:
        policy1.implement_reform(reform_mods)

    # the default set up increments year to 2013
    calc1 = Calculator(records=records1, policy=policy1)

    if growth_assumptions:
        calc1.growth.update_economic_growth(growth_assumptions)

    # this increment_year function extrapolates all PUF variables to the next year
    # so this step takes the calculator to the start_year
    for i in range(calculator_start_year-2013):
        calc1.increment_year()

    return calc1


def get_data(baseline=False, start_year=2016, reform={}):
    '''
    --------------------------------------------------------------------
    This function creates dataframes of micro data from the
    tax calculator
    --------------------------------------------------------------------
    INPUTS:
    baseline        = boolean, =True if baseline tax policy, =False if reform 
    start_year      = integer, first year of budget window
    reform          = dictionary, reform parameters

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    micro_data_dict = dictionary, contains pandas dataframe for each year 
                      of budget window.  Dataframe contain mtrs, etrs, income variables, age
                      from tax-calculator and PUF-CPS match

    OUTPUT:
        ./micro_data_policy.pkl
        ./micro_data_baseline.pkl

    RETURNS: micro_data_dict
    --------------------------------------------------------------------
    '''

    calc1 = get_calculator(baseline=baseline, calculator_start_year=start_year,
                           reform=reform)

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
                                'p22250','p23250')

    # note that use total pension income (e01500) since don't have both the
    # taxable (e01700) and non-taxable pension income separately
    # don't appear to have variable for non-taxable IRA distributions
    # capital_income_sources = ('e00300', 'e00400', 'e00600',
    #                             'e00650', 'e01400',
    #                             'e01500', 'e02000',
    #                             'p22250','p23250')
    capital_income_sources = ('e00300', 'e00400', 'e00600',
                                'e00650', 'e01400',
                                'e01700', 'e02000',
                                'p22250','p23250')

    # calculating MTRs separately - can skip items with zero tax
    all_mtrs = {income_source: calc1.mtr(income_source) for income_source in capital_income_sources_taxed}
    # Get each column of income sources - need to include non-taxable capital income
    record_columns = [getattr(calc1.records, income_source) for income_source in capital_income_sources]
    # weighted average of all those MTRs
    # weighted average of all those MTRs
    total = sum(map(abs,record_columns))
    # i.e., capital_gain_mtr = (e00300 * mtr_iit_300 + e00400 * mtr_iit_400 + ... + e23250 * mtr_iit_23250) /
    #                           sum_of_all_ten_variables
    # Note that all_mtrs gives fica (0), iit (1), and combined (2) mtrs  - we'll use the combined - hence all_mtrs[source][2]
    capital_gain_mtr = [ abs(col) * all_mtrs[source][2] for col, source in zip(record_columns, capital_income_sources_taxed)]
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
    temp[:,3] = calc1.records.age_head
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

    micro_data_dict[str(start_year)] = DataFrame(data = temp,
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
        temp[:,3] = calc1.records.age_head
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
        print 'year: ', str(calc1.current_year)

    if reform:
        pkl_path = "micro_data_policy.pkl"
    else:
        pkl_path = "micro_data_baseline.pkl"
    pickle.dump(micro_data_dict, open(pkl_path, "wb"))

    return micro_data_dict
