'''
------------------------------------------------------------------------
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
from ogusa.utils import DEFAULT_START_YEAR, TC_LAST_YEAR, RECORDS_START_YEAR


def get_calculator(baseline, calculator_start_year, reform=None,
                   data=None, weights=None, records_start_year=RECORDS_START_YEAR):
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
    if data is not None and "cps" in data:
        records1 = Records.cps_constructor()
    elif data is not None:
        records1 = Records(data=data, weights=weights,
                           start_year=records_start_year)
    else:
        records1 = Records()

    if baseline:
        # Should not be a reform if baseline is True
        assert not reform

    if not baseline:
        print("REFORM", reform)
        print("TYPE", type(reform))
        policy1.implement_reform(reform)

    # the default set up increments year to 2013
    calc1 = Calculator(records=records1, policy=policy1)

    # this increment_year function extrapolates all PUF variables to
    # the next year so this step takes the calculator to the start_year
    if calculator_start_year > TC_LAST_YEAR:
        raise RuntimeError("Start year is beyond data extrapolation.")
    while calc1.current_year < calculator_start_year:
        calc1.increment_year()

    # running all the functions and calculates taxes
    calc1.calc_all()

    return calc1


def get_data(baseline=False, start_year=DEFAULT_START_YEAR, reform={},
             data=None):
    '''
    --------------------------------------------------------------------
    This function creates dataframes of micro data from the
    tax calculator
    --------------------------------------------------------------------
    INPUTS:
    baseline        = boolean, =True if baseline tax policy,
                               =False if reform
    start_year      = integer, first year of budget window
    reform          = dictionary, reform parameters

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    micro_data_dict = dictionary, contains pandas dataframe for each
                      year of budget window.  Dataframe contain mtrs,
                      etrs, income variables, age from tax-calculator
                      and PUF-CPS match

    OUTPUT:
        ./micro_data_policy.pkl
        ./micro_data_baseline.pkl

    RETURNS: micro_data_dict
    --------------------------------------------------------------------
    '''
    calc1 = get_calculator(baseline=baseline,
                           calculator_start_year=start_year,
                           reform=reform, data=data)

    # running marginal tax rate function for wage and salaries of
    # primary three results returned for fica tax, iit tax, and combined
    # mtr_iit: marginal tax rate of individual income tax
    [mtr_fica, mtr_iit, mtr_combined] = calc1.mtr('e00200p')

    # the sum of the two e-variables here are self-employed income
    [mtr_fica_sey, mtr_iit_sey, mtr_combined_sey] = calc1.mtr('e00900p')

    # find mtr on capital income
    mtr_combined_capinc = cap_inc_mtr(calc1)

    # create a temporary array to save all variables we need
    length = len(calc1.array('s006'))
    temp = np.empty((length, 11))
    # Put values of variables in temp array
    # most e-variable definition can be found here
    # https://docs.google.com/spreadsheets/d/1WlgbgEAMwhjMI8s9eG117bBEKFioXUY0aUTfKwHwXdA/edit#gid=1029315862
    temp[:, 0] = mtr_combined
    temp[:, 1] = mtr_combined_sey
    temp[:, 2] = mtr_combined_capinc
    temp[:, 3] = calc1.array('age_head')
    temp[:, 4] = calc1.array('e00200')
    temp[:, 5] = calc1.array('sey')
    temp[:, 6] = calc1.array('sey') + calc1.array('e00200')
    temp[:, 7] = calc1.array('expanded_income')
    temp[:, 8] = calc1.array('combined')
    temp[:, 9] = calc1.current_year * np.ones(length)
    temp[:, 10] = calc1.array('s006')

    # dictionary of data frames to return
    micro_data_dict = {}

    micro_data_dict[str(start_year)] = DataFrame(
        data=temp, columns=['MTR wage', 'MTR self-employed Wage',
                            'MTR capital income', 'Age',
                            'Wage and Salaries',
                            'Self-Employed Income',
                            'Wage + Self-Employed Income',
                            'Adjusted Total income',
                            'Total Tax Liability', 'Year', 'Weights'])

    # Repeat the process for each year
    # Increment years into the future but not beyond TC_LAST_YEAR
    for i in range(TC_LAST_YEAR - start_year):
        calc1.increment_year()
        [mtr_fica, mtr_iit, mtr_combined] = calc1.mtr('e00200p')
        [mtr_fica_sey, mtr_iit_sey, mtr_combined_sey] =\
            calc1.mtr('e00900p')
        # find mtr on capital income
        mtr_combined_capinc = cap_inc_mtr(calc1)


        temp = np.empty((length, 11))
        temp[:, 0] = mtr_combined
        temp[:, 1] = mtr_combined_sey
        temp[:, 2] = mtr_combined_capinc
        temp[:, 3] = calc1.array('age_head')
        temp[:, 4] = calc1.array('e00200')
        temp[:, 5] = calc1.array('sey')
        temp[:, 6] = calc1.array('sey') + calc1.array('e00200')
        temp[:, 7] = calc1.array('expanded_income')
        temp[:, 8] = calc1.array('combined')
        temp[:, 9] = calc1.current_year * np.ones(length)
        temp[:, 10] = calc1.array('s006')

        micro_data_dict[str(calc1.current_year)] = DataFrame(
            data=temp, columns=['MTR wage', 'MTR self-employed Wage',
                                'MTR capital income', 'Age',
                                'Wage and Salaries',
                                'Self-Employed Income',
                                'Wage + Self-Employed Income',
                                'Adjusted Total income',
                                'Total Tax Liability', 'Year',
                                'Weights'])
        print 'year: ', str(calc1.current_year)

    if reform:
        pkl_path = "micro_data_policy.pkl"
    else:
        pkl_path = "micro_data_baseline.pkl"
    pickle.dump(micro_data_dict, open(pkl_path, "wb"))

    return micro_data_dict


def cap_inc_mtr(calc1):
    # find mtr on capital income
    capital_income_sources_taxed = (
        'e00300', 'e00400', 'e00600', 'e00650', 'e01400', 'e01700',
        'p22250', 'p23250', 'e26270')

    # PUF does not have variable for non-taxable IRA distributions
    capital_income_sources = (
        'e00300', 'e00400', 'e00600', 'e00650', 'e01400', 'e01700',
        'p22250', 'p23250', 'e26270')

    # calculating MTRs separately - can skip items with zero tax
    all_mtrs = {income_source: calc1.mtr(income_source) for
                income_source in capital_income_sources_taxed}
    # Get each column of income sources, to include non-taxable income
    record_columns = [calc1.array(x) for x in capital_income_sources]
    # weighted average of all those MTRs
    total = (sum(map(abs, record_columns)) +
             np.abs(calc1.array('e02000') - calc1.array('e26270')))
    # Note that all_mtrs gives fica (0), iit (1), and combined (2) mtrs
    # We'll use the combined - hence all_mtrs[source][2]
    capital_mtr = [abs(col) * all_mtrs[source][2] for col, source in
                   zip(record_columns, capital_income_sources_taxed)]
    mtr_combined_capinc = np.zeros_like(total)
    mtr_combined_capinc[total != 0] = (
        sum(capital_mtr + (calc1.mtr('e02000')[2] *
                           np.abs(calc1.array('e02000') -
                                  calc1.array('e26270'))))[total != 0] /
        total[total != 0])
    # no capital income taxpayers
    # give all the weight to interest income
    mtr_combined_capinc[total == 0] = all_mtrs['e00300'][2][total == 0]

    return mtr_combined_capinc
