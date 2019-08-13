'''
------------------------------------------------------------------------
This program extracts tax rate and income data from the microsimulation
model (Tax-Calculator).
------------------------------------------------------------------------
'''
from taxcalc import *
from pandas import DataFrame
from dask.distributed import Client
from dask import compute, delayed
import dask.multiprocessing
import numpy as np
import pickle
import pkg_resources
from ogusa.utils import DEFAULT_START_YEAR, TC_LAST_YEAR, PUF_START_YEAR


def get_calculator(baseline, calculator_start_year, reform=None,
                   data=None, gfactors=None, weights=None,
                   records_start_year=PUF_START_YEAR):
    '''
    This function creates the tax calculator object with the policy
    specified in reform and the data specified with the data kwarg.

    Args:
        baseline (boolean): True if baseline tax policy
        calculator_start_year (int): first year of budget window
        reform (dictionary): IIT policy reform parameters, None if
            baseline
        data (DataFrame or str): DataFrame or path to datafile for
            Records object
        gfactors (Tax-Calculator GrowthFactors object): growth factors
            to use to extrapolate data over budget window
        weights (DataFrame): weights for Records object
        records_start_year (int): the start year for the data and
            weights dfs (default is set to the PUF start year as defined
            in the Tax-Calculator project)

    Returns:
        calc1 (Tax-Calculator Calculator object): Calulaotr object with
            current_year equal to calculator_start_year

    '''
    # create a calculator
    policy1 = Policy()
    if data is not None and "cps" in data:
        records1 = Records.cps_constructor()
        # impute short and long term capital gains if using CPS data
        # in 2012 SOI data 6.587% of CG as short-term gains
        records1.p22250 = 0.06587 * records1.e01100
        records1.p23250 = (1 - 0.06587) * records1.e01100
        # set total capital gains to zero
        records1.e01100 = np.zeros(records1.e01100.shape[0])
    elif data is not None:
        records1 = Records(data=data, gfactors=gfactors, weights=weights,
                           start_year=records_start_year)
    else:
        records1 = Records()

    if baseline:
        if not reform:
            print("Running current law policy baseline")
        else:
            print("Baseline policy is: ", reform)
    else:
        if not reform:
            print("Running with current law as reform")
        else:
            print("Reform policy is: ", reform)
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
             data=None, client=None, num_workers=1):
    '''
    This function creates dataframes of micro data with marginal tax
    rates and information to compute effective tax rates from the
    Tax-Calculator output.  The resulting dictionary of dataframes is
    returned and saved to disk in a pickle file.

    Args:
        baseline (boolean): True if baseline tax policy
        calculator_start_year (int): first year of budget window
        reform (dictionary): IIT policy reform parameters, None if
            baseline
        data (DataFrame or str): DataFrame or path to datafile for
            Records object
        client (Dask Client object): client for Dask multiprocessing
        num_workers (int): number of workers to use for Dask
            multiprocessing

    Returns:
        micro_data_dict (dict): dict of Pandas Dataframe, one for each
            year from start_year to the maximum year Tax-Calculator can
            analyze
        taxcalc_version (str): version of Tax-Calculator used

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
        data=temp, columns=['MTR wage income', 'MTR SE income',
                            'MTR capital income', 'Age', 'Wage income',
                            'SE income', 'Wage + SE income',
                            'Adjusted total income',
                            'Total tax liability', 'Year', 'Weights'])

    # Repeat the process for each year
    # Increment years into the future but not beyond TC_LAST_YEAR
    lazy_values = []
    for year in range(start_year + 1, TC_LAST_YEAR + 1):
        lazy_values.append(
            delayed(taxcalc_advance)(calc1, year, length))
    results = compute(*lazy_values, scheduler=dask.multiprocessing.get,
                      num_workers=num_workers)
    # for i, result in results.items():
    for i, result in enumerate(results):
        year = start_year + 1 + i
        micro_data_dict[str(year)] = DataFrame(
            data=result, columns=['MTR wage income', 'MTR SE income',
                                  'MTR capital income', 'Age',
                                  'Wage income', 'SE income',
                                  'Wage + SE income',
                                  'Adjusted total income',
                                  'Total tax liability', 'Year',
                                  'Weights'])

    if reform:
        pkl_path = "micro_data_policy.pkl"
    else:
        pkl_path = "micro_data_baseline.pkl"
    pickle.dump(micro_data_dict, open(pkl_path, "wb"))

    # Do some garbage collection
    del (calc1, temp, mtr_fica, mtr_iit, mtr_combined, mtr_fica_sey,
         mtr_iit_sey, mtr_combined_sey, mtr_combined_capinc)

    taxcalc_version = pkg_resources.get_distribution("taxcalc").version
    return micro_data_dict, taxcalc_version


def taxcalc_advance(calc1, year, length):
    '''
    This function advances the year used in Tax-Calculator and save the
    results to a numpy array.

    Args:
        calc1 (Tax-Calculator Calculator object): TC calculator
        year (int): year to advance data to
        length (int): length of TC Records object

    Returns:
        temp (Numpy array): an array of microdata with marginal tax
            rates and other information computed in TC
    '''
    calc1.advance_to_year(year)
    print('year: ', str(calc1.current_year))
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

    return temp


def cap_inc_mtr(calc1):
    '''
    This function computes the marginal tax rate on capital income,
    which is calculated as weighted average of the marginal tax rates
    on different sources of capital income.

    Args:
        calc1 (Tax-Calculator Calculator object): TC calculator

    Returns:
        mtr_combined_capinc (Numpy array): array with marginal tax rates
            for each observation in the TC Records object

    '''
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
