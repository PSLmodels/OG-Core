'''
------------------------------------------------------------------------
This program extracts tax rate and income data from CSV files output
from the  TPC microsimulation model.
------------------------------------------------------------------------
'''
import pandas as pd
import numpy as np
import pickle
import os

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_PATH = os.path.join(CUR_DIR, 'TPC_microsim_output',
                            'TPC_baseline')


# will keep only the get_data function
# will take as arguments: path to files, start year, end year
# will return taxcalc version to fit other OG-USA functions, but will return a
# null value

# To work with TPC output, will import txfunc from OG-USA, will use this
# get_data to read in data to it
# estimate tax functions and save them (may have to do a modified txfunc that
# calls this new get_data script)
# then go back and run OG-USA as normal with paths to cached tax function
# parameter estimates
def get_data(baseline=True, start_year=2020, end_year=2030,
             path=DEFAULT_PATH):
    '''
    This function creates dataframes of micro data with marginal tax
    rates and information to compute effective tax rates from the
    Tax-Calculator output.  The resulting dictionary of dataframes is
    returned and saved to disk in a pickle file.

    Args:
        baseline (bool): True if using data from the microsimulation
            baseline
        start_year (int): first year of microsim results to use
        end_year (int): last year of microsim results to use
        path (str): path to folder containing the annual CSV files

    Returns:
        micro_data_dict (dict): dict of Pandas Dataframe, one for each
            year from start_year to end_year

    '''
    # assert that keyword args are ok
    assert end_year >= start_year
    assert start_year >= 2020
    assert end_year <= 2030

    # read CSV files into DataFrames to return
    micro_data_dict = {}
    for y in range(start_year, end_year + 1):
        file = os.path.join(path, str(y) + '_named.csv')
        df = pd.read_csv(file)
        # rename some vars and do simple calculations
        df['total_tax_liab'] = df['DEFICIT'] + df['PAYROLLTAX']
        df['total_labinc'] = df['WS'] + df['BUSINC']
        df.rename(columns={
            'WGT': 'weight', 'HEAD_AGE': 'age',
            'MARKET_INCOME': 'market_income',
            'PAYROLLTAX': 'payroll_tax_liab',
            'MTR_INCINC': 'MTR_INTINC', 'MTR_SHRGAIN': 'MTR_CGSHR',
            'MTR_RENTROY': 'MTR_TOTRNTROY',
            'MTR_PARTNERSHIP_NONPASS': 'MTR_PART_NONPASS',
            'MTR_PARTNERSHIP_PASS': 'MTR_PART_PASS',
            'MTR_SCORP_NONPASS': 'MTR_S_NONPASS',
            'MTR_SCORP_PASS': 'MTR_S_PASS',
            'MTR_ESTATESTRUSTS': 'MTR_ESTNET'}, inplace=True)
        df['mtr_labinc'] = ((
            df['MTR_WS_HEAD'] * df['WS'] + df['MTR_BUSINC_HEAD'] *
            df['BUSINC']) / (df['WS'] + df['BUSINC']))
        df['total_capinc'] = df['market_income'] - df['total_labinc']
        df['etr'] = df['total_tax_liab'] / df['market_income']
        # Compute mtr on capital income
        df['mtr_capinc'] = cap_inc_mtr(df)
        # Keep just columns interested in later
        df = df[['payroll_tax_liab', 'age', 'market_income', 'weight',
                 'year', 'total_tax_liab', 'total_labinc',
                 'total_capinc', 'mtr_labinc', 'mtr_capinc',
                 'etr']].copy()
        # add DataFrame to dict
        micro_data_dict[str(y)] = df

    if baseline:
        pkl_path = "micro_data_baseline.pkl"
    else:
        pkl_path = "micro_data_policy.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(micro_data_dict, f)

    return micro_data_dict, 'TPC model'  # second arg is model version


def cap_inc_mtr(df):  # pragma: no cover
    '''
    This function computes the marginal tax rate on capital income,
    which is calculated as a weighted average of the marginal tax rates
    on different sources of capital income.

    Args:
        df (Pandas DataFrame): Data from TPC microsimualtion model

    Returns:
        mtr_combined_capinc (Numpy array): array with marginal tax rates
            for each observation in df

    '''
    capital_income_sources = (
        'INTINC', 'TXEXINT', 'TAXDIV', 'QUALDIV', 'TAXIRA', 'PENAGI',
        'CGSHR', 'CGLNG', 'TOTRNTROY', 'PART_NONPASS',
        'PART_PASS', 'S_NONPASS', 'S_PASS', 'ESTNET')

    # calculating MTRs separately - can skip items with zero tax
    all_mtrs = {income_source: df['MTR_' + income_source].to_numpy() for
                income_source in capital_income_sources}
    # Get each column of income sources, to include non-taxable income
    record_columns = [df[x].to_numpy() for x in capital_income_sources]
    # Compute weighted average of all those MTRs
    # first find total capital income
    total_cap_inc = (sum(map(abs, record_columns)))
    capital_mtr = [abs(col) * all_mtrs[source] for col, source in
                   zip(record_columns, capital_income_sources)]
    mtr_combined_capinc = np.zeros_like(total_cap_inc)
    mtr_combined_capinc[total_cap_inc != 0] = (
        sum(capital_mtr)[total_cap_inc != 0] /
        total_cap_inc[total_cap_inc != 0])
    # If no capital income, make MTR equal to that on interest income
    mtr_combined_capinc[total_cap_inc == 0] = (
        all_mtrs['INTINC'][total_cap_inc == 0])
    return mtr_combined_capinc
