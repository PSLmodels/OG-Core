import numpy as np
import pandas as pd
from ogusa.utils import save_return_table
from ogusa.constants import VAR_LABELS


def tax_rate_table(base_TxFuncEst, base_params, reform_TxFuncEst=None,
                   reform_params=None, rate_type='ETR', start_year=2019,
                   num_years=10, table_format='tex', path=None):
    '''
    Table of average tax rates over several years.

    Args:
        base_TxFuncEst(dictionary): Baseline tax function parameter
            estimates
        reform_TxFuncEst (dictionary): Reform tax function parameter
            estimates
        rate_type (string): Tax rate to include in table
        start_year (integer): year to start table
        num_years (integer): number of years to include in table
        table_format (string): format to save/return table as
        path (string): path to save table to

    Returns:
        table (string or DataFrame): table of tax rates
    '''
    assert (isinstance(start_year, int))
    assert (isinstance(num_years, int))
    # Make sure both runs cover same time period
    if reform_TxFuncEst is not None:
        assert (base_params.start_year == reform_params.start_year)
    start_index = start_year - base_params.start_year
    years = list(np.arange(start_year, start_year + num_years, 1))
    if reform_TxFuncEst is None:
        if rate_type == 'ETR':
            rates = base_TxFuncEst['tfunc_avg_etr'] * 100
        elif rate_type == 'MTRx':
            rates = base_TxFuncEst['tfunc_avg_mtrx'] * 100
        elif rate_type == 'MTRy':
            rates = base_TxFuncEst['tfunc_avg_mtry'] * 100
        elif rate_type == 'all':
            etr_rates = base_TxFuncEst['tfunc_avg_etr'] * 100
            mtrx_rates = base_TxFuncEst['tfunc_avg_mtrx'] * 100
            mtry_rates = base_TxFuncEst['tfunc_avg_mtry'] * 100
        else:
            raise ValueError(
                'Value {!r} is not a valid rate_type'.format(rate_type))
        if rate_type == 'all':
            # In case num_years is greater than number of years
            # tax function estimates are for
            len_rates = len(etr_rates[start_index: start_index +
                                      num_years])
            table = {'Year': years[:len_rates],
                     VAR_LABELS['ETR']:
                     etr_rates[start_index: start_index + num_years],
                     VAR_LABELS['MTRx']:
                     mtrx_rates[start_index: start_index + num_years],
                     VAR_LABELS['MTRy']:
                     mtry_rates[start_index: start_index + num_years]}
        else:
            len_rates = len(rates[start_index: start_index + num_years])
            table = {'Year': years[:len_rates],
                     VAR_LABELS[rate_type]:
                     rates[start_index: start_index + num_years]}
    else:
        if rate_type == 'ETR':
            base_rates = base_TxFuncEst['tfunc_avg_etr'] * 100
            reform_rates = reform_TxFuncEst['tfunc_avg_etr'] * 100
        elif rate_type == 'MTRx':
            base_rates = base_TxFuncEst['tfunc_avg_mtrx'] * 100
            reform_rates = reform_TxFuncEst['tfunc_avg_mtrx'] * 100
        elif rate_type == 'MTRy':
            base_rates = base_TxFuncEst['tfunc_avg_mtrx'] * 100
            reform_rates = reform_TxFuncEst['tfunc_avg_mtrx'] * 100
        elif rate_type == 'all':
            base_etr_rates = base_TxFuncEst['tfunc_avg_etr'] * 100
            base_mtrx_rates = base_TxFuncEst['tfunc_avg_mtrx'] * 100
            base_mtry_rates = base_TxFuncEst['tfunc_avg_mtry'] * 100
            reform_etr_rates = reform_TxFuncEst['tfunc_avg_etr'] * 100
            reform_mtrx_rates = reform_TxFuncEst['tfunc_avg_mtrx'] * 100
            reform_mtry_rates = reform_TxFuncEst['tfunc_avg_mtry'] * 100
        else:
            raise ValueError(
                'Value {!r} is not a valid rate_type'.format(rate_type))
        if rate_type == 'all':
            len_rates = len(base_etr_rates[start_index: start_index +
                                           num_years])
            table = {
                'Year': years[:len_rates],
                'Baseline ' + VAR_LABELS['ETR']:
                base_etr_rates[start_index: start_index + num_years],
                'Reform ' + VAR_LABELS['ETR']:
                reform_etr_rates[start_index: start_index + num_years],
                'Differences in ' + VAR_LABELS['ETR']:
                reform_etr_rates[start_index: start_index + num_years]
                - base_etr_rates[start_index: start_index + num_years],
                'Baseline ' + VAR_LABELS['MTRx']:
                base_mtrx_rates[start_index: start_index + num_years],
                'Reform ' + VAR_LABELS['MTRx']:
                reform_mtrx_rates[start_index: start_index + num_years],
                'Differences in ' + VAR_LABELS['MTRx']:
                reform_mtrx_rates[start_index: start_index + num_years]
                - base_mtrx_rates[start_index: start_index + num_years],
                'Baseline ' + VAR_LABELS['MTRy']:
                base_mtry_rates[start_index: start_index + num_years],
                'Reform ' + VAR_LABELS['MTRy']:
                reform_mtry_rates[start_index: start_index + num_years],
                'Differences in ' + VAR_LABELS['MTRy']:
                reform_mtry_rates[start_index: start_index + num_years]
                - base_mtry_rates[start_index: start_index + num_years]}
        else:
            len_rates = len(base_rates[start_index: start_index +
                                       num_years])
            table = {'Year': years[:len_rates],
                     'Baseline ' + VAR_LABELS[rate_type]:
                     base_rates[start_index: start_index + num_years],
                     'Reform ' + VAR_LABELS[rate_type]:
                     reform_rates[start_index: start_index + num_years],
                     'Difference':
                     reform_rates[start_index: start_index + num_years]
                     - base_rates[start_index: start_index + num_years]}
    table_df = (pd.DataFrame.from_dict(table, orient='columns')).transpose()
    table_df.columns = table_df.iloc[0].astype('int').astype('str')
    table_df.reindex(table_df.index.drop('Year'))
    table_str = save_return_table(table_df, table_format, path,
                                  precision=2)

    return table_str
