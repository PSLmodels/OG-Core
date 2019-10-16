import numpy as np
import pandas as pd
import os
import pickle
from ogusa.constants import VAR_LABELS, ToGDP_LABELS
from ogusa.utils import save_return_table
cur_path = os.path.split(os.path.abspath(__file__))[0]


def macro_table(base_tpi, base_params, reform_tpi=None,
                reform_params=None,
                var_list=['Y', 'C', 'K', 'L', 'r', 'w'],
                output_type='pct_diff', num_years=10, include_SS=True,
                include_overall=True, start_year=2019,
                table_title=None, table_format=None,
                path=None):
    '''
    Create a table of macro aggregates.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-USA Specifications class): baseline parameters
            object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-USA Specifications class): reform parameters
            object
        var_list (list): names of variable to use in table
        output_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baselien
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform (reform-base)
            'levels': variables in model units
        num_years (integer): number of years to include in table
        include_SS (bool): whether to include the steady-state results
            in the table
        include_overall (bool): whether to include results over the
            entire budget window as a column in the table
        start_year (integer): year to start table
        table_title (string): title for plot
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    '''
    assert (isinstance(start_year, int))
    assert (isinstance(num_years, int))
    # Make sure both runs cover same time period
    if reform_tpi is not None:
        assert (base_params.start_year == reform_params.start_year)
    year_vec = np.arange(start_year, start_year + num_years)
    start_index = start_year - base_params.start_year
    # Check that reform included if doing pct_diff or diff plot
    if output_type == 'pct_diff' or output_type == 'diff':
        assert (reform_tpi is not None)
    year_list = year_vec.tolist()
    if include_overall:
        year_list.append(str(year_vec[0]) + '-' + str(year_vec[-1]))
    if include_SS:
        year_list.append('SS')
    table_dict = {'Year': year_list}
    for i, v in enumerate(var_list):
        if output_type == 'pct_diff':
            # multiple by 100 so in percentage points
            results = (((reform_tpi[v] - base_tpi[v]) / base_tpi[v]) *
                       100)
            results_years = results[start_index: start_index +
                                    num_years]
            results_overall = (
                (reform_tpi[v][start_index: start_index +
                               num_years].sum() -
                 base_tpi[v][start_index: start_index +
                             num_years].sum()) /
                base_tpi[v][start_index: start_index +
                            num_years].sum())
            results_SS = results[-1]
            results_for_table = results_years
            if include_overall:
                results_for_table = np.append(
                    results_for_table, results_overall)
            if include_SS:
                results_for_table = np.append(
                    results_for_table, results_SS)
            table_dict[VAR_LABELS[v]] = results_for_table
        elif output_type == 'diff':
            results = reform_tpi[v] - base_tpi[v]
            results_years = results[start_index: start_index +
                                    num_years]
            results_overall = (
                (reform_tpi[v][start_index: start_index +
                               num_years].sum() -
                 base_tpi[v][start_index: start_index +
                             num_years].sum()))
            results_SS = results[-1]
            results_for_table = results_years
            if include_overall:
                results_for_table = np.append(
                    results_for_table, results_overall)
            if include_SS:
                results_for_table = np.append(
                    results_for_table, results_SS)
            table_dict[VAR_LABELS[v]] = results_for_table
        else:
            results_years = base_tpi[v][start_index: start_index +
                                        num_years]
            results_overall = results_years.sum()
            results_SS = base_tpi[-1]
            results_for_table = results_years
            if include_overall:
                results_for_table = np.append(
                    results_for_table, results_overall)
            if include_SS:
                results_for_table = np.append(
                    results_for_table, results_SS)
            table_dict[VAR_LABELS[v] + ' Baseline'] = results_for_table
            if reform_tpi is not None:
                results_years = reform_tpi[v][start_index: start_index +
                                              num_years]
                results_overall = results_years.sum()
                results_SS = reform_tpi[-1]
                results_for_table = results_years
                if include_overall:
                    results_for_table = np.append(
                        results_for_table, results_overall)
                if include_SS:
                    results_for_table = np.append(
                        results_for_table, results_SS)
                table_dict[VAR_LABELS[v] + ' Reform'] = results_for_table
        # Make df with dict so can use pandas functions
        table_df = pd.DataFrame.from_dict(table_dict, orient='columns'
                                          ).set_index('Year').transpose()
        table = save_return_table(table_df, table_format, path)

    return table


def macro_table_SS(base_ss, reform_ss,
                   var_list=['Yss', 'Css', 'Kss', 'Lss', 'rss', 'wss'],
                   table_title=None, table_format=None, path=None):
    '''
    Create a table of macro aggregates from the steady-state solutions.

    Args:
        base_ss (dictionary): SS output from baseline run
        reform_ss (dictionary): SS output from reform run
        var_list (list): names of variable to use in table
        table_title (string): title for plot
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    '''
    table_dict = {'Variable': [], 'Baseline': [], 'Reform': [],
                  '% Change (or pp diff)': []}
    for i, v in enumerate(var_list):
        table_dict['Variable'].append(VAR_LABELS[v])
        table_dict['Baseline'].append(base_ss[v])
        table_dict['Reform'].append(reform_ss[v])
        if v != 'D/Y':
            diff = (reform_ss[v] - base_ss[v]) / base_ss[v]
        else:
            diff = (reform_ss['Dss'] / reform_ss['Yss'] -
                    base_ss['Dss'] / base_ss['Yss'])
        table_dict['% Change (or pp diff)'].append(diff)
        # Make df with dict so can use pandas functions
        table_df = pd.DataFrame.from_dict(
            table_dict, orient='columns').set_index('Variable')
        table = save_return_table(table_df, table_format, path,
                                  precision=3)

    return table
