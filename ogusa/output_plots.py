import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
from ogusa.constants import VAR_LABELS, ToGDP_LABELS, GROUP_LABELS
import ogusa.utils as utils
cur_path = os.path.split(os.path.abspath(__file__))[0]
style_file = os.path.join(cur_path, 'OGUSAplots.mplstyle')
plt.style.use(style_file)


def plot_aggregates(base_tpi, base_params, reform_tpi=None,
                    reform_params=None, var_list=['Y', 'C', 'K', 'L'],
                    plot_type='pct_diff', num_years_to_plot=50,
                    start_year=2019, vertical_line_years=None,
                    plot_title=None, path=None):
    '''
    Create a plot of macro aggregates.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-USA Specifications class): baseline parameters object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-USA Specifications class): reform parameters object
        var_list (list): names of variable to plot
        plot_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baselien
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform (reform-base)
            'levels': plot variables in model units
            'cbo': plots variables in levels relative to CBO baseline
                projection (only available for macro variables in CBO
                long-term forecasts)
        num_years_to_plot (integer): number of years to include in plot
        start_year (integer): year to start plot
        vertical_line_years (list): list of integers for years want
            vertical lines at
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of macro aggregates

    '''
    assert (isinstance(start_year, int))
    assert (isinstance(num_years_to_plot, int))
    # Make sure both runs cover same time period
    if reform_tpi is not None:
        assert (base_params.start_year == reform_params.start_year)
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    start_index = start_year - base_params.start_year
    # Check that reform included if doing pct_diff or diff plot
    if plot_type == 'pct_diff' or plot_type == 'diff':
        assert (reform_tpi is not None)
    fig1, ax1 = plt.subplots()
    for i, v in enumerate(var_list):
        if plot_type == 'pct_diff':
            plot_var = (reform_tpi[v] - base_tpi[v]) / base_tpi[v]
            ylabel = r'Pct. change'
            plt.plot(year_vec,
                     plot_var[start_index: start_index +
                              num_years_to_plot], label=VAR_LABELS[v])
        elif plot_type == 'diff':
            plot_var = reform_tpi[v] - base_tpi[v]
            ylabel = r'Difference (Model Units)'
            plt.plot(year_vec,
                     plot_var[start_index: start_index +
                              num_years_to_plot], label=VAR_LABELS[v])
        elif plot_type == 'levels':
            plt.plot(year_vec,
                     base_tpi[v][start_index: start_index +
                                 num_years_to_plot],
                     label='Baseline ' + VAR_LABELS[v])
            if reform_tpi is not None:
                plt.plot(year_vec,
                         reform_tpi[v][start_index: start_index +
                                       num_years_to_plot],
                         label='Reform ' + VAR_LABELS[v])
            ylabel = r'Model Units'
        elif plot_type == 'cbo':
            # This option is not complete.  Need to think about how to
            # best load CBO forecasts
            plt.plot(year_vec,
                     base_tpi[v][start_index: start_index +
                                 num_years_to_plot],
                     label='Baseline ' + VAR_LABELS[v])
            plt.plot(year_vec,
                     reform_tpi[v][start_index: start_index +
                                   num_years_to_plot],
                     label='Reform ' + VAR_LABELS[v])
            ylabel = r'Trillions of \$'
        else:
            print('Please enter a valid plot type')
            assert(False)
    # vertical markers at certain years
    if vertical_line_years is not None:
        for yr in vertical_line_years:
            plt.axvline(x=yr, linewidth=0.5, linestyle='--', color='k')
    plt.xlabel(r'Year $t$')
    plt.ylabel(ylabel)
    if plot_title is not None:
        plt.title(plot_title, fontsize=15)
    vals = ax1.get_yticks()
    if plot_type == 'pct_diff':
        ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xlim((base_params.start_year - 1, base_params.start_year +
              num_years_to_plot))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path is not None:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight")
    else:
        return fig1
    plt.close()


def ss_3Dplot(base_params, base_ss, reform_params=None, reform_ss=None,
              var='bssmat_splus1', plot_type='levels', plot_title=None,
              path=None):
    '''
    Create a 3d plot of household decisions.

    Args:
        base_params (OG-USA Specifications class): baseline parameters object
        base_ss (dictionary): SS output from baseline run
        reform_params (OG-USA Specifications class): reform parameters object
        reform_ss (dictionary): SS output from reform run
        var (string): name of variable to plot
        plot_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baselien
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform (reform-base)
            'levels': plot variables in model units
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of household decisions

    '''
    if reform_params:
        assert(base_params.J == reform_params.J)
        assert(base_params.starting_age == reform_params.starting_age)
        assert(base_params.ending_age == reform_params.ending_age)
        assert(base_params.S ==  reform_params.S)
    domain = np.linspace(base_params.starting_age, base_params.ending_age,base_params.S)
    Jgrid = np.zeros(base_params.J)
    for j in range(base_params.J):
        Jgrid[j:] += base_params.lambdas[j]
    if plot_type == 'levels':
        data=base_ss[var].T
    elif plot_type == 'diff':
        data=(reform_ss[var]-base_ss[var]).T
    elif plot_type == 'pct_diff':
        data=((reform_ss[var]-base_ss[var])/base_ss[var]).T
    cmap1 = matplotlib.cm.get_cmap('jet')
    X, Y = np.meshgrid(domain, Jgrid)
    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')
    ax5.set_xlabel(r'age-$s$')
    ax5.set_ylabel(r'ability type-$j$')
    ax5.set_zlabel(r'individual savings $\bar{b}_{j,s}$')
    ax5.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=cmap1)
    if plot_title:
        plt.title(plot_title)
    if path:
        plt.savefig(path)
    else:
        return plt


def plot_gdp_ratio(base_tpi, base_params, reform_tpi=None,
                   reform_params=None, var_list=['D'],
                   num_years_to_plot=50,
                   start_year=2019, vertical_line_years=None,
                   plot_title=None, path=None):
    '''
    Create a plot of some variable to GDP.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-USA Specifications class): baseline parameters object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-USA Specifications class): reform parameters object
        p (OG-USA Specifications class): parameters object
        var_list (list): names of variable to plot
        num_years_to_plot (integer): number of years to include in plot
        start_year (integer): year to start plot
        vertical_line_years (list): list of integers for years want
            vertical lines at
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of ratio of a variable to GDP
    '''
    assert (isinstance(start_year, int))
    assert (isinstance(num_years_to_plot, int))
    # Make sure both runs cover same time period
    if reform_tpi is not None:
        assert (base_params.start_year == reform_params.start_year)
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    start_index = start_year - base_params.start_year
    fig1, ax1 = plt.subplots()
    for i, v in enumerate(var_list):
        plot_var_base = (base_tpi[v][:base_params.T] /
                         base_tpi['Y'][:base_params.T])
        if reform_tpi is not None:
            plot_var_reform = (reform_tpi[v][:base_params.T] /
                               reform_tpi['Y'][:base_params.T])
            plt.plot(year_vec, plot_var_base[start_index: start_index +
                                             num_years_to_plot],
                     label='Baseline ' + ToGDP_LABELS[v])
            plt.plot(year_vec, plot_var_reform[start_index: start_index +
                                               num_years_to_plot],
                     label='Reform ' + ToGDP_LABELS[v])
        else:
            plt.plot(year_vec, plot_var_base[start_index: start_index +
                                             num_years_to_plot],
                     label=ToGDP_LABELS[v])
    ylabel = r'Percent of GDP'
    # vertical markers at certain years
    if vertical_line_years is not None:
        for yr in vertical_line_years:
            plt.axvline(x=yr, linewidth=0.5, linestyle='--', color='k')
    plt.xlabel(r'Year $t$')
    plt.ylabel(ylabel)
    if plot_title is not None:
        plt.title(plot_title, fontsize=15)
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xlim((base_params.start_year - 1, base_params.start_year +
              num_years_to_plot))
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path is not None:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight")
    else:
        return fig1
    plt.close()


def ability_bar(base_tpi, base_params, reform_tpi,
                reform_params, var='n_mat', num_years=5,
                start_year=2019,
                plot_title=None, path=None):
    '''
    Plots percentage changes from baseline by ability group for a
    given variable.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-USA Specifications class): baseline parameters object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-USA Specifications class): reform parameters object
        var (string): name of variable to plot
        num_year (integer): number of years to compute changes over
        start_year (integer): year to start plot
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of results by ability type
    '''
    assert (isinstance(start_year, int))
    assert (isinstance(num_years, int))
    # Make sure both runs cover same time period
    if reform_tpi is not None:
        assert (base_params.start_year == reform_params.start_year)
    N = base_params.J
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars
    start_index = start_year - base_params.start_year
    omega_to_use = base_params.omega[:base_params.T, :].reshape(
        base_params.T, base_params.S, 1)
    base_val = (base_tpi[var] * omega_to_use)[
        start_index:start_index + num_years, :, :].sum(1).sum(0)
    reform_val = (reform_tpi[var] * omega_to_use)[
        start_index:start_index + num_years, :, :].sum(1).sum(0)
    var_to_plot = (reform_val - base_val) / base_val
    ax.bar(ind, var_to_plot * 100, width, bottom=0)
    ax.set_xticks(ind + width / 4)
    ax.set_xticklabels(('0-25%', '25-50%', '50-70%', '70-80%', '80-90%',
                        '90-99%', 'Top 1%'))
    plt.ylabel(r'Percentage Change in ' + VAR_LABELS[var])
    if plot_title is not None:
        plt.title(plot_title, fontsize=15)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path is not None:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight")
    else:
        return fig
    plt.close()


def ss_profiles(base_ss, base_params, reform_ss=None,
                reform_params=None, by_j=True, var='nssmat',
                plot_data=False,
                plot_title=None, path=None):
    '''
    Plot lifecycle profiles of given variable in the SS.

    Args:
        base_ss (dictionary): SS output from baseline run
        base_params (OG-USA Specifications class): baseline parameters
            object
        reform_ss (dictionary): SS output from reform run
        reform_params (OG-USA Specifications class): reform parameters
            object
        var (string): name of variable to plot
        plot_data (bool): whether to plot data values for given variable
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of lifecycle profiles

    '''
    if reform_ss is not None:
        assert (base_params.S == reform_params.S)
        assert (base_params.starting_age == reform_params.starting_age)
        assert (base_params.ending_age == reform_params.ending_age)
    age_vec = np.arange(base_params.starting_age,
                        base_params.starting_age + base_params.S)
    fig1, ax1 = plt.subplots()
    if by_j:
        cm = plt.get_cmap('coolwarm')
        ax1.set_prop_cycle(color=[cm(1. * i / 7) for i in range(7)])
        for j in range(base_params.J):
            plt.plot(age_vec, base_ss[var][:, j],
                     label='Baseline, j = ' + str(j))
            if reform_ss is not None:
                plt.plot(age_vec, reform_ss[var][:, j],
                         label='Reform, j = ' + str(j), linestyle='--')
    else:
        base_var = (
            base_ss[var][:, :] *
            base_params.lambdas.reshape(1, base_params.J)).sum(axis=1)
        plt.plot(age_vec, base_var, label='Baseline')
        if reform_ss is not None:
            reform_var = (
                reform_ss[var][:, :] *
                reform_params.lambdas.reshape(1, reform_params.J)).sum(axis=1)
            plt.plot(age_vec, reform_var, label='Reform', linestyle='--')
        if plot_data:
            assert var == 'nssmat'
            labor_file = utils.read_file(
                cur_path, "data/labor/cps_hours_by_age_hourspct.txt")
            data = pd.read_csv(labor_file, header=0, delimiter='\t')
            piv = data.pivot(index='age', columns='hours_pct',
                             values='mean_hrs')
            lab_mat_basic = np.array(piv)
            lab_mat_basic /= np.nanmax(lab_mat_basic)
            piv2 = data.pivot(index='age', columns='hours_pct',
                              values='num_obs')
            weights = np.array(piv2)
            weights /= np.nansum(weights, axis=1).reshape(
                60, 1)
            weighted = np.nansum((lab_mat_basic * weights), axis=1)
            weighted = np.append(weighted, np.zeros(20))
            weighted[60:] = np.nan
            plt.plot(age_vec, weighted, linewidth=2.0, label='Data',
                     linestyle=':')
    plt.xlabel(r'Age')
    plt.ylabel(VAR_LABELS[var])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if plot_title is not None:
        plt.title(plot_title, fontsize=15)
    if path is not None:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight")
    else:
        return fig1
    plt.close()


def plot_all(base_output_path, reform_output_path, save_path):
    '''
    Function to plot all default output plots.

    Args:
        base_output_path (str): path to baseline results
        reform_output_path (str): path to reform results
        save_path (str): path to save plots to

    Returns:
        None: All output figures saved to disk.

    '''
    # Read in data
    # Read in TPI output and parameters
    base_tpi = utils.safe_read_pickle(
        os.path.join(base_output_path, 'TPI', 'TPI_vars.pkl')
    )
    base_ss = utils.safe_read_pickle(
        os.path.join(base_output_path, 'SS', 'SS_vars.pkl')
    )

    base_params = utils.safe_read_pickle(
        os.path.join(base_output_path, 'model_params.pkl')
    )

    reform_tpi = utils.safe_read_pickle(
        os.path.join(reform_output_path, 'TPI', 'TPI_vars.pkl')
    )

    reform_ss = utils.safe_read_pickle(
        os.path.join(reform_output_path, 'SS', 'SS_vars.pkl')
    )

    reform_params = utils.safe_read_pickle(
        os.path.join(reform_output_path, 'model_params.pkl')
    )

    # Percentage changes in macro vars (Y, K, L, C)
    plot_aggregates(base_tpi, base_params, reform_tpi=reform_tpi,
                    reform_params=reform_params,
                    var_list=['Y', 'K', 'L', 'C'], plot_type='pct_diff',
                    num_years_to_plot=150,
                    vertical_line_years=[
                        base_params.start_year + base_params.tG1,
                        base_params.start_year + base_params.tG2],
                    plot_title='Percentage Changes in Macro Aggregates',
                    path=os.path.join(save_path, 'MacroAgg_PctChange.png'))

    # Percentage change in fiscal vars (D, G, TR, Rev)
    plot_aggregates(base_tpi, base_params, reform_tpi=reform_tpi,
                    reform_params=reform_params,
                    var_list=['D', 'G', 'TR', 'total_revenue'],
                    plot_type='pct_diff', num_years_to_plot=150,
                    vertical_line_years=[
                        base_params.start_year + base_params.tG1,
                        base_params.start_year + base_params.tG2],
                    plot_title='Percentage Changes in Fiscal Variables',
                    path=os.path.join(save_path, 'Fiscal_PctChange.png'))

    # r and w in baseline and reform -- vertical lines at tG1, tG2
    plot_aggregates(base_tpi, base_params, reform_tpi=reform_tpi,
                    reform_params=reform_params,
                    var_list=['r'],
                    plot_type='levels', num_years_to_plot=150,
                    vertical_line_years=[
                        base_params.start_year + base_params.tG1,
                        base_params.start_year + base_params.tG2],
                    plot_title='Real Interest Rates Under Baseline and Reform',
                    path=os.path.join(save_path, 'InterestRates.png'))

    plot_aggregates(base_tpi, base_params, reform_tpi=reform_tpi,
                    reform_params=reform_params,
                    var_list=['w'],
                    plot_type='levels', num_years_to_plot=150,
                    vertical_line_years=[
                        base_params.start_year + base_params.tG1,
                        base_params.start_year + base_params.tG2],
                    plot_title='Wage Rates Under Baseline and Reform',
                    path=os.path.join(save_path, 'WageRates.png'))

    # Debt-GDP in base and reform-- vertical lines at tG1, tG2
    plot_gdp_ratio(base_tpi, base_params, reform_tpi, reform_params,
                   var_list=['D'], num_years_to_plot=150,
                   start_year=2019, vertical_line_years=[
                           base_params.start_year + base_params.tG1,
                           base_params.start_year + base_params.tG2],
                   plot_title='Debt-to-GDP',
                   path=os.path.join(save_path, 'DebtGDPratio.png'))

    # Tax revenue to GDP in base and reform-- vertical lines at tG1, tG2
    plot_gdp_ratio(base_tpi, base_params, reform_tpi, reform_params,
                   var_list=['total_revenue'], num_years_to_plot=150,
                   start_year=2019, vertical_line_years=[
                           base_params.start_year + base_params.tG1,
                           base_params.start_year + base_params.tG2],
                   plot_title='Tax Revenue to GDP',
                   path=os.path.join(save_path, 'RevenueGDPratio.png'))

    # Pct change in c, n, b, y, etr, mtrx, mtry by ability group over 10 years
    var_list = ['c_path', 'n_mat', 'bmat_splus1', 'etr_path',
                'mtrx_path', 'mtry_path']
    title_list = ['consumption', 'labor supply', 'savings',
                  'effective tax rates',
                  'marginal tax rates on labor income',
                  'marginal tax rates on capital income']
    path_list = ['Cons', 'Labor', 'Save', 'ETR', 'MTRx', 'MTRy']
    for i, v in enumerate(var_list):
        ability_bar(base_tpi, base_params, reform_tpi, reform_params,
                    var=v, num_years=10, start_year=2019,
                    plot_title='Percentage changes in ' + title_list[i],
                    path=os.path.join(save_path, 'PctChange' +
                                      path_list[i] + '.png'))

    # lifetime profiles, base vs reform, SS for c, n, b, y - not by j
    var_list = ['cssmat', 'nssmat', 'bssmat_splus1', 'etr_ss',
                'mtrx_ss', 'mtry_ss']
    for i, v in enumerate(var_list):
        ss_profiles(base_ss, base_params, reform_ss, reform_params,
                    by_j=False, var=v,
                    plot_title='Lifecycle Profile of ' + title_list[i],
                    path=os.path.join(save_path, 'SSLifecycleProfile' +
                                      path_list[i] + '.png'))

    # lifetime profiles, c, n , b, y by j, separately for base and reform
    for i, v in enumerate(var_list):
        ss_profiles(base_ss, base_params,
                    by_j=True, var=v,
                    plot_title='Lifecycle Profile of ' + title_list[i],
                    path=os.path.join(save_path, 'SSLifecycleProfile' +
                                      path_list[i] + '_Baseline.png'))
        ss_profiles(reform_ss, reform_params,
                    by_j=True, var=v,
                    plot_title='Lifecycle Profile of ' + title_list[i],
                    path=os.path.join(save_path, 'SSLifecycleProfile' +
                                      path_list[i] + '_Reform.png'))
