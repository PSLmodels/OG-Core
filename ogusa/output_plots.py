import numpy as np
import os
import matplotlib.pyplot as plt
from ogusa.constants import VAR_LABELS, ToGDP_LABELS
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
        p (OG-USA Specifications class): parameters object
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
        fig (Matplotlib plot object): plot of immigration rates
    '''
    assert (isinstance(start_year, int))
    # Make sure both runs cover same time period
    if reform_tpi is not None:
        assert (base_params.start_year == reform_params.start_year)
    year_vec = np.arange(base_params.start_year, base_params.start_year
                         + num_years_to_plot)
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
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    if path is not None:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight")
    else:
        return fig1
    plt.close()


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
        fig (Matplotlib plot object): plot of immigration rates
    '''
    assert (isinstance(start_year, int))
    # Make sure both runs cover same time period
    if reform_tpi is not None:
        assert (base_params.start_year == reform_params.start_year)
    year_vec = np.arange(base_params.start_year, base_params.start_year
                         + num_years_to_plot)
    start_index = start_year - base_params.start_year
    fig1, ax1 = plt.subplots()
    for i, v in enumerate(var_list):
        plot_var_base = base_tpi[v] / base_tpi['Y']
        if reform_tpi is not None:
            plot_var_reform = reform_tpi[v] / reform_tpi['Y']
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
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    if path is not None:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight")
    else:
        return fig1
    plt.close()


def ability_bar(base_tpi, base_params, reform_tpi,
                    reform_parmas, var='nmat',
                    num_years=5,
                    start_year=2019, vertical_line_years=None,
                    plot_title=None, path=None):
    '''
    Plots percentage changes from baseline by ability group for a
    given variable
    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-USA Specifications class): baseline parameters object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-USA Specifications class): reform parameters object
        p (OG-USA Specifications class): parameters object
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
        fig (Matplotlib plot object): plot of immigration rates
    '''
    N = base_params.J
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars
    start_index = start_year - base_params.start_year
    var_to_plot = (
        ((reform_tpi[var] * base_params.omega)[
            start_index:start_index + num_years, :, :].sum(1).sum(0)
         - (base_tpi[var] * base_params.omega)[
             start_index:start_index + num_years, :, :].sum(1).sum(0))
        / (reform_tpi[var] * base_params.omega)[
            start_index:start_index + num_years, :, :].sum(1).sum(0))
    ax.bar(ind, var_to_plot * 100, width,
                bottom=0)
    ax.set_xticks(ind + width / 4)
    ax.set_xticklabels(('0-25%', '25-50%', '50-70%', '70-80%', '80-90%',
                        '90-99%', 'Top 1%'))
    plt.ylabel(r'Percentage Change')
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    if path is not None:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight")
    else:
        return fig
    plt.close()
