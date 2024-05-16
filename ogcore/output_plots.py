import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from ogcore.constants import (
    VAR_LABELS,
    ToGDP_LABELS,
    DEFAULT_START_YEAR,
)
import ogcore.utils as utils
from ogcore.utils import Inequality


def plot_aggregates(
    base_tpi,
    base_params,
    reform_tpi=None,
    reform_params=None,
    var_list=["Y", "C", "K", "L"],
    plot_type="pct_diff",
    stationarized=True,
    num_years_to_plot=50,
    start_year=DEFAULT_START_YEAR,
    forecast_data=None,
    forecast_units=None,
    vertical_line_years=None,
    plot_title=None,
    path=None,
):
    """
    Create a plot of macro aggregates.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var_list (list): names of variable to plot
        plot_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baseline
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform
                (reform-base)
            'levels': plot variables in model units
            'forecast': plots variables in levels relative to baseline
                economic forecast
        stationarized (bool): whether used stationarized variables (False
            only affects pct_diff right now)
        num_years_to_plot (integer): number of years to include in plot
        start_year (integer): year to start plot
        forecast_data (array_like): baseline economic forecast series,
            must have length = num_year_to_plot
        forecast_units (str): units that baseline economic forecast is in
        vertical_line_years (list): list of integers for years want
            vertical lines at
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of macro aggregates

    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years_to_plot, int)
    assert num_years_to_plot <= base_params.T
    # Make sure both runs cover same time period
    if reform_tpi:
        assert base_params.start_year == reform_params.start_year
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    start_index = start_year - base_params.start_year
    # Check that reform included if doing pct_diff or diff plot
    if plot_type == "pct_diff" or plot_type == "diff":
        assert reform_tpi is not None
    fig1, ax1 = plt.subplots()
    for i, v in enumerate(var_list):
        if plot_type == "pct_diff":
            if v in ["r_gov", "r", "r_p"]:
                # Compute just percentage point changes for rates
                plot_var = reform_tpi[v] - base_tpi[v]
            else:
                if stationarized:
                    plot_var = (reform_tpi[v] - base_tpi[v]) / base_tpi[v]
                else:
                    pct_changes = utils.pct_change_unstationarized(
                        base_tpi,
                        base_params,
                        reform_tpi,
                        reform_params,
                        output_vars=[v],
                    )
                    plot_var = pct_changes[v]
            ylabel = r"Pct. change"
            plt.plot(
                year_vec,
                plot_var[start_index : start_index + num_years_to_plot],
                label=VAR_LABELS[v],
            )
        elif plot_type == "diff":
            plot_var = reform_tpi[v] - base_tpi[v]
            ylabel = r"Difference (Model Units)"
            plt.plot(
                year_vec,
                plot_var[start_index : start_index + num_years_to_plot],
                label=VAR_LABELS[v],
            )
        elif plot_type == "levels":
            plt.plot(
                year_vec,
                base_tpi[v][start_index : start_index + num_years_to_plot],
                label="Baseline " + VAR_LABELS[v],
            )
            if reform_tpi:
                plt.plot(
                    year_vec,
                    reform_tpi[v][
                        start_index : start_index + num_years_to_plot
                    ],
                    label="Reform " + VAR_LABELS[v],
                )
            ylabel = r"Model Units"
        elif plot_type == "forecast":
            # Need reform and baseline to ensure plot makes sense
            assert reform_tpi is not None
            # Plot forecast of baseline
            plot_var_base = forecast_data
            plt.plot(
                year_vec, plot_var_base, label="Baseline " + VAR_LABELS[v]
            )
            # Plot change from baseline forecast
            pct_change = ((reform_tpi[v] - base_tpi[v]) / base_tpi[v])[
                start_index : start_index + num_years_to_plot
            ]
            plot_var_reform = (1 + pct_change) * forecast_data
            plt.plot(
                year_vec, plot_var_reform, label="Reform " + VAR_LABELS[v]
            )
            # making units labels will not work if multiple variables
            # and they are in different units
            ylabel = forecast_units
        else:
            print("Please enter a valid plot type")
            assert False
    # vertical markers at certain years
    if vertical_line_years:
        for yr in vertical_line_years:
            plt.axvline(x=yr, linewidth=0.5, linestyle="--", color="k")
    plt.xlabel(r"Year $t$")
    plt.ylabel(ylabel)
    if plot_title:
        plt.title(plot_title, fontsize=15)
    ax1.set_yticks(ax1.get_yticks().tolist())
    vals = ax1.get_yticks()
    if plot_type == "pct_diff":
        ax1.set_yticklabels(["{:,.2%}".format(x) for x in vals])
    plt.xlim(
        (
            base_params.start_year - 1,
            base_params.start_year + num_years_to_plot,
        )
    )
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig1
    plt.close()


def plot_industry_aggregates(
    base_tpi,
    base_params,
    reform_tpi=None,
    reform_params=None,
    var_list=["Y_vec"],
    ind_names_list=None,
    plot_type="pct_diff",
    num_years_to_plot=50,
    start_year=DEFAULT_START_YEAR,
    forecast_data=None,
    forecast_units=None,
    vertical_line_years=None,
    plot_title=None,
    path=None,
):
    """
    Create a plot of macro aggregates by industry.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var_list (list): names of variable to plot
        plot_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baseline
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform
                (reform-base)
            'levels': plot variables in model units
            'forecast': plots variables in levels relative to baseline
                economic forecast
        num_years_to_plot (integer): number of years to include in plot
        start_year (integer): year to start plot
        forecast_data (array_like): baseline economic forecast series,
            must have length = num_year_to_plot
        forecast_units (str): units that baseline economic forecast is in
        vertical_line_years (list): list of integers for years want
            vertical lines at
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of macro aggregates

    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years_to_plot, int)
    assert num_years_to_plot <= base_params.T
    dims = base_tpi[var_list[0]].shape[1]
    if ind_names_list:
        assert len(ind_names_list) == dims
    else:
        ind_names_list = [str(i) for i in range(dims)]
    # Make sure both runs cover same time period
    if reform_tpi:
        assert base_params.start_year == reform_params.start_year
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    start_index = start_year - base_params.start_year
    # Check that reform included if doing pct_diff or diff plot
    if plot_type == "pct_diff" or plot_type == "diff":
        assert reform_tpi is not None
    fig1, ax1 = plt.subplots()
    for i, v in enumerate(var_list):
        if len(var_list) == 1:
            var_label = ""
        else:
            var_label = VAR_LABELS[v]
        for m in range(dims):
            if plot_type == "pct_diff":
                plot_var = (
                    reform_tpi[v][:, m] - base_tpi[v][:, m]
                ) / base_tpi[v][:, m]
                ylabel = r"Pct. change"
                plt.plot(
                    year_vec,
                    plot_var[start_index : start_index + num_years_to_plot],
                    label=var_label + " " + ind_names_list[m],
                )
            elif plot_type == "diff":
                plot_var = reform_tpi[v][:, m] - base_tpi[v][:, m]
                ylabel = r"Difference (Model Units)"
                plt.plot(
                    year_vec,
                    plot_var[start_index : start_index + num_years_to_plot],
                    label=var_label + " " + ind_names_list[m],
                )
            elif plot_type == "levels":
                plt.plot(
                    year_vec,
                    base_tpi[v][
                        start_index : start_index + num_years_to_plot, m
                    ],
                    label="Baseline " + var_label + " " + ind_names_list[m],
                )
                if reform_tpi:
                    plt.plot(
                        year_vec,
                        reform_tpi[v][
                            start_index : start_index + num_years_to_plot, m
                        ],
                        label="Reform " + var_label + " " + ind_names_list[m],
                    )
                ylabel = r"Model Units"
            elif plot_type == "forecast":
                # Need reform and baseline to ensure plot makes sense
                assert reform_tpi is not None
                # Plot forecast of baseline
                plot_var_base = forecast_data
                plt.plot(
                    year_vec,
                    plot_var_base,
                    label="Baseline " + var_label + " " + ind_names_list[m],
                )
                # Plot change from baseline forecast
                pct_change = (
                    reform_tpi[v][
                        start_index : start_index + num_years_to_plot, m
                    ]
                    - base_tpi[v][
                        start_index : start_index + num_years_to_plot, m
                    ]
                ) / base_tpi[v][
                    start_index : start_index + num_years_to_plot, m
                ]
                plot_var_reform = (1 + pct_change) * forecast_data
                plt.plot(
                    year_vec,
                    plot_var_reform,
                    label="Reform " + var_label + " " + ind_names_list[m],
                )
                # making units labels will not work if multiple variables
                # and they are in different units
                ylabel = forecast_units
            else:
                print("Please enter a valid plot type")
                assert False
    # vertical markers at certain years
    if vertical_line_years:
        for yr in vertical_line_years:
            plt.axvline(x=yr, linewidth=0.5, linestyle="--", color="k")
    plt.xlabel(r"Year $t$")
    plt.ylabel(ylabel)
    if plot_title:
        plt.title(plot_title, fontsize=15)
    ax1.set_yticks(ax1.get_yticks().tolist())
    vals = ax1.get_yticks()
    if plot_type == "pct_diff":
        ax1.set_yticklabels(["{:,.2%}".format(x) for x in vals])
    plt.xlim(
        (
            base_params.start_year - 1,
            base_params.start_year + num_years_to_plot,
        )
    )
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig1
    plt.close()


def ss_3Dplot(
    base_params,
    base_ss,
    reform_params=None,
    reform_ss=None,
    var="bssmat_splus1",
    plot_type="levels",
    plot_title=None,
    path=None,
):
    """
    Create a 3d plot of household decisions.

    Args:
        base_params (OG-Core Specifications class): baseline parameters object
        base_ss (dictionary): SS output from baseline run
        reform_params (OG-Core Specifications class): reform parameters object
        reform_ss (dictionary): SS output from reform run
        var (string): name of variable to plot
        plot_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baseline
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform (reform-base)
            'levels': plot variables in model units
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of household decisions

    """
    if reform_params:
        assert base_params.J == reform_params.J
        assert base_params.starting_age == reform_params.starting_age
        assert base_params.ending_age == reform_params.ending_age
        assert base_params.S == reform_params.S
    domain = np.linspace(
        base_params.starting_age, base_params.ending_age, base_params.S
    )
    Jgrid = np.zeros(base_params.J)
    for j in range(base_params.J):
        Jgrid[j:] += base_params.lambdas[j]
    if plot_type == "levels":
        data = base_ss[var].T
    elif plot_type == "diff":
        data = (reform_ss[var] - base_ss[var]).T
    elif plot_type == "pct_diff":
        data = ((reform_ss[var] - base_ss[var]) / base_ss[var]).T
    cmap1 = matplotlib.cm.get_cmap("jet")
    X, Y = np.meshgrid(domain, Jgrid)
    fig5, ax5 = plt.subplots(subplot_kw={"projection": "3d"})
    ax5.set_xlabel(r"age-$s$")
    ax5.set_ylabel(r"ability type-$j$")
    ax5.set_zlabel(r"individual savings $\bar{b}_{j,s}$")
    ax5.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=cmap1)
    if plot_title:
        plt.title(plot_title)
    if path:
        plt.savefig(path, dpi=300)
    else:
        return plt


def plot_gdp_ratio(
    base_tpi,
    base_params,
    reform_tpi=None,
    reform_params=None,
    var_list=["D"],
    plot_type="levels",
    num_years_to_plot=50,
    start_year=DEFAULT_START_YEAR,
    vertical_line_years=None,
    plot_title=None,
    path=None,
):
    """
    Create a plot of some variable to GDP.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-Core Specifications class): baseline parameters object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-Core Specifications class): reform parameters object
        p (OG-Core Specifications class): parameters object
        var_list (list): names of variable to plot
        plot_type (string): type of plot, can be:
            'diff': plots difference between baseline and reform
                (reform-base)
            'levels': plot variables in model units
        num_years_to_plot (integer): number of years to include in plot
        start_year (integer): year to start plot
        vertical_line_years (list): list of integers for years want
            vertical lines at
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of ratio of a variable to GDP
    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years_to_plot, int)
    assert num_years_to_plot <= base_params.T
    if plot_type == "diff":
        assert reform_tpi is not None
    # Make sure both runs cover same time period
    if reform_tpi:
        assert base_params.start_year == reform_params.start_year
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    start_index = start_year - base_params.start_year
    fig1, ax1 = plt.subplots()
    for i, v in enumerate(var_list):
        if plot_type == "levels":
            plot_var_base = (
                base_tpi[v][: base_params.T] / base_tpi["Y"][: base_params.T]
            )
            if reform_tpi:
                plot_var_reform = (
                    reform_tpi[v][: base_params.T]
                    / reform_tpi["Y"][: base_params.T]
                )
                plt.plot(
                    year_vec,
                    plot_var_base[
                        start_index : start_index + num_years_to_plot
                    ],
                    label="Baseline " + ToGDP_LABELS[v],
                )
                plt.plot(
                    year_vec,
                    plot_var_reform[
                        start_index : start_index + num_years_to_plot
                    ],
                    label="Reform " + ToGDP_LABELS[v],
                )
            else:
                plt.plot(
                    year_vec,
                    plot_var_base[
                        start_index : start_index + num_years_to_plot
                    ],
                    label=ToGDP_LABELS[v],
                )
        else:  # if plotting differences in ratios
            var_base = (
                base_tpi[v][: base_params.T] / base_tpi["Y"][: base_params.T]
            )
            var_reform = (
                reform_tpi[v][: base_params.T]
                / reform_tpi["Y"][: base_params.T]
            )
            plot_var = var_reform - var_base
            plt.plot(
                year_vec,
                plot_var[start_index : start_index + num_years_to_plot],
                label=ToGDP_LABELS[v],
            )
    ylabel = r"Percent of GDP"
    # vertical markers at certain years
    if vertical_line_years:
        for yr in vertical_line_years:
            plt.axvline(x=yr, linewidth=0.5, linestyle="--", color="k")
    plt.xlabel(r"Year $t$")
    plt.ylabel(ylabel)
    if plot_title:
        plt.title(plot_title, fontsize=15)
    ax1.set_yticks(ax1.get_yticks().tolist())
    vals = ax1.get_yticks()
    if plot_type == "levels":
        ax1.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    else:
        ax1.set_yticklabels(["{:,.2%}".format(x) for x in vals])
    plt.xlim(
        (
            base_params.start_year - 1,
            base_params.start_year + num_years_to_plot,
        )
    )
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig1
    plt.close()


def ability_bar(
    base_tpi,
    base_params,
    reform_tpi,
    reform_params,
    var="n_mat",
    num_years=5,
    start_year=DEFAULT_START_YEAR,
    plot_title=None,
    path=None,
):
    """
    Plots percentage changes from baseline by ability group for a
    given variable.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var (string): name of variable to plot
        num_year (integer): number of years to compute changes over
        start_year (integer): year to start plot
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of results by ability type
    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years, (int, np.integer))
    # Make sure both runs cover same time period
    if reform_tpi:
        assert base_params.start_year == reform_params.start_year
    N = base_params.J
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars
    start_index = start_year - base_params.start_year
    omega_to_use = base_params.omega[: base_params.T, :].reshape(
        base_params.T, base_params.S, 1
    )
    base_val = (
        (base_tpi[var] * omega_to_use)[
            start_index : start_index + num_years, :, :
        ]
        .sum(1)
        .sum(0)
    )
    reform_val = (
        (reform_tpi[var] * omega_to_use)[
            start_index : start_index + num_years, :, :
        ]
        .sum(1)
        .sum(0)
    )
    var_to_plot = (reform_val - base_val) / base_val
    ax.bar(ind, var_to_plot * 100, width, bottom=0)
    ax.set_xticks(ind + width / 4)
    ax.set_xticklabels(list(lambda_labels(base_params.lambdas).values()))
    plt.xticks(rotation=45)
    plt.ylabel(r"Percentage Change in " + VAR_LABELS[var])
    if plot_title:
        plt.title(plot_title, fontsize=15)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig
    plt.close()


def ability_bar_ss(
    base_ss,
    base_params,
    reform_ss,
    reform_params,
    var="nssmat",
    plot_title=None,
    path=None,
):
    """
    Plots percentage changes from baseline by ability group for a
    given variable.

    Args:
        base_ss (dictionary): SS output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_ss (dictionary): SS output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var (string): name of variable to plot
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of results by ability type
    """
    N = base_params.J
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars
    base_val = (
        base_ss[var] * base_params.omega_SS.reshape(base_params.S, 1)
    ).sum(0)
    reform_val = (
        reform_ss[var] * reform_params.omega_SS.reshape(reform_params.S, 1)
    ).sum(0)
    var_to_plot = (reform_val - base_val) / base_val
    ax.bar(ind, var_to_plot * 100, width, bottom=0)
    ax.set_xticks(ind + width / 4)
    ax.set_xticklabels(list(lambda_labels(base_params.lambdas).values()))
    plt.xticks(rotation=45)
    plt.ylabel(r"Percentage Change in " + VAR_LABELS[var])
    if plot_title:
        plt.title(plot_title, fontsize=15)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig
    plt.close()


def tpi_profiles(
    base_tpi,
    base_params,
    reform_tpi=None,
    reform_params=None,
    by_j=True,
    var="n_mat",
    num_years=5,
    start_year=DEFAULT_START_YEAR,
    plot_title=None,
    path=None,
):
    """
    Plot lifecycle profiles of given variable in the SS.

    Args:
        base_ss (dictionary): TPI output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_ss (dictionary): TPI output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var (string): name of variable to plot
        num_year (integer): number of years to compute changes over
        start_year (integer): year to start plot
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of lifecycle profiles

    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years, (int, np.integer))
    if reform_tpi:
        assert base_params.start_year == reform_params.start_year
        assert base_params.S == reform_params.S
        assert base_params.starting_age == reform_params.starting_age
        assert base_params.ending_age == reform_params.ending_age
    age_vec = np.arange(
        base_params.starting_age, base_params.starting_age + base_params.S
    )
    fig1, ax1 = plt.subplots()
    start_idx = start_year - base_params.start_year
    end_idx = start_idx + num_years
    if by_j:
        cm = plt.get_cmap("coolwarm")
        ax1.set_prop_cycle(color=[cm(1.0 * i / 7) for i in range(7)])
        for j in range(base_params.J):
            plt.plot(
                age_vec,
                base_tpi[var][start_idx:end_idx, :, j].sum(axis=0) / num_years,
                label="Baseline, j = " + str(j),
            )
            if reform_tpi:
                plt.plot(
                    age_vec,
                    reform_tpi[var][start_idx:end_idx, :, j].sum(axis=0)
                    / num_years,
                    label="Reform, j = " + str(j),
                    linestyle="--",
                )
    else:
        base_var = (
            base_tpi[var][start_idx:end_idx, :, :]
            * base_params.lambdas.reshape(1, 1, base_params.J)
        ).sum(axis=2).sum(axis=0) / num_years
        plt.plot(age_vec, base_var, label="Baseline")
        if reform_tpi:
            reform_var = (
                reform_tpi[var][start_idx:end_idx, :, :]
                * reform_params.lambdas.reshape(1, 1, base_params.J)
            ).sum(axis=2).sum(axis=0) / num_years
            plt.plot(age_vec, reform_var, label="Reform", linestyle="--")
    plt.xlabel(r"Age")
    plt.ylabel(VAR_LABELS[var])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if plot_title:
        plt.title(plot_title, fontsize=15)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig1
    plt.close()


def ss_profiles(
    base_ss,
    base_params,
    reform_ss=None,
    reform_params=None,
    by_j=True,
    var="nssmat",
    plot_data=None,
    plot_title=None,
    path=None,
):
    """
    Plot lifecycle profiles of given variable in the SS.

    Args:
        base_ss (dictionary): SS output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_ss (dictionary): SS output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var (string): name of variable to plot
        plot_data (array_like): series of data to add to plot
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of lifecycle profiles

    """
    if reform_ss:
        assert base_params.S == reform_params.S
        assert base_params.starting_age == reform_params.starting_age
        assert base_params.ending_age == reform_params.ending_age
    age_vec = np.arange(
        base_params.starting_age, base_params.starting_age + base_params.S
    )
    fig1, ax1 = plt.subplots()
    if by_j:
        cm = plt.get_cmap("coolwarm")
        ax1.set_prop_cycle(color=[cm(1.0 * i / 7) for i in range(7)])
        for j in range(base_params.J):
            plt.plot(
                age_vec, base_ss[var][:, j], label="Baseline, j = " + str(j)
            )
            if reform_ss:
                plt.plot(
                    age_vec,
                    reform_ss[var][:, j],
                    label="Reform, j = " + str(j),
                    linestyle="--",
                )
    else:
        base_var = (
            base_ss[var][:, :] * base_params.lambdas.reshape(1, base_params.J)
        ).sum(axis=1)
        plt.plot(age_vec, base_var, label="Baseline")
        if reform_ss:
            reform_var = (
                reform_ss[var][:, :]
                * reform_params.lambdas.reshape(1, reform_params.J)
            ).sum(axis=1)
            plt.plot(age_vec, reform_var, label="Reform", linestyle="--")
        if plot_data is not None:
            plt.plot(
                age_vec, plot_data, linewidth=2.0, label="Data", linestyle=":"
            )
    plt.xlabel(r"Age")
    plt.ylabel(VAR_LABELS[var])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if plot_title:
        plt.title(plot_title, fontsize=15)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig1
    plt.close()


def plot_all(base_output_path, reform_output_path, save_path):
    """
    Function to plot all default output plots.

    Args:
        base_output_path (str): path to baseline results
        reform_output_path (str): path to reform results
        save_path (str): path to save plots to

    Returns:
        None: All output figures saved to disk.

    """
    # Make directory in case it doesn't exist
    utils.mkdirs(save_path)
    # Read in data
    # Read in TPI output and parameters
    base_tpi = utils.safe_read_pickle(
        os.path.join(base_output_path, "TPI", "TPI_vars.pkl")
    )
    base_ss = utils.safe_read_pickle(
        os.path.join(base_output_path, "SS", "SS_vars.pkl")
    )

    base_params = utils.safe_read_pickle(
        os.path.join(base_output_path, "model_params.pkl")
    )

    reform_tpi = utils.safe_read_pickle(
        os.path.join(reform_output_path, "TPI", "TPI_vars.pkl")
    )

    reform_ss = utils.safe_read_pickle(
        os.path.join(reform_output_path, "SS", "SS_vars.pkl")
    )

    reform_params = utils.safe_read_pickle(
        os.path.join(reform_output_path, "model_params.pkl")
    )

    # Percentage changes in macro vars (Y, K, L, C)
    plot_aggregates(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "K", "L", "C"],
        plot_type="pct_diff",
        num_years_to_plot=min(base_params.T, 150),
        start_year=base_params.start_year,
        vertical_line_years=[
            base_params.start_year + base_params.tG1,
            base_params.start_year + base_params.tG2,
        ],
        plot_title="Percentage Changes in Macro Aggregates",
        path=os.path.join(save_path, "MacroAgg_PctChange.png"),
    )

    # Percentage change in fiscal vars (D, G, TR, Rev)
    plot_aggregates(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["D", "TR", "total_tax_revenue"],
        plot_type="pct_diff",
        num_years_to_plot=min(base_params.T, 150),
        start_year=base_params.start_year,
        vertical_line_years=[
            base_params.start_year + base_params.tG1,
            base_params.start_year + base_params.tG2,
        ],
        plot_title="Percentage Changes in Fiscal Variables",
        path=os.path.join(save_path, "Fiscal_PctChange.png"),
    )

    # r and w in baseline and reform -- vertical lines at tG1, tG2
    plot_aggregates(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["r"],
        plot_type="levels",
        num_years_to_plot=min(base_params.T, 150),
        start_year=base_params.start_year,
        vertical_line_years=[
            base_params.start_year + base_params.tG1,
            base_params.start_year + base_params.tG2,
        ],
        plot_title="Real Interest Rates Under Baseline and Reform",
        path=os.path.join(save_path, "InterestRates.png"),
    )

    plot_aggregates(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["w"],
        plot_type="levels",
        num_years_to_plot=min(base_params.T, 150),
        start_year=base_params.start_year,
        vertical_line_years=[
            base_params.start_year + base_params.tG1,
            base_params.start_year + base_params.tG2,
        ],
        plot_title="Wage Rates Under Baseline and Reform",
        path=os.path.join(save_path, "WageRates.png"),
    )

    # Gov't spending toGDP in base and reform-- vertical lines at tG1, tG2
    plot_gdp_ratio(
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        var_list=["G"],
        num_years_to_plot=min(base_params.T, 150),
        start_year=base_params.start_year,
        vertical_line_years=[
            base_params.start_year + base_params.tG1,
            base_params.start_year + base_params.tG2,
        ],
        plot_title="Gov't Spending-to-GDP",
        path=os.path.join(save_path, "SpendGDPratio.png"),
    )

    # Debt-GDP in base and reform-- vertical lines at tG1, tG2
    plot_gdp_ratio(
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        var_list=["D"],
        num_years_to_plot=min(base_params.T, 150),
        start_year=base_params.start_year,
        vertical_line_years=[
            base_params.start_year + base_params.tG1,
            base_params.start_year + base_params.tG2,
        ],
        plot_title="Debt-to-GDP",
        path=os.path.join(save_path, "DebtGDPratio.png"),
    )

    # Tax revenue to GDP in base and reform-- vertical lines at tG1, tG2
    plot_gdp_ratio(
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        var_list=["total_tax_revenue"],
        num_years_to_plot=min(base_params.T, 150),
        start_year=base_params.start_year,
        vertical_line_years=[
            base_params.start_year + base_params.tG1,
            base_params.start_year + base_params.tG2,
        ],
        plot_title="Tax Revenue to GDP",
        path=os.path.join(save_path, "RevenueGDPratio.png"),
    )

    # Pct change in c, n, b, y, etr, mtrx, mtry by ability group over 10 years
    var_list = [
        "c_path",
        "n_mat",
        "bmat_splus1",
        "etr_path",
        "mtrx_path",
        "mtry_path",
        "y_before_tax_mat",
    ]
    title_list = [
        "consumption",
        "labor supply",
        "savings",
        "effective tax rates",
        "marginal tax rates on labor income",
        "marginal tax rates on capital income",
        "before tax income",
    ]
    path_list = ["Cons", "Labor", "Save", "ETR", "MTRx", "MTRy", "Income"]
    for i, v in enumerate(var_list):
        ability_bar(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var=v,
            num_years=10,
            start_year=base_params.start_year,
            plot_title="Percentage changes in " + title_list[i],
            path=os.path.join(save_path, "PctChange_" + path_list[i] + ".png"),
        )

    # lifetime profiles, base vs reform, SS for c, n, b, y - not by j
    var_list = [
        "cssmat",
        "nssmat",
        "bssmat_splus1",
        "etr_ss",
        "mtrx_ss",
        "mtry_ss",
    ]
    for i, v in enumerate(var_list):
        ss_profiles(
            base_ss,
            base_params,
            reform_ss,
            reform_params,
            by_j=False,
            var=v,
            plot_title="Lifecycle Profile of " + title_list[i],
            path=os.path.join(
                save_path, "SSLifecycleProfile_" + path_list[i] + ".png"
            ),
        )

    # lifetime profiles, c, n , b, y by j, separately for base and reform
    for i, v in enumerate(var_list):
        ss_profiles(
            base_ss,
            base_params,
            by_j=True,
            var=v,
            plot_title="Lifecycle Profile of " + title_list[i],
            path=os.path.join(
                save_path,
                "SSLifecycleProfile_" + path_list[i] + "_Baseline.png",
            ),
        )
        ss_profiles(
            reform_ss,
            reform_params,
            by_j=True,
            var=v,
            plot_title="Lifecycle Profile of " + title_list[i],
            path=os.path.join(
                save_path, "SSLifecycleProfile_" + path_list[i] + "_Reform.png"
            ),
        )


def inequality_plot(
    base_tpi,
    base_params,
    reform_tpi=None,
    reform_params=None,
    var="c_path",
    ineq_measure="gini",
    pctiles=None,
    plot_type="levels",
    num_years_to_plot=50,
    start_year=DEFAULT_START_YEAR,
    vertical_line_years=None,
    plot_title=None,
    path=None,
):
    """
    Plot measures of inequality over the time path.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var(string): name of variable to plot
        ineq_measure (string): inequality measure to plot, can be:
            'gini': Gini coefficient
            'var_of_logs': variance of logs
            'pct_ratio': percentile ratio
            'top_share': top share of total
        pctiles (tuple or None): percentiles for percentile ratios
            (numerator, denominator) or percentile for top share (not
            required for Gini or var_of_logs)
        plot_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baselien
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform
                (reform-base)
            'levels': plot variables in model units
        num_years_to_plot (integer): number of years to include in plot
        start_year (integer): year to start plot
        vertical_line_years (list): list of integers for years want
            vertical lines at
        plot_title (string): title for plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of inequality measure

    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years_to_plot, int)
    assert num_years_to_plot <= base_params.T
    # Make sure both runs cover same time period
    if reform_tpi:
        assert base_params.start_year == reform_params.start_year
    assert ineq_measure in ["gini", "var_of_logs", "pct_ratio", "top_share"]
    if (ineq_measure == "pct_ratio") | (ineq_measure == "top_share"):
        assert pctiles
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    # Check that reform included if doing pct_diff or diff plot
    if plot_type == "pct_diff" or plot_type == "diff":
        assert reform_tpi is not None
    fig1, ax1 = plt.subplots()
    base_values = np.zeros(num_years_to_plot)
    for t in range(num_years_to_plot):
        idx = (t + start_year) - base_params.start_year
        ineq = Inequality(
            base_tpi[var][idx, :, :],
            base_params.omega[idx, :],
            base_params.lambdas,
            base_params.S,
            base_params.J,
        )
        if ineq_measure == "gini":
            base_values[t] = ineq.gini()
            ylabel = r"Gini Coefficient"
        elif ineq_measure == "var_of_logs":
            base_values[t] = ineq.var_of_logs()
            ylabel = r"var(ln(" + VAR_LABELS[var] + r"))"
        elif ineq_measure == "pct_ratio":
            base_values[t] = ineq.ratio_pct1_pct2(pctiles[0], pctiles[1])
            ylabel = r"Ratio"
        elif ineq_measure == "top_share":
            base_values[t] = ineq.top_share(pctiles)
            ylabel = r"Share of Total " + VAR_LABELS[var]
    if reform_tpi:
        reform_values = np.zeros_like(base_values)
        for t in range(num_years_to_plot):
            idx = (t + start_year) - base_params.start_year
            ineq = Inequality(
                reform_tpi[var][idx, :, :],
                reform_params.omega[idx, :],
                reform_params.lambdas,
                reform_params.S,
                reform_params.J,
            )
            if ineq_measure == "gini":
                reform_values[t] = ineq.gini()
            elif ineq_measure == "var_of_logs":
                reform_values[t] = ineq.var_of_logs()
            elif ineq_measure == "pct_ratio":
                reform_values[t] = ineq.ratio_pct1_pct2(pctiles[0], pctiles[1])
            elif ineq_measure == "top_share":
                reform_values[t] = ineq.top_share(pctiles)
    if plot_type == "pct_diff":
        plot_var = (reform_values - base_values) / base_values
        ylabel = r"Pct. change"
        plt.plot(year_vec, plot_var)
    elif plot_type == "diff":
        plot_var = reform_values - base_values
        ylabel = r"Difference"
        plt.plot(year_vec, plot_var)
    elif plot_type == "levels":
        plt.plot(year_vec, base_values, label="Baseline")
        if reform_tpi:
            plt.plot(year_vec, reform_values, label="Reform")
    # vertical markers at certain years
    if vertical_line_years:
        for yr in vertical_line_years:
            plt.axvline(x=yr, linewidth=0.5, linestyle="--", color="k")
    plt.xlabel(r"Year $t$")
    plt.ylabel(ylabel)
    if plot_title:
        plt.title(plot_title, fontsize=15)
    vals = ax1.get_yticks()
    if plot_type == "pct_diff":
        ax1.set_yticklabels(["{:,.2%}".format(x) for x in vals])
    plt.xlim(
        (
            base_params.start_year - 1,
            base_params.start_year + num_years_to_plot,
        )
    )
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if path:
        fig_path1 = os.path.join(path)
        plt.savefig(fig_path1, bbox_inches="tight", dpi=300)
    else:
        return fig1
    plt.close()


def lambda_labels(lambdas):
    """
    Creates string labels of percentage groups for ability types given
    any number of ability types.

    Args:
        lambdas (np.array): array of lambdas for each ability type

    Returns:
        labels (dict): dict of string labels for ability types
    """
    lambdas_100 = [x * 100 for x in lambdas]
    lambdas_cumsum = list(np.cumsum(lambdas_100))
    lambdas_cumsum = [0.00] + lambdas_cumsum
    lambda_dict = {}
    rounded = []  # list of rounded values, strings
    for i in range(len(lambdas_cumsum)):
        # condition formatting to number of digits need, but not more than 2
        if lambdas_cumsum[i] % 1 == 0:
            round_i = f"{lambdas_cumsum[i]:.0f}"
        elif lambdas_cumsum[i] % 0.1 == 0:
            round_i = f"{lambdas_cumsum[i]:.1f}"
        else:
            round_i = f"{lambdas_cumsum[i]:.2f}"
        rounded.append(round_i)
    for i in range(1, len(lambdas_cumsum) - 1):
        lambda_dict[i - 1] = rounded[i - 1] + "-" + rounded[i] + "%"
        # and do rounding for the top
        top_number = 100 - lambdas_cumsum[-2]
        if top_number % 1 == 0:
            top_number_str = f"{top_number:.0f}"
        elif top_number % 0.1 == 0:
            top_number_str = f"{top_number:.1f}"
        else:
            top_number_str = f"{top_number:.2f}"
    lambda_dict[i] = "Top " + top_number_str + "%"

    return lambda_dict
