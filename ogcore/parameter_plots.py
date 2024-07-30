import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from ogcore.constants import GROUP_LABELS
from ogcore import utils, txfunc
from ogcore.constants import DEFAULT_START_YEAR, VAR_LABELS


def plot_imm_rates(
    imm_rates,
    start_year=DEFAULT_START_YEAR,
    years_to_plot=[DEFAULT_START_YEAR],
    include_title=False,
    source="United Nations, World Population Prospects",
    path=None,
):
    """
    Plot fertility rates from the data

    Args:
        imm_rates (NumPy array): immigration rates for each of
            totpers
        start_year (int): first year of data
        years_to_plot (list): list of years to plot
        source (str): data source for fertility rates
        path (str): path to save figure to, if None then figure
            is returned

    Returns:
        fig (Matplotlib plot object): plot of fertility rates

    """
    # create line styles to cycle through
    fig, ax = plt.subplots()
    for y in years_to_plot:
        i = start_year - y
        plt.plot(imm_rates[i, :], c="blue", label="Year " + str(y))
    # plt.title('Fertility rates by age ($f_{s}$)',
    #     fontsize=20)
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Immigration rate $i_{s}$")
    plt.legend(loc="upper left")
    plt.text(
        -5,
        -0.05,
        "Source: " + source,
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    if include_title:
        plt.title("Immigration Rates")
    # Save or return figure
    if path:
        output_path = os.path.join(path, "imm_rates")
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        fig.show()
        return fig


def plot_mort_rates(
    p_list,
    labels=[""],
    years=[DEFAULT_START_YEAR],
    survival_rates=False,
    include_title=False,
    path=None,
):
    """
    Create a plot of mortality rates from OG-Core parameterization.

    Args:
        p_list (list): list of parameters objects
        labels (list): list of labels for the legend
        survival_rates (bool): whether to plot survival rates instead
            of mortality rates
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of mortality rates

    """
    p0 = p_list[0]
    age_per = np.linspace(p0.E, p0.E + p0.S, p0.S)
    fig, ax = plt.subplots()
    for y in years:
        t = y - p0.start_year
        for i, p in enumerate(p_list):
            if survival_rates:
                plt.plot(
                    age_per,
                    np.cumprod(1 - p.rho[t, :]),
                    label=labels[i] + " " + str(y),
                )
            else:
                plt.plot(age_per, p.rho[t, :], label=labels[i] + " " + str(y))
    plt.xlabel(r"Age $s$ (model periods)")
    if survival_rates:
        plt.ylabel(r"Cumulative Survival Rates")
        plt.legend(loc="lower left")
        title = "Survival Rates"
    else:
        plt.ylabel(r"Mortality Rates $\rho_{s}$")
        plt.legend(loc="upper left")
        title = "Mortality Rates"
    vals = ax.get_yticks()
    ax.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    if include_title:
        plt.title(title)
    if path is None:
        return fig
    else:
        if survival_rates:
            fig_path = os.path.join(path, "survival_rates")
        else:
            fig_path = os.path.join(path, "mortality_rates")
        plt.savefig(fig_path, dpi=300)


def plot_pop_growth(
    p,
    start_year=DEFAULT_START_YEAR,
    num_years_to_plot=150,
    include_title=False,
    path=None,
):
    """
    Create a plot of population growth rates by year.

    Args:
        p (OG-Core Specifications class): parameters object
        start_year (integer): year to begin plotting
        num_years_to_plot (integer): number of years to plot
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of immigration rates

    """
    assert isinstance(start_year, int)
    assert isinstance(num_years_to_plot, int)
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    start_index = start_year - p.start_year
    fig, ax = plt.subplots()
    plt.plot(year_vec, p.g_n[start_index : start_index + num_years_to_plot])
    plt.xlabel(r"Year $t$")
    plt.ylabel(r"Population Growth Rate $g_{n, t}$")
    vals = ax.get_yticks()
    ax.set_yticklabels(["{:,.2%}".format(x) for x in vals])
    if include_title:
        plt.title("Population Growth Rates")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "pop_growth_rates")
        plt.savefig(fig_path, dpi=300)


def plot_population(p, years_to_plot=["SS"], include_title=False, path=None):
    """
    Plot the distribution of the population over age for various years.

    Args:
        p (OG-Core Specifications class): parameters object
        years_to_plot (list): list of years to plot, 'SS' will denote
            the steady-state period
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of population distribution

    """
    for i, v in enumerate(years_to_plot):
        assert isinstance(v, int) | (v == "SS")
        if isinstance(v, int):
            assert v >= p.start_year
    age_vec = np.arange(p.E, p.S + p.E)
    fig, ax = plt.subplots()
    for i, v in enumerate(years_to_plot):
        if v == "SS":
            pop_dist = p.omega_SS
        else:
            pop_dist = p.omega[v - p.start_year, :]
        plt.plot(age_vec, pop_dist, label=str(v) + " pop.")
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Pop. dist'n $\omega_{s}$")
    plt.legend(loc="lower left")
    if include_title:
        plt.title("Population Distribution by Year")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "pop_distribution")
        plt.savefig(fig_path, dpi=300)


def plot_ability_profiles(
    p, p2=None, t=None, log_scale=False, include_title=False, path=None
):
    """
    Create a plot of earnings ability profiles.

    Args:
        p (OG-Core Specifications class): parameters object
        t (int): model period for year, if None, then plot ability matrix for SS
        log_scale (bool): whether to plot in log points
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of earnings ability profiles

    """
    if t is None:
        t = -1
    age_vec = np.arange(p.starting_age, p.starting_age + p.S)
    fig, ax = plt.subplots()
    cm = plt.get_cmap("coolwarm")
    ax.set_prop_cycle(color=[cm(1.0 * i / p.J) for i in range(p.J)])
    for j in range(p.J):
        if log_scale:
            plt.plot(age_vec, np.log(p.e[t, :, j]), label=GROUP_LABELS[p.J][j])
        else:
            plt.plot(age_vec, p.e[t, :, j], label=GROUP_LABELS[p.J][j])
    if p2 is not None:
        for j in range(p.J):
            if log_scale:
                plt.plot(
                    age_vec,
                    np.log(p2.e[t, :, j]),
                    linestyle="--",
                    label=GROUP_LABELS[p.J][j],
                )
            else:
                plt.plot(
                    age_vec,
                    p2.e[t, :, j],
                    linestyle="--",
                    label=GROUP_LABELS[p.J][j],
                )
    plt.xlabel(r"Age")
    if log_scale:
        plt.ylabel(r"ln(Earnings ability)")
    else:
        plt.ylabel(r"Earnings ability")
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncols=5)
    if include_title:
        plt.title("Lifecycle Profiles of Effective Labor Units")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "ability_profiles")
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)


def plot_elliptical_u(p, plot_MU=True, include_title=False, path=None):
    """
    Create a plot of showing the fit of the elliptical utility function.

    Args:
        p (OG-Core Specifications class): parameters object
        plot_MU (boolean): whether plot marginal utility or utility in
            levels
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of elliptical vs CFE utility

    """
    theta = 1 / p.frisch
    N = 101
    n_grid = np.linspace(0.01, 0.8, num=N)
    if plot_MU:
        CFE = (1.0 / p.ltilde) * ((n_grid / p.ltilde) ** theta)
        ellipse = (
            1.0
            * p.b_ellipse
            * (1.0 / p.ltilde)
            * (
                (1.0 - (n_grid / p.ltilde) ** p.upsilon)
                ** ((1.0 / p.upsilon) - 1.0)
            )
            * (n_grid / p.ltilde) ** (p.upsilon - 1.0)
        )
    else:
        CFE = ((n_grid / p.ltilde) ** (1 + theta)) / (1 + theta)
        k = 1.0  # we don't estimate k, so not in parameters
        ellipse = (
            p.b_ellipse
            * ((1 - ((n_grid / p.ltilde) ** p.upsilon)) ** (1 / p.upsilon))
            + k
        )
    fig, ax = plt.subplots()
    plt.plot(n_grid, CFE, label="CFE")
    plt.plot(n_grid, ellipse, label="Elliptical U")
    if include_title:
        if plot_MU:
            plt.title("Marginal Utility of CFE and Elliptical")
        else:
            plt.title("Constant Frisch Elasticity vs. Elliptical Utility")
    plt.xlabel(r"Labor Supply")
    if plot_MU:
        plt.ylabel(r"Marginal Utility")
    else:
        plt.ylabel(r"Utility")
    plt.legend(loc="upper left")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "ellipse_v_CFE")
        plt.savefig(fig_path, dpi=300)


def plot_chi_n(
    p_list,
    labels=[""],
    years_to_plot=[DEFAULT_START_YEAR],
    include_title=False,
    path=None,
):
    """
    Create a plot of showing the values of the chi_n parameters.

    Args:
        p_list (list): parameters objects
        labels (list): labels for legend
        years_to_plot (list): list of years to plot
        include_title (boolean): whether to include a title in the plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of chi_n parameters

    """
    p0 = p_list[0]
    age = np.linspace(p0.starting_age, p0.ending_age, p0.S)
    fig, ax = plt.subplots()
    for y in years_to_plot:
        for i, p in enumerate(p_list):
            plt.plot(
                age,
                p.chi_n[y - p.start_year, :],
                label=labels[i] + " " + str(y),
            )
    if include_title:
        plt.title("Utility Weight on the Disutility of Labor Supply")
    plt.xlabel("Age, $s$")
    plt.ylabel(r"$\chi^{n}_{s}$")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "chi_n_values")
        plt.savefig(fig_path, dpi=300)


def plot_fert_rates(
    fert_rates_list,
    labels=[""],
    start_year=DEFAULT_START_YEAR,
    years_to_plot=[DEFAULT_START_YEAR],
    include_title=False,
    source="United Nations, World Population Prospects",
    path=None,
):
    """
    Plot fertility rates from the data

    Args:
        fert_rates_list (list): list of Numpy arrays of fertility rates
            for each model period and age
        labels (list): list of labels for the legend
        start_year (int): first year of data
        years_to_plot (list): list of years to plot
        include_title (bool): whether to include a title in the plot
        source (str): data source for fertility rates
        path (str): path to save figure to, if None then figure
            is returned

    Returns:
        fig (Matplotlib plot object): plot of fertility rates

    """
    # create line styles to cycle through
    fig, ax = plt.subplots()
    for y in years_to_plot:
        i = start_year - y
        for i, fert_rates in enumerate(fert_rates_list):
            plt.plot(fert_rates[i, :], label=labels[i] + " " + str(y))
    if include_title:
        plt.title("Fertility rates by age ($f_{s}$)", fontsize=20)
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Fertility rate $f_{s}$")
    plt.legend(loc="upper right")
    plt.text(
        -5,
        -0.023,
        "Source: " + source,
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    # Save or return figure
    if path:
        output_path = os.path.join(path, "fert_rates")
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        fig.show()
        return fig


def plot_mort_rates_data(
    mort_rates,
    start_year=DEFAULT_START_YEAR,
    years_to_plot=[DEFAULT_START_YEAR],
    source="United Nations, World Population Prospects",
    path=None,
):
    """
    Plots mortality rates from the data.

    Args:
        mort_rates (array_like): mortality rates for each of
            totpers
        start_year (int): first year of data
        years_to_plot (list): list of years to plot
        source (str): data source for fertility rates
        path (str): path to save figure to, if None then figure
            is returned

    Returns:
        fig (Matplotlib plot object): plot of mortality rates

    """
    # create line styles to cycle through
    fig, ax = plt.subplots()
    for y in years_to_plot:
        i = start_year - y
        plt.plot(mort_rates[i, :], c="blue", label="Year " + str(y))
    # plt.title('Fertility rates by age ($f_{s}$)',
    #     fontsize=20)
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Mortality rate $rho_{s}$")
    plt.legend(loc="upper left")
    plt.text(
        -5,
        -0.223,
        "Source: " + source,
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    # Save or return figure
    if path:
        output_path = os.path.join(path, "mort_rates")
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        fig.show()
        return fig


def plot_g_n(p_list, label_list=[""], include_title=False, path=None):
    """
    Create a plot of population growth rates from OG-Core parameterization.

    Args:
        p_list (list): list of OG-Core Specifications objects
        label_list (list): list of labels for the legend
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of immigration rates

    """
    p0 = p_list[0]
    years = np.arange(p0.start_year, p0.start_year + p0.T)
    fig, ax = plt.subplots()
    for i, p in enumerate(p_list):
        plt.plot(years, p.g_n[: p.T], label=label_list[i])
    plt.xlabel(r"Year $s$ (model periods)")
    plt.ylabel(r"Population Growth Rate $g_{n,t}$")
    plt.legend(loc="upper right")
    vals = ax.get_yticks()
    ax.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    if include_title:
        plt.title("Population Growth Rates")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "pop_growth_rates")
        plt.savefig(fig_path, dpi=300)


def plot_omega_fixed(age_per_EpS, omega_SS_orig, omega_SSfx, E, S, path=None):
    """
    Plot the steady-state population distribution implied by the data
    on fertility and mortality rates versus the the steady-state
    population distribution after adjusting immigration rates so that
    the stationary distribution is achieved a reasonable number of
    model periods.

    Args:
        age_per_EpS (array_like): list of ages over which to plot
            population distribution
        omega_SS_orig (Numpy array): population distribution in SS
            without adjustment to immigration rates
        omega_SSfx (Numpy array): population distribution in SS
            after adjustment to immigration rates
        E (int): age at which household becomes economically active
        S (int): number of years which household is economically active
        path (str): path to save figure to, if None then figure
            is returned

    Returns:
        fig (Matplotlib plot object): plot of SS population distribution
            before and after adjustment to immigration rates

    """
    fig, ax = plt.subplots()
    plt.plot(age_per_EpS, omega_SS_orig, label="Original Dist'n")
    plt.plot(age_per_EpS, omega_SSfx, label="Fixed Dist'n")
    plt.title("Original steady-state population distribution vs. fixed")
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Pop. dist'n $\omega_{s}$")
    plt.xlim((0, E + S + 1))
    plt.legend(loc="upper right")
    # Save or return figure
    if path:
        output_path = os.path.join(path, "OrigVsFixSSpop")
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        return fig


def plot_imm_fixed(
    age_per_EpS, imm_rates_orig, imm_rates_adj, E, S, path=None
):
    """
    Plot the immigration rates implied by the data on population,
    mortality, and fertility versus the adjusted immigration rates
    needed to achieve a stationary distribution of the population in a
    reasonable number of model periods.

    Args:
        age_per_EpS (array_like): list of ages over which to plot
            population distribution
        imm_rates_orig (Numpy array): immigration rates by age
        imm_rates_adj (Numpy array): adjusted immigration rates by age
        E (int): age at which household becomes economically active
        S (int): number of years which household is economically active
        path (str): path to save figure to, if None then figure
            is returned

    Returns:
        fig (Matplotlib plot object): plot of immigration rates found
            from residuals and the adjusted rates to hit SS sooner

    """
    fig, ax = plt.subplots()
    plt.plot(age_per_EpS, imm_rates_orig, label="Original Imm. Rates")
    plt.plot(age_per_EpS, imm_rates_adj, label="Adj. Imm. Rates")
    plt.title("Original immigration rates vs. adjusted")
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Imm. rates $i_{s}$")
    plt.xlim((0, E + S + 1))
    plt.legend(loc="upper center")
    # Save or return figure
    if path:
        output_path = os.path.join(path, "OrigVsAdjImm")
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        return fig


def plot_population_path(
    age_per_EpS,
    omega_path_lev,
    omega_SSfx,
    start_year,
    year1,
    year2,
    S,
    path=None,
):
    """
    Plot the distribution of the population over age for various years.

    Args:
        age_per_EpS (array_like): list of ages over which to plot
            population distribution
        initial_pop_pct (array_like): initial year population distribution
        omega_path_lev (Numpy array): number of households by age
            over the transition path
        omega_SSfx (Numpy array): number of households by age
            in the SS
        start_year (int): first year of data (so can get index of year1
            and year2)
        year1 (int): first year of data to plot
        year2 (int): second year of data to plot
        S (int): number of years which household is economically active
        path (str): path to save figure to, if None then figure
            is returned

    Returns:
        fig (Matplotlib plot object): plot of population distribution
            at points along the time path

    """
    fig, ax = plt.subplots()
    plt.plot(
        age_per_EpS,
        (
            omega_path_lev[start_year - year1, :]
            / omega_path_lev[start_year - year1, :].sum()
        ),
        label=str(year1) + " pop.",
    )
    plt.plot(
        age_per_EpS,
        (
            omega_path_lev[start_year - year2, :]
            / omega_path_lev[start_year - year2, :].sum()
        ),
        label=str(year2) + " pop.",
    )
    plt.plot(
        age_per_EpS,
        (
            omega_path_lev[int(0.5 * S), :]
            / omega_path_lev[int(0.5 * S), :].sum()
        ),
        label="T=" + str(int(0.5 * S)) + " pop.",
    )
    plt.plot(
        age_per_EpS,
        (omega_path_lev[int(S), :] / omega_path_lev[int(S), :].sum()),
        label="T=" + str(int(S)) + " pop.",
    )
    plt.plot(age_per_EpS, omega_SSfx, label="Adj. SS pop.")
    plt.title("Population distribution at points in time path")
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Pop. dist'n $\omega_{s}$")
    plt.legend(loc="lower left")
    # Save or return figure
    if path:
        output_path = os.path.join(path, "PopDistPath")
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        return fig


def gen_3Dscatters_hist(df, s, t, output_dir):
    """
    Create 3-D scatterplots and corresponding 3D histogram of ETR, MTRx,
    and MTRy as functions of labor income and capital income with
    truncated data in the income dimension

    Args:
        df (Pandas DataFrame): 11 variables with N observations of tax
            rates
        s (int): age of individual, >= 21
        t (int): year of analysis, >= 2016
        path (str): output directory for saving plot files

    Returns:
        None

    """
    from ogcore.txfunc import MAX_INC_GRAPH, MIN_INC_GRAPH

    # Truncate the data
    df_trnc = df[
        (df["total_labinc"] > MIN_INC_GRAPH)
        & (df["total_labinc"] < MAX_INC_GRAPH)
        & (df["total_capinc"] > MIN_INC_GRAPH)
        & (df["total_capinc"] < MAX_INC_GRAPH)
    ]
    inc_lab = df_trnc["total_labinc"]
    inc_cap = df_trnc["total_capinc"]
    etr_data = df_trnc["etr"]
    mtrx_data = df_trnc["mtr_labinc"]
    mtry_data = df_trnc["mtr_capinc"]

    # Plot 3D scatterplot of ETR data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(inc_lab, inc_cap, etr_data, c="r", marker="o")
    ax.set_xlabel("Total Labor Income")
    ax.set_ylabel("Total Capital Income")
    ax.set_zlabel("ETR")
    plt.title(
        "ETR, Lab. Inc., and Cap. Inc., Age=" + str(s) + ", Year=" + str(t)
    )
    filename = "ETR_age_" + str(s) + "_Year_" + str(t) + "_data.png"
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches="tight", dpi=300)
    plt.close()

    # Plot 3D histogram for all data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bin_num = int(30)
    hist, xedges, yedges = np.histogram2d(inc_lab, inc_cap, bins=bin_num)
    hist = hist / hist.sum()
    x_midp = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
    y_midp = yedges[:-1] + 0.5 * (yedges[1] - yedges[0])
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    ypos, xpos = np.meshgrid(y_midp, x_midp)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements)
    dx = (xedges[1] - xedges[0]) * np.ones_like(bin_num)
    dy = (yedges[1] - yedges[0]) * np.ones_like(bin_num)
    dz = hist.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color="b", zsort="average")
    ax.set_xlabel("Total Labor Income")
    ax.set_ylabel("Total Capital Income")
    ax.set_zlabel("Percent of obs.")
    plt.title(
        "Histogram by lab. inc., and cap. inc., Age="
        + str(s)
        + ", Year="
        + str(t)
    )
    filename = "Hist_Age_" + str(s) + "_Year_" + str(t) + ".png"
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches="tight", dpi=300)
    plt.close()

    # Plot 3D scatterplot of MTRx data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(inc_lab, inc_cap, mtrx_data, c="r", marker="o")
    ax.set_xlabel("Total Labor Income")
    ax.set_ylabel("Total Capital Income")
    ax.set_zlabel("Marginal Tax Rate, Labor Inc.)")
    plt.title(
        "MTR Labor Income, Lab. Inc., and Cap. Inc., Age="
        + str(s)
        + ", Year="
        + str(t)
    )
    filename = "MTRx_Age_" + str(s) + "_Year_" + str(t) + "_data.png"
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches="tight", dpi=300)
    plt.close()

    # Plot 3D scatterplot of MTRy data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(inc_lab, inc_cap, mtry_data, c="r", marker="o")
    ax.set_xlabel("Total Labor Income")
    ax.set_ylabel("Total Capital Income")
    ax.set_zlabel("Marginal Tax Rate (Capital Inc.)")
    plt.title(
        "MTR Capital Income, Cap. Inc., and Cap. Inc., Age="
        + str(s)
        + ", Year="
        + str(t)
    )
    filename = "MTRy_Age_" + str(s) + "_Year_" + str(t) + "_data.png"
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches="tight", dpi=300)
    plt.close()

    # Garbage collection
    del df, df_trnc, inc_lab, inc_cap, etr_data, mtrx_data, mtry_data


def txfunc_graph(
    s,
    t,
    df,
    X,
    Y,
    txrates,
    rate_type,
    tax_func_type,
    params_to_plot,
    output_dir,
):
    """
    This function creates a 3D plot of the fitted tax function against
    the data.

    Args:
        s (int): age of individual, >= 21
        t (int): year of analysis, >= 2016
        df (Pandas DataFrame): 11 variables with N observations of tax
            rates
        X (Pandas DataSeries): labor income
        Y (Pandas DataSeries): capital income
        Y (Pandas DataSeries): tax rates from the data
        rate_type (str): type of tax rate: mtrx, mtry, etr
        tax_func_type (str): functional form of tax functions
        params_to_plot (array_like or function): tax function parameters or
            nonparametric function
        path (str): output directory for saving plot files

    Returns:
        None

    """
    cmap1 = matplotlib.cm.get_cmap("summer")

    # Make comparison plot with full income domains
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, txrates, c="r", marker="o")
    ax.set_xlabel("Total Labor Income")
    ax.set_ylabel("Total Capital Income")
    if rate_type == "etr":
        tx_label = "ETR"
    elif rate_type == "mtrx":
        tx_label = "MTRx"
    elif rate_type == "mtry":
        tx_label = "MTRy"
    ax.set_zlabel(tx_label)
    plt.title(
        tx_label
        + " vs. Predicted "
        + tx_label
        + ": Age="
        + str(s)
        + ", Year="
        + str(t)
    )

    gridpts = 50
    X_vec = np.exp(np.linspace(np.log(5), np.log(X.max()), gridpts))
    Y_vec = np.exp(np.linspace(np.log(5), np.log(Y.max()), gridpts))
    X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
    txrate_grid = txfunc.get_tax_rates(
        params_to_plot,
        X_grid,
        Y_grid,
        None,
        tax_func_type,
        rate_type,
        for_estimation=False,
    )
    ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1, linewidth=0)
    filename = tx_label + "_age_" + str(s) + "_Year_" + str(t) + "_vsPred.png"
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches="tight", dpi=300)
    plt.close()

    # Make comparison plot with truncated income domains
    df_trnc_gph = df[
        (df["total_labinc"] > 5)
        & (df["total_labinc"] < 800000)
        & (df["total_capinc"] > 5)
        & (df["total_capinc"] < 800000)
    ]
    X_gph = df_trnc_gph["total_labinc"]
    Y_gph = df_trnc_gph["total_capinc"]
    if rate_type == "etr":
        txrates_gph = df_trnc_gph["etr"]
    elif rate_type == "mtrx":
        txrates_gph = df_trnc_gph["mtr_labinc"]
    elif rate_type == "mtry":
        txrates_gph = df_trnc_gph["mtr_capinc"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_gph, Y_gph, txrates_gph, c="r", marker="o")
    ax.set_xlabel("Total Labor Income")
    ax.set_ylabel("Total Capital Income")
    ax.set_zlabel(tx_label)
    plt.title(
        "Truncated "
        + tx_label
        + ", Lab. Inc., and Cap. "
        + "Inc., Age="
        + str(s)
        + ", Year="
        + str(t)
    )

    gridpts = 50
    X_vec = np.exp(np.linspace(np.log(5), np.log(X_gph.max()), gridpts))
    Y_vec = np.exp(np.linspace(np.log(5), np.log(Y_gph.max()), gridpts))
    X_grid, Y_grid = np.meshgrid(X_vec, Y_vec)
    txrate_grid = txfunc.get_tax_rates(
        params_to_plot,
        X_grid,
        Y_grid,
        None,
        tax_func_type,
        rate_type,
        for_estimation=False,
    )
    ax.plot_surface(X_grid, Y_grid, txrate_grid, cmap=cmap1, linewidth=0)
    filename = (
        tx_label + "trunc_age_" + str(s) + "_Year_" + str(t) + "_vsPred.png"
    )
    fullpath = os.path.join(output_dir, filename)
    fig.savefig(fullpath, bbox_inches="tight", dpi=300)
    plt.close()


def txfunc_sse_plot(age_vec, sse_mat, start_year, varstr, output_dir, round):
    """
    Plot sum of squared errors of tax functions over age for each year
    of budget window.

    Args:
        age_vec (numpy array): vector of ages, length S
        sse_mat (Numpy array): SSE for each estimated tax function,
            size is BW x S
        start_year (int): first year of budget window
        varstr (str): name of tax function being evaluated
        path (str): path to save graph to
        round (int): which round of sweeping for outliers (0, 1, or 2)

    Returns:
        None

    """
    fig, ax = plt.subplots()
    BW = sse_mat.shape[0]
    for y in range(BW):
        plt.plot(age_vec, sse_mat[y, :], label=str(start_year + y))
    plt.legend(loc="upper left")
    titletext = (
        "Sum of Squared Errors by age and Tax Year"
        + " minus outliers (Round "
        + str(round)
        + "): "
        + varstr
    )
    plt.title(titletext)
    plt.xlabel(r"age $s$")
    plt.ylabel(r"SSE")
    graphname = "SSE_" + varstr + "_Round" + str(round)
    output_path = os.path.join(output_dir, graphname)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_income_data(
    ages, abil_midp, abil_pcts, emat, t=None, path=None, filesuffix=""
):
    """
    This function graphs ability matrix in 3D, 2D, log, and nolog

    Args:
        ages (Numpy array) ages represented in sample, length S
        abil_midp (Numpy array): midpoints of income percentile bins in
            each ability group
        abil_pcts (Numpy array): percent of population in each lifetime
            income group, length J
        emat (Numpy array): effective labor units by age and lifetime
            income group, size TxSxJ
        t (int): model period for year, if None, then plot SS values
        filesuffix (str): suffix to be added to plot files

    Returns:
        None

    """
    if t is None:
        t = -1
    J = abil_midp.shape[0]
    abil_mesh, age_mesh = np.meshgrid(abil_midp, ages)
    cmap1 = matplotlib.cm.get_cmap("summer")
    if path:
        # Make sure that directory is created
        utils.mkdirs(path)
        if J == 1:
            # Plot of 2D, J=1 in levels
            plt.figure()
            plt.plot(ages, emat[t, :, :])
            filename = "ability_2D_lev" + filesuffix
            fullpath = os.path.join(path, filename)
            plt.savefig(fullpath, dpi=300)
            plt.close()

            # Plot of 2D, J=1 in logs
            plt.figure()
            plt.plot(ages, np.log(emat[t, :, :]))
            filename = "ability_2D_log" + filesuffix
            fullpath = os.path.join(path, filename)
            plt.savefig(fullpath, dpi=300)
            plt.close()
        else:
            # Plot of 3D, J>1 in levels
            fig10, ax10 = plt.subplots(subplot_kw={"projection": "3d"})
            ax10.plot_surface(
                age_mesh,
                abil_mesh,
                emat[t, :, :],
                rstride=8,
                cstride=1,
                cmap=cmap1,
            )
            ax10.set_xlabel(r"age-$s$")
            ax10.set_ylabel(r"ability type -$j$")
            ax10.set_zlabel(r"ability $e_{j,s}$")
            filename = "ability_3D_lev" + filesuffix
            fullpath = os.path.join(path, filename)
            plt.savefig(fullpath, dpi=300)
            plt.close()

            # Plot of 3D, J>1 in logs
            fig11, ax11 = plt.subplots(subplot_kw={"projection": "3d"})
            ax11.plot_surface(
                age_mesh,
                abil_mesh,
                np.log(emat[t, :, :]),
                rstride=8,
                cstride=1,
                cmap=cmap1,
            )
            ax11.set_xlabel(r"age-$s$")
            ax11.set_ylabel(r"ability type -$j$")
            ax11.set_zlabel(r"log ability $log(e_{j,s})$")
            filename = "ability_3D_log" + filesuffix
            fullpath = os.path.join(path, filename)
            plt.savefig(fullpath, dpi=300)
            plt.close()

            if J <= 10:  # Restricted because of line and marker types
                # Plot of 2D lines from 3D version in logs
                ax = plt.subplot(111)
                linestyles = np.array(
                    [
                        "-",
                        "--",
                        "-.",
                        ":",
                    ]
                )
                markers = np.array(["x", "v", "o", "d", ">", "|"])
                pct_lb = 0
                for j in range(J):
                    this_label = (
                        str(int(np.rint(pct_lb)))
                        + " - "
                        + str(int(np.rint(pct_lb + 100 * abil_pcts[j])))
                        + "%"
                    )
                    pct_lb += 100 * abil_pcts[j]
                    if j <= 3:
                        ax.plot(
                            ages,
                            np.log(emat[t, :, j]),
                            label=this_label,
                            linestyle=linestyles[j],
                            color="black",
                        )
                    elif j > 3:
                        ax.plot(
                            ages,
                            np.log(emat[t, :, j]),
                            label=this_label,
                            marker=markers[j - 4],
                            color="black",
                        )
                ax.axvline(x=80, color="black", linestyle="--")
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                ax.set_xlabel(r"age-$s$")
                ax.set_ylabel(r"log ability $log(e_{j,s})$")
                filename = "ability_2D_log" + filesuffix
                fullpath = os.path.join(path, filename)
                plt.savefig(fullpath, dpi=300)
                plt.close()
    else:
        if J <= 10:  # Restricted because of line and marker types
            # Plot of 2D lines from 3D version in logs
            ax = plt.subplot(111)
            linestyles = np.array(
                [
                    "-",
                    "--",
                    "-.",
                    ":",
                ]
            )
            markers = np.array(["x", "v", "o", "d", ">", "|"])
            pct_lb = 0
            for j in range(J):
                this_label = (
                    str(int(np.rint(pct_lb)))
                    + " - "
                    + str(int(np.rint(pct_lb + 100 * abil_pcts[j])))
                    + "%"
                )
                pct_lb += 100 * abil_pcts[j]
                if j <= 3:
                    ax.plot(
                        ages,
                        np.log(emat[t, :, j]),
                        label=this_label,
                        linestyle=linestyles[j],
                        color="black",
                    )
                elif j > 3:
                    ax.plot(
                        ages,
                        np.log(emat[t, :, j]),
                        label=this_label,
                        marker=markers[j - 4],
                        color="black",
                    )
            ax.axvline(x=80, color="black", linestyle="--")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_xlabel(r"age-$s$")
            ax.set_ylabel(r"log ability $log(e_{j,s})$")

            return ax


def plot_2D_taxfunc(
    year,
    start_year,
    tax_param_list,
    age=None,
    E=21,  # Age at which agents become economically active in the model
    tax_func_type=["DEP"],
    rate_type="etr",
    over_labinc=True,
    other_inc_val=1000,
    max_inc_amt=1000000,
    data_list=None,
    labels=["1st Functions"],
    title=None,
    path=None,
):
    """
    This function plots OG-Core tax functions in two dimensions.
    The tax rates are plotted over capital or labor income, as
    entered by the user.

    Args:
        year (int): year of policy tax functions represent
        start_year (int): first year tax functions estimated for in
            tax_param_list elements
        tax_param_list (list): list of arrays containing tax function
            parameters
        age (int): age for tax functions to plot, use None if tax
            function parameters were not age specific
        tax_func_type (list): list of strings in ["DEP", "DEP_totalinc",
            "GS", "linear"] and specifies functional form of tax functions
            in tax_param_list
        rate_type (str): string that is in ["etr", "mtrx", "mtry"] and
            determines the type of tax rate that is plotted
        over_labinc (bool): indicates that x-axis of the plot is over
            labor income, if False then plot is over capital income
        other_inc_val (scalar): dollar value at which to hold constant
            the amount of income that is not represented on the x-axis
        max_inc_amt (scalar): largest income amount to represent on the
            x-axis of the plot
        data_list (list): list of DataFrames with data to scatter plot
            with tax functions, needs to be of format output from
            ogcore.get_micro_data.get_data
        labels (list): list of labels for tax function parameters
        title (str): title for the plot
        path (str): path to which to save plot, if None then figure
            returned

    Returns:
        fig (Matplotlib plot object): plot of tax functions

    """
    # Check that inputs are valid
    assert isinstance(start_year, int)
    assert isinstance(year, int)
    assert year >= start_year
    # if list of tax function types less than list of params, assume
    # all the same functional form
    if len(tax_func_type) < len(tax_param_list):
        tax_func_type = [tax_func_type[0]] * len(tax_param_list)
    for i, v in enumerate(tax_func_type):
        assert v in ["DEP", "DEP_totalinc", "GS", "linear", "mono", "mono2D"]
    assert rate_type in ["etr", "mtrx", "mtry"]
    assert len(tax_param_list) == len(labels)

    # Set age and year to look at
    if age is not None:
        assert isinstance(age, int)
        assert age >= E
        s = (
            age - E
        )  # Note: assumed age is given in E + model periods (but age below is also assumed to be calendar years)
    else:
        s = 0  # if not age-specific, all ages have the same values
    t = year - start_year

    # create rate_key to correspond to keys in tax func dicts
    rate_key = "tfunc_" + rate_type + "_params_S"

    # Set income range to plot over (min income value hard coded to 5)
    inc_sup = np.exp(np.linspace(np.log(5), np.log(max_inc_amt), 100))
    # Set income value for other income
    inc_fix = other_inc_val

    if over_labinc:
        key1 = "total_labinc"
        X = inc_sup
        Y = inc_fix
    else:
        key1 = "total_capinc"
        X = inc_fix
        Y = inc_sup

    # get tax rates for each point in the income support and plot
    fig, ax = plt.subplots()
    for i, tax_params in enumerate(tax_param_list):
        tax_params = tax_params[rate_key][t][s]
        rates = txfunc.get_tax_rates(
            tax_params,
            X,
            Y,
            None,
            tax_func_type[i],
            rate_type,
            for_estimation=False,
        )
        plt.plot(inc_sup, rates, label=labels[i])

    # plot raw data (if passed)
    if data_list is not None:
        rate_type_dict = {
            "etr": "etr",
            "mtrx": "mtr_labinc",
            "mtry": "mtr_capinc",
        }
        # censor data to range of the plot
        for d, data in enumerate(data_list):
            data_to_plot = data[str(year)].copy()
            if age is not None:
                data_to_plot.drop(
                    data_to_plot[data_to_plot["age"] != age].index,
                    inplace=True,
                )
            # other censoring
            data_to_plot.drop(
                data_to_plot[data_to_plot[key1] > max_inc_amt].index,
                inplace=True,
            )
            # other censoring used in txfunc.py
            data_to_plot = txfunc.tax_data_sample(data_to_plot)
            # set number of bins to 100 or bins of $1000 dollars
            n_bins = min(100, np.floor_divide(max_inc_amt, 1000))
            # need to compute weighted averages by group...

            def weighted_mean(x, cols, w="weight"):
                try:
                    return pd.Series(
                        np.average(x[cols], weights=x[w], axis=0), cols
                    )
                except ZeroDivisionError:
                    return 0

            data_to_plot["inc_bin"] = pd.cut(data_to_plot[key1], n_bins)
            groups = data_to_plot.groupby("inc_bin", observed=True).apply(
                weighted_mean, [rate_type_dict[rate_type], key1]
            )
            plt.scatter(
                groups[key1], groups[rate_type_dict[rate_type]], alpha=0.1
            )
    # add legend, labels, etc to plot
    plt.legend(loc="center right")
    if title:
        plt.title(title)
    if over_labinc:
        plt.xlabel(r"Labor income")
    else:
        plt.xlabel(r"Capital income")
    plt.ylabel(VAR_LABELS[rate_type])
    if path is None:
        return fig
    else:
        plt.savefig(path, dpi=300)
