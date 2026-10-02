import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from ogcore.constants import GROUP_LABELS
from ogcore import utils, txfunc
from ogcore.constants import DEFAULT_START_YEAR, VAR_LABELS


# Line styles cycled over when plotting one line per income group j
# alongside a colormap over years (see plot_fert_rates)
J_LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
    (0, (3, 2, 1, 2, 1, 2)),
]


def _check_ndim(arr, name, allowed_ndims):
    """
    Check that a demographic object has one of the allowed numbers of
    dimensions.  Objects are allowed either without an income-group
    (J) dimension (backward compatible with earlier versions of
    OG-Core) or with J as the last dimension.

    Args:
        arr (array_like): object to check
        name (str): name of the object, used in the error message
        allowed_ndims (tuple): allowed numbers of dimensions

    Returns:
        arr (NumPy array): the object as a NumPy array of floats

    """
    arr = np.asarray(arr, dtype=float)
    assert arr.ndim in allowed_ndims, (
        name
        + " must have "
        + " or ".join(str(n) for n in allowed_ndims)
        + " dimensions (with income groups j as the last dimension, if "
        + "present), got shape "
        + str(arr.shape)
    )
    return arr


def _avg_over_J(rates, omega=None):
    """
    Average demographic rates over the income-group (J) dimension,
    which is assumed to be the last axis of ``rates``.

    Args:
        rates (NumPy array): array of rates whose last axis indexes
            income groups j
        omega (NumPy array or None): population distribution with the
            same shape as ``rates`` used as weights.  The weights are
            normalized within each cell of the leading dimensions (e.g.,
            within each age) so that they sum to one across j.  If
            None, a simple (unweighted) mean across j is used.

    Returns:
        avg (NumPy array): rates averaged over j (last axis removed)

    """
    rates = np.asarray(rates, dtype=float)
    if omega is None:
        return rates.mean(axis=-1)
    omega = np.asarray(omega, dtype=float)
    assert omega.shape == rates.shape, (
        "omega must have the same shape as the rates being averaged, "
        + "got "
        + str(omega.shape)
        + " and "
        + str(rates.shape)
    )
    omega_sum = omega.sum(axis=-1, keepdims=True)
    # where an age has zero population, fall back to equal weights
    weights = np.divide(
        omega,
        omega_sum,
        out=np.full(omega.shape, 1.0 / omega.shape[-1]),
        where=omega_sum != 0,
    )
    return (rates * weights).sum(axis=-1)


def _sum_over_J(dist):
    """
    Collapse a population distribution over the income-group (J)
    dimension (assumed to be the last axis) by summing, which yields
    the marginal distribution over the remaining dimensions.

    Args:
        dist (NumPy array): population distribution whose last axis
            indexes income groups j

    Returns:
        marginal (NumPy array): distribution summed over j

    """
    return np.asarray(dist, dtype=float).sum(axis=-1)


def _plot_age_profile(
    ax,
    arr,
    label,
    by_J,
    x=None,
    omega=None,
    collapse="avg",
    transform=None,
    **plot_kwargs,
):
    """
    Plot an age profile that may or may not have an income-group (J)
    dimension.

    If ``arr`` is one-dimensional (S,), a single line is plotted.  If
    ``arr`` is two-dimensional (S x J) and ``by_J`` is True, one line
    is plotted for each income group j and labeled with the j value.
    If ``arr`` is (S x J) and ``by_J`` is False, the array is first
    collapsed over j (summed for distributions, population-weighted
    averaged for rates) and a single line is plotted.

    Args:
        ax (Matplotlib axes object): axes to plot on
        arr (NumPy array): age profile, (S,) or (S x J)
        label (str): legend label for the series (", j=k" is appended
            when plotting by income group)
        by_J (bool): whether to plot a separate line for each j
        x (array_like or None): x-axis values, if None then the index
            of ``arr`` is used
        omega (NumPy array or None): population distribution of the
            same shape as ``arr`` used to weight the average across
            j when ``collapse`` is "avg"
        collapse (str): "sum" to sum over j (for distributions) or
            "avg" to take the population-weighted average over j (for
            rates) when ``by_J`` is False
        transform (callable or None): function applied to each (S,)
            series just before it is plotted (e.g., to compute
            cumulative survival rates from mortality rates)
        plot_kwargs (dict): additional keyword arguments passed to
            ``ax.plot``

    Returns:
        None

    """
    assert collapse in ["sum", "avg"], "collapse must be 'sum' or 'avg'"
    arr = np.asarray(arr, dtype=float)
    assert arr.ndim in [1, 2], (
        "Expected an age profile of shape (S,) or (S, J), got shape "
        + str(arr.shape)
    )
    if transform is None:

        def transform(y):
            return y

    if arr.ndim == 2:
        if by_J:
            series = [
                (arr[:, j], label + ", j=" + str(j))
                for j in range(arr.shape[-1])
            ]
        elif collapse == "sum":
            series = [(_sum_over_J(arr), label)]
        else:
            series = [(_avg_over_J(arr, omega), label)]
    else:
        series = [(arr, label)]
    for y, lab in series:
        if x is None:
            ax.plot(transform(y), label=lab, **plot_kwargs)
        else:
            ax.plot(x, transform(y), label=lab, **plot_kwargs)


def plot_imm_rates(
    imm_rates,
    start_year=DEFAULT_START_YEAR,
    years_to_plot=[DEFAULT_START_YEAR],
    include_title=False,
    source="United Nations, World Population Prospects",
    path=None,
    by_J=False,
    omega=None,
):
    """
    Plot immigration rates from the data

    Args:
        imm_rates (NumPy array): immigration rates for each year and
            age (T x S) or for each year, age, and income group
            (T x S x J)
        start_year (int): first year of data
        years_to_plot (list): list of years to plot
        include_title (bool): whether to include a title in the plot
        source (str): data source for immigration rates
        path (str): path to save figure to, if None then figure
            is returned
        by_J (bool): if imm_rates has an income-group dimension, plot
            a separate line for each income group j (labeled by j).
            If False, the rates are averaged across income groups so
            that one line is plotted per year.
        omega (NumPy array): population distribution with the same
            shape as imm_rates, used to weight the average across
            income groups when by_J is False.  If None, a simple mean
            across income groups is used.

    Returns:
        fig (Matplotlib plot object): plot of immigration rates

    """
    imm_rates = _check_ndim(imm_rates, "imm_rates", (2, 3))
    if omega is not None:
        omega = _check_ndim(omega, "omega", (imm_rates.ndim,))
    plot_by_J = by_J and imm_rates.ndim == 3
    fig, ax = plt.subplots()
    for y in years_to_plot:
        i = y - start_year
        omega_i = None if omega is None else omega[i]
        _plot_age_profile(
            ax,
            imm_rates[i],
            "Year " + str(y),
            by_J,
            omega=omega_i,
            collapse="avg",
            **({} if plot_by_J else {"c": "blue"}),
        )
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Immigration rate $i_{s}$")
    plt.legend(loc="upper left")
    plt.text(
        -5,
        -0.05,
        "Source: " + source,
        fontsize=9,
    )
    if include_title:
        plt.title("Immigration Rates by Age")
    # Save or return figure
    if path:
        output_path = os.path.join(path, "imm_rates")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
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
    by_J=False,
):
    """
    Create a plot of mortality rates from OG-Core parameterization.

    Args:
        p_list (list): list of parameters objects
        labels (list): list of labels for the legend
        years (list): list of years to plot
        survival_rates (bool): whether to plot survival rates instead
            of mortality rates
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to
        by_J (bool): if mortality rates vary by income group, plot a
            separate line for each income group j (labeled by j).  If
            False, the rates are averaged across income groups using
            the population weights in omega so that one line is
            plotted per year and parameters object.

    Returns:
        fig (Matplotlib plot object): plot of mortality rates

    """
    p0 = p_list[0]
    for p in p_list:
        _check_ndim(p.rho, "rho", (2, 3))
        _check_ndim(p.omega, "omega", (2, 3))
    age_per = np.linspace(p0.E, p0.E + p0.S, p0.S)
    fig, ax = plt.subplots()
    if survival_rates:

        def transform(rho_t):
            return np.cumprod(1 - rho_t)

    else:
        transform = None
    for y in years:
        t = y - p0.start_year
        for i, p in enumerate(p_list):
            omega_t = None
            if p.rho.ndim == 3 and not by_J:
                # population weights used to average over j
                if p.omega.ndim == 3:
                    omega_t = p.omega[t, :, :]
                else:
                    omega_t = np.tile(p.lambdas.reshape(1, p.J), (p.S, 1))
            _plot_age_profile(
                ax,
                p.rho[t],
                labels[i] + " " + str(y),
                by_J,
                x=age_per,
                omega=omega_t,
                collapse="avg",
                transform=transform,
            )
    plt.xlabel(r"Age $s$ (model periods)")
    if survival_rates:
        plt.ylabel(r"Cumulative Survival Rates")
        plt.legend(loc="lower left")
        title = "Survival Rates"
    else:
        plt.ylabel(r"Mortality Rates $\rho_{s}$")
        plt.legend(loc="upper left")
        title = "Mortality Rates"
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ticks_loc])
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
    # g_n stores the pre-time-path boundary growth separately in
    # g_n_preTP; prepend it to recover the year-aligned growth path.
    g_n_full = np.append(p.g_n_preTP, p.g_n)
    plt.plot(
        year_vec,
        g_n_full[start_index : start_index + num_years_to_plot],
    )
    plt.xlabel(r"Year $t$")
    plt.ylabel(r"Population Growth Rate $g_{n, t}$")
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels(["{:,.2%}".format(x) for x in ticks_loc])
    if include_title:
        plt.title("Population Growth Rates")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "pop_growth_rates")
        plt.savefig(fig_path, dpi=300)


def plot_population(
    p, years_to_plot=["SS"], include_title=False, path=None, by_J=False
):
    """
    Plot the distribution of the population over age for various years.

    Args:
        p (OG-Core Specifications class): parameters object
        years_to_plot (list): list of years to plot, 'SS' will denote
            the steady-state period
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to
        by_J (bool): if the population distribution has an
            income-group dimension, plot a separate line for each
            income group j (labeled by j).  If False, the distribution
            is summed across income groups so that one line showing
            the overall population distribution by age is plotted per
            year.

    Returns:
        fig (Matplotlib plot object): plot of population distribution

    """
    for i, v in enumerate(years_to_plot):
        assert isinstance(v, int) | (v == "SS")
        if isinstance(v, int):
            assert v >= p.start_year
    _check_ndim(p.omega_SS, "omega_SS", (1, 2))
    _check_ndim(p.omega, "omega", (2, 3))
    age_vec = np.arange(p.E, p.S + p.E)
    fig, ax = plt.subplots()
    for i, v in enumerate(years_to_plot):
        if v == "SS":
            pop_dist = p.omega_SS
        else:
            pop_dist = p.omega[v - p.start_year]
        _plot_age_profile(
            ax, pop_dist, str(v) + " pop.", by_J, x=age_vec, collapse="sum"
        )
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
        t (int): model period for year, if None, then plot ability
            matrix for SS
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
    plt.plot(n_grid, CFE, label="Constant Frisch elasticity")
    plt.plot(n_grid, ellipse, label="Elliptical disutility")
    if include_title:
        if plot_MU:
            plt.title("Marginal Utility of CFE and Elliptical")
        else:
            plt.title("Constant Frisch Elasticity vs. Elliptical Utility")
    plt.xlabel(r"Labor Supply $n_{j,s,t}$")
    if plot_MU:
        plt.ylabel(r"Marginal disutility")
    else:
        plt.ylabel(r"Disutility")
    plt.legend(loc="upper left")
    plt.grid(color="gray", linestyle=":", linewidth=1, alpha=0.5)
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
    include_title=False,
    source="United Nations, World Population Prospects",
    path=None,
    by_J=False,
    omega_list=None,
):
    """
    Plot fertility rates from the data

    Args:
        fert_rates_list (list): list of Numpy arrays of fertility rates
            by age (S,) or by age and income group (S x J), one array
            per series (e.g., per year)
        labels (list): list of labels for the legend
        include_title (bool): whether to include a title in the plot
        source (str): data source for fertility rates
        path (str): path to save figure to, if None then figure
            is returned
        by_J (bool): if the fertility rates have an income-group
            dimension, plot a separate line for each income group j
            (labeled by j).  If False, the rates are averaged across
            income groups so that one line is plotted per series.
        omega_list (list): list of Numpy arrays of the population
            distribution, each with the same shape as the corresponding
            element of fert_rates_list, used to weight the average
            across income groups when by_J is False.  If None, a
            simple mean across income groups is used.

    Returns:
        fig (Matplotlib plot object): plot of fertility rates

    """
    fig, ax = plt.subplots()
    num_series = len(fert_rates_list)
    assert num_series == len(labels), (
        "Number of series must match number of labels"
    )
    if omega_list is not None:
        assert len(omega_list) == num_series, (
            "Number of series must match number of omega arrays"
        )
    fert_rates_list = [
        _check_ndim(f, "fert_rates_list[" + str(i) + "]", (1, 2))
        for i, f in enumerate(fert_rates_list)
    ]
    if omega_list is not None:
        omega_list = [
            _check_ndim(
                om, "omega_list[" + str(i) + "]", (fert_rates_list[i].ndim,)
            )
            for i, om in enumerate(omega_list)
        ]
    plot_by_J = by_J and any(f.ndim == 2 for f in fert_rates_list)

    if num_series > 4:
        cm = plt.get_cmap("coolwarm")
        # Attempt to convert labels to numeric values for colormap scaling
        try:
            label_values = [int(label) for label in labels]
        except (ValueError, TypeError):
            label_values = list(range(num_series))

        vmin, vmax = min(label_values), max(label_values)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        max_J = 1
        for i, fert_rates in enumerate(fert_rates_list):
            color = cm(norm(label_values[i]))
            if fert_rates.ndim == 2 and plot_by_J:
                # one line per j: color denotes the series (year),
                # line style denotes the income group
                max_J = max(max_J, fert_rates.shape[-1])
                for j in range(fert_rates.shape[-1]):
                    ax.plot(
                        fert_rates[:, j],
                        color=color,
                        linestyle=J_LINESTYLES[j % len(J_LINESTYLES)],
                        label=str(labels[i]) + ", j=" + str(j),
                    )
            else:
                omega_i = None if omega_list is None else omega_list[i]
                _plot_age_profile(
                    ax,
                    fert_rates,
                    str(labels[i]),
                    by_J=False,
                    omega=omega_i,
                    collapse="avg",
                    color=color,
                )
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Year")
        if plot_by_J:
            # legend keyed on line style to identify income groups
            handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=J_LINESTYLES[j % len(J_LINESTYLES)],
                    label="j=" + str(j),
                )
                for j in range(max_J)
            ]
            ax.legend(handles=handles, loc="upper right")
    else:
        for i, fert_rates in enumerate(fert_rates_list):
            omega_i = None if omega_list is None else omega_list[i]
            _plot_age_profile(
                ax,
                fert_rates,
                str(labels[i]),
                by_J,
                omega=omega_i,
                collapse="avg",
            )
        ax.legend(loc="upper right")

    if include_title:
        ax.set_title("Fertility rates by age ($f_{s}$)", fontsize=20)
    ax.set_xlabel(r"Age $s$")
    ax.set_ylabel(r"Fertility rate $f_{s}$")
    ax.text(
        -5,
        -0.023,
        "Source: " + source,
        fontsize=9,
    )

    # Save or return figure
    if path:
        output_path = os.path.join(path, "fert_rates")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
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
    by_J=False,
    omega=None,
):
    """
    Plots mortality rates from the data.

    Args:
        mort_rates (array_like): mortality rates for each year and
            age (T x S) or for each year, age, and income group
            (T x S x J)
        start_year (int): first year of data
        years_to_plot (list): list of years to plot
        source (str): data source for mortality rates
        path (str): path to save figure to, if None then figure
            is returned
        by_J (bool): if mort_rates has an income-group dimension, plot
            a separate line for each income group j (labeled by j).
            If False, the rates are averaged across income groups so
            that one line is plotted per year.
        omega (NumPy array): population distribution with the same
            shape as mort_rates, used to weight the average across
            income groups when by_J is False.  If None, a simple mean
            across income groups is used.

    Returns:
        fig (Matplotlib plot object): plot of mortality rates

    """
    mort_rates = _check_ndim(mort_rates, "mort_rates", (2, 3))
    if omega is not None:
        omega = _check_ndim(omega, "omega", (mort_rates.ndim,))
    plot_by_J = by_J and mort_rates.ndim == 3
    fig, ax = plt.subplots()
    for y in years_to_plot:
        i = y - start_year
        omega_i = None if omega is None else omega[i]
        _plot_age_profile(
            ax,
            mort_rates[i],
            "Year " + str(y),
            by_J,
            omega=omega_i,
            collapse="avg",
            **({} if plot_by_J else {"c": "blue"}),
        )
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
        plt.plot(
            years,
            np.append(p.g_n_preTP, p.g_n[: p.T - 1]),
            label=label_list[i],
        )
    plt.xlabel(r"Year $s$ (model periods)")
    plt.ylabel(r"Population Growth Rate $g_{n,t}$")
    if label_list[0] != "":
        plt.legend(loc="upper right")
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels(["{:,.0%}".format(x) for x in ticks_loc])
    if include_title:
        plt.title("Population Growth Rates")
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "pop_growth_rates")
        plt.savefig(fig_path, dpi=300)


def plot_omega_fixed(
    age_per_EpS, omega_SS_orig, omega_SSfx, E, S, path=None, by_J=False
):
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
            without adjustment to immigration rates, by age (E+S,) or
            by age and income group (E+S x J)
        omega_SSfx (Numpy array): population distribution in SS
            after adjustment to immigration rates, by age (E+S,) or
            by age and income group (E+S x J)
        E (int): age at which household becomes economically active
        S (int): number of years which household is economically active
        path (str): path to save figure to, if None then figure
            is returned
        by_J (bool): if the distributions have an income-group
            dimension, plot a separate line for each income group j
            (labeled by j).  If False, the distributions are summed
            across income groups so that one line is plotted for each
            distribution.

    Returns:
        fig (Matplotlib plot object): plot of SS population distribution
            before and after adjustment to immigration rates

    """
    omega_SS_orig = _check_ndim(omega_SS_orig, "omega_SS_orig", (1, 2))
    omega_SSfx = _check_ndim(omega_SSfx, "omega_SSfx", (1, 2))
    fig, ax = plt.subplots()
    _plot_age_profile(
        ax,
        omega_SS_orig,
        "Original Dist'n",
        by_J,
        x=age_per_EpS,
        collapse="sum",
    )
    _plot_age_profile(
        ax, omega_SSfx, "Fixed Dist'n", by_J, x=age_per_EpS, collapse="sum"
    )
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
    age_per_EpS,
    imm_rates_orig,
    imm_rates_adj,
    E,
    S,
    path=None,
    by_J=False,
    omega=None,
):
    """
    Plot the immigration rates implied by the data on population,
    mortality, and fertility versus the adjusted immigration rates
    needed to achieve a stationary distribution of the population in a
    reasonable number of model periods.

    Args:
        age_per_EpS (array_like): list of ages over which to plot
            population distribution
        imm_rates_orig (Numpy array): immigration rates by age (E+S,)
            or by age and income group (E+S x J)
        imm_rates_adj (Numpy array): adjusted immigration rates by age
            (E+S,) or by age and income group (E+S x J)
        E (int): age at which household becomes economically active
        S (int): number of years which household is economically active
        path (str): path to save figure to, if None then figure
            is returned
        by_J (bool): if the immigration rates have an income-group
            dimension, plot a separate line for each income group j
            (labeled by j).  If False, the rates are averaged across
            income groups so that one line is plotted for each set of
            rates.
        omega (NumPy array): population distribution with the same
            shape as the immigration rates, used to weight the average
            across income groups when by_J is False.  If None, a
            simple mean across income groups is used.

    Returns:
        fig (Matplotlib plot object): plot of immigration rates found
            from residuals and the adjusted rates to hit SS sooner

    """
    imm_rates_orig = _check_ndim(imm_rates_orig, "imm_rates_orig", (1, 2))
    imm_rates_adj = _check_ndim(imm_rates_adj, "imm_rates_adj", (1, 2))
    if omega is not None:
        omega = _check_ndim(omega, "omega", (1, 2))
    fig, ax = plt.subplots()
    _plot_age_profile(
        ax,
        imm_rates_orig,
        "Original Imm. Rates",
        by_J,
        x=age_per_EpS,
        omega=omega,
        collapse="avg",
    )
    _plot_age_profile(
        ax,
        imm_rates_adj,
        "Adj. Imm. Rates",
        by_J,
        x=age_per_EpS,
        omega=omega,
        collapse="avg",
    )
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
    by_J=False,
):
    """
    Plot the distribution of the population over age for various years.

    Args:
        age_per_EpS (array_like): list of ages over which to plot
            population distribution
        omega_path_lev (Numpy array): number of households by age
            (T+S x E+S) or by age and income group (T+S x E+S x J)
            over the transition path
        omega_SSfx (Numpy array): population distribution by age (E+S,)
            or by age and income group (E+S x J) in the SS
        start_year (int): first year of data (so can get index of year1
            and year2)
        year1 (int): first year of data to plot
        year2 (int): second year of data to plot
        S (int): number of years which household is economically active
        path (str): path to save figure to, if None then figure
            is returned
        by_J (bool): if the population distributions have an
            income-group dimension, plot a separate line for each
            income group j (labeled by j).  If False, the distributions
            are summed across income groups so that one line showing
            the overall population distribution by age is plotted for
            each point in time.

    Returns:
        fig (Matplotlib plot object): plot of population distribution
            at points along the time path

    """
    omega_path_lev = _check_ndim(omega_path_lev, "omega_path_lev", (2, 3))
    omega_SSfx = _check_ndim(omega_SSfx, "omega_SSfx", (1, 2))
    fig, ax = plt.subplots()
    periods = [
        (year1 - start_year, str(year1) + " pop."),
        (year2 - start_year, str(year2) + " pop."),
        (int(0.5 * S), "T=" + str(int(0.5 * S)) + " pop."),
        (int(S), "T=" + str(int(S)) + " pop."),
    ]
    for t, label in periods:
        # population distribution across all (age, income group) cells
        pop_dist = omega_path_lev[t] / omega_path_lev[t].sum()
        _plot_age_profile(
            ax, pop_dist, label, by_J, x=age_per_EpS, collapse="sum"
        )
    _plot_age_profile(
        ax, omega_SSfx, "Adj. SS pop.", by_J, x=age_per_EpS, collapse="sum"
    )
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
    cmap1 = matplotlib.colormaps.get_cmap("summer")

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
    cmap1 = matplotlib.colormaps["summer"]
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
        # Note: assumed age is given in E + model periods (but age
        # below is also assumed to be calendar years)
        s = age - E
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
