import numpy as np
import os
import matplotlib.pyplot as plt
cur_path = os.path.split(os.path.abspath(__file__))[0]
style_file = os.path.join(cur_path, 'OGUSAplots.mplstyle')
plt.style.use(style_file)


def plot_imm_rates(p, year=2019, include_title=False, path=None):
    '''
    Create a plot of immigration rates from OG-USA parameterization.

    Args:
        p (OG-USA Specifications class): parameters object
        year (integer): year of mortality ratese to plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of immigration rates
    '''
    assert (isinstance(year, int))
    age_per = np.linspace(p.E, p.E + p.S, p.S)
    fig, ax = plt.subplots()
    plt.scatter(age_per, p.imm_rates[year - p.start_year, :], s=40,
                marker='d')
    plt.plot(age_per, p.imm_rates[year - p.start_year, :])
    plt.xlabel(r'Age $s$ (model periods)')
    plt.ylabel(r'Imm. rate $i_{s}$')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    if include_title:
            plt.title('Immigration Rates in ' + str(year))
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "imm_rates_orig")
        plt.savefig(fig_path)


def plot_mort_rates(p, include_title=False, path=None):
    '''
    Create a plot of mortality rates from OG-USA parameterization.

    Args:
        p (OG-USA Specifications class): parameters object
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of immigration rates
    '''
    age_per = np.linspace(p.E, p.E + p.S, p.S)
    fig, ax = plt.subplots()
    plt.plot(age_per, p.rho)
    plt.xlabel(r'Age $s$ (model periods)')
    plt.ylabel(r'Mortality Rates $\rho_{s}$')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    if include_title:
            plt.title('Mortality Rates')
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "mortality_rates")
        plt.savefig(fig_path)


def plot_pop_growth(p, start_year=2019, include_title=False,
                    num_years_to_plot=150, path=None):
    '''
    Create a plot of population growth rates by year.

    Args:
        p (OG-USA Specifications class): parameters object
        start_year (integer): year to begin plotting
        num_years_to_plot (integer): number of years to plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of immigration rates
    '''
    assert (isinstance(start_year, int))
    assert (isinstance(num_years_to_plot, int))
    year_vec = np.arange(start_year, start_year + num_years_to_plot)
    start_index = start_year - p.start_year
    fig, ax = plt.subplots()
    plt.plot(year_vec, p.g_n[start_index: start_index +
                             num_years_to_plot])
    plt.xlabel(r'Year $t$')
    plt.ylabel(r'Population Growth Rate $g_{n, t}$')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    if include_title:
            plt.title('Population Growth Rates')
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "pop_growth_rates")
        plt.savefig(fig_path)


def plot_ability_profiles(p, include_title=False, path=None):
    '''
    Create a plot of earnings ability profiles.

    Args:
        p (OG-USA Specifications class): parameters object
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of earnings ability profiles
    '''
    age_vec = np.arange(p.starting_age, p.starting_age + p.S)
    fig, ax = plt.subplots()
    for j in range(p.J):
        plt.plot(age_vec, p.e[:, j], label='j = ' + str(j))
    plt.xlabel(r'Age')
    plt.ylabel(r'Earnings ability')
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    if include_title:
            plt.title('Lifecycle Profiles of Effective Labor Units')
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "ability_profiles")
        plt.savefig(fig_path)


def plot_elliptical_u(p, plot_MU=True, include_title=False, path=None):
    '''
    Create a plot of showing the fit of the elliptical utility function.

    Args:
        p (OG-USA Specifications class): parameters object
        plot_MU (boolean): whether plot marginal utility or utility in
            levels
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of elliptical vs CFE utility
    '''
    theta = 1 / p.frisch
    N = 101
    n_grid = np.linspace(0.01, 0.8, num=N)
    if plot_MU:
        CFE = (1.0 / p.ltilde) * ((n_grid / p.ltilde) ** theta)
        ellipse = (1.0 * p.b_ellipse * (1.0 / p.ltilde) *
                   ((1.0 - (n_grid / p.ltilde) ** p.upsilon) **
                    ((1.0 / p.upsilon) - 1.0)) *
                   (n_grid / p.ltilde) ** (p.upsilon - 1.0))
    else:
        CFE = ((n_grid / p.ltilde) ** (1 + theta)) / (1 + theta)
        k = 1.0  # we don't estimate k, so not in parameters
        ellipse = (p.b_ellipse * ((1 - ((n_grid / p.ltilde) **
                                        p.upsilon)) **
                                  (1 / p.upsilon)) + k)
    fig, ax = plt.subplots()
    plt.plot(n_grid, CFE, label='CFE')
    plt.plot(n_grid, ellipse, label='Elliptical U')
    if include_title:
        if plot_MU:
            plt.title('Marginal Utility of CFE and Elliptical')
        else:
            plt.title('Constant Frisch Elasticity vs. Elliptical Utility')
    plt.xlabel(r'Labor Supply')
    if plot_MU:
        plt.ylabel(r'Marginal Utility')
    else:
        plt.ylabel(r'Utility')
    plt.legend(loc='center right')
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "ellipse_v_CFE")
        plt.savefig(fig_path)
