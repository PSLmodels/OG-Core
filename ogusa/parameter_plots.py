import numpy as np
import os
import matplotlib.pyplot as plt
from ogusa.constants import GROUP_LABELS
CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
style_file = os.path.join(CUR_PATH, 'OGUSAplots.mplstyle')
plt.style.use(style_file)


def plot_imm_rates(p, year=2019, include_title=False, path=None):
    '''
    Create a plot of immigration rates from OG-USA parameterization.

    Args:
        p (OG-USA Specifications class): parameters object
        year (integer): year of mortality ratese to plot
        include_title (bool): whether to include a title in the plot
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
        include_title (bool): whether to include a title in the plot
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


def plot_pop_growth(p, start_year=2019, num_years_to_plot=150,
                    include_title=False, path=None):
    '''
    Create a plot of population growth rates by year.

    Args:
        p (OG-USA Specifications class): parameters object
        start_year (integer): year to begin plotting
        num_years_to_plot (integer): number of years to plot
        include_title (bool): whether to include a title in the plot
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


def plot_population(p, years_to_plot=['SS'], include_title=False,
                    path=None):
    '''
    Plot the distribution of the population over age for various years.

    Args:
        p (OG-USA Specifications class): parameters object
        years_to_plot (list): list of years to plot, 'SS' will denote
            the steady-state period
        include_title (bool): whether to include a title in the plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of population distribution

    '''
    for i, v in enumerate(years_to_plot):
        assert (isinstance(v, int) | (v == 'SS'))
        if isinstance(v, int):
            assert (v >= p.start_year)
    age_vec = np.arange(p.E, p.S + p.E)
    fig, ax = plt.subplots()
    for i, v in enumerate(years_to_plot):
        if v == 'SS':
            pop_dist = p.omega_SS
        else:
            pop_dist = p.omega[v - p.start_year, :]
        plt.plot(age_vec, pop_dist, label=str(v) + ' pop.')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r"Pop. dist'n $\omega_{s}$")
    plt.legend(loc='lower left')
    if include_title:
        plt.title('Population Distribution by Year')
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "pop_distribution")
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
    cm = plt.get_cmap('coolwarm')
    ax.set_prop_cycle(color=[cm(1. * i / 7) for i in range(7)])
    for j in range(p.J):
        plt.plot(age_vec, p.e[:, j], label=GROUP_LABELS[j])
    plt.xlabel(r'Age')
    plt.ylabel(r'Earnings ability')
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    if include_title:
        plt.title('Lifecycle Profiles of Effective Labor Units')
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "ability_profiles")
        plt.savefig(fig_path, bbox_inches='tight')


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


def plot_chi_n(p, include_title=False, path=None):
    '''
    Create a plot of showing the values of the chi_n parameters.

    Args:
        p (OG-USA Specifications class): parameters object
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of chi_n parameters

    '''
    age = np.linspace(p.starting_age, p.ending_age, p.S)
    fig, ax = plt.subplots()
    plt.plot(age, p.chi_n)
    if include_title:
        plt.title('Utility Weight on the Disutility of Labor Supply')
    plt.xlabel('Age, $s$')
    plt.ylabel(r'$\chi^{n}_{s}$')
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "chi_n_values")
        plt.savefig(fig_path)


def plot_fert_rates(fert_func, age_midp, totpers, min_yr, max_yr,
                    fert_data, fert_rates):
    '''
    Plot fertility rates from the data along with smoothed function to
    use for model fertility rates.

    Args:
        fert_func (Scipy interpolation object): interpolated fertility
            rates
        age_midp (NumPy array): midpoint of age for each age group in
            data
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        fert_data (NumPy array): fertility rates by age group from data
        fert_rates (NumPy array): fitted fertility rates for each of
            totpers

    Returns:
        None

    '''
    # Generate finer age vector and fertility rate vector for
    # graphing cubic spline interpolating function
    age_fine_pred = np.linspace(age_midp[0], age_midp[-1], 300)
    fert_fine_pred = fert_func(age_fine_pred)
    age_fine = np.hstack((min_yr, age_fine_pred, max_yr))
    fert_fine = np.hstack((0, fert_fine_pred, 0))
    age_mid_new = (np.linspace(np.float(max_yr) / totpers, max_yr,
                               totpers) - (0.5 * np.float(max_yr) /
                                           totpers))

    fig, ax = plt.subplots()
    plt.scatter(age_midp, fert_data, s=70, c='blue', marker='o',
                label='Data')
    plt.scatter(age_mid_new, fert_rates, s=40, c='red', marker='d',
                label='Model period (integrated)')
    plt.plot(age_fine, fert_fine, label='Cubic spline')
    # plt.title('Fitted fertility rate function by age ($f_{s}$)',
    #     fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Fertility rate $f_{s}$')
    plt.legend(loc='upper right')
    plt.text(-5, -0.023,
             'Source: National Vital Statistics Reports, ' +
             'Volume 64, Number 1, January 15, 2015.', fontsize=9)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    # Create directory if OUTPUT directory does not already exist
    output_dir = os.path.join(CUR_PATH, 'OUTPUT', 'Demographics')
    if os.access(output_dir, os.F_OK) is False:
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'fert_rates')
    plt.savefig(output_path)
    plt.close()


def plot_mort_rates_data(totpers, min_yr, max_yr, age_year_all,
                         mort_rates_all, infmort_rate,
                         mort_rates):
    '''
    Plots mortality rates from the model and data.

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        age_year_all (array_like): ages in mortality rate data
        mort_rates_all (array_like): mortality rates by age from data,
            average across males and females
        infmort_rate (scalar): infant mortality rate
        mort_rates (array_like): fitted mortality rates for each of
            totpers

    Returns:
        None

    '''
    age_mid_new = (np.linspace(np.float(max_yr) / totpers, max_yr,
                               totpers) - (0.5 * np.float(max_yr) /
                                           totpers))
    fig, ax = plt.subplots()
    plt.scatter(np.hstack([0, age_year_all]),
                np.hstack([infmort_rate, mort_rates_all]),
                s=20, c='blue', marker='o', label='Data')
    plt.scatter(np.hstack([0, age_mid_new]),
                np.hstack([infmort_rate, mort_rates]),
                s=40, c='red', marker='d',
                label='Model period (cumulative)')
    plt.plot(np.hstack([0, age_year_all[min_yr - 1:max_yr]]),
             np.hstack([infmort_rate,
                        mort_rates_all[min_yr - 1:max_yr]]))
    plt.axvline(x=max_yr, color='red', linestyle='-', linewidth=1)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    # plt.title('Fitted mortality rate function by age ($rho_{s}$)',
    #     fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Mortality rate $\rho_{s}$')
    plt.legend(loc='upper left')
    plt.text(-5, -0.2,
             'Source: Actuarial Life table, 2011 Social Security ' +
             'Administration.', fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    # Create directory if OUTPUT directory does not already exist
    output_dir = os.path.join(CUR_PATH, 'OUTPUT', 'Demographics')
    if os.access(output_dir, os.F_OK) is False:
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'mort_rates')
    plt.savefig(output_path)
    plt.close()


def plot_omega_fixed(age_per_EpS, omega_SS_orig, omega_SSfx, E, S):
    '''
    Plot the steady-state population distribution implied by the data
    on fertility and mortality rates versus the the steady-state
    population distribution after adjusting immigration rates so that
    the stationary distribution is achieved a reasonable number of
    model periods.

    Args:

    Returns:
        None

    '''
    fig, ax = plt.subplots()
    plt.plot(age_per_EpS, omega_SS_orig, label="Original Dist'n")
    plt.plot(age_per_EpS, omega_SSfx, label="Fixed Dist'n")
    plt.title('Original steady-state population distribution vs. fixed')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r"Pop. dist'n $\omega_{s}$")
    plt.xlim((0, E + S + 1))
    plt.legend(loc='upper right')
    # Create directory if OUTPUT directory does not already exist
    output_dir = os.path.join(CUR_PATH, 'OUTPUT', 'Demographics')
    if os.access(output_dir, os.F_OK) is False:
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'OrigVsFixSSpop')
    plt.savefig(output_path)
    plt.close()


def plot_imm_fixed(age_per_EpS, imm_rates_orig, imm_rates_adj, E, S):
    '''
    Plot the immigration rates implied by the data on population,
    mortality, and fertility versus the adjusted immigration rates
    needed to achieve a stationary distribution of the population in a
    reasonable number of model periods.

    Args:

    Returns:

    '''
    fig, ax = plt.subplots()
    plt.plot(age_per_EpS, imm_rates_orig, label='Original Imm. Rates')
    plt.plot(age_per_EpS, imm_rates_adj, label='Adj. Imm. Rates')
    plt.title('Original immigration rates vs. adjusted')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Imm. rates $i_{s}$')
    plt.xlim((0, E + S + 1))
    plt.legend(loc='upper center')
    # Create directory if OUTPUT directory does not already exist
    output_dir = os.path.join(CUR_PATH, 'OUTPUT', 'Demographics')
    if os.access(output_dir, os.F_OK) is False:
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'OrigVsAdjImm')
    plt.savefig(output_path)
    plt.close()


def plot_population_path(age_per_EpS, pop_2013_pct, omega_path_lev,
                         omega_SSfx, curr_year, E, S):
    '''
    Plot the distribution of the population over age for various years.

    Args:

    Returns:

    '''
    fig, ax = plt.subplots()
    plt.plot(age_per_EpS, pop_2013_pct, label='2013 pop.')
    plt.plot(age_per_EpS, (omega_path_lev[:, 0] /
                           omega_path_lev[:, 0].sum()),
             label=str(curr_year) + ' pop.')
    plt.plot(age_per_EpS, (omega_path_lev[:, int(0.5 * S)] /
                           omega_path_lev[:, int(0.5 * S)].sum()),
             label='T=' + str(int(0.5 * S)) + ' pop.')
    plt.plot(age_per_EpS, (omega_path_lev[:, int(S)] /
                           omega_path_lev[:, int(S)].sum()),
             label='T=' + str(int(S)) + ' pop.')
    plt.plot(age_per_EpS, omega_SSfx, label='Adj. SS pop.')
    plt.title('Population distribution at points in time path')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r"Pop. dist'n $\omega_{s}$")
    plt.legend(loc='lower left')
    # Create directory if OUTPUT directory does not already exist
    output_dir = os.path.join(CUR_PATH, 'OUTPUT', 'Demographics')
    if os.access(output_dir, os.F_OK) is False:
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'PopDistPath')
    plt.savefig(output_path)
    plt.close()
