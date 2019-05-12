import numpy as np
import os
import matplotlib.pyplot as plt
cur_path = os.path.split(os.path.abspath(__file__))[0]
style_file = os.path.join(cur_path, 'OGUSAplots.mplstyle')
plt.style.use(style_file)


def plot_imm_rates(p, year=2019, path=None):
    '''
    Create a plot of immigration rates from OG-USA parameterization.

    Args:
        p (OG-USA Specifications class): parameters object
        year (integer): year of mortality ratese to plot
        path (string): path to save figure to

    Returns:
        fig (Matplotlib plot object): plot of immigration rates
    '''
    age_per = np.linspace(p.E, p.E + p.S, p.S)
    fig, ax = plt.subplots()
    plt.scatter(age_per, p.imm_rates[year - p.start_year, :], s=40,
                marker='d')
    plt.plot(age_per, p.imm_rates[year - p.start_year, :])
    plt.xlabel(r'Age $s$ (model periods)')
    plt.ylabel(r'Imm. rate $i_{s}$')

    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "imm_rates_orig")
        plt.savefig(fig_path)


def plot_ability_profiles(p, path=None):
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
    if path is None:
        return fig
    else:
        fig_path = os.path.join(path, "imm_rates_orig")
        plt.savefig(fig_path)
