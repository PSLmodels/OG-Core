import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('OGUSAplots.mplstyle')


def plot_imm_rates(p, year=2019, path=None):
    '''
    Create a plot of immigration rates from OG-USA parameterization.
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
