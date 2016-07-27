'''
------------------------------------------------------------------------
Last updated 7/17/2015

Computes the average labor participation rate for each age cohort.

This py-file calls the following other file(s):
            data/labor/cps_hours_by_age_hourspct.txt

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Demographics/labor_dist_data_withfit.png
            OUTPUT/Demographics/data_labor_dist.png
            OUTPUT/Saved_moments/labor_data_moments.pkl
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''
import os
import numpy as np
import pandas as pd
import cPickle as pickle
import utils


'''
------------------------------------------------------------------------
    Import Data
------------------------------------------------------------------------
'''
def get_labor_data():

    cur_path = os.path.split(os.path.abspath(__file__))[0]
    LABOR_DIR = os.path.join(cur_path, "data", "labor")

    '''
    Need to:
    1) read in raw CPS files
    2) do collapsing
    3) return pandas DF with raw CPS data (just variables needed - age, hours, weight)
    4) return np array "weighted"

    5) moments() will take CPS and calc moments
    6) VCV will boot strap CPS and call moments() with each boostrapped sample
'''

    # Create variables for number of age groups in data (S_labor) and number
    # of percentiles (J_labor)
    S_labor = 60
    J_labor = 99

    labor_file = utils.read_file(cur_path,
                                 "data/labor/cps_hours_by_age_hourspct.txt")
    data = pd.read_table(labor_file, header=0)

    piv = data.pivot(index='age', columns='hours_pct', values='mean_hrs')
    lab_mat_basic = np.array(piv)
    lab_mat_basic /= np.nanmax(lab_mat_basic)

    piv2 = data.pivot(index='age', columns='hours_pct', values='num_obs')
    weights = np.array(piv2)
    weights /= np.nansum(weights, axis=1).reshape(S_labor, 1)
    weighted = np.nansum((lab_mat_basic * weights), axis=1)


'''
------------------------------------------------------------------------
    Compute moments from labor data
------------------------------------------------------------------------
'''
def compute_labor_moments(cps):
    '''
    ------------------------------------------------------------------------
    Inputs:
        cps         = pandas DF, raw data from SCF
    Objects created in the function:
        labor_dist_data = [S,] array of labor moments

    Returns:
        labor_dist_data
    ------------------------------------------------------------------------
    '''
    # Fit a line to the last few years of the average labor participation which extends from
    # ages 76 to 100.
    slope = (weighted[56] - weighted[49]) / (56 - 49)
    intercept = weighted[56] - slope * 56
    extension = slope * (np.linspace(56, 80, 23)) + intercept
    to_dot = slope * (np.linspace(45, 56, 11)) + intercept

    labor_dist_data = np.zeros(80)
    labor_dist_data[:57] = weighted[:57]
    labor_dist_data[57:] = extension
    
    return labor_dist_data


def VCV_moments(cps, n, bin_weights, J):
    '''
    ------------------------------------------------------------------------
        Compute Variance-Covariance matrix for labor moments by 
        bootstrapping data

        Inputs:
            data        = pandas DF, raw data from CPS
            n           = interger, number of bootstrap iterations to run
            bin_weights = ability weights (Jx1 array)
            J           = number of ability groups (scalar)
        Objects created in the function:
            labor_moments_boot = [n,S] array, bootstrapped labor moments
            boot = pandas DF, boostrapped dataframe
            VCV  = [S,S] array, variance-covariance matrix of labor moments
        Output:
            VCV

    ------------------------------------------------------------------------
    '''
    labor_moments_boot = np.zeros((n,J+2))
    for i in range(n):
        boot = scf[np.random.randint(2, size=len(scf.index)).astype(bool)]
        labor_moments_boot[i,:] = compute_wealth_moments(boot, bin_weights, J)

    VCV = np.cov(wealth_moments_boot.T)

    return VCV

def labor_data_graphs(weighted, output_dir):
    '''
    ------------------------------------------------------------------------
    Plot graphs
    ------------------------------------------------------------------------
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    domain = np.linspace(20, 80, S_labor)
    Jgrid = np.linspace(1, 100, J_labor)
    X, Y = np.meshgrid(domain, Jgrid)
    cmap2 = matplotlib.cm.get_cmap('summer')

    plt.plot(domain, weighted, color='black', label='Data')
    plt.plot(np.linspace(76, 100, 23), extension, color='black',
             linestyle='-.', label='Extrapolation')
    plt.plot(np.linspace(65, 76, 11), to_dot,
             linestyle='--', color='black')
    plt.axvline(x=76, color='black', linestyle='--')
    plt.xlabel(r'age-$s$')
    plt.ylabel(r'individual labor supply $/bar{l}_s$')
    plt.legend()
    plt.savefig(os.path.join(
        baseline_dir, 'Demographics/labor_dist_data_withfit.png'))

    fig10 = plt.figure()
    ax10 = fig10.gca(projection='3d')
    ax10.plot_surface(X, Y, lab_mat_basic.T,
                      rstride=1, cstride=2, cmap=cmap2)
    ax10.set_xlabel(r'age-$s$')
    ax10.set_ylabel(r'ability type -$j$')
    ax10.set_zlabel(r'labor $e_j(s)$')
    plt.savefig(os.path.join(baseline_dir, 'Demographics/data_labor_dist'))



