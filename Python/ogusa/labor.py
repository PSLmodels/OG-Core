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
import scipy.ndimage.filters as filter


'''
------------------------------------------------------------------------
    Import Data
------------------------------------------------------------------------
'''
def get_labor_data():

    # read in "raw" CPS data to calculate moments
    # these data were actually cleaned in hours_data_cps_setup.do
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, '../Data/Current_Population_Survey/cps_est_ability_hours_1992to2013.dta')
    filename = os.path.abspath(os.path.realpath(filename))
    cps = pd.read_stata(filename, columns=['year', 'age', 'hours',
                                           'hours_unit', 'wtsupp'])
    return cps

#     '''
#     Need to:
#     1) read in raw CPS files
#     2) do collapsing
#     3) return pandas DF with raw CPS data (just variables needed - age, hours, weight)
#     4) return np array "weighted"
#
#     5) moments() will take CPS and calc moments
#     6) VCV will boot strap CPS and call moments() with each boostrapped sample
# '''
#
#     # Create variables for number of age groups in data (S_labor) and number
#     # of percentiles (J_labor)
#     S_labor = 60
#     J_labor = 99
#
#     labor_file = utils.read_file(cur_path,
#                                  "data/labor/cps_hours_by_age_hourspct.txt")
#     data = pd.read_table(labor_file, header=0)
#
#     piv = data.pivot(index='age', columns='hours_pct', values='mean_hrs')
#     lab_mat_basic = np.array(piv)
#     lab_mat_basic /= np.nanmax(lab_mat_basic)
#
#     piv2 = data.pivot(index='age', columns='hours_pct', values='num_obs')
#     weights = np.array(piv2)
#     weights /= np.nansum(weights, axis=1).reshape(S_labor, 1)
#     weighted = np.nansum((lab_mat_basic * weights), axis=1)


'''
------------------------------------------------------------------------
    Compute moments from labor data
------------------------------------------------------------------------
'''
def compute_labor_moments(cps, S):
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

    # Find fraction of total time people work on average by age
    cps['hours_wgt'] = cps['hours']*cps['wtsupp']
    columns = ['hours_wgt', 'wgt', 'avg_hours']
    by_age = pd.DataFrame(columns=columns)
    # by_age = by_age.fillna(0) # with 0s rather than NaNs

    by_age['hours_wgt'] = cps.groupby(['age'])['hours_wgt'].sum()
    by_age['wgt'] = cps.groupby(['age'])['wtsupp'].sum()
    by_age['avg_hours'] = by_age['hours_wgt']/by_age['wgt']

    # get fraction of time endowment worked (assume time
    # endowment is 24 hours minus required time to sleep)
    by_age['frac_work'] = by_age['avg_hours']/(365*16.)

    # Data have sufficient obs through age  80
    # Fit a line to the last few years of the average labor participation which extends from
    # ages 76 to 100.
    slope = (by_age['frac_work'][-1] - by_age['frac_work'][-15]) / (15.)
    # intercept = by_age['frac_work'][-1] - slope*len(by_age['frac_work'])
    # extension = slope * (np.linspace(56, 80, 23)) + intercept
    # to_dot = slope * (np.linspace(45, 56, 11)) + intercept

    labor_dist_data = np.zeros(80)
    labor_dist_data[:60] = by_age['frac_work']
    labor_dist_data[60:] = by_age['frac_work'][-1] + slope*xrange(20)

    # the above computes moments if the model period is a year
    # the following adjusts those moments in case it is smaller
    labor_dist_out = filter.uniform_filter(labor_dist_data,size=int(80/S))[::int(80/S)]

    return labor_dist_out


def VCV_moments(cps, n, bin_weights, S):
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
    labor_moments_boot = np.zeros((n,S))
    for i in range(n):
        boot = cps[np.random.randint(2, size=len(cps.index)).astype(bool)]
        labor_moments_boot[i,:] = compute_labor_moments(boot, S)

    VCV = np.cov(labor_moments_boot.T)

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
