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

cur_path = os.path.split(os.path.abspath(__file__))[0]
LABOR_DIR = os.path.join(cur_path, "data", "labor")

'''
------------------------------------------------------------------------
    Read in data, convert to numpy array, and weight
------------------------------------------------------------------------
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
    Fit a line to extrapolate ages 80-100
------------------------------------------------------------------------
'''


def labor_data_moments(flag_graphs, output_dir="./OUTPUT"):
    '''
    Pickles the labor particpation rates for the data (Sx1 array) and generates graphs,
    depending on if flag_graphs is True or False (True means it graphs).
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
    if flag_graphs:
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
            output_dir, 'Demographics/labor_dist_data_withfit.png'))

        fig10 = plt.figure()
        ax10 = fig10.gca(projection='3d')
        ax10.plot_surface(X, Y, lab_mat_basic.T,
                          rstride=1, cstride=2, cmap=cmap2)
        ax10.set_xlabel(r'age-$s$')
        ax10.set_ylabel(r'ability type -$j$')
        ax10.set_zlabel(r'labor $e_j(s)$')
        plt.savefig(os.path.join(output_dir, 'Demographics/data_labor_dist'))

    var_names = ['labor_dist_data']
    dictionary = {}
    for key in var_names:
        dictionary[key] = locals()[key]
    pickle.dump(dictionary, open(os.path.join(
        output_dir, "Saved_moments/labor_data_moments.pkl"), "wb"))
