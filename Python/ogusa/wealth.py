'''
------------------------------------------------------------------------
Last updated 7/17/2015

Returns the wealth for all ages of a certain percentile.

This py-file calls the following other file(s):
            data/wealth/scf2007to2013_wealth_age_all_percentiles.csv
            utils.py

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/Demographics/distribution_of_wealth_data.png
            OUTPUT/Demographics/distribution_of_wealth_data_log.png
            OUTPUT/Saved_moments/wealth_data_moments.pkl
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import utils
import os
from scipy import stats
import cPickle as pickle

cur_path = os.path.split(os.path.abspath(__file__))[0]
WEALTH_DIR = os.path.join(cur_path, "data", "wealth")

'''
------------------------------------------------------------------------
    Import Data
------------------------------------------------------------------------
'''

# read in SCF data collapsed by age and percentile for graphs
wealth_file = utils.read_file(
    cur_path, "data/wealth/scf2007to2013_wealth_age_all_percentiles.csv")
data = pd.read_table(wealth_file, sep=',', header=0)

# read in raw SCF data to calculate moments
fileDir = os.path.dirname(os.path.realpath('__file__'))
year_list = [2013, 2010, 2007]
scf_dict = {}
for year in year_list:
    filename = os.path.join(fileDir, '../Data/Survey_of_Consumer_Finances/rscfp'+str(year)+'.dta')
    filename = os.path.abspath(os.path.realpath(filename))
    print 'filename = ', filename
    scf_dict[str(year)] = pd.read_stata(filename)

scf = scf_dict['2013'].append(scf_dict['2010'].append(scf_dict['2007'],ignore_index=True),ignore_index=True)



'''
------------------------------------------------------------------------
    Graph Data
------------------------------------------------------------------------
'''


def wealth_data_graphs(output_dir):
    '''
    Graphs wealth distribution and its log
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    to_graph = np.array(data)[:, 1:-1]

    domain = np.linspace(18, 95, 78)
    Jgrid = np.linspace(1, 99, 99)
    X, Y = np.meshgrid(domain, Jgrid)
    cmap2 = matplotlib.cm.get_cmap('summer')
    fig10 = plt.figure()
    ax10 = fig10.gca(projection='3d')
    ax10.plot_surface(X, Y, (to_graph).T, rstride=1, cstride=2, cmap=cmap2)
    ax10.set_xlabel(r'age-$s$')
    ax10.set_ylabel(r'percentile')
    ax10.set_zlabel(r'wealth')
    plt.savefig(os.path.join(
        outputdir, '/Demographics/distribution_of_wealth_data'))

    fig10 = plt.figure()
    ax10 = fig10.gca(projection='3d')
    ax10.plot_surface(X, Y, np.log(to_graph).T,
                      rstride=1, cstride=2, cmap=cmap2)
    ax10.set_xlabel(r'age-$s$')
    ax10.set_ylabel(r'percentile')
    ax10.set_zlabel(r'log of wealth')
    plt.savefig(os.path.join(
        outputdir, '/Demographics/distribution_of_wealth_data_log'))

'''
------------------------------------------------------------------------
    Get wealth moments
------------------------------------------------------------------------
'''
# Restrict the data: it has other columns that give weights and indexes the age

def get_wealth_data(bin_weights, J, flag_graphs):
    '''
    Inputs:
        bin_weights = ability weights (Jx1 array)
        J = number of ability groups (scalar)
        flag_graphs = whether or not to graph distribution (bool)
        output_dir = path to the starting data
    Output:
        Saves a pickle of the desired wealth percentiles.  Graphs those levels.
    '''
    if flag_graphs:
        wealth_data_graphs(output_dir)


    # calculate percentile shares (percentiles based on lambdas input)
    scf.sort(columns='networth', ascending=True, inplace=True)
    scf['weight_networth'] = scf['wgt']*scf['networth']
    total_weight_wealth = scf.weight_networth.sum()
    cumsum = scf.wgt.cumsum()
    pct_wealth = np.zeros((bin_weights.shape[0],))
    top_pct_wealth = np.zeros((bin_weights.shape[0],))
    wealth = np.zeros((bin_weights.shape[0],))
    cum_weights = bin_weights.cumsum()
    for i in range(bin_weights.shape[0]):
        cutoff = scf.wgt.sum() / (1./cum_weights[i])
        pct_wealth[i] = scf.networth[cumsum >= cutoff].iloc[0]
        top_pct_wealth[i] = 1 - ((scf.weight_networth[cumsum < cutoff].sum())/total_weight_wealth)
        wealth[i] = ((scf.weight_networth[cumsum < cutoff].sum())/total_weight_wealth)


    wealth_share = np.zeros((bin_weights.shape[0],))
    wealth_share[0] = wealth[0]
    wealth_share[1:] = wealth[1:]-wealth[0:-1]

    # print 'Weighted percentiles: ', pct_wealth
    # print 'Weighted shares: ', top_pct_wealth
    # print 'Wealth by group: ', wealth_share
    # print cum_weights

    cutoff = scf.wgt.sum() / (1./.1)
    pct_10 = scf.networth[cumsum >= cutoff].iloc[0]
    cutoff = scf.wgt.sum() / (1./.9)
    pct_90 = scf.networth[cumsum >= cutoff].iloc[0]
    top_10_share = 1 - ((scf.weight_networth[cumsum < cutoff].sum())/total_weight_wealth)
    ratio_90_10 = pct_90/pct_10 
    cutoff = scf.wgt.sum() / (1./.99)
    pct_99 = scf.networth[cumsum >= cutoff].iloc[0]
    top_1_share = 1 - ((scf.weight_networth[cumsum < cutoff].sum())/total_weight_wealth)


    print ratio_90_10, top_10_share, top_1_share
    quit()



    perc_array = np.zeros(J)
    # convert bin_weights to integers to index the array of data moments
    bins2 = (bin_weights * 100).astype(int)
    perc_array = np.cumsum(bins2)
    perc_array -= 1
    wealth_data_array = np.zeros((78, J))
    wealth_data_array[:, 0] = data2[:, :perc_array[0]].mean(axis=1)
    # pull out the data moments for each percentile
    for j in xrange(1, J):
        wealth_data_array[:, j] = data2[
            :, perc_array[j - 1]:perc_array[j]].mean(axis=1)

    # Look at the percent difference between the fits for the first age group (20-44) and second age group (45-65)
    #   The wealth_data_array moment indices are shifted because they start at age 18
    # The :: indices is so that we look at the 2 moments for the lowest group,
    # the 2 moments for the second lowest group, etc in order
    wealth_moments = np.zeros(2 * J)
    wealth_moments[0::2] = np.mean(wealth_data_array[2:26],axis=0)
    wealth_moments[1::2] = np.mean(wealth_data_array[26:47],axis=0)

    return wealth_moments
