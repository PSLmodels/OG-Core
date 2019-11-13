import os
import numpy as np
import pandas as pd
from ogusa import utils

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]


def get_wealth_data(scf_yrs_list=[2016, 2013, 2010, 2007], web=True,
                    directory=None):
    '''
    Reads wealth data from the 2007, 2010, 2013, and 2016 Survey of
    Consumer Finances (SCF) files.

    Args:
        scf_yrs_list (list): list of SCF years to import. Currently the
            largest set of years that will work is
            [2016, 2013, 2010, 2007]
        web (Boolean): =True if function retrieves data from internet
        directory (string or None): local directory location if data are
            stored on local drive, not use internet (web=False)


    Returns:
        df_scf (Pandas DataFrame): pooled cross-sectional data from SCFs

    '''
    # Hard code cpi list for given years (2015=100)
    cpi_dict = {'cpi2016': 101.2615832057050,
                'cpi2013': 98.2870778608004,
                'cpi2010': 91.9999409325069,
                'cpi2007': 87.4799768230408}
    if web:
        # Throw an error if the machine is not connected to the internet
        if utils.not_connected():
            err_msg = ('SCF DATA ERROR: The local machine is not ' +
                       'connected to the internet and web=True was ' +
                       'selected.')
            raise RuntimeError(err_msg)

        file_urls = []
        for yr in scf_yrs_list:
            zipfilename = ('https://www.federalreserve.gov/econres/' +
                           'files/scfp' + str(yr) + 's.zip')
            file_urls.append(zipfilename)

        file_paths = utils.fetch_files_from_web(file_urls)

    if not web and directory is None:
        # Thow an error if web=False no source of files is given
        err_msg = ('SCF DATA ERROR: No local directory was ' +
                   'specified as the source for the data.')
        raise ValueError(err_msg)

    elif not web and directory is not None:
        file_paths = []
        full_directory = os.path.expanduser(directory)
        filename_list = []
        for yr in scf_yrs_list:
            filename = 'rscfp' + str(yr) + '.dta'
            filename_list.append(filename)

        for name in filename_list:
            file_paths.append(os.path.join(full_directory, name))
        # Check to make sure the necessary files are present in the
        # local directory
        err_msg = ('hrs_by_age() ERROR: The file %s was not found in ' +
                   'the directory %s')
        for path in file_paths:
            if not os.path.isfile(path):
                raise ValueError(err_msg % (path, full_directory))

    # read in raw SCF data to calculate moments
    scf_dict = {}
    for filename, year in zip(file_paths, scf_yrs_list):
        df_yr = pd.read_stata(filename, columns=['networth', 'wgt'])
        # Add inflation adjusted net worth
        cpi = cpi_dict['cpi' + str(year)]
        df_yr['networth_infadj'] = df_yr['networth'] * cpi
        scf_dict[str(year)] = df_yr

    df_scf = scf_dict[str(scf_yrs_list[0])]
    num_yrs = len(scf_yrs_list)
    if num_yrs >= 2:
        for year in scf_yrs_list[1:]:
            df_scf = df_scf.append(scf_dict[str(year)],
                                   ignore_index=True)

    return df_scf


def compute_wealth_moments(scf, bin_weights):
    '''
    This function computes moments from the distribution of wealth
    using SCF data.

    Args:
        scf (Pandas DataFrame): pooled cross-sectional data from SCFs
        bin_weights (Numpy Array) = ability weights
        J (int) = number of ability groups

    Returns:
        wealth_moments (Numpy Array): array of wealth moments

    '''
    # calculate percentile shares (percentiles based on lambdas input)
    scf.sort_values(by='networth_infadj', ascending=True, inplace=True)
    scf['weight_networth'] = scf['wgt'] * scf['networth_infadj']
    total_weight_wealth = scf.weight_networth.sum()
    cumsum = scf.wgt.cumsum()
    J = bin_weights.shape[0]
    pct_wealth = np.zeros(J)
    top_pct_wealth = np.zeros(J)
    wealth = np.zeros((J,))
    cum_weights = bin_weights.cumsum()
    for i in range(J):
        cutoff = scf.wgt.sum() / (1. / cum_weights[i])
        pct_wealth[i] = scf.networth[cumsum >= cutoff].iloc[0]
        top_pct_wealth[i] = 1 - (
            (scf.weight_networth[cumsum < cutoff].sum()) /
            total_weight_wealth)
        wealth[i] = ((
            scf.weight_networth[cumsum < cutoff].sum()) /
            total_weight_wealth)

    wealth_share = np.zeros(J)
    wealth_share[0] = wealth[0]
    wealth_share[1:] = wealth[1:] - wealth[0:-1]

    # compute gini coeff
    scf.sort_values(by='networth_infadj', ascending=True, inplace=True)
    p = (scf.wgt.cumsum() / scf.wgt.sum()).values
    nu = ((scf.wgt * scf.networth_infadj).cumsum()).values
    nu = nu / nu[-1]
    gini_coeff = (nu[1:] * p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

    # compute variance in logs
    df = scf.drop(scf[scf['networth_infadj'] <= 0.0].index)
    df['ln_networth'] = np.log(df['networth_infadj'])
    df.sort_values(by='ln_networth', ascending=True, inplace=True)
    weight_mean = ((df.ln_networth * df.wgt).sum()) / (df.wgt.sum())
    var_ln_wealth = (((
        df.wgt * ((df.ln_networth - weight_mean) ** 2)).sum()) *
        (1. / (df.wgt.sum() - 1)))

    wealth_moments = np.append(
        [wealth_share], [gini_coeff, var_ln_wealth])

    return wealth_moments
