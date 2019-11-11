import numpy as np
import pandas as pd
from ogusa import utils

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]


def get_wealth_data(scf_yrs_list=[2013, 2010, 2007], web=True,
                    directory=None):
    '''
    Reads wealth data from the 2007, 2010, and 2013 Survey of Consumer
    Finances (SCF) files.

    Args:
        scf_yrs_list (list of 4-digit integers): list of SCF years to
            import
        web (Boolean): =True if function retrieves data from internet
        directory (string or None): local directory location if data are
            stored on local drive, not use internet (web=False)


    Returns:
        scf (Pandas DataFrame): pooled cross-sectional data from SCFs

    '''
    if web:
        # Throw an error if the machine is not connected to the internet
        if utils.not_connected():
            err_msg = ('SCF DATA ERROR: The local machine is not ' +
                       'connected to the internet and web=True was ' +
                       'selected.')
            raise RuntimeError(err_msg)

        file_urls = file_names_for_range(beg_yr, beg_mth, end_yr, end_mth, web)

        file_paths = fetch_files_from_web(file_urls)

    if not web and directory is None:
        # Thow an error if web=False no source of files is given
        err_msg = ('SCF DATA ERROR: No local directory was ' +
                   'specified as the source for the data.')
        raise ValueError(err_msg)

    # elif not web and directory is not None:
    #     full_directory = os.path.expanduser(directory)
    #     file_list = file_names_for_range(beg_yr, beg_mth, end_yr, end_mth, web)

    #     for name in file_list:
    #         file_paths.append(os.path.join(full_directory, name))
    #     # Check to make sure the necessary files are present in the
    #     # local directory
    #     err_msg = ('hrs_by_age() ERROR: The file %s was not found in ' +
    #                'the directory %s')
    #     for path in file_paths:
    #         if not os.path.isfile(path):
    #             raise RuntimeError(err_msg % (path, full_directory))

    # read in raw SCF data to calculate moments
    scf_dir = os.path.join(CUR_PATH, '..', 'Data',
                          'Survey_of_Consumer_Finances')
    year_list = [2013, 2010, 2007]
    scf_dict = {}
    for year in year_list:
        filename = os.path.join(scf_dir, 'rscfp' + str(year) + '.dta')
        scf_dict[str(year)] = pd.read_stata(
            filename, columns=['networth', 'wgt'])

    scf = scf_dict['2013'].append(
        scf_dict['2010'].append(
            scf_dict['2007'], ignore_index=True), ignore_index=True)

    return scf


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
    scf.sort_values(by='networth', ascending=True, inplace=True)
    scf['weight_networth'] = scf['wgt']*scf['networth']
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
    wealth_share[1:] = wealth[1:]-wealth[0:-1]

    # compute gini coeff
    scf.sort_values(by='networth', ascending=True, inplace=True)
    p = (scf.wgt.cumsum() / scf.wgt.sum()).values
    nu = ((scf.wgt * scf.networth).cumsum()).values
    nu = nu / nu[-1]
    gini_coeff = (nu[1:] * p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

    # compute variance in logs
    df = scf.drop(scf[scf['networth'] <= 0.0].index)
    df['ln_networth'] = np.log(df['networth'])
    df.sort_values(by='ln_networth', ascending=True, inplace=True)
    weight_mean = ((df.ln_networth * df.wgt).sum()) / (df.wgt.sum())
    var_ln_wealth = (((
        df.wgt * ((df.ln_networth - weight_mean) ** 2)).sum()) *
                     (1. / (df.wgt.sum() - 1)))

    wealth_moments = np.append(
        [wealth_share], [gini_coeff, var_ln_wealth])

    return wealth_moments
