'''
------------------------------------------------------------------------
This script reads in data generated from the Tax Calculator model.
It then estimates tax functions tau_{s,t}(x,y), where
tau_{s,t} is the effective tax rate, marginal tax rate on labor income,
or the marginal tax rate on capital income, for a given age (s) in a
particular year (t). x is total labor income, and y is total capital
income.
------------------------------------------------------------------------
'''
# Import packages
import time
import os
import numpy as np
import scipy.optimize as opt
from dask import delayed, compute
import dask.multiprocessing
import pickle
import ogcore.parameter_plots as pp
from ogcore.constants import DEFAULT_START_YEAR, SHOW_RUNTIME
import warnings


if not SHOW_RUNTIME:
    warnings.simplefilter("ignore", RuntimeWarning)

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
MIN_OBS = 240  # 240 is 8 parameters to estimate X 30 obs per parameter
MAX_ETR = 0.65
MAX_MTR = 0.99
MIN_INCOME = 5
MIN_INC_GRAPH = 5
MAX_INC_GRAPH = 500000

'''
------------------------------------------------------------------------
Define Functions
------------------------------------------------------------------------
'''


def get_tax_rates(params, X, Y, wgts, tax_func_type, rate_type,
                  analytical_mtrs=False, mtr_capital=False,
                  for_estimation=True):
    '''
    Generates tax rates given income data and the parameters of the tax
    functions.

    Args:
        params (tuple): parameters of the tax function, varies by
            tax_func_type
        X (array_like): labor income data
        Y (array_like): capital income data
        wgts (array_like): weights for data observations
        tax_func_type (str): functional form of tax functions
        rate_type (str): type of tax rate: mtrx, mtry, etr
        analytical_mtrs (bool): whether to compute marginal tax rates
            from the total tax function (for DEP functions only)
        mtr_capital (bool): whether analytical mtr on capital income
        for_estimation (bool): whether the results are used in
            estimation, if True, then tax rates are computed as
            deviations from the mean

    Returns:
        txrates (array_like): model tax rates for each observation

    '''
    X2 = X ** 2
    Y2 = Y ** 2
    income = X + Y
    if tax_func_type == 'GS':
        phi0, phi1, phi2 = (
            np.squeeze(params[..., 0]), np.squeeze(params[..., 1]),
            np.squeeze(params[..., 2]))
        if rate_type == 'etr':
            txrates = (
                (phi0 * (income - ((income ** -phi1) + phi2) **
                         (-1 / phi1))) / income)
        else:  # marginal tax rate function
            txrates = (
                phi0 * (1 - (income ** (-phi1 - 1) *
                        ((income ** -phi1) + phi2) ** ((-1 - phi1) /
                                                       phi1))))
    elif tax_func_type == 'DEP':
        (A, B, C, D, max_x, max_y, share, min_x, min_y, shift_x,
         shift_y, shift) = (
             np.squeeze(params[..., 0]), np.squeeze(params[..., 1]),
             np.squeeze(params[..., 2]), np.squeeze(params[..., 3]),
             np.squeeze(params[..., 4]), np.squeeze(params[..., 5]),
             np.squeeze(params[..., 6]), np.squeeze(params[..., 7]),
             np.squeeze(params[..., 8]), np.squeeze(params[..., 9]),
             np.squeeze(params[..., 10]), np.squeeze(params[..., 11]))
        Etil = A + B
        Ftil = C + D
        if for_estimation:
            shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
            shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
            X2bar = (X2 * wgts).sum() / wgts.sum()
            Xbar = (X * wgts).sum() / wgts.sum()
            Y2bar = (Y2 * wgts).sum() / wgts.sum()
            Ybar = (Y * wgts).sum() / wgts.sum()
            X2til = (X2 - X2bar) / X2bar
            Xtil = (X - Xbar) / Xbar
            Y2til = (Y2 - Y2bar) / Y2bar
            Ytil = (Y - Ybar) / Ybar
            tau_x = (((max_x - min_x) * (A * X2til + B * Xtil + Etil) /
                      (A * X2til + B * Xtil + Etil + 1)) + min_x)
            tau_y = (((max_y - min_y) * (C * Y2til + D * Ytil + Ftil) /
                      (C * Y2til + D * Ytil + Ftil + 1)) + min_y)
            txrates = (((tau_x + shift_x) ** share) *
                       ((tau_y + shift_y) ** (1 - share))) + shift
        else:
            if analytical_mtrs:
                tau_x = ((max_x - min_x) * (A * X2 + B * X) /
                         (A * X2 + B * X + 1) + min_x)
                tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
                         (C * Y2 + D * Y + 1) + min_y)
                etr = (((tau_x + shift_x) ** share) *
                       ((tau_y + shift_y) ** (1 - share))) + shift
                if mtr_capital:
                    d_etr = (
                        (1-share) * ((tau_y + shift_y) ** (-share)) *
                        (max_y - min_y) * ((2 * C * Y + D) /
                                           ((C * Y2 + D * Y + 1) ** 2))
                        * ((tau_x + shift_x) ** share))
                    txrates = d_etr * income + etr
                else:
                    d_etr = (
                        share * ((tau_x + shift_x) ** (share - 1)) *
                        (max_x - min_x) * ((2 * A * X + B) /
                                           ((A * X2 + B * X + 1) ** 2))
                        * ((tau_y + shift_y) ** (1 - share)))
                    txrates = d_etr * income + etr
            else:
                tau_x = (((max_x - min_x) * (A * X2 + B * X) /
                          (A * X2 + B * X + 1)) + min_x)
                tau_y = (((max_y - min_y) * (C * Y2 + D * Y) /
                          (C * Y2 + D * Y + 1)) + min_y)
                txrates = (((tau_x + shift_x) ** share) *
                           ((tau_y + shift_y) ** (1 - share))) + shift
    elif tax_func_type == 'DEP_totalinc':
        A, B, max_income, min_income, shift_income, shift = (
             np.squeeze(params[..., 0]), np.squeeze(params[..., 1]),
             np.squeeze(params[..., 2]), np.squeeze(params[..., 3]),
             np.squeeze(params[..., 4]), np.squeeze(params[..., 5]))
        Etil = A + B
        income2 = income ** 2
        if for_estimation:
            shift_income = (np.maximum(-min_income, 0.0) + 0.01 *
                            (max_income - min_income))
            income2bar = (income2 * wgts).sum() / wgts.sum()
            Ibar = (income * wgts).sum() / wgts.sum()
            income2til = (income2 - income2bar) / income2bar
            Itil = (income - Ibar) / Ibar
            tau_income = (((max_income - min_income) *
                           (A * income2til + B * Itil + Etil) /
                           (A * income2til + B * Itil + Etil + 1)) +
                          min_income)
            txrates = tau_income + shift_income + shift
        else:
            if analytical_mtrs:
                d_etr = ((max_income - min_income) *
                         ((2 * A * income + B) /
                          ((A * income2 + B * income + 1) ** 2)))
                etr = (((max_income - min_income) *
                       ((A * income2 + B * income) /
                        (A * income2 + B * income + 1)) + min_income) +
                       shift_income + shift)
                txrates = (d_etr * income) + (etr)
            else:
                tau_income = (((max_income - min_income) *
                               (A * income2 + B * income) /
                               (A * income2 + B * income + 1)) + min_income)
                txrates = tau_income + shift_income + shift
    elif tax_func_type == 'linear':
        rate = np.squeeze(params[..., 0])
        txrates = rate * np.ones_like(income)

    return txrates


def wsumsq(params, *args):
    '''
    This function generates the weighted sum of squared deviations of
    predicted values of tax rates (ETR, MTRx, or MTRy) from the tax
    rates from the data for the Cobb-Douglas functional form of the tax
    function.

    Args:
        params (tuple): tax function parameter values
        args (tuple): contains (fixed_tax_func_params, X, Y, txrates,
            wgts, tax_func_type, rate_type)
        fixed_tax_func_params (tuple): value of parameters of tax
            functions that are not estimated
        X (array_like): labor income data
        Y (array_like): capital income data
        txrates (array_like): tax rates data
        wgts (array_like): weights for data observations
        tax_func_type (str): functional form of tax functions
        rate_type (str): type of tax rate: mtrx, mtry, etr

    Returns:
        wssqdev (scalar): weighted sum of squared deviations, >0

    '''
    (fixed_tax_func_params, X, Y, txrates, wgts, tax_func_type,
     rate_type) = args
    params_all = np.append(params, fixed_tax_func_params)
    txrates_est = get_tax_rates(
        params_all, X, Y, wgts, tax_func_type, rate_type)
    errors = txrates_est - txrates
    wssqdev = (wgts * (errors ** 2)).sum()

    return wssqdev


def find_outliers(sse_mat, age_vec, se_mult, start_year, varstr,
                  graph=False):
    '''
    This function takes a matrix of sum of squared errors (SSE) from
    tax function estimations for each age (s) in each year of the budget
    window (t) and marks estimations that have outlier SSE.

    Args:
        sse_mat (Numpy array): SSE for each estimated tax function,
            size is SxBW
        age_vec (numpy array): vector of ages, length S
        se_mult (scalar): multiple of standard deviations before
            consider estimate an outlier
        start_year (int): first year of budget window
        varstr (str): name of tax function being evaluated
        graph (bool): whether to output graphs

    Returns:
        sse_big_mat (Numpy array): indicators of whether tax function
            is outlier, size is SxBW

    '''
    # Mark outliers from estimated MTRx functions
    thresh = (sse_mat[sse_mat > 0].mean() +
              se_mult * sse_mat[sse_mat > 0].std())
    sse_big_mat = sse_mat > thresh
    print(varstr, ": ", str(sse_big_mat.sum()),
          " observations tagged as outliers.")
    output_dir = os.path.join(CUR_PATH, 'OUTPUT', 'TaxFunctions')
    if graph:
        pp.txfunc_sse_plot(age_vec, sse_mat, start_year, varstr,
                           output_dir, 0)
    if sse_big_mat.sum() > 0:
        # Mark the outliers from the first sweep above. Then mark the
        # new outliers in a second sweep
        sse_mat_new = sse_mat.copy()
        sse_mat_new[sse_big_mat] = np.nan
        thresh2 = (sse_mat_new[sse_mat_new > 0].mean() + se_mult *
                   sse_mat_new[sse_mat_new > 0].std())
        sse_big_mat += sse_mat_new > thresh2
        print(varstr, ": ", "After second round, ",
              str(sse_big_mat.sum()),
              " observations tagged as outliers (cumulative).")
        if graph:
            pp.txfunc_sse_plot(age_vec, sse_mat_new, start_year, varstr,
                               output_dir, 1)
        if (sse_mat_new > thresh2).sum() > 0:
            # Mark the outliers from the second sweep above
            sse_mat_new2 = sse_mat_new.copy()
            sse_mat_new2[sse_big_mat] = np.nan
            if graph:
                pp.txfunc_sse_plot(age_vec, sse_mat_new2, start_year,
                                   varstr, output_dir, 2)

    return sse_big_mat


def replace_outliers(param_arr, sse_big_mat):
    '''
    This function replaces outlier estimated tax function parameters
    with linearly interpolated tax function tax function parameters

    Args:
        param_arr (Numpy array): estimated tax function parameters,
            size is SxBWx#tax params
        sse_big_mat (Numpy array): indicators of whether tax function
            is outlier, size is SxBW

    Returns:
        param_arr_adj (Numpy array): estimated and interpolated tax
            function parameters, size SxBWx#tax params

    '''
    numparams = param_arr.shape[2]
    age_ind = np.arange(0, sse_big_mat.shape[0])
    param_arr_adj = param_arr.copy()
    for t in range(sse_big_mat.shape[1]):
        big_cnt = 0
        for s in age_ind:
            # Smooth out ETR tax function outliers
            if sse_big_mat[s, t] and s < sse_big_mat.shape[0] - 1:
                # For all outlier observations, increase the big_cnt by
                # 1 and set the param_arr_adj equal to nan
                big_cnt += 1
                param_arr_adj[s, t, :] = np.nan
            if not sse_big_mat[s, t] and big_cnt > 0 and s == big_cnt:
                # When the current function is not an outlier but the last
                # one was and this string of outliers is at the beginning
                # ages, set the outliers equal to this period's tax function
                reshaped = param_arr_adj[s, t, :].reshape(
                    (1, 1, numparams))
                param_arr_adj[:big_cnt, t, :] = np.tile(
                    reshaped, (big_cnt, 1))
                big_cnt = 0

            if not sse_big_mat[s, t] and big_cnt > 0 and s > big_cnt:
                # When the current function is not an outlier but the last
                # one was and this string of outliers is in the interior of
                # ages, set the outliers equal to a linear interpolation
                # between the two bounding non-outlier functions
                diff = (param_arr_adj[s, t, :] -
                        param_arr_adj[s - big_cnt - 1, t, :])
                slopevec = diff / (big_cnt + 1)
                slopevec = slopevec.reshape(1, numparams)
                tiled_slopevec = np.tile(slopevec, (big_cnt, 1))

                interceptvec = \
                    param_arr_adj[s - big_cnt - 1, t, :].reshape(
                        1, numparams)
                tiled_intvec = np.tile(interceptvec, (big_cnt, 1))

                reshaped_arange = np.arange(1, big_cnt+1).reshape(
                    big_cnt, 1)
                tiled_reshape_arange = np.tile(reshaped_arange,
                                               (1, numparams))

                param_arr_adj[s-big_cnt:s, t, :] = (
                    tiled_intvec + tiled_slopevec * tiled_reshape_arange
                )

                big_cnt = 0
            if sse_big_mat[s, t] and s == sse_big_mat.shape[0] - 1:
                # When the last ages are outliers, set the parameters equal
                # to the most recent non-outlier tax function
                big_cnt += 1
                param_arr_adj[s, t, :] = np.nan
                reshaped = param_arr_adj[s - big_cnt, t, :].reshape(
                    1, 1, numparams)
                param_arr_adj[s - big_cnt + 1:, t, :] = np.tile(
                    reshaped, (big_cnt, 1))

    return param_arr_adj


def txfunc_est(df, s, t, rate_type, tax_func_type, numparams,
               output_dir, graph):
    '''
    This function uses tax tax rate and income data for individuals of a
    particular age (s) and a particular year (t) to estimate the
    parameters of a Cobb-Douglas aggregation function of two ratios of
    polynomials in labor income and capital income, respectively.

    Args:
        df (Pandas DataFrame): 11 variables with N observations of tax
            rates
        s (int): age of individual, >= 21
        t (int): year of analysis, >= 2016
        rate_type (str): type of tax rate: mtrx, mtry, etr
        tax_func_type (str): functional form of tax functions
        numparams (int): number of parameters in the tax functions
        output_dir (str): output directory for saving plot files
        graph (bool): whether to plot the estimated functions compared
            to the data

    Returns:
        (tuple): tax function estimation output:

            * params (Numpy array): vector of estimated parameters
            * wsse (scalar): weighted sum of squared deviations from
                minimization
            * obs (int): number of observations in the data, > 600

    '''
    X = df['total_labinc']
    Y = df['total_capinc']
    wgts = df['weight']
    X2 = X ** 2
    Y2 = Y ** 2
    X2bar = (X2 * wgts).sum() / wgts.sum()
    Xbar = (X * wgts).sum() / wgts.sum()
    Y2bar = (Y2 * wgts).sum() / wgts.sum()
    Ybar = (Y * wgts).sum() / wgts.sum()
    income = X + Y
    income2 = income ** 2
    Ibar = (income * wgts).sum() / wgts.sum()
    income2bar = (income2 * wgts).sum() / wgts.sum()
    if rate_type == 'etr':
        txrates = df['etr']
    elif rate_type == 'mtrx':
        txrates = df['mtr_labinc']
    elif rate_type == 'mtry':
        txrates = df['mtr_capinc']
    x_10pctl = df['total_labinc'].quantile(0.1)
    y_10pctl = df['total_capinc'].quantile(0.1)
    x_20pctl = df['total_labinc'].quantile(.2)
    y_20pctl = df['total_capinc'].quantile(.2)
    min_x = txrates[(df['total_capinc'] < y_10pctl)].min()
    min_y = txrates[(df['total_labinc'] < x_10pctl)].min()

    if tax_func_type == 'DEP':
        # '''
        # Estimate DeBacker, Evans, Phillips (2018) ratio of polynomial
        # tax functions.
        # '''
        Atil_init = 1.0
        Btil_init = 1.0
        Ctil_init = 1.0
        Dtil_init = 1.0
        max_x_init = np.minimum(
            txrates[(df['total_capinc'] < y_20pctl)].max(),
            MAX_ETR + 0.05)
        max_y_init = np.minimum(
            txrates[(df['total_labinc'] < x_20pctl)].max(),
            MAX_ETR + 0.05)
        shift = txrates[(df['total_labinc'] < x_20pctl) |
                        (df['total_capinc'] < y_20pctl)].min()
        share_init = 0.5
        params_init = np.array([Atil_init, Btil_init, Ctil_init,
                                Dtil_init, max_x_init, max_y_init,
                                share_init])
        shift_x = 0.0  # temp value
        shift_y = 0.0  # temp value
        tx_objs = (np.array([min_x, min_y, shift_x, shift_y, shift]), X,
                   Y, txrates, wgts, tax_func_type, rate_type)
        lb_max_x = np.maximum(min_x, 0.0) + 1e-4
        lb_max_y = np.maximum(min_y, 0.0) + 1e-4
        bnds = ((1e-12, None), (1e-12, None), (1e-12, None),
                (1e-12, None), (lb_max_x, MAX_ETR + 0.15),
                (lb_max_y, MAX_ETR + 0.15), (0, 1))
        params_til = opt.minimize(wsumsq, params_init, args=(tx_objs),
                                  method="L-BFGS-B", bounds=bnds,
                                  tol=1e-15)
        Atil, Btil, Ctil, Dtil, max_x, max_y, share = params_til.x
        # message = ("(max_x, min_x)=(" + str(max_x) + ", " + str(min_x) +
        #     "), (max_y, min_y)=(" + str(max_y) + ", " + str(min_y) + ")")
        # print(message)
        wsse = params_til.fun
        obs = df.shape[0]
        shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
        shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
        params = np.zeros(numparams)
        params[:4] = (np.array([Atil, Btil, Ctil, Dtil]) /
                      np.array([X2bar, Xbar, Y2bar, Ybar]))
        params[4:] = np.array([max_x, max_y, share, min_x, min_y,
                              shift_x, shift_y, shift])
        params_to_plot = params
    elif tax_func_type == 'DEP_totalinc':
        # '''
        # Estimate DeBacker, Evans, Phillips (2018) ratio of polynomial
        # tax functions as a function of total income.
        # '''
        Atil_init = 1.0
        Btil_init = 1.0
        max_x_init = np.minimum(
            txrates[(df['total_capinc'] < y_20pctl)].max(),
            MAX_ETR + 0.05)
        max_y_init = np.minimum(
            txrates[(df['total_labinc'] < x_20pctl)].max(),
            MAX_ETR + 0.05)
        max_income_init = max(max_x_init, max_y_init)
        min_income = min(min_x, min_y)
        shift = txrates[(df['total_labinc'] < x_20pctl) |
                        (df['total_capinc'] < y_20pctl)].min()
        share_init = 0.5
        shift_inc = 0.0  # temp value
        params_init = np.array([Atil_init, Btil_init, max_income_init])
        tx_objs = (np.array([min_income, shift_inc, shift]), X, Y,
                   txrates, wgts, tax_func_type, rate_type)
        lb_max_income = np.maximum(min_income, 0.0) + 1e-4
        bnds = ((1e-12, None), (1e-12, None), (lb_max_income,
                                               MAX_ETR + 0.15))
        params_til = opt.minimize(wsumsq, params_init, args=(tx_objs),
                                  method="L-BFGS-B", bounds=bnds,
                                  tol=1e-15)
        Atil, Btil, max_income = params_til.x
        wsse = params_til.fun
        obs = df.shape[0]
        shift_income = (np.maximum(-min_income, 0.0) + 0.01 *
                        (max_income - min_income))
        params = np.zeros(numparams)
        params[:2] = (np.array([Atil, Btil]) /
                      np.array([income2bar, Ibar]))
        params[2:] = np.array([max_income, min_income, shift_income,
                              shift])
        params_to_plot = params
    elif tax_func_type == "GS":
        # '''
        # Estimate Gouveia-Strauss parameters via least squares.
        # Need to use a different functional form than for DEP function.
        # '''
        phi0_init = 1.0
        phi1_init = 1.0
        phi2_init = 1.0
        params_init = np.array([phi0_init, phi1_init, phi2_init])
        tx_objs = (np.array([None]), X, Y, txrates, wgts, tax_func_type,
                   rate_type)
        bnds = ((1e-12, None), (1e-12, None), (1e-12, None))
        params_til = opt.minimize(wsumsq, params_init, args=(tx_objs),
                                  method="L-BFGS-B", bounds=bnds,
                                  tol=1e-15)
        phi0til, phi1til, phi2til = params_til.x
        wsse = params_til.fun
        obs = df.shape[0]
        params = np.zeros(numparams)
        params[:3] = np.array([phi0til, phi1til, phi2til])
        params_to_plot = params
    elif tax_func_type == "linear":
        # '''
        # For linear rates, just take the mean ETR or MTR by age-year.
        # Can use DEP form and set all parameters except for the shift
        # parameter to zero.
        # '''
        params = np.zeros(numparams)
        wsse = 0.0
        obs = df.shape[0]
        params[0] = (
            (txrates * wgts * income).sum() / (income * wgts).sum())
        params_to_plot = params
    else:
        raise RuntimeError("Choice of tax function is not in the set of"
                           + " possible tax functions.  Please select"
                           + " from: DEP, DEP_totalinc, GS, linear.")
    if graph:
        pp.txfunc_graph(s, t, df, X, Y, txrates, rate_type,
                        tax_func_type, get_tax_rates, params_to_plot,
                        output_dir)

    # Garbage collection
    del df, txrates

    return params, wsse, obs


def tax_data_sample(data, max_etr=MAX_ETR, min_income=MIN_INCOME,
                    max_mtr=MAX_MTR):
    '''
    Function to create sample tax data for estimation by dropping
    observations with extreme values.

    Args:
        data (DataFrame): raw data from microsimulation model

    Returns:
        data (DataFrame): selected sample

    '''
    # drop all obs with ETR > MAX_ETR
    data.drop(data[data['etr'] > MAX_ETR].index, inplace=True)
    # drop all obs with ETR < MIN_ETR
    # set min ETR to value at 10th percentile in distribution of ETRs
    min_etr = data['etr'].quantile(q=0.10)
    data.drop(data[data['etr'] < min_etr].index, inplace=True)
    # drop all obs with total market income, labor income, or
    # capital income < MIN_INCOME
    data.drop(data[(data['market_income'] < MIN_INCOME) |
                   (data['total_labinc'] < MIN_INCOME) |
                   (data['total_capinc'] < MIN_INCOME)].index,
              inplace=True)
    # drop all obs with MTR on capital income > MAX_MTR
    data.drop(data[data['mtr_capinc'] > MAX_MTR].index,
              inplace=True)
    # drop all obs with MTR on capital income < min_cap_mtr
    # set min MTR to value at 10th percentile in distribution of MTRs
    min_cap_mtr = data['mtr_capinc'].quantile(q=0.10)
    data.drop(data[data['mtr_capinc'] < min_cap_mtr].index,
              inplace=True)
    # drop all obs with MTR on labor income > MAX_MTR
    data.drop(data[data['mtr_labinc'] > MAX_MTR].index, inplace=True)
    # drop all obs with MTR on labor income < min_lab_mtr
    min_lab_mtr = data['mtr_labinc'].quantile(q=0.10)
    data.drop(data[data['mtr_labinc'] < min_lab_mtr].index, inplace=True)

    return data


def tax_func_loop(t, data, start_year, s_min, s_max, age_specific,
                  tax_func_type, analytical_mtrs, desc_data, graph_data,
                  graph_est, output_dir, numparams):
    '''
    Estimates tax functions for a particular year.  Looped over.

    Args:
        t (int): year of tax data to estimated tax functions for
        data (Pandas DataFrame): tax return data for year t
        start_yr (int): first year of budget window
        s_min (int): minimum age to estimate tax functions for
        s_max (int): maximum age to estimate tax functions for
        age_specific (bool): whether to estimate age specific tax
            functions
        tax_func_type (str): functional form of tax functions
        analytical_mtrs (bool): whether to use the analytical derivation
            of the marginal tax rates (and thus only need to estimate
            the effective tax rate functions)
        desc_data (bool): whether to print descriptive statistics
        graph_data (bool): whether to plot data
        graph_est (bool): whether to plot estimated coefficients
        output_dir (str): path to save output to
        numparams (int): number of parameters in tax functions

    Returns:
        (tuple): tax function estimation output:

            * TotPop_yr (int): total population derived from micro data
            * Pct_age (Numpy array): fraction of observations that are
                in each age bin
            * AvgInc (scalar): mean income in the data
            * AvgETR (scalar): mean effective tax rate in data
            * AvgMTRx (scalar): mean marginal tax rate on labor income
                in data
            * AvgMTRy (scalar): mean marginal tax rate on capital income
                in data
            * frac_tax_payroll (scalar): fraction of total tax revenue
                the comes from payroll taxes
            * etrparam_arr (Numpy array): parameters of the effective
                tax rate functions
            * etr_wsumsq_arr (Numpy array): weighted sum of squares from
                estimation of the effective tax rate functions
            * etr_obs_arr (Numpy array): weighted sum of squares from
                estimation of the effective tax rate functions
            * mtrxparam_arr (Numpy array): parameters of the marginal
                tax rate on labor income functions
            * mtrx_wsumsq_arr (Numpy array): weighted sum of squares
                from estimation of the marginal tax rate on labor income
                functions
            * mtrx_obs_arr (Numpy array): weighted sum of squares from
                estimation of the marginal tax rate on labor income
                functions
            * mtryparam_arr (Numpy array): parameters of the marginal
                tax rate on capital income functions
            * mtry_wsumsq_arr (Numpy array): weighted sum of squares
                from estimation of the marginal tax rate on capital
                income functions
            * mtry_obs_arr (Numpy array): weighted sum of squares from
                estimation of the marginal tax rate on capital income
                functions

    '''
    # initialize arrays for output
    etrparam_arr = np.zeros((s_max - s_min + 1, numparams))
    mtrxparam_arr = np.zeros((s_max - s_min + 1, numparams))
    mtryparam_arr = np.zeros((s_max - s_min + 1, numparams))
    etr_wsumsq_arr = np.zeros(s_max - s_min + 1)
    etr_obs_arr = np.zeros(s_max - s_min + 1)
    mtrx_wsumsq_arr = np.zeros(s_max - s_min + 1)
    mtrx_obs_arr = np.zeros(s_max - s_min + 1)
    mtry_wsumsq_arr = np.zeros(s_max - s_min + 1)
    mtry_obs_arr = np.zeros(s_max - s_min + 1)
    PopPct_age = np.zeros(s_max - s_min + 1)

    # Calculate average total income in each year
    AvgInc = (
        ((data['market_income'] * data['weight']).sum()) /
        data['weight'].sum())

    # Calculate average ETR and MTRs (weight by population weights
    #    and income) for each year
    AvgETR = (
        ((data['etr']*data['market_income'] * data['weight']).sum()) /
        (data['market_income'] * data['weight']).sum())

    AvgMTRx = (
        ((data['mtr_labinc'] * data['market_income'] *
          data['weight']).sum()) / (data['market_income'] *
                                    data['weight']).sum())

    AvgMTRy = (
        ((data['mtr_capinc'] * data['market_income'] *
          data['weight']).sum()) / (data['market_income'] *
                                    data['weight']).sum())

    # Caulcatoe fraction of total tax liability that is from payroll
    # taxes
    frac_tax_payroll = (
        (data['payroll_tax_liab'] * data['weight']).sum() /
        (data['total_tax_liab'] * data['weight']).sum())

    # Calculate total population in each year
    TotPop_yr = data['weight'].sum()

    # Clean up the data by dropping outliers
    data = tax_data_sample(data)

    # Create an array of the different ages in the data
    min_age = int(np.maximum(data['age'].min(), s_min))
    max_age = int(np.minimum(data['age'].max(), s_max))
    if age_specific:
        ages_list = np.arange(min_age, max_age + 1)
    else:
        ages_list = np.arange(0, 1)

    NoData_cnt = np.min(min_age - s_min, 0)

    # Each age s must be done in serial
    for s in ages_list:
        if age_specific:
            print("Year=", t, "Age=", s)
            df = data[data['age'] == s]
            PopPct_age[s-min_age] = \
                df['weight'].sum() / TotPop_yr

        else:
            print("year=", t, "age= all ages")
            df = data
            PopPct_age[0] = \
                df['weight'].sum() / TotPop_yr
        df_etr = df.loc[df[
            (np.isfinite(df['etr'])) &
            (np.isfinite(df['total_labinc'])) &
            (np.isfinite(df['total_capinc'])) &
            (np.isfinite(df['weight']))].index,
                        ['mtr_labinc', 'mtr_capinc',
                         'total_labinc', 'total_capinc',
                         'etr', 'weight']].copy()
        df_mtrx = df.loc[df[
            (np.isfinite(df['mtr_labinc'])) &
            (np.isfinite(df['total_labinc'])) &
            (np.isfinite(df['total_capinc'])) &
            (np.isfinite(df['weight']))].index,
                         ['mtr_labinc', 'total_labinc',
                          'total_capinc', 'weight']].copy()
        df_mtry = df.loc[df[
            (np.isfinite(df['mtr_capinc'])) &
            (np.isfinite(df['total_labinc'])) &
            (np.isfinite(df['total_capinc'])) &
            (np.isfinite(df['weight']))].index,
                         ['mtr_capinc', 'total_labinc',
                          'total_capinc', 'weight']].copy()
        df_minobs = np.min([df_etr.shape[0], df_mtrx.shape[0],
                            df_mtry.shape[0]])
        del df

        if df_minobs < MIN_OBS and s < max_age:
            # '''
            # --------------------------------------------------------
            # Don't estimate function on this iteration if obs < 500.
            # Will fill in later with interpolated values
            # --------------------------------------------------------
            # '''
            message = ("Insuff. sample size for age " + str(s) +
                       " in year " + str(t))
            print(message)
            NoData_cnt += 1
            etrparam_arr[s-s_min, :] = np.nan
            mtrxparam_arr[s-s_min, :] = np.nan
            mtryparam_arr[s-s_min, :] = np.nan

        elif df_minobs < MIN_OBS and s == max_age:
            # '''
            # --------------------------------------------------------
            # If last period does not have sufficient data, fill in
            # final missing age data with last positive year
            # --------------------------------------------------------
            # lastp_etr  = (numparams,) vector, vector of parameter
            #              estimates from previous age with sufficient
            #              observations
            # lastp_mtrx = (numparams,) vector, vector of parameter
            #              estimates from previous age with sufficient
            #              observations
            # lastp_mtry = (numparams,) vector, vector of parameter
            #              estimates from previous age with sufficient
            #              observations
            # --------------------------------------------------------
            # '''
            message = ("Max age (s=" + str(s) + ") insuff. data in"
                       + " year " + str(t) +
                       ". Fill in final ages with " +
                       "insuff. data with most recent successful " +
                       "estimate.")
            print(message)
            NoData_cnt += 1
            lastp_etr = etrparam_arr[s - NoData_cnt - s_min, :]
            etrparam_arr[s-NoData_cnt - s_min + 1:, :] = np.tile(
                lastp_etr.reshape((1, numparams)),
                (NoData_cnt + s_max - max_age, 1))
            lastp_mtrx = mtrxparam_arr[s - NoData_cnt - s_min, :]
            mtrxparam_arr[s - NoData_cnt - s_min + 1:, :] = np.tile(
                lastp_mtrx.reshape((1, numparams)),
                (NoData_cnt + s_max - max_age, 1))
            lastp_mtry = mtryparam_arr[s - NoData_cnt - s_min, :]
            mtryparam_arr[s - NoData_cnt - s_min + 1:, :] = np.tile(
                lastp_mtry.reshape((1, numparams)),
                (NoData_cnt + s_max - max_age, 1))

        else:
            # Estimate parameters for age with sufficient data
            if desc_data:
                # print some desciptive stats
                message = ("Descriptive ETR statistics for age=" +
                           str(s) + " in year " + str(t))
                print(message)
                print(df_etr.describe())
                message = ("Descriptive MTRx statistics for age=" +
                           str(s) + " in year " + str(t))
                print(message)
                print(df_mtrx.describe())
                message = ("Descriptive MTRy statistics for age=" +
                           str(s) + " in year " + str(t))
                print(message)
                print(df_mtry.describe())

            if graph_data:
                pp.gen_3Dscatters_hist(df_etr, s, t, output_dir)

            # Estimate effective tax rate function ETR(x,y)
            (etrparams, etr_wsumsq_arr[s - s_min],
             etr_obs_arr[s - s_min]) = txfunc_est(
                    df_etr, s, t, 'etr', tax_func_type, numparams,
                    output_dir, graph_est)
            etrparam_arr[s - s_min, :] = etrparams
            del df_etr

            # Estimate marginal tax rate of labor income function
            # MTRx(x,y)
            (mtrxparams, mtrx_wsumsq_arr[s - s_min],
                mtrx_obs_arr[s - s_min]) = txfunc_est(
                    df_mtrx, s, t, 'mtrx', tax_func_type, numparams,
                    output_dir, graph_est)
            mtrxparam_arr[s - s_min, :] = mtrxparams
            del df_mtrx
            # Estimate marginal tax rate of capital income function
            # MTRy(x,y)
            (mtryparams, mtry_wsumsq_arr[s - s_min],
             mtry_obs_arr[s-s_min]) = txfunc_est(
                 df_mtry, s, t, 'mtry', tax_func_type, numparams,
                 output_dir, graph_est)
            mtryparam_arr[s - s_min, :] = mtryparams

            del df_mtry

            if NoData_cnt > 0 & NoData_cnt == s - s_min:
                # '''
                # ----------------------------------------------------
                # Fill in initial blanks with first positive data
                # estimates. This includes the case in which
                # min_age > s_min
                # ----------------------------------------------------
                # '''
                message = "Fill in all previous blank ages"
                print(message)
                etrparam_arr[:s - s_min, :] = np.tile(
                    etrparams.reshape((1, numparams)), (s - s_min, 1))
                mtrxparam_arr[:s - s_min, :] = np.tile(
                    mtrxparams.reshape((1, numparams)), (s - s_min, 1))
                mtryparam_arr[:s - s_min, :, :] = np.tile(
                    mtryparams.reshape((1, numparams)), (s - s_min, 1))

            elif NoData_cnt > 0 & NoData_cnt < s - s_min:
                # '''
                # ----------------------------------------------------
                # Fill in interior data gaps with linear interpolation
                # between bracketing positive data ages. In all of
                # these cases min_age < s <= max_age.
                # ----------------------------------------------------
                # tvals        = (NoData_cnt+2,) vector, linearly
                #                space points between 0 and 1
                # x0_etr       = (NoData_cnt x 10) matrix, positive
                #                estimates at beginning of no data
                #                spell
                # x1_etr       = (NoData_cnt x 10) matrix, positive
                #                estimates at end (current period) of
                #                no data spell
                # lin_int_etr  = (NoData_cnt x 10) matrix, linearly
                #                interpolated etr parameters between
                #                x0_etr and x1_etr
                # x0_mtrx      = (NoData_cnt x 10) matrix, positive
                #                estimates at beginning of no data
                #                spell
                # x1_mtrx      = (NoData_cnt x 10) matrix, positive
                #                estimates at end (current period) of
                #                no data spell
                # lin_int_mtrx = (NoData_cnt x 10) matrix, linearly
                #                interpolated mtrx parameters between
                #                x0_mtrx and x1_mtrx
                # ----------------------------------------------------
                # '''
                message = ("Linearly interpolate previous blank " +
                           "tax functions")
                print(message)
                tvals = np.linspace(0, 1, NoData_cnt + 2)
                x0_etr = np.tile(
                    etrparam_arr[s - NoData_cnt - s_min - 1,
                                 :].reshape((1, numparams)),
                    (NoData_cnt, 1))
                x1_etr = np.tile(etrparams.reshape((1, numparams)),
                                 (NoData_cnt, 1))
                lin_int_etr = (
                    x0_etr + tvals[1:-1].reshape((NoData_cnt, 1)) *
                    (x1_etr - x0_etr))
                etrparam_arr[s - NoData_cnt - min_age:s - min_age, :] =\
                    lin_int_etr
                x0_mtrx = np.tile(
                    mtrxparam_arr[s-NoData_cnt-s_min-1,
                                  :].reshape((1, numparams)),
                    (NoData_cnt, 1))
                x1_mtrx = np.tile(
                    mtrxparams.reshape((1, numparams)), (NoData_cnt, 1))
                lin_int_mtrx = (
                    x0_mtrx + tvals[1:-1].reshape((NoData_cnt, 1)) *
                    (x1_mtrx - x0_mtrx))
                mtrxparam_arr[s - NoData_cnt - min_age:s - min_age,
                              :] = lin_int_mtrx
                x0_mtry = np.tile(
                    mtryparam_arr[s - NoData_cnt - s_min - 1,
                                  :].reshape((1, numparams)),
                    (NoData_cnt, 1))
                x1_mtry = np.tile(
                    mtryparams.reshape((1, numparams)), (NoData_cnt, 1))
                lin_int_mtry = (x0_mtry + tvals[1:-1].reshape((
                    NoData_cnt, 1)) * (x1_mtry - x0_mtry))
                mtryparam_arr[s - NoData_cnt - min_age:s - min_age,
                              :] = lin_int_mtry

            NoData_cnt == 0

            if s == max_age and max_age < s_max:
                # '''
                # ----------------------------------------------------
                # If the last age estimates, and max_age< s_max, fill
                # in the remaining ages with these last estimates
                # ----------------------------------------------------
                # '''
                message = "Fill in all old tax functions."
                print(message)
                etrparam_arr[s - s_min + 1:, :] = np.tile(
                    etrparams.reshape((1, numparams)),
                    (s_max - max_age, 1))
                mtrxparam_arr[s - s_min + 1:, :] = np.tile(
                    mtrxparams.reshape((1, numparams)),
                    (s_max - max_age, 1))
                mtryparam_arr[s - s_min + 1:, :] = np.tile(
                    mtryparams.reshape((1, numparams)),
                    (s_max - max_age, 1))

    return (TotPop_yr, PopPct_age, AvgInc, AvgETR, AvgMTRx, AvgMTRy,
            frac_tax_payroll,
            etrparam_arr, etr_wsumsq_arr, etr_obs_arr,
            mtrxparam_arr, mtrx_wsumsq_arr, mtrx_obs_arr,
            mtryparam_arr, mtry_wsumsq_arr, mtry_obs_arr)


def tax_func_estimate(micro_data, BW, S, starting_age, ending_age,
                      start_year=DEFAULT_START_YEAR, baseline=True,
                      analytical_mtrs=False, tax_func_type='DEP',
                      age_specific=False, reform={}, data=None,
                      desc_data=False, graph_data=False,
                      graph_est=False, client=None, num_workers=1,
                      tax_func_path=None):
    '''
    This function performs analysis on the source data from Tax-
    Calculator and estimates functions for the effective tax rate (ETR),
    marginal tax rate on labor income (MTRx), and marginal tax rate on
    capital income (MTRy).

    Args:
        micro_data (dict): Dictionary of DataFrames with micro data
        BW (int): number of years in the budget window (the period
            over which tax policy is assumed to vary)
        S (int): number of model periods a model agent is economically
            active for
        starting_age (int): minimum age to estimate tax functions for
        ending_age (int): maximum age to estimate tax functions for
        start_yr (int): first year of budget window
        baseline (bool): whether these are the baseline tax functions
        analytical_mtrs (bool): whether to use the analytical derivation
            of the marginal tax rates (and thus only need to estimate
            the effective tax rate functions)
        tax_func_type (str): functional form of tax functions
        age_specific (bool): whether to estimate age specific tax
            functions
        reform (dict): policy reform dictionary for Tax-Calculator
        data (str or Pandas DataFrame): path to or data to use in
            Tax-Calculator
        client (Dask client object): client
        num_workers (int): number of workers to use for parallelization
            with Dask
        tax_func_path (str): path to save pickle with estimated tax
            function parameters to

    Returns:
        dict_param (dict): dictionary with tax function parameters

    '''
    s_min = starting_age + 1
    s_max = ending_age
    start_year = int(start_year)
    end_yr = int(start_year + BW - 1)
    print('BW = ', BW, "begin year = ", start_year,
          "end year = ", end_yr)
    tax_func_type_num_params_dict = {
        'DEP': 12, 'DEP_totalinc': 6, 'GS': 3, 'linear': 1}
    numparams = int(tax_func_type_num_params_dict[tax_func_type])
    years_list = np.arange(start_year, end_yr + 1)
    if age_specific:
        ages_list = np.arange(s_min, s_max+1)
    else:
        ages_list = np.arange(0, 1)
    # initialize arrays for output
    etrparam_arr = np.zeros((s_max - s_min + 1, BW, numparams))
    mtrxparam_arr = np.zeros((s_max - s_min + 1, BW, numparams))
    mtryparam_arr = np.zeros((s_max - s_min + 1, BW, numparams))
    etr_wsumsq_arr = np.zeros((s_max - s_min + 1, BW))
    etr_obs_arr = np.zeros((s_max - s_min + 1, BW))
    mtrx_wsumsq_arr = np.zeros((s_max - s_min + 1, BW))
    mtrx_obs_arr = np.zeros((s_max - s_min + 1, BW))
    mtry_wsumsq_arr = np.zeros((s_max - s_min + 1, BW))
    mtry_obs_arr = np.zeros((s_max - s_min + 1, BW))
    AvgInc = np.zeros(BW)
    AvgETR = np.zeros(BW)
    AvgMTRx = np.zeros(BW)
    AvgMTRy = np.zeros(BW)
    frac_tax_payroll = np.zeros(BW)
    TotPop_yr = np.zeros(BW)
    PopPct_age = np.zeros((s_max-s_min+1, BW))

    # '''
    # --------------------------------------------------------------------
    # Solve for tax functions for each year (t) and each age (s)
    # --------------------------------------------------------------------
    # start_time = scalar, current processor time in seconds (float)
    # output_dir = string, directory to which plots will be saved
    # micro_data = dictionary, BW (one for each year) DataFrames,
    #              each of which has variables with observations from
    #              Tax-Calculator
    # t          = integer >= start_year, index for year of analysis
    # --------------------------------------------------------------------
    # '''
    start_time = time.time()
    if not tax_func_path:
        output_dir = os.path.join(CUR_PATH, 'OUTPUT', 'TaxFunctions')
    else:
        output_dir = os.path.join(os.path.dirname(tax_func_path), 'OUTPUT',
                                  'TaxFunctions')
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    lazy_values = []
    for t in years_list:
        lazy_values.append(
            delayed(tax_func_loop)(
                t, micro_data[str(t)], start_year, s_min, s_max,
                age_specific, tax_func_type, analytical_mtrs, desc_data,
                graph_data, graph_est, output_dir, numparams))
    if client:
        futures = client.compute(lazy_values, num_workers=num_workers)
        results = client.gather(futures)
    else:
        results = results = compute(
            *lazy_values, scheduler=dask.multiprocessing.get,
            num_workers=num_workers)

    # Garbage collection
    del micro_data

    # for i, result in results.items():
    for i, result in enumerate(results):
        (TotPop_yr[i], PopPct_age[:, i], AvgInc[i],
         AvgETR[i], AvgMTRx[i], AvgMTRy[i], frac_tax_payroll[i],
         etrparam_arr[:, i, :], etr_wsumsq_arr[:, i],
         etr_obs_arr[:, i], mtrxparam_arr[:, i, :],
         mtrx_wsumsq_arr[:, i], mtrx_obs_arr[:, i],
         mtryparam_arr[:, i, :], mtry_wsumsq_arr[:, i],
         mtry_obs_arr[:, i]) = result

    message = ("Finished tax function loop through " +
               str(len(years_list)) + " years and " +
               str(len(ages_list)) + " ages per year.")
    print(message)
    elapsed_time = time.time() - start_time

    # Print tax function computation time
    if elapsed_time < 60:  # less than a minute
        secs = round(elapsed_time, 3)
        message = "Tax function estimation time: " + str(secs) + " sec"
        print(message)
    elif elapsed_time >= 60 and elapsed_time < 3600:  # less than hour
        mins = int(elapsed_time / 60)
        secs = round(((elapsed_time / 60) - mins) * 60, 1)
        message = ("Tax function estimation time: " + str(mins) +
                   " min, " + str(secs) + " sec")
        print(message)
    elif elapsed_time >= 3600 and elapsed_time < 86400:  # less than day
        hours = int(elapsed_time / (60 * 60))
        mins = int((elapsed_time - (hours * 60 * 60)) / 60)
        secs = round(elapsed_time - (hours * 60 * 60) - (mins * 60), 1)
        message = ("Tax function estimation time: " + str(hours) +
                   " hour(s), " + str(mins) + " min(s), " + str(secs) +
                   " sec(s)")
        print(message)

    # '''
    # --------------------------------------------------------------------
    # Replace outlier tax functions (SSE>mean+2.5*std) with linear
    # linear interpolation. We make two passes (filtering runs).
    # --------------------------------------------------------------------
    # '''
    if age_specific:
        age_sup = np.linspace(s_min, s_max, s_max-s_min+1)
        se_mult = 3.5
        etr_sse_big = find_outliers(etr_wsumsq_arr / etr_obs_arr,
                                    age_sup, se_mult, start_year, "ETR",
                                    graph=graph_est)
        if etr_sse_big.sum() > 0:
            etrparam_arr_adj = replace_outliers(etrparam_arr,
                                                etr_sse_big)
        elif etr_sse_big.sum() == 0:
            etrparam_arr_adj = etrparam_arr

        mtrx_sse_big = find_outliers(mtrx_wsumsq_arr / mtrx_obs_arr,
                                     age_sup, se_mult, start_year,
                                     "MTRx", graph=graph_est)
        if mtrx_sse_big.sum() > 0:
            mtrxparam_arr_adj = replace_outliers(mtrxparam_arr,
                                                 mtrx_sse_big)
        elif mtrx_sse_big.sum() == 0:
            mtrxparam_arr_adj = mtrxparam_arr

        mtry_sse_big = find_outliers(mtry_wsumsq_arr / mtry_obs_arr,
                                     age_sup, se_mult, start_year,
                                     "MTRy", graph=graph_est)
        if mtry_sse_big.sum() > 0:
            mtryparam_arr_adj = replace_outliers(mtryparam_arr,
                                                 mtry_sse_big)
        elif mtry_sse_big.sum() == 0:
            mtryparam_arr_adj = mtryparam_arr

    # '''
    # --------------------------------------------------------------------
    # Generate tax function parameters for S < s_max - s_min + 1
    # --------------------------------------------------------------------
    # etrparam_arr_S  = S x BW x 10 array, this is an array in which S
    #                   is less-than-or-equal-to s_max-s_min+1. We use
    #                   weighted averages of parameters in relevant age
    #                   groups
    # mtrxparam_arr_S = S x BW x 10 array, this is an array in which S
    #                   is less-than-or-equal-to s_max-s_min+1. We use
    #                   weighted averages of parameters in relevant age
    #                   groups
    # age_cuts     = (S+1,) vector, linspace of age cutoffs of S+1 points
    #                between 0 and S+1
    # yrcut_lb     = integer >= 0, index of lower bound age for S bin
    # yrcut_ub     = integer >= 0, index of upper bound age for S bin
    # rmndr_pct_lb = scalar in [0,1], discounted weight on lower bound age
    # rmndr_pct_ub = scalar in [0,1], discounted weight on upper bound age
    # age_wgts     = ages x BW x 10 array, age weights for each age in
    #                each year copied back 10 times in the 3rd dimension
    # --------------------------------------------------------------------
    # '''
    if age_specific:
        if S == s_max - s_min + 1:
            etrparam_arr_S = etrparam_arr_adj
            mtrxparam_arr_S = mtrxparam_arr_adj
            mtryparam_arr_S = mtryparam_arr_adj
        elif S < s_max - s_min + 1:
            etrparam_arr_S = etrparam_arr_adj
            mtrxparam_arr_S = mtrxparam_arr_adj
            mtryparam_arr_S = mtryparam_arr_adj
            etrparam_arr_S = np.zeros((S, BW, numparams))
            mtrxparam_arr_S = np.zeros((S, BW, numparams))
            mtryparam_arr_S = np.zeros((S, BW, numparams))
            age_cuts = np.linspace(0, s_max - s_min + 1, S + 1)
            yrcut_lb = int(age_cuts[0])
            rmndr_pct_lb = 1.
            for s in np.arange(S):
                yrcut_ub = int(np.floor(age_cuts[s + 1]))
                rmndr_pct_ub = (age_cuts[s + 1] -
                                np.floor(age_cuts[s + 1]))
                if rmndr_pct_ub == 0.:
                    rmndr_pct_ub = 1.
                    yrcut_ub -= 1
                age_wgts = np.dstack(
                    [PopPct_age[yrcut_lb:yrcut_ub + 1, :]] * numparams)
                age_wgts[0, :, :] *= rmndr_pct_lb
                age_wgts[yrcut_ub-yrcut_lb, :, :] *= rmndr_pct_ub
                etrparam_arr_S[s, :, :] = (
                    etrparam_arr_adj[yrcut_lb:yrcut_ub + 1, :, :] *
                    age_wgts).sum(axis=0)
                mtrxparam_arr_S[s, :, :] = (
                    mtrxparam_arr_adj[yrcut_lb:yrcut_ub + 1, :, :] *
                    age_wgts).sum(axis=0)
                mtryparam_arr_S[s, :, :] = (
                    mtryparam_arr_adj[yrcut_lb:yrcut_ub + 1, :, :] *
                    age_wgts).sum(axis=0)
                yrcut_lb = yrcut_ub
                rmndr_pct_lb = 1 - rmndr_pct_ub
        else:
            print('S is larger than the difference between the minimum'
                  + ' age and the maximum age specified.  Please choose'
                  + ' and S such that a model period equals at least'
                  + ' one calendar year.')

        print('Big S: ', S)
        print('max age, min age: ', s_max, s_min)
    else:
        etrparam_arr_S = np.tile(np.reshape(
            etrparam_arr[0 - s_min, :, :],
            (1, BW, etrparam_arr.shape[2])), (S, 1, 1))
        mtrxparam_arr_S = np.tile(np.reshape(
            mtrxparam_arr[0-s_min, :, :],
            (1, BW, mtrxparam_arr.shape[2])), (S, 1, 1))
        mtryparam_arr_S = np.tile(np.reshape(
            mtryparam_arr[0-s_min, :, :],
            (1, BW, mtryparam_arr.shape[2])), (S, 1, 1))

    # Save tax function parameters array and computation time in
    # dictionary
    dict_params = dict(
        [('tfunc_etr_params_S', etrparam_arr_S),
         ('tfunc_mtrx_params_S', mtrxparam_arr_S),
         ('tfunc_mtry_params_S', mtryparam_arr_S),
         ('tfunc_avginc', AvgInc), ('tfunc_avg_etr', AvgETR),
         ('tfunc_avg_mtrx', AvgMTRx), ('tfunc_avg_mtry', AvgMTRy),
         ('tfunc_frac_tax_payroll', frac_tax_payroll),
         ('tfunc_etr_sumsq', etr_wsumsq_arr),
         ('tfunc_mtrx_sumsq', mtrx_wsumsq_arr),
         ('tfunc_mtry_sumsq', mtry_wsumsq_arr),
         ('tfunc_etr_obs', etr_obs_arr),
         ('tfunc_mtrx_obs', mtrx_obs_arr),
         ('tfunc_mtry_obs', mtry_obs_arr), ('tfunc_time', elapsed_time),
         ('tax_func_type', tax_func_type),
         ('start_year', start_year), ('BW', BW)])

    if tax_func_path:
        with open(tax_func_path, "wb") as f:
            pickle.dump(dict_params, f)

    return dict_params
