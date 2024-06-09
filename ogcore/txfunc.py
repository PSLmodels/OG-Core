"""
------------------------------------------------------------------------
This script reads in data generated from a tax-benefit microsimulation model.
It then estimates tax functions tau_{s,t}(x,y), where
tau_{s,t} is the effective tax rate, marginal tax rate on labor income,
or the marginal tax rate on capital income, for a given age (s) in a
particular year (t). x is total labor income, and y is total capital
income.
------------------------------------------------------------------------
"""

# Import packages
import time
import os
import numpy as np
import scipy.optimize as opt
from dask import delayed, compute
import dask.multiprocessing
import pickle
import cloudpickle
from scipy.interpolate import interp1d as intp
import matplotlib.pyplot as plt
import ogcore.parameter_plots as pp
from ogcore.constants import DEFAULT_START_YEAR, SHOW_RUNTIME
from ogcore import utils
import warnings
from pygam import LinearGAM, s, te
from matplotlib import cm
import random


if not SHOW_RUNTIME:
    warnings.simplefilter("ignore", RuntimeWarning)

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
MIN_OBS = 240  # 240 is 8 parameters to estimate X 30 obs per parameter
MAX_ETR = 0.65
MAX_MTR = 0.99
MIN_INCOME = 5
MIN_INC_GRAPH = 5
MAX_INC_GRAPH = 500000

"""
------------------------------------------------------------------------
Define Functions
------------------------------------------------------------------------
"""


def get_tax_rates(
    params,
    X,
    Y,
    wgts,
    tax_func_type,
    rate_type,
    analytical_mtrs=False,
    mtr_capital=False,
    for_estimation=True,
):
    """
    Generates tax rates given income data and the parameters of the tax
    functions.

    Args:
        params (list): list of parameters of the tax function, or
            nonparametric function for tax function type "mono"
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

    """
    X2 = X**2
    Y2 = Y**2
    income = X + Y
    if tax_func_type != "mono":
        params = np.array(
            params
        )  # easier to use arrays for calculations below, except when can't (bc lists of functions)
    if tax_func_type == "GS":
        phi0, phi1, phi2 = (
            np.squeeze(params[..., 0]),
            np.squeeze(params[..., 1]),
            np.squeeze(params[..., 2]),
        )
        if rate_type == "etr":
            txrates = (
                phi0 * (income - ((income**-phi1) + phi2) ** (-1 / phi1))
            ) / income
        else:  # marginal tax rate function
            txrates = phi0 * (
                1
                - (
                    income ** (-phi1 - 1)
                    * ((income**-phi1) + phi2) ** ((-1 - phi1) / phi1)
                )
            )
    if tax_func_type == "HSV":
        lambda_s, tau_s = (
            np.squeeze(params[..., 0]),
            np.squeeze(params[..., 1]),
        )
        if rate_type == "etr":
            txrates = 1 - (lambda_s * (income ** (-tau_s)))
        else:  # marginal tax rate function
            txrates = 1 - (lambda_s * (1 - tau_s) * (income ** (-tau_s)))
    elif tax_func_type == "DEP":
        (
            A,
            B,
            C,
            D,
            max_x,
            max_y,
            share,
            min_x,
            min_y,
            shift_x,
            shift_y,
            shift,
        ) = (
            np.squeeze(params[..., 0]),
            np.squeeze(params[..., 1]),
            np.squeeze(params[..., 2]),
            np.squeeze(params[..., 3]),
            np.squeeze(params[..., 4]),
            np.squeeze(params[..., 5]),
            np.squeeze(params[..., 6]),
            np.squeeze(params[..., 7]),
            np.squeeze(params[..., 8]),
            np.squeeze(params[..., 9]),
            np.squeeze(params[..., 10]),
            np.squeeze(params[..., 11]),
        )
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
            tau_x = (
                (max_x - min_x)
                * (A * X2til + B * Xtil + Etil)
                / (A * X2til + B * Xtil + Etil + 1)
            ) + min_x
            tau_y = (
                (max_y - min_y)
                * (C * Y2til + D * Ytil + Ftil)
                / (C * Y2til + D * Ytil + Ftil + 1)
            ) + min_y
            txrates = (
                ((tau_x + shift_x) ** share)
                * ((tau_y + shift_y) ** (1 - share))
            ) + shift
        else:
            if analytical_mtrs:
                tau_x = (max_x - min_x) * (A * X2 + B * X) / (
                    A * X2 + B * X + 1
                ) + min_x
                tau_y = (max_y - min_y) * (C * Y2 + D * Y) / (
                    C * Y2 + D * Y + 1
                ) + min_y
                etr = (
                    ((tau_x + shift_x) ** share)
                    * ((tau_y + shift_y) ** (1 - share))
                ) + shift
                if mtr_capital:
                    d_etr = (
                        (1 - share)
                        * ((tau_y + shift_y) ** (-share))
                        * (max_y - min_y)
                        * ((2 * C * Y + D) / ((C * Y2 + D * Y + 1) ** 2))
                        * ((tau_x + shift_x) ** share)
                    )
                    txrates = d_etr * income + etr
                else:
                    d_etr = (
                        share
                        * ((tau_x + shift_x) ** (share - 1))
                        * (max_x - min_x)
                        * ((2 * A * X + B) / ((A * X2 + B * X + 1) ** 2))
                        * ((tau_y + shift_y) ** (1 - share))
                    )
                    txrates = d_etr * income + etr
            else:
                tau_x = (
                    (max_x - min_x) * (A * X2 + B * X) / (A * X2 + B * X + 1)
                ) + min_x
                tau_y = (
                    (max_y - min_y) * (C * Y2 + D * Y) / (C * Y2 + D * Y + 1)
                ) + min_y
                txrates = (
                    ((tau_x + shift_x) ** share)
                    * ((tau_y + shift_y) ** (1 - share))
                ) + shift
    elif tax_func_type == "DEP_totalinc":
        A, B, max_income, min_income, shift_income, shift = (
            np.squeeze(params[..., 0]),
            np.squeeze(params[..., 1]),
            np.squeeze(params[..., 2]),
            np.squeeze(params[..., 3]),
            np.squeeze(params[..., 4]),
            np.squeeze(params[..., 5]),
        )
        Etil = A + B
        income2 = income**2
        if for_estimation:
            shift_income = np.maximum(-min_income, 0.0) + 0.01 * (
                max_income - min_income
            )
            income2bar = (income2 * wgts).sum() / wgts.sum()
            Ibar = (income * wgts).sum() / wgts.sum()
            income2til = (income2 - income2bar) / income2bar
            Itil = (income - Ibar) / Ibar
            tau_income = (
                (max_income - min_income)
                * (A * income2til + B * Itil + Etil)
                / (A * income2til + B * Itil + Etil + 1)
            ) + min_income
            txrates = tau_income + shift_income + shift
        else:
            if analytical_mtrs:
                d_etr = (max_income - min_income) * (
                    (2 * A * income + B)
                    / ((A * income2 + B * income + 1) ** 2)
                )
                etr = (
                    (
                        (max_income - min_income)
                        * (
                            (A * income2 + B * income)
                            / (A * income2 + B * income + 1)
                        )
                        + min_income
                    )
                    + shift_income
                    + shift
                )
                txrates = (d_etr * income) + (etr)
            else:
                tau_income = (
                    (max_income - min_income)
                    * (A * income2 + B * income)
                    / (A * income2 + B * income + 1)
                ) + min_income
                txrates = tau_income + shift_income + shift
    elif tax_func_type == "linear":
        rate = np.squeeze(params[..., 0])
        txrates = rate * np.ones_like(income)
    elif tax_func_type == "mono":
        if for_estimation:
            mono_interp = params[0]
            txrates = mono_interp(income)
        else:
            if np.isscalar(income):
                txrates = params[0](income)
            elif income.ndim == 1:
                # for s in range(income.shape[0]):
                #     txrates[s] = params[s][0](income[s])
                if (income.shape[0] == len(params)) and (
                    len(params) > 1
                ):  # for case where loops over S
                    txrates = [
                        params[s][0](income[s]) for s in range(income.shape[0])
                    ]
                else:
                    txrates = [
                        params[0](income[i]) for i in range(income.shape[0])
                    ]
            elif (
                income.ndim == 2
            ):  # I think only calls here are for loops over S and J
                # for s in range(income.shape[0]):
                #     for j in range(income.shape[1]):
                #         txrates[s, j] = params[s][j][0](income[s, j])
                txrates = [
                    [
                        params[s][j][0](income[s, j])
                        for j in range(income.shape[1])
                    ]
                    for s in range(income.shape[0])
                ]
            else:  # to catch 3D arrays, looping over T, S, J
                # for t in range(income.shape[0]):
                #     for s in range(income.shape[1]):
                #         for j in range(income.shape[2]):
                #             txrates[t, s, j] = params[t][s][j][0](
                #                 income[t, s, j]
                #             )
                txrates = [
                    [
                        [
                            params[t][s][j][0](income[t, s, j])
                            for j in range(income.shape[2])
                        ]
                        for s in range(income.shape[1])
                    ]
                    for t in range(income.shape[0])
                ]
        txrates = np.array(txrates)
    elif tax_func_type == "mono2D":
        if for_estimation:
            mono_interp = params[0]
            txrates = mono_interp([[X, Y]])
        else:
            if np.isscalar(X) and np.isscalar(Y):
                txrates = params[0]([[X, Y]])
            elif X.ndim == 1 and np.isscalar(Y):
                if (X.shape[0] == len(params)) and (
                    len(params) > 1
                ):  # for case where loops over S
                    txrates = [
                        params[s][0]([[X[s], Y]])
                        for s in range(income.shape[0])
                    ]
                else:
                    txrates = [
                        params[0](income[i]) for i in range(income.shape[0])
                    ]
            elif np.isscalar(X) and Y.ndim == 1:
                if (Y.shape[0] == len(params)) and (
                    len(params) > 1
                ):  # for case where loops over S
                    txrates = [
                        params[s][0]([[X, Y[s]]])
                        for s in range(income.shape[0])
                    ]
                else:
                    txrates = [
                        params[0](income[i]) for i in range(income.shape[0])
                    ]
            elif X.ndim == 1 and Y.ndim == 1:
                if (X.shape[0] == Y.shape[0] == len(params)) and (
                    len(params) > 1
                ):
                    txrates = [
                        params[s][0]([[X[s], Y[s]]]) for s in range(X.shape[0])
                    ]
                else:
                    txrates = [
                        params[0]([[X[i], Y[i]]]) for i in range(X.shape[0])
                    ]
            elif X.ndim == 2 and Y.ndim == 2:
                txrates = [
                    [
                        params[s][j][0]([[X[s, j], Y[s, j]]])
                        for j in range(X.shape[1])
                    ]
                    for s in range(X.shape[0])
                ]
            else:  # to catch 3D arrays, looping over T, S, J
                txrates = [
                    [
                        [
                            params[t][s][j][0]([[X[t, s, j], Y[t, s, j]]])
                            for j in range(income.shape[2])
                        ]
                        for s in range(income.shape[1])
                    ]
                    for t in range(income.shape[0])
                ]
        txrates = np.squeeze(np.array(txrates))

    return txrates


def wsumsq(params, *args):
    """
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

    """
    (
        fixed_tax_func_params,
        X,
        Y,
        txrates,
        wgts,
        tax_func_type,
        rate_type,
    ) = args
    params_all = np.append(params, fixed_tax_func_params)
    txrates_est = get_tax_rates(
        params_all, X, Y, wgts, tax_func_type, rate_type
    )
    errors = txrates_est - txrates
    wssqdev = (wgts * (errors**2)).sum()

    return wssqdev


def find_outliers(sse_mat, age_vec, se_mult, start_year, varstr, graph=False):
    """
    This function takes a matrix of sum of squared errors (SSE) from
    tax function estimations for each age (s) in each year of the budget
    window (t) and marks estimations that have outlier SSE.

    Args:
        sse_mat (Numpy array): SSE for each estimated tax function,
            size is BW x S
        age_vec (numpy array): vector of ages, length S
        se_mult (scalar): multiple of standard deviations before
            consider estimate an outlier
        start_year (int): first year of budget window
        varstr (str): name of tax function being evaluated
        graph (bool): whether to output graphs

    Returns:
        sse_big_mat (bool array_like): indicators of whether tax function
            is outlier, size is BW x S

    """
    # Mark outliers from estimated MTRx functions
    thresh = sse_mat[sse_mat > 0].mean() + se_mult * sse_mat[sse_mat > 0].std()
    sse_big_mat = sse_mat > thresh
    print(
        varstr,
        ": ",
        str(sse_big_mat.sum()),
        " observations tagged as outliers.",
    )
    output_dir = os.path.join(CUR_PATH, "OUTPUT", "TaxFunctions")
    if graph:
        pp.txfunc_sse_plot(age_vec, sse_mat, start_year, varstr, output_dir, 0)
    if sse_big_mat.sum() > 0:
        # Mark the outliers from the first sweep above. Then mark the
        # new outliers in a second sweep
        sse_mat_new = sse_mat.copy()
        sse_mat_new[sse_big_mat] = np.nan
        thresh2 = (
            sse_mat_new[sse_mat_new > 0].mean()
            + se_mult * sse_mat_new[sse_mat_new > 0].std()
        )
        sse_big_mat += sse_mat_new > thresh2
        print(
            varstr,
            ": ",
            "After second round, ",
            str(sse_big_mat.sum()),
            " observations tagged as outliers (cumulative).",
        )
        if graph:
            pp.txfunc_sse_plot(
                age_vec, sse_mat_new, start_year, varstr, output_dir, 1
            )
        if (sse_mat_new > thresh2).sum() > 0:
            # Mark the outliers from the second sweep above
            sse_mat_new2 = sse_mat_new.copy()
            sse_mat_new2[sse_big_mat] = np.nan
            if graph:
                pp.txfunc_sse_plot(
                    age_vec, sse_mat_new2, start_year, varstr, output_dir, 2
                )

    return sse_big_mat


def replace_outliers(param_list, sse_big_mat):
    """
    This function replaces outlier estimated tax function parameters
    with linearly interpolated tax function tax function parameters

    Args:
        param_list (list): estimated tax function parameters or nonparametric
            functions, size is BW x S x #TaxParams
        sse_big_mat (bool, array_like): indicators of whether tax function
            is outlier, size is BW x S

    Returns:
        param_arr_adj (array_like): estimated and interpolated tax function
            parameters, size BW x S x #TaxParams

    """
    numparams = len(param_list[0][0])
    S = sse_big_mat.shape[1]
    param_list_adj = param_list.copy()
    for t in range(sse_big_mat.shape[0]):
        big_cnt = 0
        for s in range(S):
            # Smooth out ETR tax function outliers
            if sse_big_mat[t, s] and s < S - 1:
                # For all outlier observations, increase the big_cnt by
                # 1 and set the param_arr_adj equal to nan
                big_cnt += 1
                param_list_adj[t][s] = np.nan
            if not sse_big_mat[t, s] and big_cnt > 0 and s == big_cnt:
                # When the current function is not an outlier but the last
                # one was and this string of outliers is at the beginning
                # ages, set the outliers equal to this period's tax function
                param_list_adj[t][:big_cnt] = [param_list_adj[t][s]] * big_cnt
                big_cnt = 0

            if not sse_big_mat[t, s] and big_cnt > 0 and s > big_cnt:
                # When the current function is not an outlier but the last
                # one was and this string of outliers is in the interior of
                # ages, set the outliers equal to a linear interpolation
                # between the two bounding non-outlier functions
                diff = (
                    param_list_adj[t][s] - param_list_adj[t][s - big_cnt - 1]
                )
                slopevec = (diff / (big_cnt + 1)).reshape(1, numparams)
                tiled_slopevec = np.tile(slopevec, (big_cnt, 1))

                interceptvec = param_list_adj[t][s - big_cnt - 1].reshape(
                    1, numparams
                )
                tiled_intvec = np.tile(interceptvec, (big_cnt, 1))

                reshaped_arange = np.arange(1, big_cnt + 1).reshape(big_cnt, 1)
                tiled_reshape_arange = np.tile(reshaped_arange, (1, numparams))

                for s_ind in range(big_cnt):
                    param_list_adj[t][s - big_cnt + s_ind] = tiled_intvec[
                        s_ind, :
                    ] + (
                        tiled_slopevec[s_ind, :]
                        * tiled_reshape_arange[s_ind, :]
                    )

                big_cnt = 0
            if sse_big_mat[t, s] and s == sse_big_mat.shape[1] - 1:
                # When the last ages are outliers, set the parameters equal
                # to the most recent non-outlier tax function
                big_cnt += 1
                param_list_adj[t][s] = np.nan
                reshaped = param_list_adj[t][s - big_cnt]
                param_list_adj[t][s - big_cnt + 1 :] = [
                    param_list_adj[t][s - big_cnt]
                ] * big_cnt

    return param_list_adj


def txfunc_est(
    df,
    s,
    t,
    rate_type,
    tax_func_type,
    numparams,
    output_dir,
    graph,
    params_init=None,
    global_opt=False,
):
    """
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

            * params (Numpy array or function object): vector of estimated
            parameters or nonparametric function object
            * wsse (scalar): weighted sum of squared deviations from
            minimization
            * obs (int): number of observations in the data, > 600

    """
    X = df["total_labinc"]
    Y = df["total_capinc"]
    wgts = df["weight"]
    X2 = X**2
    Y2 = Y**2
    X2bar = (X2 * wgts).sum() / wgts.sum()
    Xbar = (X * wgts).sum() / wgts.sum()
    Y2bar = (Y2 * wgts).sum() / wgts.sum()
    Ybar = (Y * wgts).sum() / wgts.sum()
    income = X + Y
    income2 = income**2
    Ibar = (income * wgts).sum() / wgts.sum()
    income2bar = (income2 * wgts).sum() / wgts.sum()
    if rate_type == "etr":
        txrates = df["etr"]
    elif rate_type == "mtrx":
        txrates = df["mtr_labinc"]
    elif rate_type == "mtry":
        txrates = df["mtr_capinc"]
    x_10pctl = df["total_labinc"].quantile(0.1)
    y_10pctl = df["total_capinc"].quantile(0.1)
    x_20pctl = df["total_labinc"].quantile(0.2)
    y_20pctl = df["total_capinc"].quantile(0.2)
    min_x = txrates[(df["total_capinc"] < y_10pctl)].min()
    min_y = txrates[(df["total_labinc"] < x_10pctl)].min()

    if tax_func_type == "DEP":
        # '''
        # Estimate DeBacker, Evans, Phillips (2018) ratio of polynomial
        # tax functions.
        # '''
        # if Atil_init not exist, set to 1.0
        if params_init is None:
            Atil_init = 1.0
            Btil_init = 1.0
            Ctil_init = 1.0
            Dtil_init = 1.0
            max_x_init = np.minimum(
                txrates[(df["total_capinc"] < y_20pctl)].max(), MAX_ETR + 0.05
            )
            max_y_init = np.minimum(
                txrates[(df["total_labinc"] < x_20pctl)].max(), MAX_ETR + 0.05
            )
            share_init = 0.5
            params_init = np.array(
                [
                    Atil_init,
                    Btil_init,
                    Ctil_init,
                    Dtil_init,
                    max_x_init,
                    max_y_init,
                    share_init,
                ]
            )
        shift = txrates[
            (df["total_labinc"] < x_20pctl) | (df["total_capinc"] < y_20pctl)
        ].min()
        shift_x = 0.0  # temp value
        shift_y = 0.0  # temp value
        tx_objs = (
            np.array([min_x, min_y, shift_x, shift_y, shift]),
            X,
            Y,
            txrates,
            wgts,
            tax_func_type,
            rate_type,
        )
        lb_max_x = np.maximum(min_x, 0.0) + 1e-4
        lb_max_y = np.maximum(min_y, 0.0) + 1e-4
        # bnds = (
        #     (1e-12, None),
        #     (1e-12, None),
        #     (1e-12, None),
        #     (1e-12, None),
        #     (lb_max_x, MAX_ETR + 0.15),
        #     (lb_max_y, MAX_ETR + 0.15),
        #     (0, 1),
        # )
        bnds = (
            (1e-12, 9999),
            (1e-12, 9999),
            (1e-12, 9999),
            (1e-12, 9999),
            (lb_max_x, MAX_ETR + 0.15),
            (lb_max_y, MAX_ETR + 0.15),
            (0, 1),
        )
        if global_opt:
            params_til = opt.differential_evolution(
                wsumsq, bounds=bnds, args=(tx_objs), seed=1
            )
        else:
            params_til = opt.minimize(
                wsumsq,
                params_init,
                args=(tx_objs),
                method="L-BFGS-B",
                bounds=bnds,
                tol=1e-15,
            )
        Atil, Btil, Ctil, Dtil, max_x, max_y, share = params_til.x
        # message = ("(max_x, min_x)=(" + str(max_x) + ", " + str(min_x) +
        #     "), (max_y, min_y)=(" + str(max_y) + ", " + str(min_y) + ")")
        # print(message)
        wsse = params_til.fun
        obs = df.shape[0]
        shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
        shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
        params = np.zeros(numparams)
        params[:4] = np.array([Atil, Btil, Ctil, Dtil]) / np.array(
            [X2bar, Xbar, Y2bar, Ybar]
        )
        params[4:] = np.array(
            [max_x, max_y, share, min_x, min_y, shift_x, shift_y, shift]
        )
        params_to_plot = params
        # set initial values to parameter estimates
        params_init = np.array(
            [
                Atil,
                Btil,
                Ctil,
                Dtil,
                max_x,
                max_y,
                share,
            ]
        )
    elif tax_func_type == "DEP_totalinc":
        # '''
        # Estimate DeBacker, Evans, Phillips (2018) ratio of polynomial
        # tax functions as a function of total income.
        # '''
        if params_init is None:
            Atil_init = 1.0
            Btil_init = 1.0
            max_x_init = np.minimum(
                txrates[(df["total_capinc"] < y_20pctl)].max(), MAX_ETR + 0.05
            )
            max_y_init = np.minimum(
                txrates[(df["total_labinc"] < x_20pctl)].max(), MAX_ETR + 0.05
            )
            max_income_init = max(max_x_init, max_y_init)
            share_init = 0.5
            params_init = np.array([Atil_init, Btil_init, max_income_init])
        shift = txrates[
            (df["total_labinc"] < x_20pctl) | (df["total_capinc"] < y_20pctl)
        ].min()
        min_income = min(min_x, min_y)
        shift_inc = 0.0  # temp value
        tx_objs = (
            np.array([min_income, shift_inc, shift]),
            X,
            Y,
            txrates,
            wgts,
            tax_func_type,
            rate_type,
        )
        lb_max_income = np.maximum(min_income, 0.0) + 1e-4
        # bnds = ((1e-12, None), (1e-12, None), (lb_max_income, MAX_ETR + 0.15))
        bnds = ((1e-12, 99999), (1e-12, 9999), (lb_max_income, MAX_ETR + 0.15))
        if global_opt:
            params_til = opt.differential_evolution(
                wsumsq, bounds=bnds, args=(tx_objs), seed=1
            )
        else:
            params_til = opt.minimize(
                wsumsq,
                params_init,
                args=(tx_objs),
                method="L-BFGS-B",
                bounds=bnds,
                tol=1e-15,
            )
        Atil, Btil, max_income = params_til.x
        wsse = params_til.fun
        obs = df.shape[0]
        shift_income = np.maximum(-min_income, 0.0) + 0.01 * (
            max_income - min_income
        )
        params = np.zeros(numparams)
        params[:2] = np.array([Atil, Btil]) / np.array([income2bar, Ibar])
        params[2:] = np.array([max_income, min_income, shift_income, shift])
        params_to_plot = params
        # set initial values to parameter estimates
        params_init = np.array([Atil, Btil, max_income])
    elif tax_func_type == "GS":
        # '''
        # Estimate Gouveia-Strauss parameters via least squares.
        # Need to use a different functional form than for DEP function.
        # '''
        if params_init is None:
            phi0_init = 1.0
            phi1_init = 1.0
            phi2_init = 1.0
            params_init = np.array([phi0_init, phi1_init, phi2_init])
        tx_objs = (
            np.array([None]),
            X,
            Y,
            txrates,
            wgts,
            tax_func_type,
            rate_type,
        )
        # bnds = ((1e-12, None), (1e-12, None), (1e-12, None))
        bnds = ((1e-12, 9999), (1e-12, 9999), (1e-12, 9999))
        if global_opt:
            params_til = opt.differential_evolution(
                wsumsq, bounds=bnds, args=(tx_objs), seed=1
            )
        else:
            params_til = opt.minimize(
                wsumsq,
                params_init,
                args=(tx_objs),
                method="L-BFGS-B",
                bounds=bnds,
                tol=1e-15,
            )
        phi0til, phi1til, phi2til = params_til.x
        wsse = params_til.fun
        obs = df.shape[0]
        params = np.zeros(numparams)
        params[:3] = np.array([phi0til, phi1til, phi2til])
        params_to_plot = params
        # set initial values to parameter estimates
        params_init = np.array([phi0til, phi1til, phi2til])
    elif tax_func_type == "HSV":
        # '''
        # Estimate Heathcote, Storesletten, Violante (2017) parameters via
        # OLS
        # '''
        constant = np.ones_like(income)
        ln_income = np.log(income)
        X_mat = np.column_stack((constant, ln_income))
        Y_vec = np.log(1 - txrates)
        param_est = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ Y_vec
        params = np.zeros(numparams)
        if rate_type == "etr":
            ln_lambda_s_hat, minus_tau_s_hat = param_est
            params[:2] = np.array([np.exp(ln_lambda_s_hat), -minus_tau_s_hat])
        else:
            constant, minus_tau_s_hat = param_est
            lambda_s_hat = np.exp(constant - np.log(1 + minus_tau_s_hat))
            params[:2] = np.array([lambda_s_hat, -minus_tau_s_hat])
        # Calculate the WSSE
        Y_hat = X_mat @ params
        # wsse = ((Y_vec - Y_hat) ** 2 * wgts).sum()
        wsse = ((Y_vec - Y_hat) ** 2).sum()
        obs = df.shape[0]
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
        params[0] = (txrates * wgts * income).sum() / (income * wgts).sum()
        params_to_plot = params
    elif tax_func_type == "mono":
        # '''
        # For monotonically increasing smoothing spline function rates, return
        # the resulting function object in place of parametric model.
        # This is the approach we will take with all nonparametric tax
        # functions and might be the best approach even for our parametric
        # tax functions.
        # '''
        # Set the number of bins to 500 unless the number of observations is
        # less-than-or-equal-to 1,000 and greater-than-or-equal-to MIN_OBS. For
        # that case create a linear function of bin number as 80% of MIN_OBS at
        # MIN_OBS and 500 bins at 1,000 observations.
        obs = df.shape[0]
        slope = (500 - np.round(0.8 * MIN_OBS)) / (1_000 - MIN_OBS)
        intercept = 500 - slope * 1_000
        bin_num = np.minimum(500, slope * obs + intercept)
        mono_interp, _, wsse_cstr, _, _ = monotone_spline(
            income, txrates, wgts, bins=bin_num
        )
        wsse = wsse_cstr
        params = [mono_interp]
        params_to_plot = params
    elif tax_func_type == "mono2D":
        obs = df.shape[0]
        mono_interp, _, wsse_cstr, _, _ = monotone_spline(
            # df[["total_labinc", "total_capinc"]].values,
            # df["etr"].values,
            # df["weight"].values,
            np.vstack((X, Y)).T,
            # X, Y,
            txrates,
            wgts,
            bins=[100, 100],
            method="pygam",
            splines=[100, 100],
        )
        wsse = wsse_cstr
        params = [mono_interp]
        params_to_plot = params
    else:
        raise RuntimeError(
            "Choice of tax function is not in the set of"
            + " possible tax functions.  Please select"
            + " from: DEP, DEP_totalinc, GS, linear, mono, mono2D."
        )
    if graph:
        pp.txfunc_graph(
            s,
            t,
            df,
            X,
            Y,
            txrates,
            rate_type,
            tax_func_type,
            params_to_plot,
            output_dir,
        )

    # Garbage collection
    del df, txrates

    return params, wsse, obs, params_init


def tax_data_sample(
    data, max_etr=MAX_ETR, min_income=MIN_INCOME, max_mtr=MAX_MTR
):
    """
    Function to create sample tax data for estimation by dropping
    observations with extreme values.

    Args:
        data (DataFrame): raw data from microsimulation model

    Returns:
        data (DataFrame): selected sample

    """
    # drop all obs with ETR > MAX_ETR
    data.drop(data[data["etr"] > MAX_ETR].index, inplace=True)
    # drop all obs with ETR < MIN_ETR
    # set min ETR to value at 10th percentile in distribution of ETRs
    min_etr = data["etr"].quantile(q=0.10)
    data.drop(data[data["etr"] < min_etr].index, inplace=True)
    # drop all obs with total market income, labor income, or
    # capital income < MIN_INCOME
    data.drop(
        data[
            (data["market_income"] < MIN_INCOME)
            | (data["total_labinc"] < MIN_INCOME)
            | (data["total_capinc"] < MIN_INCOME)
        ].index,
        inplace=True,
    )
    # drop all obs with MTR on capital income > MAX_MTR
    data.drop(data[data["mtr_capinc"] > MAX_MTR].index, inplace=True)
    # drop all obs with MTR on capital income < min_cap_mtr
    # set min MTR to value at 10th percentile in distribution of MTRs
    min_cap_mtr = data["mtr_capinc"].quantile(q=0.10)
    data.drop(data[data["mtr_capinc"] < min_cap_mtr].index, inplace=True)
    # drop all obs with MTR on labor income > MAX_MTR
    data.drop(data[data["mtr_labinc"] > MAX_MTR].index, inplace=True)
    # drop all obs with MTR on labor income < min_lab_mtr
    min_lab_mtr = data["mtr_labinc"].quantile(q=0.10)
    data.drop(data[data["mtr_labinc"] < min_lab_mtr].index, inplace=True)

    return data


def tax_func_loop(
    t,
    data,
    start_year,
    s_min,
    s_max,
    age_specific,
    tax_func_type,
    analytical_mtrs,
    desc_data,
    graph_data,
    graph_est,
    output_dir,
    numparams,
):
    """
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

    """
    # initialize arrays for output
    etrparam_list = np.zeros(s_max - s_min + 1).tolist()
    mtrxparam_list = np.zeros(s_max - s_min + 1).tolist()
    mtryparam_list = np.zeros(s_max - s_min + 1).tolist()
    etr_wsumsq_arr = np.zeros(s_max - s_min + 1)
    etr_obs_arr = np.zeros(s_max - s_min + 1)
    mtrx_wsumsq_arr = np.zeros(s_max - s_min + 1)
    mtrx_obs_arr = np.zeros(s_max - s_min + 1)
    mtry_wsumsq_arr = np.zeros(s_max - s_min + 1)
    mtry_obs_arr = np.zeros(s_max - s_min + 1)
    PopPct_age = np.zeros(s_max - s_min + 1)

    # Calculate average total income in each year
    AvgInc = ((data["market_income"] * data["weight"]).sum()) / data[
        "weight"
    ].sum()

    # Calculate average ETR and MTRs (weight by population weights
    #    and income) for each year
    AvgETR = ((data["etr"] * data["market_income"] * data["weight"]).sum()) / (
        data["market_income"] * data["weight"]
    ).sum()

    AvgMTRx = (
        (data["mtr_labinc"] * data["market_income"] * data["weight"]).sum()
    ) / (data["market_income"] * data["weight"]).sum()

    AvgMTRy = (
        (data["mtr_capinc"] * data["market_income"] * data["weight"]).sum()
    ) / (data["market_income"] * data["weight"]).sum()

    # Caulcatoe fraction of total tax liability that is from payroll
    # taxes
    frac_tax_payroll = (data["payroll_tax_liab"] * data["weight"]).sum() / (
        data["total_tax_liab"] * data["weight"]
    ).sum()

    # Calculate total population in each year
    TotPop_yr = data["weight"].sum()

    # Clean up the data by dropping outliers
    data = tax_data_sample(data)

    # Create an array of the different ages in the data
    min_age = int(np.maximum(data["age"].min(), s_min))
    max_age = int(np.minimum(data["age"].max(), s_max))
    if age_specific:
        ages_list = np.arange(min_age, max_age + 1)
    else:
        ages_list = np.arange(0, 1)

    NoData_cnt = np.min(min_age - s_min, 0)

    # Each age s must be done in serial
    # Set initial values
    # TODO: update this, so if using DEP or GS initial parameters are estimated on all ages first
    if tax_func_type in ["DEP", "DEP_totalinc", "GS"]:
        s = 0
        df = data
        df_etr = df.loc[
            df[
                (np.isfinite(df["etr"]))
                & (np.isfinite(df["total_labinc"]))
                & (np.isfinite(df["total_capinc"]))
                & (np.isfinite(df["weight"]))
            ].index,
            [
                "mtr_labinc",
                "mtr_capinc",
                "total_labinc",
                "total_capinc",
                "etr",
                "weight",
            ],
        ].copy()
        df_mtrx = df.loc[
            df[
                (np.isfinite(df["mtr_labinc"]))
                & (np.isfinite(df["total_labinc"]))
                & (np.isfinite(df["total_capinc"]))
                & (np.isfinite(df["weight"]))
            ].index,
            ["mtr_labinc", "total_labinc", "total_capinc", "weight"],
        ].copy()
        df_mtry = df.loc[
            df[
                (np.isfinite(df["mtr_capinc"]))
                & (np.isfinite(df["total_labinc"]))
                & (np.isfinite(df["total_capinc"]))
                & (np.isfinite(df["weight"]))
            ].index,
            ["mtr_capinc", "total_labinc", "total_capinc", "weight"],
        ].copy()
        # Estimate effective tax rate function ETR(x,y)
        (
            etrparams,
            etr_wsumsq_arr[s - s_min],
            etr_obs_arr[s - s_min],
            params_init_etr,
        ) = txfunc_est(
            df_etr,
            s,
            t,
            "etr",
            tax_func_type,
            numparams,
            output_dir,
            False,
            None,
            True,
        )
        etrparam_list[s - s_min] = etrparams
        del df_etr

        # Estimate marginal tax rate of labor income function
        # MTRx(x,y)
        (
            mtrxparams,
            mtrx_wsumsq_arr[s - s_min],
            mtrx_obs_arr[s - s_min],
            params_init_mtrx,
        ) = txfunc_est(
            df_mtrx,
            s,
            t,
            "mtrx",
            tax_func_type,
            numparams,
            output_dir,
            False,
            None,
            True,
        )
        del df_mtrx
        # Estimate marginal tax rate of capital income function
        # MTRy(x,y)
        (
            mtryparams,
            mtry_wsumsq_arr[s - s_min],
            mtry_obs_arr[s - s_min],
            params_init_mtry,
        ) = txfunc_est(
            df_mtry,
            s,
            t,
            "mtry",
            tax_func_type,
            numparams,
            output_dir,
            False,
            None,
            True,
        )
        mtryparam_list[s - s_min] = mtryparams
    else:
        params_init_etr = None
        params_init_mtrx = None
        params_init_mtry = None
    for s in ages_list:
        if age_specific:
            print("Year=", t, "Age=", s)
            df = data[data["age"] == s]
            PopPct_age[s - min_age] = df["weight"].sum() / TotPop_yr

        else:
            print("year=", t, "age= all ages")
            df = data
            PopPct_age[0] = df["weight"].sum() / TotPop_yr
        df_etr = df.loc[
            df[
                (np.isfinite(df["etr"]))
                & (np.isfinite(df["total_labinc"]))
                & (np.isfinite(df["total_capinc"]))
                & (np.isfinite(df["weight"]))
            ].index,
            [
                "mtr_labinc",
                "mtr_capinc",
                "total_labinc",
                "total_capinc",
                "etr",
                "weight",
            ],
        ].copy()
        df_mtrx = df.loc[
            df[
                (np.isfinite(df["mtr_labinc"]))
                & (np.isfinite(df["total_labinc"]))
                & (np.isfinite(df["total_capinc"]))
                & (np.isfinite(df["weight"]))
            ].index,
            ["mtr_labinc", "total_labinc", "total_capinc", "weight"],
        ].copy()
        df_mtry = df.loc[
            df[
                (np.isfinite(df["mtr_capinc"]))
                & (np.isfinite(df["total_labinc"]))
                & (np.isfinite(df["total_capinc"]))
                & (np.isfinite(df["weight"]))
            ].index,
            ["mtr_capinc", "total_labinc", "total_capinc", "weight"],
        ].copy()
        df_minobs = np.min(
            [df_etr.shape[0], df_mtrx.shape[0], df_mtry.shape[0]]
        )
        del df

        if df_minobs < MIN_OBS and s < max_age:
            # '''
            # --------------------------------------------------------
            # Don't estimate function on this iteration if obs < 240.
            # Will fill in later with interpolated values
            # --------------------------------------------------------
            # '''
            message = (
                "Insuff. sample size for age " + str(s) + " in year " + str(t)
            )
            print(message)
            NoData_cnt += 1
            etrparam_list[s - s_min] = None
            mtrxparam_list[s - s_min] = None
            mtryparam_list[s - s_min] = None

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
            message = (
                "Max age (s="
                + str(s)
                + ") insuff. data in"
                + " year "
                + str(t)
                + ". Fill in final ages with "
                + "insuff. data with most recent successful "
                + "estimate."
            )
            print(message)
            NoData_cnt += 1
            lastp_etr = etrparam_list[s - NoData_cnt - s_min]
            etrparam_list[s - NoData_cnt - s_min + 1 :] = [lastp_etr] * (
                NoData_cnt + s_max - max_age
            )
            lastp_mtrx = mtrxparam_list[s - NoData_cnt - s_min]
            mtrxparam_list[s - NoData_cnt - s_min + 1 :] = [lastp_mtrx] * (
                NoData_cnt + s_max - max_age
            )
            lastp_mtry = mtryparam_list[s - NoData_cnt - s_min]
            mtryparam_list[s - NoData_cnt - s_min + 1 :] = [lastp_mtry] * (
                NoData_cnt + s_max - max_age
            )

        else:
            # Estimate parameters for age with sufficient data
            if desc_data:
                # print some desciptive stats
                message = (
                    "Descriptive ETR statistics for age="
                    + str(s)
                    + " in year "
                    + str(t)
                )
                print(message)
                print(df_etr.describe())
                message = (
                    "Descriptive MTRx statistics for age="
                    + str(s)
                    + " in year "
                    + str(t)
                )
                print(message)
                print(df_mtrx.describe())
                message = (
                    "Descriptive MTRy statistics for age="
                    + str(s)
                    + " in year "
                    + str(t)
                )
                print(message)
                print(df_mtry.describe())

            if graph_data:
                pp.gen_3Dscatters_hist(df_etr, s, t, output_dir)

            # Estimate effective tax rate function ETR(x,y)
            (
                etrparams,
                etr_wsumsq_arr[s - s_min],
                etr_obs_arr[s - s_min],
                params_init,
            ) = txfunc_est(
                df_etr,
                s,
                t,
                "etr",
                tax_func_type,
                numparams,
                output_dir,
                graph_est,
                params_init_etr,
            )
            etrparam_list[s - s_min] = etrparams
            del df_etr

            # Estimate marginal tax rate of labor income function
            # MTRx(x,y)
            (
                mtrxparams,
                mtrx_wsumsq_arr[s - s_min],
                mtrx_obs_arr[s - s_min],
                params_init,
            ) = txfunc_est(
                df_mtrx,
                s,
                t,
                "mtrx",
                tax_func_type,
                numparams,
                output_dir,
                graph_est,
                params_init_mtrx,
            )
            mtrxparam_list[s - s_min] = mtrxparams
            del df_mtrx
            # Estimate marginal tax rate of capital income function
            # MTRy(x,y)
            (
                mtryparams,
                mtry_wsumsq_arr[s - s_min],
                mtry_obs_arr[s - s_min],
                params_init,
            ) = txfunc_est(
                df_mtry,
                s,
                t,
                "mtry",
                tax_func_type,
                numparams,
                output_dir,
                graph_est,
                params_init_mtry,
            )
            mtryparam_list[s - s_min] = mtryparams

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
                etrparam_list[: s - s_min] = [etrparams] * (s - s_min)
                mtrxparam_list[: s - s_min] = [mtrxparams] * (s - s_min)
                mtryparam_list[: s - s_min] = [mtryparams] * (s - s_min)

            elif (
                (NoData_cnt > 0)
                & (NoData_cnt < s - s_min)
                & (tax_func_type != "mono")
            ):
                # '''
                # -------------------------------------------------------------
                # For all parametric tax function types (not "mono"), fill in
                # interior data gaps with linear interpolation between
                # bracketing positive data ages. In all of these cases,
                # min_age < s <= max_age.
                # -------------------------------------------------------------
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
                # -------------------------------------------------------------
                # '''
                message = "Linearly interpolate previous blank tax functions"
                print(message)
                tvals = np.linspace(0, 1, NoData_cnt + 2)
                x0_etr = np.tile(
                    etrparam_list[s - NoData_cnt - s_min - 1].reshape(
                        (1, numparams)
                    ),
                    (NoData_cnt, 1),
                )
                x1_etr = np.tile(
                    etrparams.reshape((1, numparams)), (NoData_cnt, 1)
                )
                lin_int_etr = x0_etr + tvals[1:-1].reshape((NoData_cnt, 1)) * (
                    x1_etr - x0_etr
                )

                x0_mtrx = np.tile(
                    mtrxparam_list[s - NoData_cnt - s_min - 1].reshape(
                        (1, numparams)
                    ),
                    (NoData_cnt, 1),
                )
                x1_mtrx = np.tile(
                    mtrxparams.reshape((1, numparams)), (NoData_cnt, 1)
                )
                lin_int_mtrx = x0_mtrx + tvals[1:-1].reshape(
                    (NoData_cnt, 1)
                ) * (x1_mtrx - x0_mtrx)

                x0_mtry = np.tile(
                    mtryparam_list[s - NoData_cnt - s_min - 1].reshape(
                        (1, numparams)
                    ),
                    (NoData_cnt, 1),
                )
                x1_mtry = np.tile(
                    mtryparams.reshape((1, numparams)), (NoData_cnt, 1)
                )
                lin_int_mtry = x0_mtry + tvals[1:-1].reshape(
                    (NoData_cnt, 1)
                ) * (x1_mtry - x0_mtry)
                for s_ind in range(NoData_cnt):
                    etrparam_list[s - NoData_cnt - min_age + s_ind] = (
                        lin_int_etr[s_ind, :]
                    )
                    mtrxparam_list[s - NoData_cnt - min_age + s_ind] = (
                        lin_int_mtrx[s_ind, :]
                    )
                    mtryparam_list[s - NoData_cnt - min_age + s_ind] = (
                        lin_int_mtry[s_ind, :]
                    )

            elif (
                (NoData_cnt > 0)
                & (NoData_cnt < s - s_min)
                & (tax_func_type == "mono")
            ):
                # '''
                # -------------------------------------------------------------
                # For all nonparametric tax function types ("mono"), fill in
                # interior data gaps with the previous estimated interpolating
                # function (no interpolation between last function and current
                # function). In all of these cases, min_age < s <= max_age.
                # -------------------------------------------------------------
                etrparam_list[s - NoData_cnt - min_age : s - min_age] = [
                    etrparam_list[s - NoData_cnt - s_min - 1]
                ] * NoData_cnt
                mtrxparam_list[s - NoData_cnt - min_age : s - min_age] = [
                    mtrxparam_list[s - NoData_cnt - s_min - 1]
                ] * NoData_cnt
                mtryparam_list[s - NoData_cnt - min_age : s - min_age] = [
                    mtryparam_list[s - NoData_cnt - s_min - 1]
                ] * NoData_cnt

            NoData_cnt == 0

            if s == max_age and max_age < s_max:
                # '''
                # ----------------------------------------------------
                # If the last age estimates, and max_age< s_max, fill
                # in the remaining ages with these last estimates
                # ----------------------------------------------------
                # '''
                message = "Fill in all remaining old age tax functions."
                print(message)
                etrparam_list[s - s_min + 1 :] = [etrparams] * (
                    s_max - max_age
                )
                mtrxparam_list[s - s_min + 1 :] = [mtrxparams] * (
                    s_max - max_age
                )
                mtryparam_list[s - s_min + 1 :] = [mtryparams] * (
                    s_max - max_age
                )

    return (
        TotPop_yr,
        PopPct_age,
        AvgInc,
        AvgETR,
        AvgMTRx,
        AvgMTRy,
        frac_tax_payroll,
        etrparam_list,
        etr_wsumsq_arr,
        etr_obs_arr,
        mtrxparam_list,
        mtrx_wsumsq_arr,
        mtrx_obs_arr,
        mtryparam_list,
        mtry_wsumsq_arr,
        mtry_obs_arr,
    )


def tax_func_estimate(
    micro_data,
    BW,
    S,
    starting_age,
    ending_age,
    start_year=DEFAULT_START_YEAR,
    analytical_mtrs=False,
    tax_func_type="DEP",
    age_specific=False,
    desc_data=False,
    graph_data=False,
    graph_est=False,
    client=None,
    num_workers=1,
    tax_func_path=None,
):
    """
    This function performs analysis on the source data from microsimulation
    model and estimates functions for the effective tax rate (ETR), marginal
    tax rate on labor income (MTRx), and marginal tax rate on capital income
    (MTRy).

    Args:
        micro_data (dict): Dictionary of DataFrames with micro data
        BW (int): number of years in the budget window (the period
            over which tax policy is assumed to vary)
        S (int): number of model periods a model agent is economically
            active for
        starting_age (int): minimum age to estimate tax functions for
        ending_age (int): maximum age to estimate tax functions for
        start_yr (int): first year of budget window
        analytical_mtrs (bool): whether to use the analytical derivation
            of the marginal tax rates (and thus only need to estimate
            the effective tax rate functions)
        tax_func_type (str): functional form of tax functions
        age_specific (bool): whether to estimate age specific tax
            functions
        client (Dask client object): client
        num_workers (int): number of workers to use for parallelization
            with Dask
        tax_func_path (str): path to save pickle with estimated tax
            function parameters to

    Returns:
        dict_param (dict): dictionary with tax function parameters

    """
    s_min = starting_age + 1
    s_max = ending_age
    start_year = int(start_year)
    end_yr = int(start_year + BW - 1)
    print("BW = ", BW, "begin year = ", start_year, "end year = ", end_yr)
    tax_func_type_num_params_dict = {
        "DEP": 12,
        "DEP_totalinc": 6,
        "GS": 3,
        "HSV": 2,
        "linear": 1,
        "mono": 1,
        "mono2D": 1,
    }
    numparams = int(tax_func_type_num_params_dict[tax_func_type])
    years_list = np.arange(start_year, end_yr + 1)
    if age_specific:
        ages_list = np.arange(s_min, s_max + 1)
    else:
        ages_list = np.arange(0, 1)
    # initialize arrays for output
    etrparam_list = np.zeros(BW).tolist()
    mtrxparam_list = np.zeros(BW).tolist()
    mtryparam_list = np.zeros(BW).tolist()
    etr_wsumsq_arr = np.zeros((BW, s_max - s_min + 1))
    etr_obs_arr = np.zeros((BW, s_max - s_min + 1))
    mtrx_wsumsq_arr = np.zeros((BW, s_max - s_min + 1))
    mtrx_obs_arr = np.zeros((BW, s_max - s_min + 1))
    mtry_wsumsq_arr = np.zeros((BW, s_max - s_min + 1))
    mtry_obs_arr = np.zeros((BW, s_max - s_min + 1))
    AvgInc = np.zeros(BW)
    AvgETR = np.zeros(BW)
    AvgMTRx = np.zeros(BW)
    AvgMTRy = np.zeros(BW)
    frac_tax_payroll = np.zeros(BW)
    TotPop_yr = np.zeros(BW)
    PopPct_age = np.zeros((BW, s_max - s_min + 1))

    # '''
    # --------------------------------------------------------------------
    # Solve for tax functions for each year (t) and each age (s)
    # --------------------------------------------------------------------
    # start_time = scalar, current processor time in seconds (float)
    # output_dir = string, directory to which plots will be saved
    # t          = integer >= start_year, index for year of analysis
    # --------------------------------------------------------------------
    # '''
    start_time = time.time()
    if not tax_func_path:
        output_dir = os.path.join(CUR_PATH, "OUTPUT", "TaxFunctions")
    else:
        output_dir = os.path.join(
            os.path.dirname(tax_func_path), "OUTPUT", "TaxFunctions"
        )
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    lazy_values = []
    for t in years_list:
        lazy_values.append(
            delayed(tax_func_loop)(
                t,
                micro_data[str(t)],
                start_year,
                s_min,
                s_max,
                age_specific,
                tax_func_type,
                analytical_mtrs,
                desc_data,
                graph_data,
                graph_est,
                output_dir,
                numparams,
            )
        )
    if client:
        futures = client.compute(lazy_values, num_workers=num_workers)
        results = client.gather(futures)
    else:
        results = results = compute(
            *lazy_values,
            scheduler=dask.multiprocessing.get,
            num_workers=num_workers,
        )

    # Garbage collection
    del micro_data

    # for i, result in results.items():
    for i, result in enumerate(results):
        (
            TotPop_yr[i],
            PopPct_age[i, :],
            AvgInc[i],
            AvgETR[i],
            AvgMTRx[i],
            AvgMTRy[i],
            frac_tax_payroll[i],
            etrparam_list[i],
            etr_wsumsq_arr[i, :],
            etr_obs_arr[i, :],
            mtrxparam_list[i],
            mtrx_wsumsq_arr[i, :],
            mtrx_obs_arr[i, :],
            mtryparam_list[i],
            mtry_wsumsq_arr[i, :],
            mtry_obs_arr[i, :],
        ) = result

    message = (
        "Finished tax function loop through "
        + str(len(years_list))
        + " years and "
        + str(len(ages_list))
        + " ages per year."
    )
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
        message = (
            "Tax function estimation time: "
            + str(mins)
            + " min, "
            + str(secs)
            + " sec"
        )
        print(message)
    elif elapsed_time >= 3600 and elapsed_time < 86400:  # less than day
        hours = int(elapsed_time / (60 * 60))
        mins = int((elapsed_time - (hours * 60 * 60)) / 60)
        secs = round(elapsed_time - (hours * 60 * 60) - (mins * 60), 1)
        message = (
            "Tax function estimation time: "
            + str(hours)
            + " hour(s), "
            + str(mins)
            + " min(s), "
            + str(secs)
            + " sec(s)"
        )
        print(message)

    # '''
    # -------------------------------------------------------------------------
    # Replace outlier tax functions (SSE>mean+2.5*std) with linear
    # interpolation for parametric tax functions (not "mono"). We make two
    # passes (filtering runs).
    # -------------------------------------------------------------------------
    # '''
    if age_specific and tax_func_type != "mono":
        age_sup = np.linspace(s_min, s_max, s_max - s_min + 1)
        se_mult = 3.5
        etr_sse_big = find_outliers(
            etr_wsumsq_arr / etr_obs_arr,
            age_sup,
            se_mult,
            start_year,
            "ETR",
            graph=graph_est,
        )
        if etr_sse_big.sum() > 0:
            etrparam_list_adj = replace_outliers(etrparam_list, etr_sse_big)
        elif etr_sse_big.sum() == 0:
            etrparam_list_adj = etrparam_list

        mtrx_sse_big = find_outliers(
            mtrx_wsumsq_arr / mtrx_obs_arr,
            age_sup,
            se_mult,
            start_year,
            "MTRx",
            graph=graph_est,
        )
        if mtrx_sse_big.sum() > 0:
            mtrxparam_list_adj = replace_outliers(mtrxparam_list, mtrx_sse_big)
        elif mtrx_sse_big.sum() == 0:
            mtrxparam_list_adj = mtrxparam_list

        mtry_sse_big = find_outliers(
            mtry_wsumsq_arr / mtry_obs_arr,
            age_sup,
            se_mult,
            start_year,
            "MTRy",
            graph=graph_est,
        )
        if mtry_sse_big.sum() > 0:
            mtryparam_list_adj = replace_outliers(mtryparam_list, mtry_sse_big)
        elif mtry_sse_big.sum() == 0:
            mtryparam_list_adj = mtryparam_list

    # '''
    # -------------------------------------------------------------------------
    # Generate tax function parameters for S < s_max - s_min + 1
    # -------------------------------------------------------------------------
    # etrparam_list_S (list BW x S x numparams): this is an array in which S is
    #     less-than-or-equal-to s_max-s_min+1. We use weighted averages of
    #     parameters in relevant age groups
    # mtrxparam_list_S (list BW x S x numparams): this is an array in which S
    #     is less-than-or-equal-to s_max-s_min+1. We use weighted averages of
    #     parameters in relevant age groups
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
    if age_specific and tax_func_type != "mono":
        if S == s_max - s_min + 1:
            etrparam_list_S = etrparam_list_adj
            mtrxparam_list_S = mtrxparam_list_adj
            mtryparam_list_S = mtryparam_list_adj
        elif S < s_max - s_min + 1:
            etrparam_list_S = np.zeros((BW, S)).tolist()
            mtrxparam_list_S = np.zeros((BW, S)).tolist()
            mtryparam_list_S = np.zeros((BW, S)).tolist()
            age_cuts = np.linspace(0, s_max - s_min + 1, S + 1)
            yrcut_lb = int(age_cuts[0])
            rmndr_pct_lb = 1.0
            for bw in range(BW):
                for s in range(S):
                    yrcut_ub = int(np.floor(age_cuts[s + 1]))
                    rmndr_pct_ub = age_cuts[s + 1] - np.floor(age_cuts[s + 1])
                    if rmndr_pct_ub == 0.0:
                        rmndr_pct_ub = 1.0
                        yrcut_ub -= 1
                    age_wgts = PopPct_age[bw, yrcut_lb : yrcut_ub + 1]
                    age_wgts[0] *= rmndr_pct_lb
                    age_wgts[yrcut_ub - yrcut_lb] *= rmndr_pct_ub
                    etrparam_list_S[bw][s] = (
                        etrparam_list_adj[bw][yrcut_lb : yrcut_ub + 1]
                        * age_wgts
                    ).sum()
                    mtrxparam_list_S[bw][s] = (
                        mtrxparam_list_adj[bw][yrcut_lb : yrcut_ub + 1]
                        * age_wgts
                    ).sum()
                    mtryparam_list_S[bw][s] = (
                        mtryparam_list_adj[bw][yrcut_lb : yrcut_ub + 1]
                        * age_wgts
                    ).sum()
                    yrcut_lb = yrcut_ub
                    rmndr_pct_lb = 1 - rmndr_pct_ub

        else:
            err_msg(
                "txfunc ERROR: S is larger than the difference between "
                + "the minimum age and the maximum age specified. Please "
                + "choose an S such that a model period equals at least "
                + "one calendar year."
            )
            raise ValueError(err_msg)
        print("Big S: ", S)
        print("max age, min age: ", s_max, s_min)
    elif age_specific and tax_func_type == "mono":
        if S == s_max - s_min + 1:
            etrparam_list_S = etrparam_list_adj
            mtrxparam_list_S = mtrxparam_list_adj
            mtryparam_list_S = mtryparam_list_adj
        elif S < s_max - s_min + 1:
            err_msg = (
                "txfunc ERROR: tax_func_type = mono and S < s_max - "
                + "s_min + 1"
            )
            raise ValueError(err_msg)
        else:
            err_msg(
                "txfunc ERROR: S is larger than the difference between "
                + "the minimum age and the maximum age specified. Please "
                + "choose an S such that a model period equals at least "
                + "one calendar year."
            )
            raise ValueError(err_msg)

        print("Big S: ", S)
        print("max age, min age: ", s_max, s_min)
    elif not age_specific:
        etrparam_list_S = np.zeros((BW, S)).tolist()
        mtrxparam_list_S = np.zeros((BW, S)).tolist()
        mtryparam_list_S = np.zeros((BW, S)).tolist()
        for bw in range(BW):
            etrparam_list_S[bw][:] = [etrparam_list[bw][0 - s_min]] * S
            mtrxparam_list_S[bw][:] = [mtrxparam_list[bw][0 - s_min]] * S
            mtryparam_list_S[bw][:] = [mtryparam_list[bw][0 - s_min]] * S

    # Save tax function parameters array and computation time in
    # dictionary
    dict_params = dict(
        [
            ("tfunc_etr_params_S", etrparam_list_S),
            ("tfunc_mtrx_params_S", mtrxparam_list_S),
            ("tfunc_mtry_params_S", mtryparam_list_S),
            ("tfunc_avginc", AvgInc),
            ("tfunc_avg_etr", AvgETR),
            ("tfunc_avg_mtrx", AvgMTRx),
            ("tfunc_avg_mtry", AvgMTRy),
            ("tfunc_frac_tax_payroll", frac_tax_payroll),
            ("tfunc_etr_sumsq", etr_wsumsq_arr),
            ("tfunc_mtrx_sumsq", mtrx_wsumsq_arr),
            ("tfunc_mtry_sumsq", mtry_wsumsq_arr),
            ("tfunc_etr_obs", etr_obs_arr),
            ("tfunc_mtrx_obs", mtrx_obs_arr),
            ("tfunc_mtry_obs", mtry_obs_arr),
            ("tfunc_time", elapsed_time),
            ("tax_func_type", tax_func_type),
            ("start_year", start_year),
            ("BW", BW),
        ]
    )

    if tax_func_path:
        with open(tax_func_path, "wb") as f:
            try:
                pickle.dump(dict_params, f)
            except AttributeError:
                cloudpickle.dump(dict_params, f)

    return dict_params


def avg_by_bin_multd(x, y, bins, weights=None):
    """
    Args:
        x (numpy array): 2d with dimensions n by m used for binning
        y (numpy array): 1d with length n
        bins (numpy array): 1d with length m, each entry must divide n
            and is number of bins for corresponding column in x
        weights (None or numpy array): 1d with length n specifying
            weight of each observation. if None then array of ones

    Returns:
        xNew (numpy array): 2d with second dimension m, first
            dimension is product of elements in bins, with each entry
            representative of bin across all the features
        yNew (numpy array): 1d with length same as first dimension
            of xWeight, weighted average of y's corresponding to each
            entry of xWeight
        weightsNew (numpy array): 1d with length same as yNew, weight
            corresponding to each xNew, yNew row
    """
    if x.shape[1] != len(bins):
        message = "Dimensions of x and bins don't match: {} != {}".format(
            x.shape[1], len(bins)
        )
        raise ValueError(message)
    else:
        size = np.prod(bins)
        tupleBins = tuple(bins)
        xNew = np.zeros((size, x.shape[1]), dtype=float)
        yNew = np.zeros(size, dtype=float)
        weightsNew = np.zeros(size, dtype=float)

        # iterate through each of the final bins, which consists of bins for each feature
        # for each, only retain entries falling in that bin
        for i in range(size):
            index = list(np.unravel_index(i, tupleBins))
            valid = np.ones(x.shape[0], dtype=bool)
            for j, v in enumerate(index):
                valid &= (
                    x[:, j] >= np.percentile(x[:, j], v * 100 / bins[j])
                ) & (x[:, j] < np.percentile(x[:, j], (v + 1) * 100 / bins[j]))
            if np.sum(valid) != 0:
                xNew[i, :] = np.average(
                    x[valid], axis=0, weights=weights[valid]
                )
                yNew[i] = np.average(y[valid], axis=0, weights=weights[valid])
                weightsNew[i] = np.sum(weights[valid])
        xNew = xNew[~(weightsNew == 0)]
        yNew = yNew[~(weightsNew == 0)]
        weightsNew = weightsNew[~(weightsNew == 0)]
        return xNew, yNew, weightsNew


def monotone_spline(
    x,
    y,
    weights,
    bins=None,
    lam=12,
    kap=1e7,
    incl_uncstr=False,
    show_plot=False,
    method="eilers",
    splines=None,
    plot_start=0,
    plot_end=100,
):
    """
    Args:
        method (string): 'eilers' or 'pygam'
        splines (None or array-like): for 'pygam' only (otherwise set None),
            number of splines used for each feature, if None use default
        plot_start/plot_end (number between 0, 100): for 'pygam' only if show_plot = True,
            start and end for percentile of data used in plot, can result in
            better visualizations if original data has strong outliers


    Returns:
        xNew (numpy array): 2d with second dimension m, first
            dimension is product of elements in bins, with each entry
            representative of bin across all the features
        yNew (numpy array): 1d with length same as first dimension
            of xWeight, weighted average of y's corresponding to each
            entry of xWeight
        weightsNew (numpy array): 1d with length same as yNew, weight
            corresponding to each xNew, yNew row
    """

    if method == "pygam":
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        if splines != None and len(splines) != x.shape[1]:
            err_msg = (
                " pygam method requires splines to be None or "
                + " same length as # of columns in x, "
                + str(len(splines))
                + " != "
                + str(x.shape[1])
            )
            raise ValueError(err_msg)

        # bin data
        if bins == None:
            x_binned, y_binned, weights_binned = x, y, weights
        else:
            x_binned, y_binned, weights_binned = avg_by_bin_multd(
                x, y, bins, weights
            )

        # setup pygam parameters- in addition to 's' spline terms, can also have 't' tensor
        # terms which are interactions between two variables. 't' terms also need monotonic constraints
        # to satisfy previous constraints, they actually impose stronger restriction

        if splines == None:
            tempCstr = s(0, constraints="monotonic_inc")
            for i in range(1, x_binned.shape[1]):
                tempCstr += s(i, constraints="monotonic_inc")
            tempUncstr = s(0)
            for i in range(1, x_binned.shape[1]):
                tempUncstr += s(i)
        else:
            tempCstr = s(0, constraints="monotonic_inc", n_splines=splines[0])
            for i in range(1, x_binned.shape[1]):
                tempCstr += s(
                    i, constraints="monotonic_inc", n_splines=splines[i]
                )
            tempUncstr = s(0, n_splines=splines[0])
            for i in range(1, x_binned.shape[1]):
                tempUncstr += s(i, n_splines=splines[i])

        # fit data
        gamCstr = LinearGAM(tempCstr).fit(x_binned, y_binned, weights_binned)
        y_cstr = gamCstr.predict(x_binned)
        wsse_cstr = (weights_binned * ((y_cstr - y_binned) ** 2)).sum()
        if incl_uncstr:
            gamUncstr = LinearGAM(tempUncstr).fit(
                x_binned, y_binned, weights_binned
            )
            y_uncstr = gamUncstr.predict(x_binned)
            wsse_uncstr = (weights_binned * ((y_uncstr - y_binned) ** 2)).sum()
        else:
            y_uncstr = None
            wsse_uncstr = None

        if show_plot:
            if x.shape[1] == 2:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                # select data in [plot_start, end] percentile across both features
                # this can be rewritten to generalize for n-dimensions, but didn't know how to plot that
                xactPlot = x[
                    (x[:, 0] >= np.percentile(x[:, 0], plot_start))
                    & (x[:, 0] <= np.percentile(x[:, 0], plot_end))
                    & (x[:, 1] >= np.percentile(x[:, 1], plot_start))
                    & (x[:, 1] <= np.percentile(x[:, 1], plot_end))
                ]
                yactPlot = y[
                    (x[:, 0] >= np.percentile(x[:, 0], plot_start))
                    & (x[:, 0] <= np.percentile(x[:, 0], plot_end))
                    & (x[:, 1] >= np.percentile(x[:, 1], plot_start))
                    & (x[:, 1] <= np.percentile(x[:, 1], plot_end))
                ]
                ax.scatter(
                    xactPlot[:, 0],
                    xactPlot[:, 1],
                    yactPlot,
                    color="black",
                    s=0.8,
                    alpha=0.25,
                )

                x0 = np.linspace(
                    np.percentile(x[:, 0], plot_start),
                    np.percentile(x[:, 0], plot_end),
                    1000,
                )
                x1 = np.linspace(
                    np.percentile(x[:, 1], plot_start),
                    np.percentile(x[:, 1], plot_end),
                    1000,
                )
                X0, X1 = np.meshgrid(x0, x1)
                yPred = gamCstr.predict(
                    np.array([X0.flatten(), X1.flatten()]).T
                )
                ax.plot_surface(
                    X0, X1, yPred.reshape(x0.shape[0], -1), color="red"
                )
                ax.set_label("Monotonic GAM spline with all data")

            if x.shape[1] == 1:
                plt.scatter(
                    x,
                    y,
                    linestyle="None",
                    color="gray",
                    s=0.8,
                    alpha=0.7,
                    label="All data",
                )
                plt.plot(
                    x,
                    y_cstr,
                    color="red",
                    alpha=1.0,
                    label="Monotonic GAM spline",
                )
                if incl_uncstr:
                    plt.plot(
                        x,
                        y_uncstr,
                        color="blue",
                        alpha=1.0,
                        label="Unconstrained GAM spline",
                    )
            plt.show()
            plt.close()

        def interp(x):
            return gamCstr.predict(x)

        return interp, y_cstr, wsse_cstr, y_uncstr, wsse_uncstr

    if method == "eilers":
        # create binned and weighted x and y data
        if bins:
            if not np.isscalar(bins):
                err_msg = (
                    "monotone_spline2 ERROR: bins value is not type scalar"
                )
                raise ValueError(err_msg)
            N = int(bins)
            x_binned, y_binned, weights_binned = utils.avg_by_bin(
                x, y, weights, N
            )

        elif not bins:
            N = len(x)
            x_binned = x
            y_binned = y
            weights_binned = weights

        # Prepare bases (Imat) and penalty
        dd = 3
        E = np.eye(N)
        D3 = np.diff(E, n=dd, axis=0)
        D1 = np.diff(E, n=1, axis=0)

        # Monotone smoothing
        ws = np.zeros(N - 1)
        weights_binned = weights_binned.reshape(len(weights_binned), 1)
        weights1 = 0.5 * weights_binned[1:, :] + 0.5 * weights_binned[:-1, :]
        weights3 = (
            0.25 * weights_binned[3:, :]
            + 0.25 * weights_binned[2:-1, :]
            + 0.25 * weights_binned[1:-2, :]
            + 0.25 * weights_binned[:-3, :]
        ).flatten()

        for it in range(30):
            Ws = np.diag(ws * kap)
            mon_cof = np.linalg.solve(
                E
                + lam * D3.T @ np.diag(weights3) @ D3
                + D1.T @ (Ws * weights1) @ D1,
                y_binned,
            )
            ws_new = (D1 @ mon_cof < 0.0) * 1
            dw = np.sum(ws != ws_new)
            ws = ws_new
            if dw == 0:
                break

        # Monotonic and non monotonic fits
        y_cstr = mon_cof
        wsse_cstr = (weights_binned * ((y_cstr - y_binned) ** 2)).sum()
        if incl_uncstr:
            y_uncstr = np.linalg.solve(
                E + lam * D3.T @ np.diag(weights3) @ D3, y_binned
            )
            wsse_uncstr = (weights_binned * ((y_uncstr - y_binned) ** 2)).sum()
        else:
            y_uncstr = None
            wsse_uncstr = None

        def mono_interp(x_vec):
            # replace last point in data with two copies further out to make smooth
            # extrapolation
            x_new = np.append(
                x_binned[:-1], [1.005 * x_binned[-1], 1.01 * x_binned[-1]]
            )
            y_cstr_new = np.append(y_cstr[:-1], [y_cstr[-1], y_cstr[-1]])
            # Create interpolating cubic spline for interior points
            inter_interpl = intp(x_new, y_cstr_new, kind="cubic")
            y_pred = np.zeros_like(x_vec)
            x_lt_min = x_vec < x_binned.min()
            x_gt_max = x_vec > x_new.max()
            x_inter = (x_vec >= x_binned.min()) & (x_vec <= x_new.max())
            y_pred[x_inter] = inter_interpl(x_vec[x_inter])
            # extrapolate the maximum for values above the maximum
            y_pred[x_gt_max] = y_cstr[-1]
            # linear extrapolation of last two points for values below the min
            slope = (y_cstr[1] - y_cstr[0]) / (x_binned[1] - x_binned[0])
            intercept = y_cstr[0] - slope * x_binned[0]
            y_pred[x_lt_min] = slope * x_vec[x_lt_min] + intercept

            return y_pred

        if show_plot:
            plt.scatter(
                x,
                y,
                linestyle="None",
                color="gray",
                s=0.8,
                alpha=0.7,
                label="All data",
            )
            if not bins:
                plt.plot(
                    x,
                    y_cstr,
                    color="red",
                    alpha=1.0,
                    label="Monotonic smooth spline",
                )
                if incl_uncstr:
                    plt.plot(
                        x,
                        y_uncstr,
                        color="blue",
                        alpha=1.0,
                        label="Unconstrained smooth spline",
                    )
            else:
                plt.scatter(
                    x_binned,
                    y_binned,
                    linestyle="None",
                    color="black",
                    s=0.8,
                    alpha=0.7,
                    label="Binned data averages",
                )
                plt.plot(
                    x,
                    mono_interp(x),
                    color="red",
                    alpha=1.0,
                    label="Monotonic smooth spline",
                )
                if incl_uncstr:
                    plt.plot(
                        x_binned,
                        y_uncstr,
                        color="blue",
                        alpha=1.0,
                        label="Unconstrained smooth spline",
                    )
            plt.legend(loc="lower right")
            plt.show()
            plt.close()

        return mono_interp, y_cstr, wsse_cstr, y_uncstr, wsse_uncstr

    err_msg = method + " method not supported, must be 'eilers' or 'pygam'"
    raise ValueError(err_msg)
