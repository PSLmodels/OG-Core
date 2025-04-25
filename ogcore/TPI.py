"""
Time path iteration (TPI) module for OG-Core.

This module contains the following functions:
    get_initial_SS_values()
    firstdoughnutring()
    twist_doughnut()
    inner_loop()
    run_TPI()
"""

# imports
import numpy as np
import pickle
import scipy.optimize as opt
from dask import delayed, compute
import dask.multiprocessing
from ogcore import tax, utils, household, firm, fiscal
from ogcore import aggregates as aggr
from ogcore.constants import SHOW_RUNTIME
import os
import warnings
import logging


if not SHOW_RUNTIME:
    warnings.simplefilter("ignore", RuntimeWarning)

"""
Set minimizer tolerance
"""
MINIMIZER_TOL = 1e-13

"""
Set flag for enforcement of solution check
"""
ENFORCE_SOLUTION_CHECKS = True

"""
Set flag for verbosity
"""
VERBOSE = True
# Configure logging
log_level = logging.INFO if VERBOSE else logging.WARNING
logging.basicConfig(
    level=log_level, format="%(message)s"  # Only show the message itself
)


def get_initial_SS_values(p):
    """
    Get values of variables for the initial period and the steady state
    equilibrium values.

    Args:
        p (OG-Core Specifications object): model parameters

    Returns:
        (tuple): initial period and steady state values:

            * initial_values (tuple): initial period variable values,
                (b_sinit, b_splus1init, factor, initial_b, initial_n)
            * ss_vars (dictionary): dictionary with steady state
                solution results
            * theta (Numpy array): steady-state retirement replacement
                rates, length J
            * baseline_values (tuple): (Ybaseline, TRbaseline, Gbaseline,
                D0_baseline), GDP, lump sum transfer, and government spending
                amounts from the baseline model run

    """
    baseline_ss = os.path.join(p.baseline_dir, "SS", "SS_vars.pkl")
    ss_baseline_vars = utils.safe_read_pickle(baseline_ss)
    factor = ss_baseline_vars["factor"]
    B0 = aggr.get_B(ss_baseline_vars["b_sp1"], p, "SS", True)
    initial_b = ss_baseline_vars["b_sp1"] * (ss_baseline_vars["B"] / B0)
    initial_n = ss_baseline_vars["n"]

    Ybaseline = None
    TRbaseline = None
    Gbaseline = None
    Ig_baseline = None
    if not p.baseline:
        baseline_tpi = os.path.join(p.baseline_dir, "TPI", "TPI_vars.pkl")
        tpi_baseline_vars = utils.safe_read_pickle(baseline_tpi)
        Ybaseline = tpi_baseline_vars["Y"]
    if p.baseline_spending:
        baseline_tpi = os.path.join(p.baseline_dir, "TPI", "TPI_vars.pkl")
        tpi_baseline_vars = utils.safe_read_pickle(baseline_tpi)
        TRbaseline = tpi_baseline_vars["TR"]
        Gbaseline = tpi_baseline_vars["G"]
        Ig_baseline = tpi_baseline_vars["I_g"]

    if p.baseline:
        ss_vars = ss_baseline_vars
    else:
        reform_ss_path = os.path.join(p.output_base, "SS", "SS_vars.pkl")
        ss_vars = utils.safe_read_pickle(reform_ss_path)
    theta = ss_vars["theta"]

    """
    ------------------------------------------------------------------------
    Set other parameters and initial values
    ------------------------------------------------------------------------
    """
    # Get an initial distribution of wealth with the initial population
    # distribution. When small_open=True, the value of K0 is used as a
    # placeholder for first-period wealth
    B0 = aggr.get_B(initial_b, p, "SS", True)

    b_sinit = np.array(
        list(np.zeros(p.J).reshape(1, p.J)) + list(initial_b[:-1])
    )
    b_splus1init = initial_b

    # Intial gov't debt and capital stock must match that in the baseline
    if not p.baseline:
        baseline_tpi = os.path.join(p.baseline_dir, "TPI", "TPI_vars.pkl")
        tpi_baseline_vars = utils.safe_read_pickle(baseline_tpi)
        D0_baseline = tpi_baseline_vars["D"][0]
        Kg0_baseline = tpi_baseline_vars["K_g"][0]
    else:
        RM0_baseline = None
        D0_baseline = None
        Kg0_baseline = None

    initial_values = (B0, b_sinit, b_splus1init, factor, initial_b, initial_n)
    baseline_values = (
        Ybaseline,
        TRbaseline,
        Gbaseline,
        Ig_baseline,
        D0_baseline,
        Kg0_baseline,
    )

    return initial_values, ss_vars, theta, baseline_values


def firstdoughnutring(
    guesses, r, w, p_tilde, bq, rm, tr, theta, factor, ubi, j, initial_b, p
):
    """
    Solves the first entries of the upper triangle of the twist doughnut. This
    is separate from the main TPI function because the values of b and n are
    scalars, so it is easier to just have a separate function for these cases.

    Args:
        guesses (Numpy array): initial guesses for b and n, length 2
        r (scalar): real interest rate
        w (scalar): real wage rate
        p_tilde (scalar): composite good price
        bq (scalar): bequest amounts by age
        rm (scalar): remittance amounts by age
        tr (scalar): government transfer amount
        theta (Numpy array): retirement replacement rates, length J
        factor (scalar): scaling factor converting model units to dollars
        ubi (scalar): individual UBI credit to household s=E+S of type j in
            period 0
        j (int): index of ability type
        initial_b (Numpy array): SxJ matrix, savings of agents alive at T=0
        p (OG-Core Specifications object): model parameters

    Returns:
        euler errors (Numpy array): errors from first order conditions,
            length 2

    """
    b_splus1 = float(guesses[0])
    n = float(guesses[1])
    b_s = float(initial_b[-2, j])

    # Find errors from FOC for savings and FOC for labor supply
    error1 = household.FOC_savings(
        np.array([r]),
        np.array([w]),
        np.array([p_tilde]),
        b_s,
        np.array([b_splus1]),
        np.array([n]),
        np.array([bq]),
        np.array([rm]),
        factor,
        np.array([tr]),
        np.array([ubi]),
        theta[j],
        p.rho[0, -1],
        p.etr_params[0][-1],
        p.mtry_params[0][-1],
        None,
        j,
        p,
        "TPI_scalar",
    )

    error2 = household.FOC_labor(
        np.array([r]),
        np.array([w]),
        np.array([p_tilde]),
        b_s,
        b_splus1,
        np.array([n]),
        np.array([bq]),
        np.array([rm]),
        factor,
        np.array([tr]),
        np.array([ubi]),
        theta[j],
        p.chi_n[0, -1],
        p.etr_params[0][-1],
        p.mtrx_params[0][-1],
        None,
        j,
        p,
        "TPI_scalar",
    )

    if n <= 0 or n >= 1:
        error2 += 1e12
    if b_splus1 <= 0:
        error1 += 1e12

    return [np.squeeze(error1)] + [np.squeeze(error2)]


def twist_doughnut(
    guesses,
    r,
    w,
    p_tilde,
    bq,
    rm,
    tr,
    theta,
    factor,
    ubi,
    j,
    s,
    t,
    etr_params,
    mtrx_params,
    mtry_params,
    initial_b,
    p,
):
    """
    Solves the upper triangle of time path iterations. These are the agents who
    are alive at time T=0 so that we do not solve for their full lifetime (so
    of their life was before the model begins).

    Args:
        guesses (list): initial guesses for b and n, length 2s
        r (Numpy array): real interest rate
        w (Numpy array): real wage rate
        p_tilde (Numpy array): composite good price
        bq (Numpy array): bequest amounts by age, length s
        rm (Numpy array): remittance amounts by age, length s
        tr (Numpy array): government transfer amount
        theta (Numpy array): retirement replacement rates, length J
        factor (scalar): scaling factor converting model units to dollars
        ubi (Numpy array): length remaining periods of life UBI payout to household
        j (int): index of ability type
        s (int): years of life remaining
        t (int): model period
        etr_params (list): ETR function parameters,
            list of lists with size = sxsxnum_params
        mtrx_params (list): labor income MTR function parameters,
            list of lists with size = sxsxnum_params
        mtry_params (list): capital income MTR function
            parameters, lists of lists with size = sxsxnum_params
        initial_b (Numpy array): savings of agents alive at T=0,
            size = SxJ
        p (OG-Core Specifications object): model parameters

    Returns:
        euler errors (Numpy array): errors from first order conditions,
            length 2s

    """
    length = int(len(guesses) / 2)
    b_guess = np.array(guesses[:length])
    n_guess = np.array(guesses[length:])

    if length == p.S:
        b_s = np.array([0] + list(b_guess[:-1]))
    else:
        b_s = np.array([initial_b[-(s + 3), j]] + list(b_guess[:-1]))

    b_splus1 = b_guess
    w_s = w[t : t + length]
    r_s = r[t : t + length]
    p_tilde_s = p_tilde[t : t + length]
    n_s = n_guess
    chi_n_s = np.diag(p.chi_n[t : t + p.S, :], max(p.S - length, 0))
    rho_s = np.diag(p.rho[t : t + p.S, :], max(p.S - length, 0))

    error1 = household.FOC_savings(
        r_s,
        w_s,
        p_tilde_s,
        b_s,
        b_splus1,
        n_s,
        bq,
        rm,
        factor,
        tr,
        ubi,
        theta,
        rho_s,
        etr_params,
        mtry_params,
        t,
        j,
        p,
        "TPI",
    )

    error2 = household.FOC_labor(
        r_s,
        w_s,
        p_tilde_s,
        b_s,
        b_splus1,
        n_s,
        bq,
        rm,
        factor,
        tr,
        ubi,
        theta,
        chi_n_s,
        etr_params,
        mtrx_params,
        t,
        j,
        p,
        "TPI",
    )

    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] += 1e12
    mask2 = n_guess > p.ltilde
    error2[mask2] += 1e12
    mask4 = b_guess <= 0
    error2[mask4] += 1e12
    mask5 = b_splus1 < 0
    error2[mask5] += 1e12

    return list(error1.flatten()) + list(error2.flatten())


def inner_loop(guesses, outer_loop_vars, initial_values, ubi, j, ind, p):
    """
    Given path of economic aggregates and factor prices, solves
    household problem.  This has been termed the inner-loop (in
    contrast to the outer fixed point loop that solves for GE factor
    prices and economic aggregates).

    Args:
        guesses (tuple): initial guesses for b and n, (guesses_b,
            guesses_n)
        outer_loop_vars (tuple): values for factor prices and economic
            aggregates used in household problem
            (r_p, r, w, p_m, BQ, RM, TR, theta)
        r_p (Numpy array): real interest rate on household portfolio
        r (Numpy array): real interest rate on private capital
        w (Numpy array): real wage rate
        p_m (Numpy array): output goods prices
        BQ (array_like): aggregate bequest amounts
        TR (Numpy array): lump sum transfer amount
        theta (Numpy array): retirement replacement rates, length J
        initial_values (tuple): initial period variable values,
            (b_sinit, b_splus1init, factor, initial_b, initial_n,
            D0_baseline)
        ubi (array_like): T+S x S x J array time series of UBI transfers in
            model units for each type-j age-s household in every period t
        j (int): index of ability type
        ind (Numpy array): integers from 0 to S-1
        p (OG-Core Specifications object): model parameters

    Returns:
        (tuple): household solution results:

            * euler_errors (Numpy array): errors from FOCs, size = Tx2S
            * b_mat (Numpy array): savings amounts, size = TxS
            * n_mat (Numpy array): labor supply amounts, size = TxS

    """
    (K0, b_sinit, b_splus1init, factor, initial_b, initial_n) = initial_values
    guesses_b, guesses_n = guesses
    r_p, r, w, p_m, BQ, RM, TR, theta = outer_loop_vars

    # compute composite good price
    p_i = (
        np.tile(p.io_matrix.reshape(1, p.I, p.M), (p.T + p.S, 1, 1))
        * np.tile(p_m.reshape(p.T + p.S, 1, p.M), (1, p.I, 1))
    ).sum(axis=2)
    p_tilde = aggr.get_ptilde(p_i[:, :], p.tau_c[:, :], p.alpha_c, "TPI")
    # compute bq
    bq = household.get_bq(BQ, None, p, "TPI")
    # compute tr
    tr = household.get_tr(TR, None, p, "TPI")
    # compute rm
    rm = household.get_rm(RM, None, p, "TPI")

    # initialize arrays
    b_mat = np.zeros((p.T + p.S, p.S))
    n_mat = np.zeros((p.T + p.S, p.S))
    euler_errors = np.zeros((p.T, 2 * p.S))

    solutions = opt.root(
        firstdoughnutring,
        [guesses_b[0, -1], guesses_n[0, -1]],
        args=(
            r_p[0],
            w[0],
            p_tilde[0],
            bq[0, -1, j],
            rm[0, -1, j],
            tr[0, -1, j],
            theta * p.replacement_rate_adjust[0],
            factor,
            ubi[0, -1, j],
            j,
            initial_b,
            p,
        ),
        method=p.FOC_root_method,
        tol=MINIMIZER_TOL,
    )
    b_mat[0, -1], n_mat[0, -1] = solutions.x[0], solutions.x[1]

    for s in range(p.S - 2):  # Upper triangle
        ind2 = np.arange(s + 2)
        b_guesses_to_use = np.diag(guesses_b[: p.S, :], p.S - (s + 2))
        n_guesses_to_use = np.diag(guesses_n[: p.S, :], p.S - (s + 2))
        theta_to_use = theta[j] * p.replacement_rate_adjust[: p.S]
        bq_to_use = np.diag(bq[: p.S, :, j], p.S - (s + 2))
        rm_to_use = np.diag(rm[: p.S, :, j], p.S - (s + 2))
        tr_to_use = np.diag(tr[: p.S, :, j], p.S - (s + 2))
        ubi_to_use = np.diag(ubi[: p.S, :, j], p.S - (s + 2))

        num_params = len(p.etr_params[0][0])
        temp_etr = [
            [p.etr_params[t][p.S - s - 2 + t][i] for i in range(num_params)]
            for t in range(s + 2)
        ]
        etr_params_to_use = [
            [temp_etr[i][j] for j in range(num_params)] for i in range(s + 2)
        ]
        temp_mtrx = [
            [p.mtrx_params[t][p.S - s - 2 + t][i] for i in range(num_params)]
            for t in range(s + 2)
        ]
        mtrx_params_to_use = [
            [temp_mtrx[i][j] for j in range(num_params)] for i in range(s + 2)
        ]
        temp_mtry = [
            [p.mtry_params[t][p.S - s - 2 + t][i] for i in range(num_params)]
            for t in range(s + 2)
        ]
        mtry_params_to_use = [
            [temp_mtry[i][j] for j in range(num_params)] for i in range(s + 2)
        ]

        solutions = opt.root(
            twist_doughnut,
            list(b_guesses_to_use) + list(n_guesses_to_use),
            args=(
                r_p,
                w,
                p_tilde,
                bq_to_use,
                rm_to_use,
                tr_to_use,
                theta_to_use,
                factor,
                ubi_to_use,
                j,
                s,
                0,
                etr_params_to_use,
                mtrx_params_to_use,
                mtry_params_to_use,
                initial_b,
                p,
            ),
            method=p.FOC_root_method,
            tol=MINIMIZER_TOL,
        )

        b_vec = solutions.x[: int(len(solutions.x) / 2)]
        b_mat[ind2, p.S - (s + 2) + ind2] = b_vec
        n_vec = solutions.x[int(len(solutions.x) / 2) :]
        n_mat[ind2, p.S - (s + 2) + ind2] = n_vec

    for t in range(0, p.T):
        b_guesses_to_use = 0.75 * np.diag(guesses_b[t : t + p.S, :])
        n_guesses_to_use = np.diag(guesses_n[t : t + p.S, :])
        theta_to_use = theta[j] * p.replacement_rate_adjust[t : t + p.S]
        bq_to_use = np.diag(bq[t : t + p.S, :, j])
        rm_to_use = np.diag(rm[t : t + p.S, :, j])
        tr_to_use = np.diag(tr[t : t + p.S, :, j])
        ubi_to_use = np.diag(ubi[t : t + p.S, :, j])

        # initialize array of diagonal elements
        num_params = len(p.etr_params[t][0])
        etr_params_to_use = [
            [p.etr_params[t + s][s][i] for i in range(num_params)]
            for s in range(p.S)
        ]
        mtrx_params_to_use = [
            [p.mtrx_params[t + s][s][i] for i in range(num_params)]
            for s in range(p.S)
        ]
        mtry_params_to_use = [
            [p.mtry_params[t + s][s][i] for i in range(num_params)]
            for s in range(p.S)
        ]

        solutions = opt.root(
            twist_doughnut,
            list(b_guesses_to_use) + list(n_guesses_to_use),
            args=(
                r_p,
                w,
                p_tilde,
                bq_to_use,
                rm_to_use,
                tr_to_use,
                theta_to_use,
                factor,
                ubi_to_use,
                j,
                None,
                t,
                etr_params_to_use,
                mtrx_params_to_use,
                mtry_params_to_use,
                initial_b,
                p,
            ),
            method=p.FOC_root_method,
            tol=MINIMIZER_TOL,
        )
        euler_errors[t, :] = solutions.fun

        b_vec = solutions.x[: p.S]
        b_mat[t + ind, ind] = b_vec
        n_vec = solutions.x[p.S :]
        n_mat[t + ind, ind] = n_vec

    # print('Type ', j, ' max euler error = ',
    #       np.absolute(euler_errors).max())

    return euler_errors, b_mat, n_mat


def run_TPI(p, client=None):
    """
    Solve for transition path equilibrium of OG-Core.

    Args:
        p (OG-Core Specifications object): model parameters
        client (Dask client object): client

    Returns:
        output (dictionary): dictionary with transition path solution
            results

    """
    # unpack tuples of parameters
    initial_values, ss_vars, theta, baseline_values = get_initial_SS_values(p)
    (B0, b_sinit, b_splus1init, factor, initial_b, initial_n) = initial_values
    (
        Ybaseline,
        TRbaseline,
        Gbaseline,
        Ig_baseline,
        D0_baseline,
        Kg0_baseline,
    ) = baseline_values

    # Create time path of UBI household benefits and aggregate UBI outlays
    ubi = p.ubi_nom_array / factor
    UBI = aggr.get_L(ubi[: p.T], p, "TPI")

    logging.info(
        f"Government spending breakpoints are tG1: {p.tG1}; and tG2: {p.tG2}"
    )

    # Initialize guesses at time paths
    # Make array of initial guesses for labor supply and savings
    guesses_b = utils.get_initial_path(initial_b, ss_vars["b_sp1"], p, "ratio")
    guesses_n = utils.get_initial_path(initial_n, ss_vars["n"], p, "ratio")
    b_mat = guesses_b
    n_mat = guesses_n
    ind = np.arange(p.S)

    # Get path for aggregate savings and labor supply
    L_init = np.ones((p.T + p.S,)) * ss_vars["L"]
    B_init = np.ones((p.T + p.S,)) * ss_vars["B"]
    L_init[: p.T] = aggr.get_L(n_mat[: p.T], p, "TPI")
    B_init[1 : p.T] = aggr.get_B(b_mat[: p.T], p, "TPI", False)[: p.T - 1]
    B_init[0] = B0
    K_init = B_init * ss_vars["K"] / ss_vars["B"]
    K = K_init
    K_d = K_init * ss_vars["K_d"] / ss_vars["K"]
    K_f = K_init * ss_vars["K_f"] / ss_vars["K"]
    L = L_init
    B = B_init
    K_g = np.ones_like(K) * ss_vars["K_g"]
    Y = np.zeros_like(K)
    Y[: p.T] = firm.get_Y(K[: p.T], K_g[: p.T], L[: p.T], p, "TPI")
    Y[p.T :] = ss_vars["Y"]
    # path for industry specific aggregates
    K_vec_init = np.ones((p.T + p.S, p.M)) * ss_vars["K_m"].reshape(1, p.M)
    L_vec_init = np.ones((p.T + p.S, p.M)) * ss_vars["L_m"].reshape(1, p.M)
    Y_vec_init = np.ones((p.T + p.S, p.M)) * ss_vars["Y_m"].reshape(1, p.M)
    # compute w
    w = np.ones_like(K) * ss_vars["w"]
    # compute goods prices
    p_m = np.ones((p.T + p.S, p.M)) * ss_vars["p_m"].reshape(1, p.M)
    p_m[: p.T, :] = firm.get_pm(
        w[: p.T], Y_vec_init[: p.T, :], L_vec_init[: p.T, :], p, "TPI"
    )
    p_m = p_m / p_m[:, -1].reshape(
        p.T + p.S, 1
    )  # normalize prices by industry M
    p_i = (
        np.tile(p.io_matrix.reshape(1, p.I, p.M), (p.T + p.S, 1, 1))
        * np.tile(p_m.reshape(p.T + p.S, 1, p.M), (1, p.I, 1))
    ).sum(axis=2)
    p_tilde = aggr.get_ptilde(p_i[:, :], p.tau_c[:, :], p.alpha_c, "TPI")
    if not any(p.zeta_K == 1):
        w[: p.T] = np.squeeze(
            firm.get_w(Y[: p.T], L[: p.T], p_m[: p.T, :], p, "TPI")
        )
    # repeat with updated w
    p_m[: p.T, :] = firm.get_pm(
        w[: p.T], Y_vec_init[: p.T, :], L_vec_init[: p.T, :], p, "TPI"
    )
    p_m = p_m / p_m[:, -1].reshape(
        p.T + p.S, 1
    )  # normalize prices by industry M
    p_i = (
        np.tile(p.io_matrix.reshape(1, p.I, p.M), (p.T + p.S, 1, 1))
        * np.tile(p_m.reshape(p.T + p.S, 1, p.M), (1, p.I, 1))
    ).sum(axis=2)
    p_tilde = aggr.get_ptilde(p_i[:, :], p.tau_c[:, :], p.alpha_c, "TPI")
    # path for interest rates
    r = np.zeros_like(Y)
    r[: p.T] = np.squeeze(
        firm.get_r(Y[: p.T], K[: p.T], p_m[: p.T, :], p, "TPI")
    )
    r[p.T :] = ss_vars["r"]
    # For case where economy is small open econ
    r[p.zeta_K == 1] = p.world_int_rate[p.zeta_K == 1]

    # initial guesses at fiscal vars
    if p.budget_balance:
        if np.abs(ss_vars["TR"]) < 1e-13:
            TR_ss2 = 0.0  # sometimes SS is very small but not zero,
            # even if taxes are zero, this get's rid of the
            # approximation error, which affects the pct changes below
        else:
            TR_ss2 = ss_vars["TR"]
        TR = np.ones(p.T + p.S) * TR_ss2
        total_tax_revenue = TR - ss_vars["agg_pension_outlays"]
        G = np.zeros(p.T + p.S)
        D = np.zeros(p.T + p.S)
        D_d = np.zeros(p.T + p.S)
        D_f = np.zeros(p.T + p.S)
        I_g = fiscal.get_I_g(Y[: p.T], None, p, "TPI")
    else:
        if p.baseline_spending:
            # Will set to TRbaseline here, but will be updated in TPI loop
            # with call to fiscal.get_TR
            TR = np.concatenate(
                (TRbaseline[: p.T], np.ones(p.S) * ss_vars["TR"])
            )
            # Will set to Ig_baseline here, but will be updated in TPI loop
            # with call to fiscal.get_I_g
            I_g = np.concatenate(
                (Ig_baseline[: p.T], np.ones(p.S) * ss_vars["I_g"])
            )
            # Will set to Gbaseline here, but will be updated in TPI loop
            # with call to fiscal.D_G_path, which also does closure rule
            G = np.concatenate((Gbaseline[: p.T], np.ones(p.S) * ss_vars["G"]))
        else:
            TR = p.alpha_T * Y
            G = np.ones(p.T + p.S) * ss_vars["G"]
            I_g = np.ones(p.T + p.S) * ss_vars["I_g"]
        D = np.ones(p.T + p.S) * ss_vars["D"]
        D_d = D * ss_vars["D_d"] / ss_vars["D"]
        D_f = D * ss_vars["D_f"] / ss_vars["D"]
    if p.baseline:
        K_g0 = p.initial_Kg_ratio * Y[0]
    else:
        K_g0 = Kg0_baseline
    K_g = fiscal.get_K_g(K_g0, I_g, p, "TPI")
    total_tax_revenue = np.ones(p.T + p.S) * ss_vars["total_tax_revenue"]

    # Compute other interest rates
    r_gov = fiscal.get_r_gov(r, p, "TPI")
    r_p = np.ones_like(r) * ss_vars["r_p"]
    MPKg = np.zeros((p.T, p.M))
    for m in range(p.M):
        MPKg[:, m] = np.squeeze(
            firm.get_MPx(
                Y_vec_init[: p.T, m], K_g[: p.T], p.gamma_g[m], p, "TPI", m
            )
        )
    r_p[: p.T] = aggr.get_r_p(
        r[: p.T],
        r_gov[: p.T],
        p_m[: p.T, :],
        K_vec_init[: p.T, :],
        K_g[: p.T],
        D[: p.T],
        MPKg,
        p,
        "TPI",
    )

    # Initialize bequests
    BQ0 = aggr.get_BQ(r_p[0], initial_b, None, p, "SS", True)
    if not p.use_zeta:
        BQ = np.zeros((p.T + p.S, p.J))
        for j in range(p.J):
            BQ[:, j] = (
                list(np.linspace(BQ0[j], ss_vars["BQ"][j], p.T))
                + [ss_vars["BQ"][j]] * p.S
            )
        BQ = np.array(BQ)
    else:
        BQ = list(np.linspace(BQ0, ss_vars["BQ"], p.T)) + [ss_vars["BQ"]] * p.S
        BQ = np.array(BQ)

    # Initialize aggregate remittances
    if p.baseline:
        RM = aggr.get_RM(Y, p, "TPI")
    else:
        # This is the reform case and is based off of Ybaseline, but allows for
        # remittance parameters to change in a reform and update the RM series
        Ybaseline_ext = np.concatenate(
            [Ybaseline, np.ones(p.S) * Ybaseline[-1]]
        )
        RM = aggr.get_RM(Ybaseline_ext, p, "TPI")

    # Start transition path iteration (TPI)
    TPIiter = 0
    TPIdist = 10
    euler_errors = np.zeros((p.T, 2 * p.S, p.J))
    TPIdist_vec = np.zeros(p.maxiter)

    # TPI loop
    while (TPIiter < p.maxiter) and (TPIdist >= p.mindist_TPI):
        outer_loop_vars = (r_p, r, w, p_m, BQ, RM, TR, theta)
        # compute composite good price
        p_i = (
            np.tile(p.io_matrix.reshape(1, p.I, p.M), (p.T + p.S, 1, 1))
            * np.tile(p_m.reshape(p.T + p.S, 1, p.M), (1, p.I, 1))
        ).sum(axis=2)
        p_tilde = aggr.get_ptilde(p_i[:, :], p.tau_c[:, :], p.alpha_c, "TPI")

        # scatter parameters to workers
        scattered_p = client.scatter(p, broadcast=True) if client else p

        euler_errors = np.zeros((p.T, 2 * p.S, p.J))
        lazy_values = []
        for j in range(p.J):
            guesses = (guesses_b[:, :, j], guesses_n[:, :, j])

            # Add the delayed computation to our list
            lazy_values.append(
                delayed(inner_loop)(
                    guesses,
                    outer_loop_vars,
                    initial_values,
                    ubi,
                    j,
                    ind,
                    scattered_p,
                )
            )
        if client:
            # Compute all the values
            futures = client.compute(lazy_values)
            # Later, gather the results when needed
            results = client.gather(futures)
        else:
            results = compute(
                *lazy_values,
                scheduler=dask.multiprocessing.get,
                num_workers=p.num_workers,
            )

        for j, result in enumerate(results):
            euler_errors[:, :, j], b_mat[:, :, j], n_mat[:, :, j] = result

        bmat_s = np.zeros((p.T, p.S, p.J))
        bmat_s[0, 1:, :] = initial_b[:-1, :]
        bmat_s[1:, 1:, :] = b_mat[: p.T - 1, :-1, :]
        bmat_splus1 = np.zeros((p.T, p.S, p.J))
        bmat_splus1[:, :, :] = b_mat[: p.T, :, :]

        num_params = len(p.etr_params[0][0])
        etr_params_4D = [
            [
                [
                    [p.etr_params[t][s][i] for i in range(num_params)]
                    for j in range(p.J)
                ]
                for s in range(p.S)
            ]
            for t in range(p.T)
        ]

        bqmat = household.get_bq(BQ, None, p, "TPI")
        rmmat = household.get_rm(RM, None, p, "TPI")
        trmat = household.get_tr(TR, None, p, "TPI")
        tax_mat = tax.net_taxes(
            r_p[: p.T],
            w[: p.T],
            bmat_s,
            n_mat[: p.T, :, :],
            bqmat[: p.T, :, :],
            factor,
            trmat[: p.T, :, :],
            ubi[: p.T, :, :],
            theta,
            0,
            None,
            False,
            "TPI",
            p.e,
            etr_params_4D,
            p,
        )
        r_p_path = utils.to_timepath_shape(r_p)
        p_tilde_path = utils.to_timepath_shape(p_tilde)
        wpath = utils.to_timepath_shape(w)
        c_mat = household.get_cons(
            r_p_path[: p.T, :, :],
            wpath[: p.T, :, :],
            p_tilde_path[: p.T, :, :],
            bmat_s,
            bmat_splus1,
            n_mat[: p.T, :, :],
            bqmat[: p.T, :, :],
            rmmat[: p.T, :, :],
            tax_mat,
            p.e,
            p,
        )
        C = aggr.get_C(c_mat, p, "TPI")

        c_i = household.get_ci(
            c_mat[: p.T, :, :],
            p_i[: p.T, :],
            p_tilde[: p.T],
            p.tau_c[: p.T, :],
            p.alpha_c,
            "TPI",
        )
        y_before_tax_mat = household.get_y(
            r_p_path[: p.T, :, :],
            wpath[: p.T, :, :],
            bmat_s[: p.T, :, :],
            n_mat[: p.T, :, :],
            p,
            "TPI",
        )

        L[: p.T] = aggr.get_L(n_mat[: p.T], p, "TPI")
        B[1 : p.T] = aggr.get_B(bmat_splus1[: p.T], p, "TPI", False)[: p.T - 1]
        w_open = firm.get_w_from_r(p.world_int_rate[: p.T], p, "TPI")

        # Find output, labor demand, capital demand for M-1 industries
        L_vec = np.zeros((p.T, p.M))
        K_vec = np.zeros((p.T, p.M))
        C_vec = np.zeros((p.T, p.I))
        K_demand_open_vec = np.zeros((p.T, p.M))
        for i_ind in range(p.I):
            C_vec[:, i_ind] = aggr.get_C(c_i[: p.T, i_ind, :, :], p, "TPI")
        Y_vec = (
            np.tile(p.io_matrix.reshape(1, p.I, p.M), (p.T, 1, 1))
            * np.tile(C_vec[: p.T, :].reshape(p.T, p.I, 1), (1, 1, p.M))
        ).sum(axis=1)
        for m_ind in range(p.M - 1):
            KYrat_m = firm.get_KY_ratio(
                r[: p.T], p_m[: p.T, :], p, "TPI", m_ind
            )
            K_vec[:, m_ind] = KYrat_m * Y_vec[:, m_ind]
            L_vec[:, m_ind] = firm.solve_L(
                Y_vec[:, m_ind], K_vec[:, m_ind], K_g, p, "TPI", m_ind
            )
            K_demand_open_vec[:, m_ind] = firm.get_K(
                p.world_int_rate[: p.T],
                w_open[: p.T],
                L_vec[: p.T, m_ind],
                p,
                "TPI",
                m_ind,
            )

        # Find output, labor demand, capital demand for last industry
        L_M = np.maximum(
            np.ones(p.T) * 0.001, L[: p.T] - L_vec[: p.T, :].sum(-1)
        )  # make sure L_M > 0
        K_demand_open_vec[:, -1] = firm.get_K(
            p.world_int_rate[: p.T], w_open[: p.T], L_M[: p.T], p, "TPI", -1
        )
        K[: p.T], K_d[: p.T], K_f[: p.T] = aggr.get_K_splits(
            B[: p.T],
            K_demand_open_vec[: p.T, :].sum(-1),
            D_d[: p.T],
            p.zeta_K[: p.T],
        )
        K_M = np.maximum(
            np.ones(p.T) * 0.001, K[: p.T] - K_vec[: p.T, :].sum(-1)
        )  # make sure K_M > 0

        L_vec[:, -1] = L_M
        K_vec[:, -1] = K_M
        Y_vec[:, -1] = firm.get_Y(
            K_vec[: p.T, -1], K_g[: p.T], L_vec[: p.T, -1], p, "TPI", -1
        )

        Y = (p_m[: p.T, :] * Y_vec[: p.T, :]).sum(-1)

        (
            total_tax_rev,
            iit_payroll_tax_revenue,
            agg_pension_outlays,
            UBI_outlays,
            bequest_tax_revenue,
            wealth_tax_revenue,
            cons_tax_revenue,
            business_tax_revenue,
            payroll_tax_revenue,
            iit_revenue,
        ) = aggr.revenue(
            r_p[: p.T],
            w[: p.T],
            bmat_s,
            n_mat[: p.T, :, :],
            bqmat[: p.T, :, :],
            c_i[: p.T, :, :, :],
            Y_vec[: p.T, :],
            L_vec[: p.T, :],
            K_vec[: p.T, :],
            p_m[: p.T, :],
            factor,
            ubi[: p.T, :, :],
            theta,
            etr_params_4D,
            p.e,
            p,
            None,
            "TPI",
        )
        total_tax_revenue[: p.T] = total_tax_rev
        dg_fixed_values = (
            Y,
            total_tax_revenue,
            agg_pension_outlays,
            UBI_outlays,
            TR,
            I_g,
            Gbaseline,
            D0_baseline,
        )
        (
            Dnew,
            G[: p.T],
            D_d[: p.T],
            D_f[: p.T],
            new_borrowing,
            debt_service,
            new_borrowing_f,
        ) = fiscal.D_G_path(r_gov, dg_fixed_values, p)
        K[: p.T], K_d[: p.T], K_f[: p.T] = aggr.get_K_splits(
            B[: p.T], K_demand_open_vec.sum(-1), D_d[: p.T], p.zeta_K[: p.T]
        )
        if not p.baseline_spending:
            I_g = fiscal.get_I_g(Y[: p.T], None, p, "TPI")
        if p.baseline:
            K_g0 = p.initial_Kg_ratio * Y[0]
        K_g = fiscal.get_K_g(K_g0, I_g, p, "TPI")
        rnew = r.copy()
        rnew[: p.T] = np.squeeze(
            firm.get_r(
                Y_vec[: p.T, -1], K_vec[: p.T, -1], p_m[: p.T, :], p, "TPI", -1
            )
        )
        # For case where economy is small open econ
        rnew[p.zeta_K == 1] = p.world_int_rate[p.zeta_K == 1]
        r_gov_new = fiscal.get_r_gov(rnew, p, "TPI")
        MPKg_vec = np.zeros((p.T, p.M))
        for m in range(p.M):
            MPKg_vec[:, m] = np.squeeze(
                firm.get_MPx(
                    Y_vec[: p.T, m], K_g[: p.T], p.gamma_g[m], p, "TPI", m
                )
            )
        r_p_new = aggr.get_r_p(
            rnew[: p.T],
            r_gov_new[: p.T],
            p_m[: p.T, :],
            K_vec[: p.T, :],
            K_g[: p.T],
            Dnew[: p.T],
            MPKg_vec,
            p,
            "TPI",
        )

        # compute w
        wnew = np.squeeze(
            firm.get_w(
                Y_vec[: p.T, -1], L_vec[: p.T, -1], p_m[: p.T, :], p, "TPI", -1
            )
        )

        # compute new prices
        new_p_m = firm.get_pm(wnew, Y_vec, L_vec, p, "TPI")
        new_p_m = new_p_m / new_p_m[:, -1].reshape(
            p.T, 1
        )  # normalize prices by industry M

        b_mat_shift = np.append(
            np.reshape(initial_b, (1, p.S, p.J)),
            b_mat[: p.T - 1, :, :],
            axis=0,
        )
        BQnew = aggr.get_BQ(r_p_new[: p.T], b_mat_shift, None, p, "TPI", False)
        bqmat_new = household.get_bq(BQnew, None, p, "TPI")
        (
            total_tax_rev,
            iit_payroll_tax_revenue,
            agg_pension_outlays,
            UBI_outlays,
            bequest_tax_revenue,
            wealth_tax_revenue,
            cons_tax_revenue,
            business_tax_revenue,
            payroll_tax_revenue,
            iit_revenue,
        ) = aggr.revenue(
            r_p_new[: p.T],
            wnew[: p.T],
            bmat_s,
            n_mat[: p.T, :, :],
            bqmat_new[: p.T, :, :],
            c_i[: p.T, :, :, :],
            Y_vec[: p.T, :],
            L_vec[: p.T, :],
            K_vec[: p.T, :],
            new_p_m[: p.T, :],
            factor,
            ubi[: p.T, :, :],
            theta,
            etr_params_4D,
            p.e,
            p,
            None,
            "TPI",
        )
        total_tax_revenue[: p.T] = total_tax_rev
        TR_new = fiscal.get_TR(
            Y[: p.T],
            TR[: p.T],
            G[: p.T],
            total_tax_revenue[: p.T],
            agg_pension_outlays[: p.T],
            UBI_outlays[: p.T],
            I_g[: p.T],
            p,
            "TPI",
        )
        RM = aggr.get_RM(Y[: p.T], p, "TPI")
        RM = np.concatenate([RM, np.ones(p.S) * RM[-1]])

        # update vars for next iteration
        w[: p.T] = utils.convex_combo(wnew[: p.T], w[: p.T], p.nu)
        r[: p.T] = utils.convex_combo(rnew[: p.T], r[: p.T], p.nu)
        r_gov[: p.T] = utils.convex_combo(r_gov_new[: p.T], r_gov[: p.T], p.nu)
        r_p[: p.T] = utils.convex_combo(r_p_new[: p.T], r_p[: p.T], p.nu)
        p_m[: p.T, :] = utils.convex_combo(
            new_p_m[: p.T, :], p_m[: p.T, :], p.nu
        )
        BQ[: p.T] = utils.convex_combo(BQnew[: p.T], BQ[: p.T], p.nu)
        D[: p.T] = Dnew[: p.T]
        if not p.baseline_spending:
            TR[: p.T] = utils.convex_combo(TR_new[: p.T], TR[: p.T], p.nu)
        guesses_b = utils.convex_combo(b_mat, guesses_b, p.nu)
        guesses_n = utils.convex_combo(n_mat, guesses_n, p.nu)
        logging.info(
            f"w diff: {(wnew[: p.T] - w[: p.T]).max()}, "
            + f"{(wnew[: p.T] - w[: p.T]).min()}"
        )
        logging.info(
            f"r diff: {(rnew[: p.T] - r[: p.T]).max()}, "
            + f"{(rnew[: p.T] - r[: p.T]).min()}"
        )
        logging.info(
            f"r_p diff: {(r_p_new[: p.T] - r_p[: p.T]).max()}, "
            + f"{(r_p_new[: p.T] - r_p[: p.T]).min()}"
        )
        logging.info(
            f"p_m diff: {(new_p_m[: p.T, :] - p_m[: p.T, :]).max()}, "
            + f"{(new_p_m[: p.T, :] - p_m[: p.T, :]).min()}"
        )
        logging.info(
            f"BQ diff: {(BQnew[: p.T] - BQ[: p.T]).max()}, "
            + f"{(BQnew[: p.T] - BQ[: p.T]).min()}"
        )
        logging.info(
            f"TR diff: {(TR_new[: p.T] - TR[: p.T]).max()}, "
            + f"{(TR_new[: p.T] - TR[: p.T]).min()}"
        )

        TPIdist = np.array(
            list(utils.pct_diff_func(r_p_new[: p.T], r_p[: p.T]))
            + list(utils.pct_diff_func(rnew[: p.T], r[: p.T]))
            + list(utils.pct_diff_func(wnew[: p.T], w[: p.T]))
            + list(
                utils.pct_diff_func(new_p_m[: p.T, :], p_m[: p.T, :]).flatten()
            )
            + list(utils.pct_diff_func(BQnew[: p.T], BQ[: p.T]).flatten())
            + list(utils.pct_diff_func(TR_new[: p.T], TR[: p.T]))
        ).max()

        TPIdist_vec[TPIiter] = TPIdist
        # After T=10, if cycling occurs, drop the value of nu
        # wait til after T=10 or so, because sometimes there is a jump up
        # in the first couple iterations
        # if TPIiter > 10:
        #     if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter - 1] > 0:
        #         nu /= 2
        #         print 'New Value of nu:', nu
        TPIiter += 1
        logging.info(f"Iteration: {TPIiter}")
        logging.info(f"\tDistance: {TPIdist}")

    # Compute effective and marginal tax rates for all agents
    num_params = len(p.mtrx_params[0][0])
    etr_params_4D = [
        [
            [
                [p.etr_params[t][s][i] for i in range(num_params)]
                for j in range(p.J)
            ]
            for s in range(p.S)
        ]
        for t in range(p.T)
    ]
    mtrx_params_4D = [
        [
            [
                [p.mtrx_params[t][s][i] for i in range(num_params)]
                for j in range(p.J)
            ]
            for s in range(p.S)
        ]
        for t in range(p.T)
    ]
    mtry_params_4D = [
        [
            [
                [p.mtry_params[t][s][i] for i in range(num_params)]
                for j in range(p.J)
            ]
            for s in range(p.S)
        ]
        for t in range(p.T)
    ]
    labor_noncompliance_rate_3D = np.tile(
        np.reshape(
            p.labor_income_tax_noncompliance_rate[: p.T, :], (p.T, 1, p.J)
        ),
        (1, p.S, 1),
    )
    capital_noncompliance_rate_3D = np.tile(
        np.reshape(
            p.capital_income_tax_noncompliance_rate[: p.T, :], (p.T, 1, p.J)
        ),
        (1, p.S, 1),
    )
    e_3D = p.e
    mtry_path = tax.MTR_income(
        r_p_path[: p.T],
        wpath[: p.T],
        bmat_s[: p.T, :, :],
        n_mat[: p.T, :, :],
        factor,
        True,
        e_3D,
        etr_params_4D,
        mtry_params_4D,
        capital_noncompliance_rate_3D,
        p,
    )
    mtrx_path = tax.MTR_income(
        r_p_path[: p.T],
        wpath[: p.T],
        bmat_s[: p.T, :, :],
        n_mat[: p.T, :, :],
        factor,
        False,
        e_3D,
        etr_params_4D,
        mtrx_params_4D,
        labor_noncompliance_rate_3D,
        p,
    )
    etr_path = tax.ETR_income(
        r_p_path[: p.T],
        wpath[: p.T],
        bmat_s[: p.T, :, :],
        n_mat[: p.T, :, :],
        factor,
        e_3D,
        etr_params_4D,
        labor_noncompliance_rate_3D,
        capital_noncompliance_rate_3D,
        p,
    )

    # Note that implicitly in this computation is that immigrants'
    # wealth is all in the form of private capital
    I_d = aggr.get_I(
        bmat_splus1[: p.T], K_d[1 : p.T + 1], K_d[: p.T], p, "TPI"
    )
    I = aggr.get_I(bmat_splus1[: p.T], K[1 : p.T + 1], K[: p.T], p, "TPI")
    # solve resource constraint
    # foreign debt service costs
    debt_service_f = fiscal.get_debt_service_f(r_p, D_f)
    net_capital_outflows = aggr.get_capital_outflows(
        r_p[: p.T],
        K_f[: p.T],
        new_borrowing_f[: p.T],
        debt_service_f[: p.T],
        p,
    )
    # Fill in arrays, noting that M-1 industries only produce consumption goods
    G_vec = np.zeros((p.T, p.M))
    G_vec[:, -1] = G[: p.T]
    # Map consumption goods back to demands for production goods
    C_m_vec = (
        np.tile(p.io_matrix.reshape(1, p.I, p.M), (p.T, 1, 1))
        * np.tile(C_vec[: p.T, :].reshape(p.T, p.I, 1), (1, 1, p.M))
    ).sum(axis=1)
    I_d_vec = np.zeros((p.T, p.M))
    I_d_vec[:, -1] = I_d[: p.T]
    I_g_vec = np.zeros((p.T, p.M))
    I_g_vec[:, -1] = I_g[: p.T]
    net_capital_outflows_vec = np.zeros((p.T, p.M))
    net_capital_outflows_vec[:, -1] = net_capital_outflows[: p.T]
    RM_vec = np.zeros((p.T, p.M))
    RM_vec[:, -1] = RM[: p.T]
    RC_error = aggr.resource_constraint(
        Y_vec,
        C_m_vec,
        G_vec,
        I_d_vec,
        I_g_vec,
        net_capital_outflows_vec,
        RM_vec,
    )
    # Compute total investment (not just domestic)
    I_total = aggr.get_I(None, K[1 : p.T + 1], K[: p.T], p, "total_tpi")

    # Compute resource constraint error
    rce_max = np.amax(np.abs(RC_error))
    logging.info(f"Max absolute value resource constraint error: {rce_max}")

    logging.info("Checking time path for violations of constraints.")
    for t in range(p.T):
        household.constraint_checker_TPI(
            b_mat[t], n_mat[t], c_mat[t], t, p.ltilde
        )

    eul_savings = euler_errors[:, : p.S, :]
    eul_laborleisure = euler_errors[:, p.S :, :]

    logging.info(f"Max Euler error, savings: {np.abs(eul_savings).max()}")
    logging.info(
        f"Max Euler error labor supply: {np.abs(eul_laborleisure).max()}"
    )

    """
    ------------------------------------------------------------------------
    Save variables/values so they can be used in other modules
    ------------------------------------------------------------------------
    """

    output = {
        "Y": Y[: p.T, ...],
        "B": B[: p.T, ...],
        "K": K[: p.T, ...],
        "K_f": K_f[: p.T],
        "K_d": K_d[: p.T, ...],
        "L": L[: p.T, ...],
        "C": C[: p.T, ...],
        "I": I[: p.T, ...],
        "I_total": I_total[: p.T, ...],
        "I_d": I_d[: p.T, ...],
        "K_g": K_g[: p.T, ...],
        "I_g": I_g[: p.T, ...],
        "BQ": BQ[: p.T, ...],
        "RM": RM[: p.T, ...],
        "Y_m": Y_vec[: p.T, ...],
        "K_m": K_vec[: p.T, ...],
        "L_m": L_vec[: p.T, ...],
        "C_i": C_vec[: p.T, ...],
        "TR": TR[: p.T, ...],
        "agg_pension_outlays": agg_pension_outlays[: p.T, ...],
        "G": G[: p.T, ...],
        "UBI": UBI[: p.T, ...],
        "total_tax_revenue": total_tax_revenue[: p.T, ...],
        "business_tax_revenue": business_tax_revenue[: p.T, ...],
        "iit_payroll_tax_revenue": iit_payroll_tax_revenue[: p.T, ...],
        "iit_revenue": iit_revenue[: p.T, ...],
        "payroll_tax_revenue": payroll_tax_revenue[: p.T, ...],
        "bequest_tax_revenue": bequest_tax_revenue[: p.T, ...],
        "wealth_tax_revenue": wealth_tax_revenue[: p.T, ...],
        "cons_tax_revenue": cons_tax_revenue[: p.T, ...],
        "D": D[: p.T, ...],
        "D_f": D_f[: p.T, ...],
        "D_d": D_d[: p.T, ...],
        "new_borrowing": new_borrowing[: p.T, ...],
        "debt_service": debt_service[: p.T, ...],
        "new_borrowing_f": new_borrowing_f[: p.T, ...],
        "debt_service_f": debt_service_f[: p.T, ...],
        "r": r[: p.T, ...],
        "r_gov": r_gov[: p.T, ...],
        "r_p": r_p[: p.T, ...],
        "w": w[: p.T, ...],
        "p_m": p_m[: p.T, ...],
        "p_i": p_i[: p.T, ...],
        "p_tilde": p_tilde[: p.T, ...],
        "b_sp1": bmat_splus1[: p.T, ...],
        "b_s": bmat_s[: p.T, ...],
        "n": n_mat[: p.T, ...],
        "c": c_mat[: p.T, ...],
        "c_i": c_i[: p.T, ...],
        "bq": bqmat[: p.T, ...],
        "rm": rmmat[: p.T, ...],
        "tr": trmat[: p.T, ...],
        "ubi": ubi[: p.T, ...],
        "before_tax_income": y_before_tax_mat[: p.T, ...],
        "hh_taxes": tax_mat[: p.T, ...],
        "etr": etr_path[: p.T, ...],
        "mtrx": mtrx_path[: p.T, ...],
        "mtry": mtry_path[: p.T, ...],
        "euler_savings": eul_savings[: p.T, ...],
        "euler_labor_leisure": eul_laborleisure[: p.T, ...],
        "resource_constraint_error": RC_error[: p.T, ...],
    }

    tpi_dir = os.path.join(p.output_base, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
    with open(tpi_vars, "wb") as f:
        pickle.dump(output, f)

    if np.any(G) < 0:
        logging.warning(
            "Government spending is negative along transition path"
            + " to satisfy budget"
        )

    if (
        (TPIiter >= p.maxiter) or (np.absolute(TPIdist) > p.mindist_TPI)
    ) and ENFORCE_SOLUTION_CHECKS:
        raise RuntimeError(
            "Transition path equlibrium not found" + " (TPIdist)"
        )

    if (np.any(np.absolute(RC_error) >= p.RC_TPI)) and ENFORCE_SOLUTION_CHECKS:
        raise RuntimeError(
            "Transition path equlibrium not found " + "(RC_error)"
        )

    if (
        np.any(np.absolute(eul_savings) >= p.mindist_TPI)
        or (np.any(np.absolute(eul_laborleisure) > p.mindist_TPI))
    ) and ENFORCE_SOLUTION_CHECKS:
        raise RuntimeError(
            "Transition path equlibrium not found " + "(eulers)"
        )

    return output
