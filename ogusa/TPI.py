from __future__ import print_function
'''
------------------------------------------------------------------------
This program solves for transition path of the distribution of wealth
and the aggregate capital stock using the time path iteration (TPI)
method, where labor in inelastically supplied.

This py-file calls the following other file(s):
            tax.py
            utils.py
            household.py
            firm.py
            OUTPUT/SS/ss_vars.pkl


This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/TPIinit/TPIinit_vars.pkl
            OUTPUT/TPI/TPI_vars.pkl
------------------------------------------------------------------------
'''

# Packages
import numpy as np
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle
import scipy.optimize as opt
from dask.distributed import Client
from dask import compute, delayed
import dask.multiprocessing
from . import tax
from . import utils
from . import household
from . import firm
from . import fiscal
from . import aggregates as aggr
import os


'''
Set minimizer tolerance
'''
MINIMIZER_TOL = 1e-13

'''
Set flag for enforcement of solution check
'''
ENFORCE_SOLUTION_CHECKS = True


'''
------------------------------------------------------------------------
Import steady state distribution, parameters and other objects from
steady state computation in ss_vars.pkl
------------------------------------------------------------------------
'''


def get_initial_SS_values(p):

    '''
    ------------------------------------------------------------------------
    Get values of variables for the initial period and the steady state.
    ------------------------------------------------------------------------
    '''
    baseline_ss = os.path.join(p.baseline_dir, "SS/SS_vars.pkl")
    ss_baseline_vars = pickle.load(open(baseline_ss, "rb"))
    factor = ss_baseline_vars['factor_ss']
    initial_b = ss_baseline_vars['bssmat_splus1']
    initial_n = ss_baseline_vars['nssmat']
    T_Hbaseline = None
    Gbaseline = None
    if p.baseline_spending:
        baseline_tpi = os.path.join(p.baseline_dir, "TPI/TPI_vars.pkl")
        tpi_baseline_vars = pickle.load(open(baseline_tpi, "rb"))
        T_Hbaseline = tpi_baseline_vars['T_H']
        Gbaseline = tpi_baseline_vars['G']

    baseline_values = (T_Hbaseline, Gbaseline)

    if p.baseline:
        theta = tax.replacement_rate_vals(
            ss_baseline_vars['nssmat'], ss_baseline_vars['wss'], factor,
            None, p)
        SS_values = (ss_baseline_vars['Kss'], ss_baseline_vars['Bss'],
                     ss_baseline_vars['Lss'], ss_baseline_vars['rss'],
                     ss_baseline_vars['wss'], ss_baseline_vars['BQss'],
                     ss_baseline_vars['T_Hss'],
                     ss_baseline_vars['revenue_ss'],
                     ss_baseline_vars['bssmat_splus1'],
                     ss_baseline_vars['nssmat'],
                     ss_baseline_vars['Yss'], ss_baseline_vars['Gss'],
                     theta)

    elif not p.baseline:
        reform_ss = os.path.join(p.output_base, "SS/SS_vars.pkl")
        ss_reform_vars = pickle.load(open(reform_ss, "rb"))
        theta = tax.replacement_rate_vals(
            ss_reform_vars['nssmat'], ss_reform_vars['wss'], factor,
            None, p)
        SS_values = (ss_reform_vars['Kss'], ss_reform_vars['Bss'],
                     ss_reform_vars['Lss'], ss_reform_vars['rss'],
                     ss_reform_vars['wss'], ss_reform_vars['BQss'],
                     ss_reform_vars['T_Hss'],
                     ss_reform_vars['revenue_ss'],
                     ss_reform_vars['bssmat_splus1'],
                     ss_reform_vars['nssmat'], ss_reform_vars['Yss'],
                     ss_reform_vars['Gss'], theta)

    ## What is going on here?  Whatever it is, why not done in pb_api.py???
    N_tilde = p.omega.sum(1)  # this should equal one in
    # each year given how we've constructed omega
    p.omega = p.omega / N_tilde.reshape(p.T + p.S, 1)

    '''
    ------------------------------------------------------------------------
    Set other parameters and initial values
    ------------------------------------------------------------------------
    '''
    # Get an initial distribution of wealth with the initial population
    # distribution. When small_open=True, the value of K0 is used as a
    # placeholder for first-period wealth
    B0 = aggr.get_K(initial_b, p, 'SS', True)

    b_sinit = np.array(list(np.zeros(p.J).reshape(1, p.J)) +
                       list(initial_b[:-1]))
    b_splus1init = initial_b

    # Intial gov't debt must match that in the baseline
    initial_debt = p.initial_debt
    if not p.baseline:
        baseline_tpi = os.path.join(p.baseline_dir, "TPI/TPI_vars.pkl")
        tpi_baseline_vars = pickle.load(open(baseline_tpi, "rb"))
        D0 = tpi_baseline_vars['D'][0]
    else:
        D0 = 0.0

    initial_values = (B0, b_sinit, b_splus1init, factor, initial_b,
                      initial_n, initial_debt, D0)

    return initial_values, SS_values, baseline_values


def firstdoughnutring(guesses, r, w, BQ, T_H, theta, factor, j,
                      initial_b, p):
    '''
    Solves the first entries of the upper triangle of the twist
    doughnut.  This is separate from the main TPI function because the
    values of b and n are scalars, so it is easier to just have a
    separate function for these cases.
    Inputs:
        guesses = guess for b and n (2x1 list)
        winit = initial wage rate (scalar)
        rinit = initial rental rate (scalar)
        BQinit = initial aggregate bequest (scalar)
        T_H_init = initial lump sum tax (scalar)
        initial_b = initial distribution of capital (SxJ array)
        factor = steady state scaling factor (scalar)
        j = which ability type is being solved for (integer)
        parameters = tuple of parameters (tuple)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rates (Jx1 array)
    Output:
        euler errors (2x1 list)
    '''
    b_splus1 = float(guesses[0])
    n = float(guesses[1])
    b_s = float(initial_b[-2, j])

    # Find errors from FOC for savings and FOC for labor supply
    # Notes: 1) using method = "SS" below because just for one period
    # 2) Set retire to true for agents in last period of life

    error1 = household.FOC_savings(np.array([r]), np.array([w]), b_s,
                                   np.array([b_splus1]), np.array([n]),
                                   np.array([BQ]), factor,
                                   np.array([T_H]), theta[j],
                                   p.e[-1, j], p.rho[-1], 0,
                                   p.etr_params[0, -1, :],
                                   p.mtry_params[0, -1, :], j, p, 'SS')

    error2 = household.FOC_labor(
        np.array([r]), np.array([w]), b_s, b_splus1, np.array([n]),
        np.array([BQ]), factor, np.array([T_H]), theta[j], p.chi_n[-1],
        p.e[-1, j], 0, p.etr_params[0, -1, :], p.mtrx_params[0, -1, :],
        j, p, 'SS')

    if n <= 0 or n >= 1:
        error2 += 1e12
    if b_splus1 <= 0:
        error1 += 1e12
    return [np.squeeze(error1)] + [np.squeeze(error2)]


def twist_doughnut(guesses, r, w, BQ, T_H, theta, factor, j, s, t, etr_params,
                   mtrx_params, mtry_params, initial_b, p):
    '''
    Parameters:
        guesses = distribution of capital and labor (various length list)
        w   = wage rate ((T+S)x1 array)
        r   = rental rate ((T+S)x1 array)
        BQ = aggregate bequests ((T+S)x1 array)
        T_H = lump sum tax over time ((T+S)x1 array)
        factor = scaling factor (scalar)
        j = which ability type is being solved for (scalar)
        s = which upper triangle loop is being solved for (scalar)
        t = which diagonal is being solved for (scalar)
        params = list of parameters (list)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortalit rate (Sx1 array)
        lambdas = ability weights (Jx1 array)
        e = ability type (SxJ array)
        initial_b = capital stock distribution in period 0 (SxJ array)
        chi_b = chi^b_j (Jx1 array)
        chi_n = chi^n_s (Sx1 array)
    Output:
        Value of Euler error (various length list)
    '''
    length = int(len(guesses) / 2)
    b_guess = np.array(guesses[:length])
    n_guess = np.array(guesses[length:])

    if length == p.S:
        b_s = np.array([0] + list(b_guess[:-1]))
    else:
        b_s = np.array([(initial_b[-(s + 3), j])] + list(b_guess[:-1]))

    b_splus1 = b_guess
    w_s = w[t:t + length]
    r_s = r[t:t + length]
    n_s = n_guess
    chi_n_s = p.chi_n[-length:]
    e_s = p.e[-length:, j]
    rho_s = p.rho[-length:]
    BQ_s = BQ[t:t + length]
    T_H_s = T_H[t:t + length]

    error1 = household.FOC_savings(r_s, w_s, b_s, b_splus1, n_s, BQ_s,
                                   factor, T_H_s, theta, e_s, rho_s,
                                   p.retire, etr_params,
                                   mtry_params, j, p, 'TPI')

    error2 = household.FOC_labor(r_s, w_s, b_s, b_splus1, n_s, BQ_s,
                                 factor, T_H_s, theta, chi_n_s, e_s,
                                 p.retire, etr_params, mtrx_params, j,
                                 p, 'TPI')

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


def inner_loop(guesses, outer_loop_vars, initial_values, j, ind, p):
    '''
    Solves inner loop of TPI.  Given path of economic aggregates and
    factor prices, solves household problem.

    Inputs:
        r          = [T,] vector, interest rate
        w          = [T,] vector, wage rate
        b          = [T,S,J] array, wealth holdings
        n          = [T,S,J] array, labor supply
        BQ         = [T,J] vector,  bequest amounts
        factor     = scalar, model income scaling factor
        T_H        = [T,] vector, lump sum transfer amount(s)

    Functions called:
        firstdoughnutring()
        twist_doughnut()

    Objects in function:

    Returns: euler_errors, b_mat, n_mat
    '''
    #unpack variables and parameters pass to function
    (K0, b_sinit, b_splus1init, factor, initial_b, initial_n,
     initial_debt, D0) = initial_values
    guesses_b, guesses_n = guesses
    r, K, BQ, T_H, theta = outer_loop_vars

    # compute w
    w = firm.get_w_from_r(r, p)

    # initialize arrays
    b_mat = np.zeros((p.T + p.S, p.S))
    n_mat = np.zeros((p.T + p.S, p.S))
    euler_errors = np.zeros((p.T, 2 * p.S))

    b_mat[0, -1], n_mat[0, -1] =\
        np.array(opt.fsolve(firstdoughnutring, [guesses_b[0, -1],
                                                guesses_n[0, -1]],
                            args=(r[0], w[0], BQ[0, j], T_H[0], theta,
                                  factor, j, initial_b, p),
                            xtol=MINIMIZER_TOL))

    for s in range(p.S - 2):  # Upper triangle
        ind2 = np.arange(s + 2)
        b_guesses_to_use = np.diag(guesses_b[:p.S, :], p.S - (s + 2))
        n_guesses_to_use = np.diag(guesses_n[:p.S, :], p.S - (s + 2))

        length_diag =\
            np.diag(p.etr_params[:p.S, :, 0], p.S-(s + 2)).shape[0]
        etr_params_to_use = np.zeros((length_diag, p.etr_params.shape[2]))
        mtrx_params_to_use = np.zeros((length_diag, p.mtrx_params.shape[2]))
        mtry_params_to_use = np.zeros((length_diag, p.mtry_params.shape[2]))
        for i in range(p.etr_params.shape[2]):
            etr_params_to_use[:, i] =\
                np.diag(p.etr_params[:p.S, :, i], p.S - (s + 2))
            mtrx_params_to_use[:, i] =\
                np.diag(p.mtrx_params[:p.S, :, i], p.S - (s + 2))
            mtry_params_to_use[:, i] =\
                np.diag(p.mtry_params[:p.S, :, i], p.S - (s + 2))

        solutions = opt.fsolve(twist_doughnut,
                               list(b_guesses_to_use) +
                               list(n_guesses_to_use),
                               args=(r, w, BQ[:, j], T_H, theta, factor,
                                     j, s, 0, etr_params_to_use,
                                     mtrx_params_to_use,
                                     mtry_params_to_use, initial_b, p),
                               xtol=MINIMIZER_TOL)

        b_vec = solutions[:int(len(solutions) / 2)]
        b_mat[ind2, p.S - (s + 2) + ind2] = b_vec
        n_vec = solutions[int(len(solutions) / 2):]
        n_mat[ind2, p.S - (s + 2) + ind2] = n_vec

    for t in range(0, p.T):
        b_guesses_to_use = .75 * \
            np.diag(guesses_b[t:t + p.S, :])
        n_guesses_to_use = np.diag(guesses_n[t:t + p.S, :])

        # initialize array of diagonal elements
        etr_params_TP = np.zeros((p.T + p.S, p.S,  p.etr_params.shape[2]))
        etr_params_TP[:p.T, :, :] = p.etr_params
        etr_params_TP[p.T:, :, :] = p.etr_params[-1, :, :]

        mtrx_params_TP = np.zeros((p.T + p.S, p.S,  p.mtrx_params.shape[2]))
        mtrx_params_TP[:p.T, :, :] = p.mtrx_params
        mtrx_params_TP[p.T:, :, :] = p.mtrx_params[-1, :, :]

        mtry_params_TP = np.zeros((p.T + p.S, p.S,  p.mtry_params.shape[2]))
        mtry_params_TP[:p.T, :, :] = p.mtry_params
        mtry_params_TP[p.T:, :, :] = p.mtry_params[-1, :, :]

        length_diag =\
            np.diag(etr_params_TP[t:t + p.S, :, 0]).shape[0]
        etr_params_to_use = np.zeros((length_diag, p.etr_params.shape[2]))
        mtrx_params_to_use = np.zeros((length_diag, p.mtrx_params.shape[2]))
        mtry_params_to_use = np.zeros((length_diag, p.mtry_params.shape[2]))

        for i in range(p.etr_params.shape[2]):
            etr_params_to_use[:, i] = np.diag(etr_params_TP[t:t + p.S, :, i])
            mtrx_params_to_use[:, i] = np.diag(mtrx_params_TP[t:t + p.S, :, i])
            mtry_params_to_use[:, i] = np.diag(mtry_params_TP[t:t + p.S, :, i])
        #
        # TPI_solver_params = (inc_tax_params_TP, tpi_params, None)
        [solutions, infodict, ier, message] =\
            opt.fsolve(twist_doughnut, list(b_guesses_to_use) +
                       list(n_guesses_to_use),
                       args=(r, w, BQ[:, j], T_H, theta, factor, j,
                             None, t, etr_params_to_use,
                             mtrx_params_to_use, mtry_params_to_use,
                             initial_b, p),
                       xtol=MINIMIZER_TOL, full_output=True)
        euler_errors[t, :] = infodict['fvec']

        b_vec = solutions[:p.S]
        b_mat[t + ind, ind] = b_vec
        n_vec = solutions[p.S:]
        n_mat[t + ind, ind] = n_vec

    print('Type ', j, ' max euler error = ', euler_errors.max())

    return euler_errors, b_mat, n_mat


def run_TPI(p, client=None):

    # unpack tuples of parameters
    initial_values, SS_values, baseline_values = get_initial_SS_values(p)
    (B0, b_sinit, b_splus1init, factor, initial_b, initial_n,
     initial_debt, D0) = initial_values
    (Kss, Bss, Lss, rss, wss, BQss, T_Hss, revenue_ss, bssmat_splus1,
     nssmat, Yss, Gss, theta) = SS_values
    (T_Hbaseline, Gbaseline) = baseline_values

    print('Government spending breakpoints are tG1: ', p.tG1,
          '; and tG2:', p.tG2)

    TPI_FIG_DIR = p.output_base
    # Initialize guesses at time paths
    # Make array of initial guesses for labor supply and savings
    domain = np.linspace(0, p.T, p.T)
    domain2 = np.tile(domain.reshape(p.T, 1, 1), (1, p.S, p.J))
    ending_b = bssmat_splus1
    guesses_b = (-1 / (domain2 + 1)) * (ending_b - initial_b) + ending_b
    ending_b_tail = np.tile(ending_b.reshape(1, p.S, p.J), (p.S, 1, 1))
    guesses_b = np.append(guesses_b, ending_b_tail, axis=0)

    domain3 = np.tile(np.linspace(0, 1, p.T).reshape(p.T, 1, 1), (1, p.S, p.J))
    guesses_n = domain3 * (nssmat - initial_n) + initial_n
    ending_n_tail = np.tile(nssmat.reshape(1, p.S, p.J), (p.S, 1, 1))
    guesses_n = np.append(guesses_n, ending_n_tail, axis=0)
    b_mat = guesses_b  # np.zeros((p.T + p.S, p.S, p.J))
    n_mat = guesses_n  # np.zeros((p.T + p.S, p.S, p.J))
    ind = np.arange(p.S)

    L_init = np.ones((p.T + p.S,)) * Lss
    B_init = np.ones((p.T + p.S,)) * Bss
    L_init[:p.T] = aggr.get_L(n_mat[:p.T], p, 'TPI')
    B_init[1:p.T] = aggr.get_K(b_mat[:p.T], p, 'TPI', False)[:p.T - 1]
    B_init[0] = B0

    if not p.small_open:
        if p.budget_balance:
            K_init = B_init
        else:
            K_init = B_init * Kss / Bss
    else:
        K_init = firm.get_K(L_init, p.tpi_firm_r, p)

    K = K_init

    L = L_init
    B = B_init
    Y = firm.get_Y(K, L, p)
    if not p.small_open:
        r = firm.get_r(Y, K, p)
    else:
        r = p.tpi_hh_r
    # compute w
    w = firm.get_w_from_r(r, p)

    BQ = np.zeros((p.T + p.S, p.J))
    BQ0 = aggr.get_BQ(r[0], initial_b, None, p, 'SS', True)
    for j in range(p.J):
        BQ[:, j] = list(np.linspace(BQ0[j], BQss[j], p.T)) + [BQss[j]] * p.S
    BQ = np.array(BQ)
    if p.budget_balance:
        if np.abs(T_Hss) < 1e-13:
            T_Hss2 = 0.0  # sometimes SS is very small but not zero,
            # even if taxes are zero, this get's rid of the approximation
            # error, which affects the perc changes below
        else:
            T_Hss2 = T_Hss
        T_H = np.ones(p.T + p.S) * T_Hss2
        REVENUE = T_H
        G = np.zeros(p.T + p.S)
    elif not p.baseline_spending:
        T_H = p.ALPHA_T * Y
    elif p.baseline_spending:
        T_H = T_Hbaseline
        T_H_new = p.T_H   # Need to set T_H_new for later reference
        G = Gbaseline
        G_0 = Gbaseline[0]

    # Initialize some inputs
    if p.budget_balance:
        D = 0.0 * Y
    else:
        D = p.debt_ratio_ss * Y

    TPIiter = 0
    TPIdist = 10
    PLOT_TPI = False
    report_tG1 = False

    euler_errors = np.zeros((p.T, 2 * p.S, p.J))
    TPIdist_vec = np.zeros(p.maxiter)

    print('analytical mtrs in tpi = ', p.analytical_mtrs)
    print('tax function type in tpi = ', p.tax_func_type)

    # TPI loop
    while (TPIiter < p.maxiter) and (TPIdist >= p.mindist_TPI):
        # Plot TPI for K for each iteration, so we can see if there is a
        # problem
        if PLOT_TPI is True:
            # K_plot = list(K) + list(np.ones(10) * Kss)
            D_plot = list(D) + list(np.ones(10) * Yss * p.debt_ratio_ss)
            plt.figure()
            plt.axhline(y=Kss, color='black', linewidth=2,
                        label=r"Steady State $\hat{K}$", ls='--')
            plt.plot(np.arange(p.T + 10), D_plot[:p.T + 10], 'b',
                     linewidth=2, label=r"TPI time path $\hat{K}_t$")
            plt.savefig(os.path.join(TPI_FIG_DIR, "TPI_D"))

        if report_tG1 is True:
            print('\tAt time tG1-1:')
            print('\t\tG = ', G[p.tG1 - 1])
            print('\t\tK = ', K[p.tG1 - 1])
            print('\t\tr = ', r[p.tG1 - 1])
            print('\t\tD = ', D[p.tG1 - 1])

        outer_loop_vars = (r, K, BQ, T_H, theta)
        # inner_loop_params = (income_tax_params, tpi_params,
        #                      initial_values, ind)

        euler_errors = np.zeros((p.T, 2 * p.S, p.J))
        lazy_values = []
        for j in range(p.J):
            guesses = (guesses_b[:, :, j], guesses_n[:, :, j])
            lazy_values.append(
                delayed(inner_loop)(guesses, outer_loop_vars,
                                    initial_values, j, ind, p))
        results = compute(*lazy_values, get=dask.multiprocessing.get,
                          num_workers=p.num_workers)
        for j, result in enumerate(results):
            euler_errors[:, :, j], b_mat[:, :, j], n_mat[:, :, j] = result

        bmat_s = np.zeros((p.T, p.S, p.J))
        bmat_s[0, 1:, :] = initial_b[:-1, :]
        bmat_s[1:, 1:, :] = b_mat[:p.T-1, :-1, :]
        bmat_splus1 = np.zeros((p.T, p.S, p.J))
        bmat_splus1[:, :, :] = b_mat[:p.T, :, :]

        L[:p.T] = aggr.get_L(n_mat[:p.T], p, 'TPI')
        B[1:p.T] = aggr.get_K(bmat_splus1[:p.T], p, 'TPI',
                              False)[:p.T - 1]
        if np.any(B) < 0:
            print('B has negative elements. B[0:9]:', B[0:9])
            print('B[T-2:T]:', B[p.T - 2, p.T])

        etr_parms4D = np.tile(
            p.etr_params.reshape(p.T, p.S, 1, p.etr_params.shape[2]),
            (1, 1, p.J, 1))
        BQ_3D = np.tile(BQ.reshape(BQ.shape[0], 1, BQ.shape[1]),
                        (1, p.S, 1))

        if not p.small_open:
            if p.budget_balance:
                K[:p.T] = B[:p.T]
            else:
                if not p.baseline_spending:
                    Y = T_H/p.ALPHA_T  # maybe unecessary

                REVENUE = np.array(list(
                    aggr.revenue(r[:p.T], w[:p.T], bmat_s,
                                 n_mat[:p.T, :, :], BQ_3D[:p.T, :, :],
                                 Y[:p.T], L[:p.T], K[:p.T], factor,
                                 theta, etr_parms4D, p, 'TPI')) +
                                   [revenue_ss] * p.S)

                # set intial debt value
                if p.baseline:
                    D_0 = p.initial_debt * Y[0]
                else:
                    D_0 = D0
                if not p.baseline_spending:
                    G_0 = p.ALPHA_G[0] * Y[0]
                dg_fixed_values = (Y, REVENUE, T_H, D_0, G_0)
                Dnew, G = fiscal.D_G_path(r, dg_fixed_values, Gbaseline,
                                          p)

                K[:p.T] = B[:p.T] - Dnew[:p.T]
                if np.any(K < 0):
                    print('K has negative elements. Setting them ' +
                          'positive to prevent NAN.')
                    K[:p.T] = np.fmax(K[:p.T], 0.05 * B[:p.T])
        else:
            K[:p.T] = firm.get_K(L[:p.T], p.tpi_firm_r[:p.T], p)
        Ynew = firm.get_Y(K[:p.T], L[:p.T], p)
        if not p.small_open:
            rnew = firm.get_r(Ynew[:p.T], K[:p.T], p)
        else:
            rnew = r.copy()
        # compute w
        wnew = firm.get_w_from_r(rnew[:p.T], p)

        b_mat_shift = np.append(np.reshape(initial_b, (1, p.S, p.J)),
                                b_mat[:p.T - 1, :, :], axis=0)
        BQnew = aggr.get_BQ(rnew[:p.T], b_mat_shift, None, p, 'TPI', False)
        BQnew_3D = np.tile(BQnew.reshape(BQnew.shape[0], 1, BQnew.shape[1]), (1, p.S, 1))
        REVENUE = np.array(list(
            aggr.revenue(rnew[:p.T], wnew[:p.T], bmat_s,
                         n_mat[:p.T, :, :], BQnew_3D[:p.T, :, :], Ynew[:p.T],
                         L[:p.T], K[:p.T], factor, theta, etr_parms4D,
                         p, 'TPI')) + [revenue_ss] * p.S)

        if p.budget_balance:
            T_H_new = REVENUE
        elif not p.baseline_spending:
            T_H_new = p.ALPHA_T[:p.T] * Ynew[:p.T]
        # If baseline_spending==True, no need to update T_H, it's fixed

        if p.small_open and not p.budget_balance:
            # Loop through years to calculate debt and gov't spending.
            # This is done earlier when small_open=False.
            if p.baseline:
                D_0 = p.initial_debt * Y[0]
            else:
                D_0 = D0
            if not p.baseline_spending:
                G_0 = p.ALPHA_G[0] * Ynew[0]
            dg_fixed_values = (Ynew, REVENUE, T_H, D_0, G_0)
            Dnew, G = fiscal.D_G_path(r, dg_fixed_values, Gbaseline, p)

        if p.budget_balance:
            Dnew = D

        w[:p.T] = wnew[:p.T]
        r[:p.T] = utils.convex_combo(rnew[:p.T], r[:p.T], p.nu)
        BQ[:p.T] = utils.convex_combo(BQnew[:p.T], BQ[:p.T], p.nu)
        D = Dnew
        Y[:p.T] = utils.convex_combo(Ynew[:p.T], Y[:p.T], p.nu)
        if not p.baseline_spending:
            T_H[:p.T] = utils.convex_combo(T_H_new[:p.T], T_H[:p.T], p.nu)
        guesses_b = utils.convex_combo(b_mat, guesses_b, p.nu)
        guesses_n = utils.convex_combo(n_mat, guesses_n, p.nu)

        print('r diff: ', (rnew[:p.T] - r[:p.T]).max(),
              (rnew[:p.T] - r[:p.T]).min())
        print('BQ diff: ', (BQnew[:p.T] - BQ[:p.T]).max(),
              (BQnew[:p.T] - BQ[:p.T]).min())
        print('T_H diff: ', (T_H_new[:p.T]-T_H[:p.T]).max(),
              (T_H_new[:p.T] - T_H[:p.T]).min())
        print('Y diff: ', (Ynew[:p.T]-Y[:p.T]).max(),
              (Ynew[:p.T] - Y[:p.T]).min())
        if not p.baseline_spending:
            if T_H.all() != 0:
                TPIdist = np.array(
                    list(utils.pct_diff_func(rnew[:p.T], r[:p.T])) +
                    list(utils.pct_diff_func(BQnew[:p.T],
                                             BQ[:p.T]).flatten()) +
                    list(utils.pct_diff_func(wnew[:p.T], w[:p.T])) +
                    list(utils.pct_diff_func(T_H_new[:p.T],
                                             T_H[:p.T]))).max()
            else:
                TPIdist = np.array(
                    list(utils.pct_diff_func(rnew[:p.T], r[:p.T])) +
                    list(utils.pct_diff_func(BQnew[:p.T],
                                             BQ[:p.T]).flatten()) +
                    list(utils.pct_diff_func(wnew[:p.T], w[:p.T])) +
                    list(np.abs(T_H[:p.T]))).max()
        else:
            TPIdist = np.array(
                list(utils.pct_diff_func(rnew[:p.T], r[:p.T])) +
                list(utils.pct_diff_func(BQnew[:p.T], BQ[:p.T]).flatten())
                + list(utils.pct_diff_func(wnew[:p.T], w[:p.T])) +
                list(utils.pct_diff_func(Ynew[:p.T], Y[:p.T]))).max()

        TPIdist_vec[TPIiter] = TPIdist
        # After T=10, if cycling occurs, drop the value of nu
        # wait til after T=10 or so, because sometimes there is a jump up
        # in the first couple iterations
        # if TPIiter > 10:
        #     if TPIdist_vec[TPIiter] - TPIdist_vec[TPIiter - 1] > 0:
        #         nu /= 2
        #         print 'New Value of nu:', nu
        TPIiter += 1
        print('Iteration:', TPIiter)
        print('\tDistance:', TPIdist)

    # Loop through years to calculate debt and gov't spending.
    # The re-assignment of G0 & D0 is necessary because Y0 may change
    # in the TPI loop.
    if not p.budget_balance:
        if p.baseline:
            D_0 = p.initial_debt * Y[0]
        else:
            D_0 = D0
        if not p.baseline_spending:
            G_0 = p.ALPHA_G[0] * Y[0]
        dg_fixed_values = (Y, REVENUE, T_H, D_0, G_0)
        D, G = fiscal.D_G_path(r, dg_fixed_values, Gbaseline, p)

    # Solve HH problem in inner loop
    outer_loop_vars = (r, K, BQ, T_H, theta)
    euler_errors = np.zeros((p.T, 2 * p.S, p.J))
    lazy_values = []
    for j in range(p.J):
        guesses = (guesses_b[:, :, j], guesses_n[:, :, j])
        lazy_values.append(
            delayed(inner_loop)(guesses, outer_loop_vars, initial_values,
                                j, ind, p))
    results = compute(*lazy_values, get=dask.multiprocessing.get,
                      num_workers=p.num_workers)
    for j, result in enumerate(results):
        euler_errors[:, :, j], b_mat[:, :, j], n_mat[:, :, j] = result

    bmat_s = np.zeros((p.T, p.S, p.J))
    bmat_s[0, 1:, :] = initial_b[:-1, :]
    bmat_s[1:, 1:, :] = b_mat[:p.T-1, :-1, :]
    bmat_splus1 = np.zeros((p.T, p.S, p.J))
    bmat_splus1[:, :, :] = b_mat[:p.T, :, :]

    L[:p.T] = aggr.get_L(n_mat[:p.T], p, 'TPI')
    B[1:p.T] = aggr.get_K(bmat_splus1[:p.T], p, 'TPI', 'False')[:p.T - 1]

    if not p.small_open:
        K[:p.T] = B[:p.T] - D[:p.T]
    else:
        K[:p.T] = firm.get_K(L[:p.T], p.tpi_firm_r[:p.T], p)
    Ynew = firm.get_Y(K[:p.T], L[:p.T], p)

    # testing for change in Y
    ydiff = Ynew[:p.T] - Y[:p.T]
    ydiff_max = np.amax(np.abs(ydiff))
    print('ydiff_max = ', ydiff_max)

    if not p.small_open:
        rnew = firm.get_r(Ynew[:p.T], K[:p.T], p)
    else:
        rnew = r
    # compute
    wnew = firm.get_w_from_r(rnew[:p.T], p)

    # Update Y
    Y = Ynew[:]

    b_mat_shift = np.append(np.reshape(initial_b, (1, p.S, p.J)),
                            b_mat[:p.T - 1, :, :], axis=0)
    BQnew = aggr.get_BQ(rnew, b_mat_shift, None, p, 'TPI', False)
    BQnew_3D = np.tile(BQnew.reshape(BQnew.shape[0], 1,
                                     BQnew.shape[1]), (1, p.S, 1))
    REVENUE = np.array(
        list(aggr.revenue(rnew[:p.T], wnew[:p.T], bmat_s,
                          n_mat[:p.T, :, :], BQnew_3D[:p.T, :, :],
                          Ynew[:p.T], L[:p.T], K[:p.T], factor, theta,
                          etr_parms4D, p, 'TPI')) + [revenue_ss] * p.S)
    tax_path = tax.total_taxes(r[:p.T], w[:p.T], bmat_s,
                               n_mat[:p.T, :, :], BQ_3D[:p.T, :, :],
                               factor, T_H[:p.T], theta, None, False,
                               'TPI', p.e, p.retire, etr_parms4D, p)
    rpath = utils.to_timepath_shape(r, p)
    wpath = utils.to_timepath_shape(w, p)
    c_path = household.get_cons(rpath[:p.T, :, :], wpath[:p.T, :, :],
                                bmat_s, bmat_splus1, n_mat[:p.T, :, :],
                                BQ_3D[:p.T, :, :], tax_path, p.e, None,
                                p)
    C = aggr.get_C(c_path, p, 'TPI')

    if not p.budget_balance:
        if p.baseline:
            D_0 = p.initial_debt * Y[0]
        else:
            D_0 = D0
        if not p.baseline_spending:
            G_0 = p.ALPHA_G[0] * Y[0]
        dg_fixed_values = (Y, REVENUE, T_H, D_0, G_0)
        D, G = fiscal.D_G_path(r, dg_fixed_values, Gbaseline, p)

    if not p.small_open:
        I = aggr.get_I(bmat_splus1[:p.T], K[1:p.T + 1], K[:p.T], p, 'TPI')
        rc_error = Y[:p.T] - C[:p.T] - I[:p.T] - G[:p.T]
    else:
        I = ((1 + p.g_n[:p.T]) * np.exp(p.g_y) * K[1:p.T + 1] -
             (1.0 - p.delta) * K[:p.T])
        BI = aggr.get_I(bmat_splus1[:p.T], B[1:p.T + 1], B[:p.T], p, 'TPI')
        new_borrowing = (D[1:p.T] * (1 + p.g_n[1:p.T]) *
                         np.exp(p.g_y) - D[:p.T - 1])
        rc_error = (Y[:p.T - 1] + new_borrowing - (
            C[:p.T - 1] + BI[:p.T - 1] + G[:p.T - 1]) +
                    (p.tpi_hh_r[:p.T - 1] * B[:p.T - 1] - (
                        p.delta + p.tpi_firm_r[:p.T - 1]) * K[:p.T - 1] -
                     p.tpi_hh_r[:p.T - 1] * D[:p.T - 1]))

    # Compute total investment (not just domestic)
    I_total = ((1 + p.g_n[:p.T]) * np.exp(p.g_y) * K[1:p.T + 1] -
               (1.0 - p.delta) * K[:p.T])

    # Compute business and invidiual income tax revenue
    business_revenue = tax.get_biz_tax(w[:p.T], Y[:p.T], L[:p.T],
                                       K[:p.T], p)
    IITpayroll_revenue = REVENUE[:p.T] - business_revenue[:p.T]
    rce_max = np.amax(np.abs(rc_error))
    print('Max absolute value resource constraint error:', rce_max)

    print('Checking time path for violations of constraints.')
    for t in range(p.T):
        household.constraint_checker_TPI(
            b_mat[t], n_mat[t], c_path[t], t, p.ltilde)

    eul_savings = euler_errors[:, :p.S, :].max(1).max(1)
    eul_laborleisure = euler_errors[:, p.S:, :].max(1).max(1)

    print('Max Euler error, savings: ', eul_savings)
    print('Max Euler error labor supply: ', eul_laborleisure)

    '''
    ------------------------------------------------------------------------
    Save variables/values so they can be used in other modules
    ------------------------------------------------------------------------
    '''

    output = {'Y': Y, 'B': B, 'K': K, 'L': L, 'C': C, 'I': I,
              'I_total': I_total, 'BQ': BQ,
              'REVENUE': REVENUE, 'business_revenue': business_revenue,
              'IITpayroll_revenue': IITpayroll_revenue, 'T_H': T_H,
              'G': G, 'D': D, 'r': r, 'w': w, 'b_mat': b_mat,
              'n_mat': n_mat, 'c_path': c_path, 'tax_path': tax_path,
              'eul_savings': eul_savings,
              'eul_laborleisure': eul_laborleisure}

    tpi_dir = os.path.join(p.output_base, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
    pickle.dump(output, open(tpi_vars, "wb"))

    if np.any(G) < 0:
        print('Government spending is negative along transition path' +
              ' to satisfy budget')

    if (((TPIiter >= p.maxiter) or
         (np.absolute(TPIdist) > p.mindist_TPI)) and
        ENFORCE_SOLUTION_CHECKS):
        raise RuntimeError('Transition path equlibrium not found' +
                           ' (TPIdist)')

    if ((np.any(np.absolute(rc_error) >= p.mindist_TPI * 10)) and
        ENFORCE_SOLUTION_CHECKS):
        raise RuntimeError('Transition path equlibrium not found ' +
                           '(rc_error)')

    if ((np.any(np.absolute(eul_savings) >= p.mindist_TPI) or
         (np.any(np.absolute(eul_laborleisure) > p.mindist_TPI))) and
        ENFORCE_SOLUTION_CHECKS):
        raise RuntimeError('Transition path equlibrium not found ' +
                           '(eulers)')

    return output
