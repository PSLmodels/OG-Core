# imports
from re import VERBOSE
import numpy as np
import scipy.optimize as opt
from dask import delayed, compute
import dask.multiprocessing
from ogcore import tax, household, firm, utils, fiscal
from ogcore import aggregates as aggr
from ogcore.constants import SHOW_RUNTIME
import os
import warnings


if not SHOW_RUNTIME:
    warnings.simplefilter("ignore", RuntimeWarning)

'''
Set minimizer tolerance
'''
MINIMIZER_TOL = 1e-13

'''
Set flag for enforcement of solution check
'''
ENFORCE_SOLUTION_CHECKS = True

'''
Set flag for verbosity
'''
VERBOSE = True

'''
------------------------------------------------------------------------
    Define Functions
------------------------------------------------------------------------
'''


def euler_equation_solver(guesses, *args):
    '''
    Finds the euler errors for certain b and n, one ability type at a
    time.

    Args:
        guesses (Numpy array): initial guesses for b and n, lenth 2S
        args (tuple): tuple of arguments (r, w, bq, TR, factor, j, p)
        w (scalar): real wage rate
        bq (Numpy array): bequest amounts by age, length S
        tr (scalar): government transfer amount by age, length S
        ubi (vector): universal basic income (UBI) payment, length S
        factor (scalar): scaling factor converting model units to dollars
        p (OG-Core Specifications object): model parameters

    Returns:
        errros (Numpy array): errors from FOCs, length 2S

    '''
    (r, w, bq, tr, ubi, factor, j, p) = args

    b_guess = np.array(guesses[:p.S])
    n_guess = np.array(guesses[p.S:])
    b_s = np.array([0] + list(b_guess[:-1]))
    b_splus1 = b_guess

    theta = tax.replacement_rate_vals(n_guess, w, factor, j, p)

    error1 = household.FOC_savings(r, w, b_s, b_splus1, n_guess, bq,
                                   factor, tr, ubi, theta, p.e[:, j], p.rho,
                                   p.tau_c[-1, :, j], p.etr_params[-1, :, :],
                                   p.mtry_params[-1, :, :], None, j, p,
                                   'SS')
    error2 = household.FOC_labor(r, w, b_s, b_splus1, n_guess, bq,
                                 factor, tr, ubi, theta, p.chi_n, p.e[:, j],
                                 p.tau_c[-1, :, j], p.etr_params[-1, :, :],
                                 p.mtrx_params[-1, :, :], None, j, p,
                                 'SS')

    # Put in constraints for consumption and savings.
    # According to the euler equations, they can be negative.  When
    # Chi_b is large, they will be.  This prevents that from happening.
    # I'm not sure if the constraints are needed for labor.
    # But we might as well put them in for now.
    mask1 = n_guess < 0
    mask2 = n_guess > p.ltilde
    mask3 = b_guess <= 0
    mask4 = np.isnan(n_guess)
    mask5 = np.isnan(b_guess)
    error2[mask1] = 1e14
    error2[mask2] = 1e14
    error1[mask3] = 1e14
    error1[mask5] = 1e14
    error2[mask4] = 1e14
    taxes = tax.net_taxes(r, w, b_s, n_guess, bq, factor, tr, ubi, theta,
                          None, j, False, 'SS', p.e[:, j],
                          p.etr_params[-1, :, :], p)
    cons = household.get_cons(r, w, b_s, b_splus1, n_guess, bq, taxes,
                              p.e[:, j], p.tau_c[-1, :, j], p)
    mask6 = cons < 0
    error1[mask6] = 1e14
    errors = np.hstack((error1, error2))

    return errors


def inner_loop(outer_loop_vars, p, client):
    '''
    This function solves for the inner loop of the SS.  That is, given
    the guesses of the outer loop variables (r, w, TR, factor) this
    function solves the households' problems in the SS.

    Args:
        outer_loop_vars (tuple): tuple of outer loop variables,
            (bssmat, nssmat, r, w, BQ, TR, factor) or
            (bssmat, nssmat, r, w, BQ, Y, TR, factor)
        bssmat (Numpy array): initial guess at savings, size = SxJ
        nssmat (Numpy array): initial guess at labor supply, size = SxJ
        BQ (array_like): aggregate bequest amount(s)
        Y (scalar): real GDP
        TR (scalar): lump sum transfer amount
        factor (scalar): scaling factor converting model units to dollars
        w (scalar): real wage rate
        p (OG-Core Specifications object): model parameters
        client (Dask client object): client

    Returns:
        (tuple): results from household solution:

            * euler_errors (Numpy array): errors terms from FOCs,
                size = 2SxJ
            * bssmat (Numpy array): savings, size = SxJ
            * nssmat (Numpy array): labor supply, size = SxJ
            * new_r (scalar): real interest rate on firm capital
            * new_r_gov (scalar): real interest rate on government debt
            * new_r_p (scalar): real interest rate on household
                portfolio
            * new_w (scalar): real wage rate
            * new_TR (scalar): lump sum transfer amount
            * new_Y (scalar): real GDP
            * new_factor (scalar): scaling factor converting model
                units to dollars
            * new_BQ (array_like): aggregate bequest amount(s)
            * average_income_model (scalar): average income in model
                units

    '''
    # unpack variables to pass to function
    bssmat, nssmat, r, w, Y, BQ, TR, factor = outer_loop_vars

    # print('IN inner looPPP -- r, w, Y = ', r, w, Y)

    # initialize array for euler errors
    euler_errors = np.zeros((2 * p.S, p.J))

    print('r, w = ', r, w)

    # w = firm.get_w_from_r(r, p, 'SS')
    r_gov = fiscal.get_r_gov(r, p)
    D, D_d, D_f, new_borrowing, debt_service, new_borrowing_f =\
        fiscal.get_D_ss(r_gov, Y, p)
    I_g = fiscal.get_I_g(Y, p.alpha_I[-1])
    K_g = fiscal.get_K_g(0, I_g, p, 'SS')
    MPKg = firm.get_MPx(Y, K_g, p.gamma_g, p, 'SS')
    K = firm.get_K_from_Y(Y, r, p, 'SS')
    r_p = aggr.get_r_p(r, r_gov, K, K_g, D, MPKg, p, 'SS')
    bq = household.get_bq(BQ, None, p, 'SS')
    tr = household.get_tr(TR, None, p, 'SS')
    ubi = p.ubi_nom_array[-1, :, :] / factor

    lazy_values = []
    if client:
        scattered_p = client.scatter(p, broadcast=True)
    else:
        scattered_p = p
    for j in range(p.J):
        guesses = np.append(bssmat[:, j], nssmat[:, j])
        euler_params = (
            r_p, w, bq[:, j], tr[:, j], ubi[:, j], factor, j,
            scattered_p)
        lazy_values.append(delayed(opt.root)(
            euler_equation_solver, guesses * .9,
            args=euler_params, method=p.FOC_root_method, tol=MINIMIZER_TOL))
    if client:
        futures = client.compute(lazy_values, num_workers=p.num_workers)
        results = client.gather(futures)
    else:
        results = results = compute(
            *lazy_values, scheduler=dask.multiprocessing.get,
            num_workers=p.num_workers)

    for j, result in enumerate(results):
        euler_errors[:, j] = result.fun
        bssmat[:, j] = result.x[:p.S]
        nssmat[:, j] = result.x[p.S:]

    L = aggr.get_L(nssmat, p, 'SS')
    B = aggr.get_B(bssmat, p, 'SS', False)
    w_open = firm.get_w_from_r(p.world_int_rate[-1], p, 'SS')
    K_demand_open = firm.get_K(p.world_int_rate[-1], w_open, L, p, 'SS')
    K, K_d, K_f = aggr.get_K_splits(B, K_demand_open, D_d, p.zeta_K[-1])

    Y = firm.get_Y(K, K_g, L, p, 'SS')
    # print('Inner loop get Y 1 = ', Y)
    I_g = fiscal.get_I_g(Y, p.alpha_I[-1])
    K_g = fiscal.get_K_g(0, I_g, p, 'SS')
    Y = firm.get_Y(K, K_g, L, p, 'SS')
    # print('Inner loop get Y 2 = ', Y)
    if p.zeta_K[-1] == 1.0:
        new_r = p.world_int_rate[-1]
    else:
        new_r = firm.get_r(Y, K, p, 'SS')
    new_w = firm.get_w(Y, L, p, 'SS')  # does this work for the open econ case?

    b_s = np.array(list(np.zeros(p.J).reshape(1, p.J)) +
                   list(bssmat[:-1, :]))
    new_r_gov = fiscal.get_r_gov(new_r, p)
    MPKg = firm.get_MPx(Y, K_g, p.gamma_g, p, 'SS')
    new_r_p = aggr.get_r_p(new_r, new_r_gov, K, K_g, D, MPKg, p, 'SS')
    average_income_model = ((new_r_p * b_s + new_w * p.e * nssmat) *
                            p.omega_SS.reshape(p.S, 1) *
                            p.lambdas.reshape(1, p.J)).sum()
    if p.baseline:
        new_factor = p.mean_income_data / average_income_model
    else:
        new_factor = factor
    new_BQ = aggr.get_BQ(new_r_p, bssmat, None, p, 'SS', False)
    new_bq = household.get_bq(new_BQ, None, p, 'SS')
    tr = household.get_tr(TR, None, p, 'SS')
    theta = tax.replacement_rate_vals(nssmat, new_w, new_factor, None, p)
    etr_params_3D = np.tile(
        np.reshape(p.etr_params[-1, :, :],
                   (p.S, 1, p.etr_params.shape[2])), (1, p.J, 1))
    taxss = tax.net_taxes(
        new_r_p, new_w, b_s, nssmat, new_bq, factor, tr, ubi, theta, None,
        None, False, 'SS', p.e, etr_params_3D, p)
    cssmat = household.get_cons(
        new_r_p, new_w, b_s, bssmat, nssmat, new_bq, taxss, p.e,
        p.tau_c[-1, :, :], p)
    total_tax_revenue, _, agg_pension_outlays, UBI_outlays, _, _, _, _, _, _ =\
        aggr.revenue(new_r_p, new_w, b_s, nssmat, new_bq, cssmat, Y, L,
                     K, factor, ubi, theta, etr_params_3D, p, 'SS')
    G = fiscal.get_G_ss(Y, total_tax_revenue, agg_pension_outlays, TR,
                        UBI_outlays, I_g, new_borrowing, debt_service, p)
    new_TR = fiscal.get_TR(Y, TR, G, total_tax_revenue, agg_pension_outlays,
                           UBI_outlays, I_g, p, 'SS')

    # print('BQ at the end of inner loop: ', new_BQ)
    return euler_errors, bssmat, nssmat, new_r, new_r_gov, new_r_p, \
        new_w, new_TR, Y, new_factor, new_BQ, average_income_model


def SS_solver(bmat, nmat, r, w, Y, BQ, TR, factor, p, client,
              fsolve_flag=False):
    '''
    Solves for the steady state distribution of capital, labor, as well
    as w, r, TR and the scaling factor, using functional iteration.

    Args:
        bmat (Numpy array): initial guess at savings, size = SxJ
        nmat (Numpy array): initial guess at labor supply, size = SxJ
        r (scalar): real interest rate
        BQ (array_like): aggregate bequest amount(s)
        TR (scalar): lump sum transfer amount
        factor (scalar): scaling factor converting model units to dollars
        Y (scalar): real GDP
        p (OG-Core Specifications object): model parameters
        client (Dask client object): client

    Returns:
        output (dictionary): dictionary with steady state solution
            results

    '''
    dist = 10
    iteration = 0
    dist_vec = np.zeros(p.maxiter)
    maxiter_ss = p.maxiter
    nu_ss = p.nu
    if fsolve_flag:  # case where already solved via SS_fsolve
        maxiter_ss = 1
    if p.baseline_spending:
        TR_ss = TR
    while (dist > p.mindist_SS) and (iteration < maxiter_ss):
        # Solve for the steady state levels of b and n, given w, r,
        # Y, BQ, TR, and factor
        # if p.baseline_spending:
        #     TR = TR_ss
        # if not p.budget_balance and not p.baseline_spending:
        #     Y = TR / p.alpha_T[-1]

        outer_loop_vars = (bmat, nmat, r, w, Y, BQ, TR, factor)

        # print('IN SS SOLVE outer loop -- r, w, Y = ', r, w, Y)

        (euler_errors, new_bmat, new_nmat, new_r, new_r_gov, new_r_p,
         new_w, new_TR, new_Y, new_factor, new_BQ,
         average_income_model) =\
            inner_loop(outer_loop_vars, p, client)

        # update guesses for next iteration
        bmat = utils.convex_combo(new_bmat, bmat, nu_ss)
        nmat = utils.convex_combo(new_nmat, nmat, nu_ss)
        w = utils.convex_combo(new_w, w, nu_ss)
        r = utils.convex_combo(new_r, r, nu_ss)
        factor = utils.convex_combo(new_factor, factor, nu_ss)
        BQ = utils.convex_combo(new_BQ, BQ, nu_ss)
        if p.baseline_spending:
            Y = utils.convex_combo(new_Y, Y, nu_ss)
            if Y != 0:
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                [utils.pct_diff_func(new_w, w)] +
                                list(utils.pct_diff_func(new_BQ, BQ)) +
                                [utils.pct_diff_func(new_Y, Y)] +
                                [utils.pct_diff_func(new_factor,
                                                     factor)]).max()
            else:
                # If Y is zero (if there is no output), a percent difference
                # will throw NaN's, so we use an absolute difference
                dist = np.array([utils.pct_diff_func(new_r, r)] +
                                [utils.pct_diff_func(new_w, w)] +
                                list(utils.pct_diff_func(new_BQ, BQ)) +
                                [abs(new_Y - Y)] +
                                [utils.pct_diff_func(new_factor,
                                                     factor)]).max()
        else:
            TR = utils.convex_combo(new_TR, TR, nu_ss)
            dist = np.array([utils.pct_diff_func(new_r, r)] +
                            [utils.pct_diff_func(new_w, w)] +
                            list(utils.pct_diff_func(new_BQ, BQ)) +
                            [utils.pct_diff_func(new_TR, TR)] +
                            [utils.pct_diff_func(new_factor, factor)]).max()

        dist_vec[iteration] = dist
        # Similar to TPI: if the distance between iterations increases, then
        # decrease the value of nu to prevent cycling
        if iteration > 10:
            if dist_vec[iteration] - dist_vec[iteration - 1] > 0:
                nu_ss /= 2.0
                print('New value of nu:', nu_ss)
        iteration += 1
        if VERBOSE:
            print('Iteration: %02d' % iteration, ' Distance: ', dist)

    # Generate the SS values of variables, including euler errors
    bssmat_s = np.append(np.zeros((1, p.J)), bmat[:-1, :], axis=0)
    bssmat_splus1 = bmat
    nssmat = nmat

    rss = r
    wss = w
    r_gov_ss = fiscal.get_r_gov(rss, p)
    TR_ss = TR
    Yss = Y
    I_g_ss = fiscal.get_I_g(Yss, p.alpha_I[-1])
    K_g_ss = fiscal.get_K_g(0, I_g_ss, p, 'SS')
    Lss = aggr.get_L(nssmat, p, 'SS')
    Bss = aggr.get_B(bssmat_splus1, p, 'SS', False)
    (Dss, D_d_ss, D_f_ss, new_borrowing, debt_service,
     new_borrowing_f) = fiscal.get_D_ss(r_gov_ss, Yss, p)
    w_open = firm.get_w_from_r(p.world_int_rate[-1], p, 'SS')
    K_demand_open_ss = firm.get_K(p.world_int_rate[-1], w_open, Lss, p, 'SS')
    Kss, K_d_ss, K_f_ss = aggr.get_K_splits(
        Bss, K_demand_open_ss, D_d_ss, p.zeta_K[-1])
    Yss = firm.get_Y(Kss, K_g_ss, Lss, p, 'SS')
    I_g_ss = fiscal.get_I_g(Yss, p.alpha_I[-1])
    K_g_ss = fiscal.get_K_g(0, I_g_ss, p, 'SS')
    MPKg = firm.get_MPx(Yss, K_g_ss, p.gamma_g, p, 'SS')
    r_p_ss = aggr.get_r_p(rss, r_gov_ss, Kss, K_g_ss, Dss, MPKg, p, 'SS')
    # Note that implicitly in this computation is that immigrants'
    # wealth is all in the form of private capital
    I_d_ss = aggr.get_I(bssmat_splus1, K_d_ss, K_d_ss, p, 'SS')
    Iss = aggr.get_I(bssmat_splus1, Kss, Kss, p, 'SS')
    BQss = new_BQ
    factor_ss = factor
    bqssmat = household.get_bq(BQss, None, p, 'SS')
    trssmat = household.get_tr(TR_ss, None, p, 'SS')
    ubissmat = p.ubi_nom_array[-1, :, :] / factor_ss
    theta = tax.replacement_rate_vals(nssmat, wss, factor_ss, None, p)

    # Compute effective and marginal tax rates for all agents
    etr_params_3D = np.tile(np.reshape(
        p.etr_params[-1, :, :], (p.S, 1, p.etr_params.shape[2])), (1, p.J, 1))
    mtrx_params_3D = np.tile(np.reshape(
        p.mtrx_params[-1, :, :], (p.S, 1, p.mtrx_params.shape[2])),
                             (1, p.J, 1))
    mtry_params_3D = np.tile(np.reshape(
        p.mtry_params[-1, :, :], (p.S, 1, p.mtry_params.shape[2])),
                             (1, p.J, 1))
    mtry_ss = tax.MTR_income(r_p_ss, wss, bssmat_s, nssmat, factor, True,
                             p.e, etr_params_3D, mtry_params_3D, p)
    mtrx_ss = tax.MTR_income(r_p_ss, wss, bssmat_s, nssmat, factor, False,
                             p.e, etr_params_3D, mtrx_params_3D, p)
    etr_ss = tax.ETR_income(r_p_ss, wss, bssmat_s, nssmat, factor, p.e,
                            etr_params_3D, p)

    taxss = tax.net_taxes(r_p_ss, wss, bssmat_s, nssmat, bqssmat,
                          factor_ss, trssmat, ubissmat, theta, None, None,
                          False, 'SS', p.e, etr_params_3D, p)
    cssmat = household.get_cons(r_p_ss, wss, bssmat_s, bssmat_splus1,
                                nssmat, bqssmat, taxss,
                                p.e, p.tau_c[-1, :, :], p)
    yss_before_tax_mat = household.get_y(
        r_p_ss, wss, bssmat_s, nssmat, p)
    Css = aggr.get_C(cssmat, p, 'SS')

    (total_tax_revenue, iit_payroll_tax_revenue, agg_pension_outlays,
     UBI_outlays, bequest_tax_revenue, wealth_tax_revenue, cons_tax_revenue,
     business_tax_revenue, payroll_tax_revenue, iit_revenue
     ) = aggr.revenue(
         r_p_ss, wss, bssmat_s, nssmat, bqssmat, cssmat, Yss, Lss, Kss,
         factor, ubissmat, theta, etr_params_3D, p, 'SS')
    Gss = fiscal.get_G_ss(
        Yss, total_tax_revenue, agg_pension_outlays, TR_ss, UBI_outlays,
        I_g_ss, new_borrowing, debt_service, p)

    # Compute total investment (not just domestic)
    Iss_total = aggr.get_I(None, Kss, Kss, p, 'total_ss')

    # solve resource constraint
    # net foreign borrowing
    debt_service_f = fiscal.get_debt_service_f(r_p_ss, D_f_ss)
    RC = aggr.resource_constraint(
        Yss, Css, Gss, I_d_ss, I_g_ss, K_f_ss, new_borrowing_f, debt_service_f,
        r_p_ss, p)
    if VERBOSE:
        print('Foreign debt holdings = ', D_f_ss)
        print('Foreign capital holdings = ', K_f_ss)
        print('resource constraint: ', RC)

    if Gss < 0:
        print('Steady state government spending is negative to satisfy'
              + ' budget')

    if ENFORCE_SOLUTION_CHECKS and (np.absolute(RC) >
                                    p.mindist_SS):
        print('Resource Constraint Difference:', RC)
        err = 'Steady state aggregate resource constraint not satisfied'
        raise RuntimeError(err)

    # check constraints
    household.constraint_checker_SS(bssmat_splus1, nssmat, cssmat, p.ltilde)

    euler_savings = euler_errors[:p.S, :]
    euler_labor_leisure = euler_errors[p.S:, :]
    if VERBOSE:
        print('Maximum error in labor FOC = ',
              np.absolute(euler_labor_leisure).max())
        print('Maximum error in savings FOC = ',
              np.absolute(euler_savings).max())

    # Return dictionary of SS results
    output = {'Kss': Kss, 'K_f_ss': K_f_ss, 'K_d_ss': K_d_ss,
              'K_g_ss': K_g_ss, 'I_g_ss': I_g_ss,
              'Bss': Bss, 'Lss': Lss, 'Css': Css, 'Iss': Iss,
              'Iss_total': Iss_total, 'I_d_ss': I_d_ss, 'nssmat': nssmat,
              'Yss': Yss, 'Dss': Dss, 'D_f_ss': D_f_ss,
              'D_d_ss': D_d_ss, 'wss': wss, 'rss': rss,
              'total_taxes_ss': taxss, 'ubissmat': ubissmat,
              'r_gov_ss': r_gov_ss, 'r_p_ss': r_p_ss, 'theta': theta,
              'BQss': BQss, 'factor_ss': factor_ss, 'bssmat_s': bssmat_s,
              'cssmat': cssmat, 'bssmat_splus1': bssmat_splus1,
              'yss_before_tax_mat': yss_before_tax_mat,
              'bqssmat': bqssmat, 'TR_ss': TR_ss, 'trssmat': trssmat,
              'Gss': Gss, 'total_tax_revenue': total_tax_revenue,
              'business_tax_revenue': business_tax_revenue,
              'iit_payroll_tax_revenue': iit_payroll_tax_revenue,
              'iit_revenue': iit_revenue,
              'payroll_tax_revenue': payroll_tax_revenue,
              'agg_pension_outlays': agg_pension_outlays,
              'UBI_outlays_SS': UBI_outlays,
              'bequest_tax_revenue': bequest_tax_revenue,
              'wealth_tax_revenue': wealth_tax_revenue,
              'cons_tax_revenue': cons_tax_revenue,
              'euler_savings': euler_savings,
              'debt_service_f': debt_service_f,
              'new_borrowing_f': new_borrowing_f,
              'debt_service': debt_service,
              'new_borrowing': new_borrowing,
              'euler_labor_leisure': euler_labor_leisure,
              'resource_constraint_error': RC,
              'etr_ss': etr_ss, 'mtrx_ss': mtrx_ss, 'mtry_ss': mtry_ss}

    return output


def SS_fsolve(guesses, *args):
    '''
    Solves for the steady state distribution of capital, labor, as well
    as w, r, TR and the scaling factor, using a root finder.

    Args:
        guesses (list): initial guesses outer loop variables (r, BQ,
            TR, factor)
        args (tuple): tuple of arguments (bssmat, nssmat, TR_ss,
            factor_ss, p, client)
        bssmat (Numpy array): initial guess at savings, size = SxJ
        nssmat (Numpy array): initial guess at labor supply, size = SxJ
        TR_ss (scalar): lump sum transfer amount
        factor_ss (scalar): scaling factor converting model units to dollars
        p (OG-Core Specifications object): model parameters
        client (Dask client object): client

    Returns:
        errors (list): errors from differences between guessed and
            implied outer loop variables

    '''
    (bssmat, nssmat, TR_ss, factor_ss, p, client) = args

    # Rename the inputs
    r = guesses[0]
    w = guesses[1]
    Y = guesses[2]
    if p.baseline:
        BQ = guesses[3:-2]
        TR = guesses[-2]
        factor = guesses[-1]
    else:
        BQ = guesses[3:-1]
        TR = guesses[-1]
        factor = factor_ss
    if p.baseline_spending:
        TR = TR_ss
    if not p.budget_balance and not p.baseline_spending:
        Y = TR / p.alpha_T[-1]

    outer_loop_vars = (bssmat, nssmat, r, w, Y, BQ, TR, factor)

    # Solve for the steady state levels of b and n, given w, r, TR and
    # factor
    (euler_errors, bssmat, nssmat, new_r, new_r_gov, new_r_p, new_w,
     new_TR, new_Y, new_factor, new_BQ, average_income_model) =\
        inner_loop(outer_loop_vars, p, client)

    # Create list of errors in general equilibrium variables
    error_r = new_r - r
    # Check and punish violations of the bounds on the interest rate
    if new_r + p.delta <= 0:
        error_r = 1e9
    error_w = new_w - w
    error_Y = new_Y - Y
    error_BQ = new_BQ - BQ
    error_TR = new_TR - TR
    # divide factor by 1000000 to put on similar scale
    error_factor = new_factor / 1000000 - factor / 1000000
    # Check and punish violations of the factor
    if new_factor <= 0:
        error_factor = 1e9
    if p.baseline:
        errors = [error_r, error_w, error_Y] + list(error_BQ) + [error_TR, error_factor]
    else:
        errors = [error_r, error_w, error_Y] + list(error_BQ) + [error_TR]
    if VERBOSE:
        print('GE loop errors = ', errors)

    return errors


def run_SS(p, client=None):
    '''
    Solve for steady-state equilibrium of OG-Core.

    Args:
        p (OG-Core Specifications object): model parameters
        client (Dask client object): client

    Returns:
        output (dictionary): dictionary with steady-state solution
            results

    '''
    # For initial guesses of w, r, TR, and factor, we use values that
    # are close to some steady state values.
    if p.baseline:
        if p.zeta_K[-1] == 1.0:
            rguess = p.world_int_rate[-1]
        else:
            rguess = p.initial_guess_r_SS
        if p.use_zeta:
            b_guess = np.ones((p.S, p.J)) * 0.0055
            n_guess = np.ones((p.S, p.J)) * .4 * p.ltilde
        else:
            b_guess = np.ones((p.S, p.J)) * 0.07
            n_guess = np.ones((p.S, p.J)) * .35 * p.ltilde
        wguess = firm.get_w_from_r(rguess, p, 'SS')
        TRguess = p.initial_guess_TR_SS
        Yguess = TRguess / p.alpha_T[-1]
        factorguess = p.initial_guess_factor_SS
        BQguess = aggr.get_BQ(rguess, b_guess, None, p, 'SS', False)
        ss_params_baseline = (b_guess, n_guess, None, None, p, client)
        if p.use_zeta:
            BQguess = 0.12231465279007188
            guesses = [rguess, wguess, Yguess, BQguess, TRguess, factorguess]
        else:
            guesses = [rguess, wguess, Yguess] + list(BQguess) + [TRguess, factorguess]
        sol = opt.root(SS_fsolve, guesses, args=ss_params_baseline,
                       method=p.SS_root_method, tol=p.mindist_SS)
        if ENFORCE_SOLUTION_CHECKS and not sol.success:
            raise RuntimeError('Steady state equilibrium not found')
        rss = sol.x[0]
        wss = sol.x[1]
        Yss = sol.x[2]
        BQss = sol.x[3:-2]
        TR_ss = sol.x[-2]
        factor_ss = sol.x[-1]
        Yss = TR_ss/p.alpha_T[-1]  # may not be right - if budget_balance
        # # = True, but that's ok - will be fixed in SS_solver
        fsolve_flag = True
        output = SS_solver(b_guess, n_guess, rss, wss, Yss, BQss, TR_ss,
                           factor_ss, p, client, fsolve_flag)
    else:
        # Use the baseline solution to get starting values for the reform
        baseline_ss_dir = os.path.join(
            p.baseline_dir, 'SS', 'SS_vars.pkl')
        ss_solutions = utils.safe_read_pickle(baseline_ss_dir)
        # use baseline solution as starting values if dimensions match
        if ss_solutions['bssmat_splus1'].shape == (p.S, p.J):
            (b_guess, n_guess, rguess, wguess, BQguess, TRguess, Yguess,
             factor) =\
                (ss_solutions['bssmat_splus1'], ss_solutions['nssmat'],
                 ss_solutions['rss'], ss_solutions['wss'],
                 ss_solutions['BQss'], ss_solutions['TR_ss'],
                 ss_solutions['Yss'], ss_solutions['factor_ss'])
        else:
            if p.use_zeta:
                b_guess = np.ones((p.S, p.J)) * 0.0055
                n_guess = np.ones((p.S, p.J)) * .4 * p.ltilde
            else:
                b_guess = np.ones((p.S, p.J)) * 0.07
                n_guess = np.ones((p.S, p.J)) * .4 * p.ltilde
            if p.zeta_K[-1] == 1.0:
                rguess = p.world_int_rate[-1]
            else:
                rguess = p.initial_guess_r_SS
            wguess = firm.get_w_from_r(rguess, p, 'SS')
            TRguess = p.initial_guess_TR_SS
            Yguess = TRguess / p.alpha_T[-1]
            factor = p.initial_guess_factor_SS
            BQguess = aggr.get_BQ(rguess, b_guess, None, p, 'SS', False)
        if p.baseline_spending:
            TR_ss = TRguess
            ss_params_reform = (b_guess, n_guess, TR_ss, factor, p, client)
            if p.use_zeta:
                guesses = [rguess, wguess, Yguess] + list([BQguess]) + [Yguess]
            else:
                guesses = [rguess, wguess, Yguess] + list(BQguess) + [Yguess]
            sol = opt.root(SS_fsolve, guesses, args=ss_params_reform,
                           method=p.SS_root_method, tol=p.mindist_SS)
            rss = sol.x[0]
            wss = sol.x[1]
            Yss = sol.x[2]
            BQss = sol.x[3:-1]
            Yss = sol.x[-1]
        else:
            ss_params_reform = (b_guess, n_guess, None, factor, p, client)
            if p.use_zeta:
                guesses = [rguess, wguess, Yguess] + list([BQguess]) + [TRguess]
            else:
                guesses = [rguess, wguess, Yguess] + list(BQguess) + [TRguess]
            sol = opt.root(SS_fsolve, guesses, args=ss_params_reform,
                           method=p.SS_root_method, tol=p.mindist_SS)
            rss = sol.x[0]
            wss = sol.x[1]
            Yss = sol.x[2]
            BQss = sol.x[3:-1]
            TR_ss = sol.x[-1]
            Yss = TR_ss/p.alpha_T[-1]  # may not be right - if
            # budget_balance = True, but that's ok - will be fixed in
            # SS_solver
        if ENFORCE_SOLUTION_CHECKS and not sol.success == 1:
            raise RuntimeError('Steady state equilibrium not found')
        # Return SS values of variables
        fsolve_flag = True
        # Return SS values of variables
        output = SS_solver(b_guess, n_guess, rss, wss, Yss, BQss, TR_ss, factor,
                           p, client, fsolve_flag)
        if output['Gss'] < 0.:
            warnings.warn('Warning: The combination of the tax policy '
                          + 'you specified and your target debt-to-GDP '
                          + 'ratio results in an infeasible amount of '
                          + 'government spending in order to close the '
                          + 'budget (i.e., G < 0)')
    return output
