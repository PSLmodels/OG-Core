# Packages
import numpy as np
import numba
from ogcore import utils

# set constants
MONTHS_IN_A_YEAR = 12
THOUSAND = 1000


def replacement_rate_vals(nssmat, wss, factor_ss, j, p):
    """
    Calculates replacement rate values for the social security system.

    Args:
        nssmat (Numpy array): initial guess at labor supply, size = SxJ
        new_w (scalar): steady state real wage rate
        factor_ss (scalar): scaling factor converting model units to
            dollars
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        theta (Numpy array): social security replacement rate value for
            lifetime income group j

    """
    if j is not None:
        e = np.squeeze(p.e[-1, :, j])  # Only computes using SS earnings
    else:
        e = np.squeeze(p.e[-1, :, :])  # Only computes using SS earnings
    # adjust number of calendar years AIME computed from int model periods
    equiv_periods = int(round((p.S / 80.0) * p.AIME_num_years)) - 1
    if e.ndim == 2:
        dim2 = e.shape[1]
    else:
        dim2 = 1
    earnings = (e * (wss * nssmat * factor_ss)).reshape(p.S, dim2)
    # get highest earning years for number of years AIME computed from
    highest_earn = (
        -1.0 * np.sort(-1.0 * earnings[: p.retire[-1], :], axis=0)
    )[:equiv_periods]
    AIME = highest_earn.sum(0) / ((12.0 * (p.S / 80.0)) * equiv_periods)
    PIA = np.zeros(dim2)
    # Compute level of replacement using AIME brackets and PIA rates
    for j in range(dim2):
        if AIME[j] < p.AIME_bkt_1:
            PIA[j] = p.PIA_rate_bkt_1 * AIME[j]
        elif AIME[j] < p.AIME_bkt_2:
            PIA[j] = p.PIA_rate_bkt_1 * p.AIME_bkt_1 + p.PIA_rate_bkt_2 * (
                AIME[j] - p.AIME_bkt_1
            )
        else:
            PIA[j] = (
                p.PIA_rate_bkt_1 * p.AIME_bkt_1
                + p.PIA_rate_bkt_2 * (p.AIME_bkt_2 - p.AIME_bkt_1)
                + p.PIA_rate_bkt_3 * (AIME[j] - p.AIME_bkt_2)
            )
    # Set the maximum monthly replacment rate from SS benefits tables
    PIA[PIA > p.PIA_maxpayment] = p.PIA_maxpayment
    if p.PIA_minpayment != 0.0:
        PIA[PIA < p.PIA_minpayment] = p.PIA_minpayment
    theta = (PIA * (12.0 * p.S / 80.0)) / (factor_ss * wss)
    return theta


def pension_amount(w, n, theta, t, j, shift, method, e, p):
    """
    Calculate public pension benefit amounts for each household.

    Args:
        w (array_like): real wage rate
        n (Numpy array): labor supply
        theta (Numpy array): social security replacement rate value for
            lifetime income group j
        t (int): time period
        j (int): index of lifetime income group
        shift (bool): whether computing for periods 0--s or 1--(s+1),
            =True for 1--(s+1)
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        e (Numpy array): effective labor units
        p (OG-Core Specifications object): model parameters

    Returns:
        pension (Numpy array): pension amount for each household

    """
    # TODO: think about how can allow for transition from one
    # pension system to another along the time path
    if p.pension_system == "US-Style Social Security":
        pension = SS_amount(w, n, theta, t, j, shift, method, e, p)
    elif p.pension_system == "Defined Benefits":
        pension = DB_amount(w, n, t, j, shift, method, e, p)
    elif p.pension_system == "Notional Defined Contribution":
        sdf
    elif p.pension_system == "Points System":
        sdf
    else:
        raise ValueError(
            "pension_system must be one of the following: "
            "'US-Style Social Security', 'Defined Benefits', "
            "'Notional Defined Contribution', 'Points System'"
        )
    return pension


def SS_amount(w, n, theta, t, j, shift, method, e, p):
    """
    Calculate public pension benefit amounts for each household under
    a US-style social security system.

    Args:
        w (array_like): real wage rate
        n (Numpy array): labor supply
        theta (Numpy array): social security replacement rate value for
            lifetime income group j
        t (int): time period
        j (int): index of lifetime income group
        shift (bool): whether computing for periods 0--s or 1--(s+1),
            =True for 1--(s+1)
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        e (Numpy array): effective labor units
        p (OG-Core Specifications object): model parameters

    Returns:
        pension (Numpy array): pension amount for each household

    """
    if j is not None:
        if method == "TPI":
            if n.ndim == 2:
                w = w.reshape(w.shape[0], 1)
    else:
        if method == "TPI":
            w = utils.to_timepath_shape(w)

    pension = np.zeros_like(n)
    if method == "SS":
        # Depending on if we are looking at b_s or b_s+1, the
        # entry for retirement will change (it shifts back one).
        # The shift boolean makes sure we start replacement rates
        # at the correct age.
        if shift is False:
            pension[p.retire[-1] :] = theta * w
        else:
            pension[p.retire[-1] - 1 :] = theta * w
    elif method == "TPI":
        length = w.shape[0]
        if not shift:
            # retireTPI is different from retire, because in TP income
            # we are counting backwards with different length lists.
            # This will always be the correct location of retirement,
            # depending on the shape of the lists.
            retireTPI = p.retire[t : t + length] - p.S
        else:
            retireTPI = p.retire[t : t + length] - 1 - p.S
        if len(n.shape) == 1:
            if not shift:
                retireTPI = p.retire[t] - p.S
            else:
                retireTPI = p.retire[t] - 1 - p.S
            pension[retireTPI:] = (
                theta[j] * p.replacement_rate_adjust[t] * w[retireTPI:]
            )
        elif len(n.shape) == 2:
            for tt in range(pension.shape[0]):
                pension[tt, retireTPI[tt] :] = (
                    theta * p.replacement_rate_adjust[t + tt] * w[tt]
                )
        else:
            for tt in range(pension.shape[0]):
                pension[tt, retireTPI[tt] :, :] = (
                    theta.reshape(1, p.J)
                    * p.replacement_rate_adjust[t + tt]
                    * w[tt]
                )
    elif method == "TPI_scalar":
        # The above methods won't work if scalars are used.  This option
        # is only called by the SS_TPI_firstdoughnutring function in TPI.
        pension = theta * p.replacement_rate_adjust[0] * w

    return pension


def DB_amount(w, e, n, j, p):
    """
    Calculate public pension from a defined benefits system.
    """
    L_inc_avg = np.zeros(0)
    L_inc_avg_s = np.zeros(p.last_career_yrs)

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]
        # TODO: think about how to handle setting w_preTP and n_preTP
        w_S = np.append((p.w_preTP * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)

        DB_s = np.zeros(p.retire)
        DB = np.zeros(p.S)
        # TODO: we set a rep_rate_py in params, but not rep_rate.  What is it???
        DB = DB_1dim_loop(
            w_S,
            p.e[:, j],
            n_S,
            p.retire,
            p.S,
            p.g_y,
            L_inc_avg_s,
            L_inc_avg,
            DB_s,
            DB,
            p.last_career_yrs,
            p.rep_rate_py,
            p.yr_contr,
        )
        DB = DB[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            DB_s = np.zeros(p.retire)
            DB = np.zeros(p.S)
            DB = DB_1dim_loop(
                w,
                e,
                n,
                p.retiremet_age,
                p.S,
                p.g_y,
                L_inc_avg_s,
                L_inc_avg,
                DB_s,
                DB,
                p.last_career_yrs,
                p.rep_rate_py,
                p.yr_contr,
            )

        elif np.ndim(n) == 2:
            DB_sj = np.zeros((p.retire, p.J))
            DB = np.zeros((p.S, p.J))
            L_inc_avg_sj = np.zeros((p.last_career_yrs, p.J))
            DB = DB_2dim_loop(
                w,
                e,
                n,
                p.retire,
                p.S,
                p.g_y,
                L_inc_avg_sj,
                L_inc_avg,
                DB_sj,
                DB,
                p.last_career_yrs,
                p.rep_rate_py,
                p.yr_contr,
            )

    return DB


def NDC_amount(w, e, n, r, Y, j, p):
    """
    Calculate public pension from a notional defined contribution
    system.
    """
    g_ndc_amount = g_ndc(
        r,
        Y,
        p.g_n_SS,
        p.g_y,
    )
    delta_ret_amount = delta_ret(r, Y, p)

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]

        w_S = np.append((p.w_preTP * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)

        NDC_s = np.zeros(p.retire)
        NDC = np.zeros(p.S)
        NDC = NDC_1dim_loop(
            w_S,
            p.emat[:, j],
            n_S,
            p.retire,
            p.S,
            p.g_y,
            p.tau_p,
            g_ndc_amount,
            delta_ret_amount,
            NDC_s,
            NDC,
        )
        NDC = NDC[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            NDC_s = np.zeros(p.retire)
            NDC = np.zeros(p.S)
            NDC = NDC_1dim_loop(
                w,
                e,
                n,
                p.retire,
                p.S,
                p.g_y,
                p.tau_p,
                g_ndc_amount,
                delta_ret_amount,
                NDC_s,
                NDC,
            )
        elif np.ndim(n) == 2:
            NDC_sj = np.zeros((p.retire, p.J))
            NDC = np.zeros((p.S, p.J))
            NDC = NDC_2dim_loop(
                w,
                e,
                n,
                p.retire,
                p.S,
                p.g_y,
                p.tau_p,
                g_ndc_amount,
                delta_ret_amount,
                NDC_sj,
                NDC,
            )

    return NDC


def PS_amount(w, e, n, j, factor, p):
    """
    Calculate public pension from a points system.
    """

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]
        w_S = np.append((p.w_preTP * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)
        L_inc_avg_s = np.zeros(p.retire)
        PS = np.zeros(p.S)
        PS = PS_1dim_loop(
            w_S,
            p.emat[:, j],
            n_S,
            p.retire,
            p.S,
            p.g_y,
            p.vpoint,
            factor,
            L_inc_avg_s,
            PS,
        )
        PS = PS[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            L_inc_avg_s = np.zeros(p.retire)
            PS = np.zeros(p.S)
            PS = PS_1dim_loop(
                w,
                e,
                n,
                p.retire,
                p.S,
                p.g_y,
                p.vpoint,
                factor,
                L_inc_avg_s,
                PS,
            )

        elif np.ndim(n) == 2:
            L_inc_avg_sj = np.zeros((p.retire, p.J))
            PS = np.zeros((p.S, p.J))
            PS = PS_2dim_loop(
                w,
                e,
                n,
                p.retire,
                p.S,
                p.J,
                p.g_y,
                p.vpoint,
                factor,
                L_inc_avg_sj,
                PS,
            )

    return PS


def deriv_theta(r, w, e, Y, per_rmn, factor, p):
    """
    Change in pension benefits for another unit of labor supply for
    pension system selected
    """
    # TODO: Add SS here...
    if p.pension_system == "Defined Benefits":
        d_theta = deriv_DB(w, e, per_rmn, p)
        d_theta = d_theta[-per_rmn:]
    elif p.pension_system == "Notional Defined Contribution":
        d_theta = deriv_NDC(r, w, e, Y, per_rmn, p)
    elif p.pension_system == "Points System":
        d_theta = deriv_PS(w, e, per_rmn, factor, p)

    return d_theta


def deriv_NDC(r, w, e, Y, per_rmn, p):
    """
    Change in NDC pension benefits for another unit of labor supply
    """
    if per_rmn == 1:
        d_theta = 0
    elif per_rmn < (p.S - p.retire + 1):
        d_theta = np.zeros(per_rmn)
    else:
        d_theta_empty = np.zeros(per_rmn)
        delta_ret_amount = delta_ret(r, Y, p)
        g_ndc_amount = g_ndc(r, Y, p)
        d_theta = deriv_NDC_loop(
            w,
            e,
            per_rmn,
            p.S,
            p.retire,
            p.tau_p,
            g_ndc_amount,
            delta_ret_amount,
            d_theta_empty,
        )

    return d_theta


def deriv_DB(w, e, per_rmn, p):
    """
    Change in DB pension benefits for another unit of labor supply
    """

    if per_rmn < (p.S - p.retire + 1):
        d_theta = np.zeros(p.S)
    else:
        d_theta = deriv_DB_loop(
            w,
            e,
            p.S,
            p.retire,
            per_rmn,
            p.last_career_yrs,
            p.rep_rate_py,
            p.yr_contr,
        )
    return d_theta


def deriv_PS(w, e, per_rmn, factor, p):
    """
    Change in points system pension benefits for another unit of labor supply
    """

    if per_rmn < (p.S - p.retire + 1):
        d_theta = np.zeros(p.S)
    else:
        d_theta_empty = np.zeros(p.S)
        d_theta = deriv_PS_loop(
            w, e, p.S, p.retire, per_rmn, d_theta_empty, p.vpoint, factor
        )
        d_theta = d_theta[-per_rmn:]

    return d_theta


# TODO: can probably assign these growth rates in the if statements in
# the pension_amount function
# TODO: create a parameter for pension growth rates -- a single param should do
def delta_point(r, Y, g_n, g_y, p):
    """
    Compute growth rate used for contributions to points system pension
    """
    # TODO: Add option to allow use to enter growth rate amount
    # Also to allow rate to vary by year
    # Do this for all these growth rates for each system
    # Might also allow for option to grow at per capital GDP growth rate
    if p.points_growth_rate == "r":
        delta_point = r
    elif p.points_growth_rate == "Curr GDP":
        delta_point = (Y[1:] - Y[:-1]) / Y[:-1]
    elif p.points_growth_rate == "LR GDP":
        delta_point = g_y + g_n
    else:
        delta_point = g_y + g_n

    return delta_point


def g_ndc(r, Y, p):
    """
    Compute growth rate used for contributions to NDC pension
    """
    if p.ndc_growth_rate == "r":
        g_ndc = r
    elif p.ndc_growth_rate == "Curr GDP":
        g_ndc = (Y[1:] - Y[:-1]) / Y[:-1]
    elif p.ndc_growth_rate == "LR GDP":
        g_ndc = p.g_y + p.g_n
    else:
        g_ndc = p.g_y + p.g_n

    return g_ndc


def g_dir(r, Y, p):
    """
    Compute growth rate used for contributions to NDC pension
    """
    if p.dir_growth_rate == "r":
        g_dir = r
    elif p.dir_growth_rate == "Curr GDP":
        g_dir = (Y[1:] - Y[:-1]) / Y[:-1]
    elif p.dir_growth_rate == "LR GDP":
        g_dir = p.g_y + p.g_n
    else:
        g_dir = p.g_y + p.g_n

    return g_dir


def delta_ret(r, Y, p):
    """
    Compute conversion coefficient for the NDC pension amount
    """
    surv_rates = 1 - p.mort_rates_SS
    dir_delta_s_empty = np.zeros(p.S - p.retire + 1)
    g_dir_value = g_dir(r, Y, p)
    print("G dir value type = ", type(p.S))
    print("G dir value type = ", type(p.retire))
    print("G dir value type = ", type(surv_rates))
    print("G dir value type = ", type(g_dir_value))
    print("G dir value type = ", type(dir_delta_s_empty))
    dir_delta = delta_ret_loop(
        p.S, p.retire, surv_rates, g_dir_value, dir_delta_s_empty
    )
    delta_ret = 1 / (dir_delta - p.k_ret)

    return delta_ret


@numba.jit
def deriv_DB_loop(
    w, e, S, S_ret, per_rmn, last_career_yrs, rep_rate_py, yr_contr
):
    d_theta = np.zeros(per_rmn)
    num_per_retire = S - S_ret
    for s in range(per_rmn):
        d_theta[s] = w[s] * e[s] * rep_rate_py * (yr_contr / last_career_yrs)
    d_theta[-num_per_retire:] = 0.0

    return d_theta


@numba.jit
def deriv_PS_loop(w, e, S, S_ret, per_rmn, d_theta, vpoint, factor):
    # TODO: do we need these constants or can we scale vpoint to annual??
    for s in range((S - per_rmn), S_ret):
        d_theta[s] = (w[s] * e[s] * vpoint * MONTHS_IN_A_YEAR) / (
            factor * THOUSAND
        )

    return d_theta


@numba.jit
def deriv_NDC_loop(w, e, per_rmn, S, S_ret, tau_p, g_ndc, delta_ret, d_theta):

    for s in range((S - per_rmn), S_ret):
        d_theta[s - (S - per_rmn)] = (
            tau_p
            * w[s - (S - per_rmn)]
            * e[s - (S - per_rmn)]
            * delta_ret
            * (1 + g_ndc) ** (S_ret - s - 1)
        )

    return d_theta


@numba.jit
def delta_ret_loop(S, S_ret, surv_rates, g_dir_value, dir_delta_s):

    cumul_surv_rates = np.ones(S - S_ret + 1)
    for s in range(S - S_ret + 1):
        surv_rates_vec = surv_rates[S_ret : S_ret + s + 1]
        surv_rates_vec[0] = 1.0
        cumul_surv_rates[s] = np.prod(surv_rates_vec)
        cumul_g_y = np.ones(S - S_ret + 1)
        cumul_g_y[s] = (1 / (1 + g_dir_value)) ** s
        dir_delta_s[s] = cumul_surv_rates[s] * cumul_g_y[s]
    dir_delta = dir_delta_s.sum()
    return dir_delta


@numba.jit
def PS_1dim_loop(w, e, n, S_ret, S, g_y, vpoint, factor, L_inc_avg_s, PS):
    # TODO: do we need these constants or can we scale vpoint to annual??
    for u in range(S_ret, S):
        for s in range(S_ret):
            L_inc_avg_s[s] = w[s] / np.exp(g_y * (u - s)) * e[s] * n[s]
        PS[u] = (MONTHS_IN_A_YEAR * vpoint * L_inc_avg_s.sum()) / (
            factor * THOUSAND
        )

    return PS


@numba.jit
def PS_2dim_loop(w, e, n, S_ret, S, J, g_y, vpoint, factor, L_inc_avg_sj, PS):
    # TODO: do we need these constants or can we scale vpoint to annual??
    for u in range(S_ret, S):
        for s in range(S_ret):
            L_inc_avg_sj[s, :] = (
                w[s] / np.exp(g_y * (u - s)) * e[s, :] * n[s, :]
            )
        PS[u, :] = (MONTHS_IN_A_YEAR * vpoint * L_inc_avg_sj.sum(axis=0)) / (
            factor * THOUSAND
        )

    return PS


@numba.jit
def DB_1dim_loop(
    w,
    e,
    n,
    S_ret,
    S,
    g_y,
    L_inc_avg_s,
    L_inc_avg,
    DB,
    last_career_yrs,
    rep_rate_py,
    yr_contr,
):

    for u in range(S_ret, S):
        for s in range(S_ret - last_career_yrs, S_ret):
            L_inc_avg_s[s - (S_ret - last_career_yrs)] = (
                w[s] / np.exp(g_y * (u - s)) * e[s] * n[s]
            )
        L_inc_avg = L_inc_avg_s.sum() / last_career_yrs
        rep_rate = yr_contr * rep_rate_py
        DB[u] = rep_rate * L_inc_avg

    return DB


@numba.jit
def DB_2dim_loop(
    w,
    e,
    n,
    S_ret,
    S,
    g_y,
    L_inc_avg_sj,
    L_inc_avg,
    DB,
    last_career_yrs,
    rep_rate_py,
    yr_contr,
):

    for u in range(S_ret, S):
        for s in range(S_ret - last_career_yrs, S_ret):
            L_inc_avg_sj[s - (S_ret - last_career_yrs), :] = (
                w[s] / np.exp(g_y * (u - s)) * e[s, :] * n[s, :]
            )
        L_inc_avg = L_inc_avg_sj.sum(axis=0) / last_career_yrs
        rep_rate = yr_contr * rep_rate_py
        DB[u, :] = rep_rate * L_inc_avg

    return DB


@numba.jit
def NDC_1dim_loop(w, e, n, S_ret, S, g_y, tau_p, g_ndc, delta_ret, NDC_s, NDC):

    for u in range(S_ret, S):
        for s in range(0, S_ret):
            NDC_s[s] = (
                tau_p
                * (w[s] / np.exp(g_y * (u - s)))
                * e[s]
                * n[s]
                * ((1 + g_ndc) ** (S_ret - s - 1))
            )
        NDC[u] = delta_ret * NDC_s.sum()
    return NDC


@numba.jit
def NDC_2dim_loop(
    w, e, n, S_ret, S, g_y, tau_p, g_ndc, delta_ret, NDC_sj, NDC
):
    for u in range(S_ret, S):
        for s in range(0, S_ret):
            NDC_sj[s, :] = (
                tau_p
                * (w[s] / np.exp(g_y * (u - s)))
                * e[s, :]
                * n[s, :]
                * ((1 + g_ndc) ** (S_ret - s - 1))
            )
        NDC[u, :] = delta_ret * NDC_sj.sum(axis=0)
    return NDC
