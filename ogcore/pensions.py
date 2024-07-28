# Packages
import numpy as np
import numba
from ogcore import utils

# set constants
MONTHS_IN_A_YEAR = 12
THOUSAND = 1000


def replacement_rate_vals(nssmat, wss, factor_ss, j, p):
    r"""
    Calculates replacement rate values for the social security system.

    .. math::
        \theta_{j,R,t+R} = \frac{PIA_{j,R,t+R} \times 12}{factor \times w_{t+R}}

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
    equiv_periods = int(round((p.S / 80.0) * p.avg_earn_num_years)) - 1
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
    # Set the maximum monthly replacement rate from SS benefits tables
    PIA[PIA > p.PIA_maxpayment] = p.PIA_maxpayment
    if p.PIA_minpayment != 0.0:
        PIA[PIA < p.PIA_minpayment] = p.PIA_minpayment
    theta = (PIA * (12.0 * p.S / 80.0)) / (factor_ss * wss)
    return theta


def pension_amount(r, w, n, Y, theta, t, j, shift, method, e, factor, p):
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
        pension = DB_amount(w, e, n, j, p)
    elif p.pension_system == "Notional Defined Contribution":
        pension = NDC_amount(w, e, n, r, Y, j, p)
    elif p.pension_system == "Points System":
        pension = PS_amount(w, e, n, j, factor, p)
    else:
        raise ValueError(
            "pension_system must be one of the following: "
            "'US-Style Social Security', 'Defined Benefits', "
            "'Notional Defined Contribution', 'Points System'"
        )
    return pension


def SS_amount(w, n, theta, t, j, shift, method, e, p):
    r"""
    Calculate public pension benefit amounts for each household under
    a US-style social security system.

    .. mathL::
        pension_{j,s,t} = \theta_j \times w_t \quad \forall s > R

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
    r"""
    Calculate public pension from a defined benefits system.

    .. math::
        pension{j,s,t} = \biggl[\frac{\sum_{s=R-ny}^{R-1}w_{t}e_{j,s,t}
            n_{j,s,t}}{ny}\biggr]\times Cy \times \alpha_{DB} \quad \forall s > R

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        DB (Numpy array): pension amount for each household
    """
    L_inc_avg = np.zeros(0)
    # Adjustment to turn years into model periods
    # TODO: could add this to parameters.py at some point
    equiv_periods = int(round((p.S / 80.0) * p.avg_earn_num_years)) - 1
    equiv_yr_contrib = int(round((p.S / 80.0) * p.yr_contrib)) - 1
    L_inc_avg_s = np.zeros(equiv_periods)

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]
        # TODO: think about how to handle setting w_preTP and n_preTP
        # TODO: will need to update how the e matrix is handled here
        # and else where to allow for it to be time varying
        w_S = np.append((p.w_preTP * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)

        DB = np.zeros(p.S)
        DB = DB_1dim_loop(
            w_S,
            p.e[:, j],
            n_S,
            p.retire,
            p.S,
            p.g_y,
            L_inc_avg_s,
            L_inc_avg,
            DB,
            equiv_periods,
            p.alpha_db,
            equiv_yr_contrib,
        )
        DB = DB[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            DB = np.zeros(p.S)
            DB = DB_1dim_loop(
                w,
                e,
                n,
                p.retire,
                p.S,
                p.g_y,
                L_inc_avg_s,
                L_inc_avg,
                DB,
                equiv_periods,
                p.alpha_db,
                equiv_yr_contrib,
            )

        elif np.ndim(n) == 2:
            DB = np.zeros((p.S, p.J))
            L_inc_avg_sj = np.zeros((equiv_periods, p.J))
            DB = DB_2dim_loop(
                w,
                e,
                n,
                p.retire,
                p.S,
                p.g_y,
                L_inc_avg_sj,
                L_inc_avg,
                DB,
                equiv_periods,
                p.alpha_db,
                equiv_yr_contrib,
            )

    return DB


def NDC_amount(w, e, n, r, Y, j, p):
    r"""
    Calculate public pension from a notional defined contribution
    system.

    .. math::
        pension{j,s,t} = \biggl[\sum_{s=E}^{R-1}\tau^{p}_{t}w_{t}
            e_{j,s,t}n_{j,s,t}(1 + g_{NDC,t})^{R-s-1}\biggr]\delta_{R, t} \quad \forall s > R

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        r (array_like): interest rate
        Y (array_like): GDP
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        NDC (Numpy array): pension amount for each household
    """
    g_ndc_amount = g_ndc(r, Y, p)
    delta_ret_amount = delta_ret(r, Y, p)

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]

        w_S = np.append((p.w_preTP * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)

        NDC_s = np.zeros(p.retire)
        NDC = np.zeros(p.S)
        NDC = NDC_1dim_loop(
            w_S,
            p.e[:, j],
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
    r"""
    Calculate public pension from a points system.

    .. math::
        pension{j,s,t} = \sum_{s=E}^{R-1}w_{t}e_{j,s,t}n_{j,s,t}\times v_{t} \quad \forall s > R

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        j (int): index of lifetime income group
        factor (scalar): scaling factor converting model units to
            dollars
        p (OG-Core Specifications object): model parameters

    Returns:
        PS (Numpy array): pension amount for each household
    """

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]
        w_S = np.append((p.w_preTP * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)
        L_inc_avg_s = np.zeros(p.retire)
        PS = np.zeros(p.S)
        PS = PS_1dim_loop(
            w_S,
            p.e[:, j],
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

    Args:
        r (array_like): interest rate
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        Y (array_like): GDP
        per_rmn (int): number of periods remaining in the model
        factor (scalar): scaling factor converting model units to

    Returns:
        d_theta (Numpy array): change in pension benefits for another
            unit of labor supply
    """
    # TODO: Add SS here...
    if p.pension_system == "Defined Benefits":
        d_theta = deriv_DB(w, e, per_rmn, p)
        d_theta = d_theta[-per_rmn:]
    elif p.pension_system == "Notional Defined Contribution":
        d_theta = deriv_NDC(r, w, e, Y, per_rmn, p)
    elif p.pension_system == "Points System":
        d_theta = deriv_PS(w, e, per_rmn, factor, p)
    else:
        raise ValueError(
            "pension_system must be one of the following: "
            "'US-style Social Security', 'Defined Benefits', "
            "'Notional Defined Contribution', 'Points System'"
        )

    return d_theta


def deriv_NDC(r, w, e, Y, per_rmn, p):
    r"""
    Change in NDC pension benefits for another unit of labor supply

    .. math::
        \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
            \begin{cases}
            \tau^{p}_{t}w_{t}e_{j,s}(1+g_{NDC,t})^{u - s}\delta_{R,t}, & \text{if}\ s<R-1 \\
            0, & \text{if}\ s \geq R \\
            \end{cases}

    Args:
        r (array_like): interest rate
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        Y (array_like): GDP
        per_rmn (int): number of periods remaining in the model
        p (OG-Core Specifications object): model parameters

    Returns:
        d_theta (Numpy array): change in NDC pension benefits for
            another unit of labor supply
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
    r"""
    Change in DB pension benefits for another unit of labor supply

    .. math::
        \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
            \begin{cases}
                0 , & \text{if}\ s < R - Cy \\
                w_{t}e_{j,s}\alpha_{DB}\times \frac{Cy}{ny}, & \text{if}\  R - Cy <= s < R  \\
                0, & \text{if}\ s \geq R \\
            \end{cases}

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        per_rmn (int): number of periods remaining in the model
        p (OG-Core Specifications object): model parameters

    Returns:
        d_theta: change in DB pension benefits for another unit of labor
            supply
    """
    equiv_periods = int(round((p.S / 80.0) * p.avg_earn_num_years)) - 1
    equiv_yr_contrib = int(round((p.S / 80.0) * p.yr_contrib)) - 1
    if per_rmn < (p.S - p.retire + 1):
        d_theta = np.zeros(p.S)
    else:
        d_theta = deriv_DB_loop(
            w,
            e,
            p.S,
            p.retire,
            per_rmn,
            equiv_periods,
            p.alpha_db,
            equiv_yr_contrib,
        )
    return d_theta


def deriv_PS(w, e, per_rmn, factor, p):
    r"""
    Change in points system pension benefits for another unit of
    labor supply

    .. math::
        \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
            \begin{cases}
                0 , & \text{if}\ s < R \\
                w_{t}e_{j,s}v_{t}, & \text{if}\ s \geq R \\
            \end{cases}

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        per_rmn (int): number of periods remaining in the model
        factor (scalar): scaling factor converting model units to
        p (OG-Core Specifications object): model parameters

    Returns:
        d_theta (Numpy array): change in points system pension benefits
            for another unit of labor supply

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
    r"""
    Compute growth rate used for contributions to points system pension

    Args:
        r (array_like): interest rate
        Y (array_like): GDP
        g_n (array_like): population growth rate
        g_y (array_like): GDP growth rate
        p (OG-Core Specifications object): model parameters

    Returns:
        delta_point (Numpy array): growth rate used for contributions to
            points
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

    Args:
        r (array_like): interest rate
        Y (array_like): GDP
        p (OG-Core Specifications object): model parameters

    Returns:
        g_ndc (Numpy array): growth rate used for contributions to NDC

    """
    if p.ndc_growth_rate == "r":
        g_ndc = r[-1]
    elif p.ndc_growth_rate == "Curr GDP":
        g_ndc = (Y[1:] - Y[:-1]) / Y[:-1]
    elif p.ndc_growth_rate == "LR GDP":
        g_ndc = p.g_y[-1] + p.g_n[-1]
    else:
        g_ndc = p.g_y[-1] + p.g_n[-1]

    return g_ndc


def g_dir(r, Y, g_y, g_n, dir_growth_rate):
    """
    Compute growth rate used for contributions to NDC pension

    Args:
        r (array_like): interest rate
        Y (array_like): GDP
        g_y (array_like): GDP growth rate
        g_n (array_like): population growth rate
        dir_growth_rate (str): growth rate used for contributions to NDC

    Returns:
        g_dir (Numpy array): growth rate used for contributions to NDC

    """
    if dir_growth_rate == "r":
        g_dir = r[-1]
    elif dir_growth_rate == "Curr GDP":
        g_dir = (Y[1:] - Y[:-1]) / Y[:-1]
    elif dir_growth_rate == "LR GDP":
        g_dir = g_y[-1] + g_n[-1]
    else:
        g_dir = g_y[-1] + g_n[-1]

    return g_dir


def delta_ret(r, Y, p):
    r"""
    Compute conversion coefficient for the NDC pension amount

    .. math::
        \delta_{R} = (dir_{R} + ind_{R} - k)^{-1}

    Args:
        r (array_like): interest rate
        Y (array_like): GDP
        p (OG-Core Specifications object): model parameters

    Returns:
        delta_ret (Numpy array): conversion coefficient for the NDC
            pension amount

    """
    surv_rates = 1 - p.mort_rates_SS
    dir_delta_s_empty = np.zeros(p.S - p.retire + 1)
    g_dir_value = g_dir(r, Y, p.g_y, p.g_n, p.dir_growth_rate)
    dir_delta = delta_ret_loop(
        p.S, p.retire, surv_rates, g_dir_value, dir_delta_s_empty
    )
    delta_ret = 1 / (dir_delta + p.indR - p.k_ret)

    return delta_ret


@numba.jit
def deriv_DB_loop(
    w, e, S, S_ret, per_rmn, avg_earn_num_years, alpha_db, yr_contr
):
    """
    Change in DB pension benefits for another unit of labor supply

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        S (int): number of periods in the model
        S_ret (int): retirement age
        per_rmn (int): number of periods remaining in the model
        avg_earn_num_years (int): number of years AIME is computed from
        alpha_db (scalar): replacement rate
        yr_contr (scalar): years of contribution

    Returns:
        d_theta (Numpy array): change in DB pension benefits for
            another unit of labor supply
    """
    d_theta = np.zeros(per_rmn)
    print("Year contribution: ", yr_contr)
    print("Average earnings years: ", avg_earn_num_years)
    num_per_retire = S - S_ret
    for s in range(per_rmn):
        d_theta[s] = w[s] * e[s] * alpha_db * (yr_contr / avg_earn_num_years)
    d_theta[-num_per_retire:] = 0.0

    return d_theta


@numba.jit
def deriv_PS_loop(w, e, S, S_ret, per_rmn, d_theta, vpoint, factor):
    """
    Change in points system pension benefits for another unit of
    labor supply

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        S (int): number of periods in the model
        S_ret (int): retirement age
        per_rmn (int): number of periods remaining in the model
        d_theta (Numpy array): change in points system pension benefits
            for another unit of labor supply
        vpoint (scalar): value of points
        factor (scalar): scaling factor converting model units to
            local currency

    Returns:
        d_theta (Numpy array): change in points system pension benefits
            for another unit of labor supply

    """
    # TODO: do we need these constants or can we scale vpoint to annual??
    for s in range((S - per_rmn), S_ret):
        d_theta[s] = (w[s] * e[s] * vpoint * MONTHS_IN_A_YEAR) / (
            factor * THOUSAND
        )

    return d_theta


@numba.jit
def deriv_NDC_loop(
    w, e, per_rmn, S, S_ret, tau_p, g_ndc_value, delta_ret_value, d_theta
):
    """
    Change in NDC pension benefits for another unit of labor supply

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        per_rmn (int): number of periods remaining in the model
        S (int): number of periods in the model
        S_ret (int): retirement age
        tau_p (scalar): tax rate
        g_ndc_value (scalar): growth rate of NDC pension
        delta_ret_value (scalar): conversion coefficient for the NDC
            pension amount
        d_theta (Numpy array): change in NDC pension benefits for
            another unit of labor supply

    Returns:
        d_theta (Numpy array): change in NDC pension benefits for
            another unit of labor supply

    """
    for s in range((S - per_rmn), S_ret):
        d_theta[s - (S - per_rmn)] = (
            tau_p
            * w[s - (S - per_rmn)]
            * e[s - (S - per_rmn)]
            * delta_ret_value
            * (1 + g_ndc_value) ** (S_ret - s - 1)
        )

    return d_theta


@numba.jit
def delta_ret_loop(S, S_ret, surv_rates, g_dir_value, dir_delta_s):
    """
    Compute conversion coefficient for the NDC pension amount

    Args:
        S (int): number of periods in the model
        S_ret (int): retirement age
        surv_rates (Numpy array): survival rates
        g_dir_value (scalar): growth rate of NDC pension
        dir_delta_s (Numpy array): conversion coefficient for the NDC
            pension amount

    Returns:
        dir_delta (scalar): conversion coefficient for the NDC pension
            amount
    """
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
    """
    Calculate public pension from a points system.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        S_ret (int): retirement age
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        vpoint (scalar): value of points
        factor (scalar): scaling factor converting model units to
            local currency
        L_inc_avg_s (Numpy array): average labor income
        PS (Numpy array): pension amount for each household

    Returns:
        PS (Numpy array): pension amount for each household

    """
    # TODO: do we need these constants or can we scale vpoint to annual??
    for u in range(S_ret, S):
        # TODO: allow for g_y to be time varying
        for s in range(S_ret):
            L_inc_avg_s[s] = w[s] / np.exp(g_y[-1] * (u - s)) * e[s] * n[s]
        PS[u] = (MONTHS_IN_A_YEAR * vpoint * L_inc_avg_s.sum()) / (
            factor * THOUSAND
        )

    return PS


@numba.jit
def PS_2dim_loop(w, e, n, S_ret, S, J, g_y, vpoint, factor, L_inc_avg_sj, PS):
    """
    Calculate public pension from a points system.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        S_ret (int): retirement age
        S (int): number of periods in the model
        J (int): number of lifetime income groups
        g_y (array_like): GDP growth rate
        vpoint (scalar): value of points
        factor (scalar): scaling factor converting model units to
            local currency
        L_inc_avg_sj (Numpy array): average labor income
        PS (Numpy array): pension amount for each household

    Returns:
        PS (Numpy array): pension amount for each household

    """
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
    avg_earn_num_years,
    alpha_db,
    yr_contr,
):
    """
    Calculate public pension from a defined benefits system.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        S_ret (int): retirement age
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        L_inc_avg_s (Numpy array): average labor income
        L_inc_avg (scalar): average labor income
        DB (Numpy array): pension amount for each household
        avg_earn_num_years (int): number of years AIME is computed from
        alpha_db (scalar): replacement rate
        yr_contr (scalar): years of contribution

    Returns:
        DB (Numpy array): pension amount for each household
    """
    for u in range(S_ret, S):
        for s in range(S_ret - avg_earn_num_years, S_ret):
            # TODO: pass t so that can pull correct g_y value
            # Just need to make if doing over time path makes sense
            # or if should just do SS
            L_inc_avg_s[s - (S_ret - avg_earn_num_years)] = (
                w[s] / np.exp(g_y[-1] * (u - s)) * e[s] * n[s]
            )
        L_inc_avg = L_inc_avg_s.sum() / avg_earn_num_years
        rep_rate = yr_contr * alpha_db
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
    avg_earn_num_years,
    alpha_db,
    yr_contr,
):
    """
    Calculate public pension from a defined benefits system.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        S_ret (int): retirement age
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        L_inc_avg_sj (Numpy array): average labor income
        L_inc_avg (scalar): average labor income
        DB (Numpy array): pension amount for each household
        avg_earn_num_years (int): number of years AIME is computed from
        alpha_db (scalar): replacement rate
        yr_contr (scalar): years of contribution

    Returns:
        DB (Numpy array): pension amount for each household

    """
    for u in range(S_ret, S):
        for s in range(S_ret - avg_earn_num_years, S_ret):
            L_inc_avg_sj[s - (S_ret - avg_earn_num_years), :] = (
                w[s] / np.exp(g_y * (u - s)) * e[s, :] * n[s, :]
            )
        L_inc_avg = L_inc_avg_sj.sum(axis=0) / avg_earn_num_years
        rep_rate = yr_contr * alpha_db
        DB[u, :] = rep_rate * L_inc_avg

    return DB


@numba.jit
def NDC_1dim_loop(w, e, n, S_ret, S, g_y, tau_p, g_ndc, delta_ret, NDC_s, NDC):
    """
    Calculate public pension from a notional defined contribution

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        S_ret (int): retirement age
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        tau_p (scalar): tax rate
        g_ndc (scalar): growth rate of NDC pension
        delta_ret (scalar): conversion coefficient for the NDC pension amount
        NDC_s (Numpy array): average labor income
        NDC (Numpy array): pension amount for each household

    Returns:
        NDC (Numpy array): pension amount for each household

    """
    for u in range(S_ret, S):
        for s in range(0, S_ret):
            # TODO: update so can take g_y from period t
            NDC_s[s] = (
                tau_p
                * (w[s] / np.exp(g_y[-1] * (u - s)))
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
    """
    Calculate public pension from a notional defined contribution

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply
        S_ret (int): retirement age
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        tau_p (scalar): tax rate
        g_ndc (scalar): growth rate of NDC pension
        delta_ret (scalar): conversion coefficient for the NDC pension amount
        NDC_sj (Numpy array): average labor income
        NDC (Numpy array): pension amount for each household

    Returns:
        NDC (Numpy array): pension amount for each household

    """
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
