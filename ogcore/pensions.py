# Packages
import numpy as np
import numba
from ogcore import utils


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


def pension_amount(w, n, t, j, shift, method, e, p):
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


def DB_amount(self, households, firms, w, e, n, j_ind):
    """
    Calculate public pension from a defined benefits system.
    """
    L_inc_avg = np.zeros(0)
    L_inc_avg_s = np.zeros(self.last_career_yrs)

    if n.shape[0] < households.S:
        per_rmn = n.shape[0]
        w_S = np.append((households.w_preTP *
                                np.ones(households.S))[:(-per_rmn)], w)
        n_S = np.append(households.n_preTP[:(-per_rmn), j_ind], n)

        DB_s = np.zeros(households.S_ret)
        DB = np.zeros(households.S)

        DB = DB_1dim_loop(w_S, households.emat[:, j_ind], n_S,
                    households.S_ret, households.S, firms.g_y, L_inc_avg_s,
                    L_inc_avg, DB_s, DB, self.last_career_yrs, self.rep_rate,
                    self.rep_rate_py, self.yr_contr)
        DB = DB[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            DB_s = np.zeros(households.S_ret)
            DB = np.zeros(households.S)
            DB = DB_1dim_loop(
                w, e, n, households.S_ret, households.S, firms.g_y,
                L_inc_avg_s, L_inc_avg, DB_s, DB, self.last_career_yrs,
                        self.rep_rate, self.rep_rate_py, self.yr_contr)

        elif np.ndim(n) == 2:
            DB_sj = np.zeros((households.S_ret, households.J))
            DB = np.zeros((households.S, households.J))
            L_inc_avg_sj = np.zeros((self.last_career_yrs, households.J))
            DB = DB_2dim_loop(
                w, e, n, households.S_ret, households.S, firms.g_y,
                L_inc_avg_sj, L_inc_avg, DB_sj, DB, self.last_career_yrs,
                        self.rep_rate, self.rep_rate_py, self.yr_contr)

    return DB


def NDC_amount(self, demographics, households, firms, w, e,
            n, r, Y, j_ind):
    """
     Calculate public pension from a notional defined contribution
     system.
    """
    self.get_g_ndc(r, Y, demographics.g_n_SS, firms.g_y, )
    self.get_delta_ret(r, Y, demographics, households, firms)

    if n.shape[0] < households.S:
        per_rmn = n.shape[0]

        w_S = np.append((households.w_preTP *
                                np.ones(households.S))[:(-per_rmn)], w)
        n_S = np.append(households.n_preTP[:(-per_rmn), j_ind], n)

        NDC_s = np.zeros(households.S_ret)
        NDC = np.zeros(households.S)
        NDC = NDC_1dim_loop(
                w_S, households.emat[:, j_ind], n_S, households.S_ret,
                households.S, firms.g_y, self.tau_p, self.g_ndc,
                self.delta_ret, NDC_s, NDC)
        NDC = NDC[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            NDC_s = np.zeros(households.S_ret)
            NDC = np.zeros(households.S)
            NDC = NDC_1dim_loop(
                w, e, n, households.S_ret, households.S, firms.g_y,
                self.tau_p, self.g_ndc, self.delta_ret, NDC_s, NDC)
        elif np.ndim(n) == 2:
            NDC_sj = np.zeros((households.S_ret, households.J))
            NDC = np.zeros((households.S, households.J))
            NDC = NDC_2dim_loop(
                w, e, n, households.S_ret, households.S, firms.g_y,
                self.tau_p, self.g_ndc, self.delta_ret, NDC_sj, NDC)

    return NDC


def PS_amount(self, demographics, households, firms, w, e, n, r, Y, lambdas,
            j_ind, factor):
    """
     Calculate public pension from a points system.
    """

    if n.shape[0] < households.S:
        per_rmn = n.shape[0]
        w_S = np.append((households.w_preTP *
                                np.ones(households.S))[:(-per_rmn)], w)
        n_S = np.append(households.n_preTP[:(-per_rmn), j_ind], n)
        L_inc_avg_s = np.zeros(households.S_ret)
        PPB = np.zeros(households.S)
        PPB = PPB_1dim_loop(w_S, households.emat[:, j_ind], n_S,
                            households.S_ret, households.S,
                            firms.g_y, self.vpoint,
                            factor, L_inc_avg_s, PPB)
        PPB = PPB[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            L_inc_avg_s = np.zeros(households.S_ret)
            PPB = np.zeros(households.S)
            PPB = PPB_1dim_loop(w, e, n, households.S_ret, households.S,
                            firms.g_y, self.vpoint,
                            factor, L_inc_avg_s, PPB)

        elif np.ndim(n) == 2:
            L_inc_avg_sj = np.zeros((households.S_ret, households.J))
            PPB = np.zeros((households.S, households.J))
            PPB = PPB_2dim_loop(w, e, n, households.S_ret, households.S,
                            households.J, firms.g_y, self.vpoint, factor,
                            L_inc_avg_sj, PPB)

    return PPB


def deriv_theta(self, demographics, households, firms, r, w, e,
                Y, per_rmn, factor):
    '''
    Change in pension benefits for another unit of labor supply for
    pension system selected
    '''
    if self.pension_system == "DB":
        d_theta = self.deriv_DB(households, firms, w, e, per_rmn)
        d_theta = d_theta[-per_rmn:]
    elif self.pension_system == "NDC":
        d_theta = self.deriv_NDC(demographics, households, firms,
                                    r, w, e, Y, per_rmn)
    elif self.pension_system == "PS":
        d_theta = self.deriv_PPB(demographics, households, firms,
                                    w, e, per_rmn, factor)

    return d_theta


def deriv_NDC(self, demographics, households, firms, r, w, e, Y,
                per_rmn):
    '''
    Change in NDC pension benefits for another unit of labor supply
    '''
    if per_rmn == 1:
        d_theta = 0
    elif per_rmn < (households.S - households.S_ret + 1):
        d_theta = np.zeros(per_rmn)
    else:
        d_theta_empty = np.zeros(per_rmn)
        self.get_delta_ret(r, Y, demographics, households, firms)
        self.get_g_ndc(r, Y, demographics.g_n_SS, firms.g_y)
        d_theta = deriv_NDC_loop(
            w, e, per_rmn, households.S,
            households.S_ret, firms.g_y, self.tau_p,
            self.g_ndc, self.delta_ret, d_theta_empty)

    return d_theta


def deriv_DB(self, households, firms, w, e, per_rmn):
    '''
    Change in DB pension benefits for another unit of labor supply
    '''

    if per_rmn < (households.S - households.S_ret + 1):
        d_theta = np.zeros(households.S)
    else:
        d_theta_empty = np.zeros(households.S)
        d_theta = deriv_DB_loop(
            w, e, households.S,households.S_ret, per_rmn, firms.g_y,
            d_theta_empty, self.last_career_yrs, self.rep_rate_py,
            self.yr_contr)
    return d_theta


def deriv_PS(self, demographics, households, firms, w, e,
                per_rmn, factor):
    '''
    Change in points system pension benefits for another unit of labor supply
    '''

    if per_rmn < (households.S - households.S_ret + 1):
        d_theta = np.zeros(households.S)
    else:
        d_theta_empty = np.zeros(households.S)
        d_theta = deriv_PS_loop(w, e, households.S, households.S_ret, per_rmn,
                                firms.g_y, d_theta_empty,
                                self.vpoint, factor)
        d_theta = d_theta[-per_rmn:]

    return d_theta


def delta_point(self, r, Y, g_n, g_y):
    '''
    Compute growth rate used for contributions to points system pension
    '''
    # TODO: Add option to allow use to enter growth rate amount
    # Also to allow rate to vary by year
    # Do this for all these growth rates for each system
    # Might also allow for option to grow at per capital GDP growth rate
    if self.points_growth_rate == 'r':
        self.delta_point = r
    elif self.points_growth_rate == 'Curr GDP':
        self.delta_point = (Y[1:] - Y[:-1]) / Y[:-1]
    elif self.points_growth_rate == 'LR GDP':
        self.delta_point = g_y + g_n
    else:
        self.delta_point = g_y + g_n


def g_ndc(self, r, Y, g_n, g_y,):
    '''
    Compute growth rate used for contributions to NDC pension
    '''
    if self.ndc_growth_rate == 'r':
        self.g_ndc = r
    elif self.ndc_growth_rate == 'Curr GDP':
        self.g_ndc = (Y[1:] - Y[:-1]) / Y[:-1]
    elif self.ndc_growth_rate == 'LR GDP':
        self.g_ndc = g_y + g_n
    else:
        self.g_ndc = g_y + g_n


def g_dir(self, r, Y, g_n, g_y):
    '''
    Compute growth rate used for contributions to NDC pension
    '''
    if self.dir_growth_rate == 'r':
        self.g_dir = r
    elif self.dir_growth_rate == 'Curr GDP':
        self.g_dir = (Y[1:] - Y[:-1]) / Y[:-1]
    elif self.dir_growth_rate == 'LR GDP':
        self.g_dir = g_y + g_n
#            self.g_dir = 0.015
    else:
        self.g_dir = g_y + g_n


def delta_ret(self, r, Y, demographics, households, firms):
    '''
    Compute conversion coefficient for the NDC pension amount
    '''
    surv_rates = 1 - demographics.mort_rates_SS
    dir_delta_s_empty = np.zeros(households.S - households.S_ret + 1)
    self.get_g_dir(r, Y, demographics.g_n_SS, firms.g_y)
    dir_delta = delta_ret_loop(
        households.S, households.S_ret, surv_rates, self.g_dir,
        dir_delta_s_empty)
    #####TODO: formula below needs to be changed if we had separate calculations for both sexes
    self.delta_ret = 1 / (dir_delta - self.k_ret)
#        print("self.delta_ret", self.delta_ret)


def pension_benefit(self, demographics, households, firms, w, e, n,
                    r, Y, lambdas, j_ind, factor):
    if self.pension_system == "DB":
        theta = self.get_DB(households, firms, w, e, n, j_ind)
    elif self.pension_system == "NDC":
        theta = self.get_NDC(
            demographics, households, firms, w, e, n, r, Y, j_ind)
    elif self.pension_system == "PS":
        theta = self.get_PPB(demographics, households, firms,
                                w, e, n, r, Y, lambdas, j_ind, factor)

    return theta


@numba.jit
def deriv_DB_loop(w, e, S, S_ret, per_rmn, g_y, d_theta,
                  last_career_yrs, rep_rate_py, yr_contr):
    d_theta = np.zeros(per_rmn)
    num_per_retire = S - S_ret
    for s in range(per_rmn):
        d_theta[s] = w[s] * e[s] * rep_rate_py * (yr_contr / last_career_yrs)
    d_theta[-num_per_retire:] = 0.0

    return d_theta


@numba.jit
def deriv_PS_loop(w, e, S, S_ret, per_rmn, g_y, d_theta, vpoint, factor):

    for s in range((S - per_rmn), S_ret):
        d_theta[s] = ((w[s] * e[s] * vpoint * constants.MONTHS_IN_A_YEAR) /
               (factor * constants.THOUSAND))

    return d_theta


@numba.jit
def deriv_NDC_loop(w, e, per_rmn, S, S_ret, g_y, tau_p, g_ndc,
                   delta_ret, d_theta):

    for s in range((S - per_rmn), S_ret):
        d_theta[s - (S - per_rmn)] = (tau_p * w[s - (S - per_rmn)] *
                                   e[s - (S - per_rmn)] * delta_ret *
                                   (1 + g_ndc) ** (S_ret - s - 1))

    return d_theta


@numba.jit
def delta_ret_loop(S, S_ret, surv_rates, g_dir, dir_delta_s):

    cumul_surv_rates = np.ones(S - S_ret + 1)
    for s in range(S - S_ret + 1):
        surv_rates_vec = surv_rates[S_ret: S_ret + s + 1]
        surv_rates_vec[0] = 1.0
        cumul_surv_rates[s] = np.prod(surv_rates_vec)
        cumul_g_y = np.ones(S - S_ret + 1)
        cumul_g_y[s] = (1/(1 + g_dir)) ** s
        dir_delta_s[s] = cumul_surv_rates[s] * cumul_g_y[s]
    dir_delta = dir_delta_s.sum()
    return dir_delta

@numba.jit
def PPB_1dim_loop(w, e, n, S_ret, S, g_y, vpoint, factor, L_inc_avg_s,
                  PPB):

    for u in range(S_ret, S):
        for s in range(S_ret):
            L_inc_avg_s[s] = \
                        w[s] / np.exp(g_y * (u - s)) * e[s] * n[s]
        PPB[u] =  ((constants.MONTHS_IN_A_YEAR * vpoint * L_inc_avg_s.sum())
                        / (factor * constants.THOUSAND))

    return PPB

@numba.jit
def PPB_2dim_loop(w, e, n, S_ret, S, J, g_y, vpoint, factor,
                  L_inc_avg_sj, PPB):

    for u in range(S_ret, S):
        for s in range(S_ret):
            L_inc_avg_sj[s, :] = \
                        w[s] / np.exp(g_y * (u - s)) * e[s, :] * n[s, :]
        PPB[u, :] =  ((constants.MONTHS_IN_A_YEAR * vpoint * L_inc_avg_sj.sum(axis=0))
                        / (factor * constants.THOUSAND))

    return PPB

@numba.jit
def DB_1dim_loop(w, e, n, S_ret, S, g_y, L_inc_avg_s,
                 L_inc_avg, DB_s, DB, last_career_yrs, rep_rate,
                 rep_rate_py, yr_contr):

    for u in range(S_ret, S):
        for s in range(S_ret - last_career_yrs,
                       S_ret):
            L_inc_avg_s[s - (S_ret - last_career_yrs)] = \
                        w[s] / np.exp(g_y * (u - s)) * e[s] * n[s]
        L_inc_avg = L_inc_avg_s.sum() / last_career_yrs
        rep_rate = yr_contr * rep_rate_py
        DB[u] = rep_rate * L_inc_avg

    return DB

@numba.jit
def DB_2dim_loop(w, e, n, S_ret, S, g_y, L_inc_avg_sj,
                 L_inc_avg, DB_sj, DB, last_career_yrs, rep_rate,
                 rep_rate_py, yr_contr):

    for u in range(S_ret, S):
        for s in range(S_ret - last_career_yrs,
                       S_ret):
            L_inc_avg_sj[s - (S_ret - last_career_yrs), :] = \
                        w[s] / np.exp(g_y * (u - s)) * e[s, :] * n[s, :]
        L_inc_avg = L_inc_avg_sj.sum(axis=0) / last_career_yrs
        rep_rate = yr_contr * rep_rate_py
        DB[u, :] = rep_rate * L_inc_avg

    return DB

@numba.jit
def NDC_1dim_loop(w, e, n, S_ret, S, g_y, tau_p, g_ndc, delta_ret,
                  NDC_s, NDC):

    for u in range(S_ret, S):
        for s in range(0, S_ret):
            NDC_s[s] = (
                tau_p *
                (w[s] / np.exp(g_y * (u - s))) * e[s] *
                n[s] * ((1 + g_ndc) **
                        (S_ret - s - 1)))
        NDC[u] = delta_ret * NDC_s.sum()
    return NDC

@numba.jit
def NDC_2dim_loop(w, e, n, S_ret, S, g_y, tau_p, g_ndc, delta_ret,
                  NDC_sj, NDC):
    for u in range(S_ret, S):
        for s in range(0, S_ret):
            NDC_sj[s, :] = (
                tau_p *
                (w[s] / np.exp(g_y * (u - s))) *
                e[s, :] * n[s, :] * ((1 + g_ndc) **
                                     (S_ret - s - 1)))
        NDC[u, :] = delta_ret * NDC_sj.sum(axis=0)
    return NDC