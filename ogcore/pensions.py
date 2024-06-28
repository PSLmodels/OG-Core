# Packages
import numpy as np
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