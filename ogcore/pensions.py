# Packages
import numpy as np
import numba
from ogcore import utils

# set constants
MONTHS_IN_A_YEAR = 12
THOUSAND = 1000


def cohort_array(name, p, by_j=False):
    """
    Return the pension parameter ``name`` as an array by period and age.

    After ``Specifications.compute_default_params`` the cohort pension
    parameters (``retire``, ``avg_earn_num_years``, ``yr_contrib``,
    ``tau_p``, ``k_ret``, ``indR``) are (T + S) x S arrays holding, for
    the household of age s in period t, the value that applies to its
    birth cohort, and ``alpha_db`` is (T + S) x S x J. This helper also
    accepts the forms a caller may set by hand: a single number is
    broadcast to the full shape, and an array without the age dimension
    (a path by period, or by period and lifetime income group) is taken
    to hold the same value at every age in a period.

    Args:
        name (str): attribute name on ``p``
        p (OG-Core Specifications object): model parameters
        by_j (bool): whether the parameter also varies by lifetime
            income group, in which case the array is (T + S) x S x J

    Returns:
        arr (Numpy array): the parameter by period and age

    Raises:
        ValueError: if the attribute cannot be broadcast to that shape

    """
    arr = np.asarray(getattr(p, name), dtype=float)
    shape = (p.T + p.S, p.S, p.J) if by_j else (p.T + p.S, p.S)
    if arr.ndim == len(shape) and arr.shape == shape:
        return arr
    if arr.ndim == 0 or (arr.size > 0 and np.all(arr == arr.flat[0])):
        # a single value (possibly an array built for other dimensions,
        # as when a test changes S after the object is created)
        return np.broadcast_to(arr.flat[0], shape)
    if arr.ndim == len(shape) - 1 and arr.shape[0] == shape[0]:
        return np.broadcast_to(np.expand_dims(arr, 1), shape)
    raise ValueError(
        f"{name} must be a number or an array of shape {shape} (or the "
        f"same without the age dimension), got shape {arr.shape}"
    )


def cohort_path(arr, t, per_rmn, p):
    """
    Lifetime path of a period-by-age array for one household.

    The household has ``per_rmn`` periods of life remaining in period
    t, so it is age S - per_rmn in period t and lives age u in period
    t - (S - per_rmn) + u: its path runs down a diagonal of ``arr``.
    Ages lived before the time path begins read the first row and ages
    after it ends read the last row, matching how the DB and NDC
    formulas treat wages and labor supply outside the path.

    Args:
        arr (Numpy array): (T + S) x S or (T + S) x S x J array
        t (int): model period of the first of the remaining periods
        per_rmn (int): number of periods of life remaining
        p (OG-Core Specifications object): model parameters

    Returns:
        path (Numpy array): length S (or S x J) path by age

    """
    age0 = p.S - per_rmn
    rows = np.clip(t - age0 + np.arange(p.S), 0, arr.shape[0] - 1)
    return arr[rows, np.arange(p.S)]


def pension_periods(p):
    """
    Convert the pension parameters measured in calendar years into
    integer model periods, by period and age.

    Args:
        p (OG-Core Specifications object): model parameters

    Returns:
        equiv_periods (Numpy array): (T + S) x S number of model periods
            over which earnings are averaged (``avg_earn_num_years``)
        equiv_yr_contrib (Numpy array): (T + S) x S number of model
            periods of contributions credited (``yr_contrib``)
        S_ret (Numpy array): (T + S) x S model period of retirement

    Raises:
        ValueError: if the averaging window is shorter than one model
            period or longer than the working life for any household,
            in which case the window would index before the start of
            the lifetime
    """
    per_year = p.S / 80.0
    avg_years = cohort_array("avg_earn_num_years", p)
    equiv_periods = np.rint(per_year * avg_years).astype(int)
    equiv_yr_contrib = np.rint(
        per_year * cohort_array("yr_contrib", p)
    ).astype(int)
    S_ret = cohort_array("retire", p).astype(int)
    if np.any(equiv_periods < 1):
        raise ValueError(
            "avg_earn_num_years must cover at least one model period: "
            f"avg_earn_num_years={avg_years.min()} with S={p.S} "
            f"gives {equiv_periods.min()} periods."
        )
    if np.any(equiv_periods > S_ret):
        bad = np.argwhere(equiv_periods > S_ret)[0]
        raise ValueError(
            "avg_earn_num_years cannot exceed the working life: "
            f"avg_earn_num_years={avg_years[tuple(bad)]} is "
            f"{equiv_periods[tuple(bad)]} model periods, but the "
            f"household of age {bad[1]} in period {bad[0]} retires after "
            f"{S_ret[tuple(bad)]} periods."
        )
    return equiv_periods, equiv_yr_contrib, S_ret


def replacement_rate_vals(nssmat, wss, factor_ss, j, p):
    r"""
    Calculates replacement rate values for the social security system.

    .. math::
        \theta_{j,R,t+R} =
        \frac{PIA_{j,R,t+R} \times 12}{factor \times w_{t+R}}

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
        dim2 = 1
    else:
        e = np.squeeze(p.e[-1, :, :]).reshape(
            (p.S, p.J)
        )  # Only computes using SS earnings
        dim2 = p.J
    # adjust number of calendar years AIME computed from int model periods.
    # The Social Security window keeps the one-period shift it has always
    # used (so pinned steady-state and time-path results are unchanged);
    # see pension_periods for the unshifted conversion the DB system uses.
    # The shift is floored at one period so that a one-year window cannot
    # produce a 0/0 replacement rate, and the window is bounds-checked.
    equiv_periods, _, S_ret = pension_periods(p)
    equiv_periods, S_ret = int(equiv_periods[-1, -1]), int(S_ret[-1, -1])
    equiv_periods = max(equiv_periods - 1, 1)
    earnings = (e * (wss * nssmat * factor_ss)).reshape(p.S, dim2)
    # get highest earning years for number of years AIME computed from
    highest_earn = (-1.0 * np.sort(-1.0 * earnings[:S_ret, :], axis=0))[
        :equiv_periods
    ]
    AIME = highest_earn.sum(0) / ((12.0 * (p.S / 80.0)) * equiv_periods)
    PIA = np.zeros(dim2)
    # Compute level of replacement using AIME brackets and PIA rates
    for jj in range(dim2):
        if AIME[jj] < p.AIME_bkt_1:
            PIA[jj] = p.PIA_rate_bkt_1 * AIME[jj]
        elif AIME[jj] < p.AIME_bkt_2:
            PIA[jj] = p.PIA_rate_bkt_1 * p.AIME_bkt_1 + p.PIA_rate_bkt_2 * (
                AIME[jj] - p.AIME_bkt_1
            )
        else:
            PIA[jj] = (
                p.PIA_rate_bkt_1 * p.AIME_bkt_1
                + p.PIA_rate_bkt_2 * (p.AIME_bkt_2 - p.AIME_bkt_1)
                + p.PIA_rate_bkt_3 * (AIME[jj] - p.AIME_bkt_2)
            )
    # Set the maximum monthly replacement rate from SS benefits tables
    PIA[PIA > p.PIA_maxpayment] = p.PIA_maxpayment
    if p.PIA_minpayment != 0.0:
        PIA[PIA < p.PIA_minpayment] = p.PIA_minpayment
    theta = (PIA * (12.0 * p.S / 80.0)) / (factor_ss * wss)
    if not p.baseline and p.baseline_theta:
        # use theta from the baseline solution
        if j is not None:
            theta = p.SS_theta[j]
        else:
            theta = p.SS_theta
    return theta


def replacement_rate_adjustment(pension, t, j, method, p):
    """
    Return the replacement rate adjustment, shaped to broadcast against
    pension.

    US-Style Social Security applies this itself inside SS_amount. The
    other three systems do not, so pension_amount applies it for them.
    Along the time path the multiplier varies with each cohort's own year,
    so it is indexed at t + tt for row tt. A 1D pension array is one
    cohort's remaining lifetime starting in period t, so entry tt is
    paid in period t + tt and gets that year's multiplier.

    The steady state uses the final value of the parameter, so a
    permanent adjustment carries into the steady state that the time
    path converges to.

    The multiplier scales the whole pension array, while SS_amount scales
    only the entries from retirement on. These are equivalent only because
    DB, NDC, and PS benefits are zero before retirement; a system that
    pays benefits before retirement would need the SS_amount treatment.

    Args:
        pension (Numpy array): pension amount for each household
        t (int): model period
        j (int): index of lifetime income group, None if all groups
        method (str): 'SS', 'TPI', or 'TPI_scalar'
        p (OG-Core Specifications object): model parameters

    Returns:
        adjustment (array_like): multiplier on the replacement rate

    """
    adjust = p.replacement_rate_adjust
    if method == "SS":
        return adjust[-1, :] if j is None else adjust[-1, j]
    if method == "TPI_scalar":
        return adjust[0, j]
    length = pension.shape[0]
    if pension.ndim == 1:
        return adjust[t : t + length, j]
    if pension.ndim == 2:
        return adjust[t : t + length, j].reshape(length, 1)
    return adjust[t : t + length, :].reshape(length, 1, p.J)


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
        pension = DB_amount(w, e, n, t, j, method, p)
        pension = pension * replacement_rate_adjustment(
            pension, t, j, method, p
        )
    elif p.pension_system == "Notional Defined Contribution":
        pension = NDC_amount(w, e, n, r, Y, t, j, method, p)
        pension = pension * replacement_rate_adjustment(
            pension, t, j, method, p
        )
    elif p.pension_system == "Points System":
        pension = PS_amount(w, e, n, j, factor, p)
        pension = pension * replacement_rate_adjustment(
            pension, t, j, method, p
        )
    else:
        raise ValueError(
            "pension_system must be one of the following: "
            "'US-Style Social Security', 'Defined Benefits', "
            "'Notional Defined Contribution', 'Points System'"
        )
    return pension


def retirement_mask(t, length, shift, method, ndim, p):
    """
    Boolean array marking the households that are at or past their own
    retirement age.

    ``p.retire`` is a (T + S) x S integer array giving, for the household
    of age s in period t, the model period at which it becomes eligible
    for the public pension. A cohort's path runs down a diagonal of that
    array, so the household path reads the cohort's own retirement age
    rather than the one in force in a single calendar year.

    Args:
        t (int): first model period covered by the arrays (ignored for
            'SS' and 'TPI_scalar')
        length (int): number of periods covered by the arrays along the
            time path; for a 1D cohort path this is the number of
            periods of life remaining
        shift (bool): whether computing for periods 0--s or 1--(s+1),
            =True for 1--(s+1), which moves eligibility one age earlier
        method (str): 'SS', 'TPI', or 'TPI_scalar'
        ndim (int): number of dimensions of the labor supply array the
            mask must broadcast against: 1 for a single cohort path or
            an S vector, 2 for S x J or (length) x S, 3 for
            (length) x S x J
        p (OG-Core Specifications object): model parameters

    Returns:
        mask (Numpy array): boolean, shaped to broadcast against the
            pension array

    """
    retire = np.asarray(p.retire)
    S = p.S
    ages = np.arange(S)
    offset = 1 if shift else 0
    if method == "SS":
        # the steady state uses the final period's retirement age
        mask = ages >= retire[-1, :] - offset
        if ndim == 2:
            mask = mask.reshape(S, 1)
    elif method == "TPI_scalar":
        # the oldest household in period 0
        mask = np.array(S - 1 >= retire[0, S - 1] - offset)
    elif method == "TPI":
        if ndim == 1:
            # One cohort with ``length`` periods of life remaining,
            # starting in period t at age S - length: its own retirement
            # ages run down the diagonal of retire with offset
            # S - length, the same extraction the household problem
            # uses for chi_n and rho.
            cohort_retire = np.diagonal(
                retire[t : t + S, :], offset=S - length
            )
            mask = ages[S - length :] >= cohort_retire - offset
        else:
            # cross sections for periods t, ..., t + length - 1
            mask = ages.reshape(1, S) >= retire[t : t + length, :] - offset
            if ndim == 3:
                mask = mask[:, :, np.newaxis]
    else:
        raise ValueError(
            "method must be one of 'SS', 'TPI', or 'TPI_scalar', got "
            f"{method!r}"
        )
    return mask


def SS_amount(w, n, theta, t, j, shift, method, e, p):
    r"""
    Calculate public pension benefit amounts for each household under
    a US-style social security system.

    .. math::
        pension_{j,s,t} = \theta_j \times w_t \quad \forall s \geq R_{t-s}

    where $R_{t-s}$ is the retirement age of the cohort born in period
    $t - s$, read from the (T + S) x S array ``p.retire``.

    Args:
        w (array_like): real wage rate
        n (Numpy array): labor supply
        theta (array_like): social security replacement rate before
            replacement_rate_adjust scaling. A length J vector when j is
            None; a scalar (or length 1 array) for group j when n is 1D
            or method is 'TPI_scalar'
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

    Notes:
        replacement_rate_adjust is applied here, and only here, for the
        US-style system, so callers pass the unadjusted theta. Along the
        time path the factor is indexed by the year the benefit is paid
        and the retirement age by the household's own cohort, so the
        single-cohort path agrees with the (T, S, J) array that the
        aggregates use. The steady state uses the final values of both.

    """
    n = np.asarray(n)
    w = np.asarray(w)
    theta = np.asarray(theta)
    adjust = p.replacement_rate_adjust
    if method == "SS":
        # Depending on if we are looking at b_s or b_s+1, the
        # entry for retirement will change (it shifts back one).
        # The shift boolean makes sure we start replacement rates
        # at the correct age.
        if j is None:
            # a scalar when J = 1 and n is an S vector, else (1, J)
            benefit = np.squeeze(adjust[-1, :] * theta * w)
            if n.ndim == 2:
                benefit = benefit.reshape(1, -1)
        else:
            benefit = adjust[-1, j] * theta * w
        benefit = np.broadcast_to(benefit, n.shape)
        mask = retirement_mask(None, None, shift, method, n.ndim, p)
    elif method == "TPI":
        length = w.shape[0]
        if n.ndim == 1:
            # One cohort's remaining lifetime: entry tt is paid in
            # period t + tt, so the adjustment is indexed by that year.
            benefit = theta * adjust[t : t + length, j] * w
        elif n.ndim == 2:
            benefit = (
                theta
                * adjust[t : t + length, j].reshape(length, 1)
                * w.reshape(length, 1)
            )
        else:
            benefit = (
                theta.reshape(1, 1, p.J)
                * adjust[t : t + length, :].reshape(length, 1, p.J)
                * utils.to_timepath_shape(w)
            )
        mask = retirement_mask(t, length, shift, method, n.ndim, p)
    elif method == "TPI_scalar":
        # The above methods won't work if scalars are used.  This option
        # is only called by the SS_TPI_firstdoughnutring function in TPI.
        benefit = theta * adjust[0, j] * w
        mask = retirement_mask(None, None, shift, method, 1, p)
    else:
        raise ValueError(
            "method must be one of 'SS', 'TPI', or 'TPI_scalar', got "
            f"{method!r}"
        )
    pension = np.where(mask, benefit, 0.0)
    return pension


def DB_amount(w, e, n, t, j, method, p):
    r"""
    Calculate public pension from a defined benefits system.

    .. math::
        pension_{j,s,t} = \biggl[\frac{\sum_{u=R-ny}^{R-1}w_{t-s+u}
            e_{j,u,t-s+u}n_{j,u,t-s+u}}{ny}\biggr]\times Cy \times
            \alpha^{DB}_{j,s,t} \quad \forall s \geq R

    where the retirement age $R$, the averaging window $ny$
    (``avg_earn_num_years``), the years of contribution $Cy$
    (``yr_contrib``), and the replacement rate $\alpha^{DB}$
    (``alpha_db``) are those of the household's own birth cohort at the
    age and period the benefit is paid, read from the period-by-age
    arrays that ``pension_regimes`` builds. With no regimes, $R$, $ny$,
    and $Cy$ are the same for everyone and $\alpha^{DB}_{j,t}$ varies
    only by lifetime income group and the calendar year of payment.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply; may be a partial lifetime path
            (length < S, for cohorts alive when the time path begins),
            a steady-state vector (S,) or matrix (S, J), or the full
            time path (T, S, J)
        t (int): model period of the first element of ``n`` (for a
            partial or full lifetime path) or of the first row of ``n``
            (for the full time path)
        j (int): index of lifetime income group, None if all groups
        method (str): adjusts calculation dimensions based on 'SS',
            'TPI', or 'TPI_scalar'
        p (OG-Core Specifications object): model parameters

    Returns:
        DB (Numpy array): pension amount for each household
    """
    ny, cy, S_ret = pension_periods(p)
    alpha_db = cohort_array("alpha_db", p, by_j=True)
    # g_y is a scalar parameter, but the numba loops index it, so pass
    # it as a 1-element array.
    g_y_arr = np.atleast_1d(np.asarray(p.g_y, dtype=float)).ravel()
    # The wage may arrive as a scalar (steady state, or one remaining
    # period in the TPI); the loops index it as a path, so broadcast it
    # to the length of the labor-supply vector.
    if np.ndim(w) == 0:
        w = np.full(n.shape[0], float(w))

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]
        # Pre-time-path wages: the recent past is anchored to the
        # period-0 wage of the current path (trend growth is handled
        # by the de-trending inside the benefit formula); labor comes
        # from the model's initial condition n_preTP (see Issue #1014)
        w_S = np.append((w[0] * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)
        # The first doughnut ring (method 'TPI_scalar') solves the
        # household in its final period of life at t = 0 and passes
        # t = None, as every other TPI_scalar branch indexes period 0
        if method == "TPI_scalar":
            t = 0
        # This household is age S - per_rmn at time t, so its benefit
        # at age u is paid in period t + u - (S - per_rmn); its own
        # cohort's pension parameters run down that diagonal
        S_ret_u = cohort_path(S_ret, t, per_rmn, p)
        ny_u = cohort_path(ny, t, per_rmn, p)
        cy_u = cohort_path(cy, t, per_rmn, p)
        alpha_db_u = cohort_path(alpha_db, t, per_rmn, p)[:, j]
        # Effective labor units over this household's lifetime: the
        # profile in effect in the period each age is lived in.  Ages
        # before the time path begins use the period-0 profile (as the
        # wage above) and ages beyond the end of the path use the
        # final (steady-state) profile.
        if np.ndim(p.e) == 3:
            e_S = cohort_path(np.asarray(p.e), t, per_rmn, p)[:, j]
        else:
            e_S = p.e[:, j]

        DB = DB_1dim_loop(
            w_S,
            e_S,
            n_S,
            S_ret_u,
            p.S,
            g_y_arr,
            np.zeros(p.S),
            ny_u,
            alpha_db_u,
            cy_u,
        )
        DB = DB[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            if method == "SS":
                row = -1
            elif method == "TPI_scalar":
                row = 0
            else:
                row = None
            if row is None:
                # full lifetime of a cohort that is age 0 at time t
                S_ret_u = cohort_path(S_ret, t, p.S, p)
                ny_u = cohort_path(ny, t, p.S, p)
                cy_u = cohort_path(cy, t, p.S, p)
                alpha_db_u = cohort_path(alpha_db, t, p.S, p)[:, j]
            else:
                S_ret_u, ny_u, cy_u = S_ret[row], ny[row], cy[row]
                alpha_db_u = alpha_db[row, :, j]
            DB = DB_1dim_loop(
                w,
                e,
                n,
                S_ret_u,
                p.S,
                g_y_arr,
                np.zeros(p.S),
                ny_u,
                alpha_db_u,
                cy_u,
            )

        elif np.ndim(n) == 2:
            # cross-section of all S x J households in a single period
            if method == "SS":
                row = -1
            elif method == "TPI_scalar":
                row = 0
            else:
                row = t
            DB = DB_2dim_loop(
                w,
                e,
                n,
                S_ret[row],
                p.S,
                g_y_arr,
                np.zeros((p.S, p.J)),
                ny[row],
                alpha_db[row],
                cy[row],
            )

        elif np.ndim(n) == 3:
            T = n.shape[0]
            w_path = np.squeeze(np.asarray(w))
            if w_path.ndim == 0:
                w_path = np.full(T, float(w_path))
            e_ss = p.e[-1] if np.ndim(p.e) == 3 else p.e
            # parameters in effect for each (period, age) of the path
            rows = np.clip(t + np.arange(T), 0, S_ret.shape[0] - 1)
            DB = DB_3dim_loop(
                w_path,
                e_ss,
                n,
                p.n_preTP,
                S_ret[rows],
                p.S,
                p.J,
                float(g_y_arr[-1]),
                ny[rows],
                np.ascontiguousarray(alpha_db[rows]),
                cy[rows],
            )

    return DB


def NDC_amount(w, e, n, r, Y, t, j, method, p):
    r"""
    Calculate public pension from a notional defined contribution
    system.

    .. math::
        pension_{j,s,t} = \biggl[\sum_{u=E}^{R-1}\tau^{p}_{u,t-s+u}
            w_{t-s+u}e_{j,u,t-s+u}n_{j,u,t-s+u}(1 + g_{NDC,t})^{R-u-1}
            \biggr]\delta_{R,s,t} \quad \forall s \geq R

    where the retirement age $R$ and the conversion coefficient
    $\delta_{R,s,t}$ (through ``k_ret`` and ``indR``) are those of the
    household's own birth cohort at the age and period the benefit is
    paid, and the contribution rate $\tau^{p}_{u,t-s+u}$ is the one the
    cohort faced at each working age u, all read from the period-by-age
    arrays that ``pension_regimes`` builds.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply; may be a partial lifetime path
            (length < S, for cohorts alive when the time path begins),
            a steady-state vector (S,) or matrix (S, J), or the full
            time path (T, S, J)
        r (array_like): interest rate
        Y (array_like): GDP
        t (int): model period of the first element of ``n`` (for a
            partial or full lifetime path) or of the first row of ``n``
            (for the full time path)
        j (int): index of lifetime income group, None if all groups
        method (str): adjusts calculation dimensions based on 'SS',
            'TPI', or 'TPI_scalar'
        p (OG-Core Specifications object): model parameters

    Returns:
        NDC (Numpy array): pension amount for each household
    """
    g_ndc_amount = g_ndc(r, Y, p)
    S_ret = cohort_array("retire", p).astype(int)
    tau_p = cohort_array("tau_p", p)
    indR = cohort_array("indR", p)
    k_ret = cohort_array("k_ret", p)
    # g_y is a scalar but the numba loops index it; the steady-state
    # wage is a scalar but the loops index it as a path.
    g_y_arr = np.atleast_1d(np.asarray(p.g_y, dtype=float)).ravel()
    if np.ndim(w) == 0:
        w = np.full(n.shape[0], float(w))

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]
        # Pre-time-path wages: the recent past is anchored to the
        # period-0 wage of the current path (trend growth is handled
        # by the de-trending inside the benefit formula); labor comes
        # from the model's initial condition n_preTP (see Issue #1014)
        w_S = np.append((w[0] * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)
        # The first doughnut ring (method 'TPI_scalar') solves the
        # household in its final period of life at t = 0 and passes
        # t = None, as every other TPI_scalar branch indexes period 0
        if method == "TPI_scalar":
            t = 0
        # this household's own cohort parameters, by age
        S_ret_u = cohort_path(S_ret, t, per_rmn, p)
        tau_p_u = cohort_path(tau_p, t, per_rmn, p)
        delta_ret_u = delta_ret(
            r,
            Y,
            p,
            S_ret_u,
            cohort_path(indR, t, per_rmn, p),
            cohort_path(k_ret, t, per_rmn, p),
        )
        NDC = NDC_1dim_loop(
            w_S,
            # steady-state earnings profile for now (see Issue #1014)
            p.e[-1, :, j] if np.ndim(p.e) == 3 else p.e[:, j],
            n_S,
            S_ret_u,
            p.S,
            g_y_arr,
            tau_p_u,
            g_ndc_amount,
            delta_ret_u,
            np.zeros(p.S),
        )
        NDC = NDC[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            if method == "SS":
                row = -1
            elif method == "TPI_scalar":
                row = 0
            else:
                row = None
            if row is None:
                # full lifetime of a cohort that is age 0 at time t
                S_ret_u = cohort_path(S_ret, t, p.S, p)
                tau_p_u = cohort_path(tau_p, t, p.S, p)
                indR_u = cohort_path(indR, t, p.S, p)
                k_ret_u = cohort_path(k_ret, t, p.S, p)
            else:
                S_ret_u, tau_p_u = S_ret[row], tau_p[row]
                indR_u, k_ret_u = indR[row], k_ret[row]
            NDC = NDC_1dim_loop(
                w,
                e,
                n,
                S_ret_u,
                p.S,
                g_y_arr,
                tau_p_u,
                g_ndc_amount,
                delta_ret(r, Y, p, S_ret_u, indR_u, k_ret_u),
                np.zeros(p.S),
            )
        elif np.ndim(n) == 2:
            # cross-section of all S x J households in a single period
            if method == "SS":
                row = -1
            elif method == "TPI_scalar":
                row = 0
            else:
                row = t
            NDC = NDC_2dim_loop(
                w,
                e,
                n,
                S_ret[row],
                p.S,
                g_y_arr,
                tau_p[row],
                g_ndc_amount,
                delta_ret(r, Y, p, S_ret[row], indR[row], k_ret[row]),
                np.zeros((p.S, p.J)),
            )
        elif np.ndim(n) == 3:
            T = n.shape[0]
            w_path = np.squeeze(np.asarray(w))
            if w_path.ndim == 0:
                w_path = np.full(T, float(w_path))
            e_ss = p.e[-1] if np.ndim(p.e) == 3 else p.e
            # parameters in effect for each (period, age) of the path
            rows = np.clip(t + np.arange(T), 0, S_ret.shape[0] - 1)
            NDC = NDC_3dim_loop(
                w_path,
                e_ss,
                n,
                p.n_preTP,
                S_ret[rows],
                p.S,
                p.J,
                float(g_y_arr[-1]),
                np.ascontiguousarray(tau_p[rows]),
                g_ndc_amount,
                delta_ret(r, Y, p, S_ret[rows], indR[rows], k_ret[rows]),
            )

    return NDC


def PS_amount(w, e, n, j, factor, p):
    r"""
    Calculate public pension from a points system.

    .. math::
        pension{j,s,t} = \sum_{s=E}^{R-1}w_{t}e_{j,s,t}n_{j,s,t}
        \times v_{t} \quad \forall s > R

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

    # Single retirement age for now (see Issue #1014); g_y is
    # a scalar but the numba loops index it; the steady-state wage is a
    # scalar but the loops index it as a path.
    S_ret = int(np.asarray(p.retire).flat[-1])
    g_y_arr = np.atleast_1d(np.asarray(p.g_y, dtype=float)).ravel()
    if np.ndim(w) == 0:
        w = np.full(n.shape[0], float(w))

    if n.shape[0] < p.S:
        per_rmn = n.shape[0]
        # Pre-time-path wages: the recent past is anchored to the
        # period-0 wage of the current path (trend growth is handled
        # by the de-trending inside the benefit formula); labor comes
        # from the model's initial condition n_preTP (see Issue #1014)
        w_S = np.append((w[0] * np.ones(p.S))[:(-per_rmn)], w)
        n_S = np.append(p.n_preTP[:(-per_rmn), j], n)
        L_inc_avg_s = np.zeros(S_ret)
        PS = np.zeros(p.S)
        PS = PS_1dim_loop(
            w_S,
            # steady-state earnings profile for now (see Issue #1014)
            p.e[-1, :, j] if np.ndim(p.e) == 3 else p.e[:, j],
            n_S,
            S_ret,
            p.S,
            g_y_arr,
            p.vpoint,
            factor,
            L_inc_avg_s,
            PS,
        )
        PS = PS[-per_rmn:]

    else:
        if np.ndim(n) == 1:
            L_inc_avg_s = np.zeros(S_ret)
            PS = np.zeros(p.S)
            PS = PS_1dim_loop(
                w,
                e,
                n,
                S_ret,
                p.S,
                g_y_arr,
                p.vpoint,
                factor,
                L_inc_avg_s,
                PS,
            )

        elif np.ndim(n) == 2:
            L_inc_avg_sj = np.zeros((S_ret, p.J))
            PS = np.zeros((p.S, p.J))
            PS = PS_2dim_loop(
                w,
                e,
                n,
                S_ret,
                p.S,
                p.J,
                g_y_arr,
                p.vpoint,
                factor,
                L_inc_avg_sj,
                PS,
            )

    return PS


def deriv_theta(r, w, e, Y, per_rmn, factor, t, j, p):
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
        t (int): model period
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        d_theta (Numpy array): change in pension benefits for another
            unit of labor supply
    """
    # TODO: Add Social Security here...
    if p.pension_system == "Defined Benefits":
        d_theta = deriv_DB(w, e, per_rmn, t, j, p)
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
            \tau^{p}_{t}w_{t}e_{j,s}(1+g_{NDC,t})^{u - s}\delta_{R,t},
            & \text{if}\ s<R-1 \\
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
    # steady-state retirement age and contribution rate
    S_ret = int(cohort_array("retire", p)[-1, -1])
    tau_p = float(cohort_array("tau_p", p)[-1, -1])
    if per_rmn == 1:
        d_theta = 0
    elif per_rmn < (p.S - S_ret + 1):
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
            S_ret,
            tau_p,
            g_ndc_amount,
            delta_ret_amount,
            d_theta_empty,
        )

    return d_theta


def deriv_DB(w, e, per_rmn, t, j, p):
    r"""
    Change in DB pension benefits for another unit of labor supply

    .. math::
        \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
            \begin{cases}
                0 , & \text{if}\ s < R - Cy \\
                w_{t}e_{j,s}\alpha^{DB}_{j,t+R-s}\times \frac{Cy}{ny},
                & \text{if}\  R - Cy <= s < R  \\
                0, & \text{if}\ s \geq R \\
            \end{cases}

    With a replacement rate that varies over time, the change in the
    benefit paid at each age u depends on the rate in effect at that
    time.  This function returns a single derivative for each working
    age s, evaluated at the rate in effect when the household retires
    (period t + R - s), as an approximation.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        per_rmn (int): number of periods remaining in the model
        t (int): model period
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        d_theta: change in DB pension benefits for another unit of labor
            supply
    """
    # steady-state values of the pension periods
    equiv_periods, equiv_yr_contrib, S_ret = (
        int(a[-1, -1]) for a in pension_periods(p)
    )
    alpha_db = cohort_array("alpha_db", p, by_j=True)
    if per_rmn < (p.S - S_ret + 1):
        d_theta = np.zeros(p.S)
    else:
        # element s of the loop is a household of age S - per_rmn + s
        # at time t, which retires in period t + S_ret - age
        age = p.S - per_rmn + np.arange(per_rmn)
        idx = np.clip(t + S_ret - age, 0, alpha_db.shape[0] - 1)
        alpha_db_s = alpha_db[idx, S_ret, j]
        d_theta = deriv_DB_loop(
            w,
            e,
            p.S,
            S_ret,
            per_rmn,
            equiv_periods,
            alpha_db_s,
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

    # Single retirement age for now (see Issue #1014)
    S_ret = int(np.asarray(p.retire).flat[-1])
    if per_rmn < (p.S - S_ret + 1):
        d_theta = np.zeros(p.S)
    else:
        d_theta_empty = np.zeros(p.S)
        d_theta = deriv_PS_loop(
            w, e, p.S, S_ret, per_rmn, d_theta_empty, p.vpoint, factor
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
    # r and g_y are scalars in the SS but paths in the TPI; take the
    # last element robustly in either case (see Issue #1169)
    if p.ndc_growth_rate == "r":
        g_ndc = np.asarray(r).flat[-1]
    elif p.ndc_growth_rate == "Curr GDP":
        g_ndc = (Y[1:] - Y[:-1]) / Y[:-1]
    else:  # "LR GDP"
        g_ndc = np.asarray(p.g_y).flat[-1] + np.asarray(p.g_n).flat[-1]

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
    # r and g_y are scalars in the SS but paths in the TPI; take the
    # last element robustly in either case (see Issue #1169)
    if dir_growth_rate == "r":
        g_dir = np.asarray(r).flat[-1]
    elif dir_growth_rate == "Curr GDP":
        g_dir = (Y[1:] - Y[:-1]) / Y[:-1]
    else:  # "LR GDP"
        g_dir = np.asarray(g_y).flat[-1] + np.asarray(g_n).flat[-1]

    return g_dir


def delta_ret(r, Y, p, S_ret=None, indR=None, k_ret=None):
    r"""
    Compute conversion coefficient for the NDC pension amount

    .. math::
        \delta_{R} = (dir_{R} + ind_{R} - k)^{-1}

    Args:
        r (array_like): interest rate
        Y (array_like): GDP
        p (OG-Core Specifications object): model parameters
        S_ret (array_like): model period of retirement; any shape, or
            None for the steady-state value (the final row of
            ``retire``)
        indR (array_like): survivor benefit adjustment, broadcastable
            against ``S_ret``, or None for the steady-state value
        k_ret (array_like): payment frequency adjustment, broadcastable
            against ``S_ret``, or None for the steady-state value

    Returns:
        delta_ret (scalar or Numpy array): conversion coefficient for
            the NDC pension amount, shaped like ``S_ret``

    """
    # A real Specifications object has no mort_rates_SS attribute; fall
    # back to the steady-state mortality rates implied by rho, averaged
    # over lifetime-income groups with their population weights (see
    # Issue #1169)
    mort_rates_SS = getattr(p, "mort_rates_SS", None)
    if mort_rates_SS is None:
        mort_rates_SS = p.rho[-1] @ np.asarray(p.lambdas).ravel()
    surv_rates = 1 - mort_rates_SS
    if S_ret is None:
        S_ret = cohort_array("retire", p)[-1, -1]
    if indR is None:
        indR = cohort_array("indR", p)[-1, -1]
    if k_ret is None:
        k_ret = cohort_array("k_ret", p)[-1, -1]
    S_ret = np.asarray(S_ret, dtype=int)
    g_dir_value = g_dir(r, Y, p.g_y, p.g_n, p.dir_growth_rate)
    # the annuity factor depends on the retirement period only, so
    # compute it once per distinct value (delta_ret_loop writes into
    # the survival-rate array it is given, so pass it a copy)
    dir_delta = np.zeros(S_ret.shape)
    for R in np.unique(S_ret):
        R = int(R)
        dir_delta[S_ret == R] = delta_ret_loop(
            p.S, R, surv_rates.copy(), g_dir_value, np.zeros(p.S - R + 1)
        )
    delta_ret = 1 / (dir_delta + np.asarray(indR) - np.asarray(k_ret))
    if delta_ret.ndim == 0:
        return float(delta_ret)
    return delta_ret


@numba.jit(nopython=True)
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
        alpha_db (Numpy array): replacement rate for each of the
            per_rmn remaining periods
        yr_contr (scalar): years of contribution

    Returns:
        d_theta (Numpy array): change in DB pension benefits for
            another unit of labor supply
    """
    d_theta = np.zeros(per_rmn)
    # print("Year contribution: ", yr_contr)
    # print("Average earnings years: ", avg_earn_num_years)
    num_per_retire = S - S_ret
    for s in range(per_rmn):
        d_theta[s] = (
            w[s] * e[s] * alpha_db[s] * (yr_contr / avg_earn_num_years)
        )
    d_theta[-num_per_retire:] = 0.0

    return d_theta


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def DB_1dim_loop(
    w, e, n, S_ret, S, g_y, DB, avg_earn_num_years, alpha_db, yr_contr
):
    """
    Calculate public pension from a defined benefits system for one
    household over its lifetime.

    Args:
        w (array_like): real wage rate at each age, length S
        e (Numpy array): effective labor units at each age, length S
        n (Numpy array): labor supply at each age, length S
        S_ret (Numpy array): retirement period that applies to the
            household at each age u, length S
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        DB (Numpy array): pension amount for each household (filled in
            and returned), length S
        avg_earn_num_years (Numpy array): number of periods earnings
            are averaged over, by age u, length S
        alpha_db (Numpy array): replacement rate in effect at each age
            u, length S
        yr_contr (Numpy array): periods of contribution credited, by
            age u, length S

    Returns:
        DB (Numpy array): pension amount for each household
    """
    for u in range(S):
        R = S_ret[u]
        if u < R:
            continue
        ny = avg_earn_num_years[u]
        L_inc_avg = 0.0
        for s in range(R - ny, R):
            L_inc_avg += w[s] / np.exp(g_y[-1] * (u - s)) * e[s] * n[s]
        DB[u] = yr_contr[u] * alpha_db[u] * L_inc_avg / ny

    return DB


@numba.jit(nopython=True)
def DB_2dim_loop(
    w, e, n, S_ret, S, g_y, DB, avg_earn_num_years, alpha_db, yr_contr
):
    """
    Calculate public pension from a defined benefits system for a
    cross section of S x J households in one period.

    Args:
        w (array_like): real wage rate, length S
        e (Numpy array): effective labor units, size S x J
        n (Numpy array): labor supply, size S x J
        S_ret (Numpy array): retirement period of the household of
            each age u, length S
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        DB (Numpy array): pension amount for each household (filled in
            and returned), size S x J
        avg_earn_num_years (Numpy array): number of periods earnings
            are averaged over, by age u, length S
        alpha_db (Numpy array): replacement rate in effect at each age
            u for each group j, size S x J
        yr_contr (Numpy array): periods of contribution credited, by
            age u, length S

    Returns:
        DB (Numpy array): pension amount for each household

    """
    for u in range(S):
        R = S_ret[u]
        if u < R:
            continue
        ny = avg_earn_num_years[u]
        L_inc_avg = np.zeros(DB.shape[1])
        for s in range(R - ny, R):
            L_inc_avg += w[s] / np.exp(g_y[-1] * (u - s)) * e[s, :] * n[s, :]
        DB[u, :] = yr_contr[u] * alpha_db[u, :] * L_inc_avg / ny

    return DB


@numba.jit(nopython=True)
def DB_3dim_loop(
    w_path,
    e_ss,
    n,
    n_preTP,
    S_ret,
    S,
    J,
    g_y,
    avg_earn_num_years,
    alpha_db,
    yr_contr,
):
    r"""
    Calculate public pension from a defined benefits system over the
    full time path.

    Used when the TPI solution evaluates taxes for all periods at once
    to compute aggregates. Each retiree's benefit is computed from their
    own cohort's wage and labor history: the wage entering the average
    for a household aged u at time t, earned at age s, is the wage at
    time t - (u - s). Histories that predate the time path use the
    period-0 wage (trend growth is handled by the de-trending in the
    benefit formula) and the model's initial labor supply, so this
    reproduces exactly what the per-cohort household solves compute and
    household behavior and aggregates are consistent. The pension
    parameters are read at (t, u), the cell of the period-by-age arrays
    that belongs to the retiree's own cohort. The inner work is
    vectorized rather than looped.

    Args:
        w_path (Numpy array): real wage rate path, length T
        e_ss (Numpy array): effective labor units, size SxJ
        n (Numpy array): labor supply, size TxSxJ
        n_preTP (Numpy array): pre-time-path labor supply, size SxJ
        S_ret (Numpy array): retirement period by period and age,
            size T x S
        S (int): number of periods in the model
        J (int): number of lifetime income groups
        g_y (scalar): growth rate of technology
        avg_earn_num_years (Numpy array): number of periods earnings
            are averaged over, by period and age, size T x S
        alpha_db (Numpy array): replacement rate per year of
            contribution by period, age, and group, size T x S x J
        yr_contr (Numpy array): periods of contribution credited, by
            period and age, size T x S

    Returns:
        DB (Numpy array): pension amount for each household, size TxSxJ
    """
    T = n.shape[0]
    DB = np.zeros((T, S, J))
    for t in range(T):
        for u in range(S):
            R = S_ret[t, u]
            if u < R:
                continue
            ny = avg_earn_num_years[t, u]
            s_idx = np.arange(R - ny, R)
            tau = t - (u - s_idx)
            w_hist = w_path[np.clip(tau, 0, T - 1)]
            n_hist = np.where(
                (tau >= 0)[:, None],
                n[np.clip(tau, 0, T - 1), s_idx, :],
                n_preTP[s_idx, :],
            )
            L = (w_hist[:, None] / np.exp(g_y * (u - s_idx))[:, None]) * (
                e_ss[s_idx, :] * n_hist
            )
            DB[t, u, :] = (
                (L.sum(axis=0) / ny) * yr_contr[t, u] * alpha_db[t, u, :]
            )

    return DB


def NDC_1dim_loop(w, e, n, S_ret, S, g_y, tau_p, g_ndc, delta_ret, NDC):
    """
    Calculate public pension from a notional defined contribution
    system for one household over its lifetime.

    Args:
        w (array_like): real wage rate at each age, length S
        e (Numpy array): effective labor units at each age, length S
        n (Numpy array): labor supply at each age, length S
        S_ret (Numpy array): retirement period that applies to the
            household at each age u, length S
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        tau_p (Numpy array): contribution rate the household faced at
            each working age s, length S
        g_ndc (scalar): growth rate of NDC pension
        delta_ret (Numpy array): conversion coefficient for the NDC
            pension amount that applies at each age u, length S
        NDC (Numpy array): pension amount for each household (filled
            in and returned), length S

    Returns:
        NDC (Numpy array): pension amount for each household

    """
    for u in range(S):
        R = S_ret[u]
        if u < R:
            continue
        NDC_s = 0.0
        for s in range(0, R):
            # TODO: update so can take g_y from period t
            NDC_s += (
                tau_p[s]
                * (w[s] / np.exp(g_y[-1] * (u - s)))
                * e[s]
                * n[s]
                * ((1 + g_ndc) ** (R - s - 1))
            )
        NDC[u] = delta_ret[u] * NDC_s
    return NDC


@numba.jit(nopython=True)
def NDC_2dim_loop(w, e, n, S_ret, S, g_y, tau_p, g_ndc, delta_ret, NDC):
    """
    Calculate public pension from a notional defined contribution
    system for a cross section of S x J households in one period.

    Args:
        w (array_like): real wage rate, length S
        e (Numpy array): effective labor units, size S x J
        n (Numpy array): labor supply, size S x J
        S_ret (Numpy array): retirement period of the household of
            each age u, length S
        S (int): number of periods in the model
        g_y (array_like): GDP growth rate
        tau_p (Numpy array): contribution rate of the household of each
            age s, length S
        g_ndc (scalar): growth rate of NDC pension
        delta_ret (Numpy array): conversion coefficient for the NDC
            pension amount of the household of each age u, length S
        NDC (Numpy array): pension amount for each household (filled
            in and returned), size S x J

    Returns:
        NDC (Numpy array): pension amount for each household

    """
    for u in range(S):
        R = S_ret[u]
        if u < R:
            continue
        NDC_s = np.zeros(NDC.shape[1])
        for s in range(0, R):
            NDC_s += (
                tau_p[s]
                * (w[s] / np.exp(g_y[-1] * (u - s)))
                * e[s, :]
                * n[s, :]
                * ((1 + g_ndc) ** (R - s - 1))
            )
        NDC[u, :] = delta_ret[u] * NDC_s
    return NDC


@numba.jit(nopython=True)
def NDC_3dim_loop(
    w_path, e_ss, n, n_preTP, S_ret, S, J, g_y, tau_p, g_ndc, delta_ret
):
    r"""
    Calculate public pension from a notional defined contribution
    system over the full time path.

    Used when the TPI solution evaluates taxes for all periods at once
    to compute aggregates. As in ``DB_3dim_loop``, each retiree's
    benefit is built from their own cohort's history: the contribution
    at age s of a household aged u at time t was made in period
    t - (u - s), at that period's wage, labor supply, and contribution
    rate. Histories that predate the time path use the period-0 wage
    and contribution rate and the model's initial labor supply, so
    this reproduces what the per-cohort household solves compute.

    Args:
        w_path (Numpy array): real wage rate path, length T
        e_ss (Numpy array): effective labor units, size SxJ
        n (Numpy array): labor supply, size TxSxJ
        n_preTP (Numpy array): pre-time-path labor supply, size SxJ
        S_ret (Numpy array): retirement period by period and age,
            size T x S
        S (int): number of periods in the model
        J (int): number of lifetime income groups
        g_y (scalar): growth rate of technology
        tau_p (Numpy array): contribution rate by period and age,
            size T x S
        g_ndc (scalar): growth rate of NDC pension
        delta_ret (Numpy array): conversion coefficient for the NDC
            pension amount by period and age, size T x S

    Returns:
        NDC (Numpy array): pension amount for each household, size
            TxSxJ
    """
    T = n.shape[0]
    NDC = np.zeros((T, S, J))
    for t in range(T):
        for u in range(S):
            R = S_ret[t, u]
            if u < R:
                continue
            s_idx = np.arange(R)
            tau = t - (u - s_idx)
            rows = np.clip(tau, 0, T - 1)
            w_hist = w_path[rows]
            n_hist = np.where(
                (tau >= 0)[:, None],
                n[rows, s_idx, :],
                n_preTP[s_idx, :],
            )
            contrib = np.zeros(R)
            for i in range(R):
                s = s_idx[i]
                contrib[i] = (
                    tau_p[rows[i], s]
                    * w_hist[i]
                    / np.exp(g_y * (u - s))
                    * (1 + g_ndc) ** (R - s - 1)
                )
            L = contrib[:, None] * e_ss[s_idx, :] * n_hist
            NDC[t, u, :] = delta_ret[t, u] * L.sum(axis=0)

    return NDC
