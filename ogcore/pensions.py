# Packages
import numpy as np
import numba
from typing import NamedTuple
from ogcore import utils

# set constants
MONTHS_IN_A_YEAR = 12
THOUSAND = 1000


@numba.jit(nopython=True)
def earnings_band_at(x, lo, hi, period):
    """
    Part of labor income ``x`` (w * e * n, in model units) that lies
    between the cutoffs in effect in the model period in which it was
    earned, min(max(x, lo), hi) - lo. With ``lo = 0`` and ``hi = inf``
    this returns ``x`` unchanged.

    Args:
        x (scalar or Numpy array): labor income, w * e * n
        lo (Numpy array): calendar path of the lower cutoff, in model
            units of labor income
        hi (Numpy array): calendar path of the upper cutoff, in model
            units of labor income
        period (int): model period in which ``x`` was earned; periods
            before or after a path read its first or last element

    Returns:
        band (scalar or Numpy array): part of ``x`` between the cutoffs
            in effect in ``period``
    """
    lo_t = lo[min(max(period, 0), lo.size - 1)]
    hi_t = hi[min(max(period, 0), hi.size - 1)]
    return np.minimum(np.maximum(x, lo_t), hi_t) - lo_t


def _band_paths(earn_lo, earn_hi, method):
    """
    Calendar paths of the earnings band cutoffs used by ``DB_amount``
    and ``NDC_amount``.

    Args:
        earn_lo (scalar, array_like, or None): lower cutoff, in model
            units of labor income, a single value or a path by period
        earn_hi (scalar, array_like, or None): upper cutoff
        method (str): 'SS', 'TPI', or 'TPI_scalar'

    Returns:
        lo, hi (Numpy arrays or None): None, None when no band is
            set. Otherwise 1-D float arrays: the full paths for 'TPI',
            and the final (steady-state) or first value for 'SS' or
            'TPI_scalar', matching the rows the other pension
            parameters are read from.
    """
    if earn_lo is None and earn_hi is None:
        return None, None
    lo = np.atleast_1d(
        np.asarray(0.0 if earn_lo is None else earn_lo, dtype=float)
    ).ravel()
    hi = np.atleast_1d(
        np.asarray(np.inf if earn_hi is None else earn_hi, dtype=float)
    ).ravel()
    if method == "SS":
        lo, hi = lo[-1:], hi[-1:]
    elif method == "TPI_scalar":
        lo, hi = lo[:1], hi[:1]
    return np.ascontiguousarray(lo), np.ascontiguousarray(hi)


def _path_from(path, t):
    """
    A calendar path starting in period t (its last element if t is
    past its end), or None if ``path`` is None.
    """
    if path is None:
        return None
    return np.ascontiguousarray(path[min(t, path.size - 1) :])


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


def _db_cohort_inputs(p, t, per_rmn, j):
    """
    The Defined Benefits parameters along one household's lifetime.

    Returns the retirement period, averaging window, years of
    contribution and replacement rate in effect at each age of the
    household that has ``per_rmn`` periods of life remaining in period
    ``t`` (see ``cohort_path``), and its effective labor units by age.
    None of these depend on the household's choices, but the household
    problem evaluates the pension many times while it solves, so they are
    cached on ``p`` by household. The cache is emptied when any of the
    arrays it is built from is reassigned (compared by identity, as in
    ``pension_periods``). The returned arrays are read-only.

    Args:
        p (OG-Core Specifications object): model parameters
        t (int): model period of the first of the remaining periods
        per_rmn (int): number of periods of life remaining
        j (int): index of lifetime income group

    Returns:
        (tuple): (S_ret_u, ny_u, cy_u, alpha_db_u, e_S), each length S
    """
    ny, cy, S_ret = pension_periods(p)
    sources = (ny, cy, S_ret, p.alpha_db, p.e)
    cache = getattr(p, "_db_cohort_cache", None)
    if cache is None or not all(a is b for a, b in zip(cache[0], sources)):
        cache = (sources, {})
        p._db_cohort_cache = cache
    key = (t, per_rmn, j)
    found = cache[1].get(key)
    if found is not None:
        return found
    alpha_db = cohort_array("alpha_db", p, by_j=True)
    if np.ndim(p.e) == 3:
        e_S = cohort_path(np.asarray(p.e), t, per_rmn, p)[:, j]
    else:
        e_S = p.e[:, j]
    found = (
        cohort_path(S_ret, t, per_rmn, p),
        cohort_path(ny, t, per_rmn, p),
        cohort_path(cy, t, per_rmn, p),
        cohort_path(alpha_db, t, per_rmn, p)[:, j],
        np.ascontiguousarray(e_S),
    )
    for arr in found:
        arr.setflags(write=False)
    cache[1][key] = found
    return found


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

    The result depends only on ``S``, ``T`` and the three parameters
    above, which do not change while the model solves, yet the
    household problem asks for it several times per evaluation. It is
    therefore cached on ``p`` and recomputed only when one of them is
    reassigned (each parameter is compared by identity, so changing an
    array's elements in place is not detected). The returned arrays are
    read-only, as they are shared between calls.
    """
    key = (p.S, p.T, p.avg_earn_num_years, p.yr_contrib, p.retire)
    cached = getattr(p, "_pension_periods_cache", None)
    if (
        cached is not None
        and cached[0][:2] == key[:2]
        and all(a is b for a, b in zip(cached[0][2:], key[2:]))
    ):
        return cached[1]
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
    result = (equiv_periods, equiv_yr_contrib, S_ret)
    for arr in result:
        arr.setflags(write=False)
    p._pension_periods_cache = (key, result)
    return result


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


def pension_amount(
    r, w, n, ndc_rates, theta, t, j, shift, method, e, factor, p
):
    """
    Calculate public pension benefit amounts for each household.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        n (Numpy array): labor supply
        ndc_rates (NDCRates): calendar paths of the NDC credited and
            discount rates (``ndc_rates_path``), used only by the
            notional defined contribution system; None for the
            steady-state rates
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
        pension = NDC_amount(w, e, n, r, ndc_rates, t, j, method, p)
        pension = pension * replacement_rate_adjustment(
            pension, t, j, method, p
        )
    elif p.pension_system == "Points System":
        pension = PS_amount(w, e, n, j, factor, p)
        pension = pension * replacement_rate_adjustment(
            pension, t, j, method, p
        )
    elif p.pension_system == "Hybrid DB-NDC":
        pension = DB_NDC_hybrid_amount(w, e, n, r, ndc_rates, t, j, method, p)
        pension = pension * replacement_rate_adjustment(
            pension, t, j, method, p
        )
    else:
        raise ValueError(
            "pension_system must be one of the following: "
            "'US-Style Social Security', 'Defined Benefits', "
            "'Notional Defined Contribution', 'Points System', "
            "'Hybrid DB-NDC'"
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


def DB_amount(w, e, n, t, j, method, p, earn_lo=None, earn_hi=None):
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
        earn_lo (scalar, array_like, or None): only labor income
            w * e * n above earn_lo enters the benefit (used by the
            hybrid DB-NDC system). In model units of labor income, a
            single value or a path by model period of length T + S;
            the cutoff in effect in the period each income was earned
            applies. None (the default) for no lower cutoff.
        earn_hi (scalar, array_like, or None): only labor income below
            earn_hi enters the benefit, as for earn_lo. None (the
            default) for no upper cutoff.

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
    earn_lo, earn_hi = _band_paths(earn_lo, earn_hi, method)

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
        # cohort's pension parameters run down that diagonal. Its
        # effective labor units are the profile in effect in the period
        # each age is lived in: ages before the time path begins use the
        # period-0 profile (as the wage above) and ages beyond the end of
        # the path use the final (steady-state) profile.
        S_ret_u, ny_u, cy_u, alpha_db_u, e_S = _db_cohort_inputs(
            p, t, per_rmn, j
        )

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
            earn_lo,
            earn_hi,
            t - (p.S - per_rmn),
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
                S_ret_u, ny_u, cy_u, alpha_db_u, _ = _db_cohort_inputs(
                    p, t, p.S, j
                )
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
                earn_lo,
                earn_hi,
                # period of age 0 (the band paths of 'SS' and
                # 'TPI_scalar' have a single value)
                t if row is None else 0,
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
                earn_lo,
                earn_hi,
                max(row, 0),
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
                # the loop counts periods from the first row of n
                _path_from(earn_lo, t),
                _path_from(earn_hi, t),
            )

    return DB


def NDC_amount(
    w, e, n, r, ndc_rates, t, j, method, p, earn_lo=None, earn_hi=None
):
    r"""
    Calculate public pension from a notional defined contribution
    system.

    .. math::
        pension_{j,s,t} = \biggl[\sum_{u=E}^{R-1}\tau^{p}_{u,t-s+u}
            w_{t-s+u}e_{j,u,t-s+u}n_{j,u,t-s+u}
            \prod_{a=u+1}^{R-1}(1 + g_{NDC,t-s+a})
            \biggr]\delta_{R,s,t} \quad \forall s \geq R

    where the retirement age $R$ and the conversion coefficient
    $\delta_{R,s,t}$ (through ``k_ret`` and ``indR``) are those of the
    household's own birth cohort at the age and period the benefit is
    paid, and the contribution rate $\tau^{p}_{u,t-s+u}$ is the one the
    cohort faced at each working age u, all read from the period-by-age
    arrays that ``pension_regimes`` builds. Each contribution earns
    the rate credited in each period between the contribution and
    retirement, $g_{NDC,t}$, and the conversion coefficient discounts
    payments at the rate in the period the household retires.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply; may be a partial lifetime path
            (length < S, for cohorts alive when the time path begins),
            a steady-state vector (S,) or matrix (S, J), or the full
            time path (T, S, J)
        r (array_like): interest rate, used for the steady-state rates
            when ``ndc_rates`` is not given
        ndc_rates (NDCRates): calendar paths of the credited and
            discount rates from ``ndc_rates_path``; anything else (such
            as None) uses the steady-state rates, with the 'r' options
            reading the last element of ``r``
        t (int): model period of the first element of ``n`` (for a
            partial or full lifetime path) or of the first row of ``n``
            (for the full time path)
        j (int): index of lifetime income group, None if all groups
        method (str): adjusts calculation dimensions based on 'SS',
            'TPI', or 'TPI_scalar'
        p (OG-Core Specifications object): model parameters
        earn_lo (scalar, array_like, or None): only labor income
            w * e * n above earn_lo enters the benefit (used by the
            hybrid DB-NDC system). In model units of labor income, a
            single value or a path by model period of length T + S;
            the cutoff in effect in the period each income was earned
            applies. None (the default) for no lower cutoff.
        earn_hi (scalar, array_like, or None): only labor income below
            earn_hi enters the benefit, as for earn_lo. None (the
            default) for no upper cutoff.

    Returns:
        NDC (Numpy array): pension amount for each household
    """
    if not isinstance(ndc_rates, NDCRates):
        ndc_rates = NDCRates(
            np.array([g_ndc(r, None, p)]), np.array([g_dir(r, None, p)])
        )
    credit = np.ascontiguousarray(ndc_rates.credit, dtype=float)
    discount = ndc_rates.discount
    S_ret = cohort_array("retire", p).astype(int)
    tau_p = cohort_array("tau_p", p)
    indR = cohort_array("indR", p)
    k_ret = cohort_array("k_ret", p)
    # g_y is a scalar but the numba loops index it; the steady-state
    # wage is a scalar but the loops index it as a path.
    g_y_arr = np.atleast_1d(np.asarray(p.g_y, dtype=float)).ravel()
    if np.ndim(w) == 0:
        w = np.full(n.shape[0], float(w))
    earn_lo, earn_hi = _band_paths(earn_lo, earn_hi, method)

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
        # the household is age S - per_rmn in period t, so it reaches
        # retirement age R in period t - (S - per_rmn) + R
        t_birth = t - (p.S - per_rmn)
        delta_ret_u = delta_ret(
            r,
            None,
            p,
            S_ret_u,
            cohort_path(indR, t, per_rmn, p),
            cohort_path(k_ret, t, per_rmn, p),
            t_ret=t_birth + S_ret_u,
            g_dir_path=discount,
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
            credit,
            t_birth,
            delta_ret_u,
            np.zeros(p.S),
            earn_lo,
            earn_hi,
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
                t_birth = t
            else:
                S_ret_u, tau_p_u = S_ret[row], tau_p[row]
                indR_u, k_ret_u = indR[row], k_ret[row]
                # the steady state (rates do not vary), or a household
                # born in period 0
                t_birth = 0
            t_ret_u = None if row == -1 else t_birth + S_ret_u
            NDC = NDC_1dim_loop(
                w,
                e,
                n,
                S_ret_u,
                p.S,
                g_y_arr,
                tau_p_u,
                credit,
                t_birth,
                delta_ret(
                    r,
                    None,
                    p,
                    S_ret_u,
                    indR_u,
                    k_ret_u,
                    t_ret_u,
                    g_dir_path=discount,
                ),
                np.zeros(p.S),
                earn_lo,
                earn_hi,
            )
        elif np.ndim(n) == 2:
            # cross-section of all S x J households in a single period
            if method == "SS":
                row = -1
            elif method == "TPI_scalar":
                row = 0
            else:
                row = t
            # the household of age u in period `row` retired in period
            # row - u + R (None in the steady state)
            t_ret_u = None if row == -1 else row - np.arange(p.S) + S_ret[row]
            NDC = NDC_2dim_loop(
                w,
                e,
                n,
                S_ret[row],
                p.S,
                g_y_arr,
                tau_p[row],
                credit,
                0 if row == -1 else row,
                delta_ret(
                    r,
                    None,
                    p,
                    S_ret[row],
                    indR[row],
                    k_ret[row],
                    t_ret_u,
                    g_dir_path=discount,
                ),
                np.zeros((p.S, p.J)),
                earn_lo,
                earn_hi,
            )
        elif np.ndim(n) == 3:
            T = n.shape[0]
            w_path = np.squeeze(np.asarray(w))
            if w_path.ndim == 0:
                w_path = np.full(T, float(w_path))
            e_ss = p.e[-1] if np.ndim(p.e) == 3 else p.e
            # parameters in effect for each (period, age) of the path
            rows = np.clip(t + np.arange(T), 0, S_ret.shape[0] - 1)
            # the household of age u in period t + i retired in period
            # t + i - u + R
            t_ret_path = (
                (t + np.arange(T))[:, None] - np.arange(p.S)[None, :]
            ) + S_ret[rows]
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
                # the loop counts periods from the first row of n
                credit[min(t, credit.size - 1) :],
                delta_ret(
                    r,
                    None,
                    p,
                    S_ret[rows],
                    indR[rows],
                    k_ret[rows],
                    t_ret_path,
                    g_dir_path=discount,
                ),
                _path_from(earn_lo, t),
                _path_from(earn_hi, t),
            )

    return NDC


def DB_NDC_hybrid_amount(w, e, n, r, ndc_rates, t, j, method, p):
    r"""
    Calculate public pension from a hybrid defined benefit and notional
    defined contribution system.

    Labor income $w_t e_{j,s,t} n_{j,s,t}$ is split into two bands at
    the cutoffs $f_{1,t} \bar{y}$ and $f_{2,t} \bar{y}$, where
    $f_{1,t}$ and $f_{2,t}$ are ``factor1_db_ndc_hybrid`` and
    ``factor2_db_ndc_hybrid`` in period t and $\bar{y}$ is
    ``avg_labor_inc_t1_model_units``, the average labor income in the
    first model period in model units, taken from outside the model.
    The cutoffs therefore do not depend on any endogenous variable.
    Both factors may vary over time; the cutoffs that apply to any
    income are those of the period in which it was earned. Income up
    to $f_{1,t} \bar{y}$ is subject to the payroll tax ``tau_payroll``
    and earns a defined benefit computed with the ``DB_amount``
    formula and parameters on that capped income. Income between
    $f_{1,t} \bar{y}$ and $f_{2,t} \bar{y}$ is subject to the NDC
    contribution rate ``tau_p`` and earns a notional defined
    contribution benefit computed with the ``NDC_amount`` formula and
    parameters. Income above $f_{2,t} \bar{y}$ earns no benefit. The
    pension is the sum:

    .. math::
        pension_{j,s,t} = pension^{DB}_{j,s,t}\bigl(\min\{w e n,
            f_1 \bar{y}\}\bigr) + pension^{NDC}_{j,s,t}\bigl(\max\{
            \min\{w e n, f_2 \bar{y}\} - f_1 \bar{y}, 0\}\bigr)
            \quad \forall s \geq R

    where the argument of each term is the labor income that enters
    that system, with the cutoffs of the period of each earning.

    Args:
        w (array_like): real wage rate
        e (Numpy array): effective labor units
        n (Numpy array): labor supply; same shapes as accepted by
            ``DB_amount`` and ``NDC_amount``
        r (array_like): interest rate, used for the steady-state NDC
            rates when ``ndc_rates`` is not given
        ndc_rates (NDCRates): calendar paths of the NDC credited and
            discount rates (``ndc_rates_path``), or None for the
            steady-state rates; see ``NDC_amount``
        t (int): model period of the first element of ``n``
        j (int): index of lifetime income group, None if all groups
        method (str): adjusts calculation dimensions based on 'SS',
            'TPI', or 'TPI_scalar'
        p (OG-Core Specifications object): model parameters

    Returns:
        pension (Numpy array): pension amount for each household

    Raises:
        ValueError: if ``factor2_db_ndc_hybrid`` is less than
            ``factor1_db_ndc_hybrid`` in any period
    """
    # paths by model period (length T + S once Specifications extends
    # them); a single value is also accepted
    factor1 = np.atleast_1d(
        np.asarray(p.factor1_db_ndc_hybrid, dtype=float)
    ).ravel()
    factor2 = np.atleast_1d(
        np.asarray(p.factor2_db_ndc_hybrid, dtype=float)
    ).ravel()
    n_per = max(factor1.size, factor2.size)
    f1_full = factor1[np.minimum(np.arange(n_per), factor1.size - 1)]
    f2_full = factor2[np.minimum(np.arange(n_per), factor2.size - 1)]
    if np.any(f2_full < f1_full):
        bad = int(np.argmax(f2_full < f1_full))
        raise ValueError(
            "factor2_db_ndc_hybrid must be at least factor1_db_ndc_hybrid "
            f"in every period: in period {bad}, factor1_db_ndc_hybrid="
            f"{f1_full[bad]} and factor2_db_ndc_hybrid={f2_full[bad]}"
        )
    # cutoffs in model units of labor income
    cut1 = factor1 * p.avg_labor_inc_t1_model_units
    cut2 = factor2 * p.avg_labor_inc_t1_model_units
    DB = DB_amount(w, e, n, t, j, method, p, earn_lo=0.0, earn_hi=cut1)
    NDC = NDC_amount(
        w,
        e,
        n,
        r,
        ndc_rates,
        t,
        j,
        method,
        p,
        earn_lo=cut1,
        earn_hi=cut2,
    )
    return DB + NDC


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


NDC_GROWTH_CHOICES = ("r", "Curr GDP", "LR GDP", "wage bill", "avg wage")


class NDCRates(NamedTuple):
    """
    Calendar paths of the rates used in a notional defined contribution
    system, built by ``ndc_rates_path``.

    Both fields are length T + S on the time path, or length 1 for a
    steady state. Periods before the path read the first element and
    periods after it the last.
    """

    #: Numpy array: rate credited to notional balances in each model
    #: period, the growth realized from the previous period (after any
    #: averaging set by ``ndc_growth_avg_years``)
    credit: np.ndarray
    #: Numpy array: rate used to discount payments in the conversion
    #: coefficient of a household retiring in each model period
    discount: np.ndarray


def _ss_growth_rate(choice, r, p):
    """
    Steady-state value of an NDC growth rate.

    Stationarized aggregates are constant in the steady state, so GDP
    and the wage bill grow at the rate of productivity and population
    growth, exp(g_y)(1 + g_n) - 1, and the average wage at the rate of
    productivity growth, exp(g_y) - 1.

    Args:
        choice (str): one of ``NDC_GROWTH_CHOICES``
        r (array_like): interest rate; its last element is used
        p (OG-Core Specifications object): model parameters

    Returns:
        rate (scalar): steady-state growth rate per model period

    """
    g_y = float(np.asarray(p.g_y).flat[-1])
    g_n_ss = float(getattr(p, "g_n_ss", np.asarray(p.g_n).flat[-1]))
    if choice == "r":
        return float(np.asarray(r).flat[-1])
    elif choice in ("LR GDP", "Curr GDP", "wage bill"):
        return np.exp(g_y) * (1.0 + g_n_ss) - 1.0
    elif choice == "avg wage":
        return np.exp(g_y) - 1.0
    raise ValueError(
        f"NDC growth rate must be one of {NDC_GROWTH_CHOICES}, got {choice!r}"
    )


def growth_rate_path(choice, r, Y, w, L, p, method):
    r"""
    Calendar path of an NDC growth rate.

    Element t is the growth realized from period t - 1 to period t,
    in real, non-stationarized terms. Stationarized aggregates are
    divided by $e^{g_y t}$ and by population, and ``p.g_n[t]`` is
    population growth from t to t + 1, so for an aggregate X

    .. math::
        g_{t} = \frac{\hat{X}_{t}}{\hat{X}_{t-1}}e^{g_y}
            (1 + g_{n,t-1}) - 1

    where X is GDP for 'Curr GDP' and the wage bill wL for
    'wage bill'. 'avg wage' is the growth of earnings per person,
    the same expression without population growth. 'r' is the
    interest rate path and 'LR GDP' the constant steady-state growth
    of GDP. Growth into period 0 uses the period-0 level as the level
    of the period before.

    Args:
        choice (str): one of ``NDC_GROWTH_CHOICES``
        r (array_like): interest rate path
        Y (array_like): GDP path, used for 'Curr GDP'
        w (array_like): real wage path, used for 'wage bill' and
            'avg wage'
        L (array_like): aggregate labor path, used for 'wage bill' and
            'avg wage'
        p (OG-Core Specifications object): model parameters
        method (str): 'SS' for the steady-state value or 'TPI' for the
            time path

    Returns:
        rate (Numpy array): length 1 in the steady state, else T + S

    """
    if method == "SS" or choice == "LR GDP":
        return np.array([_ss_growth_rate(choice, r, p)])
    n_per = p.T + p.S

    def _extend(x):
        # paths may stop at T; hold the last value
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()[:n_per]
        return np.append(x, np.full(n_per - x.size, x[-1]))

    if choice == "r":
        return _extend(r)
    if choice == "Curr GDP":
        X = _extend(Y)
    elif choice in ("wage bill", "avg wage"):
        X = _extend(w) * _extend(L)
    else:
        raise ValueError(
            f"NDC growth rate must be one of {NDC_GROWTH_CHOICES}, got "
            f"{choice!r}"
        )
    g_y = float(np.asarray(p.g_y).flat[-1])
    X_prev = np.append(X[0], X[:-1])
    rate = X / X_prev * np.exp(g_y)
    if choice == "wage bill" or choice == "Curr GDP":
        g_n_ss = float(getattr(p, "g_n_ss", np.asarray(p.g_n).flat[-1]))
        g_n_preTP = float(getattr(p, "g_n_preTP", g_n_ss))
        # population growth from t - 1 to t
        g_n_prev = np.append(g_n_preTP, _extend_pop(p.g_n, n_per, g_n_ss))
        rate = rate * (1.0 + g_n_prev[:n_per])
    return rate - 1.0


def _extend_pop(g_n, n_per, g_n_ss):
    """
    Population growth path g_n[t] (growth from t to t + 1) for
    t = 0, ..., n_per - 2, padded with the steady-state rate.
    """
    g_n = np.atleast_1d(np.asarray(g_n, dtype=float)).ravel()[: n_per - 1]
    return np.append(g_n, np.full(n_per - 1 - g_n.size, g_n_ss))


def ndc_rates_path(r, Y, w, L, p, method):
    """
    Credited return and annuity discount rate paths for an NDC system.

    The credited return is ``ndc_growth_rate`` averaged over the
    ``ndc_growth_avg_years`` years up to and including each period
    (periods before the path take the period-0 rate). The discount
    rate is ``dir_growth_rate``; a household's conversion coefficient
    uses its value in the period the household retires.

    Args:
        r (array_like): interest rate path (scalar in the steady state)
        Y (array_like): GDP path
        w (array_like): real wage path
        L (array_like): aggregate labor path
        p (OG-Core Specifications object): model parameters
        method (str): 'SS' or 'TPI'

    Returns:
        rates (NDCRates): credit and discount rate paths

    """
    credit = growth_rate_path(p.ndc_growth_rate, r, Y, w, L, p, method)
    years = float(getattr(p, "ndc_growth_avg_years", 1))
    window = max(1, int(np.rint(years * p.S / 80.0)))
    if window > 1 and credit.size > 1:
        padded = np.append(np.full(window - 1, credit[0]), credit)
        credit = np.convolve(padded, np.ones(window) / window, "valid")
    discount = growth_rate_path(p.dir_growth_rate, r, Y, w, L, p, method)
    return NDCRates(credit, discount)


def g_ndc(r, Y, p):
    """
    Steady-state growth rate credited to NDC balances.

    Args:
        r (array_like): interest rate; its last element is used
        Y (array_like): GDP (unused: GDP growth in the steady state
            follows from g_y and g_n)
        p (OG-Core Specifications object): model parameters

    Returns:
        g_ndc (scalar): growth rate credited to NDC balances

    """
    return _ss_growth_rate(p.ndc_growth_rate, r, p)


def g_dir(r, Y, p):
    """
    Steady-state rate used to discount payments in the NDC conversion
    coefficient.

    Args:
        r (array_like): interest rate; its last element is used
        Y (array_like): GDP (unused: GDP growth in the steady state
            follows from g_y and g_n)
        p (OG-Core Specifications object): model parameters

    Returns:
        g_dir (scalar): discount rate in the conversion coefficient

    """
    return _ss_growth_rate(p.dir_growth_rate, r, p)


def annuity_mort_rates(p):
    """
    Mortality rates by period and age used to annuitize NDC balances.

    These are the model's mortality rates ``rho``, averaged over
    lifetime income groups with their population weights ``lambdas``,
    so that a single conversion coefficient applies to every income
    group. A ``mort_rates_SS`` attribute set on ``p`` by hand (a vector
    by age) overrides ``rho`` and is used in every period.

    Args:
        p (OG-Core Specifications object): model parameters

    Returns:
        mort_rates (Numpy array): (T + S) x S mortality rates, where
            row t, column s is the probability that a household of age
            s in period t dies before period t + 1

    """
    mort_rates_SS = getattr(p, "mort_rates_SS", None)
    if mort_rates_SS is not None:
        mort_rates_SS = np.asarray(mort_rates_SS, dtype=float).ravel()
        return np.broadcast_to(
            mort_rates_SS, (p.T + p.S, mort_rates_SS.shape[0])
        )
    rho = np.asarray(p.rho, dtype=float)
    if rho.ndim == 3:
        rho = rho @ np.asarray(p.lambdas, dtype=float).ravel()
    return np.broadcast_to(rho, (p.T + p.S, rho.shape[-1]))


def delta_ret(
    r, Y, p, S_ret=None, indR=None, k_ret=None, t_ret=None, g_dir_path=None
):
    r"""
    Compute conversion coefficient for the NDC pension amount

    .. math::
        \delta_{R} = (dir_{R} + ind_{R} - k)^{-1}

    The mortality rates entering $dir_{R}$ are set by the parameter
    ``ndc_mortality``:

    * 'steady state': the steady-state mortality rates, for every
      cohort
    * 'period': the mortality rates by age in the period in which the
      household retires (a period life table)
    * 'cohort': the mortality rates the household's own birth cohort
      faces at each age after retirement (a projected cohort life
      table), the rates that also enter the household's problem

    Payments are discounted at ``dir_growth_rate``. Given a calendar
    path of that rate (``g_dir_path``), each household uses its value
    in the period it retires, held fixed over the payments; otherwise
    the steady-state rate is used.

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
        t_ret (array_like): model period in which the household reaches
            age ``S_ret``, broadcastable against ``S_ret``; may lie
            before 0 or after T + S - 1, in which case the first or
            last row of the mortality rates is used. None for the
            steady state.
        g_dir_path (array_like): calendar path of the discount rate
            (``NDCRates.discount``), read at ``t_ret`` (clipped to the
            path); None to use the steady-state rate

    Returns:
        delta_ret (scalar or Numpy array): conversion coefficient for
            the NDC pension amount, shaped like ``S_ret``

    """
    mort_rates = annuity_mort_rates(p)
    n_rows, n_ages = mort_rates.shape
    if S_ret is None:
        S_ret = cohort_array("retire", p)[-1, -1]
    if indR is None:
        indR = cohort_array("indR", p)[-1, -1]
    if k_ret is None:
        k_ret = cohort_array("k_ret", p)[-1, -1]
    S_ret = np.asarray(S_ret, dtype=int)
    # discount rate of each household: the rate in its retirement period
    if g_dir_path is None or t_ret is None:
        g_dir_value = np.full(S_ret.shape, g_dir(r, Y, p))
    else:
        g_dir_path = np.atleast_1d(np.asarray(g_dir_path, dtype=float))
        idx = np.clip(np.asarray(t_ret, dtype=int), 0, g_dir_path.size - 1)
        g_dir_value = np.broadcast_to(g_dir_path[idx], S_ret.shape)
    method = getattr(p, "ndc_mortality", "steady state")
    if t_ret is None or method == "steady state":
        t_ret = np.full(S_ret.shape, n_rows - 1)
    elif method == "period":
        t_ret = np.clip(np.asarray(t_ret, dtype=int), 0, n_rows - 1)
    elif method == "cohort":
        # a cohort retiring S periods or more before the path begins
        # reads the first row at every age, and one retiring after it
        # ends reads the last row
        t_ret = np.clip(np.asarray(t_ret, dtype=int), -p.S, n_rows - 1)
    else:
        raise ValueError(
            "ndc_mortality must be one of 'steady state', 'period', "
            f"or 'cohort', got {method!r}"
        )
    t_ret = np.broadcast_to(t_ret, S_ret.shape)
    # the annuity factor depends only on the retirement age, the
    # mortality rates' period of retirement, and the discount rate, so
    # compute it once per distinct combination
    keys, inverse = np.unique(
        np.stack(
            [S_ret.ravel(), t_ret.ravel(), np.ravel(g_dir_value)], axis=1
        ),
        axis=0,
        return_inverse=True,
    )
    ages = np.arange(n_ages)
    dir_delta_pairs = np.zeros(keys.shape[0])
    for i, (R, tR, g) in enumerate(keys):
        R, tR = int(R), int(tR)
        if method == "cohort":
            # age a is reached in period tR + a - R
            rows = np.clip(tR - R + ages, 0, n_rows - 1)
        else:
            rows = np.full(n_ages, tR)
        surv_rates = np.ascontiguousarray(1.0 - mort_rates[rows, ages])
        dir_delta_pairs[i] = delta_ret_loop(p.S, R, surv_rates, g)
    dir_delta = dir_delta_pairs[np.ravel(inverse)].reshape(S_ret.shape)
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
def delta_ret_loop(S, S_ret, surv_rates, g_dir_value):
    r"""
    Compute the annuity factor in the NDC conversion coefficient

    .. math::
        dir_{R} = \sum_{k=0}^{S-R-1}\left[\prod_{i=0}^{k-1}
            (1 - \hat{\rho}_{R+i})\right](1 + g_{dir})^{-k}

    The pension is paid at ages R, ..., S - 1. Following the household
    mortality convention, ``1 - rho[s]`` is the probability of
    surviving from age s to age s + 1, so the payment at age R + k is
    received with probability prod_{i=0}^{k-1} (1 - rho[R + i]) by a
    household alive at retirement age R.

    Args:
        S (int): number of periods in the model
        S_ret (int): retirement age
        surv_rates (Numpy array): length S survival rates by age,
            1 - rho
        g_dir_value (scalar): discount rate used in the conversion
            coefficient

    Returns:
        dir_delta (scalar): expected discounted number of pension
            payments per unit of annual benefit
    """
    dir_delta = 0.0
    cumul_surv = 1.0
    discount = 1.0
    for k in range(S - S_ret):
        dir_delta += cumul_surv * discount
        cumul_surv *= surv_rates[S_ret + k]
        discount /= 1 + g_dir_value
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
    w,
    e,
    n,
    S_ret,
    S,
    g_y,
    DB,
    avg_earn_num_years,
    alpha_db,
    yr_contr,
    earn_lo=None,
    earn_hi=None,
    t_birth=0,
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
        earn_lo (Numpy array): calendar path of the lower cutoff of
            the earnings band, in model units of labor income; only
            labor income w * e * n above it counts. None (the default)
            applies no band.
        earn_hi (Numpy array): calendar path of the upper cutoff of
            the earnings band; only labor income below it counts
        t_birth (int): model period in which the household is age 0,
            so earnings at age s were made in period t_birth + s

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
            if earn_lo is None:
                L_inc_avg += w[s] * e[s] * n[s] / np.exp(g_y[-1] * (u - s))
            else:
                x = earnings_band_at(
                    w[s] * e[s] * n[s], earn_lo, earn_hi, t_birth + s
                )
                L_inc_avg += x / np.exp(g_y[-1] * (u - s))
        DB[u] = yr_contr[u] * alpha_db[u] * L_inc_avg / ny

    return DB


@numba.jit(nopython=True)
def DB_2dim_loop(
    w,
    e,
    n,
    S_ret,
    S,
    g_y,
    DB,
    avg_earn_num_years,
    alpha_db,
    yr_contr,
    earn_lo=None,
    earn_hi=None,
    t_cross=0,
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
        earn_lo (Numpy array): calendar path of the lower cutoff of
            the earnings band, in model units of labor income; only
            labor income w * e * n above it counts. None (the default)
            applies no band.
        earn_hi (Numpy array): calendar path of the upper cutoff of
            the earnings band; only labor income below it counts
        t_cross (int): model period of the cross section; the
            household of age u earned at age s in period t_cross - u + s

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
            if earn_lo is None:
                L_inc_avg += (
                    w[s] * e[s, :] * n[s, :] / np.exp(g_y[-1] * (u - s))
                )
            else:
                x = earnings_band_at(
                    w[s] * e[s, :] * n[s, :],
                    earn_lo,
                    earn_hi,
                    t_cross - u + s,
                )
                L_inc_avg += x / np.exp(g_y[-1] * (u - s))
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
    earn_lo=None,
    earn_hi=None,
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
        earn_lo (Numpy array): calendar path of the lower cutoff of
            the earnings band, in model units of labor income; only
            labor income w * e * n above it counts. None (the default)
            applies no band.
        earn_hi (Numpy array): calendar path of the upper cutoff of
            the earnings band; only labor income below it counts
            Both paths start in the period of the first row of n.

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
            x = e_ss[s_idx, :] * n_hist
            if earn_lo is None:
                L = (w_hist[:, None] / np.exp(g_y * (u - s_idx))[:, None]) * x
            else:
                # band of labor income, with the cutoffs in effect in
                # the period of each earning
                x = w_hist[:, None] * x
                for i in range(ny):
                    x[i, :] = earnings_band_at(
                        x[i, :], earn_lo, earn_hi, tau[i]
                    )
                L = x / np.exp(g_y * (u - s_idx))[:, None]
            DB[t, u, :] = (
                (L.sum(axis=0) / ny) * yr_contr[t, u] * alpha_db[t, u, :]
            )

    return DB


@numba.jit(nopython=True)
def NDC_1dim_loop(
    w,
    e,
    n,
    S_ret,
    S,
    g_y,
    tau_p,
    g_ndc,
    t_birth,
    delta_ret,
    NDC,
    earn_lo=None,
    earn_hi=None,
):
    r"""
    Calculate public pension from a notional defined contribution
    system for one household over its lifetime.

    The contribution made at working age s earns the rate credited in
    each later period up to the period before retirement:

    .. math::
        \prod_{a=s+1}^{R-1}(1 + g_{NDC,t_0 + a})

    where $t_0$ is the household's period of birth (age 0).

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
        g_ndc (Numpy array): calendar path of the credited rate
            (``NDCRates.credit``), a 1-D float array; periods outside
            it read the first or last element
        t_birth (int): model period in which the household is age 0
        delta_ret (Numpy array): conversion coefficient for the NDC
            pension amount that applies at each age u, length S
        NDC (Numpy array): pension amount for each household (filled
            in and returned), length S
        earn_lo (Numpy array): calendar path of the lower cutoff of
            the earnings band, in model units of labor income; only
            labor income w * e * n above it counts. None (the default)
            applies no band.
        earn_hi (Numpy array): calendar path of the upper cutoff of
            the earnings band; only labor income below it counts
            The cutoffs are read in period t_birth + s for age s.

    Returns:
        NDC (Numpy array): pension amount for each household

    """
    # credited rate at each age of the household's life
    g_age = g_ndc[np.clip(t_birth + np.arange(S), 0, g_ndc.size - 1)]
    for u in range(S):
        R = S_ret[u]
        if u < R:
            continue
        NDC_s = 0.0
        growth = 1.0
        for s in range(R - 1, -1, -1):
            # TODO: update so can take g_y from period t
            if earn_lo is None:
                x = e[s] * n[s]
                NDC_s += (
                    tau_p[s] * (w[s] / np.exp(g_y[-1] * (u - s))) * x * growth
                )
            else:
                x = earnings_band_at(
                    w[s] * e[s] * n[s], earn_lo, earn_hi, t_birth + s
                )
                NDC_s += tau_p[s] * (x / np.exp(g_y[-1] * (u - s))) * growth
            growth *= 1.0 + g_age[s]
        NDC[u] = delta_ret[u] * NDC_s
    return NDC


@numba.jit(nopython=True)
def NDC_2dim_loop(
    w,
    e,
    n,
    S_ret,
    S,
    g_y,
    tau_p,
    g_ndc,
    t_cross,
    delta_ret,
    NDC,
    earn_lo=None,
    earn_hi=None,
):
    """
    Calculate public pension from a notional defined contribution
    system for a cross section of S x J households in one period.

    As in ``NDC_1dim_loop``, contributions compound at the rates
    credited in the periods the household lived through: the household
    of age u in period ``t_cross`` was age s in period
    t_cross - u + s.

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
        g_ndc (Numpy array): calendar path of the credited rate
        t_cross (int): model period of the cross section
        delta_ret (Numpy array): conversion coefficient for the NDC
            pension amount of the household of each age u, length S
        NDC (Numpy array): pension amount for each household (filled
            in and returned), size S x J
        earn_lo (Numpy array): calendar path of the lower cutoff of
            the earnings band, in model units of labor income; only
            labor income w * e * n above it counts. None (the default)
            applies no band.
        earn_hi (Numpy array): calendar path of the upper cutoff of
            the earnings band; only labor income below it counts
            The cutoffs are read in period t_cross - u + s.

    Returns:
        NDC (Numpy array): pension amount for each household

    """
    n_g = g_ndc.shape[0]
    for u in range(S):
        R = S_ret[u]
        if u < R:
            continue
        NDC_s = np.zeros(NDC.shape[1])
        growth = 1.0
        for s in range(R - 1, -1, -1):
            if earn_lo is None:
                x = e[s, :] * n[s, :]
                NDC_s += (
                    tau_p[s] * (w[s] / np.exp(g_y[-1] * (u - s))) * x * growth
                )
            else:
                x = earnings_band_at(
                    w[s] * e[s, :] * n[s, :],
                    earn_lo,
                    earn_hi,
                    t_cross - u + s,
                )
                NDC_s += tau_p[s] * (x / np.exp(g_y[-1] * (u - s))) * growth
            idx = min(max(t_cross - u + s, 0), n_g - 1)
            growth *= 1.0 + g_ndc[idx]
        NDC[u, :] = delta_ret[u] * NDC_s
    return NDC


@numba.jit(nopython=True)
def NDC_3dim_loop(
    w_path,
    e_ss,
    n,
    n_preTP,
    S_ret,
    S,
    J,
    g_y,
    tau_p,
    g_ndc,
    delta_ret,
    earn_lo=None,
    earn_hi=None,
):
    r"""
    Calculate public pension from a notional defined contribution
    system over the full time path.

    Used when the TPI solution evaluates taxes for all periods at once
    to compute aggregates. As in ``DB_3dim_loop``, each retiree's
    benefit is built from their own cohort's history: the contribution
    at age s of a household aged u at time t was made in period
    t - (u - s), at that period's wage, labor supply, and contribution
    rate, and earns the rates credited in periods t - (u - s) + 1
    through t - (u - R) - 1. Histories that predate the time path use
    the period-0 wage, contribution rate, and credited rate and the
    model's initial labor supply, so this reproduces what the
    per-cohort household solves compute.

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
        g_ndc (Numpy array): calendar path of the credited rate
        delta_ret (Numpy array): conversion coefficient for the NDC
            pension amount by period and age, size T x S
        earn_lo (Numpy array): calendar path of the lower cutoff of
            the earnings band, in model units of labor income; only
            labor income w * e * n above it counts. None (the default)
            applies no band.
        earn_hi (Numpy array): calendar path of the upper cutoff of
            the earnings band; only labor income below it counts
            Both paths start in the period of the first row of n.

    Returns:
        NDC (Numpy array): pension amount for each household, size
            TxSxJ
    """
    T = n.shape[0]
    n_g = g_ndc.shape[0]
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
            growth = 1.0
            for i in range(R - 1, -1, -1):
                s = s_idx[i]
                if earn_lo is None:
                    contrib[i] = (
                        tau_p[rows[i], s]
                        * w_hist[i]
                        / np.exp(g_y * (u - s))
                        * growth
                    )
                else:
                    # the wage enters through the band of labor income
                    contrib[i] = (
                        tau_p[rows[i], s] / np.exp(g_y * (u - s)) * growth
                    )
                idx = min(max(tau[i], 0), n_g - 1)
                growth *= 1.0 + g_ndc[idx]
            x = e_ss[s_idx, :] * n_hist
            if earn_lo is not None:
                # band of labor income, with the cutoffs in effect in
                # the period of each contribution
                x = w_hist[:, None] * x
                for i in range(R):
                    x[i, :] = earnings_band_at(
                        x[i, :], earn_lo, earn_hi, tau[i]
                    )
            L = contrib[:, None] * x
            NDC[t, u, :] = delta_ret[t, u] * L.sum(axis=0)

    return NDC
