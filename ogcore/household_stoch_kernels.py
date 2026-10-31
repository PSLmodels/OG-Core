"""
------------------------------------------------------------------------
Compiled kernels for the stochastic-income household solver.

The solver in ``household_stoch`` spends nearly all of its time in a few
elementwise computations: marginal utilities and their inverses, the
OG-Core tax functions and their derivatives, the per-point Newton
iterations that invert the household's first-order conditions, monotone
cubic interpolation and the forward iteration of the distribution
(Young's method). This module implements them with numba.

Every kernel the solver calls has a numpy fallback:

* utilities, interpolation and the distribution have plain numpy/scipy
  versions (``*_np``), used when numba is unavailable or disabled;
* the tax functions fall back to OG-Core's own
  ``ogcore.txfunc.get_tax_rates`` (``household_stoch._NumpyBackend``),
  which also covers the nonparametric "mono"/"mono2D" tax functions that
  numba cannot compile.

Set the environment variable ``OGCORE_STOCH_DISABLE_NUMBA=1`` (or the
module attribute ``USE_NUMBA = False``) to force the numpy path.

Conventions: X and Y are labor and capital income in dollars (model
units times ``factor``), as in ``ogcore.txfunc.get_tax_rates``. "Raw"
tax rates are before the noncompliance and filer adjustments that
``ogcore.tax.ETR_income``/``MTR_income`` apply.
------------------------------------------------------------------------
"""

import os

import numpy as np
import scipy.interpolate as itp

try:
    import numba
except ImportError:  # pragma: no cover - numba is an ogcore dependency
    numba = None

HAVE_NUMBA = numba is not None
USE_NUMBA = HAVE_NUMBA and os.environ.get(
    "OGCORE_STOCH_DISABLE_NUMBA", ""
).lower() not in ("1", "true", "yes")


def _jit(func):
    """numba.njit with on-disk caching, or the plain function."""
    if numba is None:  # pragma: no cover
        return func
    return numba.njit(cache=True)(func)


def _jit_parallel(func):
    """
    numba.njit(parallel=True): loops over ``prange`` run on all cores
    (limit with the environment variable NUMBA_NUM_THREADS).
    """
    if numba is None:  # pragma: no cover
        return func
    return numba.njit(cache=True, parallel=True)(func)


prange = range if numba is None else numba.prange


# Stitching points, as in ogcore.household.marg_ut_cons/marg_ut_labor and
# household_stoch.marg_ut_beq
MU_C_EPS = 0.003
MU_B_EPS = 0.001
MDU_N_EPS = 1e-6

# Parametric tax function forms the numba path supports
TAX_FUNC_CODES = {"DEP": 0, "DEP_totalinc": 1, "GS": 2, "HSV": 3, "linear": 4}
_DEP, _DEP_TOTALINC, _GS, _HSV, _LINEAR = 0, 1, 2, 3, 4
# Smallest total income (dollars) at which GS/HSV are evaluated; they
# are singular at zero income
_MIN_INCOME = 1e-6

# Newton status codes returned by the per-point solvers. INFEASIBLE
# marks endogenous-grid knots that were not solved because the Euler
# equation implies c <= 0 there; CORNER marks a within-period solution
# with labor supply at a bound (``labor_floor`` or ltilde), where the
# labor FOC holds as an inequality.
CONVERGED, NOT_CONVERGED, BAD_VALUE, INFEASIBLE, CORNER = 0, 1, 2, 3, 4


"""
------------------------------------------------------------------------
    Marginal utilities
------------------------------------------------------------------------
"""


@_jit
def _mu_c(c, sigma):
    # ogcore.household.marg_ut_cons, including its stitching below c=eps
    if c < MU_C_EPS:
        b2 = (-sigma * (MU_C_EPS ** (-sigma - 1))) / 2
        b1 = (MU_C_EPS ** (-sigma)) - 2 * b2 * MU_C_EPS
        return 2 * b2 * c + b1
    return c ** (-sigma)


@_jit
def _mu_c_deriv(c, sigma):
    """Derivative of ``_mu_c`` in c."""
    if c < MU_C_EPS:
        return -sigma * (MU_C_EPS ** (-sigma - 1))
    return -sigma * c ** (-sigma - 1)


@_jit
def _inv_mu_c(value, sigma):
    """Exact inverse of ``_mu_c`` and its derivative dc/dvalue."""
    if value > MU_C_EPS ** (-sigma):
        b2 = (-sigma * (MU_C_EPS ** (-sigma - 1))) / 2
        b1 = (MU_C_EPS ** (-sigma)) - 2 * b2 * MU_C_EPS
        return (value - b1) / (2 * b2), 1.0 / (2 * b2)
    c = value ** (-1.0 / sigma)
    return c, -c / (sigma * value)


@_jit
def _mu_beq(b, sigma, chi_b):
    # household_stoch.marg_ut_beq: chi_b * b**(-sigma), stitched linearly
    # below b = eps
    if b < MU_B_EPS:
        b2 = -sigma * (MU_B_EPS ** (-sigma - 1))
        b1 = (MU_B_EPS ** (-sigma)) - b2 * MU_B_EPS
        return chi_b * (b2 * b + b1)
    return chi_b * b ** (-sigma)


@_jit
def _mdu_interior(n, b_ellipse, upsilon, ltilde):
    x = n / ltilde
    xu = x**upsilon
    x_um1 = xu / x  # x**(upsilon - 1)
    q = (1 - xu) ** ((1 - upsilon) / upsilon)
    level = (b_ellipse / ltilde) * x_um1 * q
    slope = (
        (b_ellipse / (ltilde * ltilde))
        * (upsilon - 1)
        * (x_um1 / x)
        * (q / (1 - xu))
    )
    return level, slope


@_jit
def _mdu_n(n, chi_n, b_ellipse, upsilon, ltilde):
    """
    Marginal disutility of labor (ogcore.household.marg_ut_labor, with
    its linear stitching near 0 and ltilde) and its derivative in n.
    """
    eps_low = MDU_N_EPS
    eps_high = ltilde - MDU_N_EPS
    if n < eps_low:
        level, slope = _mdu_interior(eps_low, b_ellipse, upsilon, ltilde)
        return chi_n * (level + slope * (n - eps_low)), chi_n * slope
    if n > eps_high:
        level, slope = _mdu_interior(eps_high, b_ellipse, upsilon, ltilde)
        return chi_n * (level + slope * (n - eps_high)), chi_n * slope
    level, slope = _mdu_interior(n, b_ellipse, upsilon, ltilde)
    return chi_n * level, chi_n * slope


@_jit
def _mu_c_vec(c, sigma):
    out = np.empty(c.shape)
    cf = c.ravel()
    of = out.ravel()
    for i in range(cf.size):
        of[i] = _mu_c(cf[i], sigma)
    return out


@_jit
def _inv_mu_c_vec(value, sigma):
    out = np.empty(value.shape)
    vf = value.ravel()
    of = out.ravel()
    for i in range(vf.size):
        of[i] = _inv_mu_c(vf[i], sigma)[0]
    return out


@_jit
def _mu_beq_vec(b, sigma, chi_b):
    out = np.empty(b.shape)
    bf = b.ravel()
    of = out.ravel()
    for i in range(bf.size):
        of[i] = _mu_beq(bf[i], sigma, chi_b)
    return out


@_jit
def _mdu_n_vec(n, chi_n, b_ellipse, upsilon, ltilde):
    level = np.empty(n.shape)
    slope = np.empty(n.shape)
    nf = n.ravel()
    lf = level.ravel()
    sf = slope.ravel()
    for i in range(nf.size):
        lf[i], sf[i] = _mdu_n(nf[i], chi_n, b_ellipse, upsilon, ltilde)
    return level, slope


def mu_c(c, sigma):
    """Marginal utility of consumption (OG-Core's, stitched below 0.003)."""
    c = np.asarray(c, dtype=float)
    if USE_NUMBA:
        return _mu_c_vec(c, float(sigma))
    return mu_c_np(c, sigma)


def inv_mu_c(value, sigma):
    """Consumption with marginal utility ``value``: inverse of ``mu_c``."""
    value = np.asarray(value, dtype=float)
    if USE_NUMBA:
        return _inv_mu_c_vec(value, float(sigma))
    return inv_mu_c_np(value, sigma)


def mu_beq(b, sigma, chi_b):
    """Marginal utility of bequests, chi_b * b**(-sigma), stitched."""
    b = np.asarray(b, dtype=float)
    if USE_NUMBA:
        return _mu_beq_vec(b, float(sigma), float(chi_b))
    return mu_beq_np(b, sigma, chi_b)


def mdu_n(n, chi_n, b_ellipse, upsilon, ltilde):
    """Marginal disutility of labor and its derivative in n."""
    n = np.asarray(n, dtype=float)
    if USE_NUMBA:
        return _mdu_n_vec(
            n, float(chi_n), float(b_ellipse), float(upsilon), float(ltilde)
        )
    return mdu_n_np(n, chi_n, b_ellipse, upsilon, ltilde)


# numpy fallbacks ------------------------------------------------------


def _mu_c_stitch(sigma):
    b2 = (-sigma * (MU_C_EPS ** (-sigma - 1))) / 2
    b1 = (MU_C_EPS ** (-sigma)) - 2 * b2 * MU_C_EPS
    return b1, b2


def mu_c_np(c, sigma):
    c = np.asarray(c, dtype=float)
    b1, b2 = _mu_c_stitch(sigma)
    low = c < MU_C_EPS
    out = np.empty(c.shape)
    out[~low] = c[~low] ** (-sigma)
    out[low] = 2 * b2 * c[low] + b1
    return out


def inv_mu_c_np(value, sigma):
    value = np.asarray(value, dtype=float)
    b1, b2 = _mu_c_stitch(sigma)
    high = value > MU_C_EPS ** (-sigma)
    out = np.empty(value.shape)
    out[~high] = value[~high] ** (-1.0 / sigma)
    out[high] = (value[high] - b1) / (2 * b2)
    return out


def mu_c_deriv_np(c, sigma):
    c = np.asarray(c, dtype=float)
    low = c < MU_C_EPS
    out = np.full(c.shape, -sigma * (MU_C_EPS ** (-sigma - 1)))
    out[~low] = -sigma * c[~low] ** (-sigma - 1)
    return out


def inv_mu_c_deriv_np(value, sigma):
    value = np.asarray(value, dtype=float)
    b1, b2 = _mu_c_stitch(sigma)
    high = value > MU_C_EPS ** (-sigma)
    c = inv_mu_c_np(value, sigma)
    out = np.empty(value.shape)
    out[~high] = -c[~high] / (sigma * value[~high])
    out[high] = 1.0 / (2 * b2)
    return out


def mu_beq_np(b, sigma, chi_b):
    b = np.asarray(b, dtype=float)
    low = b < MU_B_EPS
    b2 = -sigma * (MU_B_EPS ** (-sigma - 1))
    b1 = (MU_B_EPS ** (-sigma)) - b2 * MU_B_EPS
    out = np.empty(b.shape)
    out[~low] = b[~low] ** (-sigma)
    out[low] = b2 * b[low] + b1
    return chi_b * out


def _mdu_interior_np(n, b_ellipse, upsilon, ltilde):
    x = n / ltilde
    xu = x**upsilon
    level = (
        (b_ellipse / ltilde)
        * (x ** (upsilon - 1))
        * ((1 - xu) ** ((1 - upsilon) / upsilon))
    )
    slope = (
        (b_ellipse / ltilde**2)
        * (upsilon - 1)
        * (x ** (upsilon - 2))
        * ((1 - xu) ** ((1 - upsilon) / upsilon - 1))
    )
    return level, slope


def mdu_n_np(n, chi_n, b_ellipse, upsilon, ltilde):
    n = np.asarray(n, dtype=float)
    eps_low, eps_high = MDU_N_EPS, ltilde - MDU_N_EPS
    level = np.empty(n.shape)
    slope = np.empty(n.shape)
    low = n < eps_low
    high = n > eps_high
    mid = ~(low | high)
    lv, sl = _mdu_interior_np(n[mid], b_ellipse, upsilon, ltilde)
    level[mid], slope[mid] = lv, sl
    for mask, edge in ((low, eps_low), (high, eps_high)):
        lv, sl = _mdu_interior_np(edge, b_ellipse, upsilon, ltilde)
        level[mask] = lv + sl * (n[mask] - edge)
        slope[mask] = sl
    return chi_n * level, chi_n * slope


@_jit
def _labor_floor(b_ellipse, upsilon, ltilde):
    # see labor_floor
    level, slope = _mdu_interior(MDU_N_EPS, b_ellipse, upsilon, ltilde)
    n_floor = max(MDU_N_EPS - level / slope, 0.0)
    if n_floor > 0:
        n_floor += 1e-3 * (MDU_N_EPS - n_floor)
    return n_floor


def labor_floor(b_ellipse, upsilon, ltilde):
    """
    Lower bound on labor supply in the within-period solve: 0, or for
    upsilon > 2 just above the point where OG-Core's linear stitching of
    the marginal disutility of labor below n = 1e-6 turns negative.
    """
    level, slope = _mdu_interior_np(MDU_N_EPS, b_ellipse, upsilon, ltilde)
    n_floor = max(MDU_N_EPS - level / slope, 0.0)
    if n_floor > 0:
        n_floor += 1e-3 * (MDU_N_EPS - n_floor)
    return float(n_floor)


"""
------------------------------------------------------------------------
    Tax functions (raw rates and their derivatives in X and Y)
------------------------------------------------------------------------
"""


@_jit
def _dep(params, X, Y):
    A = params[0]
    B = params[1]
    C = params[2]
    D = params[3]
    max_x = params[4]
    max_y = params[5]
    share = params[6]
    min_x = params[7]
    min_y = params[8]
    shift_x = params[9]
    shift_y = params[10]
    shift = params[11]
    X2 = X * X
    Y2 = Y * Y
    den_x = A * X2 + B * X + 1
    den_y = C * Y2 + D * Y + 1
    tau_x = ((max_x - min_x) * (A * X2 + B * X) / den_x) + min_x
    tau_y = ((max_y - min_y) * (C * Y2 + D * Y) / den_y) + min_y
    base_x = tau_x + shift_x
    base_y = tau_y + shift_y
    pow_x = base_x**share
    pow_y = base_y ** (1 - share)
    rate = pow_x * pow_y + shift
    d_tau_x = (max_x - min_x) * ((2 * A * X + B) / (den_x * den_x))
    d_tau_y = (max_y - min_y) * ((2 * C * Y + D) / (den_y * den_y))
    # base**(share - 1) = pow_x / base_x etc. saves two powers; a term
    # that does not depend on its income source (share 0 or 1, or a flat
    # tau) has zero derivative even where the power is singular
    if share == 0 or d_tau_x == 0:
        d_x = 0.0
    elif base_x > 0:
        d_x = share * (pow_x / base_x) * d_tau_x * pow_y
    else:
        d_x = share * (base_x ** (share - 1)) * d_tau_x * pow_y
    if share == 1 or d_tau_y == 0:
        d_y = 0.0
    elif base_y > 0:
        d_y = (1 - share) * (pow_y / base_y) * d_tau_y * pow_x
    else:
        d_y = (1 - share) * (base_y ** (-share)) * d_tau_y * pow_x
    return rate, d_x, d_y


@_jit
def _dep_totalinc(params, X, Y):
    A = params[0]
    B = params[1]
    max_i = params[2]
    min_i = params[3]
    shift_i = params[4]
    shift = params[5]
    income = X + Y
    income2 = income * income
    den = A * income2 + B * income + 1
    rate = (
        ((max_i - min_i) * (A * income2 + B * income) / den)
        + min_i
        + shift_i
        + shift
    )
    d = (max_i - min_i) * ((2 * A * income + B) / (den * den))
    return rate, d, d


@_jit
def _gs_etr(params, X, Y):
    phi0 = params[0]
    phi1 = params[1]
    phi2 = params[2]
    income = max(X + Y, _MIN_INCOME)
    u = income**-phi1 + phi2
    g = u ** (-1 / phi1)
    rate = (phi0 * (income - g)) / income
    g_prime = income ** (-phi1 - 1) * u ** (-1 / phi1 - 1)
    d = phi0 * (g - g_prime * income) / (income * income)
    return rate, d, d


@_jit
def _gs_mtr(params, X, Y):
    phi0 = params[0]
    phi1 = params[1]
    phi2 = params[2]
    income = max(X + Y, _MIN_INCOME)
    u = income**-phi1 + phi2
    rate = phi0 * (1 - (income ** (-phi1 - 1) * u ** ((-1 - phi1) / phi1)))
    d = phi0 * phi2 * (1 + phi1) * income ** (-phi1 - 2) * u ** (-1 / phi1 - 2)
    return rate, d, d


@_jit
def _hsv_etr(params, X, Y):
    lambda_s = params[0]
    tau_s = params[1]
    income = max(X + Y, _MIN_INCOME)
    rate = 1 - (lambda_s * (income ** (-tau_s)))
    d = lambda_s * tau_s * income ** (-tau_s - 1)
    return rate, d, d


@_jit
def _hsv_mtr(params, X, Y):
    lambda_s = params[0]
    tau_s = params[1]
    income = max(X + Y, _MIN_INCOME)
    rate = 1 - (lambda_s * (1 - tau_s) * (income ** (-tau_s)))
    d = lambda_s * (1 - tau_s) * tau_s * income ** (-tau_s - 1)
    return rate, d, d


@_jit
def _etr_form(code, params, X, Y):
    if code == _DEP:
        return _dep(params, X, Y)
    if code == _DEP_TOTALINC:
        return _dep_totalinc(params, X, Y)
    if code == _GS:
        return _gs_etr(params, X, Y)
    if code == _HSV:
        return _hsv_etr(params, X, Y)
    return params[0], 0.0, 0.0


@_jit
def _mtr_form(code, params, X, Y):
    if code == _DEP:
        return _dep(params, X, Y)
    if code == _DEP_TOTALINC:
        return _dep_totalinc(params, X, Y)
    if code == _GS:
        return _gs_mtr(params, X, Y)
    if code == _HSV:
        return _hsv_mtr(params, X, Y)
    return params[0], 0.0, 0.0


@_jit
def _analytical_mtr(code, etr_params, X, Y, capital):
    # ogcore.txfunc.get_tax_rates with analytical_mtrs=True (DEP forms):
    # d(etr)/d(income source) * income + etr
    etr, d_x, d_y = _etr_form(code, etr_params, X, Y)
    if capital:
        return d_y * (X + Y) + etr
    return d_x * (X + Y) + etr


@_jit
def _mtr(code, analytical, etr_params, mtr_params, X, Y, capital):
    """
    Raw marginal tax rate on labor (capital=False) or capital income and
    its derivatives in X and Y, following ogcore.tax.MTR_income: with
    analytical MTRs the ETR parameters are used.
    """
    if analytical and (code == _DEP or code == _DEP_TOTALINC):
        # derivative of the analytical MTR by forward differences; only
        # the Newton Jacobian uses it
        rate = _analytical_mtr(code, etr_params, X, Y, capital)
        h_x = 1e-6 * max(abs(X), 1.0)
        h_y = 1e-6 * max(abs(Y), 1.0)
        rate_x = _analytical_mtr(code, etr_params, X + h_x, Y, capital)
        rate_y = _analytical_mtr(code, etr_params, X, Y + h_y, capital)
        return rate, (rate_x - rate) / h_x, (rate_y - rate) / h_y
    if analytical:
        return _mtr_form(code, etr_params, X, Y)
    return _mtr_form(code, mtr_params, X, Y)


@_jit
def _tax_rates_vec(code, kind, analytical, etr_params, mtr_params, X, Y):
    rate = np.empty(X.shape)
    d_x = np.empty(X.shape)
    d_y = np.empty(X.shape)
    for i in range(X.size):
        if kind == 0:
            r, dx, dy = _etr_form(code, etr_params, X[i], Y[i])
        else:
            r, dx, dy = _mtr(
                code, analytical, etr_params, mtr_params, X[i], Y[i], kind == 2
            )
        rate[i] = r
        d_x[i] = dx
        d_y[i] = dy
    return rate, d_x, d_y


def tax_rates(
    tax_func_type,
    kind,
    X,
    Y,
    etr_params,
    mtr_params=None,
    analytical_mtrs=False,
):
    """
    Raw tax rates and their derivatives in X and Y (numba).

    Args:
        tax_func_type (str): one of ``TAX_FUNC_CODES``
        kind (str): "etr", "mtrx" (labor income) or "mtry" (capital)
        X, Y (array_like): labor and capital income in dollars
        etr_params (array_like): ETR parameters for one age, (K,)
        mtr_params (array_like): MTR parameters (ignored for "etr" or
            with analytical MTRs)
        analytical_mtrs (bool): ``p.analytical_mtrs``

    Returns:
        rate, d_rate/dX, d_rate/dY (Numpy arrays, shape of X)
    """
    X, Y = np.broadcast_arrays(
        np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    )
    shape = X.shape
    etr_params = np.ascontiguousarray(etr_params, dtype=float)
    mtr_params = (
        etr_params
        if mtr_params is None
        else np.ascontiguousarray(mtr_params, dtype=float)
    )
    kind_code = {"etr": 0, "mtrx": 1, "mtry": 2}[kind]
    out = _tax_rates_vec(
        TAX_FUNC_CODES[tax_func_type],
        kind_code,
        bool(analytical_mtrs),
        etr_params,
        mtr_params,
        np.ascontiguousarray(X.ravel()),
        np.ascontiguousarray(Y.ravel()),
    )
    return tuple(o.reshape(shape) for o in out)


"""
------------------------------------------------------------------------
    Per-point household systems and Newton solvers

    Notation (one point = one asset level and shock for one type/age):
    wez = w e z, labor income per unit of n (model units)
    oth = bq + rm + tr + ubi + pension - bequest tax - min. consumption
          expenditure (all model units)
    Income tax liability (ogcore.tax.income_tax_liab with
    ETR_income's noncompliance and filer adjustments) is

        T_I = filer * etr(X, Y) * (X (1 - lnc) + Y (1 - knc)) / factor
              + tau_p * wez * n,

    with X = wez n factor and Y = r b factor; wealth tax
    T_W = filer_w * p_w h_w b^2 / (h_w b + m_w) (zero for b < 0).
------------------------------------------------------------------------
"""


@_jit
def _taxes_nonneg(
    b,
    n,
    wez,
    r,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    code,
    analytical,
    etr_params,
    mtrx_params,
):
    # _taxes for b >= 0
    X = wez * n * factor
    Y = r * b * factor
    etr, etr_x, etr_y = _etr_form(code, etr_params, X, Y)
    base = X * (1 - lnc) + Y * (1 - knc)
    T = filer_i * etr * base / factor + tau_p * wez * n
    dT_dn = (filer_i * (etr_x * base + etr * (1 - lnc)) + tau_p) * wez
    dT_db = filer_i * (etr_y * base + etr * (1 - knc)) * r
    if b > 0 and p_w != 0:
        den = h_w * b + m_w
        T += filer_w * p_w * h_w * b * b / den
        dT_db += filer_w * p_w * h_w * b * (h_w * b + 2 * m_w) / (den * den)
    mtr, mtr_x, mtr_y = _mtr(
        code, analytical, etr_params, mtrx_params, X, Y, False
    )
    adj = (1 - lnc) * filer_i
    return (
        T,
        dT_dn,
        dT_db,
        mtr * adj,
        mtr_x * adj * wez * factor,
        mtr_y * adj * r * factor,
    )


@_jit
def _taxes(
    b,
    n,
    wez,
    r,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    code,
    analytical,
    etr_params,
    mtrx_params,
):
    """
    Taxes paid and their derivatives at (b, n).

    Negative assets only occur at endogenous-grid knots below the
    borrowing limit (see ``_egm_point``); the tax functions are not
    defined for negative capital income, so there taxes are those at
    b = 0.

    Returns T (income + payroll + wealth tax), dT/dn, dT/db, the
    effective labor MTR (noncompliance and filer applied) and its
    derivatives in n and b.
    """
    T, dT_dn, dT_db, mtr, dm_dn, dm_db = _taxes_nonneg(
        max(b, 0.0),
        n,
        wez,
        r,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        code,
        analytical,
        etr_params,
        mtrx_params,
    )
    if b < 0:
        return T, dT_dn, 0.0, mtr, dm_dn, 0.0
    return T, dT_dn, dT_db, mtr, dm_dn, dm_db


@_jit
def _egm_eval(
    b,
    n,
    A,
    rhs,
    scale2,
    wez,
    oth,
    r,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
):
    # residuals of the labor FOC (f1) and budget constraint (f2) and the
    # Jacobian with respect to (n, b); see _egm_point
    T, dT_dn, dT_db, mtrx, dm_dn, dm_db = _taxes(
        b,
        n,
        wez,
        r,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        code,
        analytical,
        etr_params,
        mtrx_params,
    )
    mdu, dmdu = _mdu_n(n, chi_n, b_ellipse, upsilon, ltilde)
    f1 = 1 - tau_p - mtrx - mdu / A
    f2 = ((1 + r) * b + wez * n + oth - T - rhs) / scale2
    J11 = -dm_dn - dmdu / A
    J12 = -dm_db
    J21 = (wez - dT_dn) / scale2
    J22 = (1 + r - dT_db) / scale2
    return f1, f2, J11, J12, J21, J22


@_jit
def _labor_given_assets(
    b,
    n0,
    A,
    rhs,
    scale2,
    wez,
    oth,
    r,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    Labor supply solving the labor FOC f1(b, n) = 0 at given assets b.
    f1 = 1 - tau_p - MTRx - MDU(n)/A normally falls in n (not always
    with analytical MTRs), and the root is bracketed in (0, ltilde);
    safeguarded Newton with bisection.

    If f1 keeps one sign the bracket collapses onto 0 or ltilde: labor
    supply is at a corner, where the FOC holds as an inequality, and
    the point counts as solved. A bracket that collapses between points
    where f1 has opposite signs counts as solved only if |f1| is small
    (f1 is continuous for OG-Core's tax functions).

    Returns n, converged flag, and (f1, f2, J11, J12, J21, J22) at the
    final point.
    """
    lo = 0.0
    hi = ltilde
    lo_seen = False  # f1 > 0 seen at lo
    hi_seen = False  # f1 <= 0 seen at hi
    n = min(max(n0, 1e-6 * ltilde), (1 - 1e-6) * ltilde)
    ok = False
    vals = _egm_eval(
        b,
        n,
        A,
        rhs,
        scale2,
        wez,
        oth,
        r,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        chi_n,
        b_ellipse,
        upsilon,
        ltilde,
        code,
        analytical,
        etr_params,
        mtrx_params,
    )
    for it in range(maxiter):
        f1 = vals[0]
        J11 = vals[2]
        if not np.isfinite(f1):
            break
        if abs(f1) < tol:
            ok = True
            break
        if f1 > 0:
            lo = n
            lo_seen = True
        else:
            hi = n
            hi_seen = True
        if hi - lo < 1e-14 * ltilde:
            ok = not (lo_seen and hi_seen) or abs(f1) < 1e-6
            break
        n_new = n - f1 / J11 if J11 < 0 else 0.5 * (lo + hi)
        if not (lo < n_new < hi):
            n_new = 0.5 * (lo + hi)
        n = n_new
        vals = _egm_eval(
            b,
            n,
            A,
            rhs,
            scale2,
            wez,
            oth,
            r,
            factor,
            tau_p,
            filer_i,
            lnc,
            knc,
            filer_w,
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
        )
    return n, ok, vals


@_jit
def _egm_newton2d(
    b0,
    n0,
    A,
    rhs,
    scale2,
    wez,
    oth,
    r,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    Plain Newton on (b, n) for the labor FOC and budget constraint (see
    ``_egm_point``), n kept inside (0, ltilde). Fast when it works;
    ``_egm_point`` falls back to a bracketed method otherwise. Below the
    borrowing limit taxes are those at b = 0 (``_taxes``), so a root
    there is the knot's position on that extension.
    """
    b = b0
    n = min(max(n0, 1e-4 * ltilde), (1 - 1e-4) * ltilde)
    for it in range(maxiter):
        f1, f2, J11, J12, J21, J22 = _egm_eval(
            b,
            n,
            A,
            rhs,
            scale2,
            wez,
            oth,
            r,
            factor,
            tau_p,
            filer_i,
            lnc,
            knc,
            filer_w,
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
        )
        if not (np.isfinite(f1) and np.isfinite(f2)):
            return b, n, False, f1, f2
        if abs(f1) < tol and abs(f2) < tol:
            return b, n, True, f1, f2
        det = J11 * J22 - J12 * J21
        if det == 0 or not np.isfinite(det):
            return b, n, False, f1, f2
        dn = (-f1 * J22 + f2 * J12) / det
        db = (f1 * J21 - f2 * J11) / det
        lam = 1.0
        while n + lam * dn <= 0 or n + lam * dn >= ltilde:
            lam *= 0.5
        n += lam * dn
        b += lam * db
    return b, n, False, np.nan, np.nan


@_jit
def _egm_point(
    b0,
    n0,
    b_min,
    A,
    rhs,
    wez,
    oth,
    r,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    Given consumption c and savings b' at one point, find (b, n) such
    that the labor FOC and budget constraint hold:

        f1 = 1 - tau_p - MTRx(b, n) - MDU(n) / A = 0,
             A = MU(c) w e z / p_tilde
        f2 = [(1 + r) b + wez n + oth - T(b, n) - rhs] / (1 + |rhs|) = 0,
             rhs = p_tilde c + e^{g_y} b'

    Let n(b) solve the labor FOC at b (``_labor_given_assets``) and
    H(b) = f2(b, n(b)), the budget surplus of choosing (c, b') with
    assets b. The borrowing limit is b_min = 0. If H(b_min) >= 0 the
    household at the limit can afford more than (c, b'), so the knot
    lies below it; below zero assets taxes are those at b = 0
    (``_taxes``), n(b) is constant and H is linear, which places the
    knot at b = b_min - H(b_min) (1 + |rhs|) / (1 + r). Otherwise the
    root lies above b_min and is found by Newton's method on H
    (H' = J22 - J21 J12 / J11) safeguarded by bisection, since the tax
    functions can make H non-monotone just above zero assets.
    """
    scale2 = 1.0 + abs(rhs)
    b, n, ok, f1, f2 = _egm_newton2d(
        b0,
        n0,
        A,
        rhs,
        scale2,
        wez,
        oth,
        r,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        chi_n,
        b_ellipse,
        upsilon,
        ltilde,
        code,
        analytical,
        etr_params,
        mtrx_params,
        tol,
        8,
    )
    if ok:
        return b, n, CONVERGED, f1, f2
    b = max(b0, b_min)
    n, ok_n, vals = _labor_given_assets(
        b,
        n0,
        A,
        rhs,
        scale2,
        wez,
        oth,
        r,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        chi_n,
        b_ellipse,
        upsilon,
        ltilde,
        code,
        analytical,
        etr_params,
        mtrx_params,
        tol,
        maxiter,
    )
    # bracket [lo, hi] on H; lo = b_min is only a bracket end once
    # H(b_min) < 0 is verified (H(b_min) >= 0 puts the knot below b_min)
    lo = b_min
    lo_verified = False
    hi = np.inf
    step_up = 0.1 * (1.0 + abs(b))
    for it in range(4 * maxiter):
        f1, H, J11, J12, J21, J22 = vals
        if not (np.isfinite(H) and np.isfinite(f1)):
            return b, n, BAD_VALUE, f1, H
        if b == b_min and H >= 0:
            if not ok_n:
                return b, n, NOT_CONVERGED, f1, H
            return b_min - H * scale2 / (1 + r), n, CONVERGED, f1, 0.0
        if ok_n and abs(H) < tol:
            return b, n, CONVERGED, f1, H
        if H < 0:
            if b >= lo:
                lo = b
                lo_verified = True
        else:
            hi = min(hi, b)
        if lo_verified and hi - lo < 1e-15 * (1.0 + abs(lo)):
            # H changes sign across a bracket at machine precision: a
            # root unless H jumps there (then no knot exists)
            if ok_n and abs(H) < 1e-8:
                return b, n, CONVERGED, f1, H
            break
        dH = J22 - J21 * J12 / J11
        if dH > 0 and np.isfinite(dH):
            b_new = b - H / dH
        else:
            b_new = np.nan
        if not (lo < b_new < hi):
            if not lo_verified and not b_new > lo:
                b_new = b_min  # check the borrowing limit
            elif hi < np.inf:
                b_new = 0.5 * (lo + hi)
            else:
                b_new = lo + step_up
                step_up *= 2.0
        b = b_new
        n, ok_n, vals = _labor_given_assets(
            b,
            n,
            A,
            rhs,
            scale2,
            wez,
            oth,
            r,
            factor,
            tau_p,
            filer_i,
            lnc,
            knc,
            filer_w,
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
            tol,
            maxiter,
        )
    return b, n, NOT_CONVERGED, vals[0], vals[1]


@_jit_parallel
def _egm_solve(
    feasible,
    b0,
    n0,
    bp,
    b_min,
    A,
    rhs,
    wez,
    oth,
    filer_i,
    lnc,
    knc,
    filer_w,
    r,
    factor,
    tau_p,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
    nb,
    nz,
):
    """
    ``_egm_point`` at every knot. Points are ordered (type, b' node,
    shock) with b' increasing along each series; knots of a series are
    solved in order, each starting from a linear extrapolation of the
    two before it (series run in parallel). Like the filer shares and
    noncompliance rates, the payroll tax rate tau_p is per point (it
    varies by type).
    """
    P = b0.size
    b = np.empty(P)
    n = np.empty(P)
    status = np.empty(P, dtype=np.int64)
    res = np.empty(P)
    n_series = P // nb
    for sr in prange(n_series):
        m = sr // nz
        iz = sr % nz
        for k in range(nb):
            i = (m * nb + k) * nz + iz
            if not feasible[i]:
                b[i] = np.nan
                n[i] = np.nan
                status[i] = INFEASIBLE
                res[i] = np.nan
                continue
            bg = b0[i]
            ng = n0[i]
            if k >= 2:
                i1 = i - nz
                i2 = i - 2 * nz
                if (
                    status[i1] == CONVERGED
                    and status[i2] == CONVERGED
                    and b[i2] >= b_min[i]
                ):
                    slope = (bp[i] - bp[i1]) / (bp[i1] - bp[i2])
                    bg = b[i1] + slope * (b[i1] - b[i2])
                    ng = n[i1] + slope * (n[i1] - n[i2])
            bi, ni, st, f1, f2 = _egm_point(
                bg,
                ng,
                b_min[i],
                A[i],
                rhs[i],
                wez[i],
                oth[i],
                r,
                factor,
                tau_p[i],
                filer_i[i],
                lnc[i],
                knc[i],
                filer_w[i],
                h_w,
                m_w,
                p_w,
                chi_n,
                b_ellipse,
                upsilon,
                ltilde,
                code,
                analytical,
                etr_params,
                mtrx_params,
                tol,
                maxiter,
            )
            b[i] = bi
            n[i] = ni
            status[i] = st
            res[i] = max(abs(f1), abs(f2))
    return b, n, status, res


@_jit
def _static_eval(
    n,
    b,
    bp_fix,
    kappa,
    wez,
    oth,
    r,
    p_tilde,
    g_y,
    sigma,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
):
    """
    Labor FOC residual when consumption comes from the budget
    constraint at assets b and labor n, with savings b' = bp_fix +
    kappa c:

        c = [(1 + r) b + wez n + oth - T(b, n) - e^{g_y} bp_fix]
            / (p_tilde + e^{g_y} kappa)
        F = MU(c) wez (1 - tau_p - MTRx(b, n)) / p_tilde - MDU(n)

    Returns F, dF/dn, c and the scale |MU wez D / p_tilde| + |MDU| that
    convergence is measured against.
    """
    T, dT_dn, dT_db, mtrx, dm_dn, dm_db = _taxes(
        b,
        n,
        wez,
        r,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        code,
        analytical,
        etr_params,
        mtrx_params,
    )
    growth = np.exp(g_y)
    denom = p_tilde + growth * kappa
    c = ((1 + r) * b + wez * n + oth - T - growth * bp_fix) / denom
    dc = (wez - dT_dn) / denom
    D = 1 - tau_p - mtrx
    mu = _mu_c(c, sigma)
    dmu = _mu_c_deriv(c, sigma)
    mdu, dmdu = _mdu_n(n, chi_n, b_ellipse, upsilon, ltilde)
    gain = mu * wez * D / p_tilde
    F = gain - mdu
    dF = (dmu * dc * D - mu * dm_dn) * wez / p_tilde - dmdu
    return F, dF, c, abs(gain) + abs(mdu)


@_jit
def _static_search(
    n0,
    lo,
    hi,
    lo_seen,
    hi_seen,
    n_floor,
    b,
    bp_fix,
    kappa,
    wez,
    oth,
    r,
    p_tilde,
    g_y,
    sigma,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    Root of the labor FOC F (``_static_eval``) in [lo, hi] by
    safeguarded Newton with bisection, starting from n0. lo_seen
    (hi_seen) says F > 0 at lo (F <= 0 at hi) is already known; an
    unverified end is n_floor or ltilde, evaluated only when a Newton
    step heads there, and is a corner if F keeps its sign there.

    Returns n, c (from the budget constraint at that n), status and
    |F| / scale.
    """
    n = min(max(n0, lo + 1e-4 * (hi - lo)), hi - 1e-4 * (hi - lo))
    F = np.nan
    scale = 1.0
    c = np.nan
    for it in range(maxiter):
        F, dF, c, scale = _static_eval(
            n,
            b,
            bp_fix,
            kappa,
            wez,
            oth,
            r,
            p_tilde,
            g_y,
            sigma,
            factor,
            tau_p,
            filer_i,
            lnc,
            knc,
            filer_w,
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
        )
        if not (np.isfinite(F) and np.isfinite(c)):
            return n, c, BAD_VALUE, np.nan
        if abs(F) <= tol * scale:
            return n, c, CONVERGED, abs(F) / scale
        if F > 0:
            if n >= ltilde:
                return n, c, CORNER, 0.0  # works the whole endowment
            lo = n
            lo_seen = True
        else:
            if n <= n_floor:
                return n, c, CORNER, 0.0  # at the lower bound
            hi = n
            hi_seen = True
        if lo_seen and hi_seen and hi - lo <= 1e-14 * ltilde:
            # F changes sign across a bracket at machine precision
            return n, c, CONVERGED, abs(F) / scale
        n_new = n - F / dF if (dF < 0 and np.isfinite(dF)) else np.nan
        if not (lo < n_new < hi):
            if F > 0 and not hi_seen:
                n_new = hi  # check for a corner at the top
            elif F <= 0 and not lo_seen:
                n_new = lo  # check for a corner at the bottom
            else:
                n_new = 0.5 * (lo + hi)
        if it < maxiter - 1:
            n = n_new
    return n, c, NOT_CONVERGED, abs(F) / scale


@_jit
def _period_utility(n, c, weight, sigma, chi_n, b_ellipse, upsilon, ltilde):
    """
    Utility of consumption (weight times OG-Core's, stitched below
    c = 0.003 to match ``_mu_c``) plus utility of leisure, up to
    constants: the within-period objective at given savings.
    """
    if c >= MU_C_EPS:
        u = np.log(c) if sigma == 1.0 else c ** (1 - sigma) / (1 - sigma)
    else:
        b2 = (-sigma * (MU_C_EPS ** (-sigma - 1))) / 2
        b1 = (MU_C_EPS ** (-sigma)) - 2 * b2 * MU_C_EPS
        if sigma == 1.0:
            u = np.log(MU_C_EPS)
        else:
            u = MU_C_EPS ** (1 - sigma) / (1 - sigma)
        u += b1 * (c - MU_C_EPS) + b2 * (c * c - MU_C_EPS * MU_C_EPS)
    x = min(max(n / ltilde, 0.0), 1.0)
    leisure = chi_n * b_ellipse * (1 - x**upsilon) ** (1 / upsilon)
    return weight * u + leisure


# points scanned for interior optima when labor is at its lower bound
# with analytical MTRs
STATIC_SCAN = 12


def period_utility_np(n, c, weight, sigma, chi_n, b_ellipse, upsilon, ltilde):
    """numpy version of ``_period_utility`` (arrays n, c, weight)."""
    n, c, weight = np.broadcast_arrays(
        np.asarray(n, dtype=float),
        np.asarray(c, dtype=float),
        np.asarray(weight, dtype=float),
    )
    u = np.empty(c.shape)
    high = c >= MU_C_EPS
    c_high = c[high]
    if sigma == 1.0:
        u[high] = np.log(c_high)
        u_eps = np.log(MU_C_EPS)
    else:
        u[high] = c_high ** (1 - sigma) / (1 - sigma)
        u_eps = MU_C_EPS ** (1 - sigma) / (1 - sigma)
    b1, b2 = _mu_c_stitch(sigma)
    c_low = c[~high]
    u[~high] = (
        u_eps
        + b1 * (c_low - MU_C_EPS)
        + b2 * (c_low * c_low - MU_C_EPS * MU_C_EPS)
    )
    x = np.clip(n / ltilde, 0.0, 1.0)
    leisure = chi_n * b_ellipse * (1 - x**upsilon) ** (1 / upsilon)
    return weight * u + leisure


@_jit
def _static_point(
    n0,
    n_floor,
    b,
    bp_fix,
    kappa,
    wez,
    oth,
    r,
    p_tilde,
    g_y,
    sigma,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    Within-period solution at one point (see ``_static_solve``).

    Returns n, c (from the budget constraint at that n), status and
    |F| / scale.
    """
    n, c, st, res = _static_search(
        n0,
        n_floor,
        ltilde,
        False,
        False,
        n_floor,
        b,
        bp_fix,
        kappa,
        wez,
        oth,
        r,
        p_tilde,
        g_y,
        sigma,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        chi_n,
        b_ellipse,
        upsilon,
        ltilde,
        code,
        analytical,
        etr_params,
        mtrx_params,
        tol,
        maxiter,
    )
    if not analytical or not (st == CONVERGED or st == CORNER):
        return n, c, st, res
    # Analytical MTRs can exceed 1 - tau_p at low labor income, so that
    # not working (a corner) and an interior root can both be local
    # optima: keep the one with the higher within-period utility.
    weight = 1.0 + np.exp(g_y) * kappa / p_tilde
    best_n, best_c, best_st, best_res = n, c, st, res
    best_u = _period_utility(
        n, c, weight, sigma, chi_n, b_ellipse, upsilon, ltilde
    )
    if st == CORNER and n <= n_floor:
        # interior optima are where F turns from + to - on a scan
        n_prev = n_floor
        F_prev = -1.0  # F <= 0 at the lower bound
        c_prev = c
        for k in range(1, STATIC_SCAN + 1):
            nk = n_floor + (ltilde - n_floor) * (k / STATIC_SCAN) ** 2
            Fk, dFk, ck, sk = _static_eval(
                nk,
                b,
                bp_fix,
                kappa,
                wez,
                oth,
                r,
                p_tilde,
                g_y,
                sigma,
                factor,
                tau_p,
                filer_i,
                lnc,
                knc,
                filer_w,
                h_w,
                m_w,
                p_w,
                chi_n,
                b_ellipse,
                upsilon,
                ltilde,
                code,
                analytical,
                etr_params,
                mtrx_params,
            )
            if not np.isfinite(Fk):
                break
            if F_prev > 0 and Fk <= 0:
                n2, c2, st2, res2 = _static_search(
                    0.5 * (n_prev + nk),
                    n_prev,
                    nk,
                    True,
                    True,
                    n_floor,
                    b,
                    bp_fix,
                    kappa,
                    wez,
                    oth,
                    r,
                    p_tilde,
                    g_y,
                    sigma,
                    factor,
                    tau_p,
                    filer_i,
                    lnc,
                    knc,
                    filer_w,
                    h_w,
                    m_w,
                    p_w,
                    chi_n,
                    b_ellipse,
                    upsilon,
                    ltilde,
                    code,
                    analytical,
                    etr_params,
                    mtrx_params,
                    tol,
                    maxiter,
                )
                if st2 == CONVERGED:
                    u2 = _period_utility(
                        n2,
                        c2,
                        weight,
                        sigma,
                        chi_n,
                        b_ellipse,
                        upsilon,
                        ltilde,
                    )
                    if u2 > best_u:
                        best_n, best_c, best_st, best_res = n2, c2, st2, res2
                        best_u = u2
            n_prev = nk
            F_prev = Fk
            c_prev = ck
        if F_prev > 0 and n_prev >= ltilde:
            # F > 0 up to ltilde: working the whole endowment is a local
            # optimum too
            u2 = _period_utility(
                ltilde,
                c_prev,
                weight,
                sigma,
                chi_n,
                b_ellipse,
                upsilon,
                ltilde,
            )
            if u2 > best_u:
                best_n, best_c, best_st, best_res = ltilde, c_prev, CORNER, 0.0
                best_u = u2
    else:
        # an interior root (or the top corner): is not working better?
        F0, dF0, c0, s0 = _static_eval(
            n_floor,
            b,
            bp_fix,
            kappa,
            wez,
            oth,
            r,
            p_tilde,
            g_y,
            sigma,
            factor,
            tau_p,
            filer_i,
            lnc,
            knc,
            filer_w,
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
        )
        if np.isfinite(F0) and np.isfinite(c0) and F0 <= 0:
            u0 = _period_utility(
                n_floor, c0, weight, sigma, chi_n, b_ellipse, upsilon, ltilde
            )
            if u0 > best_u:
                best_n, best_c, best_st, best_res = n_floor, c0, CORNER, 0.0
    return best_n, best_c, best_st, best_res


@_jit_parallel
def _static_solve(
    n0,
    b,
    bp_fix,
    kappa,
    wez,
    oth,
    filer_i,
    lnc,
    knc,
    filer_w,
    r,
    p_tilde,
    g_y,
    sigma,
    factor,
    tau_p,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    Solve the within-period problem at each point: labor supply n and
    consumption c given assets b and the savings rule b' = bp_fix +
    kappa c. Consumption comes from the budget constraint, so it holds
    exactly at every n; n solves the labor FOC F(n) = 0 (see
    ``_static_eval``) on [n_floor, ltilde] (``labor_floor``).

    F > 0 at n_floor and F < 0 at ltilde bracket a root. If F <= 0 at
    n_floor (for example where OG-Core's analytical MTRs exceed
    1 - tau_p at low labor income) or F >= 0 at ltilde, labor supply is
    at that bound (status CORNER), where the FOC holds as an inequality.
    The bounds are only evaluated when Newton steps head there. With
    analytical MTRs F need not be monotone; then the corner at n_floor
    and interior roots where F turns from + to - (on a scan of
    ``STATIC_SCAN`` points) are compared by within-period utility
    (``_period_utility``; at the last age the bequest, kappa c, enters
    with it), so the solution is the global optimum among them. With
    estimated MTRs OG-Core's household is defined by its FOCs, which
    need not be those of any utility comparison, so the root the search
    from n0 reaches is the solution.

    Inputs indexed [i] (including tau_p, the payroll tax rate) are per
    point. Returns n, c, status and |F| / scale, per point.
    """
    P = b.size
    n_out = np.empty(P)
    c_out = np.empty(P)
    status = np.empty(P, dtype=np.int64)
    res = np.empty(P)
    n_floor = _labor_floor(b_ellipse, upsilon, ltilde)
    for i in prange(P):
        n, c, st, rs = _static_point(
            n0[i],
            n_floor,
            b[i],
            bp_fix[i],
            kappa[i],
            wez[i],
            oth[i],
            r,
            p_tilde,
            g_y,
            sigma,
            factor,
            tau_p[i],
            filer_i[i],
            lnc[i],
            knc[i],
            filer_w[i],
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
            tol,
            maxiter,
        )
        n_out[i] = n
        c_out[i] = c
        status[i] = st
        res[i] = rs
    return n_out, c_out, status, res


"""
------------------------------------------------------------------------
    Monotone cubic (PCHIP) interpolation, scipy-compatible, with linear
    extrapolation
------------------------------------------------------------------------
"""


@_jit
def _pchip_edge(h0, h1, m0, m1):
    # scipy.interpolate.PchipInterpolator._edge_case
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0
    return d


@_jit
def _pchip_slopes(x, y, d):
    K = x.size
    if K == 2:
        m = (y[1] - y[0]) / (x[1] - x[0])
        d[0] = m
        d[1] = m
        return
    for k in range(1, K - 1):
        h0 = x[k] - x[k - 1]
        h1 = x[k + 1] - x[k]
        m0 = (y[k] - y[k - 1]) / h0
        m1 = (y[k + 1] - y[k]) / h1
        if np.sign(m0) != np.sign(m1) or m0 == 0 or m1 == 0:
            d[k] = 0.0
        else:
            w1 = 2 * h1 + h0
            w2 = h1 + 2 * h0
            d[k] = (w1 + w2) / (w1 / m0 + w2 / m1)
    h0 = x[1] - x[0]
    h1 = x[2] - x[1]
    d[0] = _pchip_edge(h0, h1, (y[1] - y[0]) / h0, (y[2] - y[1]) / h1)
    h0 = x[K - 1] - x[K - 2]
    h1 = x[K - 2] - x[K - 3]
    d[K - 1] = _pchip_edge(
        h0, h1, (y[K - 1] - y[K - 2]) / h0, (y[K - 2] - y[K - 3]) / h1
    )


@_jit
def _pchip_eval_sorted(x, y, d, xq, out):
    """Evaluate at sorted ``xq``; linear extrapolation outside [x0, xK]."""
    K = x.size
    k = 0
    for q in range(xq.size):
        xv = xq[q]
        if xv <= x[0]:
            out[q] = y[0] + (xv - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
            continue
        if xv >= x[K - 1]:
            out[q] = y[K - 1] + (xv - x[K - 1]) * (y[K - 1] - y[K - 2]) / (
                x[K - 1] - x[K - 2]
            )
            continue
        while x[k + 1] < xv:
            k += 1
        h = x[k + 1] - x[k]
        t = (xv - x[k]) / h
        t2 = t * t
        t3 = t2 * t
        out[q] = (
            (2 * t3 - 3 * t2 + 1) * y[k]
            + (t3 - 2 * t2 + t) * h * d[k]
            + (-2 * t3 + 3 * t2) * y[k + 1]
            + (t3 - t2) * h * d[k + 1]
        )


@_jit
def _pchip_many(x, y, valid, xq):
    """
    PCHIP through the valid points of each series.

    Args:
        x (2-D array): knots, (M, K), increasing over valid entries
        y (3-D array): values, (V, M, K)
        valid (2-D bool array): which knots to use, (M, K)
        xq (2-D array): sorted query points, (M, Q)

    Returns:
        out (3-D array): (V, M, Q)
    """
    V = y.shape[0]
    M = x.shape[0]
    Q = xq.shape[1]
    out = np.empty((V, M, Q))
    for m in range(M):
        idx = np.nonzero(valid[m])[0]
        xs = x[m, idx]
        d = np.empty(idx.size)
        for v in range(V):
            ys = y[v, m, idx]
            if idx.size == 1:
                out[v, m, :] = ys[0]
                continue
            _pchip_slopes(xs, ys, d)
            _pchip_eval_sorted(xs, ys, d, xq[m], out[v, m])
    return out


@_jit
def _pchip_query(x, y, valid, starts, xq):
    """
    PCHIP of series m (knots x[m], values y[m] over valid entries) at
    queries xq[starts[m]:starts[m + 1]], in any order; linear
    extrapolation outside the knots.
    """
    M = x.shape[0]
    out = np.empty(xq.size)
    for m in range(M):
        q0 = starts[m]
        q1 = starts[m + 1]
        if q1 == q0:
            continue
        idx = np.nonzero(valid[m])[0]
        xs = x[m, idx]
        ys = y[m, idx]
        K = idx.size
        if K == 1:
            out[q0:q1] = ys[0]
            continue
        d = np.empty(K)
        _pchip_slopes(xs, ys, d)
        for q in range(q0, q1):
            xv = xq[q]
            if xv <= xs[0]:
                out[q] = ys[0] + (xv - xs[0]) * (ys[1] - ys[0]) / (
                    xs[1] - xs[0]
                )
            elif xv >= xs[K - 1]:
                out[q] = ys[K - 1] + (xv - xs[K - 1]) * (
                    ys[K - 1] - ys[K - 2]
                ) / (xs[K - 1] - xs[K - 2])
            else:
                k = np.searchsorted(xs, xv, side="right") - 1
                h = xs[k + 1] - xs[k]
                t = (xv - xs[k]) / h
                t2 = t * t
                t3 = t2 * t
                out[q] = (
                    (2 * t3 - 3 * t2 + 1) * ys[k]
                    + (t3 - 2 * t2 + t) * h * d[k]
                    + (-2 * t3 + 3 * t2) * ys[k + 1]
                    + (t3 - t2) * h * d[k + 1]
                )
    return out


def pchip_query(x, y, valid, rows, xq):
    """
    PCHIP of series rows[q] at xq[q] for each query q (any order); see
    ``_pchip_query``. Uses scipy when numba is disabled.
    """
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    valid = np.ascontiguousarray(valid, dtype=np.bool_)
    rows = np.asarray(rows, dtype=np.int64)
    xq = np.asarray(xq, dtype=float)
    order = np.argsort(rows, kind="stable")
    counts = np.bincount(rows, minlength=x.shape[0])
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    xq_sorted = np.ascontiguousarray(xq[order])
    if USE_NUMBA:
        vals = _pchip_query(x, y, valid, starts, xq_sorted)
    else:
        vals = np.empty(xq.size)
        for m in range(x.shape[0]):
            q0, q1 = starts[m], starts[m + 1]
            if q1 > q0:
                idx = np.nonzero(valid[m])[0]
                vals[q0:q1] = pchip_np(x[m, idx], y[m, idx], xq_sorted[q0:q1])
    out = np.empty(xq.size)
    out[order] = vals
    return out


def pchip_np(x, y, xq):
    """scipy PCHIP with linear extrapolation (numpy fallback, 1 series)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    if x.size == 1:
        return np.full(xq.shape, y[0])
    out = itp.PchipInterpolator(x, y, extrapolate=True)(xq)
    lo = xq < x[0]
    hi = xq > x[-1]
    out[lo] = y[0] + (xq[lo] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    out[hi] = y[-1] + (xq[hi] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


def pchip_many(x, y, valid, xq):
    """
    PCHIP interpolation of several value series per knot set; see
    ``_pchip_many``. Uses scipy when numba is disabled.
    """
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    valid = np.ascontiguousarray(valid, dtype=np.bool_)
    xq = np.ascontiguousarray(xq, dtype=float)
    if USE_NUMBA:
        return _pchip_many(x, y, valid, xq)
    V, M = y.shape[0], x.shape[0]
    out = np.empty((V, M, xq.shape[1]))
    for m in range(M):
        idx = np.nonzero(valid[m])[0]
        for v in range(V):
            out[v, m] = pchip_np(x[m, idx], y[v, m, idx], xq[m])
    return out


@_jit
def _pchip_at(x, y, d, xv):
    """
    PCHIP through (x, y) with slopes d (``_pchip_slopes``) at one point,
    linear beyond the ends; the same arithmetic as ``_pchip_query``.
    """
    K = x.size
    if xv <= x[0]:
        return y[0] + (xv - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if xv >= x[K - 1]:
        return y[K - 1] + (xv - x[K - 1]) * (y[K - 1] - y[K - 2]) / (
            x[K - 1] - x[K - 2]
        )
    k = np.searchsorted(x, xv, side="right") - 1
    h = x[k + 1] - x[k]
    t = (xv - x[k]) / h
    t2 = t * t
    t3 = t2 * t
    return (
        (2 * t3 - 3 * t2 + 1) * y[k]
        + (t3 - 2 * t2 + t) * h * d[k]
        + (-2 * t3 + 3 * t2) * y[k + 1]
        + (t3 - t2) * h * d[k + 1]
    )


@_jit
def _pchip_slopes_rows(x, y):
    """PCHIP slopes of every row of (x, y), all points used, (R, K)."""
    R, K = x.shape
    d = np.empty((R, K))
    for i in range(R):
        _pchip_slopes(x[i], y[i], d[i])
    return d


@_jit
def _monotone_valid(x, ok):
    """
    Knots to keep so that x is strictly increasing over kept entries:
    scan from the top and drop any point that is not below every point
    kept above it (the low end of an endogenous grid is the least
    reliable part).
    """
    M, K = x.shape
    keep = np.zeros((M, K), dtype=np.bool_)
    for m in range(M):
        upper = np.inf
        for k in range(K - 1, -1, -1):
            if ok[m, k] and x[m, k] < upper:
                keep[m, k] = True
                upper = x[m, k]
    return keep


def monotone_valid(x, ok):
    x = np.ascontiguousarray(x, dtype=float)
    ok = np.ascontiguousarray(ok, dtype=np.bool_)
    if USE_NUMBA:
        return _monotone_valid(x, ok)
    keep = np.zeros(x.shape, dtype=bool)
    for m in range(x.shape[0]):
        upper = np.inf
        for k in range(x.shape[1] - 1, -1, -1):
            if ok[m, k] and x[m, k] < upper:
                keep[m, k] = True
                upper = x[m, k]
    return keep


"""
------------------------------------------------------------------------
    Savings at grid nodes from the Euler equation directly
------------------------------------------------------------------------
"""

# relative tolerance on consumption when solving the Euler equation
# directly at a node
EULER_ROOT_TOL = 1e-11
# outcomes of the direct solve at a node: the Euler equation holds; the
# borrowing constraint binds (Euler inequality); savings at a jump of
# phi from + to - (with analytical MTRs: a kink of the value function,
# where the labor choice switches; the Euler equation holds as a pair
# of inequalities); failed
EULER_ROOT, EULER_AT_LIMIT, EULER_KINK, EULER_FAILED = 0, 1, 2, 3


@_jit
def _node_or_beyond(xr, k):
    # grid node k, or for k >= nb beyond the top of the grid (up to
    # twice the top)
    nb = xr.size
    if k < nb:
        return xr[k]
    return xr[nb - 1] * (1 + 0.25 * 2.0 ** (k - nb))


@_jit
def _euler_phi(
    bp,
    n0,
    n_floor,
    b,
    xr,
    yr,
    dr,
    wez,
    oth,
    r,
    p_tilde,
    g_y,
    sigma,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    # c(b, b') - c_E(b') and the labor supply at (b, b')
    n, c, st, res = _static_point(
        n0,
        n_floor,
        b,
        bp,
        0.0,
        wez,
        oth,
        r,
        p_tilde,
        g_y,
        sigma,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        chi_n,
        b_ellipse,
        upsilon,
        ltilde,
        code,
        analytical,
        etr_params,
        mtrx_params,
        tol,
        maxiter,
    )
    if st != CONVERGED and st != CORNER:
        return np.nan, n, c
    return c - _pchip_at(xr, yr, dr, bp), n, c


@_jit
def _euler_direct_point(
    b,
    g,
    n_guess,
    n_floor,
    xr,
    yr,
    dr,
    wez,
    oth,
    r,
    p_tilde,
    g_y,
    sigma,
    factor,
    tau_p,
    filer_i,
    lnc,
    knc,
    filer_w,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    Savings at one node with assets b: the root of phi(b') = c(b, b') -
    c_E(b'), c_E by PCHIP over the knots (xr, yr, slopes dr); see
    ``household_stoch._solve_euler_direct``. Returns b', n and the
    outcome (EULER_ROOT, EULER_AT_LIMIT, EULER_KINK or EULER_FAILED).

    A bracket that collapses where phi jumps from + to - is a kink of
    the value function only with analytical MTRs, where the
    within-period solve picks the best labor choice and a jump means
    that choice switches; then the end of the bracket with c > 0 is
    returned. Otherwise (OG-Core's household is defined by its FOCs,
    and jumps come from tax functions that are inconsistent at extreme
    incomes) it is a failure.
    """
    nb = xr.size
    g = min(max(g, xr[0]), xr[nb - 1])
    k_lo = min(max(np.searchsorted(xr, g, side="right") - 1, 0), nb - 2)
    k_hi = k_lo + 1
    x_lo = xr[k_lo]
    x_hi = xr[k_hi]
    f_lo, n_lo, c_lo = _euler_phi(
        x_lo,
        n_guess,
        n_floor,
        b,
        xr,
        yr,
        dr,
        wez,
        oth,
        r,
        p_tilde,
        g_y,
        sigma,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        chi_n,
        b_ellipse,
        upsilon,
        ltilde,
        code,
        analytical,
        etr_params,
        mtrx_params,
        tol,
        maxiter,
    )
    f_hi, n_hi, c_hi = _euler_phi(
        x_hi,
        n_lo,
        n_floor,
        b,
        xr,
        yr,
        dr,
        wez,
        oth,
        r,
        p_tilde,
        g_y,
        sigma,
        factor,
        tau_p,
        filer_i,
        lnc,
        knc,
        filer_w,
        h_w,
        m_w,
        p_w,
        chi_n,
        b_ellipse,
        upsilon,
        ltilde,
        code,
        analytical,
        etr_params,
        mtrx_params,
        tol,
        maxiter,
    )
    # bracket: phi(x_lo) > 0 >= phi(x_hi), stepping over grid nodes
    step = 1
    bracketed = False
    for it in range(4 * nb):
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return np.nan, n_guess, EULER_FAILED
        if f_lo > 0 and f_hi <= 0:
            bracketed = True
            break
        if not f_lo > 0:  # the root is below
            if k_lo == 0:  # the borrowing constraint binds
                return x_lo, n_lo, EULER_AT_LIMIT
            k_hi = k_lo
            x_hi = x_lo
            f_hi = f_lo
            n_hi = n_lo
            c_hi = c_lo
            k_lo = max(k_lo - step, 0)
            x_lo = xr[k_lo]
            f_lo, n_lo, c_lo = _euler_phi(
                x_lo,
                n_hi,
                n_floor,
                b,
                xr,
                yr,
                dr,
                wez,
                oth,
                r,
                p_tilde,
                g_y,
                sigma,
                factor,
                tau_p,
                filer_i,
                lnc,
                knc,
                filer_w,
                h_w,
                m_w,
                p_w,
                chi_n,
                b_ellipse,
                upsilon,
                ltilde,
                code,
                analytical,
                etr_params,
                mtrx_params,
                tol,
                maxiter,
            )
        else:  # the root is above
            if k_hi >= nb + 2:
                return np.nan, n_guess, EULER_FAILED
            k_lo = k_hi
            x_lo = x_hi
            f_lo = f_hi
            n_lo = n_hi
            c_lo = c_hi
            if k_lo >= nb - 1:
                k_hi = k_lo + 1
            else:
                k_hi = min(k_lo + step, nb - 1)
            x_hi = _node_or_beyond(xr, k_hi)
            f_hi, n_hi, c_hi = _euler_phi(
                x_hi,
                n_lo,
                n_floor,
                b,
                xr,
                yr,
                dr,
                wez,
                oth,
                r,
                p_tilde,
                g_y,
                sigma,
                factor,
                tau_p,
                filer_i,
                lnc,
                knc,
                filer_w,
                h_w,
                m_w,
                p_w,
                chi_n,
                b_ellipse,
                upsilon,
                ltilde,
                code,
                analytical,
                etr_params,
                mtrx_params,
                tol,
                maxiter,
            )
        step *= 2
    if not bracketed:
        return np.nan, n_guess, EULER_FAILED
    # regula falsi, Illinois variant
    a = x_lo
    fa = f_lo
    n_a = n_lo
    c_a = c_lo
    xb = x_hi
    fb = f_hi
    n_b = n_hi
    c_b = c_hi
    for it in range(200):
        size = abs(c_b)
        if abs(fb) <= EULER_ROOT_TOL * size:
            return xb, n_b, EULER_ROOT
        if abs(xb - a) <= 1e-14 * (1.0 + abs(a)):
            if abs(fb) <= 1e-8 * size:
                return xb, n_b, EULER_ROOT
            if analytical:
                # phi jumps from + to -: a kink of the value function
                if c_b > 0:
                    return xb, n_b, EULER_KINK
                if c_a > 0:
                    return a, n_a, EULER_KINK
            return np.nan, n_guess, EULER_FAILED
        x = xb - fb * (xb - a) / (fb - fa)
        if not (min(a, xb) < x < max(a, xb)):
            x = 0.5 * (a + xb)
        f, n_x, c_x = _euler_phi(
            x,
            n_b,
            n_floor,
            b,
            xr,
            yr,
            dr,
            wez,
            oth,
            r,
            p_tilde,
            g_y,
            sigma,
            factor,
            tau_p,
            filer_i,
            lnc,
            knc,
            filer_w,
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
            tol,
            maxiter,
        )
        if not np.isfinite(f):
            return np.nan, n_guess, EULER_FAILED
        if f * fb < 0:
            a = xb
            fa = fb
            n_a = n_b
            c_a = c_b
        else:
            fa = 0.5 * fa
        xb = x
        fb = f
        n_b = n_x
        c_b = c_x
    return np.nan, n_guess, EULER_FAILED


@_jit_parallel
def _euler_direct_solve(
    b,
    g,
    n_guess,
    rows,
    X,
    Y,
    D,
    wez,
    oth,
    filer_i,
    lnc,
    knc,
    filer_w,
    r,
    p_tilde,
    g_y,
    sigma,
    factor,
    tau_p,
    h_w,
    m_w,
    p_w,
    chi_n,
    b_ellipse,
    upsilon,
    ltilde,
    code,
    analytical,
    etr_params,
    mtrx_params,
    tol,
    maxiter,
):
    """
    ``_euler_direct_point`` at each point; point i uses knot row
    rows[i] of X (savings levels), Y (Euler consumption) and D (PCHIP
    slopes). Inputs indexed [i] (including tau_p) are per point.
    """
    P = b.size
    bp = np.empty(P)
    n = np.empty(P)
    status = np.empty(P, dtype=np.int64)
    n_floor = _labor_floor(b_ellipse, upsilon, ltilde)
    for i in prange(P):
        row = rows[i]
        out = _euler_direct_point(
            b[i],
            g[i],
            n_guess[i],
            n_floor,
            X[row],
            Y[row],
            D[row],
            wez[i],
            oth[i],
            r,
            p_tilde,
            g_y,
            sigma,
            factor,
            tau_p[i],
            filer_i[i],
            lnc[i],
            knc[i],
            filer_w[i],
            h_w,
            m_w,
            p_w,
            chi_n,
            b_ellipse,
            upsilon,
            ltilde,
            code,
            analytical,
            etr_params,
            mtrx_params,
            tol,
            maxiter,
        )
        bp[i] = out[0]
        n[i] = out[1]
        status[i] = out[2]
    return bp, n, status


"""
------------------------------------------------------------------------
    Distribution: Young's (2010) non-stochastic histogram method
------------------------------------------------------------------------
"""


@_jit
def _young_step(dist, b_next, grid, z_trans):
    """
    Move the distribution one age forward.

    Args:
        dist (3-D array): mass at (type, asset node, shock), (M, nb, nz)
        b_next (3-D array): savings chosen at each node, (M, nb, nz)
        grid (2-D array): asset grid of each type, (M, nb)
        z_trans (2-D array): shock transition matrix, (nz, nz)

    Returns:
        new (3-D array): unnormalized mass next age, (M, nb, nz)
        above (1-D array): mass whose savings exceed the grid top, (M,)
        below (1-D array): mass whose savings fall below the grid, (M,)
    """
    M, nb, nz = dist.shape
    new = np.zeros((M, nb, nz))
    above = np.zeros(M)
    below = np.zeros(M)
    for m in range(M):
        g = grid[m]
        for i in range(nb):
            for iz in range(nz):
                mass = dist[m, i, iz]
                if mass == 0.0:
                    continue
                x = b_next[m, i, iz]
                if not np.isfinite(x):
                    raise ValueError("savings policy is not finite")
                if x >= g[nb - 1]:
                    k = nb - 2
                    wt = 1.0
                    if x > g[nb - 1]:
                        above[m] += mass
                elif x <= g[0]:
                    k = 0
                    wt = 0.0
                    if x < g[0]:
                        below[m] += mass
                else:
                    k = np.searchsorted(g, x, side="right") - 1
                    wt = (x - g[k]) / (g[k + 1] - g[k])
                for iz2 in range(nz):
                    q = mass * z_trans[iz, iz2]
                    new[m, k, iz2] += q * (1.0 - wt)
                    new[m, k + 1, iz2] += q * wt
    return new, above, below


def young_step_np(dist, b_next, grid, z_trans):
    """numpy fallback for ``_young_step``."""
    M, nb, nz = dist.shape
    new = np.zeros((M, nb, nz))
    above = np.zeros(M)
    below = np.zeros(M)
    for m in range(M):
        g = grid[m]
        # non-finite savings only where there is no mass (young_step
        # checks); keep them out of the arithmetic
        x = np.where(np.isfinite(b_next[m]), b_next[m], g[0])
        above[m] = dist[m][x > g[-1]].sum()
        below[m] = dist[m][x < g[0]].sum()
        xc = np.clip(x, g[0], g[-1])
        k = np.clip(np.searchsorted(g, xc, side="right") - 1, 0, nb - 2)
        wt = (xc - g[k]) / (g[k + 1] - g[k])
        lo = dist[m] * (1 - wt)  # (nb, nz) mass going to node k
        hi = dist[m] * wt  # mass going to node k + 1
        for iz in range(nz):
            for iz2 in range(nz):
                prob = z_trans[iz, iz2]
                np.add.at(new[m, :, iz2], k[:, iz], lo[:, iz] * prob)
                np.add.at(new[m, :, iz2], k[:, iz] + 1, hi[:, iz] * prob)
    return new, above, below


def young_step(dist, b_next, grid, z_trans):
    """
    One step of Young's method; see ``_young_step``.

    Raises:
        ValueError: if the savings policy is NaN or infinite where the
            distribution has mass
    """
    args = [
        np.ascontiguousarray(a, dtype=float)
        for a in (dist, b_next, grid, z_trans)
    ]
    if not np.isfinite(args[1][args[0] != 0]).all():
        raise ValueError(
            "young_step: the savings policy is NaN or infinite where the "
            "distribution has mass"
        )
    if USE_NUMBA:
        return _young_step(*args)
    return young_step_np(*args)
