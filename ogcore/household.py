"""
------------------------------------------------------------------------
Household functions.
------------------------------------------------------------------------
"""

# Packages
import numpy as np
from ogcore import tax, utils

"""
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
"""


def marg_ut_cons(c, sigma):
    r"""
    Compute the marginal utility of consumption.

    .. math::
        MU_{c} = c^{-\sigma}

    Args:
        c (array_like): household consumption
        sigma (scalar): coefficient of relative risk aversion

    Returns:
        output (array_like): marginal utility of consumption

    """
    if np.ndim(c) == 0:
        c = np.array([c])
    epsilon = 0.003
    cvec_cnstr = c < epsilon
    MU_c = np.zeros(c.shape)
    MU_c[~cvec_cnstr] = c[~cvec_cnstr] ** (-sigma)
    b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
    b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
    MU_c[cvec_cnstr] = 2 * b2 * c[cvec_cnstr] + b1
    output = MU_c
    output = np.squeeze(output)

    return output


def marg_ut_labor(n, chi_n, p):
    r"""
    Compute the marginal disutility of labor.

    .. math::
        MDU_{l} = \chi^n_{s}\biggl(\frac{b}{\tilde{l}}\biggr)
        \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^{\upsilon-1}
        \Biggl[1-\biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon
        \Biggr]^{\frac{1-\upsilon}{\upsilon}}

    Args:
        n (array_like): household labor supply
        chi_n (array_like): utility weights on disutility of labor
        p (OG-Core Specifications object): model parameters

    Returns:
        output (array_like): marginal disutility of labor supply

    """
    nvec = n
    if np.ndim(nvec) == 0:
        nvec = np.array([nvec])
    eps_low = 0.000001
    eps_high = p.ltilde - 0.000001
    nvec_low = nvec < eps_low
    nvec_high = nvec > eps_high
    nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
    MDU_n = np.zeros(nvec.shape)
    MDU_n[nvec_uncstr] = (
        (p.b_ellipse / p.ltilde)
        * ((nvec[nvec_uncstr] / p.ltilde) ** (p.upsilon - 1))
        * (
            (1 - ((nvec[nvec_uncstr] / p.ltilde) ** p.upsilon))
            ** ((1 - p.upsilon) / p.upsilon)
        )
    )
    b2 = (
        0.5
        * p.b_ellipse
        * (p.ltilde ** (-p.upsilon))
        * (p.upsilon - 1)
        * (eps_low ** (p.upsilon - 2))
        * (
            (1 - ((eps_low / p.ltilde) ** p.upsilon))
            ** ((1 - p.upsilon) / p.upsilon)
        )
        * (
            1
            + ((eps_low / p.ltilde) ** p.upsilon)
            * ((1 - ((eps_low / p.ltilde) ** p.upsilon)) ** (-1))
        )
    )
    b1 = (p.b_ellipse / p.ltilde) * (
        (eps_low / p.ltilde) ** (p.upsilon - 1)
    ) * (
        (1 - ((eps_low / p.ltilde) ** p.upsilon))
        ** ((1 - p.upsilon) / p.upsilon)
    ) - (
        2 * b2 * eps_low
    )
    MDU_n[nvec_low] = 2 * b2 * nvec[nvec_low] + b1
    d2 = (
        0.5
        * p.b_ellipse
        * (p.ltilde ** (-p.upsilon))
        * (p.upsilon - 1)
        * (eps_high ** (p.upsilon - 2))
        * (
            (1 - ((eps_high / p.ltilde) ** p.upsilon))
            ** ((1 - p.upsilon) / p.upsilon)
        )
        * (
            1
            + ((eps_high / p.ltilde) ** p.upsilon)
            * ((1 - ((eps_high / p.ltilde) ** p.upsilon)) ** (-1))
        )
    )
    d1 = (p.b_ellipse / p.ltilde) * (
        (eps_high / p.ltilde) ** (p.upsilon - 1)
    ) * (
        (1 - ((eps_high / p.ltilde) ** p.upsilon))
        ** ((1 - p.upsilon) / p.upsilon)
    ) - (
        2 * d2 * eps_high
    )
    MDU_n[nvec_high] = 2 * d2 * nvec[nvec_high] + d1
    output = MDU_n * np.squeeze(chi_n)
    output = np.squeeze(output)
    return output


def get_bq(BQ, j, p, method):
    r"""
    Calculate bequests to each household.

    .. math::
        bq_{j,s,t} = \zeta_{j,s}\frac{BQ_{t}}{\lambda_{j}\omega_{s,t}}

    Args:
        BQ (array_like): aggregate bequests
        j (int): index of lifetime ability group
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        bq (array_like): bequests received by each household

    """
    if p.use_zeta:
        if j is not None:
            if method == "SS":
                bq = (p.zeta[:, j] * BQ) / (p.lambdas[j] * p.omega_SS)
            else:
                len_T = BQ.shape[0]
                bq = (
                    np.reshape(p.zeta[:, j], (1, p.S)) * BQ.reshape((len_T, 1))
                ) / (p.lambdas[j] * p.omega[:len_T, :])
        else:
            if method == "SS":
                bq = (p.zeta * BQ) / (
                    p.lambdas.reshape((1, p.J)) * p.omega_SS.reshape((p.S, 1))
                )
            else:
                len_T = BQ.shape[0]
                bq = (
                    np.reshape(p.zeta, (1, p.S, p.J))
                    * utils.to_timepath_shape(BQ)
                ) / (
                    p.lambdas.reshape((1, 1, p.J))
                    * p.omega[:len_T, :].reshape((len_T, p.S, 1))
                )
    else:
        if j is not None:
            if method == "SS":
                bq = np.tile(BQ[j], p.S) / p.lambdas[j]
            if method == "TPI":
                len_T = BQ.shape[0]
                bq = np.tile(
                    np.reshape(BQ[:, j] / p.lambdas[j], (len_T, 1)), (1, p.S)
                )
        else:
            if method == "SS":
                BQ_per = BQ / np.squeeze(p.lambdas)
                bq = np.tile(np.reshape(BQ_per, (1, p.J)), (p.S, 1))
            if method == "TPI":
                len_T = BQ.shape[0]
                BQ_per = BQ / p.lambdas.reshape(1, p.J)
                bq = np.tile(np.reshape(BQ_per, (len_T, 1, p.J)), (1, p.S, 1))
    return bq


def get_tr(TR, j, p, method):
    r"""
    Calculate transfers to each household.

    .. math::
        tr_{j,s,t} = \zeta_{j,s}\frac{TR_{t}}{\lambda_{j}\omega_{s,t}}

    Args:
        TR (array_like): aggregate transfers
        j (int): index of lifetime ability group
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        tr (array_like): transfers received by each household

    """
    if j is not None:
        if method == "SS":
            tr = (p.eta[-1, :, j] * TR) / (p.lambdas[j] * p.omega_SS)
        else:
            len_T = TR.shape[0]
            tr = (p.eta[:len_T, :, j] * TR.reshape((len_T, 1))) / (
                p.lambdas[j] * p.omega[:len_T, :]
            )
    else:
        if method == "SS":
            tr = (p.eta[-1, :, :] * TR) / (
                p.lambdas.reshape((1, p.J)) * p.omega_SS.reshape((p.S, 1))
            )
        else:
            len_T = TR.shape[0]
            tr = (p.eta[:len_T, :, :] * utils.to_timepath_shape(TR)) / (
                p.lambdas.reshape((1, 1, p.J))
                * p.omega[:len_T, :].reshape((len_T, p.S, 1))
            )

    return tr


def get_cons(r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, p):
    r"""
    Calculate household consumption.

    .. math::
        c_{j,s,t} =  \frac{(1 + r_{t})b_{j,s,t} + w_t e_{j,s} n_{j,s,t}
        + bq_{j,s,t} + tr_{j,s,t} - T_{j,s,t} -
        e^{g_y}b_{j,s+1,t+1}}{1 - \tau^{c}_{s,t}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): the ratio of real GDP to nominal GDP
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        net_tax (Numpy array): household net taxes paid
        e (Numpy array): effective labor units
        p (OG-Core Specifications object): model parameters

    Returns:
        cons (Numpy array): household consumption

    """
    cons = (
        (1 + r) * b + w * e * n + bq - b_splus1 * np.exp(p.g_y) - net_tax
    ) / p_tilde
    return cons


def get_ci(c_s, p_i, p_tilde, tau_c, alpha_c, method="SS"):
    r"""
    Compute consumption of good i given amount of composite consumption
    and prices.

    .. math::
        c_{i,j,s,t} = \frac{c_{s,j,t}}{\alpha_{i,j}p_{i,j}}

    Args:
        c_s (array_like): composite consumption
        p_i (array_like): prices for consumption good i
        p_tilde (array_like): composite good price
        tau_c (array_like): consumption tax rate
        alpha_c (array_like): consumption share parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        c_si (array_like): consumption of good i
    """
    if method == "SS":
        I = alpha_c.shape[0]
        S = c_s.shape[0]
        J = c_s.shape[1]
        tau_c = tau_c.reshape(I, 1, 1)
        alpha_c = alpha_c.reshape(I, 1, 1)
        p_tilde.reshape(1, 1, 1)
        p_i = p_i.reshape(I, 1, 1)
        c_s = c_s.reshape(1, S, J)
        c_si = alpha_c * (((1 + tau_c) * p_i) / p_tilde) ** (-1) * c_s
    else:  # Time path case
        I = alpha_c.shape[0]
        T = p_i.shape[0]
        S = c_s.shape[1]
        J = c_s.shape[2]
        tau_c = tau_c.reshape(T, I, 1, 1)
        alpha_c = alpha_c.reshape(1, I, 1, 1)
        p_tilde = p_tilde.reshape(T, 1, 1, 1)
        p_i = p_i.reshape(T, I, 1, 1)
        c_s = c_s.reshape(T, 1, S, J)
        c_si = alpha_c * (((1 + tau_c) * p_i) / p_tilde) ** (-1) * c_s
    return c_si


def FOC_savings(
    r,
    w,
    p_tilde,
    b,
    b_splus1,
    n,
    bq,
    factor,
    tr,
    ubi,
    theta,
    rho,
    etr_params,
    mtry_params,
    t,
    j,
    p,
    method,
):
    r"""
    Computes Euler errors for the FOC for savings in the steady state.
    This function is usually looped through over J, so it does one
    lifetime income group at a time.

    .. math::
        \frac{c_{j,s,t}^{-\sigma}}{\tilde{p}_{t}} = e^{-\sigma g_y}
        \biggl[\chi^b_j\rho_s(b_{j,s+1,t+1})^{-\sigma} +
        \beta_j\bigl(1 - \rho_s\bigr)\Bigl(\frac{1 + r_{t+1}
        \bigl[1 - \tau^{mtry}_{s+1,t+1}\bigr]}{\tilde{p}_{t+1}}\Bigr)
        (c_{j,s+1,t+1})^{-\sigma}\biggr]

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        b_splus2 (Numpy array): household savings two periods ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        factor (scalar): scaling factor converting model units to dollars
        tr (Numpy array): government transfers to household
        ubi (Numpy array): universal basic income payment
        theta (Numpy array): social security replacement rate for each
            lifetime income group
        rho (Numpy array): mortality rates
        etr_params (list): parameters of the effective tax rate
            functions
        mtry_params (list): parameters of the marginal tax rate
            on capital income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        euler (Numpy array): Euler error from FOC for savings

    """
    if j is not None:
        chi_b = p.chi_b[j]
        beta = p.beta[j]
        if method == "SS":
            tax_noncompliance = p.capital_income_tax_noncompliance_rate[-1, j]
            e = np.squeeze(p.e[-1, :, j])
        elif method == "TPI_scalar":
            tax_noncompliance = p.capital_income_tax_noncompliance_rate[0, j]
            e = np.squeeze(p.e[0, :, j])
        else:
            length = r.shape[0]
            tax_noncompliance = p.capital_income_tax_noncompliance_rate[
                t : t + length, j
            ]
            e_long = np.concatenate(
                (
                    p.e,
                    np.tile(p.e[-1, :, :].reshape(1, p.S, p.J), (p.S, 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + p.S, :, j], max(p.S - length, 0))
    else:
        chi_b = p.chi_b
        beta = p.beta
        if method == "SS":
            tax_noncompliance = p.capital_income_tax_noncompliance_rate[-1, :]
            e = np.squeeze(p.e[-1, :, :])
        elif method == "TPI_scalar":
            tax_noncompliance = p.capital_income_tax_noncompliance_rate[0, :]
            e = np.squeeze(p.e[0, :, :])
        else:
            length = r.shape[0]
            tax_noncompliance = p.capital_income_tax_noncompliance_rate[
                t : t + length, :
            ]
            e_long = np.concatenate(
                (
                    p.e,
                    np.tile(p.e[-1, :, :].reshape(1, p.S, p.J), (p.S, 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + p.S, :, :], max(p.S - length, 0))
    e = np.squeeze(e)
    if method == "SS":
        h_wealth = p.h_wealth[-1]
        m_wealth = p.m_wealth[-1]
        p_wealth = p.p_wealth[-1]
        p_tilde = np.ones_like(p.rho[-1, :]) * p_tilde
    elif method == "TPI_scalar":
        h_wealth = p.h_wealth[0]
        m_wealth = p.m_wealth[0]
        p_wealth = p.p_wealth[0]
    else:
        h_wealth = p.h_wealth[t]
        m_wealth = p.m_wealth[t]
        p_wealth = p.p_wealth[t]
    taxes = tax.net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        theta,
        t,
        j,
        False,
        method,
        e,
        etr_params,
        p,
    )
    cons = get_cons(r, w, p_tilde, b, b_splus1, n, bq, taxes, e, p)
    deriv = (
        (1 + r)
        - (
            r
            * tax.MTR_income(
                r,
                w,
                b,
                n,
                factor,
                True,
                e,
                etr_params,
                mtry_params,
                tax_noncompliance,
                p,
            )
        )
        - tax.MTR_wealth(b, h_wealth, m_wealth, p_wealth)
    )
    savings_ut = (
        rho * np.exp(-p.sigma * p.g_y) * chi_b * b_splus1 ** (-p.sigma)
    )
    euler_error = np.zeros_like(n)
    if n.shape[0] > 1:
        euler_error[:-1] = (
            marg_ut_cons(cons[:-1], p.sigma) * (1 / p_tilde[:-1])
            - beta
            * (1 - rho[:-1])
            * deriv[1:]
            * marg_ut_cons(cons[1:], p.sigma)
            * (1 / p_tilde[1:])
            * np.exp(-p.sigma * p.g_y)
            - savings_ut[:-1]
        )
        euler_error[-1] = (
            marg_ut_cons(cons[-1], p.sigma) * (1 / p_tilde[-1])
            - savings_ut[-1]
        )
    else:
        euler_error[-1] = (
            marg_ut_cons(cons[-1], p.sigma) * (1 / p_tilde[-1])
            - savings_ut[-1]
        )

    return euler_error


def FOC_labor(
    r,
    w,
    p_tilde,
    b,
    b_splus1,
    n,
    bq,
    factor,
    tr,
    ubi,
    theta,
    chi_n,
    etr_params,
    mtrx_params,
    t,
    j,
    p,
    method,
):
    r"""
    Computes errors for the FOC for labor supply in the steady
    state.  This function is usually looped through over J, so it does
    one lifetime income group at a time.

    .. math::
        w_t e_{j,s}\bigl(1 - \tau^{mtrx}_{s,t}\bigr)
       \frac{(c_{j,s,t})^{-\sigma}}{ \tilde{p}_{t}} = \chi^n_{s}
        \biggl(\frac{b}{\tilde{l}}\biggr)\biggl(\frac{n_{j,s,t}}
        {\tilde{l}}\biggr)^{\upsilon-1}\Biggl[1 -
        \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon\Biggr]
        ^{\frac{1-\upsilon}{\upsilon}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        factor (scalar): scaling factor converting model units to dollars
        tr (Numpy array): government transfers to household
        ubi (Numpy array): universal basic income payment
        theta (Numpy array): social security replacement rate for each
            lifetime income group
        chi_n (Numpy array): utility weight on the disutility of labor
            supply
        e (Numpy array): effective labor units
        etr_params (list): parameters of the effective tax rate
            functions
        mtrx_params (list): parameters of the marginal tax rate
            on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        FOC_error (Numpy array): error from FOC for labor supply

    """
    if method == "SS":
        tau_payroll = p.tau_payroll[-1]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = p.tau_payroll[0]
    else:
        length = r.shape[0]
        tau_payroll = p.tau_payroll[t : t + length]
    if j is not None:
        if method == "SS":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, j]
            e = np.squeeze(p.e[-1, :, j])
        elif method == "TPI_scalar":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, j]
            e = np.squeeze(p.e[0, -1, j])
        else:
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[
                t : t + length, j
            ]
            e_long = np.concatenate(
                (
                    p.e,
                    np.tile(p.e[-1, :, :].reshape(1, p.S, p.J), (p.S, 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + p.S, :, j], max(p.S - length, 0))
    else:
        if method == "SS":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, :]
            e = np.squeeze(p.e[-1, :, :])
        elif method == "TPI_scalar":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, :]
            e = np.squeeze(p.e[0, -1, :])
        else:
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[
                t : t + length, :
            ]
            e_long = np.concatenate(
                (
                    p.e,
                    np.tile(p.e[-1, :, :].reshape(1, p.S, p.J), (p.S, 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + p.S, :, j], max(p.S - length, 0))
    if method == "SS":
        tau_payroll = p.tau_payroll[-1]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = p.tau_payroll[0]
    else:
        length = r.shape[0]
        tau_payroll = p.tau_payroll[t : t + length]
    if method == "TPI":
        if b.ndim == 2:
            r = r.reshape(r.shape[0], 1)
            w = w.reshape(w.shape[0], 1)
            tau_payroll = tau_payroll.reshape(tau_payroll.shape[0], 1)

    taxes = tax.net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        theta,
        t,
        j,
        False,
        method,
        e,
        etr_params,
        p,
    )
    cons = get_cons(r, w, p_tilde, b, b_splus1, n, bq, taxes, e, p)
    deriv = (
        1
        - tau_payroll
        - tax.MTR_income(
            r,
            w,
            b,
            n,
            factor,
            False,
            e,
            etr_params,
            mtrx_params,
            tax_noncompliance,
            p,
        )
    )
    FOC_error = marg_ut_cons(cons, p.sigma) * (
        1 / p_tilde
    ) * w * deriv * e - marg_ut_labor(n, chi_n, p)

    return FOC_error


def get_y(r_p, w, b_s, n, p, method):
    r"""
    Compute household income before taxes.

    .. math::
        y_{j,s,t} = r_{p,t}b_{j,s,t} + w_{t}e_{j,s}n_{j,s,t}

    Args:
        r_p (array_like): real interest rate on the household portfolio
        w (array_like): real wage rate
        b_s (Numpy array): household savings coming into the period
        n (Numpy array): household labor supply
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
    """
    if method == "SS":
        e = np.squeeze(p.e[-1, :, :])
    elif method == "TPI":
        e = p.e
    y = r_p * b_s + w * e * n

    return y


def constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    """
    Checks constraints on consumption, savings, and labor supply in the
    steady state.

    Args:
        bssmat (Numpy array): steady state distribution of capital
        nssmat (Numpy array): steady state distribution of labor
        cssmat (Numpy array): steady state distribution of consumption
        ltilde (scalar): upper bound of household labor supply

    Returns:
        None

    Raises:
        Warnings: if constraints are violated, warnings printed

    """
    print("Checking constraints on capital, labor, and consumption.")

    if (bssmat < 0).any():
        print("\tWARNING: There is negative capital stock")
    flag2 = False
    if (nssmat < 0).any():
        print(
            "\tWARNING: Labor supply violates nonnegativity ", "constraints."
        )
        flag2 = True
    if (nssmat > ltilde).any():
        print("\tWARNING: Labor supply violates the ltilde constraint.")
        flag2 = True
    if flag2 is False:
        print(
            "\tThere were no violations of the constraints on labor",
            " supply.",
        )
    if (cssmat < 0).any():
        print("\tWARNING: Consumption violates nonnegativity", " constraints.")
    else:
        print(
            "\tThere were no violations of the constraints on", " consumption."
        )


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, ltilde):
    """
    Checks constraints on consumption, savings, and labor supply along
    the transition path. Does this for each period t separately.

    Args:
        b_dist (Numpy array): distribution of capital at time t
        n_dist (Numpy array): distribution of labor at time t
        c_dist (Numpy array): distribution of consumption at time t
        t (int): time period
        ltilde (scalar): upper bound of household labor supply

    Returns:
        None

    Raises:
        Warnings: if constraints are violated, warnings printed

    """
    if (b_dist <= 0).any():
        print(
            "\tWARNING: Aggregate capital is less than or equal to ",
            "zero in period %.f." % t,
        )
    if (n_dist < 0).any():
        print(
            "\tWARNING: Labor supply violates nonnegativity",
            " constraints in period %.f." % t,
        )
    if (n_dist > ltilde).any():
        print(
            "\tWARNING: Labor suppy violates the ltilde constraint",
            " in period %.f." % t,
        )
    if (c_dist < 0).any():
        print(
            "\tWARNING: Consumption violates nonnegativity",
            " constraints in period %.f." % t,
        )
