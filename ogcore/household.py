"""
------------------------------------------------------------------------
Household functions.
------------------------------------------------------------------------
"""

# Packages
import numpy as np
import scipy.optimize as opt
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

def marg_ut_beq(b, sigma, j, p):
    r"""
    Compute the marginal utility of savings.

    .. math::
        MU_{b} = \chi^b_{j}b_{j,s,t}^{-\sigma}

    Args:
        b (array_like): household savings
        chi_b (array_like): utility weights on savings
        p (OG-Core Specifications object): model parameters

    Returns:
        output (array_like): marginal utility of savings

    """
    if np.ndim(b) == 0:
        b = np.array([b])
    epsilon = 0.0001
    bvec_cnstr = b < epsilon
    MU_b = np.zeros(b.shape)
    MU_b[~bvec_cnstr] = p.chi_b[j] * b[~bvec_cnstr] ** (-sigma)
    b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
    b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
    MU_b[bvec_cnstr] = 2 * b2 * b[bvec_cnstr] + b1
    output = MU_b
    output = np.squeeze(output)
    return output

def inv_mu_c(value, sigma):
    r"""
    Compute the inverse of the marginal utility of consumption.

    .. math::
        c = \left(\frac{1}{val}\right)^{-1/\sigma}

    Args:
        value (array_like): marginal utility of consumption
        sigma (scalar): coefficient of relative risk aversion

    Returns:
        output (array_like): household consumption

    """
    if np.ndim(value) == 0:
        value = np.array([value])
    output = value ** (-1 / sigma) # need value > 0
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


def c_from_n(
n, 
b, 
p_tilde, 
r, 
w, 
factor, 
e, 
z, 
chi_n, 
etr_params, 
mtrx_params, 
t, 
j, 
p, 
method
):
    r"""
    Calculate household consumption from labor supply Euler equation for group j.
    
    .. math::
        c_{j,s,t} = \left[ \frac{p_t e^{g_y(1-\sigma)}\chi_s^n h'(n_{j,s,t})}{
        w_t e_{j, s}z_{j, s}(1- \tau^{mtrx}_{s,t})} \right]^{-1/\sigma}

    Args: 
        n (array_like): household labor supply
        b (array_like): household savings
        p_tilde (array_like): composite good price
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        factor (scalar): scaling factor converting model units to dollars
        e (array_like): effective labor units (deterministic)
        z (array_like): productivity (stochastic)
        chi_n (array_like): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate
            functions
        mtrx_params (list): parameters of the marginal tax rate
            on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
    
    Returns: 
        c (array_like): consumption implied by labor choice
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
            e*z,
            etr_params,
            mtrx_params,
            tax_noncompliance,
            p,
        )
    )
    numerator = p_tilde * np.exp(p.g_y * (1-p.sigma)) * marg_ut_labor(n, chi_n, p)
    denominator = w * e * z * deriv
    c = inv_mu_c(numerator / denominator, p.sigma)

    return c


def b_from_c_EOL(c, p_tilde, j, sigma, p):
    r"""
    Calculate household bequests at the end of life from the savings Euler equation.

    .. math::
        b_{j, E+S+1, t+1} = [\chi_j^b \tilde p_t]^{\frac{1}{\sigma}} * c_{j, E+S, t}

    Args:
        c (array_like): household consumption
        p_tilde (array_like): composite good price
        j (int): index of ability type
        sigma (scalar): coefficient of relative risk aversion
        p (OG-Core Specifications object): model parameters

    Returns:
        b (array_like): household savings at the end of life
    """
    b = c * (p.chi_b[j] * p_tilde) ** (1 / sigma)
    return b


def get_cons(r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, p):
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
        z (array_like): labor productivity
        p (OG-Core Specifications object): model parameters

    Returns:
        cons (Numpy array): household consumption

    """
    cons = (
        (1 + r) * b + w * e * z * n + bq - b_splus1 * np.exp(p.g_y) - net_tax
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


def c_from_b_splus1(
    r_splus1,
    w_splus1,
    p_tilde_splus1,
    p_tilde,
    b_splus1,
    n_splus1_policy,
    c_splus1_policy,
    factor,
    rho,
    etr_params,
    mtry_params,
    j,
    t,
    e_splus1,
    z_index,
    p,
    method
):
    r"""
    Calculate household consumption in period s from assets at period s+1 using 
    the savings Euler equation.

    .. math::
        c_{j,s,t} = (\tilde{p}_t)^{-\frac{1}{\sigma}} e^{g_y} 
        \biggl[\chi^b_j\rho_s(b_{j,s+1,t+1})^{-\sigma} +
        \beta_j\bigl(1 - \rho_s\bigr)\Bigl(\frac{1 + r_{t+1}
        \bigl[1 - \tau^{mtry}_{s+1,t+1}\bigr]}{\tilde{p}_{t+1}}\Bigr)
        \mathbb{E}[(c_{j,s+1,t+1})^{-\sigma}]\biggr]^{-\frac{1}{\sigma}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        b_splus1 (array_like): household savings one period ahead
        n_splus1_policy (array_like): household labor supply one period ahead across b, z
        c_splus1_policy (array_like): household consumption one period ahead across b, z
        factor (scalar): scaling factor converting model units to dollars
        rho (array_like): mortality rates
        etr_params (list): parameters of the effective tax rate
            functions
        mtry_params (list): parameters of the marginal tax rate
            on capital income functions
        j (int): index of ability type
        t (int): model period
        e_splus1 (array_like): effective labor units one period ahead
        z_index (array_like): index in productivity grid
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
    
    returns:
        c (array_like): household consumption in current period
    """
    beta = p.beta[j]
    if method == "SS":
        tax_noncompliance = p.capital_income_tax_noncompliance_rate[-1, j]
        h_wealth = p.h_wealth[-1]
        m_wealth = p.m_wealth[-1]
        p_wealth = p.p_wealth[-1]
    elif method == "TPI_scalar":
        tax_noncompliance = p.capital_income_tax_noncompliance_rate[0, j]
        h_wealth = p.h_wealth[0]
        m_wealth = p.m_wealth[0]
        p_wealth = p.p_wealth[0]
    else:
        tax_noncompliance = p.capital_income_tax_noncompliance_rate[t, j]
        h_wealth = p.h_wealth[t]
        m_wealth = p.m_wealth[t]
        p_wealth = p.p_wealth[t]
    bequest_utility = rho * marg_ut_beq(b_splus1, p.sigma, j, p)
    consumption_utility = np.zeros_like(b_splus1) # length nb vector
    for (zp_index, zp) in enumerate(p.z_grid):
        deriv = (
            (1 + r_splus1)
            - (
                r_splus1
                * tax.MTR_income(
                    r_splus1,
                    w_splus1,
                    b_splus1,
                    n_splus1_policy[:, zp_index],
                    factor,
                    True,
                    e_splus1*zp,
                    etr_params,
                    mtry_params,
                    tax_noncompliance,
                    p,
                )
            )
            - tax.MTR_wealth(b_splus1, h_wealth, m_wealth, p_wealth)
        )
        consumption_utility += deriv * marg_ut_cons(c_splus1_policy[:, zp_index], p.sigma) / p_tilde_splus1
    prob_z_splus1 = p.Z[z_index, :] # markov matrix for z transition - need to add to parameters
    E_MU_c = consumption_utility @ prob_z_splus1
    c = inv_mu_c(
        (p_tilde * np.exp(-p.sigma * p.g_y) * (bequest_utility + beta * (1 - rho) * E_MU_c)),
        p.sigma
    )
    return c


def FOC_labor(
    r,
    w,
    p_tilde,
    b,
    c,
    n,
    factor,
    e,
    z,
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
        w_t z e_{j,s}\bigl(1 - \tau^{mtrx}_{s,t}\bigr)
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
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, j]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = p.tau_payroll[0]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, j]
    else:
        tau_payroll = p.tau_payroll[t]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[t, j]

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
    FOC_error = marg_ut_cons(c, p.sigma) * (
        1 / p_tilde
    ) * w * deriv * e * z - marg_ut_labor(n, chi_n, p)

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

def BC_residual(
    c, 
    n, 
    b, 
    b_splus1, 
    r, 
    w,
    p_tilde, 
    e, 
    z, 
    bq, 
    net_tax,
    p
):
    r"""
    Compute the residuals of the household budget constraint.
    
    .. math::
        c_{j,s,t} + b_{j,s+1,t+1} - (1 + r_{t})b_{j,s,t} = w_{t}e_{j,s}n_{j,s,t} + bq_{j,s,t} + tr_{j,s,t} - T_{j,s,t}
    """

    BC_error = (
        (1 + r) * b + w * e * z * n + bq - b_splus1 * np.exp(p.g_y) - net_tax
    ) - p_tilde * c
    return BC_error


def EOL_system(
        n, 
        b, 
        p_tilde, 
        r, 
        w, 
        tr,
        ubi,
        bq,
        theta,
        factor, 
        e, 
        z, 
        chi_n, 
        etr_params, 
        mtrx_params, 
        t, 
        j, 
        p, 
        method
):
    r"""
    Compute the residuals of the household budget constraint at the end of life given a 
    guess for labor supply. Solve first for consumption given labor supply and then for
    savings given consumption. Then check the budget constraint.
    
    Args:
        n (array_like): household labor supply
        b (array_like): household savings
        p_tilde (array_like): composite good price
        r (scalar): the real interest rate
        w (scalar): the real wage rate
        factor (scalar): scaling factor converting model units to dollars
        e (scalar): effective labor units
        z (scalar): productivity
        chi_n (scalar): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate functions
        mtrx_params (list): parameters of the marginal tax rate on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
        
        Returns:
            BC_error (array_like): residuals of the household budget constraint"""
    c = c_from_n(n, b, p_tilde, r, w, factor, e, z, chi_n, etr_params, mtrx_params, t, j, p, method)
    b_splus1 = b_from_c_EOL(c, p_tilde, j, p.sigma, p)
    net_tax = tax.net_taxes(r, w, b, n, bq, factor, tr, ubi, theta, t, j, False, method, e, etr_params, p)
    BC_error = BC_residual(c, n, b, b_splus1, r, w, p_tilde, e, z, bq, net_tax, p)
    return BC_error


def HH_system(x, 
              c, 
              b_splus1, 
              r, 
              w, 
              p_tilde, 
              factor, 
              tr, 
              ubi, 
              bq, 
              theta,
              e, 
              z, 
              chi_n, 
              etr_params, 
              mtrx_params, 
              j, 
              t, 
              p, 
              method):
    r"""
    Compute the residuals of the household budget constraint and labor supply Euler equation given a guess
    of household assets and labor choice. This is for use in a root finder to solve the household problem at
    age s < E+S.
    
    Args: 
        x (array_like): vector containing household assets b and labor supply n
        c (array_like): household consumption
        b_splus1 (array_like): household savings one period ahead
        r (scalar): the real interest rate
        w (scalar): the real wage rate
        p_tilde (scalar): composite good price
        factor (scalar): scaling factor converting model units to dollars
        e (scalar): effective labor units
        z (scalar): productivity
        chi_n (scalar): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate functions
        mtrx_params (list): parameters of the marginal tax rate on labor income functions
        j (int): index of ability type
        t (int): model period
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
        
        Returns:
            HH_error (array_like): residuals of the household budget constraint and labor supply Euler equation"""
    b = x[0]
    n = x[1]
    net_tax = tax.net_taxes(r, w, b, n, bq, factor, tr, ubi, theta, t, j, False, method, e, etr_params, p)
    BC_error = BC_residual(c, n, b, b_splus1, r, w, p_tilde, e, z, bq, net_tax, p)
    FOC_error = FOC_labor(r, w, p_tilde, b, c, n, factor, e, z, chi_n, etr_params, mtrx_params, t, j, p, method)
    HH_error = np.array([BC_error, FOC_error])
    return HH_error


def solve_HH(
    r,
    w,
    p_tilde,
    factor,
    tr,
    bq,
    ubi,
    b_grid,
    sigma,
    theta,
    chi_n,
    rho,
    e,
    etr_params,
    mtrx_params,
    mtry_params,
    j,
    t,
    p,
    method,
):
    # solve household problem on transition path using endogenous grid method
    nb = len(b_grid)


    if method == "SS":
        r = np.repeat(r, p.S)
        w = np.repeat(w, p.S)
        p_tilde = np.repeat(p_tilde, p.S)


    # initialize policy functions on grid
    b_policy = np.zeros((p.S, nb, p.nz))
    c_policy = np.zeros((p.S, nb, p.nz))
    n_policy = np.zeros((p.S, nb, p.nz))

    # start at the end of life 
    for z_index, z in enumerate(p.z_grid): # need to add z_grid to parameters
        for b_index, b in enumerate(b_grid):

            # use root finder to solve problem at end of life
            args = (b, 
                    p_tilde[-1], 
                    r[-1], 
                    w[-1], 
                    tr[-1], 
                    ubi, 
                    bq[-1], 
                    theta,
                    factor, 
                    e[-1], 
                    z, 
                    chi_n[-1], 
                    etr_params, 
                    mtrx_params, 
                    t[-1], 
                    j + p.S, 
                    p, 
                    method)
            n = opt.brentq(EOL_system, 
                           0.0, 
                           p.ltilde, 
                           args=args)
            n_policy[-1, b_index, z_index] = n
            c_policy[-1, b_index, z_index] = c_from_n(n, 
                                                      b, 
                                                      p_tilde[-1], 
                                                      r[-1], 
                                                      w[-1], 
                                                      factor, 
                                                      e[-1], 
                                                      z, 
                                                      chi_n[-1], 
                                                      etr_params, 
                                                      mtrx_params, 
                                                      t + p.S, 
                                                      j, 
                                                      p, 
                                                      method)
            b_policy[-1, b_index, z_index] = b_from_c_EOL(c_policy[-1, b_index, z_index], 
                                                          p_tilde[-1], 
                                                          j, 
                                                          p.sigma, 
                                                          p)
    
    # iterate backwards with Euler equation
    for s in range(p.S-2, -1, -1):
        for z_index, z in enumerate(p.z_grid):
            c = c_from_b_splus1(r[s+1], 
                                w[s+1],
                                p_tilde[s+1],
                                b_grid,
                                n_policy[s+1, :, z_index],
                                c_policy[s+1, :, z_index],
                                factor,
                                rho[s],
                                etr_params,
                                mtry_params,
                                j,
                                t[s+1],
                                e[s+1],
                                z_index,
                                p,
                                method)
            b = np.zeros_like(b_grid)
            n = np.zeros_like(b_grid)
            for b_splus1_index, b_splus1 in b_grid:

                args = (c[b_splus1_index], 
                    b_grid, 
                    r[s], 
                    w[s], 
                    p_tilde[s], 
                    factor, 
                    tr[s], 
                    ubi, 
                    bq[s], 
                    theta, 
                    e[s], 
                    z, 
                    chi_n[s], 
                    etr_params, 
                    mtrx_params, 
                    j, 
                    t[s], 
                    p, 
                    method)
                initial_guess = np.array([b_splus1, n_policy[s+1, b_splus1_index, z_index]])
                x = opt.root(HH_system, initial_guess, args=args)
                b[b_splus1_index] = x[0]
                n[b_splus1_index] = x[1]
            c_policy[s, :, z_index] = np.interp(b_grid, b, c)
            n_policy[s, :, z_index] = np.interp(b_grid, b, n)
            b_policy[s, :, z_index] = np.interp(b_grid, b, b_grid)

    return b_policy, c_policy, n_policy