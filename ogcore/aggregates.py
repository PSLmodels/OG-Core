"""
-------------------------------------------------------------------------------
Functions to compute economic aggregates.
-------------------------------------------------------------------------------
"""

# Packages
import numpy as np
from ogcore import tax, pensions

"""
-------------------------------------------------------------------------------
    Functions
-------------------------------------------------------------------------------
"""


def get_L(n, p, method):
    r"""
    Calculate aggregate labor supply.

    .. math::
        L_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}\lambda_{j}n_{j,s,t}

    Args:
        n (Numpy array): labor supply of households
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        L (array_like): aggregate labor supply

    """
    if method == "SS":
        L_presum = (
            np.squeeze(p.e[-1, :, :])
            * np.transpose(p.omega_SS * p.lambdas)
            * n
        )
        L = L_presum.sum()
    elif method == "TPI":
        L_presum = (n * (p.e * np.squeeze(p.lambdas))) * np.tile(
            np.reshape(p.omega[: p.T, :], (p.T, p.S, 1)), (1, 1, p.J)
        )
        L = L_presum.sum(1).sum(1)
    return L


def get_I(b_splus1, K_p1, K, p, method):
    r"""
    Calculate aggregate investment.

    .. math::
        I_{t} = (1 + g_{n,t+1})e^{g_{y}}(K_{t+1} - \sum_{s=E}^{E+S}
        \sum_{j=0}^{J}\omega_{s+1,t}i_{s+1,t}\lambda_{j}b_{j,s+1,t+1} \
        (1+ g_{n,t+1})) - (1 - \delta)K_{t}

    Args:
        b_splus1 (Numpy array): savings of households
        K_p1 (array_like): aggregate capital, one period ahead
        K (array_like): aggregate capital
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI', also compute total investment (not net of immigrants)

    Returns:
        aggI (array_like): aggregate investment

    """
    if method == "SS":
        omega_extended = np.append(p.omega_SS[1:], [0.0])
        imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
        part2 = (
            (
                b_splus1
                * np.transpose((omega_extended * imm_extended) * p.lambdas)
            ).sum()
        ) / (1 + p.g_n_ss)
        aggI = (1 + p.g_n_ss) * np.exp(p.g_y) * (K_p1 - part2) - (
            1.0 - p.delta
        ) * K
    elif method == "TPI":
        omega_shift = np.append(p.omega[: p.T, 1:], np.zeros((p.T, 1)), axis=1)
        imm_shift = np.append(
            p.imm_rates[: p.T, 1:], np.zeros((p.T, 1)), axis=1
        )
        part2 = (
            (
                (b_splus1 * np.squeeze(p.lambdas))
                * np.tile(
                    np.reshape(imm_shift * omega_shift, (p.T, p.S, 1)),
                    (1, 1, p.J),
                )
            )
            .sum(1)
            .sum(1)
        ) / (1 + np.squeeze(np.hstack((p.g_n[1 : p.T], p.g_n_ss))))
        aggI = (
            1 + np.squeeze(np.hstack((p.g_n[1 : p.T], p.g_n_ss)))
        ) * np.exp(p.g_y) * (K_p1 - part2) - (1.0 - p.delta) * K
    elif method == "total_ss":
        aggI = ((1 + p.g_n_ss) * np.exp(p.g_y) - 1 + p.delta) * K
    elif method == "total_tpi":
        aggI = (1 + p.g_n[1 : p.T + 1]) * np.exp(p.g_y) * K_p1 - (
            1.0 - p.delta
        ) * K

    return aggI


def get_B(b, p, method, preTP):
    r"""
    Calculate aggregate savings

    .. math::
        B_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}\lambda_{j}b_{j,s,t}

    Args:
        b (Numpy array): savings of households
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        preTP (bool): whether calculation is for the pre-time path
            period amount of savings.  If True, then need to use
            `omega_S_preTP`.

    Returns:
        B (array_like): aggregate supply of savings

    """
    if method == "SS":
        if preTP:
            part1 = b * np.transpose(p.omega_S_preTP * p.lambdas)
            omega_extended = np.append(p.omega_S_preTP[1:], [0.0])
            imm_extended = np.append(p.imm_rates[0, 1:], [0.0])
            pop_growth_rate = p.g_n[0]
        else:
            part1 = b * np.transpose(p.omega_SS * p.lambdas)
            omega_extended = np.append(p.omega_SS[1:], [0.0])
            imm_extended = np.append(p.imm_rates[-1, 1:], [0.0])
            pop_growth_rate = p.g_n_ss
        part2 = b * np.transpose(omega_extended * imm_extended * p.lambdas)
        B_presum = part1 + part2
        B = B_presum.sum()
        B /= 1.0 + pop_growth_rate
    elif method == "TPI":
        part1 = (b * np.squeeze(p.lambdas)) * np.tile(
            np.reshape(p.omega[: p.T, :], (p.T, p.S, 1)), (1, 1, p.J)
        )
        omega_shift = np.append(p.omega[: p.T, 1:], np.zeros((p.T, 1)), axis=1)
        imm_shift = np.append(
            p.imm_rates[: p.T, 1:], np.zeros((p.T, 1)), axis=1
        )
        part2 = (b * np.squeeze(p.lambdas)) * np.tile(
            np.reshape(imm_shift * omega_shift, (p.T, p.S, 1)), (1, 1, p.J)
        )
        B_presum = part1 + part2
        B = B_presum.sum(1).sum(1)
        B /= 1.0 + np.hstack((p.g_n[1 : p.T], p.g_n_ss))
    return B


def get_BQ(r, b_splus1, j, p, method, preTP):
    r"""
    Calculation of aggregate bequests.  If `use_zeta` is False, then
    computes aggregate bequests within each lifetime income group.

    .. math::
        BQ_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\rho_{s}\omega_{s,t}
        \lambda_{j}b_{j,s+1,1}

    Args:
        r (array_like): the real interest rate
        b_splus1 (numpy array): household savings one period ahead
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on SS or
            TPI
        preTP (bool): whether calculation is for the pre-time path
            period amount of savings.  If True, then need to use
            `omega_S_preTP`.

    Returns:
        BQ (array_like): aggregate bequests, overall or by lifetime
            income group, depending on `use_zeta` value.

    """
    if method == "SS":
        if preTP:
            omega = p.omega_S_preTP
            pop_growth_rate = p.g_n[0]
            rho = p.rho[0, :]
        else:
            omega = p.omega_SS
            pop_growth_rate = p.g_n_ss
            rho = p.rho[-1, :]
        if j is not None:
            BQ_presum = omega * rho * b_splus1 * p.lambdas[j]
        else:
            BQ_presum = np.transpose(omega * (rho * p.lambdas)) * b_splus1
        BQ = BQ_presum.sum(0)
        BQ *= (1.0 + r) / (1.0 + pop_growth_rate)
    elif method == "TPI":
        pop = np.append(
            p.omega_S_preTP.reshape(1, p.S), p.omega[: p.T - 1, :], axis=0
        )
        rho = np.append(
            p.rho[0, :].reshape(1, p.S), p.rho[: p.T - 1, :], axis=0
        )

        if j is not None:
            BQ_presum = (b_splus1 * p.lambdas[j]) * (pop * rho)
            BQ = BQ_presum.sum(1)
            BQ *= (1.0 + r) / (1.0 + p.g_n[: p.T])
        else:
            BQ_presum = (b_splus1 * np.squeeze(p.lambdas)) * np.tile(
                np.reshape(pop * rho, (p.T, p.S, 1)), (1, 1, p.J)
            )
            BQ = BQ_presum.sum(1)
            BQ *= np.tile(
                np.reshape((1.0 + r) / (1.0 + p.g_n[: p.T]), (p.T, 1)),
                (1, p.J),
            )
    if p.use_zeta:
        if method == "SS":
            BQ = BQ.sum()
        else:
            if not j:
                BQ = BQ.sum(1)
    return BQ


def get_C(c, p, method):
    r"""
    Calculation of aggregate consumption.

    Set up to only take one consumption good at a time. This
    function is called in a loop to get consumption for all goods.

    .. math::
        C_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}
        \lambda_{j}c_{j,s,t}

    Args:
        c (Numpy array): consumption of households
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        C (array_like): aggregate consumption

    """

    if method == "SS":
        aggC = (
            (c * np.transpose(p.omega_SS * p.lambdas).reshape(1, p.S, p.J))
            .sum(-1)
            .sum(-1)
        )
    elif method == "TPI":
        aggC = (
            (
                (c * np.squeeze(p.lambdas))
                * np.tile(
                    np.reshape(p.omega[: p.T, :], (p.T, p.S, 1)), (1, 1, p.J)
                )
            )
            .sum(-1)
            .sum(-1)
        )
    return aggC


def revenue(
    r,
    w,
    b,
    n,
    bq,
    c,
    Y,
    L,
    K,
    p_m,
    factor,
    ubi,
    theta,
    etr_params,
    e,
    p,
    m,
    method,
):
    r"""
    Calculate aggregate tax revenue.

    .. math::
        R_{t} = \sum_{s=E}^{E+S}\sum_{j=0}^{J}\omega_{s,t}\lambda_{j}
        (T_{j,s,t} + \tau^{p}_{t}w_{t}e_{j,s}n_{j,s,t} - \theta_{j}
        w_{t} + \tau^{bq}bq_{j,s,t} + \tau^{c}_{s,t}c_{j,s,t} +
        \tau^{w}_{t}b_{j,s,t}) +
        \sum_{m=1}^{M}\tau^{b}_{m,t}(Y_{m,t}-w_{t}L_{m,t}) -
        \tau^{b}_{m,t}\delta^{\tau}_{m,t}K^{\tau}_{m,t}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        b (Numpy array): household savings
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        c (Numpy array): household consumption
        Y (array_like): aggregate output
        L (array_like): aggregate labor
        K (array_like): aggregate capital
        p_m (array_like): output prices
        factor (scalar): scaling factor converting model units to
            dollars
        ubi (array_like): universal basic income household distributions
        theta (Numpy array): social security replacement rate for each
            lifetime income group
        etr_params (list): list of parameters of the effective tax rate
            functions
        e (Numpy array): effective labor units
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        total_tax_revenue (array_like): aggregate tax revenue
        iit_payroll_tax_revenue (array_like): aggregate income and
            payroll tax revenue
        agg_pension_outlays (array_like): aggregate outlays for gov't
            pensions
        UBI_outlays (array_like): aggregate universal basic income (UBI)
            outlays
        bequest_tax_revenue (array_like): aggregate bequest tax revenue
        wealth_tax_revenue (array_like): aggregate wealth tax revenue
        cons_tax_revenue (array_like): aggregate consumption tax revenue
        business_tax_revenue (array_like): aggregate business tax
            revenue
        payroll_tax_revenue (array_like): aggregate payroll tax revenue
        iit_tax_revenue (array_like): aggregate income tax revenue

    """
    inc_pay_tax_liab = tax.income_tax_liab(
        r, w, b, n, factor, 0, None, method, e, etr_params, p
    )
    pension_benefits = pensions.pension_amount(
        r, w, n, Y, theta, 0, None, False, method, e, factor, p
    )
    bq_tax_liab = tax.bequest_tax_liab(r, b, bq, 0, None, method, p)
    w_tax_liab = tax.wealth_tax_liab(r, b, 0, None, method, p)
    if method == "SS":
        p_i = np.dot(p.io_matrix, p_m)
        pop_weights = np.transpose(p.omega_SS * p.lambdas)
        iit_payroll_tax_revenue = (inc_pay_tax_liab * pop_weights).sum()
        agg_pension_outlays = (pension_benefits * pop_weights).sum()
        UBI_outlays = (ubi * pop_weights).sum()
        wealth_tax_revenue = (w_tax_liab * pop_weights).sum()
        bequest_tax_revenue = (bq_tax_liab * pop_weights).sum()
        cons_tax_revenue = (
            ((p.tau_c[-1, :] * p_i).reshape(p.I, 1, 1) * c).sum(axis=0)
            * pop_weights
        ).sum()
        payroll_tax_revenue = p.frac_tax_payroll[-1] * iit_payroll_tax_revenue
    elif method == "TPI":
        p_i = (
            np.tile(p.io_matrix.reshape(1, p.I, p.M), (p.T, 1, 1))
            * np.tile(p_m[: p.T, :].reshape(p.T, 1, p.M), (1, p.I, 1))
        ).sum(axis=2)
        pop_weights = np.squeeze(p.lambdas) * np.tile(
            np.reshape(p.omega[: p.T, :], (p.T, p.S, 1)), (1, 1, p.J)
        )
        iit_payroll_tax_revenue = (
            (inc_pay_tax_liab * pop_weights).sum(1).sum(1)
        )
        agg_pension_outlays = (pension_benefits * pop_weights).sum(1).sum(1)
        UBI_outlays = (ubi[: p.T, :, :] * pop_weights).sum(1).sum(1)
        wealth_tax_revenue = (w_tax_liab * pop_weights).sum(1).sum(1)
        bequest_tax_revenue = (bq_tax_liab * pop_weights).sum(1).sum(1)
        cons_tax_revenue = (
            (
                ((p.tau_c[: p.T, :] * p_i).reshape(p.T, p.I, 1, 1) * c).sum(
                    axis=1
                )
                * pop_weights
            )
            .sum(1)
            .sum(1)
        )
        payroll_tax_revenue = (
            p.frac_tax_payroll[: p.T] * iit_payroll_tax_revenue
        )
    business_tax_revenue = tax.get_biz_tax(w, Y, L, K, p_m, p, m, method).sum(
        -1
    )
    iit_revenue = iit_payroll_tax_revenue - payroll_tax_revenue

    total_tax_revenue = (
        iit_payroll_tax_revenue
        + wealth_tax_revenue
        + bequest_tax_revenue
        + cons_tax_revenue
        + business_tax_revenue
    )

    return (
        total_tax_revenue,
        iit_payroll_tax_revenue,
        agg_pension_outlays,
        UBI_outlays,
        bequest_tax_revenue,
        wealth_tax_revenue,
        cons_tax_revenue,
        business_tax_revenue,
        payroll_tax_revenue,
        iit_revenue,
    )


def get_r_p(r, r_gov, p_m, K_vec, K_g, D, MPKg_vec, p, method):
    r"""
    Compute the interest rate on the household's portfolio of assets,
    a mix of government debt and private equity.

    .. math::
        r_{p,t} = \frac{r_{gov,t}D_{t} + r_{K,t}K_{t}}{D_{t} + K_{t}}

    Args:
        r (array_like): the real interest rate
        r_gov (array_like): the real interest rate on government debt
        p_m (array_like): good prices
        K_vec (array_like): aggregate capital demand from each industry
        K_g (array_like): aggregate public capital
        D (array_like): aggregate government debt
        MPKg_vec (array_like): marginal product of government capital
            for each industry
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        r_p (array_like): the real interest rate on the household portfolio

    """
    if method == "SS":
        tau_b = p.tau_b[-1, :]
        T = 1
    else:
        T = p.T
        tau_b = p.tau_b[: p.T, :].reshape((p.T, p.M))
        K_g = K_g.reshape((p.T, 1))
        r = r.reshape((p.T, 1))
        r_gov = r_gov.reshape((p.T, 1))
        D = D.reshape((p.T, 1))
        p_m = p_m.reshape((p.T, p.M))
        MPKg_vec = MPKg_vec.reshape((p.T, p.M))
        K_vec = K_vec.reshape((p.T, p.M))
    r_K = r + (
        ((1 - tau_b) * p_m * MPKg_vec * K_g).sum(axis=-1).reshape((T, 1))
        / K_vec.sum(axis=-1).reshape((T, 1))
    )
    r_p = ((r_gov * D) + (r_K * K_vec.sum(axis=-1).reshape((T, 1)))) / (
        D + K_vec.sum(axis=-1).reshape((T, 1))
    )

    return np.squeeze(r_p)


def resource_constraint(Y, C, G, I_d, I_g, net_capital_flows):
    r"""
    Compute the error in the resource constraint.

    .. math::
      \text{rc_error} = \hat{Y}_t - \hat{C}_t -
      \Bigl(e^{g_y}\bigl[1 + \tilde{g}_{n,t+1}\bigr]\hat{K}^d_{t+1} -
      \hat{K}^d_t\Bigr) - \delta\hat{K}_t - \hat{G}_t - \hat{I}_{g,t} -
      \text{net capital outflows}_t

    Args:
        Y (array_like): aggregate output by industry
        C (array_like): aggregate consumption by industry
        G (array_like): aggregate government spending by industry
        I_d (array_like): aggregate private investment from domestic households
        I_g (array_like): investment in government capital
        net_capital_flows (array_like): net capital outflows

    Returns:
        rc_error (array_like): error in the resource constraint

    """
    rc_error = Y - C - I_d - I_g - G - net_capital_flows

    return rc_error


def get_capital_outflows(r, K_f, new_borrowing_f, debt_service_f, p):
    r"""
    Compute net capital outflows for open economy parameterizations

    .. math::
      \text{net capital flows} &= r_{p,t}\hat{K}^f_t ... \\
      &\quad\quad + \Bigl(e^{g_y}\bigl[1 +
      \tilde{g}_{n,t+1}\bigr]\hat{D}^f_{t+1} - \hat{D}^f_t\Bigr) -
      r_{p,t}\hat{D}^f_t \quad\forall t

    Args:
        r (array_like): the real interest rate
        K_f (array_like): aggregate capital that is foreign-owned
        new_borrowing_f (array_like): new borrowing of government debt
            from foreign investors
        debt_service_f (array_like): interest payments on government
            debt owned by foreigners
        p (OG-Core Specifications object): model parameters

    Returns:
        new_flow (array_like): net capital outflows
    """
    net_flow = (r + p.delta) * K_f - new_borrowing_f + debt_service_f

    return net_flow


def get_K_splits(B, K_demand_open, D_d, zeta_K):
    r"""
    Returns total domestic capital as well as amounts of domestic
    capital held by domestic and foreign investors separately.

    .. math::
        \begin{split}
            \hat{K}_{t} &= \hat{K}^{f}_{t} + \hat{K}^{d}_{t}\\
            \hat{K}^{d}_{t} &= \hat{B}_{t} + \hat{D}^{d}_{t}\\
            \hat{K}^{f}_{t} &= \zeta_{D}\left(\hat{K}^{open}_{t} -
                K^{d}_{t}\right)
        \end{split}

    Args:
        B (array_like): aggregate savings by domestic households
        K_demand_open (array_like): capital demand at the world
            interest rate
        D_d (array_like): governmet debt held by domestic households
        zeta_K (array_like): fraction of excess capital demand satisfied
            by foreign investors

    Returns:
        (tuple): series of capital stocks:

            * K (array_like): total capital
            * K_d (array_like): capital held by domestic households
            * K_f (array_like): capital held by foreign households

    """
    K_d = B - D_d
    if np.any(K_d < 0):
        print(
            "K_d has negative elements. Setting them "
            + "positive to prevent NAN."
        )
        K_d = np.fmax(K_d, 0.05 * B)
    K_f = zeta_K * (K_demand_open - B + D_d)
    K = K_f + K_d

    return K, K_d, K_f


def get_ptilde(p_i, tau_c, alpha_c, method="SS"):
    r"""
    Calculate price of composite good.

    .. math::
        \tilde{p}_{t} = \prod_{i=1}^{I} \left(\frac{(1 +
        \tau^{c}_{i,t})p_{i,j}}{\alpha_{i,j}}\right)^{\alpha_{i,j}}

    Args:
        p_i (array_like): prices for consumption good i
        tau_c (array_like): consumption taxes on good i
        alpha_c (array_like): consumption share parameters

    Returns:
        p_tilde (array_like): tax-inclusive price of composite good
    """
    if method == "SS":
        p_tilde = np.prod((((1 + tau_c) * p_i) / alpha_c) ** alpha_c)
    else:  # TPI case
        alpha_c = alpha_c.reshape(1, alpha_c.shape[0])
        p_tilde = np.prod((((1 + tau_c) * p_i) / alpha_c) ** alpha_c, axis=1)

    return p_tilde
