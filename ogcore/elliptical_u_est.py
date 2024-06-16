"""
------------------------------------------------------------------------
This script takes a Frisch elasticity parameter and then estimates the
parameters of the elliptical utility fuction that correspond to a
constant Frisch elasticity function with the input Frisch elasticity.
------------------------------------------------------------------------
"""

# Import packages
import numpy as np
import scipy.optimize as opt


def CFE_u(theta, l_tilde, n):
    """
    Disutility of labor supply from the constant Frisch elasticity
    utility function.

    Args:
        theta (scalar): inverse of the Frisch elasticity of labor supply
        l_tilde (scalar): maximum amount of labor supply
        n (array_like): labor supply amount

    Returns:
        u (array_like): disutility of labor supply

    """
    u = ((n / l_tilde) ** (1 + theta)) / (1 + theta)

    return u


def CFE_mu(theta, l_tilde, n):
    """
    Marginal disutility of labor supply from the constant Frisch
    elasticity utility function

    Args:
        theta (scalar):  inverse of the Frisch elasticity of labor supply
        l_tilde (scalar): maximum amount of labor supply
        n (array_like): labor supply amount

    Returns:
        mu (array_like): marginal disutility of labor supply

    """
    mu = (1.0 / l_tilde) * ((n / l_tilde) ** theta)

    return mu


def elliptical_u(b, k, upsilon, l_tilde, n):
    """
    Disutility of labor supply from the elliptical utility function

    Args:
        b (scalar):  scale parameter of elliptical utility function
        k (scalar):  shift parameter of elliptical utility function
        upsilon (scalar):  curvature parameter of elliptical utility
            function
        l_tilde (scalar): maximum amount of labor supply
        n (array_like): labor supply amount

    Returns:
        u (array_like): disutility of labor supply

    """
    u = b * ((1 - ((n / l_tilde) ** upsilon)) ** (1 / upsilon)) + k

    return u


def elliptical_mu(b, upsilon, l_tilde, n):
    """
    Marginal disutility of labor supply from the elliptical utility
    function

    Args:
        b (scalar):  scale parameter of elliptical utility function
        upsilon (scalar):  curvature parameter of elliptical utility
            function
        l_tilde (scalar): maximum amount of labor supply
        n (array_like): labor supply amount

    Returns:
        mu (array_like): marginal disutility of labor supply

    """
    mu = (
        b
        * (1.0 / l_tilde)
        * ((1.0 - (n / l_tilde) ** upsilon) ** ((1.0 / upsilon) - 1.0))
        * (n / l_tilde) ** (upsilon - 1.0)
    )

    return mu


def sumsq(params, *objs):
    """
    This function generates the sum of squared deviations between the
    constant Frisch elasticity function and the elliptical utility
    function.

    Args:
        params (tuple): parameters to estimate, (b, k, upsilon)
        objs (tuple): other parameters of utility function,
            (theta, l_tilde, n_grid)

    Returns:
        ssqdev (scalar): sum of squared errors

    """
    theta, l_tilde, n_grid = objs
    b, k, upsilon = params
    CFE = CFE_u(theta, l_tilde, n_grid)
    ellipse = elliptical_u(b, k, upsilon, l_tilde, n_grid)
    errors = CFE - ellipse
    ssqdev = (errors**2).sum()
    return ssqdev


def sumsq_MU(params, *objs):
    """
    This function generates the sum of squared deviations between the
    marginals of the constant Frisch elasticity function and the
    elliptical utility function

    Args:
        params (tuple): parameters to estimate, (b, k, upsilon)
        objs (tuple): other parameters of utility function,
            (theta, l_tilde, n_grid)

    Returns:
        ssqdev (scalar): sum of squared errors

    """
    theta, l_tilde, n_grid = objs
    b, upsilon = params
    CFE_MU = CFE_mu(theta, l_tilde, n_grid)
    ellipse_MU = elliptical_mu(b, upsilon, l_tilde, n_grid)
    # CFE_MU = (1.0 / l_tilde) * ((n_grid / l_tilde) ** theta)
    # ellipse_MU = (b * (1.0 / l_tilde) * ((1.0 - (n_grid / l_tilde) **
    #                                       upsilon) **
    #                                      ((1.0 / upsilon) - 1.0)) *
    #               (n_grid / l_tilde) ** (upsilon - 1.0))
    errors = CFE_MU - ellipse_MU
    ssqdev = (errors**2).sum()
    return ssqdev


def estimation(frisch, l_tilde):
    """
    This function estimates the parameters of an elliptical utility
    function that fits a constant frisch elasticity function.

    Args:
        frisch (scalar):  Frisch elasticity of labor supply
        l_tilde (scalar): maximum amount of labor supply

    Returns:
        b_MU_til (scalar): estimated b from elliptical utility function
        upsilon_MU_til (scalar): estimated upsilon from elliptical
            utility function

    """
    # Set parameters
    theta = 1 / frisch
    N = 101

    # Estimate parameters of elliptical utility function
    # Initial guesses
    b_init = 0.6701
    # k_init = -.6548
    upsilon_init = 2.3499
    # don't estimate near edge of range of labor supply
    n_grid = np.linspace(0.01, 0.8, num=N)

    # Estimate params using marginal utilities
    ellipse_MU_params_init = np.array([b_init, upsilon_init])
    ellipse_MU_objs = (theta, l_tilde, n_grid)
    bnds_MU = ((None, None), (None, None))
    ellipse_MU_params_til = opt.minimize(
        sumsq_MU,
        ellipse_MU_params_init,
        args=(ellipse_MU_objs),
        method="L-BFGS-B",
        bounds=bnds_MU,
        tol=1e-15,
    )
    (b_MU_til, upsilon_MU_til) = ellipse_MU_params_til.x

    return b_MU_til, upsilon_MU_til
