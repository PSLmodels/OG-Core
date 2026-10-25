"""
Shared helpers for the household_stoch tests.

Constructing ``ogcore.parameters.Specifications`` takes a few seconds,
so tests start from deep copies of one cached instance.
"""

import copy
import functools
import json
import os

import numpy as np
import scipy.interpolate as itp
from ogcore import SS
from ogcore.parameters import Specifications

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
# OG-Core's test calibration (S = 40, J = 2, T = 120)
OGCORE_TESTING_PARAMS = os.path.join(CUR_PATH, "testing_params.json")
# Largest acceptable Euler error in an OG-Core benchmark solution
OGCORE_EULER_TOL = 1e-8


@functools.lru_cache(maxsize=None)
def _default_specs():
    return Specifications()


@functools.lru_cache(maxsize=None)
def _ogcore_test_calibration():
    p = Specifications(baseline=True)
    with open(OGCORE_TESTING_PARAMS) as f:
        p.update_specifications(json.load(f))
    return p


def default_specs():
    """A fresh copy of OG-Core's default Specifications (J = 7)."""
    return copy.deepcopy(_default_specs())


def ogcore_test_calibration():
    """A fresh copy of OG-Core's J = 2 test calibration."""
    return copy.deepcopy(_ogcore_test_calibration())


def set_shocks(p, z_grid, z_trans):
    """
    Give p the shock process (z_grid, z_trans) through its parameters,
    with the stochastic household solver. Call before changing other
    attributes of p by hand: the update recomputes derived parameters.
    """
    z_grid = np.atleast_1d(np.asarray(z_grid, dtype=float))
    p.update_specifications(
        {
            "household_solver": "egm",
            "nz": z_grid.shape[0],
            "z_grid": z_grid.tolist(),
            "z_trans": np.atleast_2d(np.asarray(z_trans, float)).tolist(),
        }
    )
    return p


def set_ar1_shocks(p, nz, rho, sigma):
    """
    Give p a Rouwenhorst AR(1) shock process with nz states, persistence
    rho and innovation standard deviation sigma, as ``set_shocks``.
    """
    p.update_specifications(
        {
            "household_solver": "egm",
            "nz": nz,
            "z_persistence": rho,
            "z_innovation_sd": sigma,
        }
    )
    return p


def assert_ogcore_converged(euler_errors, js=None, tol=OGCORE_EULER_TOL):
    """
    Fail if OG-Core's household solution (Euler errors from
    SS.inner_loop, shape 2S x J) has not converged for types ``js``.
    """
    euler_errors = np.asarray(euler_errors)
    if js is None:
        js = range(euler_errors.shape[1])
    max_err = {j: np.abs(euler_errors[:, j]).max() for j in js}
    bad = {j: err for j, err in max_err.items() if not err < tol}
    assert not bad, (
        "OG-Core benchmark did not converge; max |Euler error| by j: "
        + ", ".join(f"j={j}: {err:.2e}" for j, err in bad.items())
    )


def simulate_profiles(b_policy, n_policy, b_grid, b_init=0.0):
    """
    Forward-simulate the deterministic (nz = 1) life-cycle profile
    implied by one type's policy functions, starting from ``b_init``,
    interpolating the policies (monotone cubic) between grid nodes.

    Args:
        b_policy, n_policy (Numpy array): policies on the grid, (S, nb)
        b_grid (Numpy array): asset grid, (nb,)

    Returns:
        b_splus1 (Numpy array): savings chosen at each age, (S,)
            (OG-Core's ``bssmat`` convention)
        n (Numpy array): labor supply at each age, (S,)
    """
    S = b_policy.shape[0]
    b_splus1 = np.zeros(S)
    n = np.zeros(S)
    b = b_init
    for s in range(S):
        n[s] = itp.PchipInterpolator(b_grid, n_policy[s])(b)
        b_splus1[s] = itp.PchipInterpolator(b_grid, b_policy[s])(b)
        b = b_splus1[s]
    return b_splus1, n


def ogcore_inner_loop(p, bssmat, nssmat, r, w, BQ, TR, factor, Y=1.4):
    """
    OG-Core's steady-state household solution at fixed prices (one
    production good). Returns Euler errors (2S x J), savings b_{s+1}
    and labor n (S x J).
    """
    outer_loop_vars = (
        np.array(bssmat, dtype=float),
        np.array(nssmat, dtype=float),
        r,
        r,
        w,
        np.array([1.0]),  # p_m
        Y,  # only enters remittances
        BQ,
        float(np.squeeze(TR)),
        0.0,  # Ig_baseline
        factor,
    )
    euler_errors, b, n = SS.inner_loop(outer_loop_vars, p, None)[:3]
    return euler_errors, b, n


def solve_ogcore_benchmark(
    p, b_guess, n_guess, r, w, BQ, TR, factor, Y=1.4, n_steps=5
):
    """
    Converged OG-Core household solution to compare against.

    SS.inner_loop is warm-started from (b_guess, n_guess). A type j that
    still does not converge is re-solved by continuation from type j - 1:
    start from j - 1's solution and per-capita bequest and step j's
    per-capita bequest to its own value. (Cold-started, inner_loop fails
    for j = 1 and j = 6 of the default calibration.)

    Returns:
        euler_errors, b, n as from ``ogcore_inner_loop``
    """
    euler, b, n = ogcore_inner_loop(
        p, b_guess, n_guess, r, w, BQ, TR, factor, Y
    )
    pop = p.omega_SS.sum(axis=0)
    for j in range(1, p.J):
        converged = np.abs(euler[:, j]).max() < OGCORE_EULER_TOL
        prev_converged = np.abs(euler[:, j - 1]).max() < OGCORE_EULER_TOL
        if converged or not prev_converged:
            continue
        bq_from, bq_to = BQ[j - 1] / pop[j - 1], BQ[j] / pop[j]
        b_j, n_j = b[:, j - 1].copy(), n[:, j - 1].copy()
        for frac in np.linspace(0.0, 1.0, n_steps + 1):
            BQ_step = np.array(BQ, dtype=float)
            BQ_step[j] = (bq_from + frac * (bq_to - bq_from)) * pop[j]
            b_start, n_start = b.copy(), n.copy()
            b_start[:, j], n_start[:, j] = b_j, n_j
            e_step, b_step, n_step = ogcore_inner_loop(
                p, b_start, n_start, r, w, BQ_step, TR, factor, Y
            )
            if not np.abs(e_step[:, j]).max() < OGCORE_EULER_TOL:
                break
            b_j, n_j = b_step[:, j], n_step[:, j]
        euler[:, j], b[:, j], n[:, j] = e_step[:, j], b_j, n_j
    return euler, b, n
