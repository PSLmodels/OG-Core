"""
Tests of household_stoch_kernels: the numba kernels against OG-Core's
functions, scipy, and their numpy fallbacks.
"""

import numpy as np
import pytest
import scipy.interpolate as itp
from ogcore import household as hh_core
from ogcore.txfunc import get_tax_rates

from ogcore import household_stoch as household
from ogcore import household_stoch_kernels as kernels
from stoch_test_utils import default_specs

RNG = np.random.default_rng(20261002)


@pytest.fixture(params=[True, False], ids=["numba", "numpy"])
def use_numba(request, monkeypatch):
    """Run a test with the numba kernels and with the numpy fallbacks."""
    monkeypatch.setattr(kernels, "USE_NUMBA", request.param)
    return request.param


# %%
# Marginal utilities


@pytest.mark.parametrize("sigma", [1.0, 1.5, 3.2])
def test_mu_c_matches_ogcore(sigma, use_numba):
    c = np.concatenate(
        [np.linspace(1e-6, 0.003, 30), np.linspace(0.003, 300, 70)]
    )
    assert np.allclose(
        kernels.mu_c(c, sigma), hh_core.marg_ut_cons(c, sigma), rtol=1e-14
    )


@pytest.mark.parametrize("sigma", [1.0, 1.5, 3.2])
def test_inv_mu_c_is_inverse(sigma, use_numba):
    c = np.concatenate(
        [np.linspace(1e-6, 0.003, 30), np.linspace(0.003, 300, 70)]
    )
    assert np.allclose(
        kernels.inv_mu_c(hh_core.marg_ut_cons(c, sigma), sigma),
        c,
        rtol=1e-12,
    )


@pytest.mark.parametrize("sigma", [1.0, 1.5, 3.2])
def test_inv_mu_c_derivative(sigma):
    v = np.concatenate([np.linspace(1e-3, 5.0, 20), np.linspace(1e4, 1e8, 20)])
    deriv = np.array([kernels._inv_mu_c(x, sigma)[1] for x in v])
    assert np.allclose(deriv, kernels.inv_mu_c_deriv_np(v, sigma))
    h = 1e-6 * v
    fd = (kernels.inv_mu_c(v + h, sigma) - kernels.inv_mu_c(v - h, sigma)) / (
        2 * h
    )
    assert np.allclose(deriv, fd, rtol=1e-6)


@pytest.mark.parametrize("upsilon", [1.45, 2.0, 2.856])
def test_mdu_n_matches_ogcore(upsilon, use_numba):
    p = default_specs()
    p.upsilon = upsilon
    p.ltilde = 1.2
    n = np.concatenate(
        [
            np.linspace(0.6e-6, 1e-6, 10),
            np.linspace(1e-6, p.ltilde - 1e-6, 80),
            np.linspace(p.ltilde - 1e-6, p.ltilde, 10),
        ]
    )
    level, slope = kernels.mdu_n(n, 2.5, p.b_ellipse, p.upsilon, p.ltilde)
    expected = hh_core.marg_ut_labor(n, 2.5, p)
    interior = (n > 1e-6) & (n < p.ltilde - 1e-6)
    assert np.allclose(level[interior], expected[interior], rtol=1e-12)
    # the linear stitches agree up to rounding of their steep slopes
    assert np.allclose(level, expected, rtol=1e-8)
    # slope is the derivative of the level
    m = np.linspace(0.05, 0.95, 30) * p.ltilde
    h = 1e-7
    up, _ = kernels.mdu_n(m + h, 2.5, p.b_ellipse, p.upsilon, p.ltilde)
    down, _ = kernels.mdu_n(m - h, 2.5, p.b_ellipse, p.upsilon, p.ltilde)
    _, sl = kernels.mdu_n(m, 2.5, p.b_ellipse, p.upsilon, p.ltilde)
    assert np.allclose(sl, (up - down) / (2 * h), rtol=1e-6)


def test_mu_beq_matches_marg_ut_beq(use_numba):
    p = default_specs()
    p.chi_b = np.array([3.0])
    b = np.concatenate([np.linspace(0, 0.001, 20), np.linspace(0.001, 50, 50)])
    assert np.allclose(
        kernels.mu_beq(b, 1.5, 3.0),
        household.marg_ut_beq(b, 1.5, 0, p),
        rtol=1e-14,
    )


# %%
# Tax functions: rates against ogcore.txfunc.get_tax_rates, derivatives
# against finite differences

TAX_PARAMS = {
    # A, B, C, D, max_x, max_y, share, min_x, min_y, shift_x, shift_y,
    # shift (incomes in dollars)
    "DEP": np.array(
        [
            2e-10,
            3e-5,
            1e-10,
            2e-5,
            0.35,
            0.25,
            0.7,
            0.02,
            0.01,
            0.005,
            0.004,
            -0.01,
        ]
    ),
    # A, B, max, min, shift_income, shift
    "DEP_totalinc": np.array([2e-10, 3e-5, 0.35, 0.02, 0.01, -0.01]),
    # phi0, phi1, phi2
    "GS": np.array([0.4, 0.8, 1e-4]),
    # lambda, tau
    "HSV": np.array([6.5, 0.18]),
    "linear": np.array([0.21]),
}
MTR_PARAMS = {
    "DEP": TAX_PARAMS["DEP"] * np.array([1, 1.2, 1, 1.1] + [1] * 8),
    "DEP_totalinc": TAX_PARAMS["DEP_totalinc"]
    * np.array([1.2, 1, 1, 1, 1, 1]),
    "GS": np.array([0.45, 0.7, 2e-4]),
    "HSV": np.array([6.0, 0.2]),
    "linear": np.array([0.25]),
}


def _incomes():
    X = np.concatenate([[0.0, 500.0], RNG.uniform(1e3, 3e5, 200)])
    Y = np.concatenate([[1e4, 0.0], RNG.uniform(0.0, 1e5, 200)])
    return X, Y


@pytest.mark.parametrize("tax_func_type", list(TAX_PARAMS))
@pytest.mark.parametrize("kind", ["etr", "mtrx", "mtry"])
@pytest.mark.parametrize("analytical", [False, True])
def test_tax_rates_match_ogcore(tax_func_type, kind, analytical):
    X, Y = _incomes()
    if tax_func_type in ("GS", "HSV"):
        X, Y = X[2:], Y[2:]  # singular at zero income
    etr_p = TAX_PARAMS[tax_func_type]
    mtr_p = MTR_PARAMS[tax_func_type]
    rate, d_x, d_y = kernels.tax_rates(
        tax_func_type, kind, X, Y, etr_p, mtr_p, analytical
    )
    if kind == "etr":
        expected = get_tax_rates(
            etr_p, X, Y, None, tax_func_type, "etr", for_estimation=False
        )
    else:
        expected = get_tax_rates(
            etr_p if analytical else mtr_p,
            X,
            Y,
            None,
            tax_func_type,
            "mtr",
            analytical,
            kind == "mtry",
            for_estimation=False,
        )
    assert np.allclose(rate, expected, rtol=1e-12, atol=1e-15)

    # derivatives: central differences
    def rate_at(x, y):
        return kernels.tax_rates(
            tax_func_type, kind, x, y, etr_p, mtr_p, analytical
        )[0]

    h = 1e-3 * np.maximum(1.0, 1e-4 * (X + Y))
    ok = (X > 2 * h) & (Y > 2 * h)
    fd_x = (rate_at(X + h, Y) - rate_at(X - h, Y)) / (2 * h)
    fd_y = (rate_at(X, Y + h) - rate_at(X, Y - h)) / (2 * h)
    scale = np.abs(fd_x).max() + np.abs(fd_y).max() + 1e-12
    assert np.allclose(d_x[ok], fd_x[ok], rtol=1e-4, atol=1e-6 * scale)
    assert np.allclose(d_y[ok], fd_y[ok], rtol=1e-4, atol=1e-6 * scale)


def test_dep_degenerate_shares():
    """
    share = 0 or 1 with a zero base must not give NaN derivatives (DEP
    parameters with only a shift, as in simple test calibrations).
    """
    params = np.zeros(12)
    params[10] = 0.2  # shift_y: a flat 20% rate
    rate, d_x, d_y = kernels.tax_rates(
        "DEP", "etr", np.array([1e4, 0.0]), np.array([0.0, 5e3]), params
    )
    assert np.allclose(rate, 0.2)
    assert np.allclose(d_x, 0.0) and np.allclose(d_y, 0.0)


def test_default_calibration_tax_rates_match_ogcore():
    """OG-Core's default DEP parameters, every age."""
    p = default_specs()
    etr = np.array(p.etr_params)[-1]
    mtrx = np.array(p.mtrx_params)[-1]
    mtry = np.array(p.mtry_params)[-1]
    X, Y = _incomes()
    for s in range(p.S):
        for kind, prm in (
            ("etr", etr[s]),
            ("mtrx", mtrx[s]),
            ("mtry", mtry[s]),
        ):
            rate, _, _ = kernels.tax_rates("DEP", kind, X, Y, etr[s], prm)
            expected = get_tax_rates(
                prm if kind != "etr" else etr[s],
                X,
                Y,
                None,
                "DEP",
                "etr" if kind == "etr" else "mtr",
                False,
                kind == "mtry",
                for_estimation=False,
            )
            assert np.allclose(rate, expected, rtol=1e-12, atol=1e-15)


# %%
# Monotone cubic interpolation


def _series(M=4, K=12, monotone=True):
    x = np.cumsum(RNG.uniform(0.05, 1.0, (M, K)), axis=1) - 0.3
    if monotone:
        y = np.cumsum(RNG.uniform(0.0, 1.0, (M, K)), axis=1)
    else:
        y = RNG.normal(size=(M, K))
    return x, y


@pytest.mark.parametrize("monotone", [True, False])
def test_pchip_many_matches_scipy(monotone, use_numba):
    x, y = _series(monotone=monotone)
    M, K = x.shape
    valid = np.ones((M, K), dtype=bool)
    valid[1, [2, 5]] = False  # skipped knots
    valid[2, 0] = False
    xq = np.sort(RNG.uniform(x.min() - 0.5, x.max() + 0.5, (M, 30)), axis=1)
    out = kernels.pchip_many(x, np.stack([y, 2 * y]), valid, xq)
    for m in range(M):
        xs, ys = x[m, valid[m]], y[m, valid[m]]
        inside = (xq[m] >= xs[0]) & (xq[m] <= xs[-1])
        ref = itp.PchipInterpolator(xs, ys)(xq[m, inside])
        assert np.allclose(out[0, m, inside], ref, rtol=1e-12, atol=1e-14)
        assert np.allclose(out[1, m], 2 * out[0, m])
        # linear extrapolation outside the knots
        below = xq[m] < xs[0]
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        assert np.allclose(
            out[0, m, below], ys[0] + slope * (xq[m, below] - xs[0])
        )
        above = xq[m] > xs[-1]
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        assert np.allclose(
            out[0, m, above], ys[-1] + slope * (xq[m, above] - xs[-1])
        )


def test_pchip_query_matches_pchip_many(use_numba):
    x, y = _series()
    M, K = x.shape
    valid = np.ones((M, K), dtype=bool)
    valid[3, 4] = False
    rows = RNG.integers(0, M, 50)
    xq = RNG.uniform(x.min(), x.max(), 50)
    out = kernels.pchip_query(x, y, valid, rows, xq)
    for q in range(50):
        m = rows[q]
        ref = kernels.pchip_np(x[m, valid[m]], y[m, valid[m]], [xq[q]])
        assert np.isclose(out[q], ref[0], rtol=1e-12)


def test_monotone_valid():
    x = np.array(
        [
            [-3.0, -1.0, 0.5, 0.2, 1.0, 2.0],  # dip: drop 0.5
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # fine
            [5.0, 0.0, 1.0, 2.0, 3.0, 4.0],  # first knot too high
        ]
    )
    ok = np.ones(x.shape, dtype=bool)
    ok[1, 3] = False
    expected = np.array(
        [
            [True, True, False, True, True, True],
            [True, True, True, False, True, True],
            [False, True, True, True, True, True],
        ]
    )
    for flag in (True, False):
        kernels.USE_NUMBA = flag
        try:
            assert (kernels.monotone_valid(x, ok) == expected).all()
        finally:
            kernels.USE_NUMBA = kernels.HAVE_NUMBA


# %%
# Young's method


def _young_inputs(nz=3):
    M, nb = 2, 15
    grid = np.stack(
        [
            household.make_asset_grid(5.0, nb),
            household.make_asset_grid(9.0, nb),
        ]
    )
    dist = RNG.uniform(size=(M, nb, nz))
    dist /= dist.sum(axis=(1, 2), keepdims=True)
    b_next = RNG.uniform(0.0, 4.5, (M, nb, nz))
    z_trans = RNG.uniform(size=(nz, nz))
    z_trans /= z_trans.sum(axis=1, keepdims=True)
    return dist, b_next, grid, z_trans


def test_young_step_conserves_mass_and_mean(use_numba):
    dist, b_next, grid, z_trans = _young_inputs()
    new, above, below = kernels.young_step(dist, b_next, grid, z_trans)
    assert np.allclose(new.sum(axis=(1, 2)), dist.sum(axis=(1, 2)))
    # the lottery between neighboring nodes preserves mean savings
    mean_new = (new * grid[:, :, None]).sum(axis=(1, 2))
    mean_old = (dist * b_next).sum(axis=(1, 2))
    assert np.allclose(mean_new, mean_old)
    # shocks move by the transition matrix
    z_old = dist.sum(axis=1)
    assert np.allclose(new.sum(axis=1), z_old @ z_trans)
    assert (above == 0).all() and (below == 0).all()


def test_young_step_numba_matches_numpy():
    dist, b_next, grid, z_trans = _young_inputs()
    b_next[0, 3, 1] = grid[0, -1]  # exactly at the top
    b_next[1, 2, 0] = grid[1, 0]  # exactly at the bottom
    out_nb = kernels._young_step(dist, b_next, grid, z_trans)
    out_np = kernels.young_step_np(dist, b_next, grid, z_trans)
    for a, b in zip(out_nb, out_np):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-15)


def test_young_step_reports_clipping(use_numba):
    dist, b_next, grid, z_trans = _young_inputs()
    b_next[0, 4, 2] = grid[0, -1] + 1.0
    b_next[1, 0, 0] = grid[1, 0] - 1.0
    new, above, below = kernels.young_step(dist, b_next, grid, z_trans)
    assert np.isclose(above[0], dist[0, 4, 2]) and above[1] == 0
    assert np.isclose(below[1], dist[1, 0, 0]) and below[0] == 0
    assert np.allclose(new.sum(axis=(1, 2)), dist.sum(axis=(1, 2)))


def test_young_step_rejects_nan_policy(use_numba):
    """
    A NaN savings policy where the distribution has mass is an error
    (the compiled kernel would otherwise write outside the type's
    block); where there is no mass it is ignored.
    """
    dist, b_next, grid, z_trans = _young_inputs()
    b_next[0, 2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        kernels.young_step(dist, b_next, grid, z_trans)
    with pytest.raises(ValueError):
        kernels._young_step(dist, b_next, grid, z_trans)
    dist[0, 2, 1] = 0.0
    new, _, _ = kernels.young_step(dist, b_next, grid, z_trans)
    assert np.isfinite(new).all()


# %%
# Within-period solve: labor supply and consumption given savings


@pytest.mark.parametrize("sigma", [1.0, 2.5])
def test_mu_c_deriv(sigma):
    c = np.array([-0.01, 0.0, 0.001, 0.0029, 0.0031, 0.5, 3.0])
    h = 1e-7
    fd = (kernels.mu_c_np(c + h, sigma) - kernels.mu_c_np(c - h, sigma)) / (
        2 * h
    )
    expected = kernels.mu_c_deriv_np(c, sigma)
    assert np.allclose(fd, expected, rtol=1e-5)
    jitted = np.array([kernels._mu_c_deriv(x, sigma) for x in c])
    assert np.allclose(jitted, expected, rtol=1e-14)


@pytest.mark.parametrize("upsilon", [1.4, 1.9, 2.85, 4.0])
def test_labor_floor(upsilon):
    """
    The lower bound on labor is 0 unless OG-Core's stitched marginal
    disutility turns negative near 0 (upsilon > 2); there it is just
    above the zero of the stitched function.
    """
    floor = kernels.labor_floor(0.5, upsilon, 1.0)
    assert np.isclose(kernels._labor_floor(0.5, upsilon, 1.0), floor)
    if upsilon < 2.0:
        assert floor == 0.0
    else:
        assert 0.0 < floor < kernels.MDU_N_EPS
        mdu, _ = kernels.mdu_n_np(np.array([floor]), 1.0, 0.5, upsilon, 1.0)
        assert mdu[0] > 0
        below, _ = kernels.mdu_n_np(np.array([0.0]), 1.0, 0.5, upsilon, 1.0)
        assert below[0] < 0


def _static_inputs(p, js=(0, 6)):
    return household.lifecycle_inputs_ss(
        p, 0.05, 1.2, 1.0, 100_000, 0.02, 0.0, 0.01, 0.0, 0.0, js=list(js)
    )


def _static_points(inp, s, nb=25, z=(0.5, 1.0, 2.0), b_max=20.0):
    z = np.asarray(z, dtype=float)
    shape = (inp.M, nb, z.size)
    grid = household.make_asset_grid(b_max, nb)
    b = household._flat(grid[None, :, None], shape)
    rng = np.random.default_rng(s)
    bp = b * rng.uniform(0.5, 1.1, b.size) + rng.uniform(0, 0.05, b.size)
    pa = household._point_arrays(inp, s, shape, z)
    return shape, z, b, bp, pa


def _budget_and_foc(p, inp, s, shape, z, b, bp, n, c):
    """OG-Core's budget residual and labor FOC at the points."""
    M, nb, nz = shape
    bc = np.empty(shape)
    foc = np.empty(shape)
    mdu = np.empty(shape)
    etr = np.array(p.etr_params)[-1, s]
    mtrx = np.array(p.mtrx_params)[-1, s]
    for m, j in enumerate(inp.js):
        for iz in range(nz):
            sl = np.s_[m, :, iz]
            bb = b.reshape(shape)[sl]
            nn = n.reshape(shape)[sl]
            cc = c.reshape(shape)[sl]
            net_tax = household.get_net_taxes(
                0.05,
                1.2,
                bb,
                nn,
                inp.bq[s, m],
                100_000,
                inp.tr[s, m],
                0.0,
                0.0,
                None,
                j,
                "SS",
                p.e[-1, s, j] * z[iz],
                etr,
                p,
            )
            bc[sl] = household.BC_residual(
                cc,
                nn,
                bb,
                bp.reshape(shape)[sl],
                0.05,
                1.2,
                1.0,
                np.ones(1),
                p.e[-1, s, j],
                z[iz],
                inp.bq[s, m],
                0.0,
                net_tax,
                p.tau_c[-1, :],
                p,
            )
            foc[sl] = household.FOC_labor(
                0.05,
                1.2,
                1.0,
                bb,
                cc,
                nn,
                100_000,
                p.e[-1, s, j],
                z[iz],
                p.chi_n[-1, s],
                etr,
                mtrx,
                None,
                j,
                p,
                "SS",
            )
            mdu[sl] = household.marg_ut_labor(nn, p.chi_n[-1, s], p)
    return bc.ravel(), foc.ravel(), mdu.ravel()


@pytest.mark.parametrize(
    "analytical", [False, True], ids=["mtr", "analytical"]
)
def test_static_solve(analytical):
    """
    Consumption comes from the budget constraint at the labor supply
    returned, so the budget holds exactly at every point (whatever the
    status); converged points satisfy OG-Core's labor FOC; corners have
    labor at its lower bound with the FOC as an inequality. The numba
    and numpy backends agree. With analytical MTRs (above 1 at high
    assets and low labor income) some points are corners.
    """
    p = default_specs()
    p.analytical_mtrs = analytical
    inp = _static_inputs(p)
    floor = kernels.labor_floor(inp.b_ellipse, inp.upsilon, inp.ltilde)
    corners = 0
    for s in (5, 40, 79):
        shape, z, b, bp, pa = _static_points(inp, s)
        out = {}
        for name, cls in (
            ("numba", household._NumbaBackend),
            ("numpy", household._NumpyBackend),
        ):
            n, c, st, res = cls(inp).static(
                s, np.full(b.size, 0.3), b, bp, np.zeros(b.size), pa, 1e-12, 60
            )
            out[name] = (n, c, st)
            assert np.isin(st, [kernels.CONVERGED, kernels.CORNER]).all()
            bc, foc, mdu = _budget_and_foc(p, inp, s, shape, z, b, bp, n, c)
            assert np.abs(bc).max() < 1e-12
            conv = st == kernels.CONVERGED
            assert np.abs(foc[conv] / mdu[conv]).max() < 1e-8
            # corners: no labor and the household would rather work
            # less, or all of the time endowment and it would rather
            # work more
            bottom = (st == kernels.CORNER) & (n < 0.5 * inp.ltilde)
            top = (st == kernels.CORNER) & ~bottom
            assert np.allclose(n[bottom], floor)
            assert (foc[bottom] <= 0).all()
            assert np.allclose(n[top], inp.ltilde)
            assert (foc[top] >= 0).all()
            corners += bottom.sum()
        assert (out["numba"][2] == out["numpy"][2]).all()
        assert np.allclose(out["numba"][0], out["numpy"][0], atol=1e-9)
        assert np.allclose(out["numba"][1], out["numpy"][1], rtol=1e-9)
    assert (corners > 0) == analytical


def test_static_solve_corners():
    """
    A labor MTR above 1 - tau_payroll puts labor supply at its lower
    bound; a negligible disutility of labor puts it at the time
    endowment. In both cases the budget holds.
    """
    p = default_specs()
    inp = _static_inputs(p, js=(2,))
    shape, z, b, bp, pa = _static_points(inp, 30, nb=8, b_max=5.0)
    floor = kernels.labor_floor(inp.b_ellipse, inp.upsilon, inp.ltilde)
    zeros = np.zeros(b.size)
    for cls in (household._NumbaBackend, household._NumpyBackend):
        high_mtr = household.LifecycleInputs(**vars(inp))
        # DEP with shift 1.1 and flat parts zero: a flat 110% MTR
        mtrx = np.zeros_like(np.asarray(inp.mtrx_params))
        mtrx[:, 11] = 1.1
        high_mtr.mtrx_params = mtrx
        n, c, st, _ = cls(high_mtr).static(
            30, np.full(b.size, 0.3), b, bp, zeros, pa, 1e-12, 60
        )
        assert (st == kernels.CORNER).all() and np.allclose(n, floor)
        lazy = household.LifecycleInputs(**vars(inp))
        lazy.chi_n = inp.chi_n * 1e-9
        n, c, st, _ = cls(lazy).static(
            30, np.full(b.size, 0.3), b, bp, zeros, pa, 1e-12, 60
        )
        assert (st == kernels.CORNER).all()
        assert np.allclose(n, inp.ltilde)
        # consumption from the budget at n = ltilde
        bc, _, _ = _budget_and_foc(p, lazy, 30, shape, z, b, bp, n, c)
        assert np.abs(bc).max() < 1e-12
