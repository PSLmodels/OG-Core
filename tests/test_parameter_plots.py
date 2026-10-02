"""
Tests of parameter_plots.py module
"""

import pytest
import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import matplotlib.image as mpimg
from ogcore import utils, parameter_plots, Specifications

# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
base_ss = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "SS_vars_baseline.pkl")
)
base_tpi = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TPI_vars_baseline.pkl")
)

reform_ss = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "SS_vars_reform.pkl")
)
reform_tpi = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TPI_vars_reform.pkl")
)
base_params = Specifications()
base_params.update_specifications(
    {
        "M": 3,
        "gamma": [0.5, 0.5, 0.5],
        "gamma_g": [0.0, 0.0, 0.0],
        "epsilon": [0.5, 0.5, 0.5],
        "I": 3,
        "alpha_c": [0.3, 0.4, 0.3],
        "io_matrix": np.eye(3),
    }
)
reform_params = Specifications()
reform_params.update_specifications(
    {
        "M": 3,
        "gamma": [0.5, 0.5, 0.5],
        "gamma_g": [0.0, 0.0, 0.0],
        "epsilon": [0.5, 0.5, 0.5],
        "I": 3,
        "alpha_c": [0.3, 0.4, 0.3],
        "io_matrix": np.eye(3),
    }
)

base_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_baseline.pkl")
)
GS_nonage_spec_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_GS_nonage.pkl")
)
if sys.version_info[1] < 11:
    mono_nonage_spec_taxfunctions = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_mono_nonage.pkl")
    )
micro_data = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
)
# rho is now (T+S, S, J) - no reshape needed


def test_plot_imm_rates():
    fig = parameter_plots.plot_imm_rates(
        base_params.imm_rates,
        base_params.start_year,
        [base_params.start_year],
        include_title=True,
    )
    assert fig


def test_plot_imm_rates_save_fig(tmpdir):
    parameter_plots.plot_imm_rates(
        base_params.imm_rates,
        base_params.start_year,
        [base_params.start_year],
        path=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "imm_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_mort_rates():
    fig = parameter_plots.plot_mort_rates([base_params], include_title=True)
    assert fig
    plt.close()


def test_plot_surv_rates():
    fig = parameter_plots.plot_mort_rates(
        [base_params], survival_rates=True, include_title=True
    )
    assert fig
    plt.close()


def test_plot_mort_rates_save_fig(tmpdir):
    parameter_plots.plot_mort_rates([base_params], path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "mortality_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_surv_rates_save_fig(tmpdir):
    parameter_plots.plot_mort_rates(
        [base_params], survival_rates=True, path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "survival_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_pop_growth():
    fig = parameter_plots.plot_pop_growth(
        base_params, start_year=int(base_params.start_year), include_title=True
    )
    assert fig
    plt.close()


def test_plot_pop_growth_rates_save_fig(tmpdir):
    parameter_plots.plot_pop_growth(
        base_params, start_year=int(base_params.start_year), path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "pop_growth_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_ability_profiles():
    p = Specifications()
    fig = parameter_plots.plot_ability_profiles(p, p2=p, include_title=True)
    assert fig
    plt.close()


def test_plot_log_ability_profiles():
    p = Specifications()
    fig = parameter_plots.plot_ability_profiles(
        p, p2=p, log_scale=True, include_title=True
    )
    assert fig
    plt.close()


def test_plot_ability_profiles_save_fig(tmpdir):
    p = Specifications()
    parameter_plots.plot_ability_profiles(p, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "ability_profiles.png"))

    assert isinstance(img, np.ndarray)


def test_plot_elliptical_u():
    fig1 = parameter_plots.plot_elliptical_u(base_params, include_title=True)
    fig2 = parameter_plots.plot_elliptical_u(
        base_params, plot_MU=False, include_title=True
    )
    assert fig1
    assert fig2
    plt.close()


def test_plot_elliptical_u_save_fig(tmpdir):
    parameter_plots.plot_elliptical_u(base_params, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "ellipse_v_CFE.png"))

    assert isinstance(img, np.ndarray)


def test_plot_chi_n():
    p = Specifications()
    fig = parameter_plots.plot_chi_n([p], include_title=True)
    assert fig
    plt.close()


def test_plot_chi_n_save_fig(tmpdir):
    p = Specifications()
    parameter_plots.plot_chi_n([p], path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "chi_n_values.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize(
    "years_to_plot",
    [["SS"], [2025], [2050, 2070]],
    ids=["SS", "2025", "List of years"],
)
def test_plot_population(years_to_plot):
    fig = parameter_plots.plot_population(
        base_params, years_to_plot=years_to_plot, include_title=True
    )
    assert fig
    plt.close()


def test_plot_population_save_fig(tmpdir):
    parameter_plots.plot_population(base_params, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "pop_distribution.png"))

    assert isinstance(img, np.ndarray)


def test_plot_fert_rates():
    totpers = base_params.S
    fert_data = (
        np.array(
            [
                0.0,
                0.0,
                0.3,
                12.3,
                47.1,
                80.7,
                105.5,
                98.0,
                49.3,
                10.4,
                0.8,
                0.0,
                0.0,
            ]
        )
        / 2000
    )
    age_midp = np.array([9, 10, 12, 16, 18.5, 22, 27, 32, 37, 42, 47, 55, 56])
    fert_func = si.interp1d(age_midp, fert_data, kind="cubic")
    fert_rates = np.random.uniform(size=totpers)
    fig = parameter_plots.plot_fert_rates([fert_rates], include_title=True)
    assert fig
    plt.close()


def test_plot_fert_rates_save_fig(tmpdir):
    totpers = base_params.S
    fert_data = (
        np.array(
            [
                0.0,
                0.0,
                0.3,
                12.3,
                47.1,
                80.7,
                105.5,
                98.0,
                49.3,
                10.4,
                0.8,
                0.0,
                0.0,
            ]
        )
        / 2000
    )
    age_midp = np.array([9, 10, 12, 16, 18.5, 22, 27, 32, 37, 42, 47, 55, 56])
    fert_func = si.interp1d(age_midp, fert_data, kind="cubic")
    fert_rates = np.random.uniform(size=totpers)
    parameter_plots.plot_fert_rates(
        [fert_rates],
        include_title=True,
        path=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "fert_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_fert_rates_many_series_numeric_labels():
    """Test plot_fert_rates with >4 series and numeric (year) labels.
    Covers the num_series > 4 branch with the try block succeeding."""
    totpers = base_params.S
    fert_rates_list = [np.random.uniform(size=totpers) for _ in range(5)]
    labels = ["2020", "2021", "2022", "2023", "2024"]
    fig = parameter_plots.plot_fert_rates(
        fert_rates_list, labels=labels, include_title=True
    )
    assert fig
    plt.close()


def test_plot_fert_rates_many_series_nonnumeric_labels():
    """Test plot_fert_rates with >4 series and non-numeric labels.
    Covers the num_series > 4 branch with the except block (fallback to range).
    """
    totpers = base_params.S
    fert_rates_list = [np.random.uniform(size=totpers) for _ in range(5)]
    labels = ["Baseline", "Reform A", "Reform B", "Reform C", "Reform D"]
    fig = parameter_plots.plot_fert_rates(
        fert_rates_list, labels=labels, include_title=False
    )
    assert fig
    plt.close()


def test_plot_fert_rates_many_series_save_fig(tmpdir):
    """Test plot_fert_rates with >4 series saves a figure to disk."""
    totpers = base_params.S
    fert_rates_list = [np.random.uniform(size=totpers) for _ in range(5)]
    labels = ["2020", "2021", "2022", "2023", "2024"]
    parameter_plots.plot_fert_rates(
        fert_rates_list, labels=labels, path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "fert_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_g_n():
    p = Specifications()
    fig = parameter_plots.plot_g_n([p], include_title=True)
    assert fig
    plt.close()


def test_plot_g_n_savefig(tmpdir):
    p = Specifications()
    parameter_plots.plot_g_n([p], include_title=True, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "pop_growth_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_mort_rates_data():
    totpers = base_params.S - 1
    mort_rates = (
        (base_params.rho[-1, 1:, :] * base_params.omega[-1, 1:, :])
        .mean(axis=-1)
        .reshape((1, totpers))
    )
    fig = parameter_plots.plot_mort_rates_data(
        mort_rates,
        path=None,
    )
    assert fig
    plt.close()


def test_plot_mort_rates_data_save_fig(tmpdir):
    totpers = base_params.S - 1
    mort_rates = (
        (base_params.rho[-1, 1:, :] * base_params.omega[-1, 1:, :])
        .mean(axis=-1)
        .reshape((1, totpers))
    )
    parameter_plots.plot_mort_rates_data(
        mort_rates,
        path=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "mort_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_omega_fixed():
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    omega_SS_orig = base_params.omega_SS
    omega_SSfx = base_params.omega_SS
    fig = parameter_plots.plot_omega_fixed(
        age_per_EpS, omega_SS_orig, omega_SSfx, E, S
    )
    assert fig
    plt.close()


def test_plot_omega_fixed_save_fig(tmpdir):
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    omega_SS_orig = base_params.omega_SS
    omega_SSfx = base_params.omega_SS
    parameter_plots.plot_omega_fixed(
        age_per_EpS, omega_SS_orig, omega_SSfx, E, S, path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "OrigVsFixSSpop.png"))

    assert isinstance(img, np.ndarray)


def test_plot_imm_fixed():
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    imm_rates_orig = base_params.imm_rates[0, :]
    imm_rates_adj = base_params.imm_rates[-1, :]
    fig = parameter_plots.plot_imm_fixed(
        age_per_EpS, imm_rates_orig, imm_rates_adj, E, S
    )
    assert fig
    plt.close()


def test_plot_imm_fixed_save_fig(tmpdir):
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    imm_rates_orig = base_params.imm_rates[0, :]
    imm_rates_adj = base_params.imm_rates[-1, :]
    parameter_plots.plot_imm_fixed(
        age_per_EpS, imm_rates_orig, imm_rates_adj, E, S, path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "OrigVsAdjImm.png"))

    assert isinstance(img, np.ndarray)


def test_plot_population_path():
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    initial_pop_pct = base_params.omega[0, :]
    omega_path_lev = base_params.omega
    omega_SSfx = base_params.omega_SS
    data_year = base_params.start_year
    curr_year = base_params.start_year
    fig = parameter_plots.plot_population_path(
        age_per_EpS,
        omega_path_lev,
        omega_SSfx,
        data_year,
        curr_year,
        curr_year + 5,
        S,
    )
    assert fig
    plt.close()


def test_plot_population_path_save_fig(tmpdir):
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    omega_path_lev = base_params.omega
    omega_SSfx = base_params.omega_SS
    curr_year = base_params.start_year
    parameter_plots.plot_population_path(
        age_per_EpS,
        omega_path_lev,
        omega_SSfx,
        curr_year,
        curr_year + 3,
        curr_year + 50,
        S,
        path=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "PopDistPath.png"))

    assert isinstance(img, np.ndarray)


# Tests of the by_J conditional behavior of the demographic plots.
# When the demographic objects have an income-group (J) dimension, the
# default (by_J=False) collapses over J to a single line per series and
# by_J=True plots one line per income group, labeled with the j value.


def _lines(fig):
    return fig.axes[0].lines


def _labels(fig):
    return [line.get_label() for line in _lines(fig)]


def _legend_labels(fig):
    legend = fig.axes[0].get_legend()
    if legend is None:
        return None
    return [text.get_text() for text in legend.get_texts()]


# Mortality rates in the default parameterization do not vary across
# income groups, so build an array that does to make sure the
# population-weighted average is actually being used.
rho_het = base_params.rho ** (0.5 + 0.2 * np.arange(base_params.J)).reshape(
    1, 1, base_params.J
)
imm_het = base_params.imm_rates * (1 + 0.5 * np.arange(base_params.J)).reshape(
    1, 1, base_params.J
)


def test_avg_over_J_unweighted():
    rates = np.array([[0.1, 0.3], [0.2, 0.6]])
    np.testing.assert_allclose(
        parameter_plots._avg_over_J(rates), np.array([0.2, 0.4])
    )


def test_avg_over_J_weighted():
    rates = np.array([[0.1, 0.3], [0.2, 0.6]])
    # weights are normalized within each age: [0.75, 0.25] and [0, 1]
    omega = np.array([[0.3, 0.1], [0.0, 0.6]])
    expected = np.array([0.75 * 0.1 + 0.25 * 0.3, 0.6])
    np.testing.assert_allclose(
        parameter_plots._avg_over_J(rates, omega), expected
    )


def test_avg_over_J_zero_pop_falls_back_to_equal_weights():
    rates = np.array([[0.1, 0.3]])
    omega = np.zeros((1, 2))
    np.testing.assert_allclose(
        parameter_plots._avg_over_J(rates, omega), np.array([0.2])
    )


def test_avg_over_J_shape_mismatch():
    with pytest.raises(AssertionError):
        parameter_plots._avg_over_J(np.ones((3, 2)), np.ones((3, 3)))


def test_sum_over_J():
    dist = np.array([[0.1, 0.3], [0.2, 0.4]])
    np.testing.assert_allclose(
        parameter_plots._sum_over_J(dist), np.array([0.4, 0.6])
    )


def test_plot_age_profile_bad_inputs():
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError):
        parameter_plots._plot_age_profile(ax, np.ones((2, 2, 2)), "x", False)
    with pytest.raises(AssertionError):
        parameter_plots._plot_age_profile(
            ax, np.ones((2, 2)), "x", False, collapse="max"
        )
    plt.close()


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_population_by_J(by_J):
    p = base_params
    J = p.J
    years = ["SS", int(p.start_year) + 5]
    fig = parameter_plots.plot_population(p, years_to_plot=years, by_J=by_J)
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == len(years) * J
        assert labels[:J] == ["SS pop., j=" + str(j) for j in range(J)]
        assert labels[J] == str(years[1]) + " pop., j=0"
        np.testing.assert_allclose(lines[J + 2].get_ydata(), p.omega[5, :, 2])
    else:
        assert len(lines) == len(years)
        assert labels == ["SS pop.", str(years[1]) + " pop."]
        np.testing.assert_allclose(
            lines[0].get_ydata(), p.omega_SS.sum(axis=-1)
        )
        np.testing.assert_allclose(
            lines[1].get_ydata(), p.omega[5].sum(axis=-1)
        )
        # the overall population distribution sums to one
        assert np.isclose(lines[0].get_ydata().sum(), 1.0)
    plt.close()


def test_plot_population_by_J_save_fig(tmpdir):
    parameter_plots.plot_population(base_params, by_J=True, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "pop_distribution.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
@pytest.mark.parametrize("survival_rates", [False, True], ids=["mort", "surv"])
def test_plot_mort_rates_by_J(by_J, survival_rates):
    p = copy.deepcopy(base_params)
    p.rho = rho_het
    J = p.J
    fig = parameter_plots.plot_mort_rates(
        [p],
        labels=["Base"],
        years=[p.start_year],
        survival_rates=survival_rates,
        by_J=by_J,
    )
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == J
        assert labels == [
            "Base " + str(p.start_year) + ", j=" + str(j) for j in range(J)
        ]
        idx = 3
        expected = p.rho[0, :, idx]
    else:
        assert len(lines) == 1
        assert labels == ["Base " + str(p.start_year)]
        idx = 0
        expected = parameter_plots._avg_over_J(p.rho[0], p.omega[0])
        # weighted average differs from the simple mean across j
        assert not np.allclose(expected, p.rho[0].mean(axis=-1))
        # mortality at the last age is one regardless of weighting
        assert np.isclose(expected[-1], 1.0)
    if survival_rates:
        expected = np.cumprod(1 - expected)
    np.testing.assert_allclose(lines[idx].get_ydata(), expected)
    plt.close()


def test_plot_mort_rates_2D_omega_uses_lambdas():
    # when omega has no J dimension, lambdas are used as the weights
    p = copy.deepcopy(base_params)
    p.rho = rho_het
    p.omega = base_params.omega.sum(axis=-1)
    fig = parameter_plots.plot_mort_rates([p], years=[p.start_year])
    lines = _lines(fig)
    assert len(lines) == 1
    expected = (p.rho[0] * p.lambdas.reshape(1, p.J)).sum(axis=-1)
    np.testing.assert_allclose(lines[0].get_ydata(), expected)
    plt.close()


def test_plot_mort_rates_by_J_save_fig(tmpdir):
    parameter_plots.plot_mort_rates([base_params], by_J=True, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "mortality_rates.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_imm_rates_by_J(by_J):
    p = base_params
    J = p.J
    years = [p.start_year, p.start_year + 10]
    fig = parameter_plots.plot_imm_rates(
        imm_het, p.start_year, years, by_J=by_J, omega=p.omega
    )
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == len(years) * J
        assert labels[J] == "Year " + str(years[1]) + ", j=0"
        np.testing.assert_allclose(lines[J + 4].get_ydata(), imm_het[10, :, 4])
    else:
        assert len(lines) == len(years)
        assert labels == ["Year " + str(y) for y in years]
        expected = parameter_plots._avg_over_J(imm_het[10], p.omega[10])
        np.testing.assert_allclose(lines[1].get_ydata(), expected)
        assert not np.allclose(expected, imm_het[10].mean(axis=-1))
    plt.close()


def test_plot_imm_rates_3D_no_omega_uses_simple_mean():
    p = base_params
    fig = parameter_plots.plot_imm_rates(imm_het, p.start_year, [p.start_year])
    lines = _lines(fig)
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0].get_ydata(), imm_het[0].mean(axis=-1))
    plt.close()


def test_plot_imm_rates_2D_ignores_by_J():
    p = base_params
    imm_2D = imm_het.mean(axis=-1)
    fig = parameter_plots.plot_imm_rates(
        imm_2D, p.start_year, [p.start_year], by_J=True
    )
    lines = _lines(fig)
    assert len(lines) == 1
    assert _labels(fig) == ["Year " + str(p.start_year)]
    np.testing.assert_allclose(lines[0].get_ydata(), imm_2D[0])
    plt.close()


def test_plot_imm_rates_by_J_save_fig(tmpdir):
    parameter_plots.plot_imm_rates(
        imm_het,
        base_params.start_year,
        [base_params.start_year],
        by_J=True,
        path=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "imm_rates.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_mort_rates_data_by_J(by_J):
    p = base_params
    J = p.J
    years = [p.start_year, p.start_year + 3]
    fig = parameter_plots.plot_mort_rates_data(
        rho_het, p.start_year, years, by_J=by_J, omega=p.omega
    )
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == len(years) * J
        assert labels[J + 1] == "Year " + str(years[1]) + ", j=1"
        np.testing.assert_allclose(lines[J + 1].get_ydata(), rho_het[3, :, 1])
    else:
        assert len(lines) == len(years)
        assert labels == ["Year " + str(y) for y in years]
        expected = parameter_plots._avg_over_J(rho_het[3], p.omega[3])
        np.testing.assert_allclose(lines[1].get_ydata(), expected)
        assert not np.allclose(expected, rho_het[3].mean(axis=-1))
    plt.close()


def test_plot_mort_rates_data_by_J_save_fig(tmpdir):
    parameter_plots.plot_mort_rates_data(
        rho_het,
        base_params.start_year,
        [base_params.start_year],
        by_J=True,
        path=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "mort_rates.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_fert_rates_by_J(by_J):
    S, J = base_params.S, base_params.J
    rng = np.random.default_rng(0)
    fert_list = [rng.uniform(size=(S, J)) for _ in range(2)]
    omega_list = [base_params.omega[0], base_params.omega[1]]
    labels_in = ["2025", "2026"]
    fig = parameter_plots.plot_fert_rates(
        fert_list, labels=labels_in, by_J=by_J, omega_list=omega_list
    )
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == 2 * J
        assert labels[J + 1] == "2026, j=1"
        np.testing.assert_allclose(
            lines[J + 1].get_ydata(), fert_list[1][:, 1]
        )
        assert _legend_labels(fig) == labels
    else:
        assert len(lines) == 2
        assert labels == labels_in
        expected = parameter_plots._avg_over_J(fert_list[1], omega_list[1])
        np.testing.assert_allclose(lines[1].get_ydata(), expected)
        assert not np.allclose(expected, fert_list[1].mean(axis=-1))
    plt.close()


def test_plot_fert_rates_2D_no_omega_uses_simple_mean():
    S, J = base_params.S, base_params.J
    rng = np.random.default_rng(1)
    fert_list = [rng.uniform(size=(S, J))]
    fig = parameter_plots.plot_fert_rates(fert_list, labels=["2025"])
    lines = _lines(fig)
    assert len(lines) == 1
    np.testing.assert_allclose(
        lines[0].get_ydata(), fert_list[0].mean(axis=-1)
    )
    plt.close()


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_fert_rates_many_series_by_J(by_J):
    """The >4 series (colormap) branch: with by_J the color denotes the
    year and the line style denotes the income group."""
    S, J = base_params.S, base_params.J
    rng = np.random.default_rng(2)
    num_series = 5
    fert_list = [rng.uniform(size=(S, J)) for _ in range(num_series)]
    labels_in = [str(2020 + i) for i in range(num_series)]
    fig = parameter_plots.plot_fert_rates(
        fert_list, labels=labels_in, by_J=by_J
    )
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == num_series * J
        assert labels[J] == "2021, j=0"
        np.testing.assert_allclose(
            lines[J + 5].get_ydata(), fert_list[1][:, 5]
        )
        # lines for the same year share a color
        assert lines[J].get_color() == lines[J + 5].get_color()
        # lines for different income groups have different line styles
        assert lines[J].get_linestyle() != lines[J + 5].get_linestyle()
        assert _legend_labels(fig) == ["j=" + str(j) for j in range(J)]
    else:
        assert len(lines) == num_series
        assert labels == labels_in
        np.testing.assert_allclose(
            lines[1].get_ydata(), fert_list[1].mean(axis=-1)
        )
        assert _legend_labels(fig) is None
    plt.close()


def test_plot_fert_rates_omega_list_length_mismatch():
    S, J = base_params.S, base_params.J
    fert_list = [np.ones((S, J)), np.ones((S, J))]
    with pytest.raises(AssertionError):
        parameter_plots.plot_fert_rates(
            fert_list, labels=["a", "b"], omega_list=[base_params.omega[0]]
        )
    plt.close()


def test_plot_fert_rates_by_J_save_fig(tmpdir):
    S, J = base_params.S, base_params.J
    fert_list = [np.random.uniform(size=(S, J))]
    parameter_plots.plot_fert_rates(
        fert_list, labels=["2025"], by_J=True, path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "fert_rates.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_omega_fixed_by_J(by_J):
    S, J = base_params.S, base_params.J
    age_per_EpS = np.arange(21, S + 21)
    omega_SS = base_params.omega_SS
    fig = parameter_plots.plot_omega_fixed(
        age_per_EpS, omega_SS, omega_SS, 0, S, by_J=by_J
    )
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == 2 * J
        assert labels[0] == "Original Dist'n, j=0"
        assert labels[J] == "Fixed Dist'n, j=0"
        np.testing.assert_allclose(lines[1].get_ydata(), omega_SS[:, 1])
    else:
        assert len(lines) == 2
        assert labels == ["Original Dist'n", "Fixed Dist'n"]
        np.testing.assert_allclose(lines[0].get_ydata(), omega_SS.sum(axis=-1))
        assert np.isclose(lines[0].get_ydata().sum(), 1.0)
    plt.close()


def test_plot_omega_fixed_by_J_save_fig(tmpdir):
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    omega_SS = base_params.omega_SS
    parameter_plots.plot_omega_fixed(
        age_per_EpS, omega_SS, omega_SS, 0, S, by_J=True, path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "OrigVsFixSSpop.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_imm_fixed_by_J(by_J):
    S, J = base_params.S, base_params.J
    age_per_EpS = np.arange(21, S + 21)
    imm_orig = imm_het[0]
    imm_adj = imm_het[-1]
    omega = base_params.omega_SS
    fig = parameter_plots.plot_imm_fixed(
        age_per_EpS, imm_orig, imm_adj, 0, S, by_J=by_J, omega=omega
    )
    lines = _lines(fig)
    labels = _labels(fig)
    if by_J:
        assert len(lines) == 2 * J
        assert labels[0] == "Original Imm. Rates, j=0"
        assert labels[J + 2] == "Adj. Imm. Rates, j=2"
        np.testing.assert_allclose(lines[J + 2].get_ydata(), imm_adj[:, 2])
    else:
        assert len(lines) == 2
        assert labels == ["Original Imm. Rates", "Adj. Imm. Rates"]
        expected = parameter_plots._avg_over_J(imm_adj, omega)
        np.testing.assert_allclose(lines[1].get_ydata(), expected)
        assert not np.allclose(expected, imm_adj.mean(axis=-1))
    plt.close()


def test_plot_imm_fixed_by_J_save_fig(tmpdir):
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    parameter_plots.plot_imm_fixed(
        age_per_EpS, imm_het[0], imm_het[-1], 0, S, by_J=True, path=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "OrigVsAdjImm.png"))

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_population_path_by_J(by_J):
    S, J = base_params.S, base_params.J
    age_per_EpS = np.arange(21, S + 21)
    omega_path = base_params.omega
    omega_SSfx = base_params.omega_SS
    start_year = base_params.start_year
    year1 = start_year
    year2 = start_year + 5
    fig = parameter_plots.plot_population_path(
        age_per_EpS,
        omega_path,
        omega_SSfx,
        start_year,
        year1,
        year2,
        S,
        by_J=by_J,
    )
    lines = _lines(fig)
    labels = _labels(fig)
    num_dists = 5
    if by_J:
        assert len(lines) == num_dists * J
        assert labels[0] == str(year1) + " pop., j=0"
        assert labels[J] == str(year2) + " pop., j=0"
        assert labels[-1] == "Adj. SS pop., j=" + str(J - 1)
        np.testing.assert_allclose(
            lines[J + 2].get_ydata(),
            omega_path[5, :, 2] / omega_path[5].sum(),
        )
    else:
        assert len(lines) == num_dists
        assert labels == [
            str(year1) + " pop.",
            str(year2) + " pop.",
            "T=" + str(int(0.5 * S)) + " pop.",
            "T=" + str(int(S)) + " pop.",
            "Adj. SS pop.",
        ]
        # second line is year2 = start_year + 5, i.e., period index 5
        np.testing.assert_allclose(
            lines[1].get_ydata(),
            omega_path[5].sum(axis=-1) / omega_path[5].sum(),
        )
        np.testing.assert_allclose(
            lines[-1].get_ydata(), omega_SSfx.sum(axis=-1)
        )
    plt.close()


def test_plot_population_path_2D_ignores_by_J():
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    omega_path = base_params.omega.sum(axis=-1)
    omega_SSfx = base_params.omega_SS.sum(axis=-1)
    start_year = base_params.start_year
    fig = parameter_plots.plot_population_path(
        age_per_EpS,
        omega_path,
        omega_SSfx,
        start_year,
        start_year,
        start_year + 5,
        S,
        by_J=True,
    )
    lines = _lines(fig)
    assert len(lines) == 5
    np.testing.assert_allclose(lines[-1].get_ydata(), omega_SSfx)
    plt.close()


def test_plot_population_path_by_J_save_fig(tmpdir):
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    start_year = base_params.start_year
    parameter_plots.plot_population_path(
        age_per_EpS,
        base_params.omega,
        base_params.omega_SS,
        start_year,
        start_year,
        start_year + 5,
        S,
        by_J=True,
        path=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "PopDistPath.png"))

    assert isinstance(img, np.ndarray)


# Tests of the dimension checks and of backward compatibility with
# demographic objects that have no income-group (J) dimension


def test_check_ndim():
    arr = parameter_plots._check_ndim([[1, 2], [3, 4]], "x", (1, 2))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    with pytest.raises(AssertionError, match="x must have 1 or 2"):
        parameter_plots._check_ndim(np.ones((2, 2, 2)), "x", (1, 2))


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_population_2D(by_J):
    # omega with no J dimension (T+S x S) and omega_SS (S,)
    p = copy.deepcopy(base_params)
    p.omega = base_params.omega.sum(axis=-1)
    p.omega_SS = base_params.omega_SS.sum(axis=-1)
    years = ["SS", int(p.start_year) + 5]
    fig = parameter_plots.plot_population(p, years_to_plot=years, by_J=by_J)
    lines = _lines(fig)
    assert len(lines) == len(years)
    assert _labels(fig) == ["SS pop.", str(years[1]) + " pop."]
    np.testing.assert_allclose(lines[0].get_ydata(), p.omega_SS)
    np.testing.assert_allclose(lines[1].get_ydata(), p.omega[5])
    plt.close()


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_mort_rates_2D_rho(by_J):
    # rho with no J dimension (T+S x S)
    p = copy.deepcopy(base_params)
    p.rho = base_params.rho.mean(axis=-1)
    fig = parameter_plots.plot_mort_rates(
        [p], labels=["Base"], years=[p.start_year], by_J=by_J
    )
    lines = _lines(fig)
    assert len(lines) == 1
    assert _labels(fig) == ["Base " + str(p.start_year)]
    np.testing.assert_allclose(lines[0].get_ydata(), p.rho[0])
    plt.close()


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_mort_rates_data_2D(by_J):
    p = base_params
    mort_2D = rho_het.mean(axis=-1)
    fig = parameter_plots.plot_mort_rates_data(
        mort_2D, p.start_year, [p.start_year], by_J=by_J
    )
    lines = _lines(fig)
    assert len(lines) == 1
    assert _labels(fig) == ["Year " + str(p.start_year)]
    np.testing.assert_allclose(lines[0].get_ydata(), mort_2D[0])
    plt.close()


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_fert_rates_1D(by_J):
    S = base_params.S
    rng = np.random.default_rng(3)
    fert_list = [rng.uniform(size=S), rng.uniform(size=S)]
    fig = parameter_plots.plot_fert_rates(
        fert_list, labels=["2025", "2026"], by_J=by_J
    )
    lines = _lines(fig)
    assert len(lines) == 2
    assert _labels(fig) == ["2025", "2026"]
    np.testing.assert_allclose(lines[1].get_ydata(), fert_list[1])
    plt.close()


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_omega_fixed_1D(by_J):
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    omega_SS = base_params.omega_SS.sum(axis=-1)
    fig = parameter_plots.plot_omega_fixed(
        age_per_EpS, omega_SS, omega_SS, 0, S, by_J=by_J
    )
    lines = _lines(fig)
    assert len(lines) == 2
    assert _labels(fig) == ["Original Dist'n", "Fixed Dist'n"]
    np.testing.assert_allclose(lines[1].get_ydata(), omega_SS)
    plt.close()


@pytest.mark.parametrize("by_J", [False, True], ids=["avg", "by_J"])
def test_plot_imm_fixed_1D(by_J):
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    imm_orig = imm_het[0].mean(axis=-1)
    imm_adj = imm_het[-1].mean(axis=-1)
    fig = parameter_plots.plot_imm_fixed(
        age_per_EpS, imm_orig, imm_adj, 0, S, by_J=by_J
    )
    lines = _lines(fig)
    assert len(lines) == 2
    assert _labels(fig) == ["Original Imm. Rates", "Adj. Imm. Rates"]
    np.testing.assert_allclose(lines[1].get_ydata(), imm_adj)
    plt.close()


def _bad_dims_imm_rates():
    parameter_plots.plot_imm_rates(
        imm_het[0, :, 0], base_params.start_year, [base_params.start_year]
    )


def _bad_dims_imm_rates_omega():
    # omega must have the same number of dimensions as the rates
    parameter_plots.plot_imm_rates(
        imm_het.mean(axis=-1),
        base_params.start_year,
        [base_params.start_year],
        omega=base_params.omega,
    )


def _bad_dims_mort_rates():
    p = copy.deepcopy(base_params)
    p.rho = base_params.rho[0, :, 0]
    parameter_plots.plot_mort_rates([p])


def _bad_dims_mort_rates_omega():
    p = copy.deepcopy(base_params)
    p.omega = base_params.omega[:, :, 0, None, None]
    parameter_plots.plot_mort_rates([p])


def _bad_dims_population():
    p = copy.deepcopy(base_params)
    p.omega_SS = base_params.omega_SS[:, :, None]
    parameter_plots.plot_population(p)


def _bad_dims_mort_rates_data():
    parameter_plots.plot_mort_rates_data(
        rho_het[:, :, :, None], base_params.start_year
    )


def _bad_dims_fert_rates():
    parameter_plots.plot_fert_rates([np.ones((2, 2, 2))], labels=["a"])


def _bad_dims_fert_rates_omega():
    parameter_plots.plot_fert_rates(
        [np.ones(base_params.S)],
        labels=["a"],
        omega_list=[base_params.omega_SS],
    )


def _bad_dims_omega_fixed():
    S = base_params.S
    parameter_plots.plot_omega_fixed(
        np.arange(S), base_params.omega[:2], base_params.omega_SS, 0, S
    )


def _bad_dims_imm_fixed():
    S = base_params.S
    parameter_plots.plot_imm_fixed(np.arange(S), imm_het[0], imm_het[:2], 0, S)


def _bad_dims_population_path():
    S = base_params.S
    parameter_plots.plot_population_path(
        np.arange(S),
        base_params.omega_SS.sum(axis=-1),
        base_params.omega_SS,
        base_params.start_year,
        base_params.start_year,
        base_params.start_year,
        S,
    )


@pytest.mark.parametrize(
    "bad_call",
    [
        _bad_dims_imm_rates,
        _bad_dims_imm_rates_omega,
        _bad_dims_mort_rates,
        _bad_dims_mort_rates_omega,
        _bad_dims_population,
        _bad_dims_mort_rates_data,
        _bad_dims_fert_rates,
        _bad_dims_fert_rates_omega,
        _bad_dims_omega_fixed,
        _bad_dims_imm_fixed,
        _bad_dims_population_path,
    ],
    ids=[
        "imm_rates",
        "imm_rates_omega",
        "mort_rates",
        "mort_rates_omega",
        "population",
        "mort_rates_data",
        "fert_rates",
        "fert_rates_omega",
        "omega_fixed",
        "imm_fixed",
        "population_path",
    ],
)
def test_bad_dimensions_raise(bad_call):
    with pytest.raises(AssertionError, match="must have"):
        bad_call()
    plt.close("all")


# TODO:
# gen_3Dscatters_hist -- requires microdata df
# txfunc_graph - require micro data df
# txfunc_sse_plot


def test_plot_income_data():
    p = Specifications()
    ages = np.linspace(20 + 0.5, 100 - 0.5, 80)
    abil_midp = np.array([0.125, 0.375, 0.6, 0.75, 0.85, 0.945, 0.995])
    abil_pcts = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
    emat = p.e
    fig = parameter_plots.plot_income_data(ages, abil_midp, abil_pcts, emat)

    assert fig
    plt.close()


def test_plot_income_data_save_fig(tmpdir):
    p = Specifications()
    ages = np.linspace(20 + 0.5, 100 - 0.5, 80)
    abil_midp = np.array([0.125, 0.375, 0.6, 0.75, 0.85, 0.945, 0.995])
    abil_pcts = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
    emat = p.e
    parameter_plots.plot_income_data(
        ages, abil_midp, abil_pcts, emat, path=tmpdir
    )
    img1 = mpimg.imread(os.path.join(tmpdir, "ability_3D_lev.png"))
    img2 = mpimg.imread(os.path.join(tmpdir, "ability_3D_log.png"))
    img3 = mpimg.imread(os.path.join(tmpdir, "ability_2D_log.png"))

    assert isinstance(img1, np.ndarray)
    assert isinstance(img2, np.ndarray)
    assert isinstance(img3, np.ndarray)


if sys.version_info[1] < 11:
    test_list = [
        (base_taxfunctions, 43, "DEP", "etr", True, None, None),
        (base_taxfunctions, 43, "DEP", "etr", False, None, "Test title"),
        (GS_nonage_spec_taxfunctions, None, "GS", "etr", True, None, None),
        (base_taxfunctions, 43, "DEP", "etr", True, [micro_data], None),
        (base_taxfunctions, 43, "DEP", "mtry", True, [micro_data], None),
        (base_taxfunctions, 43, "DEP", "mtrx", True, [micro_data], None),
        (mono_nonage_spec_taxfunctions, None, "mono", "etr", True, None, None),
    ]
    id_list = [
        "over_labinc=True",
        "over_labinc=False",
        "Non age-specific",
        "with data",
        "MTR capital income",
        "MTR labor income",
        "Mono functions",
    ]
else:
    test_list = [
        (base_taxfunctions, 43, "DEP", "etr", True, None, None),
        (base_taxfunctions, 43, "DEP", "etr", False, None, "Test title"),
        (GS_nonage_spec_taxfunctions, None, "GS", "etr", True, None, None),
        (base_taxfunctions, 43, "DEP", "etr", True, [micro_data], None),
        (base_taxfunctions, 43, "DEP", "mtry", True, [micro_data], None),
        (base_taxfunctions, 43, "DEP", "mtrx", True, [micro_data], None),
    ]
    id_list = [
        "over_labinc=True",
        "over_labinc=False",
        "Non age-specific",
        "with data",
        "MTR capital income",
        "MTR labor income",
    ]


@pytest.mark.parametrize(
    "tax_funcs,age,tax_func_type,rate_type,over_labinc,data,title",
    test_list,
    ids=id_list,
)
def test_plot_2D_taxfunc(
    tax_funcs, age, tax_func_type, rate_type, over_labinc, data, title
):
    """
    Test of plot_2D_taxfunc
    """
    if sys.version_info[1] < 11:
        fig = parameter_plots.plot_2D_taxfunc(
            2030,
            2021,
            [tax_funcs],
            age=age,
            tax_func_type=[tax_func_type],
            rate_type=rate_type,
            over_labinc=over_labinc,
            data_list=data,
            title=title,
        )

        assert fig
        plt.close()
    else:
        assert True


def test_plot_2D_taxfunc_save_fig(tmpdir):
    """
    Test of plot_2D_taxfunc saving figures to disk
    """
    path_to_save = os.path.join(tmpdir, "plot_save_file.png")
    parameter_plots.plot_2D_taxfunc(
        2022, 2021, [base_taxfunctions], age=43, path=path_to_save
    )
    img1 = mpimg.imread(path_to_save)

    assert isinstance(img1, np.ndarray)
