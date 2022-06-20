"""
Tests of parameter_plots.py module
"""

from tracemalloc import start
import pytest
import os
import numpy as np
import scipy.interpolate as si
import matplotlib.image as mpimg
from ogcore import utils, parameter_plots, Specifications


# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
base_params = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "model_params_baseline.pkl")
)
base_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_baseline.pkl")
)
GS_nonage_spec_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_GS_nonage.pkl")
)
micro_data = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
)


def test_plot_imm_rates():
    fig = parameter_plots.plot_imm_rates(base_params, include_title=True)
    assert fig


def test_plot_imm_rates_save_fig(tmpdir):
    parameter_plots.plot_imm_rates(base_params, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "imm_rates_orig.png"))

    assert isinstance(img, np.ndarray)


def test_plot_mort_rates():
    fig = parameter_plots.plot_mort_rates(base_params, include_title=True)
    assert fig


def test_plot_mort_rates_save_fig(tmpdir):
    parameter_plots.plot_mort_rates(base_params, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "mortality_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_pop_growth():
    fig = parameter_plots.plot_pop_growth(
        base_params, start_year=2023, include_title=True
    )
    assert fig


def test_plot_pop_growth_rates_save_fig(tmpdir):
    parameter_plots.plot_pop_growth(base_params, start_year=2023, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "pop_growth_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_ability_profiles():
    fig = parameter_plots.plot_ability_profiles(
        base_params, include_title=True
    )
    assert fig


def test_plot_ability_profiles_save_fig(tmpdir):
    parameter_plots.plot_ability_profiles(base_params, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "ability_profiles.png"))

    assert isinstance(img, np.ndarray)


def test_plot_elliptical_u():
    fig1 = parameter_plots.plot_elliptical_u(base_params, include_title=True)
    fig2 = parameter_plots.plot_elliptical_u(
        base_params, plot_MU=False, include_title=True
    )
    assert fig1
    assert fig2


def test_plot_elliptical_u_save_fig(tmpdir):
    parameter_plots.plot_elliptical_u(base_params, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "ellipse_v_CFE.png"))

    assert isinstance(img, np.ndarray)


def test_plot_chi_n():
    fig = parameter_plots.plot_chi_n(base_params, include_title=True)
    assert fig


def test_plot_chi_n_save_fig(tmpdir):
    parameter_plots.plot_chi_n(base_params, path=tmpdir)
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


def test_plot_population_save_fig(tmpdir):
    parameter_plots.plot_population(base_params, path=tmpdir)
    img = mpimg.imread(os.path.join(tmpdir, "pop_distribution.png"))

    assert isinstance(img, np.ndarray)


def test_plot_fert_rates():
    totpers = base_params.S
    min_yr = 20
    max_yr = 100
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
    fig = parameter_plots.plot_fert_rates(
        fert_func, age_midp, totpers, min_yr, max_yr, fert_data, fert_rates
    )
    assert fig


def test_plot_fert_rates_save_fig(tmpdir):
    totpers = base_params.S
    min_yr = 20
    max_yr = 100
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
        fert_func,
        age_midp,
        totpers,
        min_yr,
        max_yr,
        fert_data,
        fert_rates,
        output_dir=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "fert_rates.png"))

    assert isinstance(img, np.ndarray)


def test_plot_mort_rates_data():
    totpers = base_params.S - 1
    min_yr = 21
    max_yr = 100
    age_year_all = np.arange(min_yr, max_yr)
    mort_rates = base_params.rho[1:]
    mort_rates_all = base_params.rho[1:]
    infmort_rate = base_params.rho[0]
    fig = parameter_plots.plot_mort_rates_data(
        totpers,
        min_yr,
        max_yr,
        age_year_all,
        mort_rates_all,
        infmort_rate,
        mort_rates,
        output_dir=None,
    )
    assert fig


def test_plot_mort_rates_data_save_fig(tmpdir):
    totpers = base_params.S - 1
    min_yr = 21
    max_yr = 100
    age_year_all = np.arange(min_yr, max_yr)
    mort_rates = base_params.rho[1:]
    mort_rates_all = base_params.rho[1:]
    infmort_rate = base_params.rho[0]
    parameter_plots.plot_mort_rates_data(
        totpers,
        min_yr,
        max_yr,
        age_year_all,
        mort_rates_all,
        infmort_rate,
        mort_rates,
        output_dir=tmpdir,
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


def test_plot_omega_fixed_save_fig(tmpdir):
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    omega_SS_orig = base_params.omega_SS
    omega_SSfx = base_params.omega_SS
    parameter_plots.plot_omega_fixed(
        age_per_EpS, omega_SS_orig, omega_SSfx, E, S, output_dir=tmpdir
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


def test_plot_imm_fixed_save_fig(tmpdir):
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    imm_rates_orig = base_params.imm_rates[0, :]
    imm_rates_adj = base_params.imm_rates[-1, :]
    parameter_plots.plot_imm_fixed(
        age_per_EpS, imm_rates_orig, imm_rates_adj, E, S, output_dir=tmpdir
    )
    img = mpimg.imread(os.path.join(tmpdir, "OrigVsAdjImm.png"))

    assert isinstance(img, np.ndarray)


def test_plot_population_path():
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    pop_2013_pct = base_params.omega[0, :]
    omega_path_lev = base_params.omega.T
    omega_SSfx = base_params.omega_SS
    curr_year = base_params.start_year
    fig = parameter_plots.plot_population_path(
        age_per_EpS, pop_2013_pct, omega_path_lev, omega_SSfx, curr_year, E, S
    )
    assert fig


def test_plot_population_path_save_fig(tmpdir):
    E = 0
    S = base_params.S
    age_per_EpS = np.arange(21, S + 21)
    pop_2013_pct = base_params.omega[0, :]
    omega_path_lev = base_params.omega.T
    omega_SSfx = base_params.omega_SS
    curr_year = base_params.start_year
    parameter_plots.plot_population_path(
        age_per_EpS,
        pop_2013_pct,
        omega_path_lev,
        omega_SSfx,
        curr_year,
        E,
        S,
        output_dir=tmpdir,
    )
    img = mpimg.imread(os.path.join(tmpdir, "PopDistPath.png"))

    assert isinstance(img, np.ndarray)


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


def test_plot_income_data_save_fig(tmpdir):
    p = Specifications()
    ages = np.linspace(20 + 0.5, 100 - 0.5, 80)
    abil_midp = np.array([0.125, 0.375, 0.6, 0.75, 0.85, 0.945, 0.995])
    abil_pcts = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])
    emat = p.e
    parameter_plots.plot_income_data(
        ages, abil_midp, abil_pcts, emat, output_dir=tmpdir
    )
    img1 = mpimg.imread(os.path.join(tmpdir, "ability_3D_lev.png"))
    img2 = mpimg.imread(os.path.join(tmpdir, "ability_3D_log.png"))
    img3 = mpimg.imread(os.path.join(tmpdir, "ability_2D_log.png"))

    assert isinstance(img1, np.ndarray)
    assert isinstance(img2, np.ndarray)
    assert isinstance(img3, np.ndarray)


@pytest.mark.parametrize(
    "tax_funcs,age,tax_func_type,rate_type,over_labinc,data,title",
    [
        (base_taxfunctions, 43, "DEP", "etr", True, None, None),
        (base_taxfunctions, 43, "DEP", "etr", False, None, "Test title"),
        (GS_nonage_spec_taxfunctions, None, "GS", "etr", True, None, None),
        (base_taxfunctions, 43, "DEP", "etr", True, [micro_data], None),
        (base_taxfunctions, 43, "DEP", "mtry", True, [micro_data], None),
        (base_taxfunctions, 43, "DEP", "mtrx", True, [micro_data], None),
    ],
    ids=[
        "over_labinc=True",
        "over_labinc=False",
        "Non age-specific",
        "with data",
        "MTR capital income",
        "MTR labor income",
    ],
)
def test_plot_2D_taxfunc(
    tax_funcs, age, tax_func_type, rate_type, over_labinc, data, title
):
    """
    Test of plot_2D_taxfunc
    """
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
