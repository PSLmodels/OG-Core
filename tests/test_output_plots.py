"""
Tests of output_plots.py module
"""

import pytest
import os
import sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ogcore import utils, output_plots, constants


# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
base_ss = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "SS_vars_baseline.pkl")
)
base_tpi = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TPI_vars_baseline.pkl")
)
if sys.version_info[1] < 11:
    base_params = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "model_params_baseline.pkl")
    )
elif sys.version_info[1] == 11:
    base_params = utils.safe_read_pickle(
        os.path.join(
            CUR_PATH, "test_io_data", "model_params_baseline_v311.pkl"
        )
    )
else:
    base_params = utils.safe_read_pickle(
        os.path.join(
            CUR_PATH, "test_io_data", "model_params_baseline_v312.pkl"
        )
    )
reform_ss = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "SS_vars_reform.pkl")
)
reform_tpi = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TPI_vars_reform.pkl")
)
if sys.version_info[1] < 11:
    reform_params = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "model_params_reform.pkl")
    )
elif sys.version_info[1] == 11:
    reform_params = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "model_params_reform_v311.pkl")
    )
else:
    reform_params = utils.safe_read_pickle(
        os.path.join(
            CUR_PATH, "test_io_data", "model_params_baseline_v312.pkl"
        )
    )
reform_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_reform.pkl")
)


test_data = [
    (
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        "pct_diff",
        False,
        None,
        None,
    ),
    (
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        "diff",
        False,
        None,
        None,
    ),
    (
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        "forecast",
        False,
        None,
        None,
    ),
    (
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        "levels",
        False,
        None,
        None,
    ),
    (base_tpi, base_params, None, None, "levels", False, None, None),
    (base_tpi, base_params, None, None, "levels", False, [2040, 2060], None),
    (
        base_tpi,
        base_params,
        None,
        None,
        "levels",
        False,
        None,
        "Test plot title",
    ),
    (
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        "pct_diff",
        True,
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    "base_tpi,base_params,reform_tpi,reform_parms,plot_type,"
    + "stationarized,vertical_line_years,plot_title",
    test_data,
    ids=[
        "Pct Diff",
        "Diff",
        "Forecast",
        "Levels w reform",
        "Levels w/o reform",
        "Vertical line included",
        "Plot title included",
        "Stationarized pct diff",
    ],
)
def test_plot_aggregates(
    base_tpi,
    base_params,
    reform_tpi,
    reform_parms,
    plot_type,
    stationarized,
    vertical_line_years,
    plot_title,
):
    fig = output_plots.plot_aggregates(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "r"],
        plot_type=plot_type,
        stationarized=stationarized,
        num_years_to_plot=20,
        start_year=int(base_params.start_year),
        forecast_data=np.ones(20),
        forecast_units="ones",
        vertical_line_years=vertical_line_years,
        plot_title=plot_title,
    )
    assert fig
    plt.close()


test_data = [
    (base_tpi, base_params, reform_tpi, reform_params, "pct_diff", None, None),
    (base_tpi, base_params, reform_tpi, reform_params, "diff", None, None),
    (base_tpi, base_params, reform_tpi, reform_params, "forecast", None, None),
    (base_tpi, base_params, reform_tpi, reform_params, "levels", None, None),
    (base_tpi, base_params, None, None, "levels", None, None),
    (base_tpi, base_params, None, None, "levels", [2040, 2060], None),
    (base_tpi, base_params, None, None, "levels", None, "Test plot title"),
]


@pytest.mark.parametrize(
    "base_tpi,base_params,reform_tpi,reform_parms,plot_type,"
    + "vertical_line_years,plot_title",
    test_data,
    ids=[
        "Pct Diff",
        "Diff",
        "Forecast",
        "Levels w reform",
        "Levels w/o reform",
        "Vertical line included",
        "Plot title included",
    ],
)
def test_plot_industry_aggregates(
    base_tpi,
    base_params,
    reform_tpi,
    reform_parms,
    plot_type,
    vertical_line_years,
    plot_title,
):
    fig = output_plots.plot_industry_aggregates(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y_m"],
        plot_type=plot_type,
        num_years_to_plot=20,
        start_year=int(base_params.start_year),
        forecast_data=np.ones(20),
        forecast_units="ones",
        vertical_line_years=vertical_line_years,
        plot_title=plot_title,
    )
    assert fig
    plt.close()


test_data = [
    (base_tpi, base_params, None, None, None, None),
    (base_tpi, base_params, reform_tpi, reform_params, None, None),
    (base_tpi, base_params, reform_tpi, reform_params, [2040, 2060], None),
    (base_tpi, base_params, None, None, None, "Test plot title"),
]


def test_plot_aggregates_save_fig(tmpdir):
    path = os.path.join(tmpdir, "test_plot.png")
    output_plots.plot_aggregates(
        base_tpi,
        base_params,
        start_year=int(base_params.start_year),
        plot_type="levels",
        path=path,
    )
    img = mpimg.imread(path)

    assert isinstance(img, np.ndarray)


def test_plot_aggregates_not_a_type(tmpdir):
    with pytest.raises(AssertionError):
        output_plots.plot_aggregates(
            base_tpi,
            base_params,
            start_year=int(base_params.start_year),
            plot_type="levels2",
        )


test_data = [
    (base_tpi, base_params, None, None, None, None, "levels"),
    (base_tpi, base_params, reform_tpi, reform_params, None, None, "levels"),
    (base_tpi, base_params, reform_tpi, reform_params, None, None, "diffs"),
    (
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        [2040, 2060],
        None,
        "levels",
    ),
    (base_tpi, base_params, None, None, None, "Test plot title", "levels"),
]


@pytest.mark.parametrize(
    "base_tpi,base_params,reform_tpi,reform_params,"
    + "vertical_line_years,plot_title,plot_type",
    test_data,
    ids=[
        "No reform",
        "With reform",
        "Differences",
        "Vertical line included",
        "Plot title included",
    ],
)
def test_plot_gdp_ratio(
    base_tpi,
    base_params,
    reform_tpi,
    reform_params,
    vertical_line_years,
    plot_title,
    plot_type,
):
    fig = output_plots.plot_gdp_ratio(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        start_year=int(base_params.start_year),
        plot_type=plot_type,
        vertical_line_years=vertical_line_years,
        plot_title=plot_title,
    )
    assert fig
    plt.close()


def test_plot_gdp_ratio_save_fig(tmpdir):
    path = os.path.join(tmpdir, "test_plot.png")
    output_plots.plot_aggregates(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        start_year=int(base_params.start_year),
        reform_params=reform_params,
        path=path,
    )
    img = mpimg.imread(path)

    assert isinstance(img, np.ndarray)


def test_ability_bar():
    fig = output_plots.ability_bar(
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        start_year=int(base_params.start_year),
        plot_title=" Test Plot Title",
    )
    assert fig
    plt.close()


def test_ability_bar_save_fig(tmpdir):
    path = os.path.join(tmpdir, "test_plot.png")
    output_plots.ability_bar(
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        start_year=int(base_params.start_year),
        path=path,
    )
    img = mpimg.imread(path)

    assert isinstance(img, np.ndarray)


def test_ability_bar_ss():
    fig = output_plots.ability_bar_ss(
        base_ss,
        base_params,
        reform_ss,
        reform_params,
        plot_title=" Test Plot Title",
    )
    assert fig
    plt.close()


data_for_plot = np.ones(80) * 0.3


@pytest.mark.parametrize(
    "by_j,plot_data",
    [(True, None), (False, None), (False, data_for_plot)],
    ids=["By j", "Not by j", "Plot data"],
)
def test_ss_profiles(by_j, plot_data):
    fig = output_plots.ss_profiles(
        base_ss,
        base_params,
        reform_ss,
        reform_params,
        by_j=by_j,
        plot_data=plot_data,
        plot_title=" Test Plot Title",
    )
    assert fig
    plt.close()


def test_ss_profiles_save_fig(tmpdir):
    path = os.path.join(tmpdir, "test_plot.png")
    output_plots.ss_profiles(
        base_ss, base_params, reform_ss, reform_params, path=path
    )
    img = mpimg.imread(path)

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize("by_j", [True, False], ids=["By j", "Not by j"])
def test_tpi_profiles(by_j):
    fig = output_plots.tpi_profiles(
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        start_year=int(base_params.start_year),
        by_j=by_j,
        plot_title=" Test Plot Title",
    )
    assert fig
    plt.close()


test_data = [
    (base_params, base_ss, None, None, "levels", None),
    (base_params, base_ss, reform_params, reform_ss, "levels", None),
    (base_params, base_ss, reform_params, reform_ss, "diff", None),
    (base_params, base_ss, reform_params, reform_ss, "pct_diff", None),
    (
        base_params,
        base_ss,
        reform_params,
        reform_ss,
        "pct_diff",
        "Test Plot Title",
    ),
]


def test_tpi_profiles_save_fig(tmpdir):
    path = os.path.join(tmpdir, "test_plot.png")
    output_plots.tpi_profiles(
        base_tpi,
        base_params,
        reform_tpi,
        reform_params,
        start_year=int(base_params.start_year),
        path=path,
    )
    img = mpimg.imread(path)

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize(
    "base_params,base_ss,reform_params,reform_ss,plot_type,plot_title",
    test_data,
    ids=[
        "Levels",
        "Levels w/ reform",
        "Differences",
        "Pct Diffs",
        "Plot title included",
    ],
)
def test_ss_3Dplot(
    base_params, base_ss, reform_params, reform_ss, plot_type, plot_title
):
    fig = output_plots.ss_3Dplot(
        base_params,
        base_ss,
        reform_params=reform_params,
        reform_ss=reform_ss,
        plot_type=plot_type,
        plot_title=plot_title,
    )
    assert fig
    plt.close()


def test_ss_3Dplot_save_fig(tmpdir):
    path = os.path.join(tmpdir, "test_plot.png")
    output_plots.ss_3Dplot(
        base_params,
        base_ss,
        reform_params=reform_params,
        reform_ss=reform_ss,
        path=path,
    )
    img = mpimg.imread(path)

    assert isinstance(img, np.ndarray)


@pytest.mark.parametrize(
    "base_tpi,base_params,reform_tpi, reform_params,ineq_measure,"
    + "pctiles,plot_type",
    [
        (base_tpi, base_params, None, None, "gini", None, "levels"),
        (
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            "gini",
            None,
            "levels",
        ),
        (
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            "var_of_logs",
            None,
            "diff",
        ),
        (
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            "pct_ratio",
            (0.9, 0.1),
            "levels",
        ),
        (
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            "top_share",
            (0.01),
            "pct_diff",
        ),
    ],
    ids=[
        "Just baseline",
        "Baseline + Reform",
        "Base + Refore, var logs, diff",
        "Base + Refore, pct ratios",
        "Base + Refore, top share, pct diff",
    ],
)
def test_inequality_plot(
    base_tpi,
    base_params,
    reform_tpi,
    reform_params,
    ineq_measure,
    pctiles,
    plot_type,
):
    fig = output_plots.inequality_plot(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        start_year=int(base_params.start_year),
        ineq_measure=ineq_measure,
        pctiles=pctiles,
        plot_type=plot_type,
    )
    assert fig
    plt.close()


def test_inequality_plot_save_fig(tmpdir):
    path = os.path.join(tmpdir, "test_plot.png")
    output_plots.inequality_plot(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        start_year=int(base_params.start_year),
        path=path,
    )
    img = mpimg.imread(path)

    assert isinstance(img, np.ndarray)


def test_plot_all(tmpdir):
    if sys.version_info[1] < 11:
        base_output_path = os.path.join(CUR_PATH, "test_io_data", "OUTPUT")
        reform_output_path = os.path.join(CUR_PATH, "test_io_data", "OUTPUT")
        output_plots.plot_all(base_output_path, reform_output_path, tmpdir)
        img1 = mpimg.imread(os.path.join(tmpdir, "MacroAgg_PctChange.png"))
        img2 = mpimg.imread(
            os.path.join(tmpdir, "SSLifecycleProfile_Cons_Reform.png")
        )
        img3 = mpimg.imread(
            os.path.join(tmpdir, "SSLifecycleProfile_Save_Reform.png")
        )

        assert isinstance(img1, np.ndarray)
        assert isinstance(img2, np.ndarray)
        assert isinstance(img3, np.ndarray)
    else:
        assert True


def test_lambda_labels():
    """
    Test of the lambda_labels function in output_plots.py
    """
    labels = output_plots.lambda_labels(
        [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]
    )
    print(labels)
    for k, v in labels.items():
        print(k, v)
        assert v == constants.GROUP_LABELS[7][k]
