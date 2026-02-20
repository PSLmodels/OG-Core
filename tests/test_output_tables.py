"""
Tests of output_tables.py module
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from ogcore import utils, output_tables
from ogcore.parameters import Specifications

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
# add investment tax credit parameter that not in cached parameters
base_params.inv_tax_credit = np.zeros(
    (base_params.T + base_params.S, base_params.M)
)
reform_params.inv_tax_credit = np.zeros(
    (reform_params.T + reform_params.S, reform_params.M)
)

test_data = [
    (base_tpi, base_params, reform_tpi, reform_params, "pct_diff", False),
    (base_tpi, base_params, reform_tpi, reform_params, "diff", False),
    (base_tpi, base_params, reform_tpi, reform_params, "levels", False),
    (base_tpi, base_params, reform_tpi, reform_params, "pct_diff", True),
]


@pytest.mark.parametrize(
    "base_tpi,base_params,reform_tpi,reform_params,output_type,stationarized",
    test_data,
    ids=["Pct Diff", "Diff", "Levels", "Unstationary pct diff"],
)
def test_macro_table(
    base_tpi,
    base_params,
    reform_tpi,
    reform_params,
    output_type,
    stationarized,
):
    df = output_tables.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        start_year=int(base_params.start_year),
        output_type=output_type,
        stationarized=stationarized,
        include_SS=True,
        include_overall=True,
    )
    assert isinstance(df, pd.DataFrame)


def test_macro_table_SS():
    df = output_tables.macro_table_SS(base_ss, reform_ss)
    assert isinstance(df, pd.DataFrame)


def test_ineq_table():
    df = output_tables.ineq_table(base_ss, base_params)
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    "base_ss,base_params,reform_ss,reform_params",
    [
        (base_ss, base_params, None, None),
        (base_ss, base_params, reform_ss, reform_params),
    ],
    ids=["Baseline only", "Base and reform"],
)
def test_gini_table(base_ss, base_params, reform_ss, reform_params):
    df = output_tables.gini_table(
        base_ss, base_params, reform_ss=reform_ss, reform_params=reform_params
    )
    assert isinstance(df, pd.DataFrame)


def test_wealth_moments_table():
    """
    Need SCF data which is too large to check into repo so this will
    be flagged so as to not run on TravisCI.
    """
    df = output_tables.wealth_moments_table(
        base_ss,
        base_params,
        data_moments=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 1.0, 2.0]),
    )
    assert isinstance(df, pd.DataFrame)


def test_time_series_table():
    df = output_tables.time_series_table(
        base_params,
        base_tpi,
        reform_params=reform_params,
        reform_tpi=reform_tpi,
    )
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    "include_business_tax,full_break_out",
    [(True, True), (True, False), (False, False), (False, True)],
    ids=[
        "Biz Tax and break out",
        "Biz tax, no break out",
        "No biz tax or break out",
        "No biz tax, break out",
    ],
)
def test_dynamic_revenue_decomposition(include_business_tax, full_break_out):
    base_params.labor_income_tax_noncompliance_rate = np.zeros(
        (base_params.T, base_params.J)
    )
    base_params.capital_income_tax_noncompliance_rate = np.zeros(
        (base_params.T, base_params.J)
    )
    base_params.income_tax_filer = np.ones((base_params.T, base_params.J))
    base_params.wealth_tax_filer = np.ones((base_params.T, base_params.J))
    reform_params.labor_income_tax_noncompliance_rate = np.zeros(
        (reform_params.T, reform_params.J)
    )
    reform_params.capital_income_tax_noncompliance_rate = np.zeros(
        (reform_params.T, reform_params.J)
    )
    reform_params.income_tax_filer = np.ones(
        (reform_params.T, reform_params.J)
    )
    reform_params.wealth_tax_filer = np.ones(
        (reform_params.T, reform_params.J)
    )
    base_params.tax_filer = np.ones(base_params.J)
    reform_params.tax_filer = np.ones(reform_params.J)
    # check if tax parameters are a numpy array
    # this is relevant for cached parameter arrays saved before
    # tax params were put in lists
    if isinstance(base_params.etr_params, np.ndarray):
        base_params.etr_params = base_params.etr_params.tolist()
    if isinstance(reform_params.etr_params, np.ndarray):
        reform_params.etr_params = reform_params.etr_params.tolist()
    print("M = ", base_params.M, reform_params.M)
    print(
        "Shape of M implied by output = ",
        base_tpi["p_m"].shape[1],
        reform_tpi["p_m"].shape[1],
    )
    df = output_tables.dynamic_revenue_decomposition(
        base_params,
        base_tpi,
        base_ss,
        reform_params,
        reform_tpi,
        reform_ss,
        start_year=int(base_params.start_year),
        include_business_tax=include_business_tax,
        full_break_out=full_break_out,
    )
    assert isinstance(df, pd.DataFrame)
