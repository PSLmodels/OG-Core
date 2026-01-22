"""
Tests of parameter_table.py module
"""

import pytest
import os
import numpy as np
from ogcore import utils, parameter_tables
from ogcore.parameters import Specifications

# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

base_ss = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "SS_vars_baseline.pkl")
)
base_tpi = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TPI_vars_baseline.pkl")
)
base_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_baseline.pkl")
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


@pytest.mark.parametrize(
    "rate_type",
    ["ETR", "MTRx", "MTRy", "all"],
    ids=["ETR", "MTRx", "MTRy", "All rates"],
)
def test_tax_rate_table(rate_type):
    str = parameter_tables.tax_rate_table(
        base_taxfunctions,
        base_params,
        reform_TxFuncEst=base_taxfunctions,
        reform_params=base_params,
        rate_type=rate_type,
    )
    assert str


def test_tax_rate_table_exception1():
    """
    Raise exception for not passing valid rate type
    """
    with pytest.raises(Exception):
        assert parameter_tables.tax_rate_table(
            base_taxfunctions, base_params, rate_type="not_valid_type"
        )


def test_tax_rate_table_exception2():
    """
    Raise exception for not passing valid rate type
    """
    with pytest.raises(Exception):
        assert parameter_tables.tax_rate_table(
            base_taxfunctions,
            base_params,
            reform_TxFuncEst=base_taxfunctions,
            reform_params=base_params,
            rate_type="not_valid_type",
        )


def test_param_table():
    p = Specifications()
    str = parameter_tables.param_table(p)
    assert str
