"""
Tests of parameter_table.py module
"""

import pytest
import os
import sys
from ogcore import utils, parameter_tables
from ogcore.parameters import Specifications


# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

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
base_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, "test_io_data", "TxFuncEst_baseline.pkl")
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
