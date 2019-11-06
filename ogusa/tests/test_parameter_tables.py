'''
Tests of parameter_table.py module
'''

import pytest
import os
from ogusa import utils, parameter_tables


# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

base_params = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'model_params_baseline.pkl'))
base_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'TxFuncEst_baseline.pkl'))


def test_tax_rate_table():
    str = parameter_tables.tax_rate_table(base_taxfunctions, base_params)
    assert str


def test_param_table():
    str = parameter_tables.param_table(base_params)
    assert str
