'''
Tests of output_plots.py module
'''

import pytest
import os
from ogusa import utils, output_plots


# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
base_ss = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'SS_vars_baseline.pkl'))
base_tpi = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'TPI_vars_baseline.pkl'))
base_params = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'model_params_baseline.pkl'))
base_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'TxFuncEst_baseline.pkl'))
reform_ss = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'SS_vars_reform.pkl'))
reform_tpi = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'TPI_vars_reform.pkl'))
reform_params = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'model_params_reform.pkl'))
reform_taxfunctions = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'TxFuncEst_reform.pkl'))


def test_plot_aggregates():
    fig = output_plots.plot_aggregates(base_tpi, base_params,
                                       reform_tpi, reform_params)
    assert fig


def test_plot_gdp_ratio():
    fig = output_plots.plot_gdp_ratio(base_tpi, base_params)
    assert fig


def test_ability_bar():
    fig = output_plots.ability_bar(base_tpi, base_params, reform_tpi,
                                   reform_params)
    assert fig


def test_ss_profiles():
    fig = output_plots.ss_profiles(base_ss, base_params, reform_ss,
                                   reform_params)
    assert fig


def test_ss_3Dplot():
        fig = output_plots.ss_3Dplot(base_params, base_ss)
        assert fig
