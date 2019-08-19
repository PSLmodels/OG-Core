'''
Tests of parameter_plots.py module
'''

import pytest
import os
from ogusa import utils, parameter_plots


# Load in test results and parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
base_params = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'model_params_baseline.pkl'))


def test_plot_imm_rates():
    fig = parameter_plots.plot_imm_rates(base_params)
    assert fig


def test_plot_mort_rates():
    fig = parameter_plots.plot_mort_rates(base_params)
    assert fig


def test_plot_pop_growth():
    fig = parameter_plots.plot_pop_growth(base_params)
    assert fig


def test_plot_ability_profiles():
    fig = parameter_plots.plot_ability_profiles(base_params)
    assert fig


def test_plot_elliptical_u():
    fig = parameter_plots.plot_elliptical_u(base_params)
    assert fig
