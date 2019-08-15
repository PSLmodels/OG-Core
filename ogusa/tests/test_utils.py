import pytest
from ogusa import utils
import numpy as np


def test_rate_conversion():
    '''
    test of utils.rate_conversion
    '''
    expected_rate = 0.3
    annual_rate = 0.3
    start_age = 20
    end_age = 80
    s = 60
    test_rate = utils.rate_conversion(annual_rate, start_age, end_age, s)
    assert(np.allclose(expected_rate, test_rate))
