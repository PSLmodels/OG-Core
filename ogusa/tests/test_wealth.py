'''
Tests of wealth.py module
'''

import pytest
import pandas as pd
import numpy as np
from ogusa import wealth


@pytest.mark.full_run
def test_get_wealth_data():
    '''
    Test of reading wealth data.

    Need SCF data which is too large to check into repo so this will
    be flagged so as to not run on TravisCI.
    '''
    df = wealth.get_wealth_data()

    assert isinstance(df, pd.DataFrame)


@pytest.mark.full_run
def test_compute_wealth_moments():
    '''
    Test of computation of wealth moments.

    Need SCF data which is too large to check into repo so this will
    be flagged so as to not run on TravisCI.
    '''
    expected_moments = np.array([
        -0.004314198, 0.019807056, 0.062817143, 0.0644862, 0.121936608,
        0.39036657, 0.344894446, 0.837576949, 4.76976036])
    df = wealth.get_wealth_data()
    test_moments = wealth.compute_wealth_moments(
        df, np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]))

    assert(np.allclose(expected_moments, test_moments))
