'''
Tests of wealth.py module
'''

# import pytest
import pandas as pd
import numpy as np
from ogusa import wealth


# @pytest.mark.full_run
def test_get_wealth_data():
    '''
    Test of reading wealth data.

    Need SCF data which is too large to check into repo so this will
    be flagged so as to not run on TravisCI.
    '''
    df = wealth.get_wealth_data()

    assert isinstance(df, pd.DataFrame)


# @pytest.mark.full_run
def test_compute_wealth_moments():
    '''
    Test of computation of wealth moments.

    Need SCF data which is too large to check into repo so this will
    be flagged so as to not run on TravisCI.
    '''
    expected_moments = np.array([
        -4.36938131e-03, 1.87063661e-02, 5.89720538e-02, 6.10665862e-02,
        1.17776715e-01, 3.87790368e-01, 3.60041151e-01, 8.45051216e-01,
        4.97530422e+00])
    df = wealth.get_wealth_data()
    test_moments = wealth.compute_wealth_moments(
        df, np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]))

    assert(np.allclose(expected_moments, test_moments, rtol=0.001))
