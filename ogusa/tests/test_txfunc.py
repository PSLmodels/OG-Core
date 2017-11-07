from ogusa import txfunc

import numpy as np
import pickle
import os

CUR_PATH = os.path.abspath(os.path.dirname(__file__))

def test_replace_outliers():
    """
    4 cases:
        s is an outlier and is 0
        s is an outlier and is in the interior (s > 0 and s < S)
        s is not an outlier but the first s - 1 ages were (s = 1 in our case)
        s is an outlier and is the max age
    """
    S = 20
    BW = 2
    numparams = 5
    param_arr = np.random.rand(S * BW * numparams).reshape(S, BW, numparams)
    sse_big_mat = ~ np.ones((S, BW), dtype=bool)
    sse_big_mat[0, 0] = True
    sse_big_mat[1, 0] = True
    sse_big_mat[S-11, 0] = True
    sse_big_mat[S-10, 0] = True
    sse_big_mat[S - 2, 0] = True
    sse_big_mat[S - 1, 0] = True

    txfunc.replace_outliers(param_arr, sse_big_mat)
