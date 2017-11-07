import pickle
from ogusa import txfunc

def test_cps_data():
    with open("../../regression/cps_test_replace_outliers.pkl", 'rb') as p:
        param_arr = pickle.load(p)
        sse_big_mat = pickle.load(p)

    txfunc.replace_outliers(param_arr, sse_big_mat)
