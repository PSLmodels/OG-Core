from ogcore import txfunc
from distributed import Client, LocalCluster
import pytest
import pandas as pd
import numpy as np
import os
import pickle
import bz2
from ogcore import utils

NUM_WORKERS = 2
CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


# function to decompress pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data


@pytest.mark.parametrize(
    "tax_func_type,expected",
    [("DEP", 0.032749763), ("GS", 0.007952744)],
    ids=["DEP", "GS"],
)
def test_wsumsq(tax_func_type, expected):
    """
    Test of the weighted sum of squares calculation
    """
    rate_type = "etr"
    A = 0.01
    B = 0.02
    C = 0.1
    D = 1.1
    max_x = 0.55
    min_x = 0.17
    max_y = 0.46
    min_y = 0.04
    shift = 0.04
    share = 0.8
    phi0 = 0.396
    phi1 = 0.7
    phi2 = 0.9
    X = np.array([32.0, 44.0, 1.6, 0.4])
    Y = np.array([32.0, 55.0, 0.9, 0.03])
    txrates = np.array([0.6, 0.5, 0.3, 0.25])
    wgts = np.array([0.1, 0.25, 0.55, 0.1])
    if tax_func_type == "DEP":
        params = A, B, C, D, max_x, max_y, share
        args = (
            (min_x, min_y, shift_x, shift_y, shift),
            X,
            Y,
            txrates,
            wgts,
            tax_func_type,
            rate_type,
        )
    elif tax_func_type == "GS":
        params = phi0, phi1, phi2
        args = None, X, Y, txrates, wgts, tax_func_type, rate_type
    test_val = txfunc.wsumsq(params, *args)

    assert np.allclose(test_val, expected)


@pytest.mark.parametrize(
    "se_mult,expected_mat",
    [
        (2, np.array([[False, False], [False, False], [False, True]])),
        (8, np.array([[False, False], [False, False], [False, False]])),
    ],
    ids=["2", "8"],
)
def test_find_outliers(se_mult, expected_mat):
    # Test the find outliers function
    sse_mat = np.array([[21.0, 22.0], [20.0, 32.0], [20.0, 100.0]])
    age_vec = np.array([40, 41])
    start_year = 2018
    varstr = "MTRy"
    test_mat = txfunc.find_outliers(
        sse_mat, age_vec, se_mult, start_year, varstr, False
    )

    assert np.allclose(test_mat, expected_mat)


def test_replace_outliers():
    """
    4 cases:
        s is an outlier and is 0
        s is an outlier and is in the interior (s > 0 and s < S)
        s is not an outlier but the first s - 1 ages were (s = 1 in our case)
        s is an outlier and is the max age
    """
    S = 10
    BW = 2
    numparams = 3
    random_state = np.random.RandomState(10)
    param_arr = random_state.rand(S * BW * numparams).reshape(S, BW, numparams)
    sse_big_mat = ~np.ones((S, BW), dtype=bool)
    sse_big_mat[0, 0] = True
    sse_big_mat[1, 0] = True
    sse_big_mat[S - 4, 0] = True
    sse_big_mat[S - 5, 0] = True
    sse_big_mat[S - 2, 0] = True
    sse_big_mat[S - 1, 0] = True

    act = txfunc.replace_outliers(param_arr, sse_big_mat)

    exp = [
        [
            [0.00394827, 0.51219226, 0.81262096],
            [0.74880388, 0.49850701, 0.22479665],
        ],
        [
            [0.00394827, 0.51219226, 0.81262096],
            [0.08833981, 0.68535982, 0.95339335],
        ],
        [
            [0.00394827, 0.51219226, 0.81262096],
            [0.61252607, 0.72175532, 0.29187607],
        ],
        [
            [0.91777412, 0.71457578, 0.54254437],
            [0.14217005, 0.37334076, 0.67413362],
        ],
        [
            [0.44183317, 0.43401399, 0.61776698],
            [0.51313824, 0.65039718, 0.60103895],
        ],
        [
            [0.3608713, 0.57495943, 0.5290622],
            [0.31923609, 0.09045935, 0.30070006],
        ],
        [
            [0.27990942, 0.71590487, 0.44035742],
            [0.62628715, 0.54758616, 0.819287],
        ],
        [
            [0.19894754, 0.8568503, 0.35165264],
            [0.75464769, 0.29596171, 0.88393648],
        ],
        [
            [0.19894754, 0.8568503, 0.35165264],
            [0.09346037, 0.82110566, 0.15115202],
        ],
        [
            [0.19894754, 0.8568503, 0.35165264],
            [0.45630455, 0.82612284, 0.25137413],
        ],
    ]

    assert np.allclose(act, exp)


expected_tuple_DEP = (
    np.array(
        [
            2.27262567e-23,
            6.52118581e-05,
            2.58989255e-13,
            5.79350547e-09,
            3.37733466e-01,
            7.99995110e-01,
            9.14366888e-01,
            9.02760087e-06,
            9.02760087e-06,
            3.37724438e-03,
            7.99986083e-03,
            9.02760087e-06,
        ]
    ),
    237677.14110076256,
    152900,
)

expected_tuple_DEP_totalinc = (
    np.array(
        [
            2.37823197e-24,
            1.23804760e-05,
            2.98681155e-01,
            9.02760087e-06,
            2.98672127e-03,
            9.02760087e-06,
        ]
    ),
    256983.45682508417,
    152900,
)
expected_tuple_linear = (0.26135747, 0.0, 152900)
expected_tuple_GS = (
    np.array([0.42897179, 0.39679734, 0.00239574]),
    242750.4467703229,
    152900,
)
expected_tuple_linear_mtrx = (0.37030104, 0.0, 152900)
expected_tuple_linear_mtry = (0.24793767, 0.0, 152900)


@pytest.mark.local  # only marking as local because platform
# affects results from scipy.opt that is called in this test - so it'll
# pass if run on Mac with MKL, but not necessarily on other platforms
@pytest.mark.parametrize(
    "rate_type,tax_func_type,numparams,expected_tuple",
    [
        ("etr", "DEP", 12, expected_tuple_DEP),
        ("etr", "DEP_totalinc", 6, expected_tuple_DEP_totalinc),
        ("etr", "GS", 3, expected_tuple_GS),
    ],
    ids=["DEP", "DEP_totalinc", "GS"],
)
def test_txfunc_est(
    rate_type, tax_func_type, numparams, expected_tuple, tmpdir
):
    """
    Test txfunc.txfunc_est() function.  The test is that given
    inputs from previous run, the outputs are unchanged.
    """
    micro_data = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
    )
    s = 80
    t = 2030
    df = txfunc.tax_data_sample(micro_data[str(t)])
    output_dir = tmpdir
    # Put old df variables into new df var names
    df.rename(
        columns={
            "MTR labor income": "mtr_labinc",
            "MTR capital income": "mtr_capinc",
            "Total labor income": "total_labinc",
            "Total capital income": "total_capinc",
            "ETR": "etr",
            "expanded_income": "market_income",
            "Weights": "weight",
        },
        inplace=True,
    )
    test_tuple = txfunc.txfunc_est(
        df, s, t, rate_type, tax_func_type, numparams, output_dir, True
    )

    for i, v in enumerate(expected_tuple):
        assert np.allclose(test_tuple[i], v)


@pytest.mark.parametrize(
    "rate_type,tax_func_type,numparams,expected_tuple",
    [
        ("etr", "linear", 1, expected_tuple_linear),
        ("mtrx", "linear", 1, expected_tuple_linear_mtrx),
        ("mtry", "linear", 1, expected_tuple_linear_mtry),
    ],
    ids=["linear", "linear, mtrx", "linear, mtry"],
)
def test_txfunc_est_on_GH(
    rate_type, tax_func_type, numparams, expected_tuple, tmpdir
):
    """
    Test txfunc.txfunc_est() function.  The test is that given
    inputs from previous run, the outputs are unchanged.
    """
    micro_data = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
    )
    s = 80
    t = 2030
    df = txfunc.tax_data_sample(micro_data[str(t)])
    output_dir = tmpdir
    # Put old df variables into new df var names
    df.rename(
        columns={
            "MTR labor income": "mtr_labinc",
            "MTR capital income": "mtr_capinc",
            "Total labor income": "total_labinc",
            "Total capital income": "total_capinc",
            "ETR": "etr",
            "expanded_income": "market_income",
            "Weights": "weight",
        },
        inplace=True,
    )
    test_tuple = txfunc.txfunc_est(
        df, s, t, rate_type, tax_func_type, numparams, output_dir, True
    )

    for i, v in enumerate(expected_tuple):
        assert np.allclose(test_tuple[i], v, rtol=0.0, atol=1e-04)


def test_txfunc_est_exception(tmpdir):
    micro_data = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
    )
    s = 80
    t = 2030
    df = txfunc.tax_data_sample(micro_data[str(t)])
    output_dir = tmpdir
    df.rename(
        columns={
            "MTR labor income": "mtr_labinc",
            "MTR capital income": "mtr_capinc",
            "Total labor income": "total_labinc",
            "Total capital income": "total_capinc",
            "ETR": "etr",
            "expanded_income": "market_income",
            "Weights": "weight",
        },
        inplace=True,
    )
    with pytest.raises(RuntimeError) as excinfo:
        txfunc.txfunc_est(df, s, t, "etr", "NotAType", 12, output_dir, False)
        assert "Choice of tax function is not in the set" in str(excinfo.value)


def test_tax_data_sample():
    """
    Test of txfunc.tax_data_sample() function
    """
    data = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
    )
    df = txfunc.tax_data_sample(data["2030"])
    assert isinstance(df, pd.DataFrame)


@pytest.mark.local
# mark as local run since results work on Mac, but differ on other
# platforms
def test_tax_func_loop():
    """
    Test txfunc.tax_func_loop() function.  The test is that given
    inputs from previous run, the outputs are unchanged.

    Note that the data for this test is too large for GitHub, so it
    won't be available there.

    """
    input_tuple = decompress_pickle(
        os.path.join(
            CUR_PATH, "test_io_data", "tax_func_loop_inputs_large.pbz2"
        )
    )
    (
        t,
        micro_data,
        beg_yr,
        s_min,
        s_max,
        age_specific,
        analytical_mtrs,
        desc_data,
        graph_data,
        graph_est,
        output_dir,
        numparams,
        tpers,
    ) = input_tuple
    tax_func_type = "DEP"
    # Rename and create vars to suit new micro_data var names
    micro_data["total_labinc"] = (
        micro_data["Wage income"] + micro_data["SE income"]
    )
    micro_data["etr"] = (
        micro_data["Total tax liability"] / micro_data["Adjusted total income"]
    )
    micro_data["total_capinc"] = (
        micro_data["Adjusted total income"] - micro_data["total_labinc"]
    )
    # use weighted avg for MTR labor - abs value because
    # SE income may be negative
    micro_data["mtr_labinc"] = micro_data["MTR wage income"] * (
        micro_data["Wage income"]
        / (micro_data["Wage income"].abs() + micro_data["SE income"].abs())
    ) + micro_data["MTR SE income"] * (
        micro_data["SE income"].abs()
        / (micro_data["Wage income"].abs() + micro_data["SE income"].abs())
    )
    micro_data.rename(
        columns={
            "Adjusted total income": "market_income",
            "MTR capital income": "mtr_capinc",
            "Total tax liability": "total_tax_liab",
            "Year": "year",
            "Age": "age",
            "expanded_income": "market_income",
            "Weights": "weight",
        },
        inplace=True,
    )
    micro_data["payroll_tax_liab"] = 0
    test_tuple = txfunc.tax_func_loop(
        t,
        micro_data,
        beg_yr,
        s_min,
        s_max,
        age_specific,
        tax_func_type,
        analytical_mtrs,
        desc_data,
        graph_data,
        graph_est,
        output_dir,
        numparams,
    )

    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "tax_func_loop_outputs.pkl")
    )

    for i, v in enumerate(expected_tuple):
        print("diff = ", np.absolute(test_tuple[i] - v).max())
        assert np.allclose(test_tuple[i], v, atol=1e-06)


A = 0.02
B = 0.01
C = 0.003
D = 3.2
max_x = 0.6
min_x = 0.05
max_y = 0.8
min_y = 0.05
shift = 0.03
share = 0.7
phi0 = 0.6
phi1 = 0.5
phi2 = 0.6
shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
avg_rate = 0.17


@pytest.mark.parametrize(
    "tax_func_type,rate_type,params,analytical_mtrs,mtr_capital,"
    + "for_estimation,expected",
    [
        (
            "DEP",
            "etr",
            np.array(
                [
                    A,
                    B,
                    C,
                    D,
                    max_x,
                    max_y,
                    share,
                    min_x,
                    min_y,
                    shift_x,
                    shift_y,
                    shift,
                ]
            ),
            False,
            False,
            True,
            np.array([0.1894527, 0.216354953, 0.107391574, 0.087371974]),
        ),
        (
            "DEP",
            "etr",
            np.array(
                [
                    A,
                    B,
                    C,
                    D,
                    max_x,
                    max_y,
                    share,
                    min_x,
                    min_y,
                    shift_x,
                    shift_y,
                    shift,
                ]
            ),
            False,
            False,
            False,
            np.array([0.669061481, 0.678657921, 0.190301075, 0.103958946]),
        ),
        (
            "GS",
            "etr",
            np.array([phi0, phi1, phi2]),
            False,
            False,
            False,
            np.array([0.58216409, 0.5876492, 0.441995766, 0.290991255]),
        ),
        (
            "GS",
            "mtrx",
            np.array([phi0, phi1, phi2]),
            False,
            False,
            False,
            np.array([0.596924843, 0.598227987, 0.518917438, 0.37824137]),
        ),
        (
            "DEP_totalinc",
            "etr",
            np.array([A, B, max_x, min_x, shift_x, shift]),
            False,
            False,
            True,
            np.array([0.110821747, 0.134980034, 0.085945843, 0.085573318]),
        ),
        (
            "DEP_totalinc",
            "etr",
            np.array([A, B, max_x, min_x, shift_x, shift]),
            False,
            False,
            False,
            np.array([0.628917903, 0.632722363, 0.15723913, 0.089863997]),
        ),
        (
            "linear",
            "etr",
            np.array([avg_rate]),
            False,
            False,
            False,
            np.array([0.17]),
        ),
        (
            "DEP",
            "mtr",
            np.array(
                [
                    A,
                    B,
                    C,
                    D,
                    max_x,
                    max_y,
                    share,
                    min_x,
                    min_y,
                    shift_x,
                    shift_y,
                    shift,
                ]
            ),
            True,
            False,
            False,
            np.array([0.7427211, 0.72450578, 0.30152417, 0.10923907]),
        ),
        (
            "DEP",
            "mtr",
            np.array(
                [
                    A,
                    B,
                    C,
                    D,
                    max_x,
                    max_y,
                    share,
                    min_x,
                    min_y,
                    shift_x,
                    shift_y,
                    shift,
                ]
            ),
            True,
            True,
            False,
            np.array([0.67250144, 0.68049134, 0.22151424, 0.25869779]),
        ),
        (
            "DEP_totalinc",
            "etr",
            np.array([A, B, max_x, min_x, shift_x, shift]),
            True,
            False,
            False,
            np.array([0.64187414, 0.63823569, 0.27160586, 0.09619512]),
        ),
    ],
    ids=[
        "DEP for estimation",
        "DEP not for estimation",
        "GS, etr",
        "GS, mtr",
        "DEP_totalinc for estimation",
        "DEP_totalinc not for estimation",
        "linear",
        "DEP, analytical MTRs",
        "DEP analytical capital MTRs",
        "DEP_totalinc, analytical MTRs",
    ],
)
def test_get_tax_rates(
    tax_func_type,
    rate_type,
    params,
    analytical_mtrs,
    mtr_capital,
    for_estimation,
    expected,
):
    """
    Teset of txfunc.get_tax_rates() function.  There are 6 cases to
    test:
    1) DEP function, for estimation
    2) DEP function, not for estimation
    3) GS function, etr
    4) GS function, mtr
    5) DEP_totalinc function, for estimation
    6) DEP_totalinc function, not for estimation
    """
    wgts = np.array([0.1, 0.25, 0.55, 0.1])
    X = np.array([32.0, 44.0, 1.6, 0.4])
    Y = np.array([32.0, 55.0, 0.9, 0.03])
    print("Params = ", params)
    test_txrates = txfunc.get_tax_rates(
        params,
        X,
        Y,
        wgts,
        tax_func_type,
        rate_type,
        analytical_mtrs,
        mtr_capital,
        for_estimation,
    )

    assert np.allclose(test_txrates, expected)


@pytest.mark.local
def test_tax_func_estimate(tmpdir, dask_client):
    """
    Test txfunc.tax_func_loop() function.  The test is that given
    inputs from previous run, the outputs are unchanged.
    """
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "tax_func_estimate_inputs.pkl")
    )
    micro_data = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
    )
    (
        BW,
        S,
        starting_age,
        ending_age,
        beg_yr,
        baseline,
        analytical_mtrs,
        age_specific,
        reform,
        data,
        client,
        num_workers,
    ) = input_tuple
    tax_func_type = "DEP"
    age_specific = False
    BW = 1
    test_path = os.path.join(tmpdir, "test_out.pkl")
    test_dict = txfunc.tax_func_estimate(
        micro_data,
        BW,
        S,
        starting_age,
        ending_age,
        start_year=2030,
        baseline=baseline,
        analytical_mtrs=analytical_mtrs,
        tax_func_type=tax_func_type,
        age_specific=age_specific,
        reform=reform,
        data=data,
        client=dask_client,
        num_workers=NUM_WORKERS,
        tax_func_path=test_path,
    )
    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "tax_func_estimate_outputs.pkl")
    )
    del expected_dict["tfunc_time"]

    for k, v in expected_dict.items():
        if isinstance(v, str):  # for testing tax_func_type object
            assert test_dict[k] == v
        else:  # for testing all other objects
            print(
                "Max diff for ", k, " = ", np.absolute(test_dict[k] - v).max()
            )
            assert np.all(np.isclose(test_dict[k], v))
