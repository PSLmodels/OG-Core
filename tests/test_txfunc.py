import sys

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
        (2, np.array([[False, False, False], [False, False, True]])),
        (8, np.array([[False, False, False], [False, False, False]])),
    ],
    ids=["2", "8"],
)
def test_find_outliers(se_mult, expected_mat):
    # Test the find outliers function
    sse_mat = np.array([[21.0, 20.0, 20.0], [22.0, 32.0, 100.0]])
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
    param_arr = random_state.rand(BW * S * numparams).reshape(BW, S, numparams)
    param_list = np.zeros((BW, S)).tolist()
    for bw in range(BW):
        for s in range(S):
            param_list[bw][s] = param_arr[bw, s, :]
    sse_big_mat = ~np.ones((BW, S), dtype=bool)
    sse_big_mat[0, 0] = True
    sse_big_mat[0, 1] = True
    sse_big_mat[0, S - 4] = True
    sse_big_mat[0, S - 5] = True
    sse_big_mat[0, S - 2] = True
    sse_big_mat[0, S - 1] = True

    act = txfunc.replace_outliers(param_list, sse_big_mat)

    exp = [
        [
            np.array([0.19806286, 0.76053071, 0.16911084]),
            np.array([0.19806286, 0.76053071, 0.16911084]),
            np.array([0.19806286, 0.76053071, 0.16911084]),
            np.array([0.08833981, 0.68535982, 0.95339335]),
            np.array([0.00394827, 0.51219226, 0.81262096]),
            np.array([0.05002219, 0.46590843, 0.76645851]),
            np.array([0.09609612, 0.41962459, 0.72029606]),
            np.array([0.14217005, 0.37334076, 0.67413362]),
            np.array([0.14217005, 0.37334076, 0.67413362]),
            np.array([0.14217005, 0.37334076, 0.67413362]),
        ],
        [
            np.array([0.8052232, 0.52164715, 0.90864888]),
            np.array([0.31923609, 0.09045935, 0.30070006]),
            np.array([0.11398436, 0.82868133, 0.04689632]),
            np.array([0.62628715, 0.54758616, 0.819287]),
            np.array([0.19894754, 0.8568503, 0.35165264]),
            np.array([0.75464769, 0.29596171, 0.88393648]),
            np.array([0.32551164, 0.1650159, 0.39252924]),
            np.array([0.09346037, 0.82110566, 0.15115202]),
            np.array([0.38411445, 0.94426071, 0.98762547]),
            np.array([0.45630455, 0.82612284, 0.25137413]),
        ],
    ]

    # test that act == exp
    mat_isclose = np.zeros((BW, S), dtype=bool)
    for bw in range(BW):
        for s in range(S):
            mat_isclose[bw, s] = np.allclose(act[bw][s], exp[bw][s])

    assert (1 - mat_isclose).sum() == 0


expected_tuple_DEP = (
    np.array(
        [
            6.34662816e-17,
            6.05703755e-05,
            2.32905968e-13,
            6.74269844e-08,
            8.00000000e-01,
            7.32264836e-04,
            8.90355422e-01,
            9.02760087e-06,
            9.02760087e-06,
            7.99990972e-03,
            7.23237235e-06,
            9.02760087e-06,
        ]
    ),
    237645.51573934805,
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
    256983.45682508394,
    152900,
)
expected_tuple_linear = (0.26135747, 0.0, 152900)
expected_tuple_GS = (
    np.array([0.2380778, 0.44519796, 0.00958987]),
    357042.1125832452,
    152900,
)
expected_tuple_linear_mtrx = (0.37030104, 0.0, 152900)
expected_tuple_linear_mtry = (0.24793767, 0.0, 152900)
expected_tuple_HSV = (
    np.array([1.68456651, 0.06149992]),
    # 792935850.6195159,
    1116189.8044088066,
    152900,
)
expected_tuple_HSV_mtrx = (
    np.array([1.35287201, 0.05298318]),
    # 642774637.8247124,
    904161.7256125457,
    152900,
)
expected_tuple_HSV_mtry = (
    np.array([1.76056732, 0.05796658]),
    # 798149828.3166732,
    1119562.1877437222,
    152900,
)


@pytest.mark.parametrize(
    "rate_type,tax_func_type,true_params",
    [
        # ("etr", "DEP", [6.28E-12, 4.36E-05, 1.04E-23, 7.77E-09, 0.80, 0.80, 0.84, -0.14, -0.15, 0.15, 0.16, -0.15]),
        (
            "etr",
            "DEP_totalinc",
            [6.28e-12, 4.36e-05, 0.35, -0.14, 0.15, -0.15],
        ),
        ("etr", "GS", [0.35, 0.25, 0.03]),
        ("etr", "linear", [0.25]),
        ("mtrx", "linear", [0.4]),
        ("mtry", "linear", [0.1]),
        ("etr", "HSV", [0.5, 0.1]),
        ("mtrx", "HSV", [0.5, 0.1]),
        ("mtry", "HSV", [0.4, 0.15]),
    ],
    # ids=["DEP", "DEP_totalinc", "GS", "linear, etr",
    ids=[
        "DEP_totalinc",
        "GS",
        "linear, etr",
        "linear, mtrx",
        "linear, mtry",
        "HSV, etr",
        "HSV, mtrx",
        "HSV, mtry",
    ],
)
def test_txfunc_est(rate_type, tax_func_type, true_params, tmpdir):
    """
    Test txfunc.txfunc_est() function.  The test the estimator can
    recover (close to) the true parameters.
    """
    # Generate data based on true parameters
    N = 20_000
    weights = np.ones(N)
    x = np.random.uniform(0, 500_000, size=N)
    y = np.random.uniform(0, 500_000, size=N)
    eps1 = np.random.normal(scale=0.00001, size=N)
    eps2 = np.random.normal(scale=0.00001, size=N)
    eps3 = np.random.normal(scale=0.00001, size=N)
    micro_data = pd.DataFrame(
        {
            "total_capinc": y,
            "total_labinc": x,
            "weight": weights,
            "total_tax": (
                (
                    txfunc.get_tax_rates(
                        true_params,
                        x,
                        y,
                        weights,
                        tax_func_type,
                        rate_type="etr",
                        for_estimation=False,
                    )
                    + eps1
                )
                * (x + y)
            ),
            "etr": (
                txfunc.get_tax_rates(
                    true_params,
                    x,
                    y,
                    weights,
                    tax_func_type,
                    rate_type="etr",
                    for_estimation=False,
                )
                + eps1
            ),
            "mtr_labinc": (
                txfunc.get_tax_rates(
                    true_params,
                    x,
                    y,
                    weights,
                    tax_func_type,
                    rate_type="mtr",
                    for_estimation=False,
                )
                + eps2
            ),
            "mtr_capinc": (
                txfunc.get_tax_rates(
                    true_params,
                    x,
                    y,
                    weights,
                    tax_func_type,
                    rate_type="mtr",
                    for_estimation=False,
                )
                + eps3
            ),
        }
    )
    micro_data["age"] = 44
    micro_data["year"] = 2025
    output_dir = tmpdir
    param_est, _, obs, _ = txfunc.txfunc_est(
        micro_data,
        44,
        2025,
        rate_type,
        tax_func_type,
        len(true_params),
        output_dir,
        True,
    )

    assert obs == micro_data.shape[0]
    print("Estimated parameters:", param_est)
    print("Diffs = ", np.absolute(param_est - true_params))
    if "DEP" in tax_func_type:
        assert np.allclose(param_est, true_params, atol=0.2, rtol=0.01)
    else:
        assert np.allclose(param_est, true_params, atol=0.0, rtol=0.01)
    # TODO: maybe the test is that the true parameters are in the
    # 95% confidence interval of the estimated parameters


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


def test_tax_func_loop():
    """
    Test txfunc.tax_func_loop() function. The test is that given inputs from
    previous run, the outputs are unchanged.
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
    tax_func_type = "HSV"
    numparams = 2
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
        if isinstance(test_tuple[i], list):
            test_tuple_obj = np.array(test_tuple[i])
            exp_tuple_obj = np.array(expected_tuple[i])
            print(
                "For element",
                i,
                ", diff =",
                np.absolute(test_tuple_obj - exp_tuple_obj).max(),
            )
        else:
            print(
                "For element",
                i,
                ", diff =",
                np.absolute(test_tuple[i] - v).max(),
            )
        assert np.allclose(test_tuple[i], v, atol=1e-06)


# DEP parameters
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
shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
# GS parameters
phi0 = 0.6
phi1 = 0.5
phi2 = 0.6
# Linear parameters
avg_rate = 0.17
# HSV parameters
lambda_s = 0.5
tau_s = 0.1


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
        (
            "HSV",
            "etr",
            np.array([lambda_s, tau_s]),
            True,
            False,
            False,
            np.array([0.670123022, 0.684204102, 0.543778232, 0.455969612]),
        ),
        (
            "HSV",
            "mtr",
            np.array([lambda_s, tau_s]),
            True,
            False,
            False,
            np.array([0.70311072, 0.715783692, 0.589400409, 0.510372651]),
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
        "HSV, etr",
        "HSV, mtr",
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
    Teset of txfunc.get_tax_rates() function.
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
    tax_func_type = "HSV"
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
        analytical_mtrs=analytical_mtrs,
        tax_func_type=tax_func_type,
        age_specific=age_specific,
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
        elif isinstance(expected_dict[k], list):
            test_dict_obj = np.array(test_dict[k])
            exp_dict_obj = np.array(expected_dict[k])
            print(
                "For element",
                k,
                ", diff =",
                np.absolute(test_dict_obj - exp_dict_obj).max(),
            )
        else:  # for testing all other objects
            print(
                "Max diff for ", k, " = ", np.absolute(test_dict[k] - v).max()
            )
            assert np.all(np.isclose(test_dict[k], v))


@pytest.mark.local
def test_monotone_spline():
    """
    Test txfunc.monotone_spline() function.

        1. Test whether the output includes a function
        2. test that the function give the correct outputs
        3. test that y_cstr, wsse_cstr, y_uncstr, wsse_uncstr are correct
    """
    # Simulate some data
    np.random.seed(10)

    # another test case for pygam method:

    # N = 100
    # xlo = 0.001
    # xhi = 2 * np.pi
    # x1 = np.arange(xlo, xhi, step=(xhi - xlo) / N)
    # x2 = np.arange(xlo, xhi, step=(xhi - xlo) / N)
    # x = np.array([x1, x2]).T
    # X1, X2 = np.meshgrid(x1, x2)
    # X1, X2 = X1.flatten(), X2.flatten()
    # X = np.zeros((X1.shape[0], 2))
    # X[:,0] = X1
    # X[:,1] = X2
    # y0 = np.exp(np.sin(X1)) * np.exp(np.cos(X2))
    # y0 = np.cos(X1)/(1.01 + np.sin(X2))
    # y0 = 1/(1 + np.exp(-(X1+X2)))
    # y = y0 + np.random.random(y0.shape) * 0.2
    # weights = np.ones(10000)
    # (
    # mono_interp,
    # y_cstr,
    # wsse_cstr,
    # y_uncstr,
    # wsse_uncstr,
    # ) = txfunc.monotone_spline(
    # X, y, weights, lam=100, incl_uncstr=True, show_plot=True, method='pygam'
    # )

    data = utils.safe_read_pickle(
        os.path.join(CUR_PATH, "test_io_data", "micro_data_dict_for_tests.pkl")
    )["2030"]
    df = data[["total_labinc", "total_capinc", "etr", "weight"]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Estimate monotonically increasing function on the data
    (
        mono_interp1,
        y_cstr1,
        wsse_cstr1,
        y_uncstr1,
        wsse_uncstr1,
    ) = txfunc.monotone_spline(
        df[["total_labinc", "total_capinc"]].values,
        df["etr"].values,
        df["weight"].values,
        bins=[100, 100],
        incl_uncstr=True,
        show_plot=False,
        method="pygam",
        splines=[100, 100],
        plot_start=25,
        plot_end=75,
    )
    # Test whether mono_interp is a function
    assert hasattr(mono_interp1, "__call__")
    # Not sure what baseline should be to add tests here for "correctness" of values

    # Simulate some data
    N = 100
    xlo = 0.001
    xhi = 2 * np.pi
    x = np.arange(xlo, xhi, step=(xhi - xlo) / N)
    y0 = np.sin(x) + np.log(x)
    y = y0 + np.random.randn(N) * 0.5
    weights = np.ones(N)

    (
        mono_interp2,
        y_cstr2,
        wsse_cstr2,
        y_uncstr2,
        wsse_uncstr2,
    ) = txfunc.monotone_spline(
        x,
        y,
        weights,
        bins=10,
        lam=100,
        incl_uncstr=True,
        show_plot=False,
        method="eilers",
    )
    # Test whether mono_interp is a function
    assert hasattr(mono_interp2, "__call__")

    # Test whether mono_interp is a function
    assert hasattr(mono_interp2, "__call__")
    # Test whether mono_interp gives the correct output
    x_vec_test = np.array([2.0, 5.0])
    y_vec_expected = np.array([0.69512331, 1.23822669])
    y_monointerp = mono_interp2(x_vec_test)
    assert np.allclose(y_monointerp, y_vec_expected)

    # Test that y_cstr, wsse_cstr, y_uncstr, wsse_uncstr are correct
    y_cstr_expected = np.array(
        [
            -0.14203509,
            0.21658404,
            0.5146192,
            0.75351991,
            0.93709706,
            1.071363,
            1.16349465,
            1.22046208,
            1.24755794,
            1.24755791,
        ]
    )
    wsse_cstr_expected = 546.0485935661629
    y_uncstr_expected = np.array(
        [
            -0.45699668,
            0.16840804,
            0.66150226,
            1.02342052,
            1.25698067,
            1.36630762,
            1.35585018,
            1.22938312,
            0.98920508,
            0.63615989,
        ]
    )
    wsse_uncstr_expected = 468.4894930349361
    assert np.allclose(y_cstr2, y_cstr_expected)
    assert np.allclose(wsse_cstr2, wsse_cstr_expected)
    assert np.allclose(y_uncstr2, y_uncstr_expected)
    assert np.allclose(wsse_uncstr2, wsse_uncstr_expected)
