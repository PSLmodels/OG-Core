import os
import sys
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_PATH, "../../"))
import pytest
import tempfile
import pickle
import numpy as np
import pandas as pd
from ogusa.utils import comp_array
from ogusa.utils import comp_scalar
from ogusa.utils import dict_compare
from ogusa.get_micro_data import get_calculator
from ogusa import SS
from ogusa import TPI

TOL = 1e-5

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
TAXDATA_PATH = os.path.join(CUR_PATH, '..', '..', 'test_data', 'puf91taxdata.csv.gz')
TAXDATA = pd.read_csv(TAXDATA_PATH, compression='gzip')
WEIGHTS_PATH = os.path.join(CUR_PATH, '..', '..', 'test_data', 'puf91weights.csv.gz')
WEIGHTS = pd.read_csv(WEIGHTS_PATH, compression='gzip')


@pytest.yield_fixture
def picklefile1():
    x = {'a': 1}
    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    pickle.dump(x, pfile)
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile2():
    y = {'a': 1, 'b': 2}

    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    pickle.dump(y, pfile)
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile3():
    x = {'a': np.array([100., 200., 300.]), 'b': 2}
    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    pickle.dump(x, pfile)
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile4():
    x = {'a': np.array([100., 200., 300.1]), 'b': 2}
    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    pickle.dump(x, pfile)
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


def test_import_ok():
    import ogusa


def test_run_small():
    from execute import runner
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    output_base = "./OUTPUT"
    input_dir = "./OUTPUT"
    user_params = {'frisch':0.41, 'debt_ratio_ss':0.4}
    runner(output_base=output_base, baseline_dir=input_dir, test=True, time_path=False, baseline=True, user_params=user_params, run_micro=False, small_open=False, budget_balance=False)
    runner(output_base=output_base, baseline_dir=input_dir, test=True, time_path=True, baseline=True, user_params=user_params, run_micro=False, small_open=False, budget_balance=False)


def test_compare_pickle_file_bad(picklefile1, picklefile2):
    from ogusa.utils import pickle_file_compare
    assert not pickle_file_compare(picklefile1.name, picklefile2.name)


def test_compare_pickle_file_bad2(picklefile3, picklefile4):
    from ogusa.utils import pickle_file_compare
    assert not pickle_file_compare(picklefile3.name, picklefile4.name)


def test_compare_pickle_file_relative(picklefile3, picklefile4):
    from ogusa.utils import pickle_file_compare
    assert pickle_file_compare(
        picklefile3.name, picklefile4.name, relative=True)


def test_compare_pickle_file_basic(picklefile1):
    from ogusa.utils import pickle_file_compare
    assert pickle_file_compare(picklefile1.name, picklefile1.name)


def test_compare_dict_basic():
    from ogusa.utils import dict_compare
    lhs = {'a': 1, 'b': 2}
    rhs = {'c': 4, 'b': 2}
    assert not dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs, tol=TOL)


def test_compare_dict_more_lhs():
    from ogusa.utils import dict_compare
    lhs = {'a': 1, 'b': 2, 'c': 3}
    rhs = {'c': 4, 'b': 2}
    assert not dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs, tol=TOL)


def test_compare_dict_diff_ndarrays():
    from ogusa.utils import dict_compare
    lhs = {'a': np.array([1, 2, 3]), 'b': 2}
    rhs = {'a': np.array([1, 3]), 'b': 2}
    assert not dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs, tol=TOL)


def test_compare_dict_diff_ndarrays2():
    from ogusa.utils import dict_compare
    lhs = {'a': np.array([1., 2., 3.]), 'b': 2}
    rhs = {'a': np.array([1., 2., 3.1]), 'b': 2}
    assert not dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs, tol=TOL)


def test_comp_array_relative():
    x = np.array([100., 200., 300.])
    y = np.array([100.01, 200.02, 300.03])
    unequal = []
    assert not comp_array("test", y, x, 1e-3, unequal)
    assert comp_array("test", y, x, 1e-3, unequal, relative=True)


def test_comp_array_relative_exception():
    x = np.array([100., 200., 300.])
    y = np.array([100.01, 200.02, 300.03])
    unequal = []
    exc = {'var': 1e-3}
    assert comp_array("var", y, x, 1e-5, unequal,
                      exceptions=exc, relative=True)


def test_comp_scalar_relative():
    x = 100
    y = 100.01
    unequal = []
    assert not comp_scalar("test", y, x, 1e-3, unequal)
    assert comp_scalar("test", y, x, 1e-3, unequal, relative=True)


def test_comp_scalar_relative_exception():
    x = 100
    y = 100.01
    unequal = []
    exc = {"var": 1e-3}
    assert comp_scalar("var", y, x, 1e-5, unequal,
                       exceptions=exc, relative=True)


def test_compare_dict_diff_ndarrays_relative():
    lhs = {'a': np.array([100., 200., 300.]), 'b': 2}
    rhs = {'a': np.array([100., 200., 300.1]), 'b': 2}
    assert dict_compare("lhs.pkle", lhs, "rhs.pkle",
                        rhs, tol=1e-3, relative=True)


def test_get_micro_data_get_calculator():

    reform = {
    2017: {
        '_II_rt1': [.09],
        '_II_rt2': [.135],
        '_II_rt3': [.225],
        '_II_rt4': [.252],
        '_II_rt5': [.297],
        '_II_rt6': [.315],
        '_II_rt7': [0.3564],
    }, }

    calc = get_calculator(baseline=False, calculator_start_year=2017,
                          reform=reform, data=TAXDATA,
                          weights=WEIGHTS, records_start_year=2009)
    assert calc.current_year == 2017

    reform = {
    2017: {
        '_II_rt1': [.09],
        '_II_rt2': [.135],
        '_II_rt3': [.225],
        '_II_rt4': [.252],
        '_II_rt5': [.297],
        '_II_rt6': [.315],
        '_II_rt7': [0.3564]
    }, }

    calc2 = get_calculator(baseline=False, calculator_start_year=2017,
                           reform=reform, data=TAXDATA,
                           weights=WEIGHTS, records_start_year=2009)
    assert calc2.current_year == 2017
