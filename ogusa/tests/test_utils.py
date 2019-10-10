import pytest
from ogusa import utils
from ogusa.utils import Inequality
import numpy as np
import tempfile
import os
import pickle
from ogusa.parameters import Specifications

TOL = 1e-5


def test_rate_conversion():
    '''
    Test of utils.rate_conversion
    '''
    expected_rate = 0.3
    annual_rate = 0.3
    start_age = 20
    end_age = 80
    s = 60
    test_rate = utils.rate_conversion(annual_rate, start_age, end_age, s)
    assert(np.allclose(expected_rate, test_rate))


# Parameter values to use for inequality tests
J = 2
S = 10
dist = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 1, 1, 2, 3, 4, 5, 6, 7, 20]])
pop_weights = np.ones(S) / S
ability_weights = np.array([0.5, 0.5])


def test_create_ineq_object():
    '''
    Test that Inequality class and be created
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    assert ineq


def test_gini():
    '''
    Test of Gini coefficient calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    gini = ineq.gini()
    assert np.allclose(gini, 0.41190476190476133)


def test_var_of_logs():
    '''
    Test of variance of logs calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    var_log = ineq.var_of_logs()
    assert np.allclose(var_log, 0.7187890928713506)


def test_ratio_pct1_pct2():
    '''
    Test of percentile ratio calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    ratio = ineq.ratio_pct1_pct2(0.9, 0.1)
    assert np.allclose(ratio, 9)


def test_top_share():
    '''
    Test of top share calculation
    '''
    ineq = Inequality(dist, pop_weights, ability_weights, S, J)
    top_share = ineq.top_share(0.05)
    assert np.allclose(top_share, 0.285714286)


def test_to_timepath_shape():
    '''
    Test of function that converts vector to time path conformable array
    '''
    in_array = np.ones(40)
    test_array = utils.to_timepath_shape(in_array)
    assert test_array.shape == (40, 1, 1)


p = Specifications()
p.T = 40
p.S = 3
p.J = 1
x1 = np.ones((p.S, p.J)) * 0.4
xT = np.ones((p.S, p.J)) * 5.0
expected1 = np.tile(np.array([
    0.4, 0.51794872, 0.63589744, 0.75384615, 0.87179487, 0.98974359,
    1.10769231, 1.22564103, 1.34358974, 1.46153846, 1.57948718,
    1.6974359, 1.81538462, 1.93333333, 2.05128205, 2.16923077,
    2.28717949, 2.40512821, 2.52307692, 2.64102564, 2.75897436,
    2.87692308, 2.99487179, 3.11282051, 3.23076923, 3.34871795,
    3.46666667, 3.58461538, 3.7025641, 3.82051282, 3.93846154,
    4.05641026, 4.17435897, 4.29230769, 4.41025641, 4.52820513,
    4.64615385, 4.76410256, 4.88205128, 5., 5.0, 5.0, 5.0]
                             ).reshape(p.T + p.S, 1, 1), (1, p.S, p.J))
expected2 = np.tile(np.array([
    0.4, 0.63287311, 0.85969757, 1.08047337, 1.29520053, 1.50387903,
    1.70650888, 1.90309007, 2.09362262, 2.27810651, 2.45654175,
    2.62892834, 2.79526627, 2.95555556, 3.10979619, 3.25798817,
    3.40013149, 3.53622617, 3.66627219, 3.79026956, 3.90821828,
    4.02011834, 4.12596976, 4.22577252, 4.31952663, 4.40723208,
    4.48888889, 4.56449704, 4.63405654, 4.69756739, 4.75502959,
    4.80644313, 4.85180802, 4.89112426, 4.92439185, 4.95161078,
    4.97278107, 4.9879027,  4.99697567, 5., 5., 5., 5.]
                             ).reshape(p.T + p.S, 1, 1), (1, p.S, p.J))
expected3 = np.tile(np.array([
    0.4, 2.72911392, 3.49243697, 3.87169811, 4.09849246, 4.24937238,
    4.35698925, 4.43761755, 4.50027855, 4.55037594, 4.59134396,
    4.62546973, 4.65433526, 4.67906977, 4.70050083, 4.71924883,
    4.73578792, 4.75048679, 4.76363636, 4.77546934, 4.78617402,
    4.79590444, 4.80478781, 4.81293014, 4.82042042, 4.82733397,
    4.83373494, 4.83967828, 4.84521139, 4.85037531, 4.85520581,
    4.85973417, 4.86398787, 4.86799117, 4.87176555, 4.87533009,
    4.87870183, 4.88189598, 4.88492623, 4.88780488, 5., 5., 5.]
                             ).reshape(p.T + p.S, 1, 1), (1, p.S, p.J))
expected4 = np.ones((p.T + p.S, p.S, p.J)) * xT


@pytest.mark.parametrize(
    'x1,xT,p,shape,expected', [
        (x1, xT, p, 'linear', expected1),
        (x1, xT, p, 'quadratic', expected2),
        (x1, xT, p, 'ratio', expected3),
        (xT, xT, p, 'linear', expected4),
        (xT, xT, p, 'quadratic', expected4),
        (xT, xT, p, 'ratio', expected4)],
    ids=['linear', 'quadratic', 'ratio', 'linear - trivial',
         'quadratic - trivial', 'ratio- trivial'])
def test_get_initial_path(x1, xT, p, shape, expected):
    '''
    Test of utils.get_inital_path function
    '''
    test_path = utils.get_initial_path(x1, xT, p, shape)
    assert np.allclose(test_path, expected)


@pytest.yield_fixture
def picklefile1():
    x = {'a': 1}
    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    with open(pfile.name, 'wb') as f:
        pickle.dump(x, f)
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile2():
    y = {'a': 1, 'b': 2}

    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    with open(pfile.name, 'wb') as f:
        pickle.dump(y, f)
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile3():
    x = {'a': np.array([100., 200., 300.]), 'b': 2}
    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    with open(pfile.name, 'wb') as f:
        pickle.dump(x, f)
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


@pytest.yield_fixture
def picklefile4():
    x = {'a': np.array([100., 200., 300.1]), 'b': 2}
    pfile = tempfile.NamedTemporaryFile(mode="a", delete=False)
    with open(pfile.name, 'wb') as f:
        pickle.dump(x, )
    pfile.close()
    # must close and then yield for Windows platform
    yield pfile
    os.remove(pfile.name)


def test_compare_pickle_file_bad(picklefile1, picklefile2):
    assert not utils.pickle_file_compare(picklefile1.name, picklefile2.name)


def test_compare_pickle_file_bad2(picklefile3, picklefile4):
    assert not utils.pickle_file_compare(picklefile3.name, picklefile4.name)


def test_compare_pickle_file_relative(picklefile3, picklefile4):
    assert utils.pickle_file_compare(
        picklefile3.name, picklefile4.name, relative=True)


def test_compare_pickle_file_basic(picklefile1):
    assert utils.pickle_file_compare(picklefile1.name, picklefile1.name)


def test_compare_dict_basic():
    from ogusa.utils import dict_compare
    lhs = {'a': 1, 'b': 2}
    rhs = {'c': 4, 'b': 2}
    assert not dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs, tol=TOL)


def test_compare_dict_more_lhs():
    lhs = {'a': 1, 'b': 2, 'c': 3}
    rhs = {'c': 4, 'b': 2}
    assert not utils.dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs,
                                  tol=TOL)


def test_compare_dict_diff_ndarrays():
    lhs = {'a': np.array([1, 2, 3]), 'b': 2}
    rhs = {'a': np.array([1, 3]), 'b': 2}
    assert not utils.dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs,
                                  tol=TOL)


def test_compare_dict_diff_ndarrays2():
    lhs = {'a': np.array([1., 2., 3.]), 'b': 2}
    rhs = {'a': np.array([1., 2., 3.1]), 'b': 2}
    assert not utils.dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs,
                                  tol=TOL)


def test_comp_array_relative():
    x = np.array([100., 200., 300.])
    y = np.array([100.01, 200.02, 300.03])
    unequal = []
    assert not utils.comp_array("test", y, x, 1e-3, unequal)
    assert utils.comp_array("test", y, x, 1e-3, unequal, relative=True)


def test_comp_array_relative_exception():
    x = np.array([100., 200., 300.])
    y = np.array([100.01, 200.02, 300.03])
    unequal = []
    exc = {'var': 1e-3}
    assert utils.comp_array("var", y, x, 1e-5, unequal, exceptions=exc,
                            relative=True)


def test_comp_scalar_relative():
    x = 100
    y = 100.01
    unequal = []
    assert not utils.comp_scalar("test", y, x, 1e-3, unequal)
    assert utils.comp_scalar("test", y, x, 1e-3, unequal, relative=True)


def test_comp_scalar_relative_exception():
    x = 100
    y = 100.01
    unequal = []
    exc = {"var": 1e-3}
    assert utils.comp_scalar("var", y, x, 1e-5, unequal, exceptions=exc,
                             relative=True)


def test_compare_dict_diff_ndarrays_relative():
    lhs = {'a': np.array([100., 200., 300.]), 'b': 2}
    rhs = {'a': np.array([100., 200., 300.1]), 'b': 2}
    assert utils.dict_compare("lhs.pkle", lhs, "rhs.pkle", rhs,
                              tol=1e-3, relative=True)
