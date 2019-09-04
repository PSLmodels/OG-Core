import pytest
from ogusa import utils
import numpy as np
from ogusa.parameters import Specifications


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
# new_param_values = {
#     'T': 40,
#     'S': 3,
#     'J': 1
# }
# # update parameters instance with new values for test
# p.update_specifications(new_param_values)
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
