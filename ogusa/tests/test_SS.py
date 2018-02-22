import pytest
import json
import numpy
import os
from ogusa import SS
os.path.abspath(os.path.dirname(__file__))

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_SS_fsolve():
    # Test SS.SS_fsolve function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = tuple(json.load(open(os.path.join(
        CUR_PATH, 'SS_fsolve_inputs.json'))))

    guesses, params = input_tuple
    test_list = SS.SS_fsolve(guesses, params)

    expected_list = tuple(json.load(open(os.path.join(
        CUR_PATH, 'SS_fsolve_outputs.json'))))

    assert(numpy.allclose(np.array(test_list), np.array(expected_list)))
