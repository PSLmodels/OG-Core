import os
import sys
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_PATH, "../../"))
import pytest
import tempfile
import pickle
import numpy as np
from ogusa.utils import comp_array
from ogusa.utils import comp_scalar
from ogusa.utils import dict_compare
from ogusa.parameters import (get_reduced_parameters, USER_MODIFIABLE_PARAMS,
                              get_full_parameters)


def test_parameters_user_modifiable():
    dd = get_reduced_parameters(False, guid='', user_modifiable=True, metadata=False)
    assert set(dd.keys()) == set(USER_MODIFIABLE_PARAMS)
    dd = get_full_parameters(False, guid='', user_modifiable=True, metadata=False)
    assert set(dd.keys()) == set(USER_MODIFIABLE_PARAMS)


def test_parameters_metadata_policy():
    dd_standard = get_reduced_parameters(False, guid='', user_modifiable=True, metadata=False)
    dd_meta = get_reduced_parameters(False, guid='', user_modifiable=True, metadata=True)
    for k, v in dd_meta.iteritems():
        assert dd_standard[k] == dd_meta[k]['value']
    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)

    dd_standard = get_full_parameters(False, guid='', user_modifiable=True, metadata=False)
    dd_meta = get_full_parameters(False, guid='', user_modifiable=True, metadata=True)
    for k, v in dd_meta.iteritems():
        assert dd_standard[k] == dd_meta[k]['value']

    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)
    assert 'validations' in dd_meta['frisch']

def test_parameters_metadata_baseline():
    dd_standard = get_reduced_parameters(True, guid='', user_modifiable=True, metadata=False)
    dd_meta = get_reduced_parameters(True, guid='', user_modifiable=True, metadata=True)
    for k, v in dd_meta.iteritems():
        assert dd_standard[k] == dd_meta[k]['value']
    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)

    dd_standard = get_full_parameters(True, guid='', user_modifiable=True, metadata=False)
    dd_meta = get_full_parameters(True, guid='', user_modifiable=True, metadata=True)
    for k, v in dd_meta.iteritems():
        assert dd_standard[k] == dd_meta[k]['value']

    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)
    assert 'validations' in dd_meta['frisch']
