import os
import sys
import pytest
import pickle
from ogusa.parameters import (get_parameters, read_tax_func_estimate,
                              USER_MODIFIABLE_PARAMS)
from ogusa import parameters
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_PATH, "../../"))

def test_parameters_user_modifiable():
    output_base = "./OUTPUT"
    dd = get_parameters(output_base, test=False, guid='',
                        user_modifiable=True, metadata=False)
    assert set(dd.keys()) == set(USER_MODIFIABLE_PARAMS)


def test_parameters_metadata_policy():
    output_base = "./OUTPUT"
    dd_standard = get_parameters(output_base, test=False, guid='',
                                 user_modifiable=True, metadata=False)
    dd_meta = get_parameters(output_base, test=False, guid='',
                             user_modifiable=True, metadata=True)
    for k, v in dd_meta.items():
        assert dd_standard[k] == dd_meta[k]['value']
    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)

    dd_standard = get_parameters(output_base, test=False, guid='',
                                 user_modifiable=True, metadata=False)
    dd_meta = get_parameters(output_base, test=False, guid='',
                             user_modifiable=True, metadata=True)
    for k, v in dd_meta.items():
        assert dd_standard[k] == dd_meta[k]['value']

    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)
    assert 'validations' in dd_meta['frisch']


def test_parameters_metadata_baseline():
    output_base = "./OUTPUT"
    dd_standard = get_parameters(output_base, test=True, guid='',
                                 user_modifiable=True, metadata=False)
    dd_meta = get_parameters(output_base, test=True, guid='',
                             user_modifiable=True, metadata=True)
    for k, v in dd_meta.items():
        assert dd_standard[k] == dd_meta[k]['value']
    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)

    dd_standard = get_parameters(output_base, test=True, guid='',
                                 user_modifiable=True, metadata=False)
    dd_meta = get_parameters(output_base, test=True, guid='',
                             user_modifiable=True, metadata=True)
    for k, v in dd_meta.items():
        assert dd_standard[k] == dd_meta[k]['value']

    assert set(dd_meta.keys()) == set(USER_MODIFIABLE_PARAMS)
    assert 'validations' in dd_meta['frisch']


# @pytest.mark.parametrize("baseline", [True, False])
# @pytest.mark.parametrize(
#     "guid,tx_func_est_path,exp_tx_func_est_path",
#     [('', None, "./TxFuncEst_{}.pkl"),
#      ('', "test.pkl", "test.pkl"),
#      (9, None, "./TxFuncEst_{}9.pkl")])
# def test_tx_func_est_path(monkeypatch, baseline, guid, tx_func_est_path,
#                           exp_tx_func_est_path):
#     """
#     Make sure tax parameter paths work as expected
#     monkeypatch is a pytest plugin that mocks functions and modules
#     """
#     output_base = "./OUTPUT"
#     mocked_fn = parameters.read_tax_func_estimate
#     baseline_policy = "baseline" if baseline else "policy"
#
#     def read_tax_func_estimate_mock(pickle_path, pickle_file):
#         assert (
#             pickle_path == exp_tx_func_est_path.format(baseline_policy) and
#             pickle_file == "TxFuncEst_{0}{1}.pkl".format(baseline_policy,
#                                                          guid)
#         )
#         return mocked_fn(pickle_path, pickle_file)
#
#     monkeypatch.setattr(parameters, "read_tax_func_estimate", read_tax_func_estimate_mock)
#     try:
#         parameters.get_parameters(output_base, test=False, baseline=baseline, guid=guid)
#     except IOError: #file doesn't exist
#         pass
