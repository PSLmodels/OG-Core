import multiprocessing
from distributed import Client, LocalCluster
import pytest
from pandas.util.testing import assert_frame_equal
import numpy as np
import os
from ogusa.constants import CPS_START_YEAR, PUF_START_YEAR, TC_LAST_YEAR
from ogusa import get_micro_data, utils
from taxcalc import GrowFactors
NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
# get path to puf if puf.csv in ogusa/ directory
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '..', 'puf.csv')


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


def test_cps():
    """
    Check that setting `data` to 'cps' uses cps data
    """
    baseline = False
    start_year = 2016
    reform = {"II_em": {2017: 10000}}

    calc = get_micro_data.get_calculator(
        baseline, start_year, reform=reform,
        records_start_year=CPS_START_YEAR, data="cps")
    # blind_head is only in the CPS file and e00700 is only in the PUF.
    # See taxcalc/records_variables.json
    assert (calc.array("blind_head").sum() > 0 and
            calc.array("e00700").sum() == 0)


def test_set_path():
    """
    Check that 'notapath.csv' is passed to taxcalc. An error
    containing 'notapath.csv' is sufficient proof for this
    """
    baseline = False
    start_year = 2016
    reform = {"II_em": {2017: 10000}}

    # In theory this path doesn't exist so there should be an IOError
    # But taxcalc checks if the path exists and if it doesn't, it tries
    # to read from an egg file. This raises a ValueError. At some point,
    # this could change. So I think it's best to catch both errors
    with pytest.raises((IOError, ValueError), match="notapath.csv"):
        get_micro_data.get_calculator(
            baseline, start_year, reform=reform,
            records_start_year=CPS_START_YEAR, data="notapath.csv")


def test_puf_path():
    """
    Check that setting `data` to None uses the puf file
    """
    baseline = False
    start_year = 2016
    reform = {"II_em": {2017: 10000}}

    # puf.csv in ogusa/
    if os.path.exists(PUF_PATH):
        calc = get_micro_data.get_calculator(
            baseline, start_year, reform=reform, data=PUF_PATH)
        # blind_head is only in the CPS file and e00700 is only in the
        # PUF.  See taxcalc/records_variables.json
        assert (calc.array('blind_head').sum() == 0 and
                calc.array('e00700').sum() > 0)
    # we do not have puf.csv
    else:
        # make sure TC is looking for puf.csv
        with pytest.raises((IOError, ValueError), match="puf.csv"):
            get_micro_data.get_calculator(
                baseline, start_year, reform=reform,
                records_start_year=CPS_START_YEAR, data=None)


iit_reform_1 = {
    'II_rt1': {2017: 0.09},
    'II_rt2': {2017: 0.135},
    'II_rt3': {2017: 0.225},
    'II_rt4': {2017: 0.252},
    'II_rt5': {2017: 0.297},
    'II_rt6': {2017: 0.315},
    'II_rt7': {2017: 0.3564}
    }


@pytest.mark.parametrize(
    'baseline,iit_reform',
    [(False, iit_reform_1), (False, {}), (True, iit_reform_1),
     (True, {})],
    ids=['Reform, Policy change given',
         'Reform, No policy change given',
         'Baseline, Policy change given',
         'Baseline, No policy change given'])
def test_get_calculator_cps(baseline, iit_reform):
    calc = get_micro_data.get_calculator(
        baseline=baseline, calculator_start_year=2017,
        reform=iit_reform, data='cps', gfactors=GrowFactors(),
        records_start_year=CPS_START_YEAR)
    assert calc.current_year == CPS_START_YEAR


def test_get_calculator_exception():
    iit_reform = {
        'II_rt1': {2017: 0.09},
        'II_rt2': {2017: 0.135},
        'II_rt3': {2017: 0.225},
        'II_rt4': {2017: 0.252},
        'II_rt5': {2017: 0.297},
        'II_rt6': {2017: 0.315},
        'II_rt7': {2017: 0.3564}
        }
    with pytest.raises(Exception):
        assert get_micro_data.get_calculator(
            baseline=False, calculator_start_year=TC_LAST_YEAR + 1,
            reform=iit_reform, data='cps', gfactors=GrowFactors(),
            records_start_year=CPS_START_YEAR)


@pytest.mark.full_run
def test_get_calculator_puf():
    iit_reform = {
        'II_rt1': {2017: 0.09},
        'II_rt2': {2017: 0.135},
        'II_rt3': {2017: 0.225},
        'II_rt4': {2017: 0.252},
        'II_rt5': {2017: 0.297},
        'II_rt6': {2017: 0.315},
        'II_rt7': {2017: 0.3564}
        }
    calc = get_micro_data.get_calculator(
        baseline=False, calculator_start_year=2017, reform=iit_reform,
        data=None,
        records_start_year=PUF_START_YEAR)
    assert calc.current_year == 2013


@pytest.mark.full_run
def test_get_calculator_puf_from_file():
    iit_reform = {
        'II_rt1': {2017: 0.09},
        'II_rt2': {2017: 0.135},
        'II_rt3': {2017: 0.225},
        'II_rt4': {2017: 0.252},
        'II_rt5': {2017: 0.297},
        'II_rt6': {2017: 0.315},
        'II_rt7': {2017: 0.3564}
        }
    calc = get_micro_data.get_calculator(
        baseline=False, calculator_start_year=2017, reform=iit_reform,
        data=PUF_PATH,
        records_start_year=PUF_START_YEAR)
    assert calc.current_year == 2013


@pytest.mark.parametrize(
    'baseline', [True, False], ids=['Baseline', 'Reform'])
def test_get_data(baseline, dask_client):
    '''
    Test of get_micro_data.get_data() function
    '''
    expected_data = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data',
                     'micro_data_dict_for_tests.pkl'))
    test_data, _ = get_micro_data.get_data(
        baseline=baseline, start_year=2029, reform={}, data='cps',
        client=dask_client, num_workers=NUM_WORKERS)
    for k, v in test_data.items():
        try:
            assert_frame_equal(
                expected_data[k], v)
        except KeyError:
            pass


def test_taxcalc_advance():
    '''
    Test of the get_micro_data.taxcalc_advance() function

    Note that this test may fail if the Tax-Calculator is not v 2.4.0
    In that case, you can use the pickeld calculator object, however
    this is too large for GitHub, so it won't be available there.
    '''
    expected_dict = utils.safe_read_pickle(os.path.join(
        CUR_PATH, 'test_io_data', 'tax_dict_for_tests.pkl'))
    test_dict = get_micro_data.taxcalc_advance(True, 2028, {}, 'cps', 2028)
    del test_dict['payroll_tax_liab']
    for k, v in test_dict.items():
        assert np.allclose(expected_dict[k], v, equal_nan=True)


@pytest.mark.full_run
def test_cap_inc_mtr():
    '''
    Test of the get_micro_data.cap_inc_mtr() function

    Note that this test may fail if the Tax-Calculator is not v 2.4.0
    In that case, you can use the pickeld caculator object, however
    this is too large for GitHub, so it won't be available there.
    '''
    # calc1 = utils.safe_read_pickle(os.path.join(
    #         CUR_PATH, 'test_io_data', 'calc_object_for_tests.pkl'))
    calc1 = get_micro_data.get_calculator(
        baseline=True, calculator_start_year=2028, reform={},
        data='cps')
    calc1.advance_to_year(2028)
    expected = np.genfromtxt(os.path.join(
            CUR_PATH, 'test_io_data',
            'mtr_combined_capinc_for_tests.csv'), delimiter=',')
    test_data = get_micro_data.cap_inc_mtr(calc1)

    assert np.allclose(expected, test_data, equal_nan=True)
