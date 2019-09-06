import pytest
import os
from ogusa.utils import CPS_START_YEAR
from ogusa import get_micro_data
from taxcalc import GrowFactors


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

    # get path to puf if puf.csv in ogusa/ directory
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    puf_path = os.path.join(cur_dir, "../puf.csv")

    # puf.csv in ogusa/
    if os.path.exists(puf_path):
        calc = get_micro_data.get_calculator(
            baseline, start_year, reform=reform, data=puf_path)
        # blind_head is only in the CPS file and e00700 is only in the PUF.
        # See taxcalc/records_variables.json
        assert (calc.array('blind_head').sum() == 0 and
                calc.array('e00700').sum() > 0)
    # we do not have puf.csv
    else:
        # make sure TC is looking for puf.csv
        with pytest.raises((IOError, ValueError), match="puf.csv"):
            get_micro_data.get_calculator(
                baseline, start_year, reform=reform,
                records_start_year=CPS_START_YEAR, data=None)


def test_get_calculator():
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
        data='cps', gfactors=GrowFactors(),
        records_start_year=CPS_START_YEAR)
    assert calc.current_year == CPS_START_YEAR
