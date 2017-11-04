import pytest


def test_cps():
    """
    Check that setting `data` to 'cps' uses cps data
    """
    from ogusa import get_micro_data
    baseline=False
    start_year=2016
    reform = {2017: {"_II_em": [10000]}}

    calc = get_micro_data.get_calculator(baseline, start_year, reform=reform,
                                         data="cps")
    # blind_head is only in the CPS file and e00700 is only in the PUF.
    # See taxcalc/records_variables.json
    assert (calc.records.blind_head.sum() > 0 and
            calc.records.e00700.sum() == 0)


def test_set_path():
    """
    Check that 'notapath.csv' is passed to taxcalc. An error
    containing 'notapath.csv' is sufficient proof for this
    """
    from ogusa import get_micro_data
    baseline=False
    start_year=2016
    reform = {2017: {"_II_em": [10000]}}

    # In theory this path doesn't exist so there should be an IOError
    # But taxcalc checks if the path exists and if it doesn't, it tries
    # to read from an egg file. This raises a ValueError. At some point,
    # this could change. So I think it's best to catch both errors
    with pytest.raises((IOError, ValueError), match="notapath.csv"):
        get_micro_data.get_calculator(baseline, start_year, reform=reform,
                                      data="notapath.csv")


def test_no_path_specified():
    """
    Check that setting `data` to None uses the puf file
    """
    from ogusa import get_micro_data
    baseline=False
    start_year=2016
    reform = {2017: {"_II_em": [10000]}}

    calc = get_micro_data.get_calculator(baseline, start_year, reform=reform,
                                         data=None)
    # blind_head is only in the CPS file and e00700 is only in the PUF.
    # See taxcalc/records_variables.json
    assert (calc.records.blind_head.sum() == 0 and
            calc.records.e00700.sum() > 0)
