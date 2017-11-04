import pytest
# from ogusa.execute.scipts import runner

def test_set_data(monkeypatch):
    """
    Check that setting `data` to 'cps' uses cps data
    """
    from ogusa.txfunc import get_tax_func_estimate, tax_func_estimate
    from ogusa import get_micro_data
    mocked_fn = get_micro_data
    baseline=False
    start_year=2016
    reform = {2017: {"_II_em": [10000]}}

    calc = get_micro_data.get_calculator(baseline, start_year, reform=reform,
                                         data="cps")
    # blind_head is only in the CPS file. See taxcalc/records_variables.json
    assert calc.records.blind_head.sum() > 0
