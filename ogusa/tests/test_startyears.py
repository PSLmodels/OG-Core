import pytest
import os
import ogusa
from ogusa import SS, TPI

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '../puf.csv')

def test_diff_start_year():
    from ogusa.scripts.execute import runner
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    output_base = "./OUTPUT"
    input_dir = "./OUTPUT"
    for year in [2013, 2017, 2026]:
        print('year = ', year)
        user_params = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
                       'start_year': year}
        runner(output_base=output_base, baseline_dir=input_dir, test=True,
               time_path=True, baseline=True, age_specific=False,
               user_params=user_params, run_micro=True,
               small_open=False, budget_balance=False, data=PUF_PATH)

#
# def test_get_micro_data_get_calculator():
#
#     reform = {
#     2017: {
#         '_II_rt1': [.09],
#         '_II_rt2': [.135],
#         '_II_rt3': [.225],
#         '_II_rt4': [.252],
#         '_II_rt5': [.297],
#         '_II_rt6': [.315],
#         '_II_rt7': [0.3564],
#     }, }
#
#     calc = get_calculator(baseline=False, calculator_start_year=2017,
#                           reform=reform, data=TAXDATA,
#                           weights=WEIGHTS, records_start_year=2009)
#     assert calc.current_year == 2017
#
#     reform = {
#     2017: {
#         '_II_rt1': [.09],
#         '_II_rt2': [.135],
#         '_II_rt3': [.225],
#         '_II_rt4': [.252],
#         '_II_rt5': [.297],
#         '_II_rt6': [.315],
#         '_II_rt7': [0.3564]
#     }, }
#
#     calc2 = get_calculator(baseline=False, calculator_start_year=2017,
#                            reform=reform, data=TAXDATA,
#                            weights=WEIGHTS, records_start_year=2009)
#     assert calc2.current_year == 2017
