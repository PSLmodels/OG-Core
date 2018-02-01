import pytest
import os
import ogusa
from ogusa import SS, TPI

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '../puf.csv')

@pytest.mark.full_run
def test_frisch():
    from ogusa.scripts.execute import runner
    output_base = "./OUTPUT"
    input_dir = "./OUTPUT"
    for frisch in [0.1, 0.4, 0.8]:
        print('Frisch elasticity = ', frisch)
        user_params = {'frisch': frisch, 'debt_ratio_ss': 1.0,
                       'start_year': 2018}
        runner(output_base=output_base, baseline_dir=input_dir, test=False,
               time_path=False, baseline=True, age_specific=True,
               user_params=user_params, run_micro=False,
               small_open=False, budget_balance=False, data=PUF_PATH)

@pytest.mark.full_run
def test_gy():
    from ogusa.scripts.execute import runner
    # # Monkey patch enforcement flag since small data won't pass checks
    # SS.ENFORCE_SOLUTION_CHECKS = True
    # TPI.ENFORCE_SOLUTION_CHECKS = True
    output_base = "./OUTPUT"
    input_dir = "./OUTPUT"
    for gy in [-0.01,  0.0, 0.03, 0.08, 0.2, 1.0]:
        print('Growth rates = ', gy)
        user_params = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
                       'start_year': 2018, 'g_y_annual': gy}
        runner(output_base=output_base, baseline_dir=input_dir, test=False,
               time_path=False, baseline=True, age_specific=True,
               user_params=user_params, run_micro=False,
               small_open=False, budget_balance=False, data=PUF_PATH)
