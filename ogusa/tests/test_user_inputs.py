import pytest
import os
import ogusa
from ogusa import SS, TPI

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '../puf.csv')


@pytest.mark.full_run
@pytest.mark.parametrize('frisch', [0.15, 0.4, 0.8],
                         ids=['Frisch 0.15', 'Frisch 0.4', 'Frisch 0.8'])
def test_frisch(frisch):
    from ogusa.scripts.execute import runner
    output_base = "./OUTPUT"
    input_dir = "./OUTPUT"
    user_params = {'frisch': frisch, 'debt_ratio_ss': 1.0}
    runner(output_base=output_base, baseline_dir=input_dir, test=False,
           time_path=False, baseline=True, age_specific=True,
           user_params=user_params, run_micro=False,
           small_open=False, budget_balance=False, data=PUF_PATH)


@pytest.mark.full_run
@pytest.mark.parametrize('g_y_annual', [-0.01, 0.05],
                         ids=['-0.01', '0.05'])
def test_gy(g_y_annual):
    from ogusa.scripts.execute import runner
    output_base = "./OUTPUT"
    input_dir = "./OUTPUT"
    user_params = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
                   'g_y_annual': g_y_annual}
    runner(output_base=output_base, baseline_dir=input_dir, test=False,
           time_path=False, baseline=True, age_specific=True,
           user_params=user_params, run_micro=False,
           small_open=False, budget_balance=False, data=PUF_PATH)
