import pytest
import os
from ogusa import SS, TPI
from ogusa.execute import runner
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '../puf.csv')


@pytest.mark.full_run
@pytest.mark.parametrize(
        'year',
        [2013, 2017, 2026],
        ids=['2013', '2017', '2026'])
def test_diff_start_year(year):
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    output_base = os.path.join(CUR_PATH, "STARTYEAR_OUTPUT")
    input_dir = os.path.join(CUR_PATH, "STARTYEAR_OUTPUT")
    user_params = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
                   'start_year': year}
    runner(output_base=output_base, baseline_dir=input_dir, test=True,
           time_path=True, baseline=True, user_params=user_params,
           run_micro=True, data=PUF_PATH)
