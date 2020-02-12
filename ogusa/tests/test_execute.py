import pytest
from ogusa import SS, TPI
from ogusa.execute import runner


@pytest.mark.full_run
def test_runner_baseline():
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    BASELINE_DIR = './OUTPUT_BASELINE_'
    runner(output_base=BASELINE_DIR, baseline_dir=BASELINE_DIR,
           test=True, time_path=True, baseline=True, run_micro=False,
           data='cps')


def test_runner_reform():
    # Monkey patch enforcement flag since small data won't pass checks
    SS.ENFORCE_SOLUTION_CHECKS = False
    TPI.ENFORCE_SOLUTION_CHECKS = False
    BASELINE_DIR = './OUTPUT_BASELINE_'
    runner(output_base=BASELINE_DIR, baseline_dir=BASELINE_DIR,
           test=True, time_path=False, baseline=False, run_micro=False,
           data='cps')
