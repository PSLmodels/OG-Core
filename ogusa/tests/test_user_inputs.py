import pytest
import os
from ogusa.execute import runner
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '..', 'puf.csv')


@pytest.mark.full_run
@pytest.mark.parametrize('frisch', [0.3, 0.4, 0.62],
                         ids=['Frisch 0.3', 'Frisch 0.4', 'Frisch 0.6'])
def test_frisch(frisch):
    output_base = os.path.join(CUR_PATH, "OUTPUT")
    input_dir = os.path.join(CUR_PATH, "OUTPUT")
    og_spec = {'frisch': frisch, 'debt_ratio_ss': 1.0}
    runner(output_base=output_base, baseline_dir=input_dir, test=False,
           time_path=False, baseline=True, og_spec=og_spec,
           run_micro=False, data=PUF_PATH)


@pytest.mark.full_run
@pytest.mark.parametrize('g_y_annual', [0.0, 0.04],
                         ids=['0.0', '0.04'])
def test_gy(g_y_annual):
    output_base = os.path.join(CUR_PATH, "OUTPUT")
    input_dir = os.path.join(CUR_PATH, "OUTPUT")
    og_spec = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
               'g_y_annual': g_y_annual}
    runner(output_base=output_base, baseline_dir=input_dir, test=False,
           time_path=False, baseline=True, og_spec=og_spec,
           run_micro=False, data=PUF_PATH)


@pytest.mark.full_run
@pytest.mark.parametrize('sigma', [1.3, 1.5, 1.7, 1.9],
                         ids=['sigma=1.3', 'sigma=1.5', 'sigma=1.7',
                              'sigma=1.9'])
def test_sigma(sigma):
    output_base = os.path.join(CUR_PATH, "OUTPUT")
    input_dir = os.path.join(CUR_PATH, "OUTPUT")
    og_spec = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
               'sigma': sigma}
    runner(output_base=output_base, baseline_dir=input_dir, test=False,
           time_path=False, baseline=True, og_spec=og_spec,
           run_micro=False, data=PUF_PATH)
