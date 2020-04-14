import pytest
import os
from ogusa.execute import runner
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '..', 'puf.csv')
TAX_FUNC_PATH = os.path.join(CUR_PATH, 'TxFuncEst_baseline.pkl')
OUTPUT_DIR = os.path.join(CUR_PATH, "OUTPUT")


@pytest.mark.full_run
@pytest.mark.parametrize('frisch', [0.3, 0.4, 0.62],
                         ids=['Frisch 0.3', 'Frisch 0.4', 'Frisch 0.6'])
def test_frisch(frisch):
    og_spec = {'frisch': frisch, 'debt_ratio_ss': 1.0}
    runner(output_base=OUTPUT_DIR, baseline_dir=OUTPUT_DIR, test=False,
           time_path=False, baseline=True, og_spec=og_spec,
           run_micro=False, tax_func_path=TAX_FUNC_PATH, data=PUF_PATH)


@pytest.mark.full_run
@pytest.mark.parametrize('g_y_annual', [0.0, 0.04],
                         ids=['0.0', '0.04'])
def test_gy(g_y_annual):
    og_spec = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
               'g_y_annual': g_y_annual}
    runner(output_base=OUTPUT_DIR, baseline_dir=OUTPUT_DIR, test=False,
           time_path=False, baseline=True, og_spec=og_spec,
           run_micro=False, tax_func_path=TAX_FUNC_PATH, data=PUF_PATH)


@pytest.mark.full_run
@pytest.mark.parametrize('sigma', [1.3, 1.5, 1.7, 1.9],
                         ids=['sigma=1.3', 'sigma=1.5', 'sigma=1.7',
                              'sigma=1.9'])
def test_sigma(sigma):
    og_spec = {'frisch': 0.41, 'debt_ratio_ss': 1.0,
               'sigma': sigma}
    runner(output_base=OUTPUT_DIR, baseline_dir=OUTPUT_DIR, test=False,
           time_path=False, baseline=True, og_spec=og_spec,
           run_micro=False, tax_func_path=TAX_FUNC_PATH, data=PUF_PATH)
