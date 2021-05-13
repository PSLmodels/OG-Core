import multiprocessing
from distributed import Client, LocalCluster
import pytest
import pickle
import numpy as np
import os
import json
from ogusa import SS, TPI, utils, firm
import ogusa.aggregates as aggr
from ogusa.parameters import Specifications
NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

TEST_PARAM_DICT = json.load(open(os.path.join(CUR_PATH, 'testing_params.json')))


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


filename1 = 'intial_SS_values_baseline.pkl'
filename2 = 'intial_SS_values_reform.pkl'
filename3 = 'intial_SS_values_reform_base_spend.pkl'
param_updates1 = {'start_year': 2020}
param_updates2 = {'start_year': 2020}
param_updates3 = {'start_year': 2020, 'baseline_spending': True}


@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates1, filename1),
                          (False, param_updates2, filename2),
                          (False, param_updates3, filename3)],
                         ids=['Baseline', 'Reform',
                              'Reform, baseline_spending'])
def test_get_initial_SS_values(baseline, param_updates, filename,
                               dask_client):
    p = Specifications(baseline=baseline, client=dask_client,
                       num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.baseline_dir = os.path.join(CUR_PATH, 'test_io_data', 'OUTPUT')
    p.output_base = os.path.join(CUR_PATH, 'test_io_data', 'OUTPUT')
    test_tuple = TPI.get_initial_SS_values(p)
    (test_initial_values, test_ss_vars, test_theta,
     test_baseline_values) = test_tuple
    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', filename))

    (exp_initial_values, exp_ss_vars, exp_theta,
     exp_baseline_values) = expected_tuple
    (B0, b_sinit, b_splus1init, factor, initial_b,
     initial_n) = exp_initial_values
    B0 = aggr.get_B(exp_ss_vars['bssmat_splus1'], p, 'SS', True)
    initial_b = (exp_ss_vars['bssmat_splus1'] *
                 (exp_ss_vars['Bss'] / B0))
    B0 = aggr.get_B(initial_b, p, 'SS', True)
    b_sinit = np.array(list(np.zeros(p.J).reshape(1, p.J)) +
                       list(initial_b[:-1]))
    b_splus1init = initial_b
    exp_initial_values = (B0, b_sinit, b_splus1init, factor, initial_b,
                          initial_n)

    for i, v in enumerate(exp_initial_values):
        assert(np.allclose(test_initial_values[i], v, equal_nan=True))

    if p.baseline_spending:
        for i, v in enumerate(exp_baseline_values):
            assert(np.allclose(test_baseline_values[i], v, equal_nan=True))

    assert(np.allclose(test_theta, exp_theta))

    for k, v in exp_ss_vars.items():
        assert(np.allclose(test_ss_vars[k], v, equal_nan=True))


def test_firstdoughnutring(dask_client):
    # Test TPI.firstdoughnutring function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'firstdoughnutring_inputs.pkl'))
    guesses, r, w, b, BQ, TR, j, params = input_tuple
    income_tax_params, tpi_params, initial_b = params
    tpi_params = tpi_params + [True]
    p = Specifications(client=dask_client, num_workers=NUM_WORKERS)
    (p.J, p.S, p.T, p.BW, beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, tau_b, delta_tau,
     tau_payroll, tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, retire, p.mean_income_data, factor, h_wealth,
     p_wealth, m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params
    p.beta = np.ones(p.J) * beta
    p.Z = np.ones(p.T + p.S) * Z
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.tau_b = np.ones(p.T + p.S) * tau_b
    p.delta_tau = np.ones(p.T + p.S) * delta_tau
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    p.tax_func_type = 'DEP'
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params, (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params, (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params, (1, 0, 2))
    etr_params_old = p.etr_params
    mtrx_params_old = p.mtrx_params
    mtry_params_old = p.mtry_params
    p.etr_params = etr_params_old.copy()
    p.etr_params[:, :, 5] = etr_params_old[:, :, 6]
    p.etr_params[:, :, 6] = etr_params_old[:, :, 11]
    p.etr_params[:, :, 7] = etr_params_old[:, :, 5]
    p.etr_params[:, :, 8] = etr_params_old[:, :, 7]
    p.etr_params[:, :, 9] = etr_params_old[:, :, 8]
    p.etr_params[:, :, 10] = etr_params_old[:, :, 9]
    p.etr_params[:, :, 11] = etr_params_old[:, :, 10]
    p.mtrx_params = mtrx_params_old.copy()
    p.mtrx_params[:, :, 5] = mtrx_params_old[:, :, 6]
    p.mtrx_params[:, :, 6] = mtrx_params_old[:, :, 11]
    p.mtrx_params[:, :, 7] = mtrx_params_old[:, :, 5]
    p.mtrx_params[:, :, 8] = mtrx_params_old[:, :, 7]
    p.mtrx_params[:, :, 9] = mtrx_params_old[:, :, 8]
    p.mtrx_params[:, :, 10] = mtrx_params_old[:, :, 9]
    p.mtrx_params[:, :, 11] = mtrx_params_old[:, :, 10]
    p.mtry_params = mtry_params_old.copy()
    p.mtry_params[:, :, 5] = mtry_params_old[:, :, 6]
    p.mtry_params[:, :, 6] = mtry_params_old[:, :, 11]
    p.mtry_params[:, :, 7] = mtry_params_old[:, :, 5]
    p.mtry_params[:, :, 8] = mtry_params_old[:, :, 7]
    p.mtry_params[:, :, 9] = mtry_params_old[:, :, 8]
    p.mtry_params[:, :, 10] = mtry_params_old[:, :, 9]
    p.mtry_params[:, :, 11] = mtry_params_old[:, :, 10]
    p.lambdas = lambdas.reshape(p.J, 1)
    p.num_workers = 1
    bq = BQ / p.lambdas[j]
    tr = TR
    test_list = TPI.firstdoughnutring(guesses, r, w, bq, tr, theta,
                                      factor, j, initial_b, p)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data',
                     'firstdoughnutring_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


file_in1 = os.path.join(CUR_PATH, 'test_io_data',
                        'twist_doughnut_inputs_2.pkl')
file_in2 = os.path.join(CUR_PATH, 'test_io_data',
                        'twist_doughnut_inputs_S.pkl')
file_out1 = os.path.join(CUR_PATH, 'test_io_data',
                         'twist_doughnut_outputs_2.pkl')
file_out2 = os.path.join(CUR_PATH, 'test_io_data',
                         'twist_doughnut_outputs_S.pkl')


@pytest.mark.parametrize('file_inputs,file_outputs',
                         [(file_in1, file_out1), (file_in2, file_out2)],
                         ids=['s<S', 's==S'])
def test_twist_doughnut(file_inputs, file_outputs):
    '''
    Test TPI.twist_doughnut function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    '''
    input_tuple = utils.safe_read_pickle(file_inputs)
    (guesses, r, w, bq, tr, theta, factor, j, s, t, tau_c, etr_params,
     mtrx_params, mtry_params, initial_b, p) = input_tuple
    input_tuple = (guesses, r, w, bq, tr, theta, factor, j, s, t, tau_c,
                   etr_params, mtrx_params, mtry_params, initial_b, p)
    test_list = TPI.twist_doughnut(*input_tuple)
    expected_list = utils.safe_read_pickle(file_outputs)
    assert(np.allclose(np.array(test_list), np.array(expected_list)))


@pytest.mark.local
def test_inner_loop(dask_client):
    # Test TPI.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'tpi_inner_loop_inputs.pkl'))
    guesses, outer_loop_vars, params, j = input_tuple
    income_tax_params, tpi_params, initial_values, ind = params
    initial_values = initial_values[:-1]
    tpi_params = tpi_params
    p = Specifications(client=dask_client, num_workers=NUM_WORKERS)
    (p.J, p.S, p.T, p.BW, beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, tau_b, delta_tau,
     tau_payroll, tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, retire, p.mean_income_data, factor, h_wealth,
     p_wealth, m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params
    p.beta = np.ones(p.J) * beta
    p.eta = p.omega.reshape(p.T + p.S, p.S, 1) * p.lambdas.reshape(1, p.J)
    p.Z = np.ones(p.T + p.S) * Z
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.tau_b = np.ones(p.T + p.S) * tau_b
    p.delta_tau = np.ones(p.T + p.S) * delta_tau
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    p.tax_func_type = 'DEP'
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params, (1, 0, 2))[:p.T, :, :]
    p.mtrx_params = np.transpose(mtrx_params, (1, 0, 2))[:p.T, :, :]
    p.mtry_params = np.transpose(mtry_params, (1, 0, 2))[:p.T, :, :]
    p.etr_params = np.append(
        p.etr_params[:, :, :], np.tile(
            p.etr_params[-1, :, :], (p.T + p.S - p.etr_params.shape[0], 1, 1)),
        axis=0)
    p.mtrx_params = np.append(
        p.mtrx_params[:, :, :], np.tile(
            p.mtrx_params[-1, :, :], (p.T + p.S - p.mtrx_params.shape[0], 1, 1)),
        axis=0)
    p.mtry_params = np.append(
        p.mtry_params[:, :, :], np.tile(
            p.mtry_params[-1, :, :], (p.T + p.S - p.mtry_params.shape[0], 1, 1)),
        axis=0)
    etr_params_old = p.etr_params
    mtrx_params_old = p.mtrx_params
    mtry_params_old = p.mtry_params
    p.etr_params = etr_params_old.copy()
    p.etr_params[:, :, 5] = etr_params_old[:, :, 6]
    p.etr_params[:, :, 6] = etr_params_old[:, :, 11]
    p.etr_params[:, :, 7] = etr_params_old[:, :, 5]
    p.etr_params[:, :, 8] = etr_params_old[:, :, 7]
    p.etr_params[:, :, 9] = etr_params_old[:, :, 8]
    p.etr_params[:, :, 10] = etr_params_old[:, :, 9]
    p.etr_params[:, :, 11] = etr_params_old[:, :, 10]
    p.mtrx_params = mtrx_params_old.copy()
    p.mtrx_params[:, :, 5] = mtrx_params_old[:, :, 6]
    p.mtrx_params[:, :, 6] = mtrx_params_old[:, :, 11]
    p.mtrx_params[:, :, 7] = mtrx_params_old[:, :, 5]
    p.mtrx_params[:, :, 8] = mtrx_params_old[:, :, 7]
    p.mtrx_params[:, :, 9] = mtrx_params_old[:, :, 8]
    p.mtrx_params[:, :, 10] = mtrx_params_old[:, :, 9]
    p.mtrx_params[:, :, 11] = mtrx_params_old[:, :, 10]
    p.mtry_params = mtry_params_old.copy()
    p.mtry_params[:, :, 5] = mtry_params_old[:, :, 6]
    p.mtry_params[:, :, 6] = mtry_params_old[:, :, 11]
    p.mtry_params[:, :, 7] = mtry_params_old[:, :, 5]
    p.mtry_params[:, :, 8] = mtry_params_old[:, :, 7]
    p.mtry_params[:, :, 9] = mtry_params_old[:, :, 8]
    p.mtry_params[:, :, 10] = mtry_params_old[:, :, 9]
    p.mtry_params[:, :, 11] = mtry_params_old[:, :, 10]
    p.lambdas = lambdas.reshape(p.J, 1)
    p.num_workers = 1
    (K0, b_sinit, b_splus1init, factor, initial_b, initial_n,
     p.omega_S_preTP, initial_debt) = initial_values
    initial_values_in = (K0, b_sinit, b_splus1init, factor, initial_b,
                         initial_n)
    (r, K, BQ, TR) = outer_loop_vars
    wss = firm.get_w_from_r(r[-1], p, 'SS')
    w = np.ones(p.T + p.S) * wss
    w[:p.T] = firm.get_w_from_r(r[:p.T], p, 'TPI')
    outer_loop_vars_in = (r, w, r, BQ, TR, theta)

    guesses = (guesses[0], guesses[1])
    test_tuple = TPI.inner_loop(guesses, outer_loop_vars_in,
                                initial_values_in, j, ind, p)

    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data',
                     'tpi_inner_loop_outputs.pkl'))

    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_tuple[i], v))


g_n_ss = 0.0012907765315350872
imm_rates = np.load(os.path.join(CUR_PATH, 'imm_rates_tpi.npy'))
e = np.load(os.path.join(CUR_PATH, 'e_tpi.npy'))
omega_SS = np.load(os.path.join(CUR_PATH, 'omega_SS_tpi.npy'))
omega_S_preTP = np.load(os.path.join(CUR_PATH, 'omega_S_preTP_tpi.npy'))
omega = np.load(os.path.join(CUR_PATH, 'omega_tpi.npy'))
g_n = np.load(os.path.join(CUR_PATH, 'g_n_tpi.npy'))
rho = np.load(os.path.join(CUR_PATH, 'old_rho.npy'))
param_updates1 = {'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e,
                  'omega': omega, 'omega_S_preTP': omega_S_preTP,
                  'g_n': g_n, 'rho': rho}
filename1 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_baseline.pkl')
param_updates2 = {'budget_balance': True, 'alpha_G': [0.0],
                  'zeta_D': [0.0], 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e,
                  'omega': omega, 'omega_S_preTP': omega_S_preTP,
                  'g_n': g_n, 'rho': rho}
filename2 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_baseline_balanced_budget.pkl')
param_updates3 = {'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e,
                  'omega': omega, 'omega_S_preTP': omega_S_preTP,
                  'g_n': g_n, 'rho': rho}
filename3 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_reform.pkl')
param_updates4 = {'baseline_spending': True, 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e,
                  'omega': omega, 'omega_S_preTP': omega_S_preTP,
                  'g_n': g_n, 'rho': rho}
filename4 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_reform_baseline_spend.pkl')
param_updates5 = {'zeta_K': [1.0], 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e,
                  'omega': omega, 'omega_S_preTP': omega_S_preTP,
                  'g_n': g_n, 'rho': rho}
filename5 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_baseline_small_open.pkl')
param_updates6 = {'zeta_K': [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2],
                  'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e,
                  'omega': omega, 'omega_S_preTP': omega_S_preTP,
                  'g_n': g_n, 'rho': rho}
filename6 = os.path.join(
    CUR_PATH, 'test_io_data',
    'run_TPI_outputs_baseline_small_open_some_periods.pkl')
param_updates7 = {'delta_tau_annual': [0.0], 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e,
                  'omega': omega, 'omega_S_preTP': omega_S_preTP,
                  'g_n': g_n, 'rho': rho}
filename7 = os.path.join(
    CUR_PATH, 'test_io_data',
    'run_TPI_outputs_baseline_delta_tau0.pkl')


@pytest.mark.local
@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates2, filename2),
                          (True, param_updates1, filename1),
                          (False, param_updates3, filename3),
                          (False, param_updates4, filename4),
                          (True, param_updates5, filename5),
                          (True, param_updates6, filename6),
                          (True, param_updates7, filename7)],
                         ids=['Baseline, balanced budget', 'Baseline',
                              'Reform', 'Reform, baseline spending',
                              'Baseline, small open',
                              'Baseline, small open some periods',
                              'Baseline, delta_tau = 0'])
def test_run_TPI_full_run(baseline, param_updates, filename, tmp_path,
                          dask_client):
    '''
    Test TPI.run_TPI function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    '''
    baseline_dir = os.path.join(CUR_PATH, 'baseline')
    if baseline:
        output_base = baseline_dir
    else:
        output_base = os.path.join(CUR_PATH, 'reform')
    p = Specifications(baseline=baseline, baseline_dir=baseline_dir,
                       output_base=output_base, client=dask_client,
                       num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)

    # Need to run SS first to get results
    SS.ENFORCE_SOLUTION_CHECKS = False
    ss_outputs = SS.run_SS(p, None)

    if p.baseline:
        utils.mkdirs(os.path.join(p.baseline_dir, "SS"))
        ss_dir = os.path.join(p.baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)
    else:
        utils.mkdirs(os.path.join(p.output_base, "SS"))
        ss_dir = os.path.join(p.output_base, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)

    test_dict = TPI.run_TPI(p, None)
    expected_dict = utils.safe_read_pickle(filename)

    for k, v in expected_dict.items():
        try:
            assert(np.allclose(test_dict[k][:p.T], v[:p.T], rtol=1e-04, atol=1e-04))
        except ValueError:
            assert(np.allclose(test_dict[k][:p.T, :, :], v[:p.T, :, :], rtol=1e-04,
                               atol=1e-04))


param_updates1 = {}
filename1 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_baseline_2.pkl')
param_updates2 = {'budget_balance': True, 'alpha_G': [0.0]}
filename2 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_baseline_balanced_budget_2.pkl')
param_updates3 = {}
filename3 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_reform_2.pkl')
param_updates4 = {'baseline_spending': True}
filename4 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_reform_baseline_spend_2.pkl')


@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates2, filename2),
                          (True, param_updates1, filename1),
                          (False, param_updates3, filename3)],
                         ids=['Baseline, balanced budget', 'Baseline',
                              'Reform'])
def test_run_TPI(baseline, param_updates, filename, tmp_path,
                 dask_client):
    '''
    Test TPI.run_TPI function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    '''
    baseline_dir = os.path.join(CUR_PATH, 'baseline')
    if baseline:
        output_base = baseline_dir
    else:
        output_base = os.path.join(CUR_PATH, 'reform')
    p = Specifications(baseline=baseline, baseline_dir=baseline_dir,
                       output_base=output_base,
                       client=dask_client, num_workers=NUM_WORKERS)
    TEST_PARAM_DICT.update(param_updates)
    p.update_specifications(TEST_PARAM_DICT)
    p.maxiter = 2  # this test runs through just two iterations
    p.BW = 10
    dict_params = utils.safe_read_pickle(os.path.join(
        CUR_PATH, 'TxFuncEst_baseline.pkl'))
    num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
    num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
    num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
    p.mean_income_data = dict_params['tfunc_avginc'][0]
    p.etr_params = np.empty((p.T + p.S, p.S, num_etr_params))
    p.etr_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_etr_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.etr_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_etr_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_etr_params), axes=[1, 0, 2]), (p.T + p.S - p.BW, 1, 1))
    p.mtrx_params = np.empty((p.T + p.S, p.S, num_mtrx_params))
    p.mtrx_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtrx_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtrx_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p.T + p.S - p.BW, 1, 1))
    p.mtry_params = np.empty((p.T + p.S, p.S, num_mtry_params))
    p.mtry_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtry_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtry_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtry_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtry_params), axes=[1, 0, 2]), (p.T + p.S - p.BW, 1, 1))

    # Need to run SS first to get results
    SS.ENFORCE_SOLUTION_CHECKS = False
    ss_outputs = SS.run_SS(p, None)

    if p.baseline:
        utils.mkdirs(os.path.join(p.baseline_dir, "SS"))
        ss_dir = os.path.join(p.baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)
    else:
        utils.mkdirs(os.path.join(p.output_base, "SS"))
        ss_dir = os.path.join(p.output_base, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)

    TPI.ENFORCE_SOLUTION_CHECKS = False
    test_dict = TPI.run_TPI(p, None)
    expected_dict = utils.safe_read_pickle(filename)

    for k, v in expected_dict.items():
        try:
            assert(np.allclose(test_dict[k][:p.T], v[:p.T], rtol=1e-04,
                               atol=1e-04))
        except ValueError:
            assert(np.allclose(test_dict[k][:p.T, :, :], v[:p.T, :, :],
                               rtol=1e-04, atol=1e-04))


param_updates5 = {'zeta_K': [1.0]}
filename5 = os.path.join(CUR_PATH, 'test_io_data',
                         'run_TPI_outputs_baseline_small_open_2.pkl')
param_updates6 = {'zeta_K': [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 0.2]}
filename6 = filename = os.path.join(
    CUR_PATH, 'test_io_data',
    'run_TPI_outputs_baseline_small_open_some_periods_2.pkl')
param_updates7 = {'delta_tau_annual': [0.0]}
filename7 = filename = os.path.join(
    CUR_PATH, 'test_io_data',
    'run_TPI_outputs_baseline_delta_tau0_2.pkl')


@pytest.mark.local
@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates5, filename5),
                          (True, param_updates6, filename6),
                          (True, param_updates7, filename7),
                          (True, param_updates1, filename1),
                          (False, param_updates4, filename4)],
                         ids=['Baseline, small open',
                              'Baseline, small open for some periods',
                              'Baseline, delta_tau = 0', 'Baseline',
                              'Reform, baseline spending'])
def test_run_TPI_extra(baseline, param_updates, filename, tmp_path,
                       dask_client):
    '''
    Test TPI.run_TPI function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    '''
    baseline_dir = os.path.join(CUR_PATH, 'baseline')
    if baseline:
        output_base = baseline_dir
    else:
        output_base = os.path.join(CUR_PATH, 'reform')
    p = Specifications(baseline=baseline, baseline_dir=baseline_dir,
                       output_base=output_base,
                       client=dask_client, num_workers=NUM_WORKERS)
    TEST_PARAM_DICT.update(param_updates)
    p.update_specifications(TEST_PARAM_DICT)
    p.maxiter = 2  # this test runs through just two iterations
    p.BW = 10
    dict_params = utils.safe_read_pickle(os.path.join(
        CUR_PATH, 'TxFuncEst_baseline.pkl'))
    num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
    num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
    num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
    p.mean_income_data = dict_params['tfunc_avginc'][0]
    p.etr_params = np.empty((p.T + p.S, p.S, num_etr_params))
    p.etr_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_etr_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.etr_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_etr_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_etr_params), axes=[1, 0, 2]), (p.T + p.S - p.BW, 1, 1))
    p.mtrx_params = np.empty((p.T + p.S, p.S, num_mtrx_params))
    p.mtrx_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtrx_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtrx_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p.T + p.S - p.BW, 1, 1))
    p.mtry_params = np.empty((p.T + p.S, p.S, num_mtry_params))
    p.mtry_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtry_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtry_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtry_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtry_params), axes=[1, 0, 2]), (p.T + p.S - p.BW, 1, 1))

    # Need to run SS first to get results
    SS.ENFORCE_SOLUTION_CHECKS = False
    ss_outputs = SS.run_SS(p, None)

    if p.baseline:
        utils.mkdirs(os.path.join(p.baseline_dir, "SS"))
        ss_dir = os.path.join(p.baseline_dir, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)
    else:
        utils.mkdirs(os.path.join(p.output_base, "SS"))
        ss_dir = os.path.join(p.output_base, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(ss_outputs, f)

    TPI.ENFORCE_SOLUTION_CHECKS = False
    test_dict = TPI.run_TPI(p, None)

    expected_dict = utils.safe_read_pickle(filename)

    for k, v in expected_dict.items():
        try:
            assert(np.allclose(test_dict[k][:p.T], v[:p.T], rtol=1e-04,
                               atol=1e-04))
        except ValueError:
            assert(np.allclose(test_dict[k][:p.T, :, :], v[:p.T, :, :],
                               rtol=1e-04,
                               atol=1e-04))
