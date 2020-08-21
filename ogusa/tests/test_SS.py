'''
Test of steady-state module
'''

import multiprocessing
from distributed import Client, LocalCluster
import pytest
import numpy as np
import os
from ogusa import SS, utils, aggregates, household, execute, constants
from ogusa.parameters import Specifications
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
NUM_WORKERS = min(multiprocessing.cpu_count(), 7)


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'SS_fsolve_inputs.pkl'))
guesses_in, params = input_tuple
params = params + (None, 1)
(bssmat, nssmat, chi_params, ss_params, income_tax_params,
 iterative_params, small_open_params, client, num_workers) = params
p1 = Specifications()
(p1.J, p1.S, p1.T, p1.BW, p1.beta, p1.sigma, p1.alpha, p1.gamma, p1.epsilon,
 Z, p1.delta, p1.ltilde, p1.nu, p1.g_y, p1.g_n_ss, tau_payroll,
 tau_bq, p1.rho, p1.omega_SS, p1.budget_balance, alpha_T,
 p1.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p1.e,
 retire, p1.mean_income_data, h_wealth, p_wealth, m_wealth,
 p1.b_ellipse, p1.upsilon) = ss_params
p1.eta = (p1.omega_SS.reshape(p1.S, 1) *
          p1.lambdas.reshape(1, p1.J)).reshape(1, p1.S, p1.J)
p1.Z = np.ones(p1.T + p1.S) * Z
p1.tau_bq = np.ones(p1.T + p1.S) * 0.0
p1.tau_payroll = np.ones(p1.T + p1.S) * tau_payroll
p1.alpha_T = np.ones(p1.T + p1.S) * alpha_T
p1.tau_b = np.ones(p1.T + p1.S) * tau_b
p1.delta_tau = np.ones(p1.T + p1.S) * delta_tau
p1.h_wealth = np.ones(p1.T + p1.S) * h_wealth
p1.p_wealth = np.ones(p1.T + p1.S) * p_wealth
p1.m_wealth = np.ones(p1.T + p1.S) * m_wealth
p1.retire = (np.ones(p1.T + p1.S) * retire).astype(int)
p1.lambdas = lambdas.reshape(p1.J, 1)
p1.imm_rates = imm_rates.reshape(1, p1.S)
p1.tax_func_type = 'DEP'
p1.zeta_K = np.array([0.0])
p1.zeta_D = np.array([0.0])
p1.initial_foreign_debt_ratio = 0.0
p1.r_gov_shift = np.array([0.0])
p1.start_year = 2019
p1.baseline = False
p1.baseline = True
p1.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
    income_tax_params
p1.etr_params = np.transpose(etr_params.reshape(
    p1.S, 1, etr_params.shape[-1]), (1, 0, 2))
p1.mtrx_params = np.transpose(mtrx_params.reshape(
    p1.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
p1.mtry_params = np.transpose(mtry_params.reshape(
    p1.S, 1, mtry_params.shape[-1]), (1, 0, 2))
p1.maxiter, p1.mindist_SS = iterative_params
p1.chi_b, p1.chi_n = chi_params
small_open, firm_r, hh_r = small_open_params
p1.world_int_rate = np.ones(p1.T + p1.S) * firm_r
p1.num_workers = 1
BQ1 = np.ones((p1.J)) * 0.00019646295986015257
guesses1 = [guesses_in[0]] + list(BQ1) + [guesses_in[1]] + [guesses_in[2]]
args1 = (bssmat, nssmat, None, None, p1, client)
expected1 = np.array([0.06858352869423862, 0.0157424466869841,
                      0.020615373965602958, 0.02225725864386594,
                      0.01359148091834126, 0.01604345296066714,
                      0.018393166562212734, 0.0033730256425707566,
                      -0.07014671511880782, 0.05424969771042221])

input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'SS_fsolve_reform_inputs.pkl'))
guesses_in2, params = input_tuple
params = params + (None, 1)
(bssmat, nssmat, chi_params, ss_params, income_tax_params,
 iterative_params, factor, small_open_params, client,
 num_workers) = params
p2 = Specifications()
(p2.J, p2.S, p2.T, p2.BW, p2.beta, p2.sigma, p2.alpha, p2.gamma, p2.epsilon,
 Z, p2.delta, p2.ltilde, p2.nu, p2.g_y, p2.g_n_ss, tau_payroll,
 tau_bq, p2.rho, p2.omega_SS, p2.budget_balance, alpha_T,
 p2.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p2.e,
 retire, p2.mean_income_data, h_wealth, p_wealth, m_wealth,
 p2.b_ellipse, p2.upsilon) = ss_params
p2.eta = (p2.omega_SS.reshape(p2.S, 1) *
          p2.lambdas.reshape(1, p2.J)).reshape(1, p2.S, p2.J)
p2.Z = np.ones(p2.T + p2.S) * Z
p2.tau_bq = np.ones(p2.T + p2.S) * 0.0
p2.tau_payroll = np.ones(p2.T + p2.S) * tau_payroll
p2.alpha_T = np.ones(p2.T + p2.S) * alpha_T
p2.tau_b = np.ones(p2.T + p2.S) * tau_b
p2.delta_tau = np.ones(p2.T + p2.S) * delta_tau
p2.h_wealth = np.ones(p2.T + p2.S) * h_wealth
p2.p_wealth = np.ones(p2.T + p2.S) * p_wealth
p2.m_wealth = np.ones(p2.T + p2.S) * m_wealth
p2.retire = (np.ones(p2.T + p2.S) * retire).astype(int)
p2.lambdas = lambdas.reshape(p2.J, 1)
p2.imm_rates = imm_rates.reshape(1, p2.S)
p2.tax_func_type = 'DEP'
p2.zeta_K = np.array([0.0])
p2.zeta_D = np.array([0.0])
p2.initial_foreign_debt_ratio = 0.0
p2.r_gov_shift = np.array([0.0])
p2.start_year = 2019
p2.baseline = False
p2.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
    income_tax_params
p2.etr_params = np.transpose(etr_params.reshape(
    p2.S, 1, etr_params.shape[-1]), (1, 0, 2))
p2.mtrx_params = np.transpose(mtrx_params.reshape(
    p2.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
p2.mtry_params = np.transpose(mtry_params.reshape(
    p2.S, 1, mtry_params.shape[-1]), (1, 0, 2))
p2.maxiter, p2.mindist_SS = iterative_params
p2.chi_b, p2.chi_n = chi_params
small_open, firm_r, hh_r = small_open_params
p2.world_int_rate = np.ones(p2.T + p2.S) * firm_r
p2.num_workers = 1
BQ2 = np.ones((p2.J)) * 0.00019646295986015257
guesses2 = [guesses_in2[0]] + list(BQ2) + [guesses_in2[1]]
args2 = (bssmat, nssmat, None, factor, p2, client)
expected2 = np.array([0.016757343762877415, 0.01435509375160598,
                      0.019450554513959047, 0.020767620498430173,
                      0.012363834824786278, 0.014583252714123543,
                      0.01716246184210253, 0.003106382567096101,
                      0.0016798428580572025])

input_tuple = utils.safe_read_pickle(
    os.path.join(
        CUR_PATH,
        'test_io_data', 'SS_fsolve_reform_baselinespend_inputs.pkl'))
guesses_in3, params = input_tuple
params = params + (None, 1)
(bssmat, nssmat, TR_ss, chi_params, ss_params, income_tax_params,
 iterative_params, factor_ss, small_open_params, client,
 num_workers) = params
p3 = Specifications()
(p3.J, p3.S, p3.T, p3.BW, p3.beta, p3.sigma, p3.alpha, p3.gamma, p3.epsilon,
 Z, p3.delta, p3.ltilde, p3.nu, p3.g_y, p3.g_n_ss, tau_payroll,
 tau_bq, p3.rho, p3.omega_SS, p3.budget_balance, alpha_T,
 p3.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p3.e,
 retire, p3.mean_income_data, h_wealth, p_wealth, m_wealth,
 p3.b_ellipse, p3.upsilon) = ss_params
p3.eta = (p3.omega_SS.reshape(p3.S, 1) *
          p3.lambdas.reshape(1, p3.J)).reshape(1, p3.S, p3.J)
p3.Z = np.ones(p3.T + p3.S) * Z
p3.tau_bq = np.ones(p3.T + p3.S) * 0.0
p3.tau_payroll = np.ones(p3.T + p3.S) * tau_payroll
p3.alpha_T = np.ones(p3.T + p3.S) * alpha_T
p3.tau_b = np.ones(p3.T + p3.S) * tau_b
p3.delta_tau = np.ones(p3.T + p3.S) * delta_tau
p3.h_wealth = np.ones(p3.T + p3.S) * h_wealth
p3.p_wealth = np.ones(p3.T + p3.S) * p_wealth
p3.m_wealth = np.ones(p3.T + p3.S) * m_wealth
p3.retire = (np.ones(p3.T + p3.S) * retire).astype(int)
p3.lambdas = lambdas.reshape(p3.J, 1)
p3.imm_rates = imm_rates.reshape(1, p3.S)
p3.tax_func_type = 'DEP'
p3.zeta_K = np.array([0.0])
p3.zeta_D = np.array([0.0])
p3.initial_foreign_debt_ratio = 0.0
p3.r_gov_shift = np.array([0.0])
p3.start_year = 2019
p3.baseline = False
p3.baseline_spending = True
p3.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
    income_tax_params
p3.etr_params = np.transpose(etr_params.reshape(
    p3.S, 1, etr_params.shape[-1]), (1, 0, 2))
p3.mtrx_params = np.transpose(mtrx_params.reshape(
    p3.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
p3.mtry_params = np.transpose(mtry_params.reshape(
    p3.S, 1, mtry_params.shape[-1]), (1, 0, 2))
p3.maxiter, p3.mindist_SS = iterative_params
p3.chi_b, p3.chi_n = chi_params
small_open, firm_r, hh_r = small_open_params
p3.world_int_rate = np.ones(p3.T + p3.S) * firm_r
p3.num_workers = 1
BQ3 = np.ones((p3.J)) * 0.00019646295986015257
guesses3 = [guesses_in3[0]] + list(BQ3) + [guesses_in3[1]]
args3 = (bssmat, nssmat, TR_ss, factor_ss, p3, client)
expected3 = np.array([0.016757345515050044, 0.014355093775301265,
                      0.019450554545951612, 0.020767620470159415,
                      0.01236383484523906, 0.014583252738190352,
                      0.01716246187036924, 0.0031063825724743474,
                      0.018664915456857223])

input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'SS_fsolve_inputs.pkl'))
guesses_in, params = input_tuple
params = params + (None, 1)
(bssmat, nssmat, chi_params, ss_params, income_tax_params,
 iterative_params, small_open_params, client, num_workers) = params
p4 = Specifications()
new_param_values = {
    'zeta_D': [0.4],
    'zeta_K': [0.2]
}
p4.update_specifications(new_param_values)
(p4.J, p4.S, p4.T, p4.BW, p4.beta, p4.sigma, p4.alpha, p4.gamma, p4.epsilon,
 Z, p4.delta, p4.ltilde, p4.nu, p4.g_y, p4.g_n_ss, tau_payroll,
 tau_bq, p4.rho, p4.omega_SS, p4.budget_balance, alpha_T,
 p4.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p4.e,
 retire, p4.mean_income_data, h_wealth, p_wealth, m_wealth,
 p4.b_ellipse, p4.upsilon) = ss_params
p4.eta = (p4.omega_SS.reshape(p4.S, 1) *
          p4.lambdas.reshape(1, p4.J)).reshape(1, p4.S, p4.J)
p4.Z = np.ones(p4.T + p4.S) * Z
p4.tau_bq = np.ones(p4.T + p4.S) * 0.0
p4.tau_payroll = np.ones(p4.T + p4.S) * tau_payroll
p4.alpha_T = np.ones(p4.T + p4.S) * alpha_T
p4.tau_b = np.ones(p4.T + p4.S) * tau_b
p4.delta_tau = np.ones(p4.T + p4.S) * delta_tau
p4.h_wealth = np.ones(p4.T + p4.S) * h_wealth
p4.p_wealth = np.ones(p4.T + p4.S) * p_wealth
p4.m_wealth = np.ones(p4.T + p4.S) * m_wealth
p4.retire = (np.ones(p4.T + p4.S) * retire).astype(int)
p4.lambdas = lambdas.reshape(p4.J, 1)
p4.imm_rates = imm_rates.reshape(1, p4.S)
p4.tax_func_type = 'DEP'
p4.baseline = True
p4.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
    income_tax_params
p4.etr_params = np.transpose(etr_params.reshape(
    p4.S, 1, etr_params.shape[-1]), (1, 0, 2))
p4.mtrx_params = np.transpose(mtrx_params.reshape(
    p4.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
p4.mtry_params = np.transpose(mtry_params.reshape(
    p4.S, 1, mtry_params.shape[-1]), (1, 0, 2))
p4.maxiter, p4.mindist_SS = iterative_params
p4.chi_b, p4.chi_n = chi_params
p4.num_workers = 1
BQ4 = np.ones((p4.J)) * 0.00019646295986015257
guesses4 = [guesses_in[0]] + list(BQ4) + [guesses_in[1]] + [guesses_in[2]]
args4 = (bssmat, nssmat, None, None, p4, client)
expected4 = np.array([0.028883118596741857, 0.014511613659907734,
                      0.019044550115699707, 0.02065761642516883,
                      0.012627889727738099, 0.014940813299332474,
                      0.016999514675696315, 0.0030878921261591253,
                      -0.06125508233576064, 0.06697984483743183])

input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data', 'SS_fsolve_inputs.pkl'))
guesses_in, params = input_tuple
params = params + (None, 1)
(bssmat, nssmat, chi_params, ss_params, income_tax_params,
 iterative_params, small_open_params, client, num_workers) = params
p5 = Specifications()
(p5.J, p5.S, p5.T, p5.BW, p5.beta, p5.sigma, p5.alpha, p5.gamma, p5.epsilon,
 Z, p5.delta, p5.ltilde, p5.nu, p5.g_y, p5.g_n_ss, tau_payroll,
 tau_bq, p5.rho, p5.omega_SS, p5.budget_balance, alpha_T,
 p5.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p5.e,
 retire, p5.mean_income_data, h_wealth, p_wealth, m_wealth,
 p5.b_ellipse, p5.upsilon) = ss_params
p5.eta = (p5.omega_SS.reshape(p5.S, 1) *
          p5.lambdas.reshape(1, p5.J)).reshape(1, p5.S, p5.J)
p5.zeta_K = np.ones(p5.T + p5.S) * 1.0
p5.world_int_rate = np.ones(p5.T + p5.S) * 0.05
p5.Z = np.ones(p5.T + p5.S) * Z
p5.tau_bq = np.ones(p5.T + p5.S) * 0.0
p5.tau_payroll = np.ones(p5.T + p5.S) * tau_payroll
p5.alpha_T = np.ones(p5.T + p5.S) * alpha_T
p5.tau_b = np.ones(p5.T + p5.S) * tau_b
p5.delta_tau = np.ones(p5.T + p5.S) * delta_tau
p5.h_wealth = np.ones(p5.T + p5.S) * h_wealth
p5.p_wealth = np.ones(p5.T + p5.S) * p_wealth
p5.m_wealth = np.ones(p5.T + p5.S) * m_wealth
p5.retire = (np.ones(p5.T + p5.S) * retire).astype(int)
p5.lambdas = lambdas.reshape(p5.J, 1)
p5.imm_rates = imm_rates.reshape(1, p5.S)
p5.tax_func_type = 'DEP'
p5.baseline = True
p5.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
    income_tax_params
p5.etr_params = np.transpose(etr_params.reshape(
    p5.S, 1, etr_params.shape[-1]), (1, 0, 2))
p5.mtrx_params = np.transpose(mtrx_params.reshape(
    p5.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
p5.mtry_params = np.transpose(mtry_params.reshape(
    p5.S, 1, mtry_params.shape[-1]), (1, 0, 2))
p5.maxiter, p5.mindist_SS = iterative_params
p5.chi_b, p5.chi_n = chi_params
p5.num_workers = 1
BQ5 = np.ones((p5.J)) * 0.00019646295986015257
guesses5 = [guesses_in[0]] + list(BQ5) + [guesses_in[1]] + [guesses_in[2]]
args5 = (bssmat, nssmat, None, None, p5, client)
expected5 = np.array([
    0.010000000000000002, 0.014267680343491978, 0.01872543808956217,
    0.02031175170902025, 0.012415197955534063, 0.014689761710816571,
    0.016714319543654727, 0.003033421127052222, -0.05535189569237248,
    0.06644990365361061])

p6 = Specifications()
(p6.J, p6.S, p6.T, p6.BW, p6.beta, p6.sigma, p6.alpha, p6.gamma, p6.epsilon,
 Z, p6.delta, p6.ltilde, p6.nu, p6.g_y, p6.g_n_ss, tau_payroll,
 tau_bq, p6.rho, p6.omega_SS, p6.budget_balance, alpha_T,
 p6.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p6.e,
 retire, p6.mean_income_data, h_wealth, p_wealth, m_wealth,
 p6.b_ellipse, p6.upsilon) = ss_params
p6.eta = (p6.omega_SS.reshape(p6.S, 1) *
          p6.lambdas.reshape(1, p6.J)).reshape(1, p6.S, p6.J)
p6.Z = np.ones(p6.T + p6.S) * Z
p6.tau_bq = np.ones(p6.T + p6.S) * 0.0
p6.tau_payroll = np.ones(p6.T + p6.S) * tau_payroll
p6.alpha_T = np.ones(p6.T + p6.S) * alpha_T
p6.tau_b = np.ones(p6.T + p6.S) * tau_b
p6.delta_tau = np.ones(p6.T + p6.S) * 0.0
p6.h_wealth = np.ones(p6.T + p6.S) * h_wealth
p6.p_wealth = np.ones(p6.T + p6.S) * p_wealth
p6.m_wealth = np.ones(p6.T + p6.S) * m_wealth
p6.retire = (np.ones(p6.T + p6.S) * retire).astype(int)
p6.lambdas = lambdas.reshape(p6.J, 1)
p6.imm_rates = imm_rates.reshape(1, p6.S)
p6.tax_func_type = 'DEP'
p6.zeta_K = np.array([0.0])
p6.zeta_D = np.array([0.0])
p6.initial_foreign_debt_ratio = 0.0
p6.r_gov_shift = np.array([0.0])
p6.start_year = 2019
p6.baseline = False
p6.baseline = True
p6.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
    income_tax_params
p6.etr_params = np.transpose(etr_params.reshape(
    p6.S, 1, etr_params.shape[-1]), (1, 0, 2))
p6.mtrx_params = np.transpose(mtrx_params.reshape(
    p6.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
p6.mtry_params = np.transpose(mtry_params.reshape(
    p6.S, 1, mtry_params.shape[-1]), (1, 0, 2))
p6.maxiter, p6.mindist_SS = iterative_params
p6.chi_b, p6.chi_n = chi_params
small_open, firm_r, hh_r = small_open_params
p6.world_int_rate = np.ones(p6.T + p6.S) * firm_r
p6.num_workers = 1
BQ6 = np.ones((p6.J)) * 0.00019646295986015257
guesses6 = [guesses_in[0]] + list(BQ6) + [guesses_in[1]] + [guesses_in[2]]
args6 = (bssmat, nssmat, None, None, p6, client)
expected6 = np.array([
    0.07215505148288945, 0.015648438358930053, 0.020459875721456536,
    0.022060136475180223, 0.013462001740302838, 0.015934473464005214,
    0.018219674546277976, 0.0033400842357827754, -0.07119442016669195,
    0.05562671567906091])


@pytest.mark.parametrize('guesses,args,expected',
                         [(guesses1, args1, expected1),
                          (guesses2, args2, expected2),
                          (guesses3, args3, expected3),
                          (guesses4, args4, expected4),
                          (guesses5, args5, expected5),
                          (guesses6, args6, expected6)],
                         ids=['Baseline, Closed', 'Reform, Closed',
                              'Reform, Baseline spending=True, Closed',
                              'Baseline, Partial Open',
                              'Baseline, Small Open',
                              'Baseline, Closed, delta_tau = 0'])
def test_SS_fsolve(guesses, args, expected):
    '''
    Test SS.SS_fsolve function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    '''
    test_list = SS.SS_fsolve(guesses, *args)
    assert(np.allclose(np.array(test_list), np.array(expected),
                       atol=1e-6))


param_updates1 = {'start_year': 2020}
filename1 = 'SS_solver_outputs_baseline.pkl'
param_updates2 = {'start_year': 2020, 'budget_balance': True,
                  'alpha_G': [0.0]}
filename2 = 'SS_solver_outputs_baseline_budget_balance.pkl'
param_updates3 = {'baseline_spending': True}
filename3 = 'SS_solver_outputs_reform_baseline_spending.pkl'
param_updates4 = {'start_year': 2020, 'zeta_K': [1.0]}
filename4 = 'SS_solver_outputs_baseline_small_open.pkl'


# Note that chaning the order in which these tests are run will cause
# failures for the baseline spending=True tests which depend on the
# output of the baseline run just prior
@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates1, filename1),
                          (True, param_updates2, filename2),
                          (False, param_updates3, filename3),
                          (True, param_updates4, filename4)],
                         ids=['Baseline', 'Baseline, budget balance',
                              'Reform, baseline spending=True',
                              'Baseline, small open'])
def test_SS_solver(baseline, param_updates, filename, dask_client):
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, client=dask_client,
                       num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    p.get_tax_function_parameters(None, run_micro=False)
    b_guess = np.ones((p.S, p.J)) * 0.07
    n_guess = np.ones((p.S, p.J)) * .35 * p.ltilde
    if p.zeta_K[-1] == 1.0:
        rguess = p.world_int_rate[-1]
    else:
        rguess = 0.06483431412921253
    TRguess = 0.05738932081035772
    factorguess = 139355.1547340256
    BQguess = aggregates.get_BQ(rguess, b_guess, None, p, 'SS', False)
    Yguess = 0.6376591201150815

    test_dict = SS.SS_solver(b_guess, n_guess, rguess, BQguess, TRguess,
                             factorguess, Yguess, p, None, False)
    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', filename))

    for k, v in expected_dict.items():
        print('Testing ', k)
        assert(np.allclose(test_dict[k], v, atol=1e-07, equal_nan=True))


param_updates5 = {'start_year': 2020, 'zeta_K': [1.0],
                  'budget_balance': True, 'alpha_G': [0.0]}
filename5 = 'SS_solver_outputs_baseline_small_open_budget_balance.pkl'
param_updates6 = {'delta_tau_annual': [0.0], 'zeta_K': [0.0],
                  'zeta_D': [0.0]}
filename6 = 'SS_solver_outputs_baseline_delta_tau0.pkl'


@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates5, filename5),
                          (True, param_updates6, filename6)],
                         ids=['Baseline, small open, budget balance',
                              'Baseline, delta_tau = 0'])
@pytest.mark.full_run
def test_SS_solver_extra(baseline, param_updates, filename, dask_client):
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, client=dask_client,
                       num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    p.get_tax_function_parameters(None, run_micro=False)
    b_guess = np.ones((p.S, p.J)) * 0.07
    n_guess = np.ones((p.S, p.J)) * .35 * p.ltilde
    if p.zeta_K[-1] == 1.0:
        rguess = p.world_int_rate[-1]
    else:
        rguess = 0.06483431412921253
    TRguess = 0.05738932081035772
    factorguess = 139355.1547340256
    BQguess = aggregates.get_BQ(rguess, b_guess, None, p, 'SS', False)
    Yguess = 0.6376591201150815

    test_dict = SS.SS_solver(b_guess, n_guess, rguess, BQguess, TRguess,
                             factorguess, Yguess, p, None, False)
    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', filename))

    for k, v in expected_dict.items():
        print('Testing ', k)
        assert(np.allclose(test_dict[k], v, atol=1e-07, equal_nan=True))


param_updates1 = {'start_year': 2020, 'zeta_K': [1.0]}
filename1 = 'inner_loop_outputs_baseline_small_open.pkl'
param_updates2 = {'start_year': 2020, 'budget_balance': True,
                  'alpha_G': [0.0]}
filename2 = 'inner_loop_outputs_baseline_balance_budget.pkl'
param_updates3 = {'start_year': 2020}
filename3 = 'inner_loop_outputs_baseline.pkl'
param_updates4 = {'start_year': 2020}
filename4 = 'inner_loop_outputs_reform.pkl'
param_updates5 = {'start_year': 2020, 'baseline_spending': True}
filename5 = 'inner_loop_outputs_reform_baselinespending.pkl'


@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates1, filename1),
                          (True, param_updates2, filename2),
                          (True, param_updates3, filename3),
                          (False, param_updates4, filename4),
                          (False, param_updates5, filename5)],
                         ids=['Baseline, Small Open',
                              'Baseline, Balanced Budget',
                              'Baseline', 'Reform',
                              'Reform, baseline spending'])
def test_inner_loop(baseline, param_updates, filename, dask_client):
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, client=dask_client,
                       num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    p.get_tax_function_parameters(None, run_micro=False)
    bssmat = np.ones((p.S, p.J)) * 0.07
    nssmat = np.ones((p.S, p.J)) * .4 * p.ltilde
    if p.zeta_K[-1] == 1.0:
        r = p.world_int_rate[-1]
    else:
        r = 0.05
    TR = 0.12
    Y = 1.3
    factor = 100000
    BQ = np.ones(p.J) * 0.00019646295986015257
    if p.budget_balance:
        outer_loop_vars = (bssmat, nssmat, r, BQ, TR, factor)
    else:
        outer_loop_vars = (bssmat, nssmat, r, BQ, Y, TR, factor)
    test_tuple = SS.inner_loop(outer_loop_vars, p, None)
    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', filename))
    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_tuple[i], v, atol=1e-05))


param_updates6 = {'delta_tau_annual': [0.0], 'zeta_K': [0.0],
                  'zeta_D': [0.0]}
filename6 = 'inner_loop_outputs_baseline_delta_tau0.pkl'


@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(False, param_updates6, filename6)],
                         ids=['Baseline, delta_tau = 0'])
@pytest.mark.full_run
def test_inner_loop_extra(baseline, param_updates, filename, dask_client):
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, client=dask_client,
                       num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    p.get_tax_function_parameters(None, run_micro=False)
    bssmat = np.ones((p.S, p.J)) * 0.07
    nssmat = np.ones((p.S, p.J)) * .4 * p.ltilde
    if p.zeta_K[-1] == 1.0:
        r = p.world_int_rate[-1]
    else:
        r = 0.05
    TR = 0.12
    Y = 1.3
    factor = 100000
    BQ = np.ones(p.J) * 0.00019646295986015257
    if p.budget_balance:
        outer_loop_vars = (bssmat, nssmat, r, BQ, TR, factor)
    else:
        outer_loop_vars = (bssmat, nssmat, r, BQ, Y, TR, factor)
    test_tuple = SS.inner_loop(outer_loop_vars, p, None)
    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', filename))
    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_tuple[i], v, atol=1e-05))


def test_euler_equation_solver(dask_client):
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'euler_eqn_solver_inputs.pkl'))
    (guesses, params) = input_tuple
    p = Specifications(client=dask_client, num_workers=NUM_WORKERS)
    (r, w, TR, factor, j, p.J, p.S, p.beta, p.sigma, p.ltilde, p.g_y,
     p.g_n_ss, tau_payroll, retire, p.mean_income_data, h_wealth,
     p_wealth, m_wealth, p.b_ellipse, p.upsilon, j, p.chi_b,
     p.chi_n, tau_bq, p.rho, lambdas, p.omega_SS, p.e,
     p.analytical_mtrs, etr_params, mtrx_params, mtry_params) = params
    p.eta = (p.omega_SS.reshape(p.S, 1) *
             p.lambdas.reshape(1, p.J)).reshape(1, p.S, p.J)
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    p.etr_params = np.transpose(etr_params.reshape(
        p.S, 1, etr_params.shape[-1]), (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params.reshape(
        p.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params.reshape(
        p.S, 1, mtry_params.shape[-1]), (1, 0, 2))
    p.tax_func_type = 'DEP'
    p.lambdas = lambdas.reshape(p.J, 1)
    b_splus1 = np.array(guesses[:p.S]).reshape(p.S, 1) + 0.005
    BQ = aggregates.get_BQ(r, b_splus1, j, p, 'SS', False)
    bq = household.get_bq(BQ, j, p, 'SS')
    tr = household.get_tr(TR, j, p, 'SS')
    args = (r, w, bq, tr, factor, j, p)
    test_list = SS.euler_equation_solver(guesses, *args)

    expected_list = np.array([
        -3.62741663e+00, -6.30068841e+00, -6.76592886e+00,
        -6.97731223e+00, -7.05777777e+00, -6.57305440e+00,
        -7.11553046e+00, -7.30569622e+00, -7.45808107e+00,
        -7.89984062e+00, -8.11466111e+00, -8.28230086e+00,
        -8.79253862e+00, -8.86994311e+00, -9.31299476e+00,
        -9.80834199e+00, -9.97333771e+00, -1.08349979e+01,
        -1.13199826e+01, -1.22890930e+01, -1.31550471e+01,
        -1.42753713e+01, -1.55721098e+01, -1.73811490e+01,
        -1.88856303e+01, -2.09570569e+01, -2.30559500e+01,
        -2.52127149e+01, -2.76119605e+01, -3.03141128e+01,
        -3.30900203e+01, -3.62799730e+01, -3.91169706e+01,
        -4.24246421e+01, -4.55740527e+01, -4.92914871e+01,
        -5.30682805e+01, -5.70043846e+01, -6.06075991e+01,
        -6.45251018e+01, -6.86128365e+01, -7.35896515e+01,
        -7.92634608e+01, -8.34733231e+01, -9.29802390e+01,
        -1.01179788e+02, -1.10437881e+02, -1.20569527e+02,
        -1.31569973e+02, -1.43633399e+02, -1.57534056e+02,
        -1.73244610e+02, -1.90066728e+02, -2.07980863e+02,
        -2.27589046e+02, -2.50241670e+02, -2.76314755e+02,
        -3.04930986e+02, -3.36196973e+02, -3.70907934e+02,
        -4.10966644e+02, -4.56684022e+02, -5.06945218e+02,
        -5.61838645e+02, -6.22617808e+02, -6.90840503e+02,
        -7.67825713e+02, -8.54436805e+02, -9.51106365e+02,
        -1.05780305e+03, -1.17435473e+03, -1.30045062e+03,
        -1.43571221e+03, -1.57971603e+03, -1.73204264e+03,
        -1.88430524e+03, -2.03403679e+03, -2.17861987e+03,
        -2.31532884e+03, -8.00654731e+03, -5.21487172e-02,
        -2.80234170e-01, 4.93894552e-01, 3.11884938e-01, 6.55799607e-01,
        5.62182419e-01,  3.86074983e-01,  3.43741491e-01,  4.22461089e-01,
        3.63707951e-01,  4.93150010e-01,  4.72813688e-01,  4.07390308e-01,
        4.94974186e-01,  4.69900128e-01,  4.37562389e-01,  5.67370182e-01,
        4.88965362e-01,  6.40728461e-01,  6.14619979e-01,  4.97173823e-01,
        6.19549666e-01,  6.51193557e-01,  4.48906118e-01,  7.93091492e-01,
        6.51249363e-01,  6.56307713e-01,  1.12948552e+00,  9.50018058e-01,
        6.79613030e-01,  9.51359123e-01,  6.31059147e-01,  7.97896887e-01,
        8.44620817e-01,  7.43683837e-01,  1.56693187e+00,  2.75630011e-01,
        5.32956891e-01,  1.57110727e+00,  1.22674610e+00, 4.63932928e-01,
        1.47225464e+00,  1.16948107e+00,  1.07965795e+00, -3.20557791e-01,
        -1.17064127e+00, -7.84880649e-01, -7.60851182e-01, -1.61415945e+00,
        -8.30363975e-01, -1.68459409e+00, -1.49260581e+00, -1.84257084e+00,
        -1.72143079e+00, -1.43131579e+00, -1.63719219e+00, -1.43874851e+00,
        -1.57207905e+00, -1.72909159e+00, -1.98778122e+00, -1.80843826e+00,
        -2.12828312e+00, -2.24768762e+00, -2.36961877e+00, -2.49117258e+00,
        -2.59914065e+00, -2.82309085e+00, -2.93613362e+00, -3.34446991e+00,
        -3.45445086e+00, -3.74962140e+00, -3.78113417e+00, -4.55643800e+00,
        -4.86929016e+00, -5.08657898e+00, -5.22054177e+00, -5.54606515e+00,
        -5.78478304e+00, -5.93652041e+00, -6.11519786e+00])

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


param_updates1 = {'start_year': 2020}
filename1 = 'run_SS_baseline_outputs.pkl'
param_updates2 = {'start_year': 2020, 'use_zeta': True}
filename2 = 'run_SS_baseline_use_zeta.pkl'
param_updates3 = {'start_year': 2020, 'zeta_K': [1.0]}
filename3 = 'run_SS_baseline_small_open.pkl'
param_updates4 = {'start_year': 2020, 'zeta_K': [1.0], 'use_zeta': True}
filename4 = 'run_SS_baseline_small_open_use_zeta.pkl'
param_updates5 = {'start_year': 2020}
filename5 = 'run_SS_reform.pkl'
param_updates6 = {'start_year': 2020, 'use_zeta': True}
filename6 = 'run_SS_reform_use_zeta.pkl'
param_updates7 = {'start_year': 2020, 'zeta_K': [1.0]}
filename7 = 'run_SS_reform_small_open.pkl'
param_updates8 = {'start_year': 2020, 'zeta_K': [1.0], 'use_zeta': True}
filename8 = 'run_SS_reform_small_open_use_zeta.pkl'
param_updates9 = {'start_year': 2020, 'baseline_spending': True}
filename9 = 'run_SS_reform_baseline_spend.pkl'
param_updates10 = {'start_year': 2020, 'baseline_spending': True,
                   'use_zeta': True}
filename10 = 'run_SS_reform_baseline_spend_use_zeta.pkl'
param_updates11 = {'delta_tau_annual': [0.0], 'zeta_K': [0.0],
                   'zeta_D': [0.0]}
filename11 = 'run_SS_baseline_delta_tau0.pkl'


# Note that chaning the order in which these tests are run will cause
# failures for the baseline spending=True tests which depend on the
# output of the baseline run just prior
@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates1, filename1),
                          (False, param_updates9, filename9),
                          (True, param_updates2, filename2),
                        #   (False, param_updates10, filename10),
                          (True, param_updates3, filename3),
                          (True, param_updates4, filename4),
                          (False, param_updates5, filename5),
                          (False, param_updates6, filename6),
                          (False, param_updates7, filename7),
                          (False, param_updates8, filename8),
                          (False, param_updates11, filename11)
                          ],
                         ids=['Baseline', 'Reform, baseline spending',
                              'Baseline, use zeta',
                            #   'Reform, baseline spending, use zeta',
                              'Baseline, small open',
                              'Baseline, small open use zeta',
                              'Reform', 'Reform, use zeta',
                              'Reform, small open',
                              'Reform, small open use zeta',
                              'Baseline, delta_tau=0'
                              ])
@pytest.mark.full_run
def test_run_SS(baseline, param_updates, filename, dask_client):
    # Test SS.run_SS function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    if baseline is False:
        tax_func_path_baseline = os.path.join(CUR_PATH,
                                              'TxFuncEst_baseline.pkl')
        tax_func_path = os.path.join(CUR_PATH,
                                     'TxFuncEst_policy.pkl')
        execute.runner(constants.BASELINE_DIR, constants.BASELINE_DIR,
                       time_path=False, baseline=True,
                       og_spec=param_updates, run_micro=False,
                       tax_func_path=tax_func_path_baseline)
    else:
        tax_func_path = os.path.join(CUR_PATH,
                                     'TxFuncEst_baseline.pkl')
    p = Specifications(baseline=baseline, client=dask_client,
                       num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.get_tax_function_parameters(None, run_micro=False,
                                  tax_func_path=tax_func_path)
    test_dict = SS.run_SS(p, None)
    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', filename))

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v))
