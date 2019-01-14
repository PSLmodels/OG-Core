from __future__ import print_function
import pytest
import numpy as np
import os
from ogusa import SS, utils
from ogusa.parameters import Specifications
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_inputs.pkl'))
guesses1, params = input_tuple
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
p1.small_open, ss_firm_r, ss_hh_r = small_open_params
p1.ss_firm_r = np.ones(p1.T + p1.S) * ss_firm_r
p1.ss_hh_r = np.ones(p1.T + p1.S) * ss_hh_r
p1.num_workers = 1
args1 = (bssmat, nssmat, None, None, p1, client)
expected1 = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_outputs.pkl'))

input_tuple = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_inputs.pkl'))
guesses2, params = input_tuple
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
p2.small_open, ss_firm_r, ss_hh_r = small_open_params
p2.ss_firm_r = np.ones(p2.T + p2.S) * ss_firm_r
p2.ss_hh_r = np.ones(p2.T + p2.S) * ss_hh_r
p2.num_workers = 1
args2 = (bssmat, nssmat, None, factor, p2, client)
expected2 = utils.safe_read_pickle(
    os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_outputs.pkl'))

input_tuple = utils.safe_read_pickle(
    os.path.join(
        CUR_PATH,
        'test_io_data/SS_fsolve_reform_baselinespend_inputs.pkl'))
guesses3, params = input_tuple
params = params + (None, 1)
(bssmat, nssmat, T_Hss, chi_params, ss_params, income_tax_params,
 iterative_params, factor_ss, small_open_params, client,
 num_workers) = params
p3 = Specifications()
(p3.J, p3.S, p3.T, p3.BW, p3.beta, p3.sigma, p3.alpha, p3.gamma, p3.epsilon,
 Z, p3.delta, p3.ltilde, p3.nu, p3.g_y, p3.g_n_ss, tau_payroll,
 tau_bq, p3.rho, p3.omega_SS, p3.budget_balance, alpha_T,
 p3.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p3.e,
 retire, p3.mean_income_data, h_wealth, p_wealth, m_wealth,
 p3.b_ellipse, p3.upsilon) = ss_params
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
p3.small_open, ss_firm_r, ss_hh_r = small_open_params
p3.ss_firm_r = np.ones(p3.T + p3.S) * ss_firm_r
p3.ss_hh_r = np.ones(p3.T + p3.S) * ss_hh_r
p3.num_workers = 1
args3 = (bssmat, nssmat, T_Hss, factor_ss, p3, client)
expected3 = utils.safe_read_pickle(
    os.path.join(
        CUR_PATH,
        'test_io_data/SS_fsolve_reform_baselinespend_outputs.pkl'))


@pytest.mark.parametrize('guesses,args,expected',
                         [(guesses1, args1, expected1),
                          (guesses2, args2, expected2),
                          (guesses3, args3, expected3)],
                         ids=['Baseline, Closed', 'Reform, Closed',
                              'Reform, Baseline spending=True, Closed'])
def test_SS_fsolve(guesses, args, expected):
    # Test SS.SS_fsolve function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    test_list = SS.SS_fsolve(guesses, *args)
    assert(np.allclose(np.array(test_list), np.array(expected),
                       atol=1e-6))


def test_SS_solver():
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_solver_inputs.pkl'))
    (b_guess_init, n_guess_init, rss, T_Hss, factor_ss, Yss, params,
     baseline, fsolve_flag, baseline_spending) = input_tuple
    (bssmat, nssmat, chi_params, ss_params, income_tax_params,
     iterative_params, small_open_params) = params

    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, tau_payroll,
     tau_bq, p.rho, p.omega_SS, p.budget_balance, alpha_T,
     p.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p.e,
     retire, p.mean_income_data, h_wealth, p_wealth, m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.Z = np.ones(p.T + p.S) * Z
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.alpha_T = np.ones(p.T + p.S) * alpha_T
    p.tau_b = np.ones(p.T + p.S) * tau_b
    p.delta_tau = np.ones(p.T + p.S) * delta_tau
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    p.lambdas = lambdas.reshape(p.J, 1)
    p.imm_rates = imm_rates.reshape(1, p.S)
    p.tax_func_type = 'DEP'
    p.baseline = baseline
    p.baseline_spending = baseline_spending
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params.reshape(
        p.S, 1, etr_params.shape[-1]), (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params.reshape(
        p.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params.reshape(
        p.S, 1, mtry_params.shape[-1]), (1, 0, 2))
    p.maxiter, p.mindist_SS = iterative_params
    p.chi_b, p.chi_n = chi_params
    p.small_open, ss_firm_r, ss_hh_r = small_open_params
    p.ss_firm_r = np.ones(p.T + p.S) * ss_firm_r
    p.ss_hh_r = np.ones(p.T + p.S) * ss_hh_r
    p.num_workers = 1
    test_dict = SS.SS_solver(b_guess_init, n_guess_init, rss, T_Hss,
                             factor_ss, Yss, p, None, fsolve_flag)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_solver_outputs.pkl'))

    # delete values key-value pairs that are not in both dicts
    del expected_dict['bssmat'], expected_dict['chi_n'], expected_dict['chi_b']
    del test_dict['etr_ss'], test_dict['mtrx_ss'], test_dict['mtry_ss']
    test_dict['IITpayroll_revenue'] = (test_dict['total_revenue_ss'] -
                                       test_dict['business_revenue'])
    del test_dict['T_Pss'], test_dict['T_BQss'], test_dict['T_Wss']
    test_dict['revenue_ss'] = test_dict.pop('total_revenue_ss')

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v))


def test_inner_loop():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/inner_loop_inputs.pkl'))
    (outer_loop_vars, params, baseline, baseline_spending) = input_tuple
    ss_params, income_tax_params, chi_params, small_open_params = params
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, tau_payroll,
     tau_bq, p.rho, p.omega_SS, p.budget_balance, alpha_T,
     p.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p.e,
     retire, p.mean_income_data, h_wealth, p_wealth, m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.Z = np.ones(p.T + p.S) * Z
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.alpha_T = np.ones(p.T + p.S) * alpha_T
    p.tau_b = np.ones(p.T + p.S) * tau_b
    p.delta_tau = np.ones(p.T + p.S) * delta_tau
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    p.lambdas = lambdas.reshape(p.J, 1)
    p.imm_rates = imm_rates.reshape(1, p.S)
    p.tax_func_type = 'DEP'
    p.baseline = baseline
    p.baseline_spending = baseline_spending
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params.reshape(
        p.S, 1, etr_params.shape[-1]), (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params.reshape(
        p.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params.reshape(
        p.S, 1, mtry_params.shape[-1]), (1, 0, 2))
    p.chi_b, p.chi_n = chi_params
    p.small_open, ss_firm_r, ss_hh_r = small_open_params
    p.ss_firm_r = np.ones(p.T + p.S) * ss_firm_r
    p.ss_hh_r = np.ones(p.T + p.S) * ss_hh_r
    p.num_workers = 1

    test_tuple = SS.inner_loop(outer_loop_vars, p, None)

    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/inner_loop_outputs.pkl'))

    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_tuple[i], v))


def test_euler_equation_solver():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/euler_eqn_solver_inputs.pkl'))
    (guesses, params) = input_tuple
    p = Specifications()
    (r, w, T_H, factor, j, p.J, p.S, p.beta, p.sigma, p.ltilde, p.g_y,
     p.g_n_ss, tau_payroll, retire, p.mean_income_data, h_wealth,
     p_wealth, m_wealth, p.b_ellipse, p.upsilon, j, p.chi_b,
     p.chi_n, tau_bq, p.rho, lambdas, p.omega_SS, p.e,
     p.analytical_mtrs, etr_params, mtrx_params, mtry_params) = params
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

    args = (r, w, T_H, factor, j, p)
    test_list = SS.euler_equation_solver(guesses, *args)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/euler_eqn_solver_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


@pytest.mark.full_run
@pytest.mark.parametrize('input_path,expected_path',
                         [('run_SS_open_unbal_inputs.pkl',
                           'run_SS_open_unbal_outputs.pkl'),
                          ('run_SS_closed_balanced_inputs.pkl',
                           'run_SS_closed_balanced_outputs.pkl')],
                         ids=['Open, Unbalanced', 'Closed Balanced'])
def test_run_SS(input_path, expected_path):
    # Test SS.run_SS function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', input_path))
    (income_tax_params, ss_params, iterative_params, chi_params,
     small_open_params, baseline, baseline_spending, baseline_dir) =\
        input_tuple
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, tau_payroll,
     tau_bq, p.rho, p.omega_SS, p.budget_balance, alpha_T,
     p.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p.e,
     retire, p.mean_income_data, h_wealth, p_wealth, m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.Z = np.ones(p.T + p.S) * Z
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.alpha_T = np.ones(p.T + p.S) * alpha_T
    p.tau_b = np.ones(p.T + p.S) * tau_b
    p.delta_tau = np.ones(p.T + p.S) * delta_tau
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    p.lambdas = lambdas.reshape(p.J, 1)
    p.imm_rates = imm_rates.reshape(1, p.S)
    p.tax_func_type = 'DEP'
    p.baseline = baseline
    p.baseline_spending = baseline_spending
    p.baseline_dir = baseline_dir
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params.reshape(
        p.S, 1, etr_params.shape[-1]), (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params.reshape(
        p.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params.reshape(
        p.S, 1, mtry_params.shape[-1]), (1, 0, 2))
    p.maxiter, p.mindist_SS = iterative_params
    p.chi_b, p.chi_n = chi_params
    p.small_open, ss_firm_r, ss_hh_r = small_open_params
    p.ss_firm_r = np.ones(p.T + p.S) * ss_firm_r
    p.ss_hh_r = np.ones(p.T + p.S) * ss_hh_r
    p.num_workers = 1
    test_dict = SS.run_SS(p, None)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', expected_path))

    # delete values key-value pairs that are not in both dicts
    del expected_dict['bssmat'], expected_dict['chi_n'], expected_dict['chi_b']
    del test_dict['etr_ss'], test_dict['mtrx_ss'], test_dict['mtry_ss']
    test_dict['IITpayroll_revenue'] = (test_dict['total_revenue_ss'] -
                                       test_dict['business_revenue'])
    del test_dict['T_Pss'], test_dict['T_BQss'], test_dict['T_Wss']
    del test_dict['resource_constraint_error']
    test_dict['revenue_ss'] = test_dict.pop('total_revenue_ss')

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v))
