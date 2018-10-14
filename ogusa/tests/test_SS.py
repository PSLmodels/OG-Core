from __future__ import print_function
import pytest
import numpy as np
import os
from ogusa import SS, utils
from ogusa.pb_api import Specifications
CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_SS_fsolve():
    # Test SS.SS_fsolve function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_inputs.pkl'))
    guesses, params = input_tuple
    params = params + (None, 1)
    (bssmat, nssmat, chi_params, ss_params, income_tax_params,
     iterative_params, small_open_params, client, num_workers) = params
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, p.tau_payroll,
     p.tau_bq, p.rho, p.omega_SS, p.budget_balance, p.alpha_T,
     p.debt_ratio_ss, p.tau_b, p.delta_tau, lambdas, imm_rates, p.e,
     p.retire, p.mean_income_data, p.h_wealth, p.p_wealth, p.m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.imm_rates = imm_rates.reshape(1, p.S)
    p.tax_func_type = 'DEP'
    p.baseline = True
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
    p.small_open, p.ss_firm_r, p.ss_hh_r = small_open_params
    p.num_workers = 1
    args = (bssmat, nssmat, p, client)

    test_list = SS.SS_fsolve(guesses, *args)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_fsolve_reform():
    # Test SS.SS_fsolve_reform function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_inputs.pkl'))
    guesses, params = input_tuple
    params = params + (None, 1)
    (bssmat, nssmat, chi_params, ss_params, income_tax_params,
     iterative_params, factor, small_open_params, client,
     num_workers) = params
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, p.tau_payroll,
     p.tau_bq, p.rho, p.omega_SS, p.budget_balance, p.alpha_T,
     p.debt_ratio_ss, p.tau_b, p.delta_tau, lambdas, imm_rates, p.e,
     p.retire, p.mean_income_data, p.h_wealth, p.p_wealth, p.m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.imm_rates = imm_rates.reshape(1, p.S)
    p.tax_func_type = 'DEP'
    p.baseline = False
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
    p.small_open, p.ss_firm_r, p.ss_hh_r = small_open_params
    p.num_workers = 1
    print('Bssmat in max and min = ', bssmat.max(), bssmat.min())
    print('Nssmat in max and min = ', nssmat.max(), nssmat.min())
    args = (bssmat, nssmat, factor, p, client)

    test_list = SS.SS_fsolve_reform(guesses, *args)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_fsolve_reform_outputs.pkl'))
    print('Results = ', test_list)
    print('Expected = ', expected_list)
    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_SS_fsolve_reform_baselinespend():
    # Test SS.SS_fsolve_reform_baselinespend function.  Provide inputs
    # to function and ensure that output returned matches what it has
    # been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(
            CUR_PATH,
            'test_io_data/SS_fsolve_reform_baselinespend_inputs.pkl'))
    guesses, params = input_tuple
    params = params + (None, 1)
    (bssmat, nssmat, T_Hss, chi_params, ss_params, income_tax_params,
     iterative_params, factor_ss, small_open_params, client,
     num_workers) = params
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, p.tau_payroll,
     p.tau_bq, p.rho, p.omega_SS, p.budget_balance, p.alpha_T,
     p.debt_ratio_ss, p.tau_b, p.delta_tau, lambdas, imm_rates, p.e,
     p.retire, p.mean_income_data, p.h_wealth, p.p_wealth, p.m_wealth,
     p.b_ellipse, p.upsilon) = ss_params

    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.imm_rates = imm_rates.reshape(1, p.S)
    p.tax_func_type = 'DEP'
    p.baseline = False
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
    p.small_open, p.ss_firm_r, p.ss_hh_r = small_open_params
    p.num_workers = 1

    args = (bssmat, nssmat, T_Hss, factor_ss, p, client)
    test_list = SS.SS_fsolve_reform_baselinespend(guesses, *args)

    expected_list = utils.safe_read_pickle(
        os.path.join(
            CUR_PATH,
            'test_io_data/SS_fsolve_reform_baselinespend_outputs.pkl'))
    print('Results = ', test_list)
    print('Expected = ', expected_list)
    assert(np.allclose(np.array(test_list), np.array(expected_list)))


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
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, p.tau_payroll,
     p.tau_bq, p.rho, p.omega_SS, p.budget_balance, p.alpha_T,
     p.debt_ratio_ss, p.tau_b, p.delta_tau, lambdas, imm_rates, p.e,
     p.retire, p.mean_income_data, p.h_wealth, p.p_wealth, p.m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.tau_bq = 0.0
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
    p.small_open, p.ss_firm_r, p.ss_hh_r = small_open_params
    p.num_workers = 1
    test_dict = SS.SS_solver(b_guess_init, n_guess_init, rss, T_Hss,
                             factor_ss, Yss, p, None, fsolve_flag)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/SS_solver_outputs.pkl'))

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
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, p.tau_payroll,
     p.tau_bq, p.rho, p.omega_SS, p.budget_balance, p.alpha_T,
     p.debt_ratio_ss, p.tau_b, p.delta_tau, lambdas, imm_rates, p.e,
     p.retire, p.mean_income_data, p.h_wealth, p.p_wealth, p.m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.tau_bq = 0.0
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
    p.small_open, p.ss_firm_r, p.ss_hh_r = small_open_params
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
     p.g_n_ss, p.tau_payroll, p.retire, p.mean_income_data, p.h_wealth,
     p.p_wealth, p.m_wealth, p.b_ellipse, p.upsilon, j, p.chi_b,
     p.chi_n, p.tau_bq, p.rho, lambdas, p.omega_SS, p.e,
     p.analytical_mtrs, etr_params, mtrx_params, mtry_params) = params

    p.etr_params = np.transpose(etr_params.reshape(
        p.S, 1, etr_params.shape[-1]), (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params.reshape(
        p.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params.reshape(
        p.S, 1, mtry_params.shape[-1]), (1, 0, 2))
    p.tax_func_type = 'DEP'
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)

    args = (r, w, T_H, factor, j, p)
    test_list = SS.euler_equation_solver(guesses, *args)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/euler_eqn_solver_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


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
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, p.tau_payroll,
     p.tau_bq, p.rho, p.omega_SS, p.budget_balance, p.alpha_T,
     p.debt_ratio_ss, p.tau_b, p.delta_tau, lambdas, imm_rates, p.e,
     p.retire, p.mean_income_data, p.h_wealth, p.p_wealth, p.m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.tau_bq = 0.0
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
    p.small_open, p.ss_firm_r, p.ss_hh_r = small_open_params
    p.num_workers = 1
    test_dict = SS.run_SS(p, None)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', expected_path))

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v))
