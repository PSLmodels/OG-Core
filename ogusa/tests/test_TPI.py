import pytest
import pickle
import numpy as np
import os
from ogusa import SS, TPI, utils
from ogusa.pb_api import Specifications

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_firstdoughnutring():
    # Test TPI.firstdoughnutring function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/firstdoughnutring_inputs.pkl'))
    guesses, r, w, b, BQ, T_H, j, params = input_tuple
    income_tax_params, tpi_params, initial_b = params
    tpi_params = tpi_params + [True]
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, p.tau_b, p.delta_tau,
     p.tau_payroll, p.tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, p.retire, p.mean_income_data, factor, p.h_wealth,
     p.p_wealth, p.m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params
    p.tax_func_type = 'DEP'
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params, (1, 0, 2))
    p.mtrx_params = np.transpose(mtrx_params, (1, 0, 2))
    p.mtry_params = np.transpose(mtry_params, (1, 0, 2))
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.num_workers = 1

    test_list = TPI.firstdoughnutring(guesses, r, w, BQ, T_H, theta,
                                      factor, j, initial_b, p)

    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/firstdoughnutring_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


def test_twist_doughnut():
    # Test TPI.twist_doughnut function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/twist_doughnut_inputs.pkl'))
    guesses, r, w, BQ, T_H, j, s, t, params = input_tuple
    income_tax_params, tpi_params, initial_b = params
    tpi_params = tpi_params + [True]
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, p.tau_b, p.delta_tau,
     p.tau_payroll, p.tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, p.retire, p.mean_income_data, factor, p.h_wealth,
     p.p_wealth, p.m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params
    p.tax_func_type = 'DEP'
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.num_workers = 1
    test_list = TPI.twist_doughnut(guesses, r, w, BQ, T_H, theta,
                                   factor, j, s, t, etr_params,
                                   mtrx_params, mtry_params, initial_b,
                                   p)
    expected_list = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/twist_doughnut_outputs.pkl'))

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


@pytest.mark.full_run
def test_inner_loop():
    # Test TPI.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/tpi_inner_loop_inputs.pkl'))
    guesses, outer_loop_vars, params, j = input_tuple
    income_tax_params, tpi_params, initial_values, ind = params
    initial_values = initial_values
    tpi_params = tpi_params
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, p.tau_b, p.delta_tau,
     p.tau_payroll, p.tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, p.retire, p.mean_income_data, factor, p.h_wealth,
     p.p_wealth, p.m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params
    p.tax_func_type = 'DEP'
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params, (1, 0, 2))[:p.T, :, :]
    p.mtrx_params = np.transpose(mtrx_params, (1, 0, 2))[:p.T, :, :]
    p.mtry_params = np.transpose(mtry_params, (1, 0, 2))[:p.T, :, :]
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.num_workers = 1
    (K0, b_sinit, b_splus1init, factor, initial_b, initial_n,
     p.omega_S_preTP, initial_debt, D0) = initial_values
    initial_values_in = (K0, b_sinit, b_splus1init, factor, initial_b,
                         initial_n, initial_debt, D0)
    (r, K, BQ, T_H) = outer_loop_vars
    outer_loop_vars_in = (r, K, BQ, T_H, theta)

    guesses = (guesses[0], guesses[1])
    test_tuple = TPI.inner_loop(guesses, outer_loop_vars_in,
                                initial_values_in, j, ind, p)

    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/tpi_inner_loop_outputs.pkl'))

    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_tuple[i], v))


@pytest.mark.full_run
def test_run_TPI():
    # Test TPI.run_TPI function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/run_TPI_inputs.pkl'))
    (income_tax_params, tpi_params, iterative_params, small_open_params,
     initial_values, SS_values, fiscal_params, biz_tax_params,
     output_dir, baseline_spending) = input_tuple
    tpi_params = tpi_params + [True]
    initial_values = initial_values + (0.0,)

    p = Specifications()
    (J, S, T, BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, p.tau_b, p.delta_tau,
     p.tau_payroll, p.tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, p.retire, p.mean_income_data, factor, p.h_wealth,
     p.p_wealth, p.m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params

    new_param_values = {
        'J': J,
        'S': S,
        'T': T
    }
    # update parameters instance with new values for test
    p.update_specifications(new_param_values, raise_errors=False)
    (J, S, T, BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     p.Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n, p.tau_b, p.delta_tau,
     p.tau_payroll, p.tau_bq, p.rho, p.omega, N_tilde, lambdas,
     p.imm_rates, p.e, p.retire, p.mean_income_data, factor, p.h_wealth,
     p.p_wealth, p.m_wealth, p.b_ellipse, p.upsilon, p.chi_b, p.chi_n,
     theta, p.baseline) = tpi_params
    p.small_open, p.tpi_firm_r, p.tpi_hh_r_params = small_open_params
    p.maxiter, p.mindist_SS, p.mindist_TPI = iterative_params
    (p.budget_balance, p.ALPHA_T, p.ALPHA_G, p.tG1, p.tG2, p.rho_G,
     p.debt_ratio_ss) = fiscal_params
    (p.tau_b, p.delta_tau) = biz_tax_params
    p.analytical_mtrs, etr_params, mtrx_params, mtry_params =\
        income_tax_params
    p.etr_params = np.transpose(etr_params, (1, 0, 2))[:p.T, :, :]
    p.mtrx_params = np.transpose(mtrx_params, (1, 0, 2))[:p.T, :, :]
    p.mtry_params = np.transpose(mtry_params, (1, 0, 2))[:p.T, :, :]
    p.tau_bq = 0.0
    p.lambdas = lambdas.reshape(p.J, 1)
    p.output = output_dir
    p.baseline_spending = baseline_spending
    p.num_workers = 1
    (K0, b_sinit, b_splus1init, factor, initial_b, initial_n,
     p.omega_S_preTP, initial_debt, D0) = initial_values

    # Need to run SS first to get results
    print('Shpaes in  = ', p.rho.shape, p.S, p.J, p.T)
    ss_outputs = SS.run_SS(p, None)

    if p.baseline:
        utils.mkdirs(os.path.join(p.baseline_dir, "SS"))
        ss_dir = os.path.join(p.baseline_dir, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
    else:
        utils.mkdirs(os.path.join(p.output_base, "SS"))
        ss_dir = os.path.join(p.output_base, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))

    test_dict = TPI.run_TPI(p, None)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data/run_TPI_outputs.pkl'))

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v, rtol=1e-04, atol=1e-04))
