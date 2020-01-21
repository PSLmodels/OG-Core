'''
Test of steady-state module
'''

import pytest
import numpy as np
import os
from ogusa import SS, utils, aggregates, household
from ogusa.parameters import Specifications
CUR_PATH = os.path.abspath(os.path.dirname(__file__))

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
p1.small_open, firm_r, hh_r = small_open_params
p1.firm_r = np.ones(p1.T + p1.S) * firm_r
p1.hh_r = np.ones(p1.T + p1.S) * hh_r
p1.num_workers = 1
BQ1 = np.ones((p1.J)) * 0.00019646295986015257
guesses1 = [guesses_in[0]] + list(BQ1) + [guesses_in[1]] + [guesses_in[2]]
args1 = (bssmat, nssmat, None, None, p1, client)
expected1 = np.array([0.28753454, 0.01889046, 0.02472582, 0.02669199,
                      0.01631467, 0.01925092, 0.02206471, 0.00407802,
                      -0.07014671494961716, 0.00626609])

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
p2.small_open, firm_r, hh_r = small_open_params
p2.firm_r = np.ones(p2.T + p2.S) * firm_r
p2.hh_r = np.ones(p2.T + p2.S) * hh_r
p2.num_workers = 1
BQ2 = np.ones((p2.J)) * 0.00019646295986015257
guesses2 = [guesses_in2[0]] + list(BQ2) + [guesses_in2[1]]
args2 = (bssmat, nssmat, None, factor, p2, client)
expected2 = np.array([0.01325131, 0.01430768, 0.01938654, 0.02069931,
                      0.01232291, 0.0145351, 0.0171059, 0.00309562,
                      0.0016798427500707008])

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
p3.small_open, firm_r, hh_r = small_open_params
p3.firm_r = np.ones(p3.T + p3.S) * firm_r
p3.hh_r = np.ones(p3.T + p3.S) * hh_r
p3.num_workers = 1
BQ3 = np.ones((p3.J)) * 0.00019646295986015257
guesses3 = [guesses_in3[0]] + list(BQ3) + [guesses_in3[1]]
args3 = (bssmat, nssmat, TR_ss, factor_ss, p3, client)
expected3 = np.array([0.01325131, 0.01430768, 0.01938654, 0.02069931,
                      0.01232291, 0.0145351, 0.0171059, 0.00309562,
                      0.01866492])


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
        os.path.join(CUR_PATH, 'test_io_data', 'SS_solver_inputs.pkl'))
    (b_guess_init, n_guess_init, rss, TR_ss, factor_ss, Yss, params,
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
    p.eta = (p.omega_SS.reshape(p.S, 1) *
             p.lambdas.reshape(1, p.J)).reshape(1, p.S, p.J)
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
    p.small_open, firm_r, hh_r = small_open_params
    p.firm_r = np.ones(p.T + p.S) * firm_r
    p.hh_r = np.ones(p.T + p.S) * hh_r
    p.frac_tax_payroll = 0.5 * np.ones(p.T + p.S)
    p.num_workers = 1

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'SS_solver_outputs.pkl'))

    BQss = expected_dict['BQss']
    test_dict = SS.SS_solver(b_guess_init, n_guess_init, rss, BQss, TR_ss,
                             factor_ss, Yss, p, None, fsolve_flag)

    # delete values key-value pairs that are not in both dicts
    del expected_dict['bssmat'], expected_dict['chi_n'], expected_dict['chi_b']
    del expected_dict['Iss_total']
    del test_dict['etr_ss'], test_dict['mtrx_ss'], test_dict['mtry_ss']
    test_dict['IITpayroll_revenue'] = (test_dict['total_revenue_ss'] -
                                       test_dict['business_revenue'])
    del test_dict['T_Pss'], test_dict['T_BQss'], test_dict['T_Wss']
    del test_dict['K_d_ss'], test_dict['K_f_ss'], test_dict['D_d_ss']
    del test_dict['D_f_ss'], test_dict['I_d_ss'], test_dict['trssmat']
    del test_dict['debt_service_f'], test_dict['new_borrowing_f']
    del test_dict['bqssmat'], test_dict['T_Css'], test_dict['Iss_total']
    del test_dict['iit_revenue'], test_dict['payroll_tax_revenue']
    test_dict['revenue_ss'] = test_dict.pop('total_revenue_ss')
    test_dict['T_Hss'] = test_dict.pop('TR_ss')

    for k, v in expected_dict.items():
        print('Testing ', k)
        assert(np.allclose(test_dict[k], v))


def test_inner_loop():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'inner_loop_inputs.pkl'))
    (outer_loop_vars_in, params, baseline, baseline_spending) = input_tuple
    ss_params, income_tax_params, chi_params, small_open_params = params
    (bssmat, nssmat, r, Y, TR, factor) = outer_loop_vars_in
    p = Specifications()
    (p.J, p.S, p.T, p.BW, p.beta, p.sigma, p.alpha, p.gamma, p.epsilon,
     Z, p.delta, p.ltilde, p.nu, p.g_y, p.g_n_ss, tau_payroll,
     tau_bq, p.rho, p.omega_SS, p.budget_balance, alpha_T,
     p.debt_ratio_ss, tau_b, delta_tau, lambdas, imm_rates, p.e,
     retire, p.mean_income_data, h_wealth, p_wealth, m_wealth,
     p.b_ellipse, p.upsilon) = ss_params
    p.eta = (p.omega_SS.reshape(p.S, 1) *
             p.lambdas.reshape(1, p.J)).reshape(1, p.S, p.J)
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
    p.small_open, firm_r, hh_r = small_open_params
    p.firm_r = np.ones(p.T + p.S) * firm_r
    p.hh_r = np.ones(p.T + p.S) * hh_r
    p.num_workers = 1
    BQ = np.ones(p.J) * 0.00019646295986015257
    outer_loop_vars = (bssmat, nssmat, r, BQ, Y, TR, factor)
    (euler_errors, new_bmat, new_nmat, new_r, new_r_gov, new_r_hh,
     new_w, new_TR, new_Y, new_factor, new_BQ,
     average_income_model) = SS.inner_loop(outer_loop_vars, p, None)
    test_tuple = (euler_errors, new_bmat, new_nmat, new_r, new_w,
                  new_TR, new_Y, new_factor, new_BQ,
                  average_income_model)

    expected_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'inner_loop_outputs.pkl'))

    for i, v in enumerate(expected_tuple):
        assert(np.allclose(test_tuple[i], v, atol=1e-05))


def test_euler_equation_solver():
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'euler_eqn_solver_inputs.pkl'))
    (guesses, params) = input_tuple
    p = Specifications()
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
    p.eta = (p.omega_SS.reshape(p.S, 1) *
             p.lambdas.reshape(1, p.J)).reshape(1, p.S, p.J)
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
    p.small_open, firm_r, hh_r = small_open_params
    p.firm_r = np.ones(p.T + p.S) * firm_r
    p.hh_r = np.ones(p.T + p.S) * hh_r
    p.num_workers = 1
    p.frac_tax_payroll = 0.5 * np.ones(p.T + p.S)
    test_dict = SS.run_SS(p, None)

    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', expected_path))

    # delete values key-value pairs that are not in both dicts
    del expected_dict['bssmat'], expected_dict['chi_n'], expected_dict['chi_b']
    del expected_dict['Iss_total']
    del test_dict['etr_ss'], test_dict['mtrx_ss'], test_dict['mtry_ss']
    test_dict['IITpayroll_revenue'] = (test_dict['total_revenue_ss'] -
                                       test_dict['business_revenue'])
    del test_dict['T_Pss'], test_dict['T_BQss'], test_dict['T_Wss']
    del test_dict['resource_constraint_error'], test_dict['T_Css']
    del test_dict['r_gov_ss'], test_dict['r_hh_ss']
    del test_dict['K_d_ss'], test_dict['K_f_ss'], test_dict['D_d_ss']
    del test_dict['D_f_ss'], test_dict['I_d_ss'], test_dict['Iss_total']
    del test_dict['debt_service_f'], test_dict['new_borrowing_f']
    del test_dict['iit_revenue'], test_dict['payroll_tax_revenue']
    test_dict['revenue_ss'] = test_dict.pop('total_revenue_ss')
    test_dict['T_Hss'] = test_dict.pop('TR_ss')

    for k, v in expected_dict.items():
        assert(np.allclose(test_dict[k], v))
