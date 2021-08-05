'''
Test of steady-state module
'''

import multiprocessing
from distributed import Client, LocalCluster
import pytest
import numpy as np
import os
import pickle
import copy
from ogcore import SS, utils, aggregates, household, constants
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle
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
(bssmat, nssmat, TR_ss, factor_ss) = input_tuple
# Parameterize the baseline, closed econ case
p1 = Specifications(baseline=True)
p1.update_specifications({'zeta_D': [0.0], 'zeta_K': [0.0]})
guesses1 = np.array([
    0.06, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051])
args1 = (bssmat, nssmat, None, None, p1, None)
expected1 = np.array([
    -0.026632037158481975, -0.0022739752626707334, -0.01871875707724979,
    -0.01791935965422934, 0.005996289165268601, 0.00964100151012603,
    -0.01953460990186908, -0.0029633389016814967, 0.1306862551496613,
    0.11574464544202477])
# Parameterize the reform, closed econ case
p2 = Specifications(baseline=False)
p2.update_specifications({'zeta_D': [0.0], 'zeta_K': [0.0]})
guesses2 = np.array([
    0.06, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07])
args2 = (bssmat, nssmat, None, 0.51, p2, None)
expected2 = np.array([
    -0.030232643078209973, -0.002371139373741624, -0.01637669829114836,
    -0.014404669292893904, 0.005875798127829608, 0.009489605242631064,
    -0.01930931544418691, -0.0029454311332426275, 0.13208003970756407])
# Parameterize the reform, closed econ, baseline spending case
p3 = Specifications(baseline=False)
p3.update_specifications({'zeta_D': [0.0], 'zeta_K': [0.0],
                          'baseline_spending': True})
guesses3 = np.array([
    0.06, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07])
args3 = (bssmat, nssmat, 0.13, 0.51, p3, None)
expected3 = np.array([
    -0.04031619469134644, 0.002754829779058155, 0.005665309738855779,
    0.008140368498332051, 0.007352479311256247, 0.010806872033967399,
    0.007183869209053399, -0.00284466958926688, 0.7269578822834164])
# Parameterize the baseline, partial open economy case (default)
p4 = Specifications(baseline=True)
guesses4 = np.array([
    0.06, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051])
args4 = (bssmat, nssmat, None, None, p4, None)
expected4 = np.array([
    -0.03615195306306068, -0.0025227018857025708, 0.0005778317722585288,
    0.004588284248481369, 0.005706423522356211, 0.00928509074411313,
    0.005887584256507098, 0.002849544478420344, 0.13774167744621094,
    0.09930811783393777])
# Parameterize the baseline, small open econ case
p5 = Specifications(baseline=True)
p5.update_specifications({'zeta_D': [0.0], 'zeta_K': [1.0]})
guesses5 = np.array([
    0.06, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051])
args5 = (bssmat, nssmat, None, 0.51, p5, None)
expected5 = np.array([
    -0.019999999999999962, -0.0021216948497802778,
    0.0013874915656953528, 0.0053198904577732575, 0.006173756535859736,
    0.009858904353577247, 0.006657850176415573, 0.0030235933473326034,
    0.13082844301733987, 0.09467304802224508])
# Parameterize the baseline closed economy, delta tau = 0 case
p6 = Specifications(baseline=True)
p6.update_specifications({'zeta_D': [0.0], 'zeta_K': [0.0],
                          'delta_tau_annual': [0.0]})
guesses6 = np.array([
    0.06, 0.016, 0.02, 0.02, 0.01, 0.01, 0.02, 0.003, -0.07, 0.051])
args6 = (bssmat, nssmat, None, None, p6, None)
expected6 = np.array([
    -0.045453339832995945, -0.002801342521138476, 0.0003419917882823524,
    0.004084012889204677, 0.005384114713826117, 0.008889155694384665,
    0.005358783502477, 0.0027296252357792237, 0.14237702198971164,
    0.1009176917078035])


@pytest.mark.parametrize(
    'guesses,args,expected',
    [(guesses1, args1, expected1),
     (guesses2, args2, expected2),
     (guesses3, args3, expected3),
     (guesses4, args4, expected4),
     (guesses5, args5, expected5),
     (guesses6, args6, expected6)],
     ids=['Baseline, Closed', 'Reform, Closed',
           'Reform, Baseline spending=True, Closed',
           'Baseline, Partial Open', 'Baseline, Small Open',
           'Baseline, Closed, delta_tau = 0'])
def test_SS_fsolve(guesses, args, expected):
    '''
    Test SS.SS_fsolve function.  Provide inputs to function and
    ensure that output returned matches what it has been before.
    '''
    test_list = SS.SS_fsolve(guesses, *args)
    assert(np.allclose(np.array(test_list), np.array(expected),
                       atol=1e-6))


# need to update parameter values that had corresponded to a different
# start year than those in the default_parameters.json file
g_n_ss = 0.0012907765315350872
imm_rates = np.load(os.path.join(CUR_PATH, 'old_imm_rates.npy'))
e = np.load(os.path.join(CUR_PATH, 'old_e.npy'))
omega_SS = np.load(os.path.join(CUR_PATH, 'old_omega_SS.npy'))
param_updates1 = {'start_year': 2020, 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename1 = 'SS_solver_outputs_baseline.pkl'
param_updates2 = {'start_year': 2020, 'budget_balance': True,
                  'alpha_G': [0.0], 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename2 = 'SS_solver_outputs_baseline_budget_balance.pkl'
param_updates3 = {'baseline_spending': True, 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename3 = 'SS_solver_outputs_reform_baseline_spending.pkl'
param_updates4 = {'start_year': 2020, 'zeta_K': [1.0],
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
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
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.BW = 10
    p.frac_tax_payroll = np.zeros(p.frac_tax_payroll.shape)
    p.output_base = CUR_PATH
    if p.baseline:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_baseline.pkl'))
    else:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_policy.pkl'))
    num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
    num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
    num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
    p.mean_income_data = dict_params['tfunc_avginc'][0]
    p.etr_params = np.empty((p.T, p.S, num_etr_params))
    p.etr_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_etr_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.etr_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_etr_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_etr_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtrx_params = np.empty((p.T, p.S, num_mtrx_params))
    p.mtrx_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtrx_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtrx_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtry_params = np.empty((p.T, p.S, num_mtry_params))
    p.mtry_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtry_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtry_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtry_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtry_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    etr_params_old = p.etr_params.copy()
    p.etr_params = etr_params_old.copy()
    p.etr_params[:, :, 5] = etr_params_old[:, :, 6]
    p.etr_params[:, :, 6] = etr_params_old[:, :, 11]
    p.etr_params[:, :, 7] = etr_params_old[:, :, 5]
    p.etr_params[:, :, 8] = etr_params_old[:, :, 7]
    p.etr_params[:, :, 9] = etr_params_old[:, :, 8]
    p.etr_params[:, :, 10] = etr_params_old[:, :, 9]
    p.etr_params[:, :, 11] = etr_params_old[:, :, 10]
    mtrx_params_old = p.mtrx_params.copy()
    p.mtrx_params = mtrx_params_old.copy()
    p.mtrx_params[:, :, 5] = mtrx_params_old[:, :, 6]
    p.mtrx_params[:, :, 6] = mtrx_params_old[:, :, 11]
    p.mtrx_params[:, :, 7] = mtrx_params_old[:, :, 5]
    p.mtrx_params[:, :, 8] = mtrx_params_old[:, :, 7]
    p.mtrx_params[:, :, 9] = mtrx_params_old[:, :, 8]
    p.mtrx_params[:, :, 10] = mtrx_params_old[:, :, 9]
    p.mtrx_params[:, :, 11] = mtrx_params_old[:, :, 10]
    mtry_params_old = p.mtry_params.copy()
    p.mtry_params = mtry_params_old.copy()
    p.mtry_params[:, :, 5] = mtry_params_old[:, :, 6]
    p.mtry_params[:, :, 6] = mtry_params_old[:, :, 11]
    p.mtry_params[:, :, 7] = mtry_params_old[:, :, 5]
    p.mtry_params[:, :, 8] = mtry_params_old[:, :, 7]
    p.mtry_params[:, :, 9] = mtry_params_old[:, :, 8]
    p.mtry_params[:, :, 10] = mtry_params_old[:, :, 9]
    p.mtry_params[:, :, 11] = mtry_params_old[:, :, 10]
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
        assert(np.allclose(test_dict[k], v, atol=1e-04, equal_nan=True))


param_updates5 = {'start_year': 2020, 'zeta_K': [1.0],
                  'budget_balance': True, 'alpha_G': [0.0],
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename5 = 'SS_solver_outputs_baseline_small_open_budget_balance.pkl'
param_updates6 = {'delta_tau_annual': [0.0], 'zeta_K': [0.0],
                  'zeta_D': [0.0], 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename6 = 'SS_solver_outputs_baseline_delta_tau0.pkl'


@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(True, param_updates5, filename5),
                          (True, param_updates6, filename6)],
                         ids=['Baseline, small open, budget balance',
                              'Baseline, delta_tau = 0'])
@pytest.mark.local
def test_SS_solver_extra(baseline, param_updates, filename, dask_client):
    # Test SS.SS_solver function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.BW = 10
    p.frac_tax_payroll = np.zeros(p.frac_tax_payroll.shape)
    p.output_base = CUR_PATH
    if p.baseline:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_baseline.pkl'))
    else:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_policy.pkl'))
    num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
    num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
    num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
    p.mean_income_data = dict_params['tfunc_avginc'][0]
    p.etr_params = np.empty((p.T, p.S, num_etr_params))
    p.etr_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_etr_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.etr_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_etr_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_etr_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtrx_params = np.empty((p.T, p.S, num_mtrx_params))
    p.mtrx_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtrx_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtrx_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtry_params = np.empty((p.T, p.S, num_mtry_params))
    p.mtry_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtry_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtry_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtry_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtry_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    etr_params_old = p.etr_params.copy()
    p.etr_params = etr_params_old.copy()
    p.etr_params[:, :, 5] = etr_params_old[:, :, 6]
    p.etr_params[:, :, 6] = etr_params_old[:, :, 11]
    p.etr_params[:, :, 7] = etr_params_old[:, :, 5]
    p.etr_params[:, :, 8] = etr_params_old[:, :, 7]
    p.etr_params[:, :, 9] = etr_params_old[:, :, 8]
    p.etr_params[:, :, 10] = etr_params_old[:, :, 9]
    p.etr_params[:, :, 11] = etr_params_old[:, :, 10]
    mtrx_params_old = p.mtrx_params.copy()
    p.mtrx_params = mtrx_params_old.copy()
    p.mtrx_params[:, :, 5] = mtrx_params_old[:, :, 6]
    p.mtrx_params[:, :, 6] = mtrx_params_old[:, :, 11]
    p.mtrx_params[:, :, 7] = mtrx_params_old[:, :, 5]
    p.mtrx_params[:, :, 8] = mtrx_params_old[:, :, 7]
    p.mtrx_params[:, :, 9] = mtrx_params_old[:, :, 8]
    p.mtrx_params[:, :, 10] = mtrx_params_old[:, :, 9]
    p.mtrx_params[:, :, 11] = mtrx_params_old[:, :, 10]
    mtry_params_old = p.mtry_params.copy()
    p.mtry_params = mtry_params_old.copy()
    p.mtry_params[:, :, 5] = mtry_params_old[:, :, 6]
    p.mtry_params[:, :, 6] = mtry_params_old[:, :, 11]
    p.mtry_params[:, :, 7] = mtry_params_old[:, :, 5]
    p.mtry_params[:, :, 8] = mtry_params_old[:, :, 7]
    p.mtry_params[:, :, 9] = mtry_params_old[:, :, 8]
    p.mtry_params[:, :, 10] = mtry_params_old[:, :, 9]
    p.mtry_params[:, :, 11] = mtry_params_old[:, :, 10]

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
        assert(np.allclose(test_dict[k], v, atol=1e-05, equal_nan=True))


param_updates1 = {'start_year': 2020, 'zeta_K': [1.0],
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename1 = 'inner_loop_outputs_baseline_small_open.pkl'
param_updates2 = {'start_year': 2020, 'budget_balance': True,
                  'alpha_G': [0.0], 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename2 = 'inner_loop_outputs_baseline_balance_budget.pkl'
param_updates3 = {'start_year': 2020, 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename3 = 'inner_loop_outputs_baseline.pkl'
param_updates4 = {'start_year': 2020, 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename4 = 'inner_loop_outputs_reform.pkl'
param_updates5 = {'start_year': 2020, 'baseline_spending': True,
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
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
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    p.BW = 10
    if p.baseline:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_baseline.pkl'))
    else:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_policy.pkl'))
    num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
    num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
    num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
    p.mean_income_data = dict_params['tfunc_avginc'][0]
    p.etr_params = np.empty((p.T, p.S, num_etr_params))
    p.etr_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_etr_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.etr_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_etr_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_etr_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtrx_params = np.empty((p.T, p.S, num_mtrx_params))
    p.mtrx_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtrx_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtrx_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtry_params = np.empty((p.T, p.S, num_mtry_params))
    p.mtry_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtry_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtry_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtry_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtry_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    etr_params_old = p.etr_params.copy()
    p.etr_params = etr_params_old.copy()
    p.etr_params[:, :, 5] = etr_params_old[:, :, 6]
    p.etr_params[:, :, 6] = etr_params_old[:, :, 11]
    p.etr_params[:, :, 7] = etr_params_old[:, :, 5]
    p.etr_params[:, :, 8] = etr_params_old[:, :, 7]
    p.etr_params[:, :, 9] = etr_params_old[:, :, 8]
    p.etr_params[:, :, 10] = etr_params_old[:, :, 9]
    p.etr_params[:, :, 11] = etr_params_old[:, :, 10]
    mtrx_params_old = p.mtrx_params.copy()
    p.mtrx_params = mtrx_params_old.copy()
    p.mtrx_params[:, :, 5] = mtrx_params_old[:, :, 6]
    p.mtrx_params[:, :, 6] = mtrx_params_old[:, :, 11]
    p.mtrx_params[:, :, 7] = mtrx_params_old[:, :, 5]
    p.mtrx_params[:, :, 8] = mtrx_params_old[:, :, 7]
    p.mtrx_params[:, :, 9] = mtrx_params_old[:, :, 8]
    p.mtrx_params[:, :, 10] = mtrx_params_old[:, :, 9]
    p.mtrx_params[:, :, 11] = mtrx_params_old[:, :, 10]
    mtry_params_old = p.mtry_params.copy()
    p.mtry_params = mtry_params_old.copy()
    p.mtry_params[:, :, 5] = mtry_params_old[:, :, 6]
    p.mtry_params[:, :, 6] = mtry_params_old[:, :, 11]
    p.mtry_params[:, :, 7] = mtry_params_old[:, :, 5]
    p.mtry_params[:, :, 8] = mtry_params_old[:, :, 7]
    p.mtry_params[:, :, 9] = mtry_params_old[:, :, 8]
    p.mtry_params[:, :, 10] = mtry_params_old[:, :, 9]
    p.mtry_params[:, :, 11] = mtry_params_old[:, :, 10]
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
        print('Max diff = ', np.absolute(test_tuple[i] - v).max())
        print('Checking item = ', i)
        assert(np.allclose(test_tuple[i], v, atol=4e-05))


param_updates6 = {'delta_tau_annual': [0.0], 'zeta_K': [0.0],
                  'zeta_D': [0.0], 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename6 = 'inner_loop_outputs_baseline_delta_tau0.pkl'


@pytest.mark.parametrize('baseline,param_updates,filename',
                         [(False, param_updates6, filename6)],
                         ids=['Baseline, delta_tau = 0'])
@pytest.mark.local
def test_inner_loop_extra(baseline, param_updates, filename, dask_client):
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.output_base = CUR_PATH
    p.BW = 10
    if p.baseline:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_baseline.pkl'))
    else:
        dict_params = utils.safe_read_pickle(os.path.join(
            p.output_base, 'TxFuncEst_policy.pkl'))
    num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
    num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
    num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
    p.mean_income_data = dict_params['tfunc_avginc'][0]
    p.etr_params = np.empty((p.T, p.S, num_etr_params))
    p.etr_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_etr_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.etr_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_etr_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_etr_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtrx_params = np.empty((p.T, p.S, num_mtrx_params))
    p.mtrx_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtrx_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtrx_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtry_params = np.empty((p.T, p.S, num_mtry_params))
    p.mtry_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtry_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtry_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtry_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtry_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    # p.get_tax_function_parameters(None, run_micro=False)
    etr_params_old = p.etr_params.copy()
    p.etr_params = etr_params_old.copy()
    p.etr_params[:, :, 5] = etr_params_old[:, :, 6]
    p.etr_params[:, :, 6] = etr_params_old[:, :, 11]
    p.etr_params[:, :, 7] = etr_params_old[:, :, 5]
    p.etr_params[:, :, 8] = etr_params_old[:, :, 7]
    p.etr_params[:, :, 9] = etr_params_old[:, :, 8]
    p.etr_params[:, :, 10] = etr_params_old[:, :, 9]
    p.etr_params[:, :, 11] = etr_params_old[:, :, 10]
    mtrx_params_old = p.mtrx_params.copy()
    p.mtrx_params = mtrx_params_old.copy()
    p.mtrx_params[:, :, 5] = mtrx_params_old[:, :, 6]
    p.mtrx_params[:, :, 6] = mtrx_params_old[:, :, 11]
    p.mtrx_params[:, :, 7] = mtrx_params_old[:, :, 5]
    p.mtrx_params[:, :, 8] = mtrx_params_old[:, :, 7]
    p.mtrx_params[:, :, 9] = mtrx_params_old[:, :, 8]
    p.mtrx_params[:, :, 10] = mtrx_params_old[:, :, 9]
    p.mtrx_params[:, :, 11] = mtrx_params_old[:, :, 10]
    mtry_params_old = p.mtry_params.copy()
    p.mtry_params = mtry_params_old.copy()
    p.mtry_params[:, :, 5] = mtry_params_old[:, :, 6]
    p.mtry_params[:, :, 6] = mtry_params_old[:, :, 11]
    p.mtry_params[:, :, 7] = mtry_params_old[:, :, 5]
    p.mtry_params[:, :, 8] = mtry_params_old[:, :, 7]
    p.mtry_params[:, :, 9] = mtry_params_old[:, :, 8]
    p.mtry_params[:, :, 10] = mtry_params_old[:, :, 9]
    p.mtry_params[:, :, 11] = mtry_params_old[:, :, 10]
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
    p = Specifications(num_workers=NUM_WORKERS)
    (r, w, TR, factor, j, p.J, p.S, beta, p.sigma, p.ltilde, p.g_y,
     p.g_n_ss, tau_payroll, retire, p.mean_income_data, h_wealth,
     p_wealth, m_wealth, p.b_ellipse, p.upsilon, j, p.chi_b,
     p.chi_n, tau_bq, p.rho, lambdas, p.omega_SS, p.e,
     p.analytical_mtrs, etr_params, mtrx_params, mtry_params) = params
    p.beta = np.ones(p.J) * beta
    p.eta = (p.omega_SS.reshape(p.S, 1) *
             p.lambdas.reshape(1, p.J)).reshape(1, p.S, p.J)
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    etr_params_old = np.transpose(etr_params.reshape(
        p.S, 1, etr_params.shape[-1]), (1, 0, 2))
    mtrx_params_old = np.transpose(mtrx_params.reshape(
        p.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
    mtry_params_old = np.transpose(mtry_params.reshape(
        p.S, 1, mtry_params.shape[-1]), (1, 0, 2))
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
    p.tax_func_type = 'DEP'
    p.lambdas = lambdas.reshape(p.J, 1)
    b_splus1 = np.array(guesses[:p.S]).reshape(p.S, 1) + 0.005
    BQ = aggregates.get_BQ(r, b_splus1, j, p, 'SS', False)
    bq = household.get_bq(BQ, j, p, 'SS')
    tr = household.get_tr(TR, j, p, 'SS')
    ubi = p.ubi_nom_array[-1, :, :] / factor
    ubi_j = ubi[:, j]
    args = (r, w, bq, tr, ubi_j, factor, j, p)
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


def test_euler_equation_solver_ubi(dask_client):
    # Test SS.inner_loop function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    input_tuple = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', 'euler_eqn_solver_inputs.pkl'))
    (guesses, params) = input_tuple
    p = Specifications(num_workers=NUM_WORKERS)
    (r, w, TR, factor, j, p.J, p.S, beta, p.sigma, p.ltilde, p.g_y,
     p.g_n_ss, tau_payroll, retire, p.mean_income_data, h_wealth,
     p_wealth, m_wealth, p.b_ellipse, p.upsilon, j, p.chi_b,
     p.chi_n, tau_bq, p.rho, lambdas, p.omega_SS, p.e,
     p.analytical_mtrs, etr_params, mtrx_params, mtry_params) = params
    new_param_values_ubi = {
        'ubi_nom_017': 1000,
        'ubi_nom_1864': 1500,
        'ubi_nom_65p': 500
    }
    p.update_specifications(new_param_values_ubi)
    p.beta = np.ones(p.J) * beta
    p.eta = (p.omega_SS.reshape(p.S, 1) *
             p.lambdas.reshape(1, p.J)).reshape(1, p.S, p.J)
    p.tau_bq = np.ones(p.T + p.S) * 0.0
    p.tau_payroll = np.ones(p.T + p.S) * tau_payroll
    p.h_wealth = np.ones(p.T + p.S) * h_wealth
    p.p_wealth = np.ones(p.T + p.S) * p_wealth
    p.m_wealth = np.ones(p.T + p.S) * m_wealth
    p.retire = (np.ones(p.T + p.S) * retire).astype(int)
    etr_params_old = np.transpose(etr_params.reshape(
        p.S, 1, etr_params.shape[-1]), (1, 0, 2))
    mtrx_params_old = np.transpose(mtrx_params.reshape(
        p.S, 1, mtrx_params.shape[-1]), (1, 0, 2))
    mtry_params_old = np.transpose(mtry_params.reshape(
        p.S, 1, mtry_params.shape[-1]), (1, 0, 2))
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
    p.tax_func_type = 'DEP'
    p.lambdas = lambdas.reshape(p.J, 1)
    b_splus1 = np.array(guesses[:p.S]).reshape(p.S, 1) + 0.005
    BQ = aggregates.get_BQ(r, b_splus1, j, p, 'SS', False)
    bq = household.get_bq(BQ, j, p, 'SS')
    tr = household.get_tr(TR, j, p, 'SS')
    ubi = p.ubi_nom_array[-1, :, :] / factor
    ubi_j = ubi[:, j]
    args = (r, w, bq, tr, ubi_j, factor, j, p)
    test_list = SS.euler_equation_solver(guesses, *args)

    expected_list = np.array(
        [-3.62746503e+00, -6.30069092e+00, -6.76593192e+00, -6.97731636e+00,
         -7.05778211e+00, -6.57306960e+00, -7.11553568e+00, -7.30569989e+00,
         -7.45808587e+00, -7.89984205e+00, -8.11466457e+00, -8.28230737e+00,
         -8.79254123e+00, -8.86994998e+00, -9.31299900e+00, -9.80834386e+00,
         -9.97334576e+00, -1.08350003e+01, -1.13199891e+01, -1.22890967e+01,
         -1.31550522e+01, -1.42753757e+01, -1.55721143e+01, -1.73811488e+01,
         -1.88856354e+01, -2.09570598e+01, -2.30559531e+01, -2.52127202e+01,
         -2.76119665e+01, -3.03141169e+01, -3.30900249e+01, -3.62799735e+01,
         -3.91169753e+01, -4.24246442e+01, -4.55740570e+01, -4.92914895e+01,
         -5.30682845e+01, -5.70043867e+01, -6.06076029e+01, -6.45251035e+01,
         -6.86128407e+01, -7.35896544e+01, -7.92634621e+01, -8.34733575e+01,
         -9.29802395e+01, -1.01179790e+02, -1.10437882e+02, -1.20569528e+02,
         -1.31569974e+02, -1.43633400e+02, -1.57534057e+02, -1.73244610e+02,
         -1.90066729e+02, -2.07980865e+02, -2.27589047e+02, -2.50241671e+02,
         -2.76314756e+02, -3.04930988e+02, -3.36196975e+02, -3.70907936e+02,
         -4.10966646e+02, -4.56684025e+02, -5.06945221e+02, -5.61838648e+02,
         -6.22617812e+02, -6.90840507e+02, -7.67825718e+02, -8.54436811e+02,
         -9.51106371e+02, -1.05780306e+03, -1.17435474e+03, -1.30045062e+03,
         -1.43571222e+03, -1.57971604e+03, -1.73204265e+03, -1.88430525e+03,
         -2.03403680e+03, -2.17861989e+03, -2.31532886e+03, -8.00654736e+03,
         1.40809846e-01, -1.12030129e-01,  6.22203459e-01,  4.46934168e-01,
         7.79360787e-01,  6.79364683e-01,  4.99804499e-01,  4.54369900e-01,
         5.31101485e-01,  4.75088975e-01,  6.02350253e-01,  5.81814144e-01,
         5.15200223e-01,  6.01786444e-01,  5.75500074e-01,  5.41699997e-01,
         6.71009437e-01,  5.91778042e-01,  7.37111582e-01,  7.10420755e-01,
         6.01374236e-01,  7.23255141e-01,  7.53531275e-01,  5.49931777e-01,
         8.92652461e-01,  7.50906356e-01,  7.55676965e-01,  1.22886378e+00,
         1.04917726e+00,  7.78731676e-01,  1.05046810e+00,  7.30225943e-01,
         8.97117997e-01,  9.43832224e-01,  8.53413605e-01,  1.67678933e+00,
         3.86259321e-01,  6.44184123e-01,  1.68230946e+00,  1.33939500e+00,
         5.77523639e-01,  1.58977855e+00,  1.29189143e+00,  1.20609011e+00,
         -1.87021981e-01, -1.02052387e+00, -6.30621486e-01, -6.01292232e-01,
         -1.44640704e+00, -6.63841122e-01, -1.49184613e+00, -1.29364048e+00,
         -1.63965117e+00, -1.54321912e+00, -1.24915822e+00, -1.44958302e+00,
         -1.24924895e+00, -1.38011848e+00, -1.53352558e+00, -1.78785553e+00,
         -1.62057536e+00, -1.92556558e+00, -2.04078887e+00, -2.15831069e+00,
         -2.27538656e+00, -2.37952138e+00, -2.59332565e+00, -2.70210349e+00,
         -3.09023596e+00, -3.19597235e+00, -3.47688206e+00, -3.50823599e+00,
         -4.24335977e+00, -4.54082061e+00, -4.74779822e+00, -4.87587521e+00,
         -5.18517218e+00, -5.41228106e+00, -5.55703856e+00, -5.72724692e+00])

    assert(np.allclose(np.array(test_list), np.array(expected_list)))


imm_rates_deltatau0 = np.load(os.path.join(CUR_PATH, 'old_imm_rates_deltatau0.npy'))
param_updates1 = {'start_year': 2020, 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
# param_updates1 = {'start_year': 2020}
filename1 = 'run_SS_baseline_outputs.pkl'
param_updates2 = {'start_year': 2020, 'use_zeta': True,
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename2 = 'run_SS_baseline_use_zeta.pkl'
param_updates3 = {'start_year': 2020, 'zeta_K': [1.0],
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename3 = 'run_SS_baseline_small_open.pkl'
param_updates4 = {'start_year': 2020, 'zeta_K': [1.0], 'use_zeta': True,
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename4 = 'run_SS_baseline_small_open_use_zeta.pkl'
param_updates5 = {'start_year': 2020, 'omega_SS': omega_SS,
                  'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename5 = 'run_SS_reform.pkl'
param_updates6 = {'start_year': 2020, 'use_zeta': True,
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename6 = 'run_SS_reform_use_zeta.pkl'
param_updates7 = {'start_year': 2020, 'zeta_K': [1.0],
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename7 = 'run_SS_reform_small_open.pkl'
param_updates8 = {'start_year': 2020, 'zeta_K': [1.0], 'use_zeta': True,
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename8 = 'run_SS_reform_small_open_use_zeta.pkl'
param_updates9 = {'start_year': 2020, 'baseline_spending': True,
                  'omega_SS': omega_SS, 'g_n_ss': g_n_ss,
                  'imm_rates': imm_rates, 'e': e}
filename9 = 'run_SS_reform_baseline_spend.pkl'
param_updates10 = {'start_year': 2020, 'baseline_spending': True,
                   'use_zeta': True, 'omega_SS': omega_SS,
                   'g_n_ss': g_n_ss, 'imm_rates': imm_rates, 'e': e}
filename10 = 'run_SS_reform_baseline_spend_use_zeta.pkl'
param_updates11 = {'delta_tau_annual': [0.0], 'zeta_K': [0.0],
                   'zeta_D': [0.0], 'omega_SS': omega_SS,
                   'g_n_ss': g_n_ss, 'imm_rates': imm_rates_deltatau0, 'e': e}
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
@pytest.mark.local
def test_run_SS(baseline, param_updates, filename, dask_client):
    # Test SS.run_SS function.  Provide inputs to function and
    # ensure that output returned matches what it has been before.
    SS.ENFORCE_SOLUTION_CHECKS = False
    if baseline is False:
        p_base = Specifications(
            output_base=constants.BASELINE_DIR,
            baseline_dir=constants.BASELINE_DIR,
            time_path=False, baseline=True,
            num_workers=NUM_WORKERS)
        p_base.update_specifications(param_updates)
        p_base.BW = 10
        p_base.frac_tax_payroll = np.zeros(p_base.frac_tax_payroll.shape)
        dict_params = utils.safe_read_pickle(os.path.join(
            CUR_PATH, 'TxFuncEst_baseline.pkl'))
        num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
        num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
        num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
        p_base.mean_income_data = dict_params['tfunc_avginc'][0]
        p_base.etr_params = np.empty((p_base.T, p_base.S, num_etr_params))
        p_base.etr_params[:p_base.BW, :, :] =\
            np.transpose(
                dict_params['tfunc_etr_params_S'][:p_base.S, :p_base.BW, :],
                axes=[1, 0, 2])
        p_base.etr_params[p_base.BW:, :, :] = np.tile(np.transpose(
            dict_params['tfunc_etr_params_S'][:p_base.S, -1, :].reshape(
                p_base.S, 1, num_etr_params), axes=[1, 0, 2]), (p_base.T - p_base.BW, 1, 1))
        p_base.mtrx_params = np.empty((p_base.T, p_base.S, num_mtrx_params))
        p_base.mtrx_params[:p_base.BW, :, :] =\
            np.transpose(
                dict_params['tfunc_mtrx_params_S'][:p_base.S, :p_base.BW, :],
                axes=[1, 0, 2])
        p_base.mtrx_params[p_base.BW:, :, :] = np.tile(np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p_base.S, -1, :].reshape(
                p_base.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p_base.T - p_base.BW, 1, 1))
        p_base.mtry_params = np.empty((p_base.T, p_base.S, num_mtry_params))
        p_base.mtry_params[:p_base.BW, :, :] =\
            np.transpose(
                dict_params['tfunc_mtry_params_S'][:p_base.S, :p_base.BW, :],
                axes=[1, 0, 2])
        p_base.mtry_params[p_base.BW:, :, :] = np.tile(np.transpose(
            dict_params['tfunc_mtry_params_S'][:p_base.S, -1, :].reshape(
                p_base.S, 1, num_mtry_params), axes=[1, 0, 2]), (p_base.T - p_base.BW, 1, 1))
        etr_params_old = p_base.etr_params
        mtrx_params_old = p_base.mtrx_params
        mtry_params_old = p_base.mtry_params
        p_base.etr_params = etr_params_old.copy()
        p_base.etr_params[:, :, 5] = etr_params_old[:, :, 6]
        p_base.etr_params[:, :, 6] = etr_params_old[:, :, 11]
        p_base.etr_params[:, :, 7] = etr_params_old[:, :, 5]
        p_base.etr_params[:, :, 8] = etr_params_old[:, :, 7]
        p_base.etr_params[:, :, 9] = etr_params_old[:, :, 8]
        p_base.etr_params[:, :, 10] = etr_params_old[:, :, 9]
        p_base.etr_params[:, :, 11] = etr_params_old[:, :, 10]
        p_base.mtrx_params = mtrx_params_old.copy()
        p_base.mtrx_params[:, :, 5] = mtrx_params_old[:, :, 6]
        p_base.mtrx_params[:, :, 6] = mtrx_params_old[:, :, 11]
        p_base.mtrx_params[:, :, 7] = mtrx_params_old[:, :, 5]
        p_base.mtrx_params[:, :, 8] = mtrx_params_old[:, :, 7]
        p_base.mtrx_params[:, :, 9] = mtrx_params_old[:, :, 8]
        p_base.mtrx_params[:, :, 10] = mtrx_params_old[:, :, 9]
        p_base.mtrx_params[:, :, 11] = mtrx_params_old[:, :, 10]
        p_base.mtry_params = mtry_params_old.copy()
        p_base.mtry_params[:, :, 5] = mtry_params_old[:, :, 6]
        p_base.mtry_params[:, :, 6] = mtry_params_old[:, :, 11]
        p_base.mtry_params[:, :, 7] = mtry_params_old[:, :, 5]
        p_base.mtry_params[:, :, 8] = mtry_params_old[:, :, 7]
        p_base.mtry_params[:, :, 9] = mtry_params_old[:, :, 8]
        p_base.mtry_params[:, :, 10] = mtry_params_old[:, :, 9]
        p_base.mtry_params[:, :, 11] = mtry_params_old[:, :, 10]
        base_ss_outputs = SS.run_SS(p_base, dask_client)
        utils.mkdirs(os.path.join(
            constants.BASELINE_DIR, "SS"))
        ss_dir = os.path.join(
            constants.BASELINE_DIR, "SS", "SS_vars.pkl")
        with open(ss_dir, "wb") as f:
            pickle.dump(base_ss_outputs, f)
    if baseline:
        dict_params = utils.safe_read_pickle(os.path.join(
            CUR_PATH, 'TxFuncEst_baseline.pkl'))
    else:
        dict_params = utils.safe_read_pickle(os.path.join(
            CUR_PATH, 'TxFuncEst_policy.pkl'))
    p = Specifications(baseline=baseline, num_workers=NUM_WORKERS)
    p.update_specifications(param_updates)
    p.BW = 10
    p.frac_tax_payroll = np.zeros(p.frac_tax_payroll.shape)
    num_etr_params = dict_params['tfunc_etr_params_S'].shape[2]
    num_mtrx_params = dict_params['tfunc_mtrx_params_S'].shape[2]
    num_mtry_params = dict_params['tfunc_mtry_params_S'].shape[2]
    p.mean_income_data = dict_params['tfunc_avginc'][0]
    p.etr_params = np.empty((p.T, p.S, num_etr_params))
    p.etr_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_etr_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.etr_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_etr_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_etr_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtrx_params = np.empty((p.T, p.S, num_mtrx_params))
    p.mtrx_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtrx_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtrx_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtrx_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtrx_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    p.mtry_params = np.empty((p.T, p.S, num_mtry_params))
    p.mtry_params[:p.BW, :, :] =\
        np.transpose(
            dict_params['tfunc_mtry_params_S'][:p.S, :p.BW, :],
            axes=[1, 0, 2])
    p.mtry_params[p.BW:, :, :] = np.tile(np.transpose(
        dict_params['tfunc_mtry_params_S'][:p.S, -1, :].reshape(
            p.S, 1, num_mtry_params), axes=[1, 0, 2]), (p.T - p.BW, 1, 1))
    # p.get_tax_function_parameters(None, run_micro=False,
    #                               tax_func_path=tax_func_path)
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

    test_dict = SS.run_SS(p, client=None)
    expected_dict = utils.safe_read_pickle(
        os.path.join(CUR_PATH, 'test_io_data', filename))

    for k, v in expected_dict.items():
        # print('Checking item = ', k)
        assert(np.allclose(test_dict[k], v, atol=1e-04))
