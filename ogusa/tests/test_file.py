from ogusa.scripts import execute
import numpy as np
import os
import pytest

mock_params = {"ALPHA_G": np.ones(6),
               'ALPHA_T':np.ones(50),
               'BW': 10,
               'E': 1,
               'J': 2,
                 'S': 10,
                 'T': 200,
                 'Z': 1,
                 'alpha': 1,
                 'alpha_G': 1,
                 'alpha_T': 1,
                 'analytical_mtrs': 1,
                 'b_ellipse': 1,
                 'beta': 1,
                 'budget_balance': 1,
                 'chi_b_guess': 1,
                 'chi_n_guess': 1,
                 'debt_ratio_ss': 1,
                 'delta': 1,
                 'delta_tau': 1,
                 'e': 1,
                 'ending_age': 1,
                 'epsilon': 1,
                 'etr_params': 1,
                 'g_n_ss': 1,
                 'g_n_vector': 1,
                 'g_y': 1,
                 'gamma': 1,
                 'h_wealth': 1,
                 'imm_rates': 1,
                 'initial_debt': 1,
                 'k_ellipse': 1,
                 'lambdas': 1,
                 'ltilde': 1,
                 'm_wealth': 1,
                 'maxiter': 1,
                 'mean_income_data': 1,
                 'mindist_SS': 1,
                 'mindist_TPI': 1,
                 'mtrx_params': 1,
                 'mtry_params': 1,
                 'nu': 1,
                 'omega': 1,
                 'omega_SS': 1,
                 'omega_S_preTP': 1,
                 'p_wealth': 1,
                 'retire': 1,
                 'rho': 1,
                 'rho_G': 1,
                 'sigma': 1,
                 'small_open': 1,
                 'ss_firm_r': 1,
                 'ss_hh_r': 1,
                 'starting_age': 1,
                 'surv_rate': 1,
                 'tG1': 1,
                 'tG2': 1,
                 'tau_b': 1,
                 'tau_bq': 1,
                 'tau_payroll': 1,
                 'tpi_firm_r': 1,
                 'tpi_hh_r': 1,
                 'upsilon': 1}


class MockWriteFiles():


    def __init__(self, baseline_dir, reform_dir):
        self.dumped_files = []
        self.baseline_dir = os.path.abspath(baseline_dir)
        self.reform_dir = os.path.abspath(reform_dir)
        self.call_runner()


    def path_diffs(self):
        for path_tup in self.dumped_files:
            np.testing.assert_equal(path_tup[0], path_tup[1])


    def call_runner(self):
        reform = {}

        T_shifts = np.zeros(50)
        T_shifts[2:10] = 0.01
        T_shifts[10:40]= -0.01
        G_shifts = np.zeros(6)
        G_shifts[0:3]  = -0.01
        G_shifts[3:6]  = -0.005

        user_params = {'frisch':0.41, 'start_year':2017, 'debt_ratio_ss':1.0,
                       'T_shifts':T_shifts, 'G_shifts':G_shifts}

        kwargs={'output_base': self.baseline_dir, 'baseline_dir': self.baseline_dir,
                'test': False, 'time_path': True, 'baseline': True,
                'analytical_mtrs': False, 'age_specific': True,
                'user_params': user_params, 'guid': 'baseline',
                'run_micro': True, 'small_open': False, 'budget_balance': False,
                'baseline_spending': False}
        self.mock(self.baseline_dir)
        execute.runner(**kwargs)

        output_base = self.reform_dir
        kwargs= {'output_base': output_base, 'baseline_dir': self.baseline_dir,
                 'test': False, 'time_path': True, 'baseline': False,
                 'analytical_mtrs': False, 'age_specific': True,
                 'user_params': user_params, 'guid': "policy", 'reform': reform,
                 'run_micro': True, 'small_open': False, 'budget_balance': False,
                 'baseline_spending': False}
        self.mock(self.reform_dir, baseline=False)
        execute.runner(**kwargs)


    def mock(self, _dir, baseline=True):
        execute.txfunc.tax_func_estimate = lambda *args: os.path.join(
            _dir,
            "TxFuncEst_baseline.pkl" if baseline else "TxFuncEst_policy.pkl"
        )
        execute.parameters.get_parameters = lambda *args, **kwargs: mock_params
        execute.txfunc.pickle.dump = self.mock_pickle_dump
        execute.pickle.dump = self.mock_pickle_dump

        execute.SS.create_steady_state_parameters = lambda **kwargs: (None, None, None, None, None)
        execute.SS.run_SS = lambda *args, **kwargs: os.path.join(
            _dir,
            "SS/SS_vars.pkl"
        )
        execute.TPI.create_tpi_params = lambda **kwargs: (None, None, None, None, None, None, None, None)
        execute.TPI.run_TPI = lambda *args, **kwargs: (
            os.path.join(_dir, "TPI/TPI_vars.pkl"),
            os.path.join(_dir, "TPI/TPI_macro_vars.pkl")
        )


    def mock_pickle_dump(self, data, file_obj):
        print('mock mock')
        self.dumped_files.append(
            (os.path.abspath(file_obj.name), data)
        )


# def test_write_files_empty():
#     mwf = MockWriteFiles('', '')
#     mwf.call_runner()
#     mwf.path_diffs()

# def test_write_files_with_name():
#     mwf = MockWriteFiles('test_baseline', 'test_reform')
#     mwf.call_runner()
#     mwf.path_diffs()
