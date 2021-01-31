'''
Script to run the Biden plan with microsimulation output from the TPC
model
'''

# import modules
import numpy as np
import pandas as pd
import multiprocessing
from distributed import Client
import matplotlib.pyplot as plt
import time
import os
from ogusa import output_tables as ot
from ogusa import output_plots as op
from ogusa import parameter_plots as pp
from ogusa.execute import runner
from ogusa.utils import Inequality
import tpc_txfunc as txfunc
from ogusa.constants import REFORM_DIR, BASELINE_DIR
from ogusa.utils import safe_read_pickle, mkdirs


def main():
    # Define parameters to use for multiprocessing
    client = Client()
    num_workers = min(multiprocessing.cpu_count(), 7)
    print('Number of workers = ', num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    main_dir = os.path.join(CUR_DIR, 'Biden_Proposal')
    tables_dir = os.path.join(main_dir, 'tables')
    plots_dir = os.path.join(main_dir, 'plots')
    base_dir = os.path.join(main_dir, BASELINE_DIR)
    reform_dir = os.path.join(main_dir, REFORM_DIR)
    # make directories
    mkdirs(base_dir)
    mkdirs(reform_dir)
    mkdirs(tables_dir)
    mkdirs(plots_dir)

    '''
    ------------------------------------------------------------------------
    Estimate tax functions
    ------------------------------------------------------------------------
    '''
    # Parameters for estimating tax functions
    tax_func_type = 'GS'
    age_specific = True
    start_year = 2021
    end_year = 2030
    BW = end_year - start_year + 1
    starting_age = 20
    ending_age = 100
    S = 80
    J = 10
    # Update from default parameters to model high income groups
    lambdas = [
        0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.0009, 0.0001]
    chi_b = np.ones(J) * 80
    zeta = np.ones((S, J)) * (1 / (S * J))
    eta = np.ones((S, J)) * (1 / S) * np.array(lambdas).reshape(1, J)
    tau_c = [0.0]
    beta_annual = np.array(
        [0.91, 0.91, 0.92, 0.93, 0.95, 0.965, 0.97, 0.98, 0.99, 0.995])

    # Path with raw data from TPC microsim
    # commented bc can't share TPC microdata
    # base_data_path = os.path.join(CUR_DIR, 'TPC_microsim_output',
    #                               'TPC_baseline_12142020')
    # reform_data_path = os.path.join(CUR_DIR, 'TPC_microsim_output',
    #                                 'biden_plan_rates_11212020')
    # Path to save tax function parameters to
    base_tax_func_path = os.path.join(
            CUR_DIR, 'TxFuncEst_TPC_baseline.pkl')
    reform_tax_func_path = os.path.join(
            CUR_DIR, 'TxFuncEst_TPC_biden.pkl')
    #  Estimate Tax Functions -- commented bc can't share TPC microdata
    # txfunc.get_tax_func_estimate(
    #     BW, S, starting_age, ending_age, baseline=True,
    #     analytical_mtrs=False, tax_func_type=tax_func_type,
    #     age_specific=age_specific, start_year=start_year, reform={},
    #     guid='', tax_func_path=base_tax_func_path, data=base_data_path,
    #     client=client, num_workers=num_workers)
    # # estimate reform tax functions
    # txfunc.get_tax_func_estimate(
    #     BW, S, starting_age, ending_age, baseline=False,
    #     analytical_mtrs=False, tax_func_type=tax_func_type,
    #     age_specific=age_specific, start_year=start_year, reform={},
    #     guid='', tax_func_path=reform_tax_func_path,
    #     data=reform_data_path, client=client, num_workers=num_workers)
    '''
    ------------------------------------------------------------------------
    Run baseline policy first
    ------------------------------------------------------------------------
    '''

    # path to pre-estimated tax function parameters
    og_spec_base = {'start_year': start_year, 'tG1': 20, 'tG2': 200,
                    'initial_debt_ratio': 1.0, 'debt_ratio_ss': 1.7,
                    'lambdas': lambdas, 'chi_b': chi_b, 'eta': eta,
                    'zeta': zeta, 'beta_annual': beta_annual,
                    'tau_c': tau_c, 'alpha_G': [0.065],
                    'tax_func_type': tax_func_type,
                    'age_specific': age_specific,
                    'initial_guess_r_SS': 0.08,
                    'initial_guess_TR_SS': 0.057,
                    'r_gov_shift': 0.02, 'mindist_TPI': 1e-4}
    kwargs = {'output_base': base_dir, 'baseline_dir': base_dir,
              'test': False, 'time_path': True, 'baseline': True,
              'og_spec': og_spec_base, 'guid': '',
              'run_micro': False, 'tax_func_path': base_tax_func_path,
              'data': 'cps', 'client': client,
              'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)

    '''
    ------------------------------------------------------------------------
    Run reform policy
    ------------------------------------------------------------------------
    '''
    # update the effective corporate income tax rate
    og_spec_reform = {'start_year': start_year, 'tG1': 20, 'tG2': 200,
                      'initial_debt_ratio': 1.0, 'debt_ratio_ss': 1.7,
                      'lambdas': lambdas, 'chi_b': chi_b, 'eta': eta,
                      'zeta': zeta, 'beta_annual': beta_annual,
                      'tau_c': tau_c, 'cit_rate': [0.21, 0.28],
                      'alpha_G': [0.065],
                      'tax_func_type': tax_func_type,
                      'age_specific': age_specific,
                      'initial_guess_r_SS': 0.08,
                      'initial_guess_TR_SS': 0.057,
                      'r_gov_shift': 0.02, 'mindist_TPI': 1e-4}
    kwargs = {'output_base': reform_dir, 'baseline_dir': base_dir,
              'test': False, 'time_path': True, 'baseline': False,
              'og_spec': og_spec_reform, 'guid': '',
              'iit_reform': {}, 'run_micro': False,
              'tax_func_path': reform_tax_func_path, 'data': 'cps',
              'client': client, 'num_workers': num_workers}

    start_time = time.time()
    runner(**kwargs)
    print('run time = ', time.time()-start_time)
    client.close()

    # Create plots and tables found in paper
    tax_base_params = safe_read_pickle(base_tax_func_path)
    tax_reform_params = safe_read_pickle(reform_tax_func_path)
    base_ss = safe_read_pickle(
        os.path.join(base_dir, 'SS', 'SS_vars.pkl'))
    base_tpi = safe_read_pickle(
        os.path.join(base_dir, 'TPI', 'TPI_vars.pkl'))
    base_params = safe_read_pickle(
        os.path.join(base_dir, 'model_params.pkl'))
    reform_ss = safe_read_pickle(
        os.path.join(reform_dir, 'SS', 'SS_vars.pkl'))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, 'TPI', 'TPI_vars.pkl'))
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, 'model_params.pkl'))

    # TABLES
    # Look at breakout of dynamic revenue effects
    ot.dynamic_revenue_decomposition(
            base_params, base_tpi, base_ss, reform_params, reform_tpi,
            reform_ss, num_years=10, include_SS=True,
            include_business_tax=True, full_break_out=False,
            include_overall=True, start_year=start_year,
            table_format='tex',
            path=os.path.join(tables_dir, 'revenue_decomposition.tex'))

    # Pct changes in macro variables
    ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['Y', 'C', 'K', 'L', 'r', 'w'],
        output_type='pct_diff', num_years=10, include_SS=True,
        include_overall=True, start_year=start_year,
        table_format='tex',
        path=os.path.join(tables_dir, 'macro_pct_changes.tex'))

    # Income distribution moments
    # income distribution from Piketty and Saez (2016 data)
    moments = {'Data': {
        '0-50% share': 14.21,
        '90-99% share': 29.40,
        'Top 1% share': 18.38,
        'Gini coeff': 0.545
            }}
    # Model distribution
    ineq_base = Inequality(
        base_ss['yss_before_tax_mat'], base_params.omega_SS,
        base_params.lambdas, base_params.S, base_params.J)
    moments['OG-USA'] = {
        '0-50% share': (1 - ineq_base.top_share(0.50)) * 100,
        '90-99% share': (ineq_base.top_share(0.10) -
                         ineq_base.top_share(0.01)) * 100,
        'Top 1% share': (ineq_base.top_share(0.01)) * 100,
        'Gini coeff': ineq_base.gini()}
    df = pd.DataFrame.from_dict(moments)
    df.to_latex(os.path.join(tables_dir, 'income_distrib_moments.tex'))

    # Wealth distribution moments
    ot.wealth_moments_table(
        base_ss, base_params, table_format='tex',
        path=os.path.join(tables_dir, 'wealth_moments_table.tex'))

    # PLOTS
    # GDP plot in levels based on CBO
    op.plot_aggregates(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params, var_list=['Y'],
        plot_type='cbo', num_years_to_plot=25,
        start_year=start_year, vertical_line_years=None,
        plot_title=None,
        path=os.path.join(plots_dir, 'GDP_Levels_CBO.png'))

    # plot debt/GDP trajectory
    op.plot_gdp_ratio(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params, var_list=['D'],
        plot_type='levels', num_years_to_plot=50,
        start_year=start_year,
        vertical_line_years=[base_params.start_year + base_params.tG1],
        plot_title=None,
        path=os.path.join(plots_dir, 'debt_to_gdp.png'))

    # Percentage changes in the capital stock
    op.plot_aggregates(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params, var_list=['K'],
        plot_type='pct_diff', num_years_to_plot=50,
        start_year=start_year,
        vertical_line_years=[base_params.start_year + base_params.tG1],
        plot_title=None,
        path=os.path.join(plots_dir, 'pct_change_K.png'))

    # Plot ability profiles
    pp.plot_ability_profiles(
        base_params, include_title=False, path=plots_dir)

    # Labor supply profiles
    op.tpi_profiles(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params, by_j=True, var='n_mat',
        num_years=10, start_year=start_year, plot_title=None,
        path=os.path.join(plots_dir, 'LaborSupply_2021-2030.png'))

    # Pct change in c, n, b, y, etr, mtrx, mtry by ability group over
    # 10 years
    var_list = ['c_path', 'n_mat', 'bmat_splus1', 'etr_path',
                'mtrx_path', 'mtry_path', 'y_before_tax_mat']
    path_list = ['Cons', 'Labor', 'Save', 'ETR', 'MTRx', 'MTRy',
                 'Income']
    for i, v in enumerate(var_list):
        op.ability_bar(
            base_tpi, base_params, reform_tpi, reform_params,
            var=v, num_years=10, start_year=base_params.start_year,
            plot_title=None,
            path=os.path.join(plots_dir, 'PctChange_' +
                              path_list[i] + '.png'))

    # Create plot comparing baseline and reform tax functions in each year
    # of the budget window for each type of tax function (ETR, MTRx, MTRy)
    min_inc_amt = 5
    max_inc_amt = 800000
    tot_inc_sup = np.exp(
        np.linspace(np.log(min_inc_amt), np.log(max_inc_amt), 100))
    tax_abbrev_list = ['etr', 'mtrx', 'mtry']
    tax_label_list = ['ETR', 'MTR, Labor Inc.', 'MTR, Capital Inc.']
    y_lim_list = [0.35, 0.47, 0.35]

    for i, tax_abbrev in enumerate(tax_abbrev_list):
        for year in range(start_year, start_year + BW):
            rate_key = 'tfunc_' + tax_abbrev + '_params_S'
            phi0_base, phi1_base, phi2_base = \
                tax_base_params[rate_key][0, year - start_year, :3]
            phi0_reform, phi1_reform, phi2_reform = \
                tax_reform_params[rate_key][0, year - start_year, :3]
            if tax_abbrev == 'etr':
                rates_base = (
                    (phi0_base * (tot_inc_sup -
                                  ((tot_inc_sup ** -phi1_base) +
                                   phi2_base) **
                                  (-1 / phi1_base))) / tot_inc_sup)
                rates_reform = (
                    (phi0_reform * (tot_inc_sup -
                     ((tot_inc_sup ** -phi1_reform) + phi2_reform) **
                     (-1 / phi1_reform))) / tot_inc_sup)
            else:
                rates_base = (
                    phi0_base * (1 - (tot_inc_sup ** (-phi1_base - 1) *
                                 ((tot_inc_sup ** -phi1_base) +
                                  phi2_base) **
                                  ((-1 - phi1_base) / phi1_base))))
                rates_reform = (
                    phi0_reform * (1 -
                                   (tot_inc_sup ** (-phi1_reform - 1) *
                                    ((tot_inc_sup ** -phi1_reform) +
                                    phi2_reform) **
                                    ((-1 - phi1_reform) / phi1_reform))))
            # Plot estimated baseline tax rates
            plt.plot(tot_inc_sup, rates_base, color='black',
                     label='Baseline')
            # Plot estimated policy tax rates
            plt.plot(tot_inc_sup, rates_reform, color='red',
                     label='Biden Proposal')
            plt.ylim((-0.01, y_lim_list[i]))
            plt.ylabel(tax_label_list[i])
            plt.xlim((min_inc_amt - 6, max_inc_amt))
            plt.xticks(
                np.array([0, 100000, 200000, 300000, 400000, 500000,
                          600000, 700000, 800000]),
                ('$0', '$100k', '$200k', '$300k', '$400k', '$500k',
                 '$600k', '$700k', '$800k'))
            plt.xlabel(r'Total income (\$s)')
            title_text = (
                'Gouveia-Strauss baseline vs. reform functions, ' +
                'all ages, ' + str(year))
            plt.title(title_text)
            plt.legend(loc='lower right')
            filename = (
                'GS_noage_' + str(year) + '_base_reform_' + tax_abbrev)
            plt.savefig(os.path.join(plots_dir, filename))
            plt.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
