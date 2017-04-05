'''
A 'smoke test' for the ogusa package. Uses a fake data set to run the
baseline
'''

import cPickle as pickle
import os
import numpy as np
import time

import ogusa
from ogusa import calibrate
ogusa.parameters.DATASET = 'REAL'


def runner(output_base, baseline_dir, test=False, time_path=True, baseline=False,
  analytical_mtrs=False, age_specific=False, reform={}, user_params={},
  guid='', run_micro=True, small_open=False, budget_balance=False, baseline_spending=False):

    #from ogusa import parameters, wealth, labor, demographics, income
    from ogusa import parameters, demographics, income, utils
    from ogusa import txfunc

    tick = time.time()
    
    # Make sure options are internally consistent
    if baseline==True and baseline_spending==True:
        print 'Inconsistent options. Setting <baseline_spending> to False, leaving <baseline> True.'
        baseline_spending = False
    if budget_balance==True and baseline_spending==True:
        print 'Inconsistent options. Setting <baseline_spending> to False, leaving <budget_balance> True.'
        baseline_spending = False

    #Create output directory structure
    saved_moments_dir = os.path.join(output_base, "Saved_moments")
    ss_dir = os.path.join(output_base, "SS")
    tpi_dir = os.path.join(output_base, "TPI")
    dirs = [saved_moments_dir, ss_dir, tpi_dir]
    for _dir in dirs:
        try:
            print "making dir: ", _dir
            os.makedirs(_dir)
        except OSError as oe:
            pass

    if run_micro:
        txfunc.get_tax_func_estimate(baseline=baseline, analytical_mtrs=analytical_mtrs, age_specific=age_specific,
                                     start_year=user_params['start_year'], reform=reform, guid=guid)
    print 'In runner, baseline is ', baseline
    run_params = ogusa.parameters.get_parameters(test=test, baseline=baseline, guid=guid)
    run_params['analytical_mtrs'] = analytical_mtrs
    run_params['small_open'] = small_open
    run_params['budget_balance'] = budget_balance

    # Modify ogusa parameters based on user input
    if 'frisch' in user_params:
        print "updating frisch and associated"
        b_ellipse, upsilon = ogusa.elliptical_u_est.estimation(user_params['frisch'],
                                                               run_params['ltilde'])
        run_params['b_ellipse'] = b_ellipse
        run_params['upsilon'] = upsilon
        run_params.update(user_params)
    if 'debt_ratio_ss' in user_params:
        run_params['debt_ratio_ss']=user_params['debt_ratio_ss']

    # Modify ogusa parameters based on user input
    if 'g_y_annual' in user_params:
        print "updating g_y_annual and associated"
        ending_age = run_params['ending_age']
        starting_age = run_params['starting_age']
        S = run_params['S']
        g_y = (1 + user_params['g_y_annual'])**(float(ending_age - starting_age) / S) - 1
        run_params['g_y'] = g_y
        run_params.update(user_params)
        
    # Modify transfer & spending ratios based on user input.
    if 'T_shifts' in user_params:
        if baseline_spending==False:
            print 'updating ALPHA_T with T_shifts in first', user_params['T_shifts'].size, 'periods.'                                            
            T_shifts = np.concatenate((user_params['T_shifts'], np.zeros(run_params['ALPHA_T'].size - user_params['T_shifts'].size)), axis=0)
            run_params['ALPHA_T'] = run_params['ALPHA_T'] + T_shifts
    if 'G_shifts' in user_params:
        if baseline_spending==False:
            print 'updating ALPHA_G with G_shifts in first', user_params['G_shifts'].size, 'periods.'                                            
            G_shifts = np.concatenate((user_params['G_shifts'], np.zeros(run_params['ALPHA_G'].size - user_params['G_shifts'].size)), axis=0)
            run_params['ALPHA_G'] = run_params['ALPHA_G'] + G_shifts

    from ogusa import SS, TPI

    calibrate_model = False
    # List of parameter names that will not be changing (unless we decide to
    # change them for a tax experiment)

    param_names = ['S', 'J', 'T', 'BW', 'lambdas', 'starting_age', 'ending_age',
                'beta', 'sigma', 'alpha', 'gamma', 'epsilon', 'nu', 'Z', 'delta', 'E',
                'ltilde', 'g_y', 'maxiter', 'mindist_SS', 'mindist_TPI',
                'analytical_mtrs', 'b_ellipse', 'k_ellipse', 'upsilon',
                'small_open', 'budget_balance', 'ss_firm_r', 'ss_hh_r', 'tpi_firm_r', 'tpi_hh_r',
                'tG1', 'tG2', 'alpha_T', 'alpha_G', 'ALPHA_T', 'ALPHA_G', 'rho_G', 'debt_ratio_ss',
                'tau_b', 'delta_tau',
                'chi_b_guess', 'chi_n_guess','etr_params','mtrx_params',
                'mtry_params','tau_payroll', 'tau_bq',
                'retire', 'mean_income_data', 'g_n_vector',
                'h_wealth', 'p_wealth', 'm_wealth',
                'omega', 'g_n_ss', 'omega_SS', 'surv_rate', 'imm_rates','e', 'rho',
                'initial_debt','omega_S_preTP']

    '''
    ------------------------------------------------------------------------
        Run SS
    ------------------------------------------------------------------------
    '''

    sim_params = {}
    for key in param_names:
        sim_params[key] = run_params[key]

    sim_params['output_dir'] = output_base
    sim_params['run_params'] = run_params
    income_tax_params, ss_parameters, iterative_params, chi_params, small_open_params = SS.create_steady_state_parameters(**sim_params)

    ss_outputs = SS.run_SS(income_tax_params, ss_parameters, iterative_params, chi_params, small_open_params, baseline, baseline_spending,
                                     baseline_dir=baseline_dir)

    '''
    ------------------------------------------------------------------------
        Pickle SS results
    ------------------------------------------------------------------------
    '''
    if baseline:
        utils.mkdirs(os.path.join(baseline_dir, "SS"))
        ss_dir = os.path.join(baseline_dir, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))
    else:
        utils.mkdirs(os.path.join(output_base, "SS"))
        ss_dir = os.path.join(output_base, "SS/SS_vars.pkl")
        pickle.dump(ss_outputs, open(ss_dir, "wb"))

    if time_path:
        '''
        ------------------------------------------------------------------------
            Run the TPI simulation
        ------------------------------------------------------------------------
        '''

        sim_params['baseline'] = baseline
        sim_params['baseline_spending'] = baseline_spending
        sim_params['input_dir'] = output_base
        sim_params['baseline_dir'] = baseline_dir


        income_tax_params, tpi_params, iterative_params, small_open_params, initial_values, SS_values, fiscal_params, biz_tax_params = TPI.create_tpi_params(**sim_params)

        tpi_output, macro_output = TPI.run_TPI(income_tax_params, tpi_params, iterative_params, small_open_params, initial_values, 
                                               SS_values, fiscal_params, biz_tax_params, output_dir=output_base, baseline_spending=baseline_spending)

        '''
        ------------------------------------------------------------------------
            Pickle TPI results
        ------------------------------------------------------------------------
        '''
        tpi_dir = os.path.join(output_base, "TPI")
        utils.mkdirs(tpi_dir)
        tpi_vars = os.path.join(tpi_dir, "TPI_vars.pkl")
        pickle.dump(tpi_output, open(tpi_vars, "wb"))

        tpi_dir = os.path.join(output_base, "TPI")
        utils.mkdirs(tpi_dir)
        tpi_vars = os.path.join(tpi_dir, "TPI_macro_vars.pkl")
        pickle.dump(macro_output, open(tpi_vars, "wb"))


        print "Time path iteration complete."
    print "It took {0} seconds to get that part done.".format(time.time() - tick)
