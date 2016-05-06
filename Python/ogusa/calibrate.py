
'''
This module should be organized as follows:

Main function:
chi_estimate() = returns chi_n, chi_b
    - calls: 
        wealth.get_wealth_data() - returns data moments on wealth distribution
        labor.labor_data_moments() - returns data moments on labor supply
        minstat() - returns min of statistical objective function
            model_moments() - returns model moments
                SS.run_SS() - return SS distributions

'''

'''
------------------------------------------------------------------------
Last updated: 4/11/2016

Uses a simulated method of moments to calibrate the chi_n adn chi_b 
parameters of OG-USA.

This py-file calls the following other file(s):
    wealth.get_wealth_data()
    labor.labor_data_moments()
    SS.run_SS

This py-file creates the following other file(s): None
------------------------------------------------------------------------
'''


def chi_estimate(income_tax_params, ss_parameters, iterative_params, chi_guesses, baseline_dir="./OUTPUT"):

    # unpack tuples of parameters
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, tau_bq, rho, omega_SS, lambdas, e, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_params

    # Generate Wealth data moments
    wealth_moments = wealth.get_wealth_data(lambdas, J, flag_graphs, output_dir)

    # Generate labor data moments
    labor_moments = labor.labor_data_moments(flag_graphs, output_dir)

    # combine moments
    data_moments = list(wealth_moments.flatten()) + list(labor_moments.flatten())

    # call minimizer
    bnds = np.ones((S+J,2))*(1e-12, None) # Need (1e-12, None) pari J+S times
    #min_args = () data_moments
    min_args = data_moments
    est_output = opt.minimize(minstat, chi_guesses,
                    args=(min_args), method="L-BFGS-B", bounds=bnds,
                    tol=1e-15)
    chi_params = est_output.x
    objective_func_min = est_out.fun 

    return chi_params




def minstat(chi_guesses, *args):
    '''
    --------------------------------------------------------------------
    This function generates the weighted sum of squared differences
    between the model and data moments.


    RETURNS: wssqdev
    --------------------------------------------------------------------
    '''
    chi_params = chi_guesses
    data_moments, income_tax_params, ss_params, iterative_params, chi_params, baseline_dir = args
    ss_output = SS.run_SS(income_tax_params, ss_params, iterative_params, chi_params, True, baseline_dir)

        # output = {'Kss': Kss, 'bssmat': bssmat, 'Lss': Lss, 'Css':Css, 'nssmat': nssmat, 'Yss': Yss,
        #       'wss': wss, 'rss': rss, 'theta': theta, 'BQss': BQss, 'factor_ss': factor_ss,
        #       'bssmat_s': bssmat_s, 'cssmat': cssmat, 'bssmat_splus1': bssmat_splus1,
        #       'T_Hss': T_Hss, 'euler_savings': euler_savings,
        #       'euler_labor_leisure': euler_labor_leisure, 'chi_n': chi_n,
        #       'chi_b': chi_b}

    model_moments = calc_moments(ss_output)

    distance = ((model_moments - data_moments)**2).sum()
    return distance


def calc_moments(**ss_output):
    '''
    --------------------------------------------------------------------
    This function calculates moments from the SS output that correspond
    to the data moments used for estimation.

    RETURNS: model_moments
    --------------------------------------------------------------------
    '''

    # wealth moments

    # labor moments

    # combine moments
    model_moments = list(model_wealth_moments.flatten()) + list(model_labor_moments.flatten())


    return model_moments

#***********************************************
#***********************************************

#OLD CODE BELOW

#***********************************************
#***********************************************

calibrate_model = False

if calibrate_model:
        global Nfeval, value_all, chi_params_all
        Nfeval = 1
        value_all = np.zeros((10000))
        chi_params_all = np.zeros((S+J,10000))
        outputs = {'solutions': solutions, 'chi_params': chi_params}
        ss_init_path = os.path.join(
            output_dir, "Saved_moments/SS_init_solutions.pkl")
        pickle.dump(outputs, open(ss_init_path, "wb"))
        function_to_minimize_X = lambda x: function_to_minimize(
            x, chi_params, income_tax_parameters, ss_parameters, iterative_params, omega_SS, rho, lambdas, tau_bq, e, output_dir)
        bnds = tuple([(1e-6, None)] * (S + J))
        # In order to scale all the parameters to estimate in the minimizer, we have the minimizer fit a vector of ones that
        # will be multiplied by the chi initial guesses inside the function.  Otherwise, if chi^b_j=1e5 for some j, and the
        # minimizer peturbs that value by 1e-8, the % difference will be extremely small, outside of the tolerance of the
        # minimizer, and it will not change that parameter.
        chi_params_scalars = np.ones(S + J)
        #chi_params_scalars = opt.minimize(function_to_minimize_X, chi_params_scalars,
        #                                  method='TNC', tol=MINIMIZER_TOL, bounds=bnds, callback=callbackF(chi_params_scalars), options=MINIMIZER_OPTIONS).x
        # chi_params_scalars = opt.minimize(function_to_minimize, chi_params_scalars, 
        #                                   args=(chi_params, income_tax_parameters, ss_parameters, iterative_params, 
        #                                     omega_SS, rho, lambdas, tau_bq, e, output_dir),
        #                                   method='TNC', tol=MINIMIZER_TOL, bounds=bnds, 
        #                                   callback=callbackF(chi_params_scalars,chi_params, income_tax_parameters, 
        #                                     ss_parameters, iterative_params, omega_SS, rho, lambdas, tau_bq, e, output_dir), 
        #                                   options=MINIMIZER_OPTIONS).x
        chi_params_scalars = opt.minimize(function_to_minimize, chi_params_scalars, 
                                          args=(chi_params, income_tax_parameters, ss_parameters, iterative_params, 
                                            omega_SS, rho, lambdas, tau_bq, e, output_dir),
                                          method='TNC', tol=MINIMIZER_TOL, bounds=bnds, 
                                          options=MINIMIZER_OPTIONS).x
        chi_params *= chi_params_scalars
        print 'The final scaling params', chi_params_scalars
        print 'The final bequest parameter values:', chi_params

        solutions_dict = pickle.load(open(ss_init_path, "rb"))
        solutions = solutions_dict['solutions']
        b_guess = solutions[:S * J]
        n_guess = solutions[S * J:2 * S * J]
        wguess, rguess, factorguess, T_Hguess = solutions[2 * S * J:]
        guesses = [wguess, rguess, T_Hguess, factorguess]
        args_ = (b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params[J:], chi_params[:J], 
             income_tax_parameters, ss_parameters, iterative_params, tau_bq, rho, lambdas, omega_SS, e)
        [solutions, infodict, ier, message] = opt.fsolve(SS_fsolve, guesses, args=args_, xtol=mindist_SS, full_output=True)
        [wguess, rguess, T_Hguess, factorguess] = solutions
        fsolve_flag = True
        solutions = SS_solver(b_guess.reshape(S, J), n_guess.reshape(S, J), wguess, rguess, T_Hguess, factorguess, chi_params[
                          J:], chi_params[:J], income_tax_parameters, ss_parameters, iterative_params, tau_bq, rho, lambdas, omega_SS, e, fsolve_flag)




def function_to_minimize(chi_params_scalars, chi_params_init, income_tax_parameters, ss_parameters, 
                         iterative_params, omega_SS, rho_vec, lambdas, tau_bq, e, output_dir):
    '''
    Inputs:
        chi_params_scalars = guesses for multipliers for chi parameters
                             ((S+J)x1 array)
        chi_params_init = chi parameters that will be multiplied
                          ((S+J)x1 array)
        params = list of parameters (list)
        omega_SS = steady state population weights (Sx1 array)
        rho_vec = mortality rates (Sx1 array)
        lambdas = ability weights (Jx1 array)
        tau_bq = bequest tax rates (Jx1 array)
        e = ability levels (Jx1 array)
    Output:
        The sum of absolute percent deviations between the actual and
        simulated wealth moments
    '''
    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = ss_parameters

    analytical_mtrs, etr_params, mtrx_params, mtry_params = income_tax_parameters

    chi_params_init *= chi_params_scalars
    # print 'Print Chi_b: ', chi_params_init[:J]
    # print 'Scaling vals:', chi_params_scalars[:J]
    ss_init_path = os.path.join(output_dir,
                                "Saved_moments/SS_init_solutions.pkl")
    solutions_dict = pickle.load(open(ss_init_path, "rb"))
    solutions = solutions_dict['solutions']

    b_guess = solutions[:(S * J)]
    n_guess = solutions[S * J:2 * S * J]
    wguess, rguess, factorguess, T_Hguess = solutions[(2 * S * J):]
    guesses = [wguess, rguess, T_Hguess, factorguess]
    args_ = (b_guess.reshape(S, J), n_guess.reshape(S, J), chi_params_init[J:], chi_params_init[:J], 
                 income_tax_parameters, ss_parameters, iterative_params, tau_bq, rho, lambdas, omega_SS, e)
    [solutions, infodict, ier, message] = opt.fsolve(SS_fsolve, guesses, args=args_, xtol=mindist_SS, full_output=True)
    [wguess, rguess, T_Hguess, factorguess] = solutions
    fsolve_flag = True
    solutions = SS_solver(b_guess.reshape(S, J), n_guess.reshape(S, J), wguess, rguess, T_Hguess, factorguess, chi_params_init[
                              J:], chi_params_init[:J], income_tax_parameters, ss_parameters, iterative_params, tau_bq, rho, lambdas, omega_SS, e, fsolve_flag)


    b_new = solutions[:(S * J)]
    n_new = solutions[(S * J):(2 * S * J)]
    w_new, r_new, factor_new, T_H_new = solutions[(2 * S * J):]
    # Wealth Calibration Euler
    error5 = list(utils.check_wealth_calibration(b_new.reshape(S, J)[:-1, :],
                                                 factor_new, ss_parameters, output_dir))
    # labor calibration euler
    labor_path = os.path.join(
        output_dir, "Saved_moments/labor_data_moments.pkl")
    lab_data_dict = pickle.load(open(labor_path, "rb"))
    labor_sim = (n_new.reshape(S, J) * lambdas.reshape(1, J)).sum(axis=1)
    if DATASET == 'SMALL':
        lab_dist_data = lab_data_dict['labor_dist_data'][:S]
    else:
        lab_dist_data = lab_data_dict['labor_dist_data']

    error6 = list(utils.pct_diff_func(labor_sim, lab_dist_data))
    # combine eulers
    output = np.array(error5 + error6)
    # Constraints
    eul_error = np.ones(J)
    for j in xrange(J):
        eul_error[j] = np.abs(Euler_equation_solver(np.append(b_new.reshape(S, J)[:, j], n_new.reshape(S, J)[:, j]), r_new, w_new,
                                                    T_H_new, factor_new, j, income_tax_parameters, ss_parameters, chi_params_init[:J], chi_params_init[J:], tau_bq, rho, lambdas, omega_SS, e)).max()
    fsolve_no_converg = eul_error.max()
    if np.isnan(fsolve_no_converg):
        fsolve_no_converg = 1e6
    if fsolve_no_converg > 1e-4:
        # If the fsovle didn't converge (was NaN or above the tolerance), then tell the minimizer that this is a bad place to be
        # and don't save the solutions as initial guesses (since they might be
        # gibberish)
        output += 1e14
    else:
        var_names = ['solutions']
        dictionary = {}
        for key in var_names:
            dictionary[key] = locals()[key]
        ss_init_path = os.path.join(
            output_dir, "Saved_moments/SS_init_solutions.pkl")
        pickle.dump(dictionary, open(ss_init_path, "wb"))
    if (chi_params_init <= 0.0).any():
        # In case the minimizer doesn't respect the bounds given
        output += 1e14
    # Use generalized method of moments to fit the chi's
    weighting_mat = np.eye(2 * J + S)
    scaling_val = 100.0
    value = np.dot(scaling_val * np.dot(output.reshape(1, 2 * J + S),
                                        weighting_mat), scaling_val * output.reshape(2 * J + S, 1))
    print 'Value of criterion function: ', value.sum()

    
    # pickle output in case not converge
    global Nfeval, value_all, chi_params_all
    value_all[Nfeval] = value.sum()
    chi_params_all[:,Nfeval] = chi_params_init
    dict_GMM = dict([('values', value_all), ('chi_params', chi_params_all)])
    ss_init_path = os.path.join(output_dir, "Saved_moments/SS_init_all.pkl")
    pickle.dump(dict_GMM, open(ss_init_path, "wb"))
    Nfeval += 1

    return value.sum()


def callbackF(chi,chi_params, income_tax_parameters, ss_parameters, iterative_params, omega_SS, rho, lambdas, tau_bq, e, output_dir):
    '''
    ------------------------------------------------------------------------
      Callback function for minimizer - to save array and function eval at each iteration
    ------------------------------------------------------------------------
    '''
    global Nfeval, value_all, chi_params_all
    #print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], rosen(Xi))
    # pickle output in case not converge
    value_all[Nfeval] = function_to_minimize(chi,chi_params, income_tax_parameters, ss_parameters, iterative_params, omega_SS, rho, lambdas, tau_bq, e, output_dir)
    chi_params_all[:,Nfeval] = chi
    dict_GMM = dict([('values', value_all), ('chi_params', chi_params_all)])
    ss_init_path = os.path.join(output_dir, "Saved_moments/SS_init_all.pkl")
    pickle.dump(dict_GMM, open(ss_init_path, "wb"))

    Nfeval += 1


def check_wealth_calibration(wealth_model, factor_model, params):
    '''
    Creates a vector of the percent differences between the
    model and data wealth moments for the two age groups for
    each J group.

    Inputs:
        wealth_model = [S,J] array, model wealth levels
        factor_model = scalar, factor to convert wealth levels to dollars
        params       = length 2 tuple, (wealth_dir, J)
        wealth_dir   = string, directory containing wealth data moments
        J            = integer, number of lifetime income groups

    Functions called: 
        pct_dif_func

    Objects in function:
        wealth path          = string, path of pickle file with wealth data moments 
        wealth_dict          = dictionary, contains wealth data moments
        wealth_model         = [S,J] array, wealth holdings of model households
        wealth_model_dollars = [S,J] array, wealth holdings of model households in dollars
        wealth_fits          = [2*J,] vector, fits for how well the model wealth levels match the data wealth levels

    Returns: wealth_fits
    '''

    wealth_dir, J = params

    # Import the wealth data moments
    wealth_path = os.path.join(
        wealth_dir, "Saved_moments/wealth_data_moments.pkl")
    wealth_dict = pickle.load(open(wealth_path, "rb"))
    # Set lowest ability group's wealth to be a positive, not negative, number
    # for the calibration
    wealth_dict['wealth_data_array'][2:26, 0] = 500.0

    # Convert wealth levels from model to dollar terms
    wealth_model_dollars = wealth_model * factor_model
    wealth_fits = np.zeros(2 * J)
    # Look at the percent difference between the fits for the first age group (20-44) and second age group (45-65)
    #   The wealth_data_array moment indices are shifted because they start at age 18
    # The :: indices is so that we look at the 2 moments for the lowest group,
    # the 2 moments for the second lowest group, etc in order
    wealth_fits[0::2] = pct_dif_func(np.mean(wealth_model_dollars[
                                      :24], axis=0), np.mean(wealth_dict['wealth_data_array'][2:26], axis=0))
    wealth_fits[1::2] = pct_dif_func(np.mean(wealth_model_dollars[
                                      24:45], axis=0), np.mean(wealth_dict['wealth_data_array'][26:47], axis=0))
    return wealth_fits
