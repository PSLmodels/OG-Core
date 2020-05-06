'''
------------------------------------------------------------------------
Functions for generating demographic objects necessary for the OG-USA
model
------------------------------------------------------------------------
'''
# Import packages
import os
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as si
import pandas as pd
from ogusa import parameter_plots as pp


# create output director for figures
CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
OUTPUT_DIR = os.path.join(CUR_PATH, 'OUTPUT', 'Demographics')
if os.access(OUTPUT_DIR, os.F_OK) is False:
    os.makedirs(OUTPUT_DIR)


'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''


def get_fert(totpers, min_yr, max_yr, graph=False):
    '''
    This function generates a vector of fertility rates by model period
    age that corresponds to the fertility rate data by age in years
    (Source: National Vital Statistics Reports, Volume 64, Number 1,
    January 15, 2015, Table 3, final 2013 data
    http://www.cdc.gov/nchs/data/nvsr/nvsr64/nvsr64_01.pdf)

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        fert_rates (Numpy array): fertility rates for each model period
            of life

    '''
    # Get current population data (2013) for weighting
    pop_file = os.path.join(
        CUR_PATH, 'data', 'demographic', 'pop_data.csv')
    pop_data = pd.read_csv(pop_file, thousands=',')
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    curr_pop = np.array(pop_data_samp['2013'], dtype='f')
    curr_pop_pct = curr_pop / curr_pop.sum()
    # Get fertility rate by age-bin data
    fert_data = (np.array([0.0, 0.0, 0.3, 12.3, 47.1, 80.7, 105.5, 98.0,
                           49.3, 10.4, 0.8, 0.0, 0.0]) / 2000)
    # Mid points of age bins
    age_midp = np.array([9, 10, 12, 16, 18.5, 22, 27, 32, 37, 42, 47,
                         55, 56])
    # Generate interpolation functions for fertility rates
    fert_func = si.interp1d(age_midp, fert_data, kind='cubic')
    # Calculate average fertility rate in each age bin using trapezoid
    # method with a large number of points in each bin.
    binsize = (max_yr - min_yr + 1) / totpers
    num_sub_bins = float(10000)
    len_subbins = (np.float64(100 * num_sub_bins)) / totpers
    age_sub = (np.linspace(np.float64(binsize) / num_sub_bins,
                           np.float64(max_yr),
                           int(num_sub_bins*max_yr)) - 0.5 *
               np.float64(binsize) / num_sub_bins)
    curr_pop_sub = np.repeat(np.float64(curr_pop_pct) / num_sub_bins,
                             num_sub_bins)
    fert_rates_sub = np.zeros(curr_pop_sub.shape)
    pred_ind = (age_sub > age_midp[0]) * (age_sub < age_midp[-1])
    age_pred = age_sub[pred_ind]
    fert_rates_sub[pred_ind] = np.float64(fert_func(age_pred))
    fert_rates = np.zeros(totpers)
    end_sub_bin = 0
    for i in range(totpers):
        beg_sub_bin = int(end_sub_bin)
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        fert_rates[i] = ((
            curr_pop_sub[beg_sub_bin:end_sub_bin] *
            fert_rates_sub[beg_sub_bin:end_sub_bin]).sum() /
            curr_pop_sub[beg_sub_bin:end_sub_bin].sum())

    if graph:
        pp.plot_fert_rates(fert_func, age_midp, totpers, min_yr, max_yr,
                           fert_data, fert_rates, output_dir=OUTPUT_DIR)

    return fert_rates


def get_mort(totpers, min_yr, max_yr, graph=False):
    '''
    This function generates a vector of mortality rates by model period
    age.
    (Source: Male and Female death probabilities Actuarial Life table,
    2011 Social Security Administration,
    http://www.ssa.gov/oact/STATS/table4c6.html)

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        mort_rates (Numpy array) mortality rates that correspond to each
            period of life
        infmort_rate (scalar): infant mortality rate from 2015 U.S. CIA
            World Factbook

    '''
    # Get mortality rate by age data
    infmort_rate = 0.00587  # taken from 2015 U.S. infant mortality rate
    mort_file = os.path.join(
        CUR_PATH, 'data', 'demographic', 'mort_rates2011.csv')
    mort_data = pd.read_csv(mort_file, thousands=',')
    age_year_all = mort_data['Age'] + 1
    mort_rates_all = (
        ((mort_data['Male Mort. Rate'] * mort_data['Num. Male Lives']) +
         (mort_data['Female Mort. Rate'] *
          mort_data['Num. Female Lives'])) /
        (mort_data['Num. Male Lives'] + mort_data['Num. Female Lives']))
    age_year_all = age_year_all[np.isfinite(mort_rates_all)]
    mort_rates_all = mort_rates_all[np.isfinite(mort_rates_all)]
    # Calculate implied mortality rates in sub-bins of mort_rates_all.
    mort_rates_mxyr = mort_rates_all[0:max_yr]
    num_sub_bins = int(100)
    len_subbins = ((np.float64((max_yr - min_yr + 1) * num_sub_bins)) /
                   totpers)
    mort_rates_sub = np.zeros(num_sub_bins * max_yr, dtype=float)
    for i in range(max_yr):
        mort_rates_sub[i * num_sub_bins:(i + 1) * num_sub_bins] =\
            (1 - ((1 - mort_rates_mxyr[i]) ** (1.0 / num_sub_bins)))
    mort_rates = np.zeros(totpers)
    end_sub_bin = 0
    for i in range(totpers):
        beg_sub_bin = int(end_sub_bin)
        end_sub_bin = int(np.rint((i + 1) * len_subbins))
        mort_rates[i] = (
            1 - (1 - (mort_rates_sub[beg_sub_bin:end_sub_bin])).prod())
    mort_rates[-1] = 1  # Mortality rate in last period is set to 1

    if graph:
        pp.plot_mort_rates_data(totpers, min_yr, max_yr, age_year_all,
                                mort_rates_all, infmort_rate,
                                mort_rates, output_dir=OUTPUT_DIR)

    return mort_rates, infmort_rate


def pop_rebin(curr_pop_dist, totpers_new):
    '''
    For cases in which totpers (E+S) is less than the number of periods
    in the population distribution data, this function calculates a new
    population distribution vector with totpers (E+S) elements.

    Args:
        curr_pop_dist (Numpy array): population distribution over N
            periods
        totpers_new (int): number of periods to which we are
            transforming the population distribution, >= 3

    Returns:
        curr_pop_new (Numpy array): new population distribution over
            totpers (E+S) periods that approximates curr_pop_dist

    '''
    # Number of periods in original data
    assert totpers_new >= 3
    totpers_orig = len(curr_pop_dist)
    if int(totpers_new) == totpers_orig:
        curr_pop_new = curr_pop_dist
    elif int(totpers_new) < totpers_orig:
        num_sub_bins = float(10000)
        curr_pop_sub = np.repeat(np.float64(curr_pop_dist) /
                                 num_sub_bins, num_sub_bins)
        len_subbins = ((np.float64(totpers_orig*num_sub_bins)) /
                       totpers_new)
        curr_pop_new = np.zeros(totpers_new, dtype=np.float64)
        end_sub_bin = 0
        for i in range(totpers_new):
            beg_sub_bin = int(end_sub_bin)
            end_sub_bin = int(np.rint((i + 1) * len_subbins))
            curr_pop_new[i] = \
                curr_pop_sub[beg_sub_bin:end_sub_bin].sum()
        # Return curr_pop_new to single precision float (float32)
        # datatype
        curr_pop_new = np.float32(curr_pop_new)

    return curr_pop_new


def get_imm_resid(totpers, min_yr, max_yr):
    '''
    Calculate immigration rates by age as a residual given population
    levels in different periods, then output average calculated
    immigration rate. We have to replace the first mortality rate in
    this function in order to adjust the first implied immigration rate
    (Source: Population data come from Annual Estimates of the Resident
    Population by Single Year of Age and Sex: April 1, 2010 to July 1,
    2013 (Both sexes) National Characteristics, Vintage 2013, US Census
    Bureau,
    http://www.census.gov/popest/data/national/asrh/2013/index.html)

    Args:
        totpers (int): total number of agent life periods (E+S), >= 3
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        graph (bool): =True if want graphical output

    Returns:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S

    '''
    pop_file = os.path.join(
        CUR_PATH, 'data', 'demographic', 'pop_data.csv')
    pop_data = pd.read_csv(pop_file, thousands=',')
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    pop_2010, pop_2011, pop_2012, pop_2013 = (
        np.array(pop_data_samp['2010'], dtype='f'),
        np.array(pop_data_samp['2011'], dtype='f'),
        np.array(pop_data_samp['2012'], dtype='f'),
        np.array(pop_data_samp['2013'], dtype='f'))
    pop_2010_EpS = pop_rebin(pop_2010, totpers)
    pop_2011_EpS = pop_rebin(pop_2011, totpers)
    pop_2012_EpS = pop_rebin(pop_2012, totpers)
    pop_2013_EpS = pop_rebin(pop_2013, totpers)
    # Create three years of estimated immigration rates for youngest age
    # individuals
    imm_mat = np.zeros((3, totpers))
    pop11vec = np.array([pop_2010_EpS[0], pop_2011_EpS[0],
                         pop_2012_EpS[0]])
    pop21vec = np.array([pop_2011_EpS[0], pop_2012_EpS[0],
                         pop_2013_EpS[0]])
    fert_rates = get_fert(totpers, min_yr, max_yr, False)
    mort_rates, infmort_rate = get_mort(totpers, min_yr, max_yr, False)
    newbornvec = np.dot(fert_rates, np.vstack((pop_2010_EpS,
                                               pop_2011_EpS,
                                               pop_2012_EpS)).T)
    imm_mat[:, 0] = ((pop21vec - (1 - infmort_rate) * newbornvec) /
                     pop11vec)
    # Estimate 3 years of immigration rates for all other-aged
    # individuals
    pop11mat = np.vstack((pop_2010_EpS[:-1], pop_2011_EpS[:-1],
                          pop_2012_EpS[:-1]))
    pop12mat = np.vstack((pop_2010_EpS[1:], pop_2011_EpS[1:],
                          pop_2012_EpS[1:]))
    pop22mat = np.vstack((pop_2011_EpS[1:], pop_2012_EpS[1:],
                          pop_2013_EpS[1:]))
    mort_mat = np.tile(mort_rates[:-1], (3, 1))
    imm_mat[:, 1:] = (pop22mat - (1 - mort_mat) * pop11mat) / pop12mat
    # Final estimated immigration rates are the averages over 3 years
    imm_rates = imm_mat.mean(axis=0)

    return imm_rates


def immsolve(imm_rates, *args):
    '''
    This function generates a vector of errors representing the
    difference in two consecutive periods stationary population
    distributions. This vector of differences is the zero-function
    objective used to solve for the immigration rates vector, similar to
    the original immigration rates vector from get_imm_resid(), that
    sets the steady-state population distribution by age equal to the
    population distribution in period int(1.5*S)

    Args:
        imm_rates (Numpy array):immigration rates that correspond to
            each period of life, length E+S
        args (tuple): (fert_rates, mort_rates, infmort_rate, omega_cur,
            g_n_SS)

    Returns:
        omega_errs (Numpy array): difference between omega_new and
            omega_cur_pct, length E+S

    '''
    fert_rates, mort_rates, infmort_rate, omega_cur_lev, g_n_SS = args
    omega_cur_pct = omega_cur_lev / omega_cur_lev.sum()
    totpers = len(fert_rates)
    OMEGA = np.zeros((totpers, totpers))
    OMEGA[0, :] = ((1 - infmort_rate) * fert_rates +
                   np.hstack((imm_rates[0], np.zeros(totpers-1))))
    OMEGA[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA[1:, 1:] += np.diag(imm_rates[1:])
    omega_new = np.dot(OMEGA, omega_cur_pct) / (1 + g_n_SS)
    omega_errs = omega_new - omega_cur_pct

    return omega_errs


def get_pop_objs(E, S, T, min_yr, max_yr, curr_year, GraphDiag=False):
    '''
    This function produces the demographics objects to be used in the
    OG-USA model package.

    Args:
        E (int): number of model periods in which agent is not
            economically active, >= 1
        S (int): number of model periods in which agent is economically
            active, >= 3
        T (int): number of periods to be simulated in TPI, > 2*S
        min_yr (int): age in years at which agents are born, >= 0
        max_yr (int): age in years at which agents die with certainty,
            >= 4
        curr_year (int): current year for which analysis will begin,
            >= 2016
        GraphDiag (bool): =True if want graphical output and printed
                diagnostics

    Returns:
        omega_path_S (Numpy array), time path of the population
            distribution from the current state to the steady-state,
            size T+S x S
        g_n_SS (scalar): steady-state population growth rate
        omega_SS (Numpy array): normalized steady-state population
            distribution, length S
        surv_rates (Numpy array): survival rates that correspond to
            each model period of life, lenght S
        mort_rates (Numpy array): mortality rates that correspond to
            each model period of life, length S
        g_n_path (Numpy array): population growth rates over the time
            path, length T + S

    '''
    # age_per = np.linspace(min_yr, max_yr, E+S)
    fert_rates = get_fert(E + S, min_yr, max_yr, graph=False)
    mort_rates, infmort_rate = get_mort(E + S, min_yr, max_yr,
                                        graph=False)
    mort_rates_S = mort_rates[-S:]
    imm_rates_orig = get_imm_resid(E + S, min_yr, max_yr)
    OMEGA_orig = np.zeros((E + S, E + S))
    OMEGA_orig[0, :] = ((1 - infmort_rate) * fert_rates +
                        np.hstack((imm_rates_orig[0], np.zeros(E+S-1))))
    OMEGA_orig[1:, :-1] += np.diag(1 - mort_rates[:-1])
    OMEGA_orig[1:, 1:] += np.diag(imm_rates_orig[1:])

    # Solve for steady-state population growth rate and steady-state
    # population distribution by age using eigenvalue and eigenvector
    # decomposition
    eigvalues, eigvectors = np.linalg.eig(OMEGA_orig)
    g_n_SS = (eigvalues[np.isreal(eigvalues)].real).max() - 1
    eigvec_raw =\
        eigvectors[:,
                   (eigvalues[np.isreal(eigvalues)].real).argmax()].real
    omega_SS_orig = eigvec_raw / eigvec_raw.sum()

    # Generate time path of the nonstationary population distribution
    omega_path_lev = np.zeros((E + S, T + S))
    pop_file = os.path.join(
        CUR_PATH, 'data', 'demographic', 'pop_data.csv')
    pop_data = pd.read_csv(pop_file, thousands=',')
    pop_data_samp = pop_data[(pop_data['Age'] >= min_yr - 1) &
                             (pop_data['Age'] <= max_yr - 1)]
    pop_2013 = np.array(pop_data_samp['2013'], dtype='f')
    # Generate the current population distribution given that E+S might
    # be less than max_yr-min_yr+1
    age_per_EpS = np.arange(1, E + S + 1)
    pop_2013_EpS = pop_rebin(pop_2013, E + S)
    pop_2013_pct = pop_2013_EpS / pop_2013_EpS.sum()
    # Age most recent population data to the current year of analysis
    pop_curr = pop_2013_EpS.copy()
    data_year = 2013
    pop_next = np.dot(OMEGA_orig, pop_curr)
    g_n_curr = ((pop_next[-S:].sum() - pop_curr[-S:].sum()) /
                pop_curr[-S:].sum())  # g_n in 2013
    pop_past = pop_curr  # assume 2012-2013 pop
    # Age the data to the current year
    for per in range(curr_year - data_year):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        g_n_curr = ((pop_next[-S:].sum() - pop_curr[-S:].sum()) /
                    pop_curr[-S:].sum())
        pop_past = pop_curr
        pop_curr = pop_next

    # Generate time path of the population distribution
    omega_path_lev[:, 0] = pop_curr.copy()
    for per in range(1, T + S):
        pop_next = np.dot(OMEGA_orig, pop_curr)
        omega_path_lev[:, per] = pop_next.copy()
        pop_curr = pop_next.copy()

    # Force the population distribution after 1.5*S periods to be the
    # steady-state distribution by adjusting immigration rates, holding
    # constant mortality, fertility, and SS growth rates
    imm_tol = 1e-14
    fixper = int(1.5 * S)
    omega_SSfx = (omega_path_lev[:, fixper] /
                  omega_path_lev[:, fixper].sum())
    imm_objs = (fert_rates, mort_rates, infmort_rate,
                omega_path_lev[:, fixper], g_n_SS)
    imm_fulloutput = opt.fsolve(immsolve, imm_rates_orig,
                                args=(imm_objs), full_output=True,
                                xtol=imm_tol)
    imm_rates_adj = imm_fulloutput[0]
    imm_diagdict = imm_fulloutput[1]
    omega_path_S = (omega_path_lev[-S:, :] /
                    np.tile(omega_path_lev[-S:, :].sum(axis=0), (S, 1)))
    omega_path_S[:, fixper:] = \
        np.tile(omega_path_S[:, fixper].reshape((S, 1)),
                (1, T + S - fixper))
    g_n_path = np.zeros(T + S)
    g_n_path[0] = g_n_curr.copy()
    g_n_path[1:] = ((omega_path_lev[-S:, 1:].sum(axis=0) -
                    omega_path_lev[-S:, :-1].sum(axis=0)) /
                    omega_path_lev[-S:, :-1].sum(axis=0))
    g_n_path[fixper + 1:] = g_n_SS
    omega_S_preTP = (pop_past.copy()[-S:]) / (pop_past.copy()[-S:].sum())
    imm_rates_mat = np.hstack((
        np.tile(np.reshape(imm_rates_orig[E:], (S, 1)), (1, fixper)),
        np.tile(np.reshape(imm_rates_adj[E:], (S, 1)), (1, T + S - fixper))))

    if GraphDiag:
        # Check whether original SS population distribution is close to
        # the period-T population distribution
        omegaSSmaxdif = np.absolute(omega_SS_orig -
                                    (omega_path_lev[:, T] /
                                     omega_path_lev[:, T].sum())).max()
        if omegaSSmaxdif > 0.0003:
            print('POP. WARNING: Max. abs. dist. between original SS ' +
                  "pop. dist'n and period-T pop. dist'n is greater than" +
                  ' 0.0003. It is ' + str(omegaSSmaxdif) + '.')
        else:
            print('POP. SUCCESS: orig. SS pop. dist is very close to ' +
                  "period-T pop. dist'n. The maximum absolute " +
                  'difference is ' + str(omegaSSmaxdif) + '.')

        # Plot the adjusted steady-state population distribution versus
        # the original population distribution. The difference should be
        # small
        omegaSSvTmaxdiff = np.absolute(omega_SS_orig - omega_SSfx).max()
        if omegaSSvTmaxdiff > 0.0003:
            print('POP. WARNING: The maximimum absolute difference ' +
                  'between any two corresponding points in the original'
                  + ' and adjusted steady-state population ' +
                  'distributions is' + str(omegaSSvTmaxdiff) + ', ' +
                  'which is greater than 0.0003.')
        else:
            print('POP. SUCCESS: The maximum absolute difference ' +
                  'between any two corresponding points in the original'
                  + ' and adjusted steady-state population ' +
                  'distributions is ' + str(omegaSSvTmaxdiff))

        # Print whether or not the adjusted immigration rates solved the
        # zero condition
        immtol_solved = \
            np.absolute(imm_diagdict['fvec'].max()) < imm_tol
        if immtol_solved:
            print('POP. SUCCESS: Adjusted immigration rates solved ' +
                  'with maximum absolute error of ' +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ', which is less than the tolerance of ' +
                  str(imm_tol))
        else:
            print('POP. WARNING: Adjusted immigration rates did not ' +
                  'solve. Maximum absolute error of ' +
                  str(np.absolute(imm_diagdict['fvec'].max())) +
                  ' is greater than the tolerance of ' + str(imm_tol))

        # Test whether the steady-state growth rates implied by the
        # adjusted OMEGA matrix equals the steady-state growth rate of
        # the original OMEGA matrix
        OMEGA2 = np.zeros((E + S, E + S))
        OMEGA2[0, :] = ((1 - infmort_rate) * fert_rates +
                        np.hstack((imm_rates_adj[0], np.zeros(E+S-1))))
        OMEGA2[1:, :-1] += np.diag(1 - mort_rates[:-1])
        OMEGA2[1:, 1:] += np.diag(imm_rates_adj[1:])
        eigvalues2, eigvectors2 = np.linalg.eig(OMEGA2)
        g_n_SS_adj = (eigvalues[np.isreal(eigvalues2)].real).max() - 1
        if np.max(np.absolute(g_n_SS_adj - g_n_SS)) > 10 ** (-8):
            print('FAILURE: The steady-state population growth rate' +
                  ' from adjusted OMEGA is different (diff is ' +
                  str(g_n_SS_adj - g_n_SS) + ') than the steady-' +
                  'state population growth rate from the original' +
                  ' OMEGA.')
        elif np.max(np.absolute(g_n_SS_adj - g_n_SS)) <= 10 ** (-8):
            print('SUCCESS: The steady-state population growth rate' +
                  ' from adjusted OMEGA is close to (diff is ' +
                  str(g_n_SS_adj - g_n_SS) + ') the steady-' +
                  'state population growth rate from the original' +
                  ' OMEGA.')

        # Do another test of the adjusted immigration rates. Create the
        # new OMEGA matrix implied by the new immigration rates. Plug in
        # the adjusted steady-state population distribution. Hit is with
        # the new OMEGA transition matrix and it should return the new
        # steady-state population distribution
        omega_new = np.dot(OMEGA2, omega_SSfx)
        omega_errs = np.absolute(omega_new - omega_SSfx)
        print('The maximum absolute difference between the adjusted ' +
              'steady-state population distribution and the ' +
              'distribution generated by hitting the adjusted OMEGA ' +
              'transition matrix is ' + str(omega_errs.max()))

        # Plot the original immigration rates versus the adjusted
        # immigration rates
        immratesmaxdiff = \
            np.absolute(imm_rates_orig - imm_rates_adj).max()
        print('The maximum absolute distance between any two points ' +
              'of the original immigration rates and adjusted ' +
              'immigration rates is ' + str(immratesmaxdiff))

        # plots
        pp.plot_omega_fixed(age_per_EpS, omega_SS_orig, omega_SSfx, E,
                            S, output_dir=OUTPUT_DIR)
        pp.plot_imm_fixed(age_per_EpS, imm_rates_orig, imm_rates_adj, E,
                          S, output_dir=OUTPUT_DIR)
        pp.plot_population_path(age_per_EpS, pop_2013_pct,
                                omega_path_lev, omega_SSfx, curr_year,
                                E, S, output_dir=OUTPUT_DIR)

    # return omega_path_S, g_n_SS, omega_SSfx, survival rates,
    # mort_rates_S, and g_n_path
    return (omega_path_S.T, g_n_SS, omega_SSfx[-S:] /
            omega_SSfx[-S:].sum(), 1-mort_rates_S, mort_rates_S,
            g_n_path, imm_rates_mat.T, omega_S_preTP)
