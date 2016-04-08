'''
------------------------------------------------------------------------
Last updated 4/8/2016

Functions for generating omega, the T x S array which describes the
demographics of the population

This py-file calls the following other file(s):
            utils.py
            data\demographic\demographic_data.csv
            data\demographic\mortality_rates.csv

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            OUTPUT/fert_rates.png
            OUTPUT/mort_rates.png
            OUTPUT/survival_rate.png
            OUTPUT/cum_mort_rate.png
            OUTPUT/imm_rates.png
            OUTPUT/Population.png
            OUTPUT/Population_growthrate.png
            OUTPUT/omega_init.png
            OUTPUT/omega_ss.png
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import scipy.optimize as opt
import utils

cur_path = os.path.split(os.path.abspath(__file__))[0]
DEMO_DIR = os.path.join(cur_path, "data", "demographic")

pd.options.mode.chained_assignment = None


'''
------------------------------------------------------------------------
    Import data sets
------------------------------------------------------------------------
Population data:
    Obtained from:
        Annual Estimates of the Resident Population by Single Year of
            Age and Sex: April 1, 2010 to July 1, 2013
             (Both sexes)
        National Characteristics, Vintage 2013
        US Census Bureau
        http://www.census.gov/popest/data/national/asrh/2013/index.html
Mortality rates data:
    Obtained from:
        Male and Female death probabilities
        Actuarial Life table, 2010
        Social Security Administration
        http://www.ssa.gov/oact/STATS/table4c6.html
Fertility rates data:
    Obtained from:
        Births and birth rates, by age of mother, US, 2010
        National Vital Statistics Reports, CDC
        http://www.cdc.gov/nchs/data/nvsr/nvsr60/nvsr60_02.pdf
        Since rates are per 1000 women, the data is divided by 1000
------------------------------------------------------------------------
'''

# Population data
demo_file = utils.read_file(cur_path, "data/demographic/demographic_data.csv")
data = pd.read_table(demo_file, sep=',', header=0)
data = data.set_index('Age')
# Remove commas in the data
for index, value in enumerate(data['2010']):
    data['2010'][index] = int(value.replace(',', ''))
for index, value in enumerate(data['2011']):
    data['2011'][index] = int(value.replace(',', ''))
for index, value in enumerate(data['2012']):
    data['2012'][index] = int(value.replace(',', ''))
for index, value in enumerate(data['2013']):
    data['2013'][index] = int(value.replace(',', ''))
# Create a copy of the data to be used elsewhere, without changing the
# main data
data_raw = data.copy(deep=True)

# Mortality rates data
#mort_data = pd.read_table(os.path.join(DEMO_DIR, 'mortality_rates.csv'), sep=',')
mort_file = utils.read_file(cur_path, "data/demographic/mortality_rates.csv")
mort_data = pd.read_table(mort_file, sep=',')
# Remove commas in the data
for index, value in enumerate(mort_data['male_weight']):
    mort_data['male_weight'][index] = float(value.replace(',', ''))
for index, value in enumerate(mort_data['female_weight']):
    mort_data['female_weight'][index] = float(value.replace(',', ''))
# Average male and female death rates
mort_data['mort_rate'] = (
    (np.array(mort_data.male_death.values).astype(float) * np.array(
        mort_data.male_weight.values).astype(float)) + (np.array(
            mort_data.female_death.values).astype(float) * np.array(
            mort_data.female_weight.values).astype(float))) / (
    np.array(mort_data.male_weight.values).astype(float) + np.array(
        mort_data.female_weight.values).astype(float))
mort_data = mort_data[mort_data.mort_rate.values < 1]
del mort_data['male_death'], mort_data[
    'female_death'], mort_data['male_weight'], mort_data[
    'female_weight'], mort_data['male_expectancy'], mort_data[
    'female_expectancy']
# As the data gives the probability of death, one minus the rate will
# give the survial rate
mort_data['surv_rate'] = 1 - mort_data.mort_rate
# Create an array of death rates of children

# Fertility rates data
fert_data = np.array(
    [.4, 34.3, 17.3, 58.3, 90.0, 108.3, 96.6, 45.9, 10.2, .7]) / 1000
# Fertility rates are given in age groups of 5 years, so the following
# are the midpoints of those groups
age_midpoint = np.array([12, 17, 16, 18.5, 22, 27, 32, 37, 42, 49.5])

'''
------------------------------------------------------------------------
    Define functions
------------------------------------------------------------------------
'''

def fit_exp_right(params, point1, point2):
    # Fit exponentials to two points for right tail of distributions
    a, b = params
    x1, y1 = point1
    x2, y2 = point2
    error1 = a*b**(-x1) - y1
    error2 = a*b**(-x2) - y2
    return [error1, error2]


def fit_exp_left(params, point1, point2):
    # Fit exponentials to two points for left tail of distributions
    a, b = params
    x1, y1 = point1
    x2, y2 = point2
    error1 = a*b**(x1) - y1
    error2 = a*b**(x2) - y2
    return [error1, error2]


def exp_int(points, a, b):
    top = a * ((1.0/(b**40)) - b**(-points))
    bottom = np.log(b)
    return top / bottom


def integrate(func, points):
    params_guess = [1, 1]
    a, b = opt.fsolve(fit_exp_right, params_guess, args=(
        [40, poly.polyval(40, func)], [49.5, .0007]))
    func_int = poly.polyint(func)
    integral = np.empty(points.shape)
    integral[points <= 40] = poly.polyval(points[points <= 40], func_int)
    integral[points > 40] = poly.polyval(40, func_int) + exp_int(
        points[points > 40], a, b)
    return np.diff(integral)

'''
------------------------------------------------------------------------
    Survival Rates
------------------------------------------------------------------------
'''


def get_survival(S, starting_age, ending_age, E):
    '''
    Parameters:
        S - Number of age cohorts (scalar)
        starting_age = initial age of cohorts (scalar)
        ending_age = ending age of cohorts (scalar)
        E = number of children (scalar)

    Returns:
        surv_array - S x 1 array of survival rates for each age cohort
        children_rate - starting_age x 1 array of survival
            rates for children
    '''
    mort_rate = np.array(mort_data.mort_rate)
    mort_poly = poly.polyfit(np.arange(mort_rate.shape[0]), mort_rate, deg=18)
    mort_int = poly.polyint(mort_poly)
    child_rate = poly.polyval(np.linspace(0, starting_age, E+1), mort_int)
    child_rate = np.diff(child_rate)
    mort_rate = poly.polyval(
        np.linspace(starting_age, ending_age, S+1), mort_int)
    mort_rate = np.diff(mort_rate)
    child_rate[child_rate < 0] = 0.0
    mort_rate[mort_rate < 0] = 0.0
    return 1.0 - mort_rate, 1.0 - child_rate

'''
------------------------------------------------------------------------
    Immigration Rates
------------------------------------------------------------------------
'''
pop_2010, pop_2011, pop_2012, pop_2013 = np.array(
    data_raw['2010'], dtype='f'), np.array(
        data_raw['2011'], dtype='f'), np.array(
        data_raw['2012'], dtype='f'), np.array(
        data_raw['2013'], dtype='f')


def get_immigration1(S, starting_age, ending_age, pop_2010, pop_2011, E):
    '''
    Parameters:
        S - Number of age cohorts
        starting_age - initial age of cohorts
        pop1 - initial population
        pop2 - population one year later

    Returns:
        im_array - S+E x 1 array of immigration rates for each
                   age cohort
    '''
    # Get survival rates for the S age groups
    surv_array, children_rate = get_survival(
        ending_age-starting_age, starting_age, ending_age, starting_age)
    surv_array = np.array(list(children_rate) + list(surv_array))
    # Only keep track of individuals in 2010 that don't die
    pop_2010 = pop_2010[:ending_age]
    # In 2011, individuals will have aged one year
    pop_2011 = pop_2011[1:ending_age+1]
    # The immigration rate will be 1 plus the percent change in
    # population (since death has already been accounted for)
    perc_change = ((pop_2011 - pop_2010) / pop_2010)
    # Remove the last entry, since individuals in the last period will die
    im_array = perc_change - (surv_array - 1)
    return im_array


def get_immigration2(S, starting_age, ending_age, E):
    '''
    Parameters:
        S - Number of age cohorts
        starting age - initial age of cohorts

    Returns:
        im_array - S x 1 array of immigration rates for each
                   age cohort
        child_imm_rate - starting_age x 1 array of immigration
            rates for children
    '''
    imm_rate_condensed1 = get_immigration1(
        S, starting_age, ending_age, pop_2010, pop_2011, E)
    imm_rate_condensed2 = get_immigration1(
        S, starting_age, ending_age, pop_2011, pop_2012, E)
    imm_rate_condensed3 = get_immigration1(
        S, starting_age, ending_age, pop_2012, pop_2013, E)
    im_array = (
        imm_rate_condensed1 + imm_rate_condensed2 + imm_rate_condensed3) / 3.0
    poly_imm = poly.polyfit(np.linspace(
        1, ending_age, ending_age-1), im_array[:-1], deg=18)
    poly_imm_int = poly.polyint(poly_imm)
    child_imm_rate = poly.polyval(np.linspace(
        0, starting_age, E+1), poly_imm_int)
    imm_rate = poly.polyval(np.linspace(
        starting_age, ending_age, S+1), poly_imm_int)
    child_imm_rate = np.diff(child_imm_rate)
    imm_rate = np.diff(imm_rate)
    imm_rate[-1] = 0.0
    return imm_rate, child_imm_rate

'''
------------------------------------------------------------------------
    Fertility Rates
------------------------------------------------------------------------
'''


def get_fert(S, starting_age, ending_age, E):
    '''
    Parameters:
        S - Number of age cohorts
        starting age - initial age of cohorts

    Returns:
        fert_rate - Sx1 array of fertility rates for each
            age cohort
        children_fertrate  - starting_age x 1 array of zeros, to be
            used in get_omega()
    '''
    # Fit a polynomial to the fertility rates
    poly_fert = poly.polyfit(age_midpoint, fert_data, deg=4)
    fert_rate = integrate(poly_fert, np.linspace(
        starting_age, ending_age, S+1))
    fert_rate /= 2.0
    children_fertrate_int = poly.polyint(poly_fert)
    children_fertrate_int = poly.polyval(np.linspace(
        0, starting_age, E + 1), children_fertrate_int)
    children_fertrate = np.diff(children_fertrate_int)
    children_fertrate /= 2.0
    children_fertrate[children_fertrate < 0] = 0
    children_fertrate[:int(10*S/float(ending_age-starting_age))] = 0
    return fert_rate, children_fertrate

'''
------------------------------------------------------------------------
    Generate graphs of mortality, fertility, and immigration rates
------------------------------------------------------------------------
'''


def rate_graphs(S, starting_age, ending_age, imm, fert, surv, child_imm,
                child_fert, child_mort, output_dir="./OUTPUT"):
    domain = np.arange(child_fert.shape[0] + S) + 1
    mort = mort_data.mort_rate
    domain2 = np.arange(mort.shape[0]) + 1
    domain4 = np.arange(child_imm.shape[0] + imm.shape[0]) + 1

    # Graph of fertility rates
    plt.figure()
    plt.plot(
        domain, list(child_fert)+list(fert), linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'fertility $f_s$')
    fert_rates = os.path.join(output_dir, "Demographics/fert_rates")
    plt.savefig(fert_rates)

    # Graph of mortality rates
    plt.figure()
    plt.plot(domain2[:ending_age-1], (1-np.array(list(child_mort)+list(surv)))[:-1], color='blue', linewidth=2)
    plt.plot(domain2[ending_age:], mort[
        ending_age:], color='blue', linestyle='--', linewidth=2)
    plt.axvline(x=ending_age, color='red', linestyle='-', linewidth=1)
    plt.xlabel(r'age $s$')
    plt.ylabel(r'mortality $\rho_s$')
    mort_rates = os.path.join(output_dir, "Demographics/mort_rates")
    plt.savefig(mort_rates)

    cum_surv_arr = np.cumprod(surv)
    domain3 = np.arange(surv.shape[0]) + 1

    # Graph of cumulative mortality rates
    plt.figure()
    plt.plot(domain3, cum_surv_arr)
    plt.xlabel(r'age $s$')
    plt.ylabel(r'survival rate $1-\rho_s$')
    surv_rates = os.path.join(output_dir, "Demographics/survival_rates")
    plt.savefig(surv_rates)
    cum_mort_rate = 1-cum_surv_arr
    plt.figure()
    plt.plot(domain3, cum_mort_rate)
    plt.xlabel(r'age $s$')
    plt.ylabel(r'cumulative mortality rate')
    cum_mort_rates = os.path.join(output_dir, "Demographics/cum_mort_rate")
    plt.savefig(cum_mort_rates)

    # Graph of immigration rates
    plt.figure()
    plt.plot(domain4, list(
        child_imm)+list(imm), linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'immigration $i_s$')
    imm_rates = os.path.join(output_dir, "Demographics/imm_rates")
    plt.savefig(imm_rates)

'''
------------------------------------------------------------------------
Generate graphs of Population
------------------------------------------------------------------------
'''


def pop_graphs(S, T, starting_age, ending_age, children, g_n, omega,
               output_dir="./OUTPUT"):
    N = omega[T].sum() + children[T].sum()
    x = children.sum(1) + omega.sum(1)
    x2 = 100 * np.diff(x)/x[:-1]

    plt.figure()
    plt.plot(np.arange(T+S)+1, x, 'b', linewidth=2)
    plt.title('Population Size (as a percent of the initial population)')
    plt.xlabel(r'Time $t$')
    # plt.ylabel('Population size, as a percent of initial population')
    pop = os.path.join(output_dir, "Demographics/imm_rates")
    plt.savefig(pop)

    plt.figure()
    plt.plot(np.arange(T+S-1)+1, x2, 'b', linewidth=2)
    plt.axhline(y=100 * g_n, color='r', linestyle='--', label=r'$\bar{g}_n$')
    plt.legend(loc=0)
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'Population growth rate $g_n$')
    # plt.title('Population Growth rate over time')
    pop_growth = os.path.join(output_dir, "Demographics/Population_growthrate")
    plt.savefig(pop_growth)

    plt.figure()
    plt.plot(np.arange(S+int(starting_age * S / (
        ending_age-starting_age)))+1, list(
        children[0, :]) + list(
        omega[0, :]), linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'$\omega_{s,1}$')
    omega_init = os.path.join(output_dir, "Demographics/omega_init")
    plt.savefig(omega_init)

    plt.figure()
    plt.plot(np.arange(S+int(starting_age * S / (
        ending_age-starting_age)))+1, list(
        children[T, :]/N) + list(
        omega[T, :]/N), linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'$\overline{\omega}$')
    omega_ss = os.path.join(output_dir, "Demographics/omega_ss")
    plt.savefig(omega_ss)

'''
------------------------------------------------------------------------
    Generate Demographics
------------------------------------------------------------------------
'''


def get_omega(S, T, starting_age, ending_age, E, flag_graphs):
    '''
    Inputs:
        S - Number of age cohorts (scalar)
        T - number of time periods in TPI (scalar)
        starting_age - initial age of cohorts (scalar)
        ending_age = ending age of cohorts (scalar)
        E = number of children (scalar)
        flag_graphs = graph variables or not (bool)
    Outputs:
        omega_big = array of all population weights over time ((T+S)x1 array)
        g_n_SS = steady state growth rate (scalar)
        omega_SS = steady state population weights (Sx1 array)
        surv_array = survival rates (Sx1 array)
        rho = mortality rates (Sx1 array)
        g_n_vec = population growth rate over time ((T+S)x1 array)
    '''
    data1 = data
    pop_data = np.array(data1['2010'])
    poly_pop = poly.polyfit(np.linspace(
        0, pop_data.shape[0]-1, pop_data.shape[0]), pop_data, deg=11)
    poly_int_pop = poly.polyint(poly_pop)
    pop_int = poly.polyval(np.linspace(
        starting_age, ending_age, S+1), poly_int_pop)
    new_omega = pop_int[1:]-pop_int[:-1]
    surv_array, children_rate = get_survival(S, starting_age, ending_age, E)
    surv_array[-1] = 0.0
    imm_array, children_im = get_immigration2(S, starting_age, ending_age, E)
    imm_array *= 0.0
    children_im *= 0.0
    fert_rate, children_fertrate = get_fert(S, starting_age, ending_age, E)
    cum_surv_rate = np.cumprod(surv_array)
    if flag_graphs:
        rate_graphs(S, starting_age, ending_age, imm_array, fert_rate, surv_array, children_im, children_fertrate, children_rate)
    children_int = poly.polyval(np.linspace(0, starting_age, E + 1), poly_int_pop)
    sum2010 = pop_int[-1] - children_int[0]
    new_omega /= sum2010
    children = np.diff(children_int)
    children /= sum2010
    children = np.tile(children.reshape(1, E), (T + S, 1))
    omega_big = np.tile(new_omega.reshape(1, S), (T + S, 1))
    # Generate the time path for each age group
    for t in xrange(1, T + S):
        # Children are born and then have to wait 20 years to enter the model
        omega_big[t, 0] = children[t-1, -1] * (children_rate[-1] + children_im[-1])
        omega_big[t, 1:] = omega_big[t-1, :-1] * (surv_array[:-1] + imm_array[:-1])
        children[t, 1:] = children[t-1, :-1] * (children_rate[:-1] + children_im[:-1])
        children[t, 0] = (omega_big[t-1, :] * fert_rate).sum(0) + (children[t-1] * children_fertrate).sum(0)
    OMEGA = np.zeros(((S + E), (S + E)))
    OMEGA[0, :] = np.array(list(children_fertrate) + list(fert_rate))
    OMEGA += np.diag(np.array(list(children_rate) + list(surv_array[:-1])) + np.array(list(children_im) + list(imm_array[:-1])), -1)
    eigvalues, eigvectors = np.linalg.eig(OMEGA)
    mask = eigvalues.real != 0
    eigvalues = eigvalues[mask]
    mask2 = eigvalues.imag == 0
    eigvalues = eigvalues[mask2].real
    g_n_SS = eigvalues - 1
    eigvectors = np.abs(eigvectors.T)
    eigvectors = eigvectors[mask]
    omega_SS = eigvectors[mask2].real
    if eigvalues.shape[0] != 1:
        ind = ((abs(omega_SS.T/omega_SS.T.sum(0) - np.array(list(children[-1, :]) + list(omega_big[-1, :])).reshape(S+E, 1)).sum(0))).argmin()
        omega_SS = omega_SS[ind]
        g_n_SS = [g_n_SS[ind]]
    omega_SS = omega_SS[E:]
    omega_SS /= omega_SS.sum()
    # Creating the different ability level bins
    if flag_graphs:
        pop_graphs(S, T, starting_age, ending_age, children, g_n_SS[0], omega_big)
    N_vector = omega_big.sum(1)
    g_n_vec = N_vector[1:] / N_vector[:-1] -1.0
    g_n_vec = np.append(g_n_vec, g_n_SS[0])
    rho = 1.0 - surv_array
    return omega_big, g_n_SS[0], omega_SS, surv_array, rho, g_n_vec
