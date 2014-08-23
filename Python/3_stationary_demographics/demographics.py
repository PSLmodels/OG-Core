'''
------------------------------------------------------------------------
Last updated 8/20/2014

Functions for generating omega, the T x S x J array which describes the
demographics of the population

This py-file calls the following other file(s):
            data\demographic\demographic_data.csv
            data\demographic\mortality_rates.csv
------------------------------------------------------------------------
'''

'''
------------------------------------------------------------------------
    Packages
------------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import scipy.optimize as opt

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
data = pd.read_table('data/demographic/demographic_data.csv', sep=',')
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
data_raw = data.copy(deep=True)

# Mortality rates data
mort_data = pd.read_table('data/demographic/mortality_rates.csv', sep=',')
for index, value in enumerate(mort_data['male_weight']):
    mort_data['male_weight'][index] = float(value.replace(',', ''))
for index, value in enumerate(mort_data['female_weight']):
    mort_data['female_weight'][index] = float(value.replace(',', ''))
# Average male and female death rates
mort_data['mort_rate'] = (
    (np.array(mort_data.male_death.values).astype(float) * np.array(mort_data.male_weight.values).astype(float)) + (np.array(mort_data.female_death.values).astype(float) * np.array(mort_data.female_weight.values).astype(float))) / (np.array(mort_data.male_weight.values).astype(float) + np.array(mort_data.female_weight.values).astype(float))
mort_data = mort_data[mort_data.mort_rate.values < 1]
del mort_data['male_death'], mort_data['female_death'], mort_data['male_weight'], mort_data['female_weight'], mort_data['male_expectancy'], mort_data['female_expectancy']
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

# Fit exponentials to two points for right tail of distributions


def fit_exp_right(params, point1, point2):
    a, b = params
    x1, y1 = point1
    x2, y2 = point2
    error1 = a*b**(-x1) - y1
    error2 = a*b**(-x2) - y2
    return [error1, error2]

# Fit exponentials to two points for left tail of distributions


def fit_exp_left(params, point1, point2):
    a, b = params
    x1, y1 = point1
    x2, y2 = point2
    error1 = a*b**(x1) - y1
    error2 = a*b**(x2) - y2
    return [error1, error2]


'''
------------------------------------------------------------------------
    Survival Rates
------------------------------------------------------------------------
'''


def get_survival(S, starting_age):
    '''
    Parameters:
        S - Number of age cohorts
        J - Number of ability types

    Returns:
        surv_array - S x J array of survival rates for each age cohort
    '''
    ending_age = starting_age + 60
    survival_rate = np.array(mort_data.surv_rate)[starting_age: ending_age]
    surv_rate_condensed = np.zeros(S)
    # If S < 60, then group years together
    for s in xrange(S):
        surv_rate_condensed[s] = np.product(
            survival_rate[s*(60/S):(s+1)*(60/S)])
    # All individuals must die if they reach the last age group
    surv_rate_condensed[-1] = 0
    children_rate = np.array(mort_data[(
        mort_data.age < starting_age)].surv_rate)
    children_rate_condensed = np.zeros(starting_age * S / 60.0)
    for s in xrange(int(starting_age * S / 60.0)):
        children_rate_condensed[s] = np.product(
            children_rate[s*(60/S):(s+1)*(60/S)])
    return surv_rate_condensed, children_rate_condensed

'''
------------------------------------------------------------------------
    Immigration Rates
------------------------------------------------------------------------
'''
pop_2010, pop_2011, pop_2012, pop_2013 = np.array(data_raw['2010'], dtype='f'), np.array(
        data_raw['2011'], dtype='f'), np.array(data_raw['2012'], dtype='f'), np.array(data_raw['2013'], dtype='f')

def get_immigration1(S, starting_age, pop_2010, pop_2011):
    '''
    Parameters:
        S - Number of age cohorts
        J - Number of ability types

    Returns:
        im_array - (S-1) x J array of immigration rates for each
                   age cohort
    '''
    ending_age = starting_age + 60
    
    # Get survival rates for the S age groups
    surv_array, children_rate = get_survival(60, starting_age)
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
    im_array2 = im_array[starting_age:]
    imm_rate_condensed = np.zeros(S)
    # If S < 60, then group years together
    for s in xrange(S):
        imm_rate_condensed[s] = np.product(1 + im_array2[
            s*(60/S):(s+1)*(60/S)]) - 1
    children_im = im_array[:starting_age]
    children_im_condensed = np.zeros(starting_age * S / 60.0)
    for s in xrange(int(starting_age * S / 60.0)):
        children_im_condensed[s] = np.product(1 + children_im[
            s*(60/S):(s+1)*(60/S)]) - 1
    return imm_rate_condensed, children_im_condensed

def get_immigration2(S, starting_age):
    imm_rate_condensed1, children_im_condensed1 = get_immigration1(S, starting_age, pop_2010, pop_2011)
    imm_rate_condensed2, children_im_condensed2 = get_immigration1(S, starting_age, pop_2011, pop_2012)
    imm_rate_condensed3, children_im_condensed3 = get_immigration1(S, starting_age, pop_2012, pop_2013)
    imm_rate = (imm_rate_condensed1 + imm_rate_condensed2 + imm_rate_condensed3) / 3.0
    child_imm_rate = (children_im_condensed1 + children_im_condensed2 + children_im_condensed3) / 3.0
    return imm_rate, child_imm_rate

'''
------------------------------------------------------------------------
    Fertility Rates
------------------------------------------------------------------------
'''


def get_fert(S, starting_age):
    '''
    Parameters:
        S - Number of age cohorts
        J - Number of ability types

    Returns:
        fert_rate - S x J array of fertility rates for each age cohort
        children  - 15 x J array of zeros, to be used in get_omega()
    '''
    ending_age = starting_age + 60
    # Fit a polynomial to the fertility rates
    poly_fert = poly.polyfit(age_midpoint, fert_data, deg=4)
    fert_rate = poly.polyval(
        np.linspace(starting_age, ending_age-1, 60), poly_fert)
    # Do not allow negative fertility rates, or nonzero rates outside of
    # a certain age range
    params = [1, 1]
    point1 = [40, poly.polyval(40,poly_fert)]
    params = opt.fsolve(fit_exp_right, params, args=(
        point1, [49.5, .0007]))
    domain = np.arange(11) + 40
    new_end = params[0] * params[1]**(-domain)
    fert_rate[40-starting_age:51-starting_age] = new_end
    for i in xrange(60):
        if np.linspace(starting_age, ending_age-1, 60)[i] >= 51 or np.linspace(
                starting_age, ending_age-1, 60)[i] < 10:
            fert_rate[i] = 0
        if fert_rate[i] < 0:
            fert_rate[i] = 0
    fert_rate_condensed = np.zeros(S)
    # If S < 60, then group years together
    for s in xrange(S):
        fert_rate_condensed[s] = np.prod(1 + fert_rate[
            s*(60/S):(s+1)*(60/S)]) - 1
    # Divide the fertility rate by 2, since it will be used for men and women
    fert_rate_condensed /= 2.0
    children_fertrate_int = poly.polyint(poly_fert)
    children_fertrate_int = poly.polyval(np.linspace(0, starting_age-.5, (starting_age * S / 60.0) + 1), children_fertrate_int) # No, I didn't cheat here a little
    children_fertrate = np.diff(children_fertrate_int)
    children_fertrate /= 2.0
    return fert_rate_condensed, children_fertrate

'''
------------------------------------------------------------------------
    Generate graphs of mortality, fertility, and immigration rates
------------------------------------------------------------------------
'''


def rate_graphs(S, starting_age, imm, fert, child_imm, child_fert):
    domain = np.arange(child_fert.shape[0] + S) + 1
    mort = np.array(mort_data.mort_rate)[:100]
    domain2 = np.arange(mort.shape[0]) + 1
    domain4 = np.arange(child_imm.shape[0] + imm.shape[0]) + 1

    plt.figure()
    plt.plot(
        domain, list(child_fert)+list(fert), linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'fertility $f_s$')
    plt.savefig('OUTPUT/fert_rates')

    plt.figure()
    plt.plot(domain2[:60+starting_age], mort[
        :60+starting_age], color='blue', linewidth=2)
    plt.plot(domain2[60+starting_age:], mort[
        60+starting_age:], color='blue', linestyle='--', linewidth=2)
    plt.axvline(x=60+starting_age, color='red', linestyle='-', linewidth=1)
    plt.xlabel(r'age $s$')
    plt.ylabel(r'mortality $\rho_s$')
    plt.savefig('OUTPUT/mort_rates')

    mort = np.array(mort_data.mort_rate)
    surv_arr = 1-mort
    cum_surv_arr = np.zeros(len(mort))
    for i in xrange(len(mort)):
        cum_surv_arr[i] = np.prod(surv_arr[:i])
    domain3 = np.arange(mort.shape[0]) + 1

    plt.figure()
    plt.plot(domain3, cum_surv_arr)
    plt.savefig('OUTPUT/survival_rate')
    cum_mort_rate = 1-cum_surv_arr
    plt.figure()
    plt.plot(domain3, cum_mort_rate)
    plt.savefig('OUTPUT/cum_mort_rate')

    plt.figure()
    plt.plot(domain4, [child_imm[0]] + list(
        child_imm)+list(imm[:-1]), linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'immigration $i_s$')
    plt.savefig('OUTPUT/imm_rates')

'''
------------------------------------------------------------------------
    Generate Omega array
------------------------------------------------------------------------
'''


def get_omega(S, J, T, bin_weights, starting_age):
    '''
    Parameters:
        S - Number of age cohorts
        J - Number of ability types

    Returns:

    '''
    ending_age = starting_age + 60
    data1 = data
    pop_data = np.array(data1['2010'])
    poly_pop = poly.polyfit(np.linspace(
        0, pop_data.shape[0]-1, pop_data.shape[0]), pop_data, deg=11)
    poly_int_pop = poly.polyint(poly_pop)
    pop_int = poly.polyval(np.linspace(
        starting_age, ending_age, S+1), poly_int_pop)
    new_omega = np.zeros(S)
    for s in xrange(S):
        new_omega[s] = pop_int[s+1] - pop_int[s]
    surv_array, children_rate = get_survival(S, starting_age)
    imm_array, children_im = get_immigration2(S, starting_age)
    fert_rate, children_fertrate = get_fert(S, starting_age)
    cum_surv_rate = np.zeros(S)
    for i in xrange(S):
        cum_surv_rate[i] = np.prod(surv_array[:i])
    rate_graphs(
        S, starting_age, imm_array, fert_rate, children_im, children_fertrate)
    children_int = poly.polyval(
        np.linspace(0, starting_age, (starting_age * S / 60.0) + 1), poly_int_pop)
    sum2010 = pop_int[-1] - children_int[0]
    new_omega /= sum2010
    children = np.diff(children_int)
    children /= sum2010
    children = np.tile(children.reshape(1, (starting_age * S / 60.0)), (T + S, 1))
    omega_big = np.tile(new_omega.reshape(1, S), (T + S, 1))
    # This is cheating 
    # children[0,0] += 0.00011 * J
    # children[0,1] += 0.00005 * J
    # children[0,2] += 0.00002 * J
    # children[0,3] += 0.000007 * J
    # Generate the time path for each age/abilty group
    for t in xrange(1, T + S):
        # Children are born and then have to wait 20 years to enter the model
        omega_big[t, 0] = children[t-1, -1] * (
            children_rate[-1] + children_im[-1])
        omega_big[t, 1:] = omega_big[t-1, :-1] * (
            surv_array[:-1] + imm_array[
                :-1])
        children[t, 1:] = children[t-1, :-1] * (
            children_rate[:-1] + children_im[:-1])
        children[t, 0] = ((omega_big[t-1, :] * fert_rate).sum(0) + (
            children[t-1] * children_fertrate).sum(0)) * (1 + children_im[0])
    OMEGA = np.zeros((S + int(starting_age * S / 60.0), S + int(starting_age * S / 60.0)))
    OMEGA[0, :] = np.array(list(children_fertrate) + list(
        fert_rate)) * (1 + children_im[0])
    OMEGA += np.diag(np.array(list(children_rate[:]) + list(
        surv_array[:-1])) + np.array(list(children_im) + list(
            imm_array[:-1])), -1)
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
        ind = ((abs(omega_SS.T/omega_SS.T.sum(0) - np.array(
            list(children[-1, :]) + list(omega_big[-1, :])).reshape(
            S+int(starting_age * S / 60.0), 1))).sum(0)).argmin()
        omega_SS = omega_SS[ind]
        g_n_SS = [g_n_SS[ind]]
    omega_SS /= omega_SS.sum()
    # Creating the different ability level bins
    omega_SS = np.tile(omega_SS.reshape(S+int(starting_age * S / 60.0), 1), (1, J)) * bin_weights.reshape(1,J)
    omega_big = np.tile(omega_big.reshape(T+S,S,1), (1,1,J)) * bin_weights.reshape(1,1,J)
    children = np.tile(children.reshape(T+S, int(starting_age * S / 60.0), 1), (1, 1, J)) * bin_weights.reshape(1,1,J)

    return omega_big, g_n_SS, omega_SS, children, surv_array
