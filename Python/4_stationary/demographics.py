'''
------------------------------------------------------------------------
Last updated 8/13/2014

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
        Births and birth rates, by age of mother, US, 2012
        National Vital Statistics Reports, CDC
        http://www.cdc.gov/nchs/data/nvsr/nvsr63/nvsr63_02.pdf
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
# Average male and female death rates
mort_data['mort_rate'] = (
    mort_data.prob_live_next_male + mort_data.prob_live_next_female) / 2
del mort_data['prob_live_next_female'], mort_data['prob_live_next_male']
# As the data gives the probability of death, one minus the rate will
# give the survial rate
mort_data['surv_rate'] = 1 - mort_data.mort_rate
# Create an array of death rates of children

# Fertility rates data
fert_data = np.array(
    [.3, 26.6, 12.3, 47.3, 81.2, 106.2, 98.7, 49.6, 10.5, .8]) / 1000
# Fertility rates are given in age groups of 5 years, so the following
# are the midpoints of those groups
age_midpoint = np.array([12, 17, 16, 18.5, 22, 27, 32, 37, 42, 49.5])

'''
------------------------------------------------------------------------
    Survival Rates
------------------------------------------------------------------------
'''


def get_survival(S, J, starting_age):
    '''
    Parameters:
        S - Number of age cohorts
        J - Number of ability types

    Returns:
        surv_array - S x J array of survival rates for each age cohort
    '''
    ending_age = starting_age + 60
    # Fit a polynomial to the data of survival rates (so that we are not
    # bound to groups that span at least a year
    # poly_surv = poly.polyfit(mort_data.age, mort_data.surv_rate, deg=10)
    # Evaluate the polynomial every year for individuals 15 to 75
    # survival_rate = poly.polyval(np.linspace(starting_age, ending_age-1, 60), poly_surv)
    survival_rate = np.array(mort_data.surv_rate)[starting_age: ending_age]
    # for i in xrange(survival_rate.shape[0]):
    #     if survival_rate[i] > 1.0:
    #         survival_rate[i] = 1.0
    surv_rate_condensed = np.zeros(S)
    # If S < 60, then group years together
    for s in xrange(S):
        surv_rate_condensed[s] = np.product(
            survival_rate[s*(60/S):(s+1)*(60/S)])
    # All individuals must die if they reach the last age group
    surv_rate_condensed[-1] = 0
    surv_array = np.tile(surv_rate_condensed.reshape(S, 1), (1, J))
    children_rate = np.array(mort_data[(mort_data.age < starting_age)].surv_rate)
    return surv_array, children_rate

'''
------------------------------------------------------------------------
    Immigration Rates
------------------------------------------------------------------------
'''


def get_immigration(S, J, starting_age):
    '''
    Parameters:
        S - Number of age cohorts
        J - Number of ability types

    Returns:
        im_array - (S-1) x J array of immigration rates for each
                   age cohort
    '''
    ending_age = starting_age + 60
    pop_2010, pop_2011 = np.array(data_raw['2010'], dtype='f'), np.array(
        data_raw['2011'], dtype='f')
    # Get survival rates for the S age groups
    surv_array, children_rate = get_survival(60, 1, starting_age)
    surv_array = np.array(list(children_rate) + list(surv_array))
    # Only keep track of individuals in 2010 that don't die
    pop_2010 = pop_2010[:ending_age] * surv_array
    # In 2011, individuals will have aged one year
    pop_2011 = pop_2011[1:ending_age+1]
    # The immigration rate will be 1 plus the percent change in
    # population (since death has already been accounted for)
    perc_change = ((pop_2011 - pop_2010) / pop_2010)
    # Remove the last entry, since individuals in the last period will die
    # Fit a polynomial to the immigration rates
    # poly_imm = poly.polyfit(np.linspace(0, ending_age-1, ending_age-1), perc_change, deg=10)
    # im_array = poly.polyval(np.linspace(0, ending_age-1, ending_age-1), poly_imm)
    im_array = perc_change
    im_array2 = im_array[starting_age-1:ending_age-1]
    imm_rate_condensed = np.zeros(S)
    # If S < 60, then group years together
    for s in xrange(S):
        imm_rate_condensed[s] = np.product(1+
            im_array2[s*(60/S):(s+1)*(60/S)]) - 1
    im_array2 = np.tile(imm_rate_condensed.reshape(S, 1), (1, J))
    children_im = np.array([im_array[0]] + list(im_array[:starting_age-1]))
    return im_array2, children_im


'''
------------------------------------------------------------------------
    Fertility Rates
------------------------------------------------------------------------
'''


def get_fert(S, J, starting_age):
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
    fert_rate = poly.polyval(np.linspace(starting_age, ending_age-1, 60), poly_fert)
    # Do not allow negative fertility rates, or nonzero rates outside of
    # a certain age range
    new_end = np.linspace(fert_rate[42-starting_age], fert_data[-1], 9)
    fert_rate[42-starting_age:51-starting_age] = new_end
    for i in xrange(60):
        if np.linspace(starting_age, ending_age-1, 60)[i] >= 51 or np.linspace(starting_age, ending_age-1, 60)[i] < 10:
            fert_rate[i] = 0
        if fert_rate[i] < 0:
            fert_rate[i] = 0
    fert_rate_condensed = np.zeros(S)
    # If S < 60, then group years together
    for s in xrange(S):
        fert_rate_condensed[s] = np.prod(1+
            fert_rate[s*(60/S):(s+1)*(60/S)]) - 1
    # plt.scatter(age_midpoint, fert_data)
    # plt.axhline(y=0, color='red')
    # plt.plot(np.linspace(starting_age, ending_age-1, 60), fert_rate)
    # plt.savefig('OUTPUT/fert_dist')
    fert_rate = np.tile(fert_rate_condensed.reshape(S, 1), (1, J))
    # Divide the fertility rate by 2, since it will be used for men and women
    fert_rate /= 2.0
    children_fertrate = poly.polyval(np.linspace(0, starting_age-1, starting_age), poly_fert)
    for i in xrange(starting_age):
        if np.linspace(0, starting_age-1, starting_age)[i] <= 10:
            children_fertrate[i] = 0
        if children_fertrate[i] < 0:
            children_fertrate[i] = 0
    return fert_rate, children_fertrate

'''
------------------------------------------------------------------------
    Generate graphs of mortality, fertility, and immigration rates
------------------------------------------------------------------------
'''


def rate_graphs(S, starting_age, imm, fert):
    domain = np.arange(S) + 1
    mort = np.array(mort_data.mort_rate)[starting_age:100]
    domain2 = np.arange(mort.shape[0]) + 1
    plt.figure(1)
    plt.plot(domain2[:60], mort[:60], color='blue', linewidth=2)
    plt.plot(domain2[60:], mort[60:], color='blue', linestyle='--', linewidth=2)
    plt.axvline(x=60, color='red', linestyle='-', linewidth=1)
    plt.xlabel(r'age $s$')
    plt.ylabel(r'mortality $\rho_s$')
    plt.savefig('OUTPUT/mort_rates')
    plt.figure(2)
    plt.plot(domain, imm, linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'immigration $i_s$')
    plt.savefig('OUTPUT/imm_rates')
    plt.figure(3)
    plt.plot(domain, fert, linewidth=2, color='blue')
    plt.xlabel(r'age $s$')
    plt.ylabel(r'fertility $f_s$')
    plt.savefig('OUTPUT/fert_rates')

'''
------------------------------------------------------------------------
    Generate Omega array
------------------------------------------------------------------------
'''


def get_omega(S, J, T, starting_age):
    '''
    Parameters:
        S - Number of age cohorts
        J - Number of ability types

    Returns:

    '''
    ending_age = starting_age + 60
    data1 = data
    data2 = data1[starting_age:ending_age]
    # Generate list of total population size for 2010, 2011, 2012 and 2013
    sum2010 = data2['2010'].values.sum()
    # For each year of the data, transform each age group's population to
    # be a fraction of the total
    pop_data = np.array(data1['2010'] / float(sum2010))
    poly_pop = poly.polyfit(np.linspace(0, pop_data.shape[0]-1, pop_data.shape[0]), pop_data, deg=11)
    poly_int_pop = poly.polyint(poly_pop)
    pop_int = poly.polyval(np.linspace(starting_age, ending_age, S+1), poly_int_pop)
    new_omega = np.zeros(S)
    for s in xrange(S):
        new_omega[s] = pop_int[s+1] - pop_int[s]
    # Each ability group contains 1/J fraction of age group
    new_omega = np.tile(new_omega.reshape(S, 1), (1, J))
    new_omega /= J
    surv_array, children_rate = get_survival(S, J, starting_age)
    imm_array, children_im = get_immigration(S, J, starting_age)
    omega_big = np.tile(new_omega.reshape(1, S, J), (T, 1, 1))
    fert_rate, children_fertrate = get_fert(S, J, starting_age)
    rate_graphs(S, starting_age, imm_array, fert_rate)
    # children = np.zeros((starting_age, J))
    # Keep track of how many individuals have been born and their survival
    # until they enter the working population
    # children_rate = np.array([1] + list(children_rate))
    # for ind in xrange(starting_age):
    #     children[ind, :] = (
    #         omega_big[0, :, :] * fert_rate).sum(0) * np.prod(
    #         children_rate[:ind] + children_im[:ind])
    # Generate the time path for each age/abilty group
    for t in xrange(1, T):
        # Children are born and then have to wait 20 years to enter the model
        # omega_big[t, 0, :] = children[-1, :] * (children_rate[-1] + imm_array[0])
        # Children are born immediately:
        omega_big[t, 0, :] = (omega_big[t-1, :, :] * fert_rate).sum(0) #* (children_rate[-1] + imm_array[0])
        omega_big[t, 1:, :] = omega_big[t-1, :-1, :] * (surv_array[:-1].reshape(1, S-1, J) + imm_array[1:].reshape(1, S-1, J))
        # children[1:, :] = children[:-1, :] * (children_rate[1:-1].reshape(
        #     starting_age-1, 1) + children_im[1:].reshape(starting_age-1, 1))
        # children[0, :] = ((omega_big[t, :, :] * fert_rate).sum(0) + (children * children_fertrate.reshape(starting_age, 1)).sum(0))* (1 + children_im[0])
    OMEGA = np.zeros((S, S))
    OMEGA[0, :] = fert_rate[:, 0] # * (children_rate[-1] + imm_array[0])
    OMEGA += np.diag(surv_array[:-1, 0] + imm_array[1:, 0], -1)
    eigvalues, eigvectors = np.linalg.eig(OMEGA)
    mask = eigvalues.real != 0
    eigvalues = eigvalues[mask]
    mask2 = eigvalues.imag == 0
    eigvalues = eigvalues[mask2].real
    if eigvalues.shape[0] != 1:
        raise Exception ('There are multiple steady state growth rates.')
    g_n_SS = eigvalues - 1
    eigvectors = np.abs(eigvectors.T)
    eigvectors = eigvectors[mask]
    omega_SS = eigvectors[mask2].real
    omega_SS = np.tile(omega_SS.reshape(S, 1), (1, J)) / J # Divide by J here?
    return omega_big, g_n_SS, omega_SS

# Known problems:
# Fitted polynomial on survival rates creates some entries that are greater than 1
# If children are not born immediately, and S < 60, then they must age 60/S years...
