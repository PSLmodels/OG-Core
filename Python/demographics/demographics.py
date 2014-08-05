import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import beta, gamma
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

data = pd.read_table('data\demographic_data.csv', sep=',')
data = data.set_index('Age')
for index, value in enumerate(data['2010']):
    data['2010'][index+16] = int(value.replace(',', ''))
for index, value in enumerate(data['2011']):
    data['2011'][index+16] = int(value.replace(',', ''))
for index, value in enumerate(data['2012']):
    data['2012'][index+16] = int(value.replace(',', ''))
for index, value in enumerate(data['2013']):
    data['2013'][index+16] = int(value.replace(',', ''))

mort_data = pd.read_table('data\mortality_rates.csv', sep=',')
mort_data['mort_rate'] = (mort_data.prob_live_next_male + mort_data.prob_live_next_female) / 2
del mort_data['prob_live_next_female'], mort_data['prob_live_next_male']
mort_data['surv_rate'] = 1 - mort_data.mort_rate
children_rate = np.array(mort_data[(mort_data.age < 15)].surv_rate)
# mort_data = mort_data[(mort_data.age < 76) & (mort_data.age >= 15)]


def get_survival(S, J):
    poly_surv = poly.polyfit(mort_data.age, mort_data.surv_rate, deg=10)
    survival_rate = poly.polyval(np.linspace(15, 75, S), poly_surv)
    surv_rate_condensed = np.zeros(S)
    for s in xrange(S):
        surv_rate_condensed[s] = np.product(survival_rate[s*(60/S):(s+1)*(60/S)])
    surv_rate_condensed[-1] = 0
    surv_array = np.tile(surv_rate_condensed.reshape(S, 1), (1, J))
    return surv_array

# from http://www.cdc.gov/nchs/data/nvsr/nvsr63/nvsr63_02.pdf
fert_data = np.array([.3, 26.6, 12.3, 47.3, 81.2, 106.2, 98.7, 49.6, 10.5, .8]) / 1000
age_midpoint = np.array([12, 17, 16, 18.5, 22, 27, 32, 37, 42, 49.5])


def get_fert(S, J):
    # polynomial fit of fertility rates
    poly_fert = poly.polyfit(age_midpoint, fert_data, deg=4)
    fert_rate = poly.polyval(np.linspace(15, 75, S), poly_fert)
    for i in xrange(S):
        if np.linspace(15,75, S)[i] > 50 or np.linspace(15,75, S)[i] < 10:
            fert_rate[i] = 0
        if fert_rate[i] < 0:
            fert_rate[i] = 0
    fert_rate = fert_rate[ 60%S : : 60/S ]
    fert_rate = np.tile(fert_rate.reshape(S,1), (1,J))
    children = np.zeros((15, J))
    return fert_rate/2.0, children


def get_omega(S, J, T):
    age_groups = np.linspace(16, 76, S+1)
    data = data[(15 < data.Age) & (data.Age < 76)]
    sums = [sum2010, sum2011, sum2012, sum2013] = [data['2010'].values.sum(), data['2011'].values.sum(), data['2012'].values.sum(), data['2013'].values.sum()]
    data['2010'] /= float(sum2010)
    data['2011'] /= float(sum2011)
    data['2012'] /= float(sum2012)
    data['2013'] /= float(sum2013)
    omega = data['2010'].values
    new_omega = np.zeros(S)
    for ind in xrange(S):
        new_omega[ind] = (data[np.array((age_groups[ind] <= data.index)) & np.array((data.index < age_groups[ind+1]))])['2010'].values.sum()
    new_omega = np.tile(new_omega.reshape(S,1), (1,J))
    new_omega /= J
    surv_array = get_survival(S, J)
    omega_big = np.tile(new_omega.reshape(1, S, J), (T, 1, 1))
    fert_rate, children = get_fert(S, J)
    for ind in xrange(15):
        children[ind, :] = (omega_big[0,:,:] * fert_rate).sum(0) * np.prod(children_rate[:ind])
    for t in xrange(1, T):
        # omega_big[t,0,:] = children[-1,:] * children_rate[-1]
        omega_big[t, 1:, :] = omega_big[t-1, :-1, :] * surv_array[:-1].reshape(1, S-1, J)
        children[1:,:] = children[:-1,:] * children_rate[:-1].reshape(14,1)
        children[0,:] = (omega_big[t,:,:] * fert_rate).sum(0)
    return omega_big
