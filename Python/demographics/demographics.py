import numpy as np
import pandas as pd

data = pd.read_table('data\demographic_data.csv', sep=',')
data = data[(15 < data.Age) & (data.Age < 76)]
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
mort_data = mort_data[(mort_data.age < 76) & (mort_data.age >= 15)]


def get_survival(S, J):
    survival_rate = np.array(mort_data.surv_rate)
    surv_rate_condensed = np.zeros(S)
    for s in xrange(S):
        surv_rate_condensed[s] = np.product(survival_rate[s*(60/S):(s+1)*(60/S)])
    surv_rate_condensed[-1] = 0
    surv_array = np.tile(surv_rate_condensed.reshape(S, 1), (1, J))
    return surv_array


def get_omega(S, J, T):
    age_groups = np.linspace(16, 76, S+1)
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
    for t in xrange(1, T):
        omega_big[t, 1:, :] = omega_big[t-1, :-1, :] * surv_array[:-1].reshape(1, S-1, J)
    return omega_big


