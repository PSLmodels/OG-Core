import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.special import beta, gamma
import matplotlib.pyplot as plt

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

# x = np.arange(35) + 15
# y = np.array([43.7]*5 + [104.0]*5 + [114.5]*5 + [93.5]*5 + [42.8]*5 + [8.5]*5 + [0.5]*5)
x = np.array([17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 27.5])
y = np.array([43.7, 104.0, 114.5, 93.5, 42.8, 8.5, 0.5]) / 2000.0
b = y.var()/y.mean()
a = y.mean()/b
params = [a, b]
# params = [3.0, 1.0]

def GB2(x, params):
    a, b, p, q = params[0], params[1], params[2], params[3]
    return a*x**(a*p-1) / ( b**(a*p) * bt(p,q) * (1+(x/b)**a)**(p+q) )

def GA(x, params):
    a, b = params.flatten()
    part1 = 1.0/((b**a) * gamma(a))
    part2 = (x**(a-1)) * (np.exp(-x/b))
    return part1 * part2

def GMM(params, x, y):
    size = len(y)
    e = (GA(x, params) - y)/y
    W = np.identity(size)
    return np.linalg.norm(e)**2
    # return (e.reshape(1,size)).dot(W.dot(e.reshape(size,1)))

def get_fertility(S, J):
    ages = np.linspace(16+(S/120.0), 76-(S/120.0), S)
    answer = opt.minimize(GMM, params, args=(x,y), method='Nelder-Mead', bounds=((0,None),(0,None)))
    opt_params = answer.x
    print answer.success
    print opt_params
    plt.plot(np.arange(50), GA(np.arange(50), opt_params))
    plt.plot(x, y)
    plt.show()
    fert_array = GA(ages, opt_params)
    return ages, fert_array

# get_fertility(60,7)

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


