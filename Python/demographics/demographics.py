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
children_rate = np.array(mort_data[(mort_data.age < 15)].surv_rate)
# mort_data = mort_data[(mort_data.age < 76) & (mort_data.age >= 15)]


def get_survival(S, J):
    poly_surv = np.poly1d(np.polyfit(mort_data.age, mort_data.surv_rate, 10))
    survival_rate = poly_surv(np.linspace(15, 75, S))
    surv_rate_condensed = np.zeros(S)
    for s in xrange(S):
        surv_rate_condensed[s] = np.product(survival_rate[s*(60/S):(s+1)*(60/S)])
    surv_rate_condensed[-1] = 0
    surv_array = np.tile(surv_rate_condensed.reshape(S, 1), (1, J))
    return surv_array

# fert_data = pd.read_table('data\\fertility_data.asc', sep=',')
# del fert_data['HRHHID'], fert_data['HRHHID2'], fert_data['OCCURNUM'], fert_data['YYYYMM']
# fert_data = fert_data[(15 <= fert_data.PRTAGE) & (fert_data.PRTAGE <= 44)]
# group = fert_data.groupby('PRTAGE')
# a = np.array(group.mean()).flatten()
# fert_data2 = pd.read_table('data\\fertility_data2.asc', sep=',')
# del fert_data2['HRHHID'], fert_data2['HRHHID2'], fert_data2['OCCURNUM'], fert_data2['YYYYMM']
# fert_data2 = fert_data2[(15 <= fert_data2.PRTAGE) & (fert_data.PRTAGE <= 44)]
# group2 = fert_data2.groupby('PRTAGE')
# b = np.array(group2.mean()).flatten()
# c = (a+b)/2
# print np.diff(c)
# domain = np.linspace(15, 44, 29)
# plt.plot(domain, np.diff(c))
# plt.savefig('plot')



# 15-19, ...,45-49
# fert_data = np.array([43.7, 104.0, 114.5, 93.5, 92.8, 8.5, .5]) / 1000


# from http://www.cdc.gov/nchs/data/nvsr/nvsr63/nvsr63_02.pdf
fert_data = np.array([.3, 26.6, 12.3, 47.3, 81.2, 106.2, 98.7, 49.6, 10.5, .8])
age_midpoint = np.array([12, 17, 16, 18.5, 22, 27, 32, 37, 42, 49.5])
# 12.5 - 17.5
ln1 = np.linspace(0, fert_data[0], 11)
# # 17.5 - 22.5
ln2 = np.linspace(fert_data[0], fert_data[1], 11)
# # 22.5 - 27.5
ln3 = np.linspace(fert_data[1], fert_data[2], 11)
# # 27.5 - 32.5
ln4 = np.linspace(fert_data[2], fert_data[3], 11)
# # 32.5 - 37.5
ln5 = np.linspace(fert_data[3], fert_data[4], 11)
# # 37.5 - 42.5
ln6 = np.linspace(fert_data[4], fert_data[5], 11)
# # 42.5 - 47.5
ln7 = np.linspace(fert_data[5], fert_data[6], 11)
# # 47.5 - 52.5
ln8 = np.linspace(fert_data[6], 0, 11)

def get_fert(S, J):
    # everything = np.array(list(ln1) + list(ln2[1:]) + list(ln3[1:]) + list(ln4[1:]) + list(ln5[1:]) + list(ln6[1:]) + list(ln7[1:]) + list(ln8[1:]))
    # sixteen_to_52 = everything[7::2]
    # fert_rate = np.array(list(sixteen_to_52) + list(np.zeros(23)))
    # polynomial fit of fertility rates
    poly_fert = np.poly1d(np.polyfit(age_midpoint, fert_data, 4))
    fert_rate = poly_fert(np.linspace(15, 75, S))
    for i in xrange(S):
        if np.linspace(15,75, S)[i] > 52 or np.linspace(15,75, S)[i] < 10:
            fert_rate[i] = 0
        if fert_rate[i] < 0:
            fert_rate[i] = 0
    fert_rate = fert_rate[ 60%S : : 60/S ]
    fert_rate = np.tile(fert_rate.reshape(S,1), (1,J))
    children = np.zeros((15, J))
    return fert_rate/2.0, children

ran1, els = get_fert(60, 1)
plt.plot(np.linspace(15, 75, 60), ran1)
plt.scatter(age_midpoint, fert_data)
plt.savefig('zzzsomething')


# # x = np.arange(35) + 15
# # y = np.array([43.7]*5 + [104.0]*5 + [114.5]*5 + [93.5]*5 + [42.8]*5 + [8.5]*5 + [0.5]*5)
# x = np.array([17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 27.5])
# y = np.array([43.7, 104.0, 114.5, 93.5, 42.8, 8.5, 0.5]) / 2000.0
# b = y.var()/y.mean()
# a = y.mean()/b
# params = [a, b]
# # params = [3.0, 1.0]

# def GB2(x, params):
#     a, b, p, q = params[0], params[1], params[2], params[3]
#     return a*x**(a*p-1) / ( b**(a*p) * bt(p,q) * (1+(x/b)**a)**(p+q) )

# def GA(x, params):
#     a, b = params.flatten()
#     part1 = 1.0/((b**a) * gamma(a))
#     part2 = (x**(a-1)) * (np.exp(-x/b))
#     return part1 * part2

# def GMM(params, x, y):
#     size = len(y)
#     e = (GA(x, params) - y)/y
#     W = np.identity(size)
#     return np.linalg.norm(e)**2
#     # return (e.reshape(1,size)).dot(W.dot(e.reshape(size,1)))

# def get_fertility(S, J):
#     ages = np.linspace(16+(S/120.0), 76-(S/120.0), S)
#     answer = opt.minimize(GMM, params, args=(x,y), method='Nelder-Mead', bounds=((0,None),(0,None)))
#     opt_params = answer.x
#     print answer.success
#     print opt_params
#     plt.plot(np.arange(50), GA(np.arange(50), opt_params))
#     plt.plot(x, y)
#     plt.show()
#     fert_array = GA(ages, opt_params)
#     return ages, fert_array

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
    fert_rate, children = get_fert(S, J)
    for ind in xrange(15):
        children[ind, :] = (omega_big[0,:,:] * fert_rate).sum(0) * np.prod(children_rate[:ind])
    for t in xrange(1, T):
        # omega_big[t,0,:] = children[-1,:] * children_rate[-1]
        omega_big[t, 1:, :] = omega_big[t-1, :-1, :] * surv_array[:-1].reshape(1, S-1, J)
        children[1:,:] = children[:-1,:] * children_rate[:-1].reshape(14,1)
        children[0,:] = (omega_big[t,:,:] * fert_rate).sum(0)
    return omega_big


x = get_omega(60,7,70).sum(1).sum(1)
# print np.diff(x)/x[:-1]
plt.axhline(y=x[-1], color='r', linewidth=2)
plt.plot(np.arange(70), x, 'b', linewidth=2)
plt.title('Population Size (as a percent of the 2010 population)')
plt.savefig('OUTPUT\Population')


