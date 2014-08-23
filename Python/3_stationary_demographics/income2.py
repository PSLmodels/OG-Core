import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly

jan_dat = pd.read_table("data/e_vec_data/jan2014.asc", sep=',', header=0)
jan_dat['wgt'] = jan_dat['PWCMPWGT']
jan_dat['age'], jan_dat['wage'] = jan_dat['PRTAGE'], jan_dat['PTERNHLY']
del jan_dat['HRHHID'], jan_dat['OCCURNUM'], jan_dat['YYYYMM'], jan_dat[
    'HRHHID2'], jan_dat['PRTAGE'], jan_dat['PTERNHLY'], jan_dat['PWCMPWGT']

feb_dat = pd.read_table("data/e_vec_data/feb2014.asc", sep=',', header=0)
feb_dat['wgt'] = feb_dat['PWCMPWGT']
feb_dat['age'], feb_dat['wage'] = feb_dat['PRTAGE'], feb_dat['PTERNHLY']
del feb_dat['HRHHID'], feb_dat['OCCURNUM'], feb_dat['YYYYMM'], feb_dat[
    'HRHHID2'], feb_dat['PRTAGE'], feb_dat['PTERNHLY'], feb_dat['PWCMPWGT']

mar_dat = pd.read_table("data/e_vec_data/mar2014.asc", sep=',', header=0)
mar_dat['wgt'] = mar_dat['PWCMPWGT']
mar_dat['age'], mar_dat['wage'] = mar_dat['PRTAGE'], mar_dat['PTERNHLY']
del mar_dat['HRHHID'], mar_dat['OCCURNUM'], mar_dat['YYYYMM'], mar_dat[
    'HRHHID2'], mar_dat['PRTAGE'], mar_dat['PTERNHLY'], mar_dat['PWCMPWGT']

apr_dat = pd.read_table("data/e_vec_data/apr2014.asc", sep=',', header=0)
apr_dat['wgt'] = apr_dat['PWCMPWGT']
apr_dat['age'], apr_dat['wage'] = apr_dat['PRTAGE'], apr_dat['PTERNHLY']
del apr_dat['HRHHID'], apr_dat['OCCURNUM'], apr_dat['YYYYMM'], apr_dat[
    'HRHHID2'], apr_dat['PRTAGE'], apr_dat['PTERNHLY'], apr_dat['PWCMPWGT']

may_dat = pd.read_table("data/e_vec_data/may2014.asc", sep=',', header=0)
may_dat['age'], may_dat['wage'] = may_dat['PRTAGE'], may_dat['PTERNHLY']
may_dat['wgt'] = may_dat['PWCMPWGT']
del may_dat['HRHHID'], may_dat['OCCURNUM'], may_dat['YYYYMM'], may_dat[
    'HRHHID2'], may_dat['PRTAGE'], may_dat['PTERNHLY'], may_dat['PWCMPWGT']

def get_e_indiv(S, J, data, bin_weights, starting_age):
    temp_ending_age = starting_age + 50
    age_groups = np.linspace(starting_age, temp_ending_age, 51)
    e = np.zeros((S, J))
    data = data[(starting_age <= data.age) & (data.age <= temp_ending_age)]
    for i in xrange(50):
        incomes = data[(age_groups[i] <= data.age) & (data.age < age_groups[i+1])]
        incomes = incomes.sort(['wage'])
        inc = np.array(incomes.wage)
        wgt_ar = np.array(incomes.wgt)
        wgt_cum = np.zeros(inc.shape[0])
        cum_weight_scalar = 0
        for k in xrange(inc.shape[0]):
            cum_weight_scalar += wgt_ar[k]
            wgt_cum[k] = cum_weight_scalar
        total_wgts = wgt_cum[-1]
        percentile = 0
        indicies  = np.zeros(J+1)
        for j, weight in enumerate(bin_weights):
            percentile += weight
            ind = 0
            while wgt_cum[ind] < total_wgts * percentile:
                ind += 1
            indicies[j+1] = ind
        for j in xrange(J):
            e[i, j] = np.mean(inc[indicies[j]:indicies[j+1]])
    e /= e.mean()
    polynomials = []
    ints = []
    for j in xrange(J):
        polynomials.append(poly.polyfit(np.arange(50), e[:50,j], deg=2))
        ints.append(poly.poly)
    
    return e

def get_e(S, J, starting_age, bin_weights):
    e = np.zeros((S, J))
    e += get_e_indiv(S, J, jan_dat, starting_age, bin_weights)
    e += get_e_indiv(S, J, feb_dat, starting_age, bin_weights)
    e += get_e_indiv(S, J, mar_dat, starting_age, bin_weights)
    e += get_e_indiv(S, J, apr_dat, starting_age, bin_weights)
    e += get_e_indiv(S, J, may_dat, starting_age, bin_weights)
    e /= 5
    return e
get_e_indiv(60,7,may_dat,[1.0/7]*7, 20)
