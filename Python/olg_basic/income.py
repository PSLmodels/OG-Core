import numpy as np
import pandas as pd

data = pd.read_table("data/income_data.asc", sep=',', header=0)
data = data.query("19 < PRTAGE < 80")
data['age'], data['wage'] = data['PRTAGE'], data['PTERNH1O']
del data['HRHHID'], data['OCCURNUM'], data['YYYYMM'], data['HRHHID2'], data['PRTAGE'], data['PTERNH1O']

def get_e(S, J):
    age_groups = np.linspace(20,80,S+1)
    e = np.zeros((S,J))
    for i in xrange(S):
        incomes = data.query('age_groups[i]<=age<age_groups[i+1]')
        inc = np.array(incomes['wage'])
        inc.sort()
        for j in xrange(J):
            e[i,j] = inc[len(inc)*(j+.5)/J]
    e /= e.mean()
    return e

def get_f(S, J):
    f = np.ones((S,J))*(1.0/J)
    return f


