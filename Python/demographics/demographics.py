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

def get_omega(S, J):
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
    return new_omega

