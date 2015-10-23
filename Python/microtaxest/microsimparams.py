import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import pandas as pd
import scipy as sp

A = pd.read_csv('2015_tau_n.csv')
A['Wage + Self-Employed Income'] = A['Wage and Salaries'] + A['Self-Employed Income']
A["Effective Tax Rate"] = A['Total Tax Liability']/A["Adjusted Total income"]
A["Total Capital Income"] = A['Adjusted Total income'] - A['Wage + Self-Employed Income']

#Weights
w = np.array(A['Weights'].values)
w = np.reshape(w,(len(w), 1))
W = sp.sparse.spdiags(w.T,0, len(w), len(w)).toarray()
#Labor Income
x = np.array(A['Wage + Self-Employed Income'].values) 
x = np.reshape(x,(len(x), 1))
#Capital Income
y = np.array(A['Total Capital Income'].values) 
y = np.reshape(y,(len(y), 1))
effective_tax_rate = np.array(A['Effective Tax Rate'].values)
effective_tax_rate = np.reshape(effective_tax_rate, (len(effective_tax_rate),1))

x_cube_bar = np.average(x**3)
x_cube = (x**3 - x_cube_bar)/x_cube_bar 

y_cube_bar = np.average(x**3)
y_cube = (y**3 - y_cube_bar)/y_cube_bar 

x2y_bar = np.average(x**2*y)
x2y = (x**2*y - x2y_bar)/x2y_bar

xy2_bar = np.average(x*y**2)
xy2 = (x*y**2 - xy2_bar)/xy2_bar

xy_bar = np.average(x*y)
xy = (x*y - xy_bar)/xy_bar

x_bar = np.average(x)
x = (x - x_bar)/x_bar

y_bar = np.average(y)
y = (y - y_bar)/y_bar

X = np.concatenate((x**3, y**3, x**2*y, x*y**2, x*y, x, y, np.reshape(np.ones(len(y)),(len(y),1))), axis = 1)
'''
print np.dot( np.linalg.inv(  np.dot( np.dot(X.T, W)),X),np.dot(np.dot(X.T,W), effective_tax_rate))
'''






'''
efftax = np.array((0.00028147,0.001697761,0.004968297,0.009946499,0.015911159,0.022463427,0.032044766,0.043890195,0.059313549,0.071657599,0.1008071,0.16372908,0.208437155,0.2170106,0.217145338,0.216026002,0.208568983,0.17475609))

y_l = np.array((11252.08343,19606.04969,26842.65344,33063.45169,38735.33288,44341.33709,52510.34395,63745.31317,83573.01535,112967.2962,168257.2995,340507.8414,776921.8912,1365583.314,1942389.306,3342054.315,7634947.804,34989584.11))

y_k = np.array((11252.08343,19606.04969,26842.65344,33063.45169,38735.33288,44341.33709,52510.34395,63745.31317,83573.01535,112967.2962,168257.2995,340507.8414,776921.8912,1365583.314,1942389.306,3342054.315,7634947.804,34989584.11))
y_l /= (y_l/2)
y_k /= (y_k/3)

def wealthcurve(params, y_l, y_k, tax):
    A,B,C,D,E,F, minimum, maximum = params
    ##RIGHT EQUATION?
    return sum(((maximum - minimum)*((A*y_l**2 + B*y_k**2 + C*y_l*y_k + D*y_l + E*y_k)/(A*y_l**2 + B*y_k**2 + C*y_l*y_k + D*y_l + E*y_k + F) - minimum)) - tax)

A,B,C,D,E,F, minimum, maximum = .0001,1.7, 60000,.22,.1,.1,.1,.5
params = A,B,C,D,E,F, minimum, maximum 
(A,B,C,D,E,F,minimum,maximum) = optimize.minimize(wealthcurve, params, args = (y_l[:16], y_k[:16], efftax[:16]), method = 'Nelder-Mead', tol = 10**-15).x
print (A,B,C,D,E,F,minimum,maximum)
values = (maximum - minimum)*((A*y_l**2 + B*y_k**2 + C*y_l*y_k + D*y_l + E*y_k)/(A*y_l**2 + B*y_k**2 + C*y_l*y_k + D*y_l + E*y_k + F))
plt.figure() 
x = np.linspace(1,18,18)
plt.plot(x,efftax*100) 
plt.plot(x,values*100) 
labels = [11252.08343,19606.04969,26842.65344,33063.45169,38735.33288,44341.33709,52510.34395,63745.31317,83573.01535,112967.2962,168257.2995,340507.8414,776921.8912,1365583.314,1942389.306,3342054.315,7634947.804,34989584.11]
plt.xticks(x, labels, rotation = 'vertical')
plt.xlabel('wealth')
plt.ylabel('average tax rate')
plt.show()
'''

