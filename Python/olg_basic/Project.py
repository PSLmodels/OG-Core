import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Number of age cohorts
S = 60

# Exogenous retirement age
# R = 5

# Parameters of the model
alpha = .35
beta = .96**(60.0/S)
delta = 1 - (.95)**(60.0/S)
gamma = 3.0
xi = 0
eta = 0
g = 0
# n = np.arange(1, S+1)<R
n = np.array([.87,.89,.91,.93,.96,.98]+list(np.ones(34))+[.95,.89,.84,
    .79,.73,.68,.63,.57,.52,.47,.4,.33,.26,.19,.12,.11,.11,.1,.1,.09])
n = n[ (60%S) : : (60/S) ]
phi = 0

# Parameters of estimation 
tol = 0.1
rho = 0.2

# Number of time periods
T = 31

def wage(K, L):

    """
    Parameters: Aggregate capital stock 
                Aggregate labor supply

    Returns:    Wage
    """

    return (1 - alpha) * ( (K/L)**alpha ) * np.exp((1-alpha)*g)

def rate(K, L):

    """
    Parameters: Aggregate capital stock
                Aggregate labor supply

    Returns:    Rental rate of capital
    """

    return alpha * ( np.exp(g) * (L/K) )**(1-alpha)

def MUc(c):

    """
    Parameters: Consumption

    Returns:    Marginal Utility of Consumption
    """

    return c**(-gamma)

def MUl(l):

    """
    Parameters: Labor

    Returns:    Marginal Utility of Labor
    """

    return - eta * l**(-xi)

def Steady_State(K_guess):

    """
    Parameters: Steady state distribution of capital guess as array 
                size S-1

    Returns:    Array of S-1 Euler equation errors
    """

    K = K_guess.sum()
    w = wage(K, n.sum())
    r = rate(K, n.sum())
    
    K1 = np.array([0] + list(K_guess[:-1]))
    K2 = K_guess
    K3 = np.array(list(K_guess[1:]) + [0])

    error = MUc((1 + r - delta)*K1 + w * n[:-1] - K2) \
    - beta * (1 + r - delta) * MUc((1 + r - delta)*K2 + w * n[1:] - K3)
    
    return error

K_guess = np.ones(S-1)*0.3
Kss = opt.fsolve(Steady_State, K_guess)
Kss = np.array([0]+list(Kss))
domain = np.linspace(0,S,S)

plt.plot(domain, Kss, label="Distribution of Capital")
plt.plot(np.arange(S), Kss.mean() * np.ones(S), 'k--', label="Steady State")
plt.title("Distribution of capital across age cohorts")
plt.legend(loc=0)
plt.show()

def EulerEq(K_opt, w_path, r_path, j):

    """
    Parameters: Guess for individual's choice of capital in each period
                of life as array size S-1,
                Array of wages over time,
                Array of rental rates over time,
                Time period the indiviual is born in

    Returns:    Array of S-1 Euler equation errors
    """

    K1 = np.array([0] + list(K_opt[:-1]))
    K2 = K_opt
    K3 = np.array(list(K_opt[1:]) + [0])

    w1 = w_path[j : j+S-1]
    w2 = w_path[j+1 : j+S]

    r1 = r_path[j : j+S-1]
    r2 = r_path[j+1 : j+S]

    n1 = n[:-1]
    n2 = n[1:]

    error = MUc((1 + r1 - delta)*K1 + w1 * n1 - K2) \
    - beta * (1 + r2 - delta) * MUc((1 + r2 - delta)*K2 + w2 * n2 - K3)
    return error

def UpperTriangle(K_opt, init, w_path, r_path, i):

    error = np.zeros(len(K_opt))

    K1 = np.array([init[i]] + list(K_opt[:-1]))
    K2 = K_opt
    K3 = np.array(list(K_opt[1:]) + [0])

    w1 = w_path[:len(K_opt)]
    w2 = w_path[1:len(K_opt) + 1]

    r1 = r_path[:len(K_opt)]
    r2 = r_path[1:len(K_opt) + 1]

    n1 = n[-(len(K_opt)+1):-1]
    n2 = n[-(len(K_opt)):]

    error = MUc((1 + r1 - delta)*K1 + w1 * n1 - K2) \
    - beta * (1 + r2 - delta) * MUc((1 + r2 - delta)*K2 + w2 * n2 - K3)

    return error

initial = (np.random.rand(S)+.5) * Kss
K_path = np.linspace(initial.sum(), Kss.sum(), T+(S-1))
w_path = wage(K_path, n.sum())
r_path = rate(K_path, n.sum())

# sup_norm = 1
# max_iter = 100
# count = 0
# while sup_norm >= tol and count < max_iter:
#     K_mat = np.zeros((T+(S-1), S))
#     K_mat[0,1:] = initial
#     # for i in xrange(S-2):
#     #     K_mat[1:S-(i+1) , 2+i:] += np.diag(opt.fsolve(UpperTriangle, \
#     #         np.ones(S-2-i)*.01*S, args=(initial, w_path, r_path, i)))
#     for j in xrange(T):
#         K_opt = np.ones(S-1)*0.1
#         K_opt = opt.fsolve(EulerEq, K_opt, args=(w_path, r_path, j), xtol=1e-10)
#         K_opt = np.diag(K_opt)
#         K_mat[j:j+S-1, 1:] += K_opt
#     K = K_mat.sum(axis=1)
#     sup_norm = np.max(K_path-K)
#     K_path = rho*K + (1-rho)*K_path
#     w_path = wage(K_path, n.sum())
#     r_path = rate(K_path, n.sum())
#     count += 1

# plt.plot(np.arange(T), K_path[:-(S-1)], label='Capital Time Path')
# plt.plot(np.arange(T), np.ones(T)*Kss.sum(), 'k--', label='Steady State')
# plt.title("Time path of aggregate capital stock")
# plt.legend(loc=0)
# plt.show()


