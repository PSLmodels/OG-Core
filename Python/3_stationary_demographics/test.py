import matplotlib.pyplot as plt
import numpy as np
import demographics

S = 80
J = 7
T = S
starting_age = 20
bin_weights = np.array([1.0/J] * J)

omega, g_n, omega_SS, children, surv_rate = demographics.get_omega(S, J, T, bin_weights, starting_age)

for i in xrange(T):
	# plt.figure()
    plt.plot(np.arange(S+int(starting_age * S / 80.0))+1, list(children[i, :, 0]) + list(omega[i, :, 0]), linewidth=2, color='blue')
    plt.show()

