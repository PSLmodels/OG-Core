import matplotlib.pyplot as plt
import numpy as np
import demographics

S = 60
J = 7
T = 3*S
starting_age = 20

omega, g_n, omega_SS, children, surv_rate = demographics.get_omega(S, J, T, starting_age)

for i in xrange(150):
	# plt.figure()
    plt.plot(np.arange(S+int(starting_age * S / 60.0))+1, list(children[i, :, 0]) + list(omega[i, :, 0]), linewidth=2, color='blue')
    plt.show()

