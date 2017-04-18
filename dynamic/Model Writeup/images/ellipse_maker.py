# Courtesy of Evan and Isaac
import matplotlib.pyplot as plt
import numpy as np
point_num = 1000
domain = np.linspace(0, 2, point_num)
y_top = 2 * np.sqrt(1-(domain-1)**2) - 1
y_bottom = -2 * np.sqrt(1-(domain-1)**2) - 1
plt.figure()
plt.plot(domain, y_top, color='blue', linewidth=2)
plt.fill_between(domain, 0, y_top, where=y_top>0, facecolor='white', hatch='//')
plt.plot(domain, y_bottom, color='blue', linewidth=2)
plt.scatter(1, -1, color='black')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlim([-2, 4])
plt.ylim([-4, 2])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.hlines(y=-1, xmin=0, xmax=2, color='black', linestyle='--')
plt.vlines(x=1, ymin=-3, ymax=1, color='black', linestyle='--')
plt.savefig('EllipseGen.pdf')
# plt.show()