#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os

directory = os.path.dirname(__file__)
value_list = []
x_value_list = []
labels = []
percentile_deviations = np.zeros([2, 9])
for i in range(9):
    gamma_value = 10 * (2 ** (i+2))
    values = np.loadtxt(directory+'/Gamma_'+str(gamma_value)+'.txt')
    val = np.mean(values)
    value_list.append(val)
    percentile_deviations[0, i] = val - np.percentile(values, 2.5)
    percentile_deviations[1, i] = - val + np.percentile(values, 97.5)
    x_value_list.append(i)
    labels.append(gamma_value)


plt.figure(figsize=(8, 6))
plt.plot(x_value_list, [-1]*9)
(_, caps, _) = plt.errorbar(x_value_list, value_list, yerr=percentile_deviations, fmt='.', capthick=1, capsize=2, elinewidth=1)
plt.xticks(x_value_list, labels)
plt.title('Mean and standard deviation of optimal values depending on $\gamma$')
for cap in caps:
    cap.set_color('red')
    cap.set_markeredgewidth(1)
plt.ylabel('$\hat{\phi}_{\mu,\gamma}^{m}(f)$')
plt.xlabel('$\gamma$')
plt.savefig(directory+'/Figure2.jpg', format='jpg', dpi=700)
plt.show()
