import numpy as np
import matplotlib.pyplot as plt
import os

directory = os.path.dirname(__file__)


# Optimal values and weights obtained by Wolfram Alpha. Formulas taken from Pflug and Pohl (2017)
penalization_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
o_weights = [1, 1, 1, 1, 1, 15/19, 10/19, 45/133, 15/76, 5/57, 0]
o_values = [2/3, 142/225, 134/225, 14/25, 118/225, 28/57, 533/1140, 898/1995, 1991/4560, 364/855, 5/12]

n1_values = -np.loadtxt(directory + '/ValuesProduct.txt')
n2_values = -np.loadtxt(directory + '/Values1Correlated.txt')
n1_weights = np.loadtxt(directory + '/WeightsProduct.txt')
n2_weights = np.loadtxt(directory + '/Weights1Correlated.txt')

plt.plot(penalization_values, o_weights, 'o', markersize=6, markeredgewidth=1,
         markeredgecolor='k', markerfacecolor='None', label='Analytic solution')
plt.plot(penalization_values, n2_weights, 'gx', label='Correlated reference measure')
plt.plot(penalization_values, n1_weights, 'r^', markersize=6, markeredgewidth=1,
         markeredgecolor='r', markerfacecolor='None', label='Product reference measure')
plt.title('weight of the second asset depending on risk aversion')
plt.ylabel('weight of second asset $x$')
plt.xlabel('risk aversion $\lambda$')
plt.legend()
plt.savefig(directory + '/WeightsHQ.jpg', format='jpg', dpi=600)
plt.show()

plt.plot(penalization_values, o_values, 'o', markersize=6, markeredgewidth=1,
         markeredgecolor='k', markerfacecolor='None', label='Analytic solution')
plt.plot(penalization_values, n2_values, 'gx', label='Correlated reference measure')
plt.plot(penalization_values, n1_values, 'r^', markersize=6, markeredgewidth=1,
         markeredgecolor='r', markerfacecolor='None', label='Product reference measure')
plt.title('optimal value depending on risk aversion')
plt.ylabel('optimal value')
plt.xlabel('risk aversion $\lambda$')
plt.legend()
plt.savefig(directory + '/ValuesHQ.jpg', format='jpg', dpi=600)
plt.show()
