import numpy as np
import matplotlib.pyplot as plt

def binary_sigmoidal(x):
  return 1 / (1 + np.exp(-x))

def bipolar_sigmoidal(x):
  return 2 / (1 + np.exp(-2 * x)) - 1

# Visualize the results
x = np.arange(-10, 10, 0.1)

binary_sigmoidal_values = binary_sigmoidal(x)
bipolar_sigmoidal_values = bipolar_sigmoidal(x)

plt.plot(x, binary_sigmoidal_values, label='Binary sigmoidal')
plt.plot(x, bipolar_sigmoidal_values, label='Bipolar sigmoidal')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
