from scipy.stats import entropy
import numpy as np

a = np.array([0.1, 0.0, 0.3, 0.6])     # real value
b = np.array([0.1, 0.0, 0.4, 0.5])

cross_entropy = entropy(a, b)
print(cross_entropy)
