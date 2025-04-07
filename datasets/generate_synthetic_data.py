import numpy as np
from scipy.stats import norm

# Create a 100x100 standard normal matrix
gauss_100 = norm(0, 1).rvs(size=(100, 100))
np.save('./datasets/gauss_100', gauss_100)

# Create a 1000x1000 standard normal matrix
gauss_1000 = norm(0, 1).rvs(size=(1000, 1000))