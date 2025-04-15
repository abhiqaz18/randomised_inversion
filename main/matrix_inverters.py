import numpy as np
from scipy.stats import norm
std_norm = norm(0, 1)

# TO-DO
# 1. Define convergence criterion, add max_iters, make iterations terminate automatically.
# 2. Allow arbitrary m to be set (as frac).
# 3. Allow arbitrary distribution to be set.
# 4. Consider preloading sampled matrices. 

def Gower_Richtarik_2016_1(A: np.matrix) -> np.matrix:
    """
    Algorithm 1 from Gower and Richtárik (2016).
    Called Stochastic Iterative Matrix Inversion (SIMI) – nonsymmetric row variant.
    Parameters: distribution D and positive definite matrix W with the same dimensions as A.
    """
    n = A.shape[0]
    I = np.matrix(np.eye(n))
    m = n // 4
    num_iters = 200

    A_inv = np.matrix(std_norm.rvs(size=(n, n)))
    for _ in range(num_iters):
        S = np.matrix(std_norm.rvs(size=(n, m)))
        L = S * np.matrix(np.linalg.inv(S.T * A * A.T * S)) * S.T
        A_inv += A.T * L * (I - A * A_inv)

    return A_inv

def Gower_Richtarik_2016_3(A: np.matrix) -> np.matrix:
    """
    Algorithm 1 from Gower and Richtárik (2016).
    Called Stochastic Iterative Matrix Inversion (SIMI) - symmetric variant.
    Parameters: distribution D and positive definite matrix W with the same dimensions as A.
    """
    assert A == A.T, 'Please ensure input matrix is symmetric.' # Does this play well with floats?
    n = A.shape[0]
    I = np.matrix(np.eye())
    m = n // 4
    num_iters = 200

    A_inv = np.matrix(std_norm.rvs(size=(n, n)))
    A_inv = (A_inv + A_inv.T) / 2
    for _ in range(num_iters):
        S = 

    return 0