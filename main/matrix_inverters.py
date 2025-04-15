import numpy as np
from scipy.stats import norm

std_norm = norm(0, 1)

# TO-DO
# 1. Allow arbitrary distribution to be set.
# 2. Consider preloading sampled matrices.

def Gower_Richtarik_2016_1(A: np.matrix, max_iters=10000, tol=1e-2, sketch_frac=None) -> np.matrix:
    """
    Algorithm 1 from Gower and Richtárik (2016).
    Called Stochastic Iterative Matrix Inversion (SIMI) – nonsymmetric row variant.
    Parameters: distribution D and positive definite matrix W with the same dimensions as A.
    """
    n = A.shape[0]
    m = int(n ** 0.5) if sketch_frac == None else int(n * sketch_frac)
    tol *= n

    W = np.matrix(np.eye(n))
    I = np.matrix(np.eye(n))

    A_inv = np.matrix(std_norm.rvs(size=(n, n)))
    for num_iters in range(max_iters):
        S = np.matrix(std_norm.rvs(size=(n, m)))
        L = S * np.linalg.inv(S.T * A * W * A.T * S) * S.T
        M = I - A * A_inv
        A_inv += W * A.T * L * M

        if np.linalg.matrix_norm(M) < tol: # is this really independent of n?
            break
    
    if num_iters == max_iters - 1:
        print(f'Warning: Max. iterations ({max_iters}) reached without convergence.')

    return A_inv

def Gower_Richtarik_2016_3(A: np.matrix, max_iters=10000, tol=1e-2, sketch_frac=None) -> np.matrix:
    """
    Algorithm 3 from Gower and Richtárik (2016).
    Called Stochastic Iterative Matrix Inversion (SIMI) - symmetric variant.
    Parameters: distribution D and positive definite matrix W with the same dimensions as A.
    """
    assert A == A.T, 'Please ensure input matrix is symmetric.' # Does this play well with floats?

    n = A.shape[0]
    m = int(n ** 0.5) if sketch_frac == None else int(n * sketch_frac)
    tol *= n

    W = np.matrix(np.eye(n))
    I = np.matrix(np.eye(n))

    A_inv = np.matrix(std_norm.rvs(size=(n, n)))
    A_inv = (A_inv + A_inv.T) / 2
    for num_iters in range(num_iters):
        S = np.matrix(std_norm.rvs(size=(n, m)))
        L = S * np.linalg.inv(S.T * A * W * A.T * S) * S.T
        T = L * A * W
        M = A_inv * A - I
        A_inv += T.T * (A * A_inv * A - A) * T - M * T - (M * T).T

        if np.linalg.matrix_norm(M) < tol:
            break
    
    if num_iters == max_iters - 1:
        print(f'Warning: Max. iterations ({max_iters}) reached without convergence.')

    return A_inv