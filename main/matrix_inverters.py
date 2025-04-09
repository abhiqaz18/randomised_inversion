import numpy as np
from scipy.stats import norm
std_norm = norm(0, 1)

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
    for i in range(num_iters):
        S = np.matrix(std_norm.rvs(size=(n, m)))
        L = S * np.matrix(np.linalg.inv(S.T * A * A.T * S)) * S.T
        A_inv += A.T * L * (I - A * A_inv)

    return A_inv

if __name__ == '__main__':
    A = np.matrix(std_norm.rvs(size=(10, 10)))
    A_inv = Gower_Richtarik_2016_1(A)
    print(np.linalg.eigvals(A * A_inv))