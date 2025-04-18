import numpy as np
import scipy as sp

std_norm = sp.stats.norm(0, 1)

# TO-DO
# 1. Clean up imports.
# 2. Replace matrices with ndarrays.
# 3. Allow arbitrary distribution to be set.
# 4. Consider preloading sampled matrices.

def inv_sqrt(A: np.matrix):
    """
    Given a s.p.d. matrix A, returns the unique s.p.d. matrix X such that X^2 = A^-1.
    """
    eig, l = sp.linalg.eigh(A)
    eig = np.diag(1 / np.sqrt(np.real(eig)))
    l = np.real(l)
    return np.matrix(l @ eig @ l.T)

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
    Input matrix must be symmetric.
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

def Gower_Richtarik_2016_4(A: np.matrix, max_iters=1000, tol=1e-2, sketch_frac=None) -> np.matrix:
    """
    Algorithm 4 from Gower and Richtárik (2016).
    Called Adaptive Randomised BFGS (AdaRBFGS).
    Input matrix must be symmetric positive definite.
    Parameters: distribution D.
    """
    #assert A == A.T, 'Please ensure input matrix is symmetric.'

    n = A.shape[0]
    m = int(n ** 0.5) if sketch_frac == None else int(n * sketch_frac)
    tol *= n

    I = np.matrix(np.eye(n))
    L = np.matrix(np.eye(n))

    for num_iters in range(max_iters):
        #if np.linalg.matrix_norm(L * L.T * A - I) < tol:
        #    break

        S_tilde = np.matrix(std_norm.rvs(size=(n, m)))
        S = L * S_tilde
        R = inv_sqrt(S_tilde.T * A * S_tilde)
        L += S * R * (inv_sqrt(S_tilde.T * S_tilde) * S_tilde.T - R * S.T * A * L)
    
    if num_iters == max_iters - 1:
        print(f'Warning: Max. iterations ({max_iters}) reached without convergence.')
    
    return L * L.T