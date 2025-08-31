import numpy as np
import scipy as sp

std_norm = sp.stats.norm(0, 1)

# TO-DO
# 1. Clean up imports.
# 2. Bracketing @ for speed.
# 3. Use sqrt then solve instead of inv_sqrt then multiply.
# 4. Allow arbitrary distribution to be set.
# 5. Consider batching random matrix generation.
# 6. Fix DRY violations - a general matrix inverter should accept a preconditioner, an iterator, and an output processor (e.g. if your iterates are L, return L @ L.T)

def inv_sqrt(A: np.ndarray) -> np.ndarray:
    """
    Given a s.p.d. matrix A, returns the unique s.p.d. matrix X such that X^2 = A^-1.
    """
    eig, l = sp.linalg.eigh(A)
    eig = np.diag(1 / np.sqrt(np.real(eig)))
    l = np.real(l)
    return l @ eig @ l.T

def Gower_Richtarik_2016_4(A: np.ndarray, max_iters=10000, tol=1e-2, tol_check_period=1000, sketch_frac=None, precondition=True) -> np.ndarray:
    """
    Algorithm 4 from Gower and Richtárik (2016).
    Called Adaptive Randomised BFGS (AdaRBFGS).
    Input matrix must be symmetric positive definite.
    Parameters: distribution D.
    """
    assert np.allclose(A, A.T), 'Please ensure input matrix is symmetric.'

    n = A.shape[0]
    m = int(n ** 0.5) if sketch_frac == None else int(n * sketch_frac)
    
    tol *= n

    I = np.eye(n)

    if precondition:
        # scale identity by trace(A) / trace(A^2)
        L = sp.linalg.cholesky(I * np.trace(A) / np.trace(A @ A), lower=True)
    else:
        L = np.eye(n)

    for num_iter in range(max_iters):
        if tol_check_period and (num_iter % tol_check_period == 0):
            if np.linalg.matrix_norm(L @ L.T @ A - I) < tol:
                break

        S_tilde = std_norm.rvs(size=(n, m), random_state=num_iter)
        S = L @ S_tilde
        R = inv_sqrt(S_tilde.T @ A @ S_tilde)
        L += S @ R @ (inv_sqrt(S_tilde.T @ S_tilde) @ S_tilde.T - R.T @ S.T @ A @ L)
    
    if num_iter == max_iters - 1:
        print(f'Warning: Max. iterations ({max_iters}) reached without convergence.')
    
    return L @ L.T

def Gower_Richtarik_2016_4_corrected(A: np.ndarray, max_iters=10000, tol=1e-2, tol_check_period=1000, sketch_frac=None, precondition=True) -> np.ndarray:
    """
    Corrected version of algorithm 4 from Gower and Richtárik (2016).
    Input matrix must be symmetric positive definite.
    Parameters: distribution D.
    """
    assert np.allclose(A, A.T), 'Please ensure input matrix is symmetric.'

    n = A.shape[0]
    m = int(n ** 0.5) if sketch_frac == None else int(n * sketch_frac)
    
    tol *= n

    I = np.eye(n)

    if precondition:
        # scale identity by trace(A) / trace(A^2)
        L = sp.linalg.cholesky(I * np.trace(A) / np.trace(A @ A), lower=True)
    else:
        L = np.eye(n)

    for num_iter in range(max_iters):
        if tol_check_period and (num_iter % tol_check_period == 0):
            if np.linalg.matrix_norm(L @ L.T @ A - I) < tol:
                break

        S_tilde = std_norm.rvs(size=(n, m), random_state=num_iter)
        S = L @ S_tilde
        R = inv_sqrt(S.T @ A @ S)
        L += S @ R @ (inv_sqrt(S_tilde.T @ S_tilde) @ S_tilde.T - R.T @ S.T @ A @ L)
    
    if num_iter == max_iters - 1:
        print(f'Warning: Max. iterations ({max_iters}) reached without convergence.')
    
    return L @ L.T