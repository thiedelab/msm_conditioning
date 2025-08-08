import numpy as np
from typing import Callable

def estimate_forward_relative_error_naive_bound(
        A_est: np.ndarray,
        b_est: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        vector_norm: Callable[[np.ndarray], float],
        matrix_norm: Callable[[np.ndarray], float],
        ) -> float:
    """
    Estimates the naive upper bound of the x vector error in equation.

    More precisely, it estimates the ||x_estimate - x||_2 error, where
    A_estimate * x_estimate = b_estimate is the experimental equation and
    A * x = b is the original equation, representing the true x vector value.

    norm: Callable[[np.ndarray], float]
        Norm function defined on vectors
    matrix_norm: Callable[[np.ndarray], float]
        Norm function defined on matrices
    """

    if(np.isclose(np.linalg.det(A), 0)):
        raise ValueError("A matrix is singular, cannot calculate the naive error bound")
    if(matrix_norm(A_est - A) * matrix_norm(np.linalg.inv(A)) >= 1):
        print("matrix_norm(A_est - A) * matrix_norm(np.linalg.inv(A)) must be >= 1, but is equal to", matrix_norm(A_est - A) * matrix_norm(np.linalg.inv(A)))
        raise ValueError("Theorem assumption doesn't hold, cannot calculate the naive error bound")

    rel_b_err = vector_norm(b_est - b) / vector_norm(b)
    rel_A_err = matrix_norm(A_est - A) / matrix_norm(A)
    cond_num = matrix_norm(A) * matrix_norm(np.linalg.inv(A))

    rel_x_err = (cond_num / (1 - cond_num * rel_A_err)) * (rel_b_err + rel_A_err)
    return rel_x_err

def matrix_spectral_norm(A: np.ndarray) -> float:
    """
    Induced 2-norm of a matrix A, same as Schatten infinity-norm.
    """
    eigenvalues = np.linalg.eigvals(np.conj(A.T) @ A)
    max_eigenvalue = np.sqrt(np.max(eigenvalues))
    return max_eigenvalue
    