import numpy as np
import numpy.typing as npt
from scipy import sparse as sps
from typing import Union


def transition_prob(x: Union[float, npt.ArrayLike]) -> Union[float, npt.ArrayLike]:
    return 0.25 / (1 + np.exp(x))

def grid_1D_hopping_process(U: npt.ArrayLike, epsilon: float, kT: float=1.) -> sps.csr_matrix:
    """
    Compute the transition matrix for a grid-hopping process on a 1D grid

    Parameters
    ----------
    U : np.ndarray
        Potential energy surface, first and only index is x
    """
    if(U.ndim != 1):
        raise ValueError("U must be a 1D array")
    U = U / kT
    n = U.size
    x_keys = np.arange(n)

    # Calculation Transition Probabilities
    delta_U_positive_x = transition_prob(U[1:] - U[:-1])
    source_keys_pos_x = x_keys[:-1]
    target_keys_pos_x = x_keys[1:]

    delta_U_negative_x = transition_prob(U[:-1] - U[1:])
    source_keys_neg_x = x_keys[1:]
    target_keys_neg_x = x_keys[:-1]

    all_U = np.concatenate(
        [delta_U_positive_x, delta_U_negative_x]
    )

    row_indices = np.concatenate(
        [source_keys_pos_x, source_keys_neg_x]
    )
    col_indices = np.concatenate(
        [target_keys_pos_x, target_keys_neg_x]
    )

    # Calculate transition matrix and approximation of the generator.
    P = sps.coo_matrix((all_U, (row_indices, col_indices)), shape=(n, n))
    P = P.tocsr()
    Pdiag = sps.diags(P.sum(axis=1).A.ravel())
    L = P - Pdiag
    L = (8 / epsilon**2) * L
    return L


def grid_hopping_process(U: npt.ArrayLike, epsilon: float, kT: float=1.) -> sps.csr_matrix:
    """
    Compute the transition matrix for a grid-hopping process on a 2D grid

    Parameters
    ----------
    U : np.ndarray
        Potential energy surface, first index is x, second index is y
    """
    U = U / kT
    n, m = U.shape
    y_keys, x_keys = np.meshgrid(np.arange(n), np.arange(m))

    # Calculation Transition Probabilities
    delta_U_positive_x = transition_prob(U[1:, :] - U[:-1, :]).flatten()
    source_keys_pos_x = n * x_keys[:-1, :].ravel() + y_keys[:-1, :].ravel()
    target_keys_pos_x = n * x_keys[1:, :].ravel() + y_keys[1:, :].ravel()

    delta_U_negative_x = transition_prob(U[:-1, :] - U[1:, :]).flatten()
    source_keys_neg_x = n * x_keys[1:, :].ravel() + y_keys[1:, :].ravel()
    target_keys_neg_x = n * x_keys[:-1, :].ravel() + y_keys[:-1, :].ravel()

    delta_U_positive_y = transition_prob(U[:, 1:] - U[:, :-1]).flatten()
    source_keys_pos_y = n * x_keys[:, :-1].ravel() + y_keys[:, :-1].ravel()
    target_keys_pos_y = n * x_keys[:, 1:].ravel() + y_keys[:, 1:].ravel()

    delta_U_negative_y = transition_prob(U[:, :-1] - U[:, 1:]).flatten()
    source_keys_neg_y = n * x_keys[:, 1:].ravel() + y_keys[:, 1:].ravel()
    target_keys_neg_y = n * x_keys[:, :-1].ravel() + y_keys[:, :-1].ravel()

    all_U = np.concatenate(
        [delta_U_positive_x, delta_U_negative_x, delta_U_positive_y, delta_U_negative_y]
    )

    row_indices = np.concatenate(
        [source_keys_pos_x, source_keys_neg_x, source_keys_pos_y, source_keys_neg_y]
    )
    col_indices = np.concatenate(
        [target_keys_pos_x, target_keys_neg_x, target_keys_pos_y, target_keys_neg_y]
    )

    # Calculate transition matrix and approximation of the generator.
    P = sps.coo_matrix((all_U, (row_indices, col_indices)), shape=(n * m, n * m))
    P = P.tocsr()
    Pdiag = sps.diags(P.sum(axis=1).A.ravel())
    L = P - Pdiag
    L = (8 / epsilon**2) * L
    return L

def grid_hopping_process_on_rect_grid(U: npt.ArrayLike, epsilon_x: float, epsilon_y: float, kT: float=1.) -> sps.csr_matrix:
    """
    Compute the transition matrix for a grid-hopping process on a 2D grid

    Parameters
    ----------
    U : np.ndarray
        Potential energy surface, first index is x, second index is y
    
    epsilon_x : float
        Grid spacing in x direction

    epsilon_y : float
        Grid spacing in y direction

    kT : float
        Temperature of the system

    Returns
    -------
    L : sps.csr_matrix
        Sparse matrix representing the generator of the grid-hopping process
    """
    U = U / kT
    n, m = U.shape
    y_keys, x_keys = np.meshgrid(np.arange(n), np.arange(m))

    # Calculation Transition Probabilities
    delta_U_positive_x = transition_prob(U[1:, :] - U[:-1, :]).flatten() * np.min([epsilon_y / epsilon_x, 1])**2
    source_keys_pos_x = n * x_keys[:-1, :].ravel() + y_keys[:-1, :].ravel()
    target_keys_pos_x = n * x_keys[1:, :].ravel() + y_keys[1:, :].ravel()

    delta_U_negative_x = transition_prob(U[:-1, :] - U[1:, :]).flatten() * np.min([epsilon_y / epsilon_x, 1])**2
    source_keys_neg_x = n * x_keys[1:, :].ravel() + y_keys[1:, :].ravel()
    target_keys_neg_x = n * x_keys[:-1, :].ravel() + y_keys[:-1, :].ravel()

    delta_U_positive_y = transition_prob(U[:, 1:] - U[:, :-1]).flatten() * np.min([epsilon_x / epsilon_y, 1])**2
    source_keys_pos_y = n * x_keys[:, :-1].ravel() + y_keys[:, :-1].ravel()
    target_keys_pos_y = n * x_keys[:, 1:].ravel() + y_keys[:, 1:].ravel()

    delta_U_negative_y = transition_prob(U[:, :-1] - U[:, 1:]).flatten() * np.min([epsilon_x / epsilon_y, 1])**2
    source_keys_neg_y = n * x_keys[:, 1:].ravel() + y_keys[:, 1:].ravel()
    target_keys_neg_y = n * x_keys[:, :-1].ravel() + y_keys[:, :-1].ravel()

    all_U = np.concatenate(
        [delta_U_positive_x, delta_U_negative_x, delta_U_positive_y, delta_U_negative_y]
    )

    row_indices = np.concatenate(
        [source_keys_pos_x, source_keys_neg_x, source_keys_pos_y, source_keys_neg_y]
    )
    col_indices = np.concatenate(
        [target_keys_pos_x, target_keys_neg_x, target_keys_pos_y, target_keys_neg_y]
    )

    # Calculate transition matrix and approximation of the generator.
    P = sps.coo_matrix((all_U, (row_indices, col_indices)), shape=(n * m, n * m))
    P = P.tocsr()
    Pdiag = sps.diags(P.sum(axis=1).A.ravel())
    L = P - Pdiag
    L = (8 / np.min([epsilon_x, epsilon_y])**2) * L
    return L
