from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from numba import njit

def merge_trajectories(traj: List[npt.ArrayLike], lag_time: int) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Convert a list of frames to a numpy array
    
    Parameters
    ----------
    traj : List[npt.ArrayLike]
        List of frames to convert to numpy array
        
    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]]
        Tuple of (traj, mask)
    merged_traj : npt.NDArray
        Numpy array of shape (n_frames, n_coords) containing all trajectories merged together into one long trajectory
    mask : npt.NDArray
        Numpy boolean array of shape (n_frames) containing a mask to indicate when pair (frame, frame+lag_time) belongs to same trajectories
        mask[t] is True when pair (t, t+lag_time) belongs to same trajectory, looking at the first frame of the pair
    """

    merged_traj = np.concatenate(traj)  # Concatenate all trajectories into a single long one
    
    n_frames = len(merged_traj)
    mask_length = n_frames
    if(mask_length < 0):
        raise ValueError("lag_time is larger than the total number of frames")
    mask = np.zeros(mask_length, dtype=bool)  # Initialize mask with zeros
    
    current_frame = 0
    for traj in traj:
        traj_length = len(traj)
        if traj_length > lag_time:
            mask[current_frame : current_frame + traj_length - lag_time] = True
        current_frame += traj_length

    # NOT be picked by first frame of the pair via cutoff, but PICKED by second frame of the pair
    mask[-lag_time:] = True
    return merged_traj, mask

def convert_to_zero_indexed(arr_with_gaps: np.ndarray, value_map: bool = False) -> np.ndarray:
    unique_labels, indices = np.unique(arr_with_gaps, return_inverse=True)
    sorted_indices = np.argsort(unique_labels)
    zero_indexed_arr = np.searchsorted(unique_labels[sorted_indices], arr_with_gaps)
    arr_no_gaps = sorted_indices[zero_indexed_arr]
    # get value map of (arr_with_gaps) -> (converted to zero indexed arr_with_gaps)
    if value_map:
        value_map = {old_value: new_value for old_value, new_value in zip(unique_labels, np.unique(arr_no_gaps))}
        return arr_no_gaps, value_map
    return arr_no_gaps

@njit
def sample_from_grid(grid: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """
    Generates random numbers from a given grid using a discrete probability density.

    Parameters:
        grid (np.ndarray): Array of shape (n_cells_in_grid, n_features) representing the flattened spatial grid.
        probabilities (np.ndarray): Array of (n_cells_in_grid, 1) representing the corresponding probabilities for each flattened grid value.

    Returns:
        np.ndarray: A vector sampled from the grid based on the given probabilities.
    """
    probabilities /= np.sum(probabilities)
    return custom_random_choice(grid, probabilities)

@njit
def custom_random_choice(grid, probabilities):
    cumulative_probabilities = np.cumsum(probabilities)
    r = np.random.rand()
    return grid[np.searchsorted(cumulative_probabilities, r)]
