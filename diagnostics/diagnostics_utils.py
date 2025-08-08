from msm_playground.msm import MSM
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional
from scipy import stats

clusters_colors = np.array([
    'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta', 'lime'
])

def save_logs(msm_obj: MSM):
    log_file = open("logs/1D_rwalk.log",'w+')
    np.set_printoptions(precision=4, suppress=True, linewidth=100)

    print("Parameters: lag_time =", msm_obj.lag_time, "N_clusters =", msm_obj.n_states, file=log_file)
    print("Transition matrix:\n", msm_obj.Tmat, file=log_file)
    print("Stationary distribution:\n", msm_obj.stationary_distribution, file=log_file)
    print("Committor function:\n", msm_obj.committor, file=log_file)
    print("\n", file=log_file)

    log_file.close()

def save_plots(msm_obj: MSM, target_stationary_distribution: np.ndarray, target_committor: np.ndarray):
    plt.clf()
    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].set_ylabel('stationary distribution')
    axs[0].set_xlabel('cluster index')
    axs[0].plot(msm_obj.stationary_distribution)
    axs[0].plot(target_stationary_distribution, linestyle='--')

    axs[1].set_ylabel('committor function')
    axs[1].set_xlabel('cluster index')
    axs[1].plot(msm_obj.committor)
    axs[1].plot(target_committor, linestyle='--')

    params_string = "lag_time = " + str(msm_obj.lag_time) + ", N_clusters = " + str(msm_obj.clustering_obj.n_clusters)
    fig.suptitle(params_string, fontsize=10)
    fig.subplots_adjust(hspace=2.5)
    fig.tight_layout()
    plt.savefig("plots/" + params_string + "")
    plt.close(fig)

def plot_1D_clustering(traj: np.ndarray, time_arr: np.ndarray, clustering_obj: Callable):
    """
    Plots both 1D clustering and time evolution of the trajectory
    """
    n_traj = len(traj)
    params_string = "N_clusters = " + str(clustering_obj.n_clusters) + ", n_traj = " + str(n_traj)
    plt.clf()
    clustering_obj(traj=np.concatenate(traj, axis=0))
    previous_traj_last_index = 0
    for i in range(n_traj):
        plt.scatter(time_arr, traj[i],
            color=clusters_colors[clustering_obj.labels[previous_traj_last_index:previous_traj_last_index
            + len(traj[0])] % len(clusters_colors)], s=0.5)
        previous_traj_last_index += len(traj[0])
    plt.title('Uniform clustering of 1D random walk')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.savefig("plots/clustering " + params_string + "")
    plt.show()

def count_hits(labels: np.ndarray, target_state_index: List[int]) -> int:
    """
    Counts the number of times the trajectory hits the given state

    Parameters
    ----------
    labels: np.ndarray
        Labels of the trajectory (labels[i] is the label of the i-th frame)
    target_state_index : List[int]
        Indices of the target state
    """

    previous_frames = labels[:-1]
    next_frames = labels[1:]

    previous_not_in_target = np.logical_not(np.isin(previous_frames, target_state_index))
    next_in_target = np.isin(next_frames, target_state_index)

    hits = np.sum(np.logical_and(previous_not_in_target, next_in_target))

    return hits

def downscale_func_on_grid(reference_func: np.ndarray, new_dimensions: Tuple[int, int]) -> np.ndarray:
    """
    Downsacales the reference function on grid to the new given grid dimensions.
    The rescaling is done by taking the average of the values.

    Parameters
    ----------
    reference_func: np.ndarray
        Reference function to rescale
    new_dimensions : Tuple
        New dimensions of the reference function

    Returns
    -------
    np.ndarray
        Rescaled reference function with the given dimensions 
    """
    old_x, old_y = reference_func.shape
    new_x, new_y = new_dimensions
    x_ratio = old_x / new_x
    y_ratio = old_y / new_y

    new_func = np.zeros(new_dimensions)

    # Create indices for the rows and columns
    rows = np.arange(new_x)[:, np.newaxis]
    cols = np.arange(new_y)

    # Calculate the corresponding indices in the original array
    start_x = np.floor(rows * x_ratio).astype(int)
    end_x = np.floor((rows + 1) * x_ratio).astype(int)
    start_y = np.floor(cols * y_ratio).astype(int)
    end_y = np.floor((cols + 1) * y_ratio).astype(int)

    # Use numpy's fancy indexing to select values from reference_func
    new_func = np.mean(reference_func[start_x, start_y][:, :, None, None], axis=(2, 3))

    return new_func    

def plot_2D_committor(committor_mesh: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray,
                       ref_potential: Optional[np.ndarray]=None):
    """
    Plots the committor function on a 2D grid

    Parameters
    ----------
    committor_mesh: np.ndarray
        Committor function on a 2D grid
    xgrid : np.ndarray
        X coordinates of the grid
    ygrid : np.ndarray
        Y coordinates of the grid
    """
    plt.clf()
    fig, ax = plt.subplots(1)
    HM = ax.pcolor(xgrid, ygrid, committor_mesh, vmin=0, vmax=1)
    if ref_potential is not None:
        ax.contour(xgrid, ygrid, ref_potential, levels=np.linspace(0, 10., 11), colors='k') # Contour lines every 1 k_B T
    ax.set_aspect('equal')
    cbar = plt.colorbar(HM, ax=ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('True Committor')

def narrow_down_exp_func_domain(probability_map_on_cluster_centers: np.ndarray, exp_func_domain: List[np.ndarray], exp_func: np.ndarray,
                                rel_probability_threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Narrows down the domain of the experimental function by removing points with low probability

    Parameters
    ----------
    probability_map_on_cluster_centers: np.ndarray
        Probability (density) map of the experimental function, 1D array of values.
    exp_func_domain : np.ndarray
        Domain of the experimental function, 1D array of vectors
    exp_func: np.ndarray
        Experimental function, 1D array of values. Function is determined only on some sparse points on the grid
    rel_probability_threshold : float
        Relative probability threshold. Points with probability lower np.max(probability_map) * rel_probability_threshold will be removed from the domain
        The higher the threshold, the more points will be removed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (new_exp_func_domain, new_probability_map_on_cluster_centers)
        new_exp_func_domain : np.ndarray
            New domain of the experimental function, 1D array of vectors
        exp_func_on_new_domain : np.ndarray
            Experimental function on the new domain, 1D array of values
    """
    max_probability = np.max(probability_map_on_cluster_centers)
    probability_threshold = max_probability * rel_probability_threshold
    indices = np.where(probability_map_on_cluster_centers > probability_threshold)
    if(len(indices[0]) == 0):
        return np.array([]), np.array([])
    new_exp_func_domain = exp_func_domain[indices]
    exp_func_on_new_domain = exp_func[indices]

    return new_exp_func_domain, exp_func_on_new_domain

def calc_func_values_on_grid(ref_func: np.ndarray, exp_func_domain: np.ndarray,
                            ref_func_grid: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Calculates the value of functions on a different grid domain

    Parameters
    ----------
    exp_func: np.ndarray
        Experimental function, 1D array of values. Function is determined only on some sparse points on the grid
    ref_func : np.ndarray
        Reference function, determined on a whole grid domain
    exp_func_domain : np.ndarray
        Domain of the experimental function, 1D array of 2D points
    ref_func_grid : Tuple[np.ndarray, np.ndarray]
        Grid domain of the reference function. Tuple of two 2D arrays: (xgrid, ygrid)

    Returns
    -------
    pointwise_error: np.ndarray
        Pointwise error between the two functions, 1D array determined on the experimental function domain
    """
    xgrid, ygrid = ref_func_grid

    x_indices = np.argmin(np.abs(xgrid.reshape(-1, 1) - exp_func_domain[:, 0]), axis=0)
    y_indices = np.argmin(np.abs(ygrid.reshape(-1, 1) - exp_func_domain[:, 1]), axis=0)
    
    # get ref_func value at geometric points of exp_func_domain, same shape as exp_func
    ref_func_values_on_exp_func_domain = ref_func[x_indices, y_indices]
    
    return ref_func_values_on_exp_func_domain

def calc_func_error_on_grid(exp_func: np.ndarray, ref_func: np.ndarray, exp_func_domain: np.ndarray,
                            ref_func_grid: Tuple[np.ndarray, np.ndarray], relative_error=False) -> np.ndarray:
    """
    Calculates the error between two functions on a grid

    Parameters
    ----------
    exp_func: np.ndarray
        Experimental function, 1D array of values. Function is determined only on some sparse points on the grid
    ref_func : np.ndarray
        Reference function, determined on a whole grid domain
    exp_func_domain : np.ndarray
        Domain of the experimental function, 1D array of 2D points
    ref_func_grid : Tuple[np.ndarray, np.ndarray]
        Grid domain of the reference function. Tuple of two 2D arrays: (xgrid, ygrid)

    Returns
    -------
    pointwise_error: np.ndarray
        Pointwise error between the two functions, 1D array determined on the experimental function domain
    """    
    ref_func_values_on_exp_func_domain = calc_func_values_on_grid(ref_func, exp_func_domain, ref_func_grid)
    pointwise_error = np.abs(exp_func - ref_func_values_on_exp_func_domain)
    if relative_error:
        pointwise_error /= np.abs(ref_func_values_on_exp_func_domain)
    
    return pointwise_error

def first_n_dominant_eigenvectors(Mat: np.ndarray, n: int=5) -> List[np.ndarray]:
    """
    Get first n dominant eigenvectors of a matrix
    
    Parameters
    ----------
    Mat : np.ndarray
        Matrix to get eigenvectors from
    n : int
        Number of eigenvectors to get
        
    Returns
    -------
    List[np.ndarray]
        List of eigenvectors
    """
    eigvals, eigvecs = np.linalg.eig(Mat)
    sorted_indices = np.argsort(np.abs(eigvals))[::-1][:n]
    if(~np.all(np.isreal(eigvals[sorted_indices]))):
        print('Warning: some eigenvalues are complex')
        print(np.where(~np.isreal(eigvals)))
    print("Eigenvalues:", eigvals[sorted_indices])
    eigvals = np.abs(eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = np.abs(eigvecs)
    eigvecs = eigvecs[:, sorted_indices]
    return [eigvecs[:, i] for i in range(n)]

def plot_dominant_eigenvectors(msm_obj: MSM, n: int=5):
    cmap = 'viridis'
    fig, axes = plt.subplots(1, n, figsize=(15, 3))

    for i, vector in enumerate(first_n_dominant_eigenvectors(msm_obj.Tmat.T, n)):
        ax = axes[i] if n > 1 else axes
        sc = ax.scatter(msm_obj.cluster_centers[:, 0], msm_obj.cluster_centers[:, 1], c=vector,
                    cmap=cmap, marker='s', s=40, alpha=0.7, vmin=np.min(vector), vmax=np.max(vector))
        ax.set_title(f'{i+1}-th left eigenvector')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Colorbar')

    plt.tight_layout()
    plt.show()

def plot_boltzmann(U, kT, xgrid, ygrid):
    probability = np.exp(-U / kT)
    probability /= np.sum(probability) 

    x = np.arange(U.shape[0])
    y = np.arange(U.shape[1])
    X, Y = np.meshgrid(x, y)

    plt.imshow(probability, cmap='viridis', origin='lower', extent=[xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]],
               vmin=np.min(probability), vmax=np.max(probability))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Boltzmann Distribution')
    plt.colorbar()

def get_measurements_error(measurements: np.ndarray, confidence_level=0.95) -> np.ndarray:
    mean = np.mean(measurements)
    std_dev = np.std(measurements, ddof=1)
    # Calculate the confidence interval
    interval = stats.t.interval(confidence_level, len(measurements) - 1, loc=mean, scale=std_dev)
    # Calculate the error as half of the confidence interval range
    error = (interval[1] - interval[0]) / 2

    return error
