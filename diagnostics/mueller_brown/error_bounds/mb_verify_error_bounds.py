from diagnostics.mueller_brown.mb_diagn_blocks import MBPipelineBlocks
from msm_playground.clustering.uniform_clustering import UniformClustering
from msm_playground.msm import MSM
from msm_playground.grid_based_reference import grid_hopping_process_on_rect_grid, grid_hopping_process
from msm_playground.course_graining import build_projection_matrix
from diagnostics.error_estimator_utils import matrix_spectral_norm, estimate_forward_relative_error_naive_bound
from diagnostics.potentials import MB_potential
from sklearn.preprocessing import normalize
import numpy.typing as npt
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from scipy.sparse import linalg as splinalg
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kT", type=float, default=40.0, help="Temperature of the system")
parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
parser.add_argument("--n-traj", type=int, default=1, help="Number of trajectories")
parser.add_argument("--traj-length", type=int, default=10000 * 30 * 9, help="Length of each trajectory")
parser.add_argument("--max-lag-time", type=int, default=500, help="committor function is benchmarked on lag time from min-lat-time to max-lag-time")
parser.add_argument("--min-lag-time", type=int, default=400, help="committor function is benchmarked on lag time from min-lat-time to max-lag-time")
parser.add_argument("--stride", type=int, default=25, help="Stride for lag time")
parser.add_argument("--seed", type=int, default=3, help="Seed for random number generator")
args = parser.parse_args()

kT = args.kT
n_clusters_per_dim = args.n_clusters
ref_grid_precision_factor = 10 # Number of clusters per dimension in the reference grid = n_clusters_per_dim * ref_grid_precision_factor before unifying clusters
n_traj = args.n_traj
traj_length = args.traj_length
max_lag_time = args.max_lag_time + 1
min_lag_time = args.min_lag_time
lag_time_stride = args.stride
seed = args.seed
burnin = 1000
dt = 1e-3 / 4
plot_folder = "plots/mb/"

def plot_matrix_as_image_plot(matrix: np.ndarray, title: str):
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig("data/{}".format(title))
    plt.close()

def build_reference_msm_on_grid(trajectory_dt: float, clusters_per_dim: int,
                                grid_left_bottom: npt.ArrayLike, grid_top_right: npt.ArrayLike,
                                kT: float) -> Tuple[MSM, npt.ArrayLike]:
    """
    Build a reference MSM on a grid with clusters_per_dim clusters per dimension.
    The grid is centered at grid_center and has dimensions grid_dimensions.

    Parameters
    ----------
    lag_time : float
        Lag time of the reference MSM in seconds.
    clusters_per_dim : int
        Number of clusters per dimension.
    grid_left_bottom : npt.ArrayLike
        Left bottom corner of the grid.
    grid_top_right : npt.ArrayLike
        Top right corner of the grid.
    kT : float
        Temperature of the system.

    Returns
    -------
    Tuple[MSM, npt.ArrayLike]
        Returns reference MSM, which has Tmat with lag time of 1 trajectory_dt.
    """
    U, (X, Y), (xgrid, ygrid) = MB_potential(clusters_per_dim,
                                             grid_left_bottom[0],
                                             grid_top_right[0],
                                             clusters_per_dim,
                                             grid_left_bottom[1],
                                             grid_top_right[1])

    mesh = np.meshgrid(xgrid, ygrid)
    list_of_states = np.vstack(list(map(np.ravel, mesh))).T
    dx = np.abs(xgrid[1] - xgrid[0])
    dy = np.abs(ygrid[1] - ygrid[0])

    assert np.isclose(dx, dy, atol=1e-3)

    L = grid_hopping_process(U, dx, kT)

    E = sps.identity(L.shape[0], format=L.format)
    epsilon = np.min([dx, dy])

    reference_Tmat_lag_time = epsilon**2 / 8
    print("Reference Tmat lag time / Trajectory dt:", reference_Tmat_lag_time / trajectory_dt)
    if (reference_Tmat_lag_time / trajectory_dt > 0.5):
        Tmat = E + trajectory_dt * L
    else:
        Tmat = E + reference_Tmat_lag_time * L
        Tmat = splinalg.matrix_power(Tmat, int(trajectory_dt // reference_Tmat_lag_time))

    reference_MSM = MSM(traj=None, clustering_obj=None, lag_time=1, seconds_between_frames=trajectory_dt)
    if(type(Tmat) == sps.csr_matrix):
        Tmat = Tmat.toarray()
    assert np.allclose(Tmat.sum(axis=1), 1)
    assert np.all(Tmat >= 0)
    reference_MSM.Tmat = Tmat
    reference_MSM.cluster_centers = list_of_states
    return reference_MSM


mb_pb = MBPipelineBlocks()

mb_pb.kT = kT
mb_pb.n_traj = n_traj
mb_pb.n_clusters_per_dim = n_clusters_per_dim
mb_pb.nsteps = traj_length
mb_pb.burnin = burnin
mb_pb.dt = dt
mb_pb.plot_folder = plot_folder

lag_times_list = np.arange(min_lag_time, max_lag_time, lag_time_stride, dtype=np.float16)
vanilla_true_error = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_true_error =  np.full(len(lag_times_list), fill_value=np.nan)
vanilla_error_upper_bound = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_error_upper_bound = np.full(len(lag_times_list), fill_value=np.nan)

# Debug
vanilla_matrix_diff_norms = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_matrix_diff_norms = np.full(len(lag_times_list), fill_value=np.nan)
vanilla_matrix_norms = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_matrix_norms = np.full(len(lag_times_list), fill_value=np.nan)
vanilla_inv_matrix_norms = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_inv_matrix_norms = np.full(len(lag_times_list), fill_value=np.nan)
vanilla_true_committor_norms = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_true_committor_norms = np.full(len(lag_times_list), fill_value=np.nan)

my_clustering = UniformClustering(n_clusters_per_dimension=n_clusters_per_dim)

EXP_VANILLA_MSM_NAME = "exp_vanilla_msm"
EXP_LAGSTOP_MSM_NAME = "exp_lagstop_msm"
EXP_TEST_MSM_NAME = "exp_test_msm"

def infinity_norm(vec: np.ndarray) -> float:
    return np.max(np.abs(vec))

def l1_norm(vec: np.ndarray) -> float:
    return np.sum(np.abs(vec))

def l2_norm(vec: np.ndarray) -> float:
    return np.sqrt(np.sum(vec**2))

CHOSEN_VEC_NORM = l2_norm


# STEP 1. Generate a trajectory within a grid

np.random.seed(seed)
initial_state = np.random.get_state()
np.random.set_state(initial_state)
grid_mean_point = np.array([-0.45 + 0.25, 0.7])
grid_dimensions = np.array([1.6, 1.6])
traj = mb_pb.simulate_simulation(seed=seed, grid_mean_point=grid_mean_point, grid_dimensions=grid_dimensions)

exp_test_msm = MSM(traj, my_clustering, 1, dt)
mb_pb.init_msm(msm=exp_test_msm, msm_name=EXP_TEST_MSM_NAME)
exp_test_msm.calculate_correlation_matrix()
simulation_is_bad = mb_pb.is_bad_simulation()
if(simulation_is_bad):
    raise ValueError("Simulation is bad. Please rerun the simulation with different seed or run it longer.")


# STEP 2. Build reference MSM on the exactly same grid

# First build reference MSM on the grid with same dimenstions but finer resolution (small cells, treated as "microstates")
# These reference MSMs have lag time of min_lag_time in trajectory_dt units
# and (n_clusters_per_dim * ref_grid_precision_factor)**2 clusters

ref_cluster_size_x = grid_dimensions[0] / n_clusters_per_dim / ref_grid_precision_factor
ref_cluster_size_y = grid_dimensions[1] / n_clusters_per_dim / ref_grid_precision_factor
ref_grid_left_bottom = grid_mean_point - ((n_clusters_per_dim * ref_grid_precision_factor - 1) / 2) * np.array([ref_cluster_size_x, ref_cluster_size_y])
ref_grid_top_right = grid_mean_point + ((n_clusters_per_dim * ref_grid_precision_factor - 1) / 2) * np.array([ref_cluster_size_x, ref_cluster_size_y])

ref_vanilla_fg_msm = build_reference_msm_on_grid(dt, n_clusters_per_dim * ref_grid_precision_factor,
                                        ref_grid_left_bottom,
                                        ref_grid_top_right,
                                        kT)
ref_vanilla_fg_Tmat_lagtime_stride = splinalg.matrix_power(ref_vanilla_fg_msm.Tmat, lag_time_stride)
ref_vanilla_fg_msm.Tmat = splinalg.matrix_power(ref_vanilla_fg_msm.Tmat, min_lag_time)
ref_vanilla_fg_msm.lag_time = min_lag_time

ref_lagstop_fg_msm = build_reference_msm_on_grid(dt, n_clusters_per_dim * ref_grid_precision_factor,
                                        ref_grid_left_bottom,
                                        ref_grid_top_right,
                                        kT)

# Setup the projection matrix and the measure for coarse graining

projection_matrix = build_projection_matrix(ref_vanilla_fg_msm.cluster_centers, exp_test_msm.cluster_centers)
measure = np.ones(ref_vanilla_fg_msm.Tmat.shape[0]) / ref_vanilla_fg_msm.Tmat.shape[0]
projection_mat_with_measure = projection_matrix.multiply(measure[:, np.newaxis])

# To ensure that the state boundaries are well-preserved, we will
# evaluate which course-grained states are inside the A and B regions
# and then pull those back to the high-res mesh.
cg_B_labels_mask = mb_pb.B_circle(exp_test_msm.cluster_centers)
cg_A_labels_mask = mb_pb.A_circle(exp_test_msm.cluster_centers)

fg_B_mask = projection_matrix @ cg_B_labels_mask
fg_A_mask = projection_matrix @ cg_A_labels_mask
fg_B_labels = np.where(fg_B_mask != 0)[0]
fg_A_labels = np.where(fg_A_mask != 0)[0]

fg_product_reactant_states = np.concatenate([fg_A_labels, fg_B_labels])
ref_lagstop_fg_msm.Tmat[fg_product_reactant_states, :] = 0
ref_lagstop_fg_msm.Tmat[fg_product_reactant_states, fg_product_reactant_states] = 1
plot_matrix_as_image_plot(ref_lagstop_fg_msm.Tmat, "ref_lagstop_fg_msm")

ref_lagstop_fg_Tmat_lagtime_stride = splinalg.matrix_power(ref_lagstop_fg_msm.Tmat, lag_time_stride)
ref_lagstop_fg_msm.Tmat = splinalg.matrix_power(ref_lagstop_fg_msm.Tmat, min_lag_time)
ref_lagstop_fg_msm.lag_time = min_lag_time

assert np.allclose(ref_lagstop_fg_Tmat_lagtime_stride.sum(axis=1), 1)
assert np.all(ref_lagstop_fg_Tmat_lagtime_stride >= 0)
assert np.allclose(np.min(ref_lagstop_fg_Tmat_lagtime_stride[fg_product_reactant_states, :]), 0)
assert np.allclose(ref_lagstop_fg_Tmat_lagtime_stride[fg_product_reactant_states, fg_product_reactant_states], 1)

for tau_idx, tau in enumerate(lag_times_list.copy().astype(int)):
    print("\ntau =", tau, "\n")
    gc.collect()

    # STEP 3. Construct experimental MSMs. Make sure they have n_clusters_per_dim**2 clusters (full grid coverage)
    exp_vanilla_msm = MSM(traj, my_clustering, tau, dt)
    exp_lagstop_msm = MSM(traj, my_clustering, tau, dt)
    mb_pb.init_msm(msm=exp_vanilla_msm, msm_name=EXP_VANILLA_MSM_NAME)
    mb_pb.init_msm(msm=exp_lagstop_msm, msm_name=EXP_LAGSTOP_MSM_NAME)

    mb_pb.plot_clustering_and_potential(my_clustering, traj, save_plot=True)

    exp_vanilla_msm.calculate_correlation_matrix(stopped_process=False)
    exp_lagstop_msm.calculate_correlation_matrix(stopped_process=True,
                                                    stop_state_index=np.concatenate([mb_pb.A_labels[EXP_LAGSTOP_MSM_NAME],
                                                    mb_pb.B_labels[EXP_LAGSTOP_MSM_NAME]]))

    assert exp_vanilla_msm.clustering_obj.n_clusters == n_clusters_per_dim**2
    assert exp_lagstop_msm.clustering_obj.n_clusters == n_clusters_per_dim**2

    # STEP 4. Get reference projected Tmat for the current tau

    # Reference Tmats
    ref_vanilla_cg_Tmat = projection_mat_with_measure.T @ ref_vanilla_fg_msm.Tmat @ projection_matrix
    ref_vanilla_cg_Tmat = normalize(ref_vanilla_cg_Tmat, norm='l1', axis=1)
    ref_vanilla_cg_msm = MSM(traj=None, clustering_obj=None,
                             lag_time=tau, seconds_between_frames=dt)
    ref_vanilla_cg_msm.Tmat = ref_vanilla_cg_Tmat
    plot_matrix_as_image_plot(ref_vanilla_cg_msm.Tmat, "ref_vanilla_cg_msm_{0}".format(tau))

    ref_lagstop_cg_Tmat = projection_mat_with_measure.T @ ref_lagstop_fg_msm.Tmat @ projection_matrix
    cg_product_reactant_states = np.concatenate([mb_pb.A_labels[EXP_LAGSTOP_MSM_NAME], mb_pb.B_labels[EXP_LAGSTOP_MSM_NAME]])
    ref_lagstop_cg_Tmat[cg_product_reactant_states, :] = 0
    ref_lagstop_cg_Tmat[cg_product_reactant_states, cg_product_reactant_states] = 1
    assert np.allclose(np.min(ref_lagstop_cg_Tmat[cg_product_reactant_states, :]), 0)
    assert np.allclose(ref_lagstop_cg_Tmat[cg_product_reactant_states, cg_product_reactant_states], 1)
    ref_lagstop_cg_Tmat = normalize(ref_lagstop_cg_Tmat, norm='l1', axis=1)
    ref_lagstop_cg_msm = MSM(traj=None, clustering_obj=None,
                             lag_time=tau, seconds_between_frames=dt)
    ref_lagstop_cg_msm.Tmat = ref_lagstop_cg_Tmat
    plot_matrix_as_image_plot(ref_lagstop_cg_msm.Tmat, "ref_lagstop_cg_msm_{0}".format(tau))

    # STEP 5. Calculate LHS and RHS errors for the chosen measure

    args_for_vanilla_committor_calculation = (mb_pb.A_labels[EXP_VANILLA_MSM_NAME], mb_pb.B_labels[EXP_VANILLA_MSM_NAME], False)
    args_for_lagstop_committor_calculation = (mb_pb.A_labels[EXP_LAGSTOP_MSM_NAME], mb_pb.B_labels[EXP_LAGSTOP_MSM_NAME], True)

    LHS_vanilla_exp_mat, RHS_vanilla_exp_vec = exp_vanilla_msm.derive_equation_for_committor(
        *args_for_vanilla_committor_calculation)
    LHS_lagstop_exp_mat, RHS_lagstop_exp_vec = exp_lagstop_msm.derive_equation_for_committor(
        *args_for_lagstop_committor_calculation)
    
    LHS_vanilla_ref_mat, RHS_vanilla_ref_vec = ref_vanilla_cg_msm.derive_equation_for_committor(
        *args_for_vanilla_committor_calculation)
    LHS_lagstop_ref_mat, RHS_lagstop_ref_vec = ref_lagstop_cg_msm.derive_equation_for_committor(
        *args_for_lagstop_committor_calculation)
    
    plot_matrix_as_image_plot(exp_vanilla_msm.Tmat, "exp_vanilla_msm_Tmat_{0}".format(tau))
    plot_matrix_as_image_plot(exp_lagstop_msm.Tmat, "exp_lagstop_msm_Tmat_{0}".format(tau))
    plot_matrix_as_image_plot(np.abs(exp_vanilla_msm.Tmat - ref_vanilla_cg_msm.Tmat), "exp_vanilla_msm_Tmat_diff_{0}".format(tau))
    plot_matrix_as_image_plot(np.abs(exp_lagstop_msm.Tmat - ref_lagstop_cg_msm.Tmat), "exp_lagstop_msm_Tmat_diff_{0}".format(tau))

    # STEP 6. Plug everything into the Theorem 1 to get error bound. Compare it with the actual error.
    
    try:
        vanilla_upper_error_bound = estimate_forward_relative_error_naive_bound(LHS_vanilla_exp_mat, RHS_vanilla_exp_vec,
                                                                                LHS_vanilla_ref_mat, RHS_vanilla_ref_vec,
                                                                                CHOSEN_VEC_NORM, matrix_spectral_norm)

        lagstop_upper_error_bound = estimate_forward_relative_error_naive_bound(LHS_lagstop_exp_mat, RHS_lagstop_exp_vec,
                                                                                LHS_lagstop_ref_mat, RHS_lagstop_ref_vec,
                                                                                CHOSEN_VEC_NORM, matrix_spectral_norm)
    except ValueError as e:
        print(e)
        print("Skipping the current tau")
        vanilla_upper_error_bound = np.nan
        lagstop_upper_error_bound = np.nan

    # STEP 7. Calculate the actual error

    vanilla_error = CHOSEN_VEC_NORM(exp_vanilla_msm.calculate_committor(*args_for_vanilla_committor_calculation) - 
                                        ref_vanilla_cg_msm.calculate_committor(*args_for_vanilla_committor_calculation))
    lagstop_error = CHOSEN_VEC_NORM(exp_lagstop_msm.calculate_committor(*args_for_lagstop_committor_calculation) - 
                                        ref_lagstop_cg_msm.calculate_committor(*args_for_lagstop_committor_calculation))
    
    vanilla_true_error[tau_idx] = vanilla_error
    lagstop_true_error[tau_idx] = lagstop_error
    vanilla_error_upper_bound[tau_idx] = vanilla_upper_error_bound
    lagstop_error_upper_bound[tau_idx] = lagstop_upper_error_bound
    
    print("Vanilla true error norm:", vanilla_error)
    print("Lagstop true error norm:", lagstop_error)
    print("Vanilla upper error bound:", vanilla_upper_error_bound)
    print("Lagstop upper error bound:", lagstop_upper_error_bound)

    vanilla_matrix_diff_norms[tau_idx] = matrix_spectral_norm(exp_vanilla_msm.Tmat - ref_vanilla_cg_msm.Tmat)
    lagstop_matrix_diff_norms[tau_idx] = matrix_spectral_norm(exp_lagstop_msm.Tmat - ref_lagstop_cg_msm.Tmat)
    vanilla_matrix_norms[tau_idx] = matrix_spectral_norm(ref_vanilla_cg_msm.Tmat)
    lagstop_matrix_norms[tau_idx] = matrix_spectral_norm(ref_lagstop_cg_msm.Tmat)
    vanilla_inv_matrix_norms[tau_idx] = matrix_spectral_norm(np.linalg.inv(ref_vanilla_cg_msm.Tmat))
    lagstop_inv_matrix_norms[tau_idx] = matrix_spectral_norm(np.linalg.inv(ref_lagstop_cg_msm.Tmat))
    vanilla_true_committor_norms[tau_idx] = CHOSEN_VEC_NORM(exp_vanilla_msm.calculate_committor(*args_for_vanilla_committor_calculation))
    lagstop_true_committor_norms[tau_idx] = CHOSEN_VEC_NORM(exp_lagstop_msm.calculate_committor(*args_for_lagstop_committor_calculation))

    # STEP 8. Setup the next iteration
    ref_vanilla_fg_msm.Tmat = ref_vanilla_fg_msm.Tmat.dot(ref_vanilla_fg_Tmat_lagtime_stride)
    ref_vanilla_fg_msm.lag_time = tau
    ref_lagstop_fg_msm.Tmat = ref_lagstop_fg_msm.Tmat.dot(ref_lagstop_fg_Tmat_lagtime_stride)
    ref_lagstop_fg_msm.lag_time = tau

plt.plot(lag_times_list, vanilla_matrix_diff_norms, label="Vanilla error matrix norm")
plt.plot(lag_times_list, lagstop_matrix_diff_norms, label="Lagstop error matrix norm")
plt.xlabel("Lag time")
plt.ylabel("Matrix norm")
plt.legend()
plt.savefig("data/matrix_norms")
plt.close()

np.save("data/mb_committor_error_of_lag_time_ntraj_{}_trajlength_{}_seed_{}.npy".format(n_traj, traj_length, seed),
        [lag_times_list,
         vanilla_true_error, lagstop_true_error,
         vanilla_error_upper_bound, lagstop_error_upper_bound,
         vanilla_matrix_diff_norms, lagstop_matrix_diff_norms,
         vanilla_matrix_norms, lagstop_matrix_norms,
         vanilla_inv_matrix_norms, lagstop_inv_matrix_norms,
         vanilla_true_committor_norms, lagstop_true_committor_norms
         ])
