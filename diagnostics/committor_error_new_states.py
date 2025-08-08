import numpy as np
from numpy import typing as npt
from diagnostics.diagnostics_utils import *
import matplotlib.pyplot as plt
import os
from diagnostics.potentials import MB_potential
from msm_playground import condition_numbers_utils
from msm_playground.grid_based_reference import grid_hopping_process
from msm_playground.matplotlib_loader import load_matplotlib_presets, CycleStyle, fix_axis_intersection_at_y0
from msm_playground import course_graining as cg
from msm_playground.course_graining import (
    build_downsampled_matrix,
    build_meshes,
)


def is_inside_circle(
    coord: npt.ArrayLike, center: npt.ArrayLike, rad: float
) -> npt.ArrayLike:
    return np.linalg.norm(coord - center, axis=-1) <= rad


def B_circle(coord: npt.ArrayLike) -> npt.ArrayLike:
    return is_inside_circle(coord, center=np.array([-0.558, 1.441]), rad=0.5)


def A_circle(coord: npt.ArrayLike) -> npt.ArrayLike:
    return is_inside_circle(coord, center=np.array([0.623, 0.028]), rad=0.5)

def _get_projected_reference(
    reference_Tmat,
    downsample_matrix,
    state_A,
    state_B,
    measure=None,
):
    """
    Evaluates the fine-grainod committor and projects it onto the
    course-grained states.
    """
    if measure is None:
        measure = np.ones(reference_Tmat.shape[0])
        measure /= np.sum(measure)

    committor = cg._build_naive_committor_from_Tmat(reference_Tmat, state_A, state_B)
    projection_matrix = downsample_matrix * measure
    projection_matrix /= np.sum(projection_matrix, axis=1, keepdims=True)
    projected_committor = projection_matrix.dot(committor)
    return projected_committor, committor


def evaluate_committors_and_cond_nums(save_dir: str = "data/mb/inf-data/committor/"):
    # Simulation Parameters
    kT = 30.0
    n_clusters_per_x_dim = 60
    x_min = -3.0
    x_max = 2.0
    n_clusters_per_y_dim = 60
    y_min = -1.5
    y_max = 3.5
    max_lagtime_factor = 300

    # Built the Boltzmann potential and the transition matrix
    U, (X, Y), (xgrid, ygrid) = MB_potential(
        dots_per_x_dim=n_clusters_per_x_dim,
        x_min=x_min,
        x_max=x_max,
        dots_per_y_dim=n_clusters_per_y_dim,
        y_min=y_min,
        y_max=y_max,
    )
    dx = (x_max - x_min) / n_clusters_per_x_dim
    dy = (y_max - y_min) / n_clusters_per_y_dim

    assert (
        dx == dy
    )  # This is a requirement for the grid_hopping_process to be well-defined

    measure = None  # Sampling measure is uniform (aka NOT Boltzmann)

    # Transition matrix is I + L dt at short time
    L = grid_hopping_process(U, dx, kT=kT).toarray()
    L = L.astype(np.float64)
    reference_Tmat_lag_time = dx**2 / 8
    reference_Tmat = np.eye(L.shape[0]) + reference_Tmat_lag_time * L

    lag_times_list = np.arange(1, max_lagtime_factor + 1) * reference_Tmat_lag_time

    # Construct the high-res and downsampled meshes
    downsample_states_per_dim_factor = 4
    high_res_mesh, downsampled_mesh = build_meshes(
        X, Y, downsample_states_per_dim_factor
    )

    # save reference Tmat and A, B states
    np.save(save_dir + "reference_Tmat.npy", reference_Tmat)
    state_A_idx = np.where(A_circle(high_res_mesh))[0]
    state_B_idx = np.where(B_circle(high_res_mesh))[0]
    np.save(save_dir + "state_A_idx.npy", state_A_idx)
    np.save(save_dir + "state_B_idx.npy", state_B_idx)
    
    
    downsample_matrix = build_downsampled_matrix(downsampled_mesh, high_res_mesh)

    # To ensure that the state boundaries are well-preserved, we will
    # evaluate which course-grained states are inside the A and B regions
    # and then pull those back to the high-res mesh.
    downsampled_B_labels_mask = B_circle(downsampled_mesh)
    downsampled_A_labels_mask = A_circle(downsampled_mesh)
    downsampled_outside_domain = np.logical_or(
        downsampled_A_labels_mask, downsampled_B_labels_mask
    ).astype("int")

    high_res_B_mask = downsampled_B_labels_mask @ downsample_matrix
    high_res_A_mask = downsampled_A_labels_mask @ downsample_matrix
    high_res_outside_domain = downsampled_outside_domain @ downsample_matrix

    unprojected_reference_Tmat_cond_num, unprojected_reference_Cmat_cond_num = cg.get_raw_Tmat_cond_num(
        reference_Tmat,
        high_res_A_mask,
        high_res_B_mask,
    )
    # Create constant arrays to make it easier to plot
    unprojected_reference_Tmat_cond_nums = np.array(
        [
            unprojected_reference_Tmat_cond_num
            for lag_time in lag_times_list
        ]
    )

    unprojected_reference_Cmat_cond_nums = np.array(
        [
            unprojected_reference_Cmat_cond_num
            for lag_time in lag_times_list
        ]
    )

    projected_reference_Tmat_cond_num, projected_reference_Cmat_cond_num = cg.get_projected_Tmat_cond_num(
        reference_Tmat,
        downsample_matrix,
        high_res_A_mask,
        high_res_B_mask,
    )
    # Create constant arrays to make it easier to plot
    projected_reference_Tmat_cond_nums = np.array(
        [
            projected_reference_Tmat_cond_num
            for lag_time in lag_times_list
        ]
    )

    projected_reference_Cmat_cond_nums = np.array(
        [
            projected_reference_Cmat_cond_num
            for lag_time in lag_times_list
        ]
    )

    ### We're ready to evaluate some committors!
    projected_reference, full_reference = _get_projected_reference(
        reference_Tmat,
        downsample_matrix,
        high_res_A_mask,
        high_res_B_mask,
        measure,
    )

    downsample_committors, lag_times = cg.calculate_naive_committors_linear_lags(
        reference_Tmat,
        downsample_matrix,
        max_lagtime_factor,
        high_res_A_mask,
        high_res_B_mask,
        measure,
    )
    naive_Tmat_cond_nums, naive_Cmat_cond_nums, lag_times = cg.calculate_naive_cond_nums(
        reference_Tmat,
        downsample_matrix,
        max_lagtime_factor,
        high_res_A_mask,
        high_res_B_mask,
        measure,
    )
    np.save("data/downsample_committors.npy", downsample_committors)

    # The lagstop committor requires a stopped transition matrix.
    outside_domain_indices = np.where(high_res_outside_domain == 1)[0]
    stopped_Tmat = cg.build_stopped_P_matrix(reference_Tmat, outside_domain_indices)

    lagstop_ds_committors, lag_times = cg.calculate_lagstop_committors_linear_lags(
        stopped_Tmat,
        downsample_matrix,
        max_lagtime_factor,
        high_res_A_mask,
        high_res_B_mask,
        measure,
    )
    lagstop_Tmat_cond_nums, lagstop_Cmat_cond_nums, lag_times = cg.calculate_lagstop_cond_nums(
        stopped_Tmat,
        downsample_matrix,
        max_lagtime_factor,
        high_res_A_mask,
        high_res_B_mask,
        measure,
    )

    #save condition numbers
    np.save(save_dir + condition_numbers_utils.UNPROJECTED_REFERENCE_TMAT_COND_NUMS_FILE, unprojected_reference_Tmat_cond_nums)
    np.save(save_dir + condition_numbers_utils.UNPROJECTED_REFERENCE_CMAT_COND_NUMS_FILE, unprojected_reference_Cmat_cond_nums)
    np.save(save_dir + condition_numbers_utils.PROJECTED_REFERENCE_TMAT_COND_NUMS_FILE, projected_reference_Tmat_cond_nums)
    np.save(save_dir + condition_numbers_utils.PROJECTED_REFERENCE_CMAT_COND_NUMS_FILE, projected_reference_Cmat_cond_nums)
    np.save(save_dir + condition_numbers_utils.NAIVE_TMAT_COND_NUMS_FILE, naive_Tmat_cond_nums)
    np.save(save_dir + condition_numbers_utils.NAIVE_CMAT_COND_NUMS_FILE, naive_Cmat_cond_nums)
    np.save(save_dir + condition_numbers_utils.LAGSTOP_TMAT_COND_NUMS_FILE, lagstop_Tmat_cond_nums)
    np.save(save_dir + condition_numbers_utils.LAGSTOP_CMAT_COND_NUMS_FILE, lagstop_Cmat_cond_nums)

    np.save(save_dir + "lagstop_ds_committors.npy", lagstop_ds_committors)
    np.save(save_dir + "high_res_mesh.npy", high_res_mesh)
    np.save(save_dir + "downsampled_mesh.npy", downsampled_mesh)
    np.save(save_dir + "high_res_grid.npy", np.stack((X, Y), axis=0))
    np.save(save_dir + "lag_times.npy", lag_times_list)
    np.save(save_dir + "projected_committor_reference.npy", projected_reference)

def plot_errors(data_dir="data/mb/inf-data/", plot_dir="plots/mb/inf-data/"):
    load_matplotlib_presets(cycle_style=CycleStyle.LINES)
    fix_axis_intersection_at_y0()
    downsample_committors = np.load(data_dir + "downsample_committors.npy")
    lagstop_ds_committors = np.load(data_dir + "lagstop_ds_committors.npy")
    lag_times = np.load(data_dir + "lag_times.npy")
    projected_reference = np.load(data_dir + "projected_committor_reference.npy")

    naive_errors = np.array(
        [
            np.sqrt(np.mean((projected_reference - committor) ** 2))
            for committor in downsample_committors
        ]
    )

    lagstop_errors = np.array(
        [
            np.sqrt(np.mean((projected_reference - committor) ** 2))
            for committor in lagstop_ds_committors
        ]
    )
    plt.plot(lag_times, naive_errors, label="Naive", marker="None")
    plt.plot(lag_times, lagstop_errors, label="Stopped Process", marker="None")
    plt.legend()
    plt.xlabel("Lag Time [s]")
    plt.ylabel("Root Mean Square Error")
    plt.savefig(plot_dir + "committor_errors")
    plt.legend()
    plt.close()


def main():
    data_dir = "data/mb/inf-data/committor/"
    plot_dir = "plots/mb/inf-data/committor/"
    # Ensure that the save directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # evaluate_committors_and_cond_nums(save_dir=data_dir)
    plot_errors(data_dir=data_dir, plot_dir=plot_dir)
    condition_numbers_utils.plot_condition_numbers_of_lag_time(
        plot_folder=plot_dir, data_folder=data_dir, lag_times=np.load(data_dir + "lag_times.npy"))


if __name__ == "__main__":
    main()
