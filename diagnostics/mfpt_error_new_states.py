import numpy as np
from numpy import typing as npt
from diagnostics.diagnostics_utils import *
import matplotlib.pyplot as plt
import os
from diagnostics.potentials import MB_potential
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
    measure=None,
):
    """
    Evaluates the fine-grainod mfpt and projects it onto the
    course-grained states.
    """
    if measure is None:
        measure = np.ones(reference_Tmat.shape[0])
        measure /= np.sum(measure)

    mfpt = cg._build_naive_mfpt_from_Tmat(reference_Tmat, state_A)
    projection_matrix = downsample_matrix * measure
    projection_matrix /= np.sum(projection_matrix, axis=1, keepdims=True)
    projected_mfpt = projection_matrix.dot(mfpt)
    return projected_mfpt, mfpt


def evaluate_mfpts(save_dir: str = "data/mb/inf-data/mfpt/"):
    # Simulation Parameters
    kT = 30.0
    n_clusters_per_x_dim = 60
    x_min = -3.0
    x_max = 2.0
    n_clusters_per_y_dim = 60
    y_min = -1.5
    y_max = 3.5
    max_lagtime_factor = 500

    # Ensure that the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
    downsample_matrix = build_downsampled_matrix(downsampled_mesh, high_res_mesh)

    # To ensure that the state boundaries are well-preserved, we will
    # evaluate which course-grained states are inside the A and B regions
    # and then pull those back to the high-res mesh.
    downsampled_B_labels_mask = B_circle(downsampled_mesh)
    downsampled_outside_domain = downsampled_B_labels_mask.astype("int")

    high_res_B_mask = downsampled_B_labels_mask @ downsample_matrix
    high_res_outside_domain = downsampled_outside_domain @ downsample_matrix

    ### We're ready to evaluate some mfpts!
    projected_reference, full_reference = _get_projected_reference(
        reference_Tmat,
        downsample_matrix,
        high_res_B_mask,
        measure,
    )

    downsample_mfpts, lag_times = cg.calculate_naive_mfpts_linear_lags(
        reference_Tmat,
        downsample_matrix,
        max_lagtime_factor,
        high_res_B_mask,
        measure,
    )
    np.save(save_dir + "downsample_mfpts.npy", downsample_mfpts)
    # The lagstop mfpt requires a stopped transition matrix.
    outside_domain_indices = np.where(high_res_outside_domain == 1)[0]
    stopped_Tmat = cg.build_stopped_P_matrix(reference_Tmat, outside_domain_indices)

    lagstop_ds_mfpts, lag_times = cg.calculate_lagstop_mfpts_linear_lags(
        stopped_Tmat,
        downsample_matrix,
        max_lagtime_factor,
        high_res_B_mask,
        measure,
    )
    np.save(save_dir + "lagstop_ds_mfpts.npy", lagstop_ds_mfpts)

    np.save(save_dir + "high_res_mesh.npy", high_res_mesh)
    np.save(save_dir + "downsampled_mesh.npy", downsampled_mesh)
    np.save(save_dir + "high_res_grid.npy", np.stack((X, Y), axis=0))
    np.save(save_dir + "lag_times.npy", lag_times_list)
    np.save(save_dir + "projected_mfpt_reference.npy", projected_reference)


def plot_errors(data_dir: str = "data/mb/inf-data/mfpt/", plot_dir: str = "plots/mb/inf-data/mfpt/"):
    load_matplotlib_presets(cycle_style=CycleStyle.LINES)
    fix_axis_intersection_at_y0()
    downsample_mfpts = np.load(data_dir + "downsample_mfpts.npy")
    lagstop_ds_mfpts = np.load(data_dir + "lagstop_ds_mfpts.npy")
    lag_times = np.load(data_dir + "lag_times.npy")
    projected_reference = np.load(data_dir + "projected_mfpt_reference.npy")

    naive_errors = np.array(
        [
            np.sqrt(np.mean((projected_reference - mfpt) ** 2))
            for mfpt in downsample_mfpts
        ]
    )

    lagstop_errors = np.array(
        [
            np.sqrt(np.mean((projected_reference - mfpt) ** 2))
            for mfpt in lagstop_ds_mfpts
        ]
    )

    fin_tau_idx = int(len(lag_times) * 0.5/0.8)
    lag_times = lag_times[:fin_tau_idx]
    naive_errors = naive_errors[:fin_tau_idx]
    lagstop_errors = lagstop_errors[:fin_tau_idx]

    plt.plot(lag_times, naive_errors, label="Naive", marker="None")
    plt.plot(lag_times, lagstop_errors, label="Stopped Process", marker="None")
    plt.legend()
    plt.xlabel("Lag Time [s]")
    plt.ylabel("Root Mean Square Error [s]")
    plt.savefig(plot_dir + "mfpt_errors")
    plt.close()

def image_plots(data_dir="data/mb/inf-data/mfpt/", plot_dir="plots/mb/inf-data/mfpt/"):
    downsample_mfpts = np.load(data_dir + "downsample_mfpts.npy")
    lagstop_ds_mfpts = np.load(data_dir + "lagstop_ds_mfpts.npy")
    lag_times = np.load(data_dir + "lag_times.npy")
    projected_reference = np.load(data_dir + "projected_mfpt_reference.npy")

    plt.imshow(np.array(projected_reference).reshape((int(projected_reference.shape[0]**0.5), -1)))
    plt.colorbar()
    plt.savefig(plot_dir + "projected_reference")
    plt.close()

    for i in range(1, len(lag_times), 10):
        plt.imshow(np.array(downsample_mfpts[i]).reshape((int(downsample_mfpts[i].shape[0]**0.5), -1)))
        plt.colorbar()
        plt.savefig(plot_dir + "downsample_mfpts_{}".format(i))
        plt.close()

        plt.imshow(np.array(lagstop_ds_mfpts[i]).reshape((int(lagstop_ds_mfpts[i].shape[0]**0.5), -1)))
        plt.colorbar()
        plt.savefig(plot_dir + "lagstop_ds_mfpts_{}".format(i))
        plt.close()

def main():
    data_dir = "data/mb/inf-data/mfpt/"
    plot_dir = "plots/mb/inf-data/mfpt/"
    # Ensure that the save directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # evaluate_mfpts(save_dir=data_dir)
    plot_errors(data_dir=data_dir, plot_dir=plot_dir)
    # image_plots(data_dir=data_dir, plot_dir=plot_dir)


if __name__ == "__main__":
    main()
