import numpy as np
import matplotlib.pyplot as plt
from msm_playground.matplotlib_loader import (
    load_matplotlib_presets,
    set_hardcoded_presets,
    CycleStyle,
    fix_axis_intersection_at_y0,
)
from diagnostics.potentials import triple_well_potential, triple_well_force
from scipy.signal import argrelextrema
from msm_playground.grid_based_reference import grid_1D_hopping_process
from msm_playground.clustering.uniform_clustering import UniformClustering
from msm_playground import sampler
from msm_playground.msm import MSM


def _calc_ref_committor_from_Tmat(reference_Tmat, state_A_idx, state_B_idx):
    ref_msm = MSM(
        traj=None,
        clustering_obj=None,
        lag_time=1,
        seconds_between_frames=1,
    )

    ref_msm.Cmat = reference_Tmat
    reference_committor = ref_msm.calculate_committor(
        state_A_index=state_A_idx, state_B_index=state_B_idx, stopped_process=True
    )
    return reference_committor


def _calc_ref_committor(U, dx, kT, state_A_coords, state_B_coords):
    # Transition matrix is I + L dt at short time
    L = grid_1D_hopping_process(U, dx, kT=kT).toarray()
    L = L.astype(np.float64)
    reference_Tmat_lag_time = dx**2 / 8
    reference_Tmat = np.eye(L.shape[0]) + reference_Tmat_lag_time * L

    return _calc_ref_committor_from_Tmat(reference_Tmat, state_A_coords, state_B_coords)


def calc_committors(save_folder):
    # Set up the sampler
    n_traj = 1
    kT = 1.0
    cfg_0 = np.array([0.0])
    nsteps = 1e6
    burnin = 1000
    dt = 1e-4
    time_arr = np.arange(0, nsteps // n_traj * dt, dt)
    upper_box_boundary = 1.0
    lower_box_boundary = -1.0
    box_periodicity_fxn = lambda x: np.clip(x, lower_box_boundary, upper_box_boundary)
    n_clusters = 50
    xx = np.linspace(lower_box_boundary, upper_box_boundary, n_clusters)
    U = triple_well_potential(xx)

    # Find the local minima of U
    local_minima_indices = argrelextrema(U, np.less)[0]
    local_minima_coords = xx[local_minima_indices]
    state_A_coords = local_minima_coords[:1]
    state_B_coords = local_minima_coords[1:]

    plot_potential(
        triple_well_potential, xx, state_A_coords, state_B_coords, "plots/triple_well/"
    )

    # Run the sampler, shooting off n_traj trajectories
    traj = [
        sampler.sample_overdamped_langevin(
            cfg_0,
            triple_well_force,
            nsteps=nsteps // n_traj,
            burnin=burnin,
            dt=dt,
            periodicity_fxn=box_periodicity_fxn,
            kT=kT,
        )
        for i in range(n_traj)
    ]

    lag_time = 250

    uniform_clustering = UniformClustering(n_clusters_per_dimension=n_clusters)
    naive_msm = MSM(traj=traj, clustering_obj=uniform_clustering, lag_time=lag_time)
    lagstop_msm = MSM(traj=traj, clustering_obj=uniform_clustering, lag_time=lag_time)

    naive_clusters = naive_msm.cluster_centers.reshape(-1)
    lagstop_clusters = lagstop_msm.cluster_centers.reshape(-1)
    assert np.allclose(naive_clusters, lagstop_clusters)
    clusters = naive_clusters

    state_A_index = [np.argmin(np.abs(clusters - coord)) for coord in state_A_coords]
    state_B_index = [np.argmin(np.abs(clusters - coord)) for coord in state_B_coords]

    target_committor = _calc_ref_committor(
        U, xx[1] - xx[0], kT, state_A_index, state_B_index
    )

    naive_msm.calculate_correlation_matrix()
    naive_committor = naive_msm.calculate_committor(
        state_A_index=state_A_index, state_B_index=state_B_index, stopped_process=False
    )
    lagstop_msm.calculate_correlation_matrix(
        stopped_process=True, stop_state_index=state_A_index + state_B_index
    )
    lagstop_committor = lagstop_msm.calculate_committor(
        state_A_index=state_A_index, state_B_index=state_B_index, stopped_process=True
    )

    np.save(save_folder + "naive_committor.npy", naive_committor)
    np.save(save_folder + "lagstop_committor.npy", lagstop_committor)
    np.save(save_folder + "target_committor.npy", target_committor)
    np.save(save_folder + "clusters.npy", clusters)
    np.save(save_folder + "state_A_index.npy", np.array(state_A_index))
    np.save(save_folder + "state_B_index.npy", np.array(state_B_index))


def plot_potential(potential_fxn, xx, state_A_coords, state_B_coords, plot_dir):
    load_matplotlib_presets()
    set_hardcoded_presets(CycleStyle.MARKERS)
    fix_axis_intersection_at_y0()

    plt.plot(
        xx, potential_fxn(xx), marker="None", label="Potential energy", linestyle="--"
    )
    plt.plot(
        state_A_coords,
        potential_fxn(state_A_coords),
        linestyle="None",
        markersize=6,
        label="State A",
    )
    plt.plot(
        state_B_coords,
        potential_fxn(state_B_coords),
        linestyle="None",
        markersize=6,
        label="State B",
    )
    plt.legend()
    plt.xlabel("X coordinate")
    plt.ylabel("Potential energy")
    plt.savefig(plot_dir + "triple_well_potential")
    plt.close()


def plot_committors(data_folder, plot_folder):
    load_matplotlib_presets()
    set_hardcoded_presets(CycleStyle.LINES)
    fix_axis_intersection_at_y0()

    naive_committor = np.load(data_folder + "naive_committor.npy")
    lagstop_committor = np.load(data_folder + "lagstop_committor.npy")
    target_committor = np.load(data_folder + "target_committor.npy")
    clusters = np.load(data_folder + "clusters.npy")
    state_A_index = np.load(data_folder + "state_A_index.npy")
    state_B_index = np.load(data_folder + "state_B_index.npy")

    plt.ylabel("Committor")
    plt.xlabel("X coordinate")
    plt.plot(clusters, naive_committor, label="Naive", marker="None")
    plt.plot(
        clusters,
        lagstop_committor,
        label="Stopped Process",
        marker="None",
        color="#d62728",
    )
    # plt.plot(clusters, target_committor, label='Target', marker="None")

    plt.plot(
        clusters[state_A_index],
        [0] * len(state_A_index),
        label="state A",
        markersize=6,
        linestyle="None",
        marker="s",
        color="#ff7f0e",
    )
    plt.plot(
        clusters[state_B_index],
        [1] * len(state_B_index),
        label="state B",
        markersize=6,
        linestyle="None",
        marker="^",
        color="#2ca02c",
    )
    plt.legend(loc="center right")

    plt_name = "Target_vs_naive_vs_lagstop_committors"
    plt.savefig(plot_folder + plt_name)
    plt.close()


if __name__ == "__main__":
    data_folder = "data/triple_well/"
    plot_folder = "plots/triple_well/"
    # calc_committors(data_folder)
    plot_committors(data_folder=data_folder, plot_folder=plot_folder)
