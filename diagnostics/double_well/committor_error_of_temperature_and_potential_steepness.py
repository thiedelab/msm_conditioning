from msm_playground.grid_based_reference import grid_1D_hopping_process
from msm_playground.msm import MSM
from diagnostics.potentials import double_well_potential
from diagnostics.diagnostics_utils import clusters_colors
from msm_playground import numba_sampler as sampler
from msm_playground.traj_utils import sample_from_grid
from msm_playground.clustering.uniform_clustering import UniformClustering

from msm_playground.matplotlib_loader import (
    load_matplotlib_presets,
    set_hardcoded_presets,
    CycleStyle,
    fix_axis_intersection_at_y0,
)

import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse as sps
import matplotlib.pyplot as plt
from numba import njit
from typing import List, Tuple
import gc
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--n-clusters", type=int, default=50, help="Number of clusters")
parser.add_argument("--n-traj", type=int, default=1_000, help="Number of trajectories")
parser.add_argument(
    "--n-committors-per-t-point",
    type=int,
    default=1_000,
    help="Number of committors per temperature point",
)
parser.add_argument(
    "--traj-length", type=int, default=20, help="Length of each trajectory"
)
parser.add_argument(
    "--lag-time",
    type=int,
    default=10,
    help="committor function is benchmarked on lag time from min-lat-time to max-lag-time",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed for random number generator"
)
args = parser.parse_args()

n_clusters = args.n_clusters
n_traj = args.n_traj
n_committors_per_point = args.n_committors_per_t_point
traj_length = args.traj_length
lag_time = args.lag_time
seed = args.seed
burnin = 0
kT = 1.0
dt = 1e-3 / 4
xgridmax = 1.0

# If you change the potential coefficients, you need to change the hardcoded scaling in numba_sampler.double_well_force as well
potential_scale = 5.0 * 2
plot_folder = "plots/double_well/"

potential_coeffs = np.array([1.0, 1.0, 0.2])
a, b, c = potential_coeffs
critical_points = np.roots([4 * a, 0, -2 * b, c])
well_1, well_2 = np.min(critical_points), np.max(critical_points)
uniform_clustering = UniformClustering(n_clusters_per_dimension=n_clusters)


def _build_ref_committor_from_Tmat(reference_Tmat, state_A, state_B):
    ref_msm = MSM(
        traj=None,
        clustering_obj=None,
        lag_time=1,
        seconds_between_frames=1,
    )

    ref_msm.Tmat = reference_Tmat
    reference_committor = ref_msm.calculate_committor(
        state_A_index=state_A, state_B_index=state_B
    )
    return reference_committor


def get_ref_committor(U, dx, state_A, state_B, kT):
    L = grid_1D_hopping_process(U, dx, kT)
    ref_Tmat = sps.identity(L.shape[0], format=L.format) + dx**2 / 8 * L

    ref_committor = _build_ref_committor_from_Tmat(ref_Tmat.toarray(), state_A, state_B)
    return ref_committor


def get_boltzmann_prob(U, kT):
    boltzmann_prob = np.exp(-U / kT)
    boltzmann_prob /= np.sum(boltzmann_prob)
    return boltzmann_prob


@njit
def is_outside_of_grid(coord: np.ndarray) -> bool:
    # unfortunately needs to be hardcoded for njit
    grid_center = 0.0

    return not np.all(np.abs(coord - grid_center) < xgridmax)


@njit
def simulate_trajectories(
    xgrid: ArrayLike,
    kT: float,
    sampling_prob: ArrayLike,
    seed: float,
    traj_length: float,
    dt: float,
    n_traj: int,
    burnin: int = 1000,
) -> List[ArrayLike]:
    if seed is not None:
        np.random.seed(seed)
        initial_state = np.random.get_state()
        np.random.set_state(initial_state)
        sampler.set_seed(seed)

    cfg_0 = np.array([sample_from_grid(grid=xgrid, probabilities=sampling_prob)])
    trajectories = np.zeros((n_traj, traj_length, 1))
    for traj_idx in range(n_traj):
        trajectories[traj_idx, :] = sampler.sample_overdamped_langevin(
            cfg_0,
            sampler.double_well_force,
            nsteps=traj_length,
            burnin=burnin,
            dt=dt,
            kT=kT,
            reject_function=is_outside_of_grid,
        )

    return trajectories


def get_exp_vanilla_committor(
    trajectories: List[ArrayLike], lag_time: int, state_A: List[int], state_B: List[int]
) -> ArrayLike:
    exp_vanilla_msm = MSM(
        trajectories, uniform_clustering, lag_time, dt, enforce_irreducibility=False
    )
    exp_vanilla_msm.calculate_correlation_matrix(stopped_process=False)
    committor = exp_vanilla_msm.calculate_committor(state_A, state_B)
    return committor


def get_exp_stopped_committor(
    trajectories: List[ArrayLike], lag_time: int, state_A: List[int], state_B: List[int]
) -> ArrayLike:
    exp_stopped_msm = MSM(
        trajectories, uniform_clustering, lag_time, dt, enforce_irreducibility=False
    )
    exp_stopped_msm.calculate_correlation_matrix(
        stopped_process=True, stop_state_index=np.concatenate([state_A, state_B])
    )

    committor = exp_stopped_msm.calculate_committor(
        state_A, state_B, stopped_process=True
    )
    Tmat = exp_stopped_msm.Tmat
    assert Tmat[state_A, state_A] == 1
    assert Tmat[state_B, state_B] == 1
    assert np.sum(Tmat[state_A, :]) == 1
    assert np.sum(Tmat[state_B, :]) == 1
    assert np.all(Tmat >= 0)
    assert np.all(Tmat <= 1)
    np.save(f"{plot_folder}Tmat.npy", Tmat)
    np.save(f"{plot_folder}state_A_idx.npy", state_A)
    np.save(f"{plot_folder}state_B_idx.npy", state_B)
    return committor


def get_weighted_sqrerror(
    exp_committor: ArrayLike, ref_committor: ArrayLike, boltzmann_prob: ArrayLike
) -> float:
    # weights correspond to the probability that a trajectory going through a given point is reactive
    weights = np.ones_like(exp_committor)
    sqrerror = (exp_committor - ref_committor) ** 2
    return np.sum(sqrerror * weights) ** 0.5


def plot_sampling_prob_and_trajectories(
    sampling_prob, xgrid, trajectories, title_suffix=""
):
    plt.plot(xgrid, sampling_prob)
    for traj_idx, traj in enumerate(trajectories):
        illusion_of_time_axis = np.linspace(
            np.min(sampling_prob), np.max(sampling_prob), traj.size
        )
        plt.plot(
            traj,
            illusion_of_time_axis,
            marker="o",
            color=clusters_colors[traj_idx % len(clusters_colors)],
            alpha=0.6,
            markersize=0.1,
            linewidth=0.1,
        )
    plt.title(f"Sampling probability and trajectories {title_suffix}")
    plt.savefig(f"{plot_folder}sampling_prob_and_trajectories_{title_suffix}")
    plt.close()


def plot_committor(
    xgrid, ref_committor, exp_vanilla_committor, exp_stopped_committor, title_suffix=""
):
    plt.plot(xgrid, ref_committor, label="Reference")
    plt.plot(xgrid, exp_vanilla_committor, label="Vanilla")
    plt.plot(xgrid, exp_stopped_committor, label="Stopped")
    plt.title(f"Committor comparison {title_suffix}")
    plt.legend()
    plt.savefig(f"{plot_folder}committor_{title_suffix}")
    plt.close()


def get_committor_errors_on_double_well(
    kT_scale: float, potential_scale: float, plot: bool = False
) -> Tuple[float, float]:
    gc.collect()

    U, xgrid = double_well_potential(
        dots_per_x_dim=n_clusters, x_min=-xgridmax, x_max=+xgridmax, a=a, b=b, c=c
    )
    U *= potential_scale
    dx = xgrid[1] - xgrid[0]
    state_A = [np.argmin(np.abs(xgrid - well_1))]
    state_B = [np.argmin(np.abs(xgrid - well_2))]
    plt.plot(xgrid, U, marker="None", label="Potential energy", linestyle="--")
    plt.plot(
        xgrid[state_A], U[state_A], linestyle="None", markersize=6, label="State A"
    )
    plt.plot(
        xgrid[state_B], U[state_B], linestyle="None", markersize=6, label="State B"
    )
    plt.legend()
    plt.xlabel("X coordinate")
    plt.ylabel("Potential energy")
    plt.savefig(f"{plot_folder}potential_energy_potential_scale={potential_scale}.jpg")
    plt.close()

    ref_committor = get_ref_committor(U, dx, state_A, state_B, kT)

    boltzmann_prob = get_boltzmann_prob(U, kT * kT_scale)

    trajectories = simulate_trajectories(
        xgrid, kT, boltzmann_prob, seed, traj_length, dt, n_traj
    )
    test_msm = MSM(
        trajectories, uniform_clustering, lag_time, dt, enforce_irreducibility=False
    )

    # If the trajectories don't visit all the states, we need to add some trajectories that stay in one place
    if len(np.squeeze(test_msm.cluster_centers)) != len(xgrid) or not np.allclose(
        np.squeeze(test_msm.cluster_centers), xgrid, atol=1 / n_clusters
    ):
        never_visited_x = np.setdiff1d(xgrid, np.unique(trajectories))
        stay_on_place_trajectories = [
            np.full((lag_time + 1, 1), never_x) for never_x in never_visited_x
        ]
        top_level_list_from_trajectories = [subarray for subarray in trajectories]
        trajectories = top_level_list_from_trajectories + stay_on_place_trajectories

    test_msm = MSM(
        trajectories, uniform_clustering, lag_time, dt, enforce_irreducibility=False
    )
    assert np.allclose(np.squeeze(test_msm.cluster_centers), xgrid, atol=1 / n_clusters)

    kT_str = "{:e}".format(kT_scale)
    title_suffix = f"_kT_scale={kT_str}_potscale={potential_scale}_seed={seed}"

    exp_vanilla_committor = get_exp_vanilla_committor(
        trajectories, lag_time, state_A, state_B
    )
    exp_stopped_committor = get_exp_stopped_committor(
        trajectories, lag_time, state_A, state_B
    )

    if plot:
        plot_sampling_prob_and_trajectories(
            boltzmann_prob, xgrid, trajectories, title_suffix=title_suffix
        )
        plot_committor(
            xgrid,
            ref_committor,
            exp_vanilla_committor,
            exp_stopped_committor,
            title_suffix=title_suffix,
        )

    vanilla_sqrerr, stopped_sqrerr = (
        get_weighted_sqrerror(exp_vanilla_committor, ref_committor, boltzmann_prob),
        get_weighted_sqrerror(exp_stopped_committor, ref_committor, boltzmann_prob),
    )

    return vanilla_sqrerr, stopped_sqrerr


def main():
    load_matplotlib_presets()
    set_hardcoded_presets(CycleStyle.MARKERS)
    fix_axis_intersection_at_y0()

    kT_scale_array = np.logspace(-2.0, 3.0, 40)
    print(f"kT_array: {kT_scale_array}")
    all_vanilla_sqrerr = np.zeros((len(kT_scale_array), n_committors_per_point))
    all_stopped_sqrerr = np.zeros((len(kT_scale_array), n_committors_per_point))

    for kT_scale_idx, kT_scale in tqdm(enumerate(kT_scale_array)):
        for committor_iteration in range(n_committors_per_point):
            vanilla_sqerr, stopped_sqerr = get_committor_errors_on_double_well(
                kT_scale, potential_scale=potential_scale
            )
            all_vanilla_sqrerr[kT_scale_idx, committor_iteration] = vanilla_sqerr
            all_stopped_sqrerr[kT_scale_idx, committor_iteration] = stopped_sqerr

    np.save(
        f"{plot_folder}_{n_clusters}_{n_traj}_{traj_length}_{lag_time}_{seed}_{n_committors_per_point}_vanilla_sqrerr.npy",
        all_vanilla_sqrerr,
    )
    np.save(
        f"{plot_folder}_{n_clusters}_{n_traj}_{traj_length}_{lag_time}_{seed}_{n_committors_per_point}_stopped_sqrerr.npy",
        all_stopped_sqrerr,
    )
    np.save(
        f"{plot_folder}_{n_clusters}_{n_traj}_{traj_length}_{lag_time}_{seed}_kT_scale_array.npy",
        kT_scale_array,
    )


if __name__ == "__main__":
    main()
