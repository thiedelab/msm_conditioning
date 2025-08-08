import numpy as np
from msm_playground.matplotlib_loader import (
    load_matplotlib_presets,
    set_hardcoded_presets,
    CycleStyle,
    fix_axis_intersection_at_y0,
)
import matplotlib.pyplot as plt

set_hardcoded_presets(CycleStyle.MARKERS)
# fix_axis_intersection_at_y0()
load_matplotlib_presets()
plot_folder = "plots/double_well/"
n_clusters = 50
n_traj = 1_000
n_committors_per_point = 1000
traj_length = 20
lag_time = 10
seed = None

error_per_cluster = np.array([0.01, 0.02, 0.03])
abs_error_threshold = n_clusters * error_per_cluster

all_vanilla_sqrerr = np.load(
    f"{plot_folder}_{n_clusters}_{n_traj}_{traj_length}_{lag_time}_{seed}_{n_committors_per_point}_vanilla_sqrerr.npy"
)
all_stopped_sqrerr = np.load(
    f"{plot_folder}_{n_clusters}_{n_traj}_{traj_length}_{lag_time}_{seed}_{n_committors_per_point}_stopped_sqrerr.npy"
)
kT_array = np.load(
    f"{plot_folder}_{n_clusters}_{n_traj}_{traj_length}_{lag_time}_{seed}_kT_scale_array.npy"
)

for threshold in abs_error_threshold:
    share_of_vanilla_better = (
        np.sum(all_vanilla_sqrerr < threshold, axis=1) / n_committors_per_point
    )
    share_of_stopped_better = (
        np.sum(all_vanilla_sqrerr < threshold, axis=1) / n_committors_per_point
    )

    plt.plot(
        kT_array,
        share_of_vanilla_better,
        label=f"Error threshold={round(threshold, 1)}",
        clip_on=True,
        markersize=3.5,
    )
plt.xscale("log")
plt.xlabel("Sampling kT [J]")
plt.ylabel("Share of accurate committors")
plt.legend()
ax = plt.gca()
ax.set_ylim([0, 1])
ax.margins(y=0.05)

# plt.title(f"Share of accurate committors \n{n_committors_per_point} committors per point, {n_traj} trajectories, {traj_length} length, {lag_time} lag time")

plt.savefig(
    f"{plot_folder}committor_error_comparison_{n_clusters}_{n_traj}_{traj_length}_{lag_time}_{seed}"
)
