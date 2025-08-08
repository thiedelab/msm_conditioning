import numpy as np
import matplotlib.pyplot as plt

UNPROJECTED_REFERENCE_TMAT_COND_NUMS_FILE = "unprojected_reference_Tmat_cond_nums.npy"
UNPROJECTED_REFERENCE_CMAT_COND_NUMS_FILE = "unprojected_reference_Cmat_cond_nums.npy"
PROJECTED_REFERENCE_TMAT_COND_NUMS_FILE = "projected_reference_Tmat_cond_nums.npy"
PROJECTED_REFERENCE_CMAT_COND_NUMS_FILE = "projected_reference_Cmat_cond_nums.npy"
NAIVE_TMAT_COND_NUMS_FILE = "naive_Tmat_cond_nums.npy"
NAIVE_CMAT_COND_NUMS_FILE = "naive_Cmat_cond_nums.npy"
LAGSTOP_TMAT_COND_NUMS_FILE = "lagstop_Tmat_cond_nums.npy"
LAGSTOP_CMAT_COND_NUMS_FILE = "lagstop_Cmat_cond_nums.npy"

def plot_condition_numbers_of_lag_time(plot_folder: str, data_folder: str, lag_times: np.ndarray = None):
    naive_Tmat_cond_nums = np.load(data_folder + NAIVE_TMAT_COND_NUMS_FILE)
    naive_Cmat_cond_nums = np.load(data_folder + NAIVE_CMAT_COND_NUMS_FILE)
    lagstop_Tmat_cond_nums = np.load(data_folder + LAGSTOP_TMAT_COND_NUMS_FILE)
    lagstop_Cmat_cond_nums = np.load(data_folder + LAGSTOP_CMAT_COND_NUMS_FILE)
    unprojected_ref_cond_nums = np.load(data_folder + UNPROJECTED_REFERENCE_TMAT_COND_NUMS_FILE)
    projected_ref_cond_nums = np.load(data_folder + PROJECTED_REFERENCE_TMAT_COND_NUMS_FILE)
    if lag_times is None:
        lag_times = np.arange(len(naive_Tmat_cond_nums))

    plt.plot(lag_times, naive_Tmat_cond_nums, label="Naive Tmat", marker="None")
    plt.plot(lag_times, lagstop_Tmat_cond_nums, label="Lagstop Tmat", marker="None")
    # plt.plot(lag_times, naive_Cmat_cond_nums, label="Naive Cmat", alpha=0.5, marker="None")
    # plt.plot(lag_times, lagstop_Cmat_cond_nums, label="Lagstop Cmat", alpha=0.5, marker="None")
    plt.plot(lag_times, unprojected_ref_cond_nums, label="Unprojected Reference", marker="None")
    plt.plot(lag_times, projected_ref_cond_nums, label="Projected Reference", marker="None")
    plt.xlabel("Lag Times")
    plt.ylabel("Condition Number")
    plt.yscale("log")
    plt.title("Condition Numbers of Infinite Data Transition Matrices")
    plt.legend()
    plt.savefig(plot_folder + "cond_nums")
    plt.close()

"""
Example of usage:

condition_numbers_utils.plot_matrices(
        [
            (naive_C_T_minus_C_0, "Naive"),
            (lagstop_C_T_minus_C_0, "Lagstop"),
            (proj_ref_C_T_minus_C_0, "Projected Reference"),
            (unproj_ref_C_T_minus_C_0, "Unprojected Reference"),
        ],
        title="C_T - C_0 on domain",
        save_filename="data/combined_matrices",
    )
"""
def plot_correlation_matrices(matrix_legend_list, title="C_T - C_0 on domain", save_filename="combined_matrices", resolution=300):

    vmin = min(np.min(matrix) for matrix, _ in matrix_legend_list)
    vmax = max(np.max(matrix) for matrix, _ in matrix_legend_list)

    num_figure_rows = 2
    num_figure_cols = 2

    fig, axes = plt.subplots(num_figure_rows, num_figure_cols, figsize=(12, 10))

    for i, (matrix, legend_name) in enumerate(matrix_legend_list):
        row = i // num_figure_cols
        col = i % num_figure_cols

        im = axes[row, col].imshow(matrix, cmap="hot", interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[row, col].set_title(legend_name)

    cbar_ax = fig.add_axes([0.1, 0.02, 0.8, 0.02])  # Adjust position for colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', fraction=0.02, pad=0.1)

    fig.suptitle(title)


    plt.savefig(save_filename, dpi=resolution)
    plt.close()
    