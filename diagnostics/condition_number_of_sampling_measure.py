import numpy as np
import matplotlib.pyplot as plt
from msm_playground.matplotlib_loader import load_matplotlib_presets, set_hardcoded_presets, CycleStyle, fix_axis_intersection_at_y0
import numpy.typing as npt
from typing import Literal

def calc_cond(Tmat_on_domain: np.ndarray, sampling_measure: npt.ArrayLike, chosen_norm: float | Literal['fro', 'nuc'] = 1) -> float:
    """Calculate the condition number of the transition matrix on a domain. Assume that the transition matrix doesn't depend on the sampling measure.
    If you give a reference Tmat as an argument, you get a lower bound of the condition number.

    Parameters
    ----------
    Tmat_on_domain : np.ndarray
        Transition matrix on a domain.
    sampling_measure : npt.ArrayLike
        Sampling measure on the domain.
    chosen_norm : float | Literal['fro', 'nuc'], optional
        Norm to calculate the condition number, by default 1, which is max(sum(abs(x), axis=0))

    Returns
    -------
    float
        Condition number of the transition matrix on the domain.
    """

    L = Tmat_on_domain - np.eye(Tmat_on_domain.shape[0])
    L = np.diag(np.sqrt(sampling_measure)) @ L
    return np.linalg.cond(L, p=1)

def stationary_distribution(transition_matrix):
    # Transpose the matrix to work with left eigenvectors
    transposed_matrix = transition_matrix.T
    eigenvalues, eigenvectors = np.linalg.eig(transposed_matrix)
    idx = np.isclose(eigenvalues, 1)
    stationary_vector = np.real(eigenvectors[:, idx]).flatten()
    stationary_vector = stationary_vector / np.sum(stationary_vector)
    
    return stationary_vector


def main():
    data_dir = "data/mb/"
    Tmat = np.load(data_dir + "reference_Tmat.npy")
    state_A_idx = np.load(data_dir + "state_A_idx.npy")
    state_B_idx = np.load(data_dir + "state_B_idx.npy")
    domain_idx = np.setdiff1d(np.arange(Tmat.shape[0]), state_A_idx)
    domain_idx = np.setdiff1d(domain_idx, state_B_idx)

    Tmat_on_domain = Tmat[np.ix_(domain_idx, domain_idx)]        
    boltzmann = stationary_distribution(Tmat)[domain_idx]
    uniform = np.ones(Tmat_on_domain.shape[0]) / Tmat_on_domain.shape[0]

    homotopy_parameters = np.linspace(0, 1.5, 100)
    sampling_measure = np.array([boltzmann * (1 - t) + uniform * t for t in homotopy_parameters])
    sampling_measure = np.array([measure / np.sum(measure) for measure in sampling_measure])

    cond = np.array([calc_cond(Tmat_on_domain, mu) for mu in sampling_measure])
    np.save(data_dir + "condition_number_of_sampling_measure.npy", cond)
    np.save(data_dir + "homotopy_parameters.npy", homotopy_parameters)
    
def plot(plots_dir = "plots/sampling_measure/"):
    set_hardcoded_presets(CycleStyle.LINES)
    # fix_axis_intersection_at_y0()
    load_matplotlib_presets()

    homotopy_parameters = np.load("data/mb/homotopy_parameters.npy")
    cond = np.load("data/mb/condition_number_of_sampling_measure.npy")

    plt.plot(homotopy_parameters, cond, marker="None")
    plt.xlabel(r"Mixing parameter $\alpha$")
    plt.ylabel("Condition number")
    plt.savefig(plots_dir + "condition_number_of_sampling_measure")

if __name__ == "__main__":
    # main()
    plot()
    