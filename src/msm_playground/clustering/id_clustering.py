from typing import List
import numpy as np
import numpy.typing as npt
from msm_playground.clustering.abstract_clustering import AbstractClustering

class IdentityClustering(AbstractClustering):
    """
    Identity clustering class for testing purposes. Creates labels that are identical to trajectory points.
    Deals only with integer-valued 0-indexed trajectory points.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, traj: List[int]) -> npt.NDArray:
        unique_points = np.unique(traj, axis=0)
        if (np.max(unique_points) != len(unique_points) - 1 or np.min(unique_points) != 0):
            raise ValueError("Identity clustering requires integer-valued 0-indexed trajectory points")

        self.n_clusters = len(traj)
        self.cluster_centers = unique_points
        self.labels = traj
        return self.labels
    
    def get_cluster_center(self, label_index):
        return label_index
    