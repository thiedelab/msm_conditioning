from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from typing import List
import numpy as np
import numpy.typing as npt
from msm_playground.clustering.abstract_clustering import AbstractClustering

class WardClustering(AbstractClustering):
    """
    Wrapper class for structured Ward hierarchical clustering
    """
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def __call__(self, traj: List[npt.ArrayLike]) -> npt.NDArray:
        ward = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
        ward.fit(traj)
        self.labels = ward.labels_
        traj_dim = traj[0].shape[-1]
        self.cluster_centers = np.zeros((self.n_clusters, traj_dim))
        for cluster_idx in range(self.n_clusters):
            self.cluster_centers[cluster_idx, :] = np.mean(traj[self.labels == cluster_idx, :], axis=0)

        return self.labels