from sklearn.cluster import kmeans_plusplus
from typing import Any, List, Optional, Tuple, Callable
import numpy as np
import numpy.typing as npt
from msm_playground.clustering.abstract_clustering import AbstractClustering

class KMeansClustering(AbstractClustering):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def __call__(self, traj: List[npt.ArrayLike]) -> npt.NDArray:
        self.cluster_centers, centroids_indices = kmeans_plusplus(traj, n_clusters=self.n_clusters)
        self.cluster_centers = self.cluster_centers[np.argsort(self.cluster_centers[:,0])]
        self.labels = self.assign_clusters(traj)
        self.recalculate_centroids(traj)
        return np.array(self.labels)

    def assign_clusters(self, traj):
        distances = np.linalg.norm(traj[:, np.newaxis] - self.cluster_centers, axis=2)
        return np.argmin(distances, axis=1)
    
    def recalculate_centroids(self, traj):
        self.cluster_centers = np.array([np.mean(traj[self.labels == i], axis=0) for i in range(self.n_clusters)])

    def get_cluster_center(self, label_index):
        return self.cluster_centers[label_index]
    