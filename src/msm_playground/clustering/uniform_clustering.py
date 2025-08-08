from typing import List
import numpy as np
import numpy.typing as npt
from msm_playground.clustering.abstract_clustering import AbstractClustering

class UniformClustering(AbstractClustering):
    def __init__(self, n_clusters_per_dimension):
        super().__init__()
        self.n_clusters_per_dim = n_clusters_per_dimension
        self.n_clusters = None

    def __call__(self, traj: List[npt.ArrayLike]) -> npt.NDArray:
        self.traj_dim = len(traj[0])
        self.min_values = np.min(traj, axis=0)
        self.max_values = np.max(traj, axis=0)
        
        cube_sizes = (self.max_values - self.min_values) / self.n_clusters_per_dim
        cube_coordinates = ((traj - self.min_values) // cube_sizes).astype(int)
        cube_coordinates = np.clip(cube_coordinates, 0, self.n_clusters_per_dim - 1)

        # Calculate labels for all points
        powers = np.arange(self.traj_dim)
        labels = np.sum(cube_coordinates * (self.n_clusters_per_dim ** powers), axis=1)
        # Cut labels to remove unused clusters
        self.unique_geom_labels, self.labels = np.unique(labels, return_inverse=True)

        # real number of clusters is less than n_clusters_per_dim ** traj_dim
        self.n_clusters = len(self.unique_geom_labels)
        self.cluster_centers = np.array([self.get_cluster_center(i) for i in range(self.n_clusters)])
        return self.labels
    
    def get_cluster_center(self, label_index):
        # Old label index containing information about the cluster coordinates
        geometrical_label_index = self.unique_geom_labels[label_index]
        powers = np.arange(self.traj_dim)

        # Convert the label index to cube coordinates
        cube_coordinates = []
        for power in reversed(powers):
            div, geometrical_label_index = divmod(geometrical_label_index, self.n_clusters_per_dim ** power)
            cube_coordinates.append(div)
        cube_coordinates = np.array(cube_coordinates[::-1])

        # Calculate the central position of the cube
        cube_sizes = (self.max_values - self.min_values) / self.n_clusters_per_dim
        cube_center = self.min_values + cube_coordinates * cube_sizes + cube_sizes / 2

        return cube_center
    