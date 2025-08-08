from typing import Any, List
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

class AbstractClustering(ABC):
    def __init__(self):
        self.labels = None
        self.clustering_centers = None

    @abstractmethod
    def __call__(self, traj: List[npt.ArrayLike]) -> npt.NDArray:
        """
        Cluster the trajectory

        Parameters
        ----------
        traj : List[npt.ArrayLike]
            List of frames to cluster of shape (n_frames, n_features). 
            If multiple trajectories are given, concatenate them along the first axis
        
        Returns
        -------
        numpy array
            Cluster labels for each frame in each trajectory
        """
        pass
    
    @abstractmethod
    def get_cluster_center(self, label_index):
        """
        Get the center of the cluster with the given label index

        Parameters
        ----------
        label_index : int
            Index of the cluster

        Returns
        -------
        numpy array
            Center of the cluster
        """
        pass
