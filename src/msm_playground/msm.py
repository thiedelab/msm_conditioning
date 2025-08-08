from typing import Any, List, Optional, Tuple, Callable
from msm_playground.traj_utils import merge_trajectories, convert_to_zero_indexed
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.optimize import lsq_linear
import pyemma
from msm_playground.clustering.abstract_clustering import AbstractClustering

class MSM():
    """
    Base class for MSMs
    """
    def __init__(self, traj: List[npt.ArrayLike], 
                 clustering_obj: AbstractClustering,
                 lag_time: int,
                 seconds_between_frames: Optional[float]=1.,
                 custom_labels: Optional[List[npt.NDArray]]=None,
                 enforce_irreducibility: bool=True):
        self.traj = traj
        self.traj_mask = None
        self.init_trajs_lengths = None
        if(traj is not None):
            self.init_trajs_lengths = [len(t) for t in traj]
            self.traj, self.traj_mask = merge_trajectories(traj, lag_time)
        self.clustering_obj = clustering_obj
        self.lag_time = int(lag_time)
        self.traj_time_step = seconds_between_frames
        self.Cmat = None
        # By convention, labels are 0-indexed, integers
        self.labels = custom_labels
        self.markov_generator = None
        self.stationary_distribution = None
        self.committor = None
        self._backward_committor = None
        self.lagstop_committor = None
        self.eigenvalues = None
        self.eigenvectors = None
        # Can be different from clustering_obj.cluster_centers if some states are never visited
        # Never use clustering_obj.cluster_centers directly, use self.cluster_centers instead
        self.cluster_centers = None

        # Reserve a copy of the original traj-related arrays, in case they are modified
        self._untouched_traj = MSM.custom_copy(self.traj)
        self._untouched_traj_mask = MSM.custom_copy(self.traj_mask)
        self._untouched_labels = MSM.custom_copy(self._labels)

        self.labels_value_map = None
        if(traj is not None and enforce_irreducibility):
            self.cut_reducible_states()

    @staticmethod
    def custom_copy(array):
        if array is None:
            return None
        else:
            return np.copy(array)
        
    @property
    def Tmat(self):
        if(self.Cmat is None):
            raise ValueError("Cannot calculate transition matrix, correlation matrix is not calculated")
        row_sums = np.sum(self.Cmat, axis=1, keepdims=True)
        return self.Cmat / row_sums
    
    @Tmat.setter
    def Tmat(self, value):
        if(self.Cmat is not None):
            print("Warning: Correlation matrix has already been set before. Overwriting it.")
        self.Cmat = value
        
    @property
    def lag_time_in_sec(self):
        return self.lag_time * self.traj_time_step

    @property
    def untouched_traj(self):
        return self._untouched_traj
    
    @property
    def untouched_traj_mask(self):
        return self._untouched_traj_mask
    
    @property
    def untouched_labels(self):
        return self._untouched_labels

    @property
    def cluster_centers(self):
        if(self._cluster_centers is None):
            if(self.labels is None):
                self.labels = self.clustering_obj(self.traj)
            self._cluster_centers = self.clustering_obj.cluster_centers
        return self._cluster_centers

    @cluster_centers.setter
    def cluster_centers(self, value):
        self._cluster_centers = value

    @property
    def n_states(self):
        if(self.Cmat is not None):
            return self.Cmat.shape[0]
        if(self.markov_generator is not None):
            return self.markov_generator.shape[0]
        if(self.labels is not None):
            return np.unique(self.labels).shape[0]
        # Can contain never left states, not trustful!
        if(self.clustering_obj is not None):
            return self.clustering_obj.n_clusters
        raise ValueError("Cannot determine number of states")
        
    @property
    def split_labels(self):
        # Returns list of nparrays of labels, split by self.init_trajs_lengths
        if(self.init_trajs_lengths is None):
            return None
        return np.split(self.labels, np.cumsum(self.init_trajs_lengths)[:-1])
    
    @property
    def labels(self):
        if(self._labels is None):
            if(self.clustering_obj is None):
                raise ValueError("Cannot determine labels, no clustering object was provided")
            self._labels = self.clustering_obj(self.traj)
            if(self._untouched_labels is None):
                self._untouched_labels = np.copy(self._labels)
        return self._labels

    @labels.setter
    def labels(self, value):
        if(value is None):
            self._labels = None
            return
        # Update mask when labels are updated
        if isinstance(value, np.ndarray):
            self._labels, self.traj_mask = merge_trajectories([value], self.lag_time)
            return
        # if value is split into multiple trajectories, merge them, update mask
        if isinstance(value, list):
            self._labels, self.traj_mask = merge_trajectories(value, self.lag_time)
            return
        raise TypeError("labels must be None, np.ndarray or List[np.ndarray]")
    
    @property
    def backward_committor(self):
        if(self._backward_committor is not None):
            return self._backward_committor
        
        return 1 - self.committor
    
    def is_reversible(self) -> bool:
        """
        Checks if the MSM is reversible
        """
        if(self.Cmat is None):
            self.calculate_correlation_matrix()
        pi = self.calculate_stationary_distribution()
        if(pi is None):
            print("Cannot calculate stationary distribution, cannot check reversibility")
            return False
        return np.allclose(self.Cmat * pi, self.Cmat.T * pi, atol=1e-1)
    
    def old_to_new_label(self, label_value: int) -> int:
        """
        Transforms a label value of original clustering to a label value of the current clustering
        If no value_map is provided, the label value is returned
        """
        if(self.labels_value_map is None):
            return label_value
        try:
            return self.labels_value_map[label_value]
        except KeyError:
            raise ValueError("No mapping for label value", label_value,
                              "Probably one of committor reactant or product states was cut out")

    def calculate_correlation_matrix(self, stopped_process: bool=False,
                                    stop_state_index: Optional[List[int]] = None) -> np.ndarray:
        if(stopped_process):
            self.Cmat = self.calculate_lagstop_correlation_matrix(stop_state_index)
            return self.Cmat
        
        if(np.min(self.labels) != 0 or np.max(self.labels) > self.n_states - 1):
            print("min:", np.min(self.labels), "max:", np.max(self.labels), "n_states:", self.n_states)
            raise ValueError("Labels must be 0-indexed integers")
        
        n_frames = len(self.labels)
        if(self.labels.shape != (n_frames,)):
            raise ValueError("Labels must be a 1D array of length n_frames")
        
        num_states = np.max(self.labels) + 1
        one_hot_labels = csr_matrix(([1] * n_frames, (range(n_frames), self.labels)), shape=(n_frames, num_states), dtype=np.int32)

        # Find the indices of the transition pairs satisfying the trajectory mask
        indices = np.where(self.traj_mask[:-self.lag_time])[0].astype(np.int32)

        # Get the one-hot vectors for the transition pairs
        source_vectors = one_hot_labels[indices].T
        target_vectors = one_hot_labels[indices + self.lag_time]

        # Compute the transition counts
        Cmat = source_vectors.dot(target_vectors).astype(np.int32)
        Cmat = Cmat.toarray().astype(np.float32)
        
        self.Cmat = Cmat
        return Cmat

    def cut_reducible_states(self):
        """
            Cuts reducible states proposed by pyemma.
            Pyemma returns largest reversibly connected set by default.
            Call in initialize method before calculating transition matrix, but after merging trajectories.
        """
        def get_bad_states()-> List[int]:
            try:
                self.active_set = pyemma.msm.estimate_markov_model(self.split_labels,
                                                                        self.lag_time, reversible=False).active_set
            except Exception as e:
                print("Pyemma error:", e)
                print("Can't cut reducible states, returning None")
                return None
            states_to_cut = np.setdiff1d(np.unique(self.labels), self.active_set)
            return states_to_cut

        if(self.split_labels is None):
            # Can't reduce states if no initial trajectories are provided
            return

        cut_rows = get_bad_states()
        if(cut_rows is None):
            return

        old_labels = np.copy(self._labels)
        self.labels_time_mask = np.ones_like(old_labels, dtype=bool)
        self.labels_time_mask &= ~(old_labels[:, None] == cut_rows).any(axis=1)

        new_labels = self._labels[self.labels_time_mask]
        # remove gaps between labels, save the mapping for later
        # value_map will be used to reassign state_A/B_idx for committor calculation
        self._labels, relative_labels_value_map = convert_to_zero_indexed(new_labels, value_map=True)
        if(self.labels_value_map is None):
            self.labels_value_map = relative_labels_value_map
        else:
            # compose the value maps
            self.labels_value_map = {k: relative_labels_value_map.get(v, v) for k, v in self.labels_value_map.items()}

        if(self.clustering_obj is not None):
            self.cluster_centers = self.cluster_centers[~np.isin(np.arange(self.cluster_centers.shape[0]), cut_rows)]
        if(self.traj is not None):
            self.traj = self.traj[self.labels_time_mask]
        if(self.traj_mask is not None):
            self.traj_mask = self.traj_mask[self.labels_time_mask]
        
    def calculate_generator(self):
        if self.Cmat is None:
            self.calculate_correlation_matrix()

        if type(self.Tmat) is csr_matrix:
            Tmat = self.Tmat.toarray()
        else:
            Tmat = self.Tmat
        self.markov_generator = (Tmat - np.identity(Tmat.shape[0])) / self.lag_time_in_sec
        return self.markov_generator

    def derive_equation_for_committor(self, state_A_index: List[int], state_B_index: List[int], stopped_process: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Left-Hand Side (LHS) matrix and Right-Hand Side (RHS) vector for committor B before A between two states equation.

        .. math::

            A \cdot \mathbf{q} = \mathbf{r}

        where:
        - :math:`A` is the matrix to be returned,
        - :math:`\mathbf{q}` is the committor probability vector,
        - :math:`\mathbf{r}` is the RHS vector to be returned.

        Parameters
        ----------
        state_A_index : List[int]
            List of indices of states in state A
        state_B_index : List[int]
            List of indices of states in state B
        stopped_process : bool, optional
            If True, calculate committor for stopped process, by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the LHS matrix A and RHS vector r of the committor equation.
        """

        if(stopped_process):
            self.Cmat = self.calculate_lagstop_correlation_matrix(
                stop_state_index=np.concatenate((state_A_index, state_B_index)))
        if self.Cmat is None:
            self.calculate_correlation_matrix()

        state_A_index = self.old_to_new_reactant_states(state_A_index)
        state_B_index = self.old_to_new_reactant_states(state_B_index)

        n_unknown_variables = self.n_states - len(state_A_index) - len(state_B_index)
        
        unknown_variables_index = [i for i in range(self.n_states) if i not in state_A_index and i not in state_B_index]
        C_0 = np.identity(self.n_states) * np.sum(self.Cmat, axis=1)
        C_T_minus_C_0 = self.Cmat - C_0
        equation_coefficients = C_T_minus_C_0[np.ix_(unknown_variables_index, unknown_variables_index)]
        values_vector = np.zeros(n_unknown_variables)
        values_vector -= np.sum(C_T_minus_C_0[np.ix_(unknown_variables_index, state_B_index)], axis=1)

        return equation_coefficients, values_vector
    
    def calculate_full_committor_from_committor_on_domain(self, committor_on_domain: np.ndarray, state_A_index: List[int], state_B_index: List[int]) -> np.ndarray:
        """
        Calculate the committor B before A between two states from the committor equation.

        Parameters
        ----------
        committor_on_domain : np.ndarray
            The committor B before A between two states on the domain (on all clusters excluding the boundary conditions).
        state_A_index : List[int]
            List of indices of states in state A
        state_B_index : List[int]
            List of indices of states in state B

        Returns
        -------
        np.ndarray
            The committor B before A between two states on the complete domain.
        """
        state_A_index = self.old_to_new_reactant_states(state_A_index)
        state_B_index = self.old_to_new_reactant_states(state_B_index)

        unknown_variables_index = [i for i in range(self.n_states) if i not in state_A_index and i not in state_B_index]
        full_committor = np.zeros(self.n_states)
        full_committor[state_A_index] = 0 # committor is 0 in states A, unnecessary line for clarity
        full_committor[state_B_index] = 1 # committor is 1 in states B
        full_committor[unknown_variables_index] = committor_on_domain

        self.committor = full_committor
        return full_committor


    def calculate_committor(self, state_A_index: List[int], state_B_index: List[int], stopped_process: bool=False) -> np.ndarray:
        """
        Calculate the committor B before A between two states.
        For disconnected states, committor outputs some number between 0 and 1, so that
        the whole committor vector satisfies the time-delayed equation for committor.

        Parameters
        ----------
        state_A_index : List[int]
            List of indices of states in state A
        state_B_index : List[int]
            List of indices of states in state B
        stopped_process : bool, optional
            If True, calculate committor for stopped process, by default False
        """
        equation_coefficients, values_vector = self.derive_equation_for_committor(state_A_index, state_B_index, stopped_process)
        
        res = lsq_linear(equation_coefficients, values_vector, bounds=(-1e6, 1 + 1e6), max_iter=1000)
        if(not res.success):
            raise ValueError("Committor calculation failed. Reason: " + res.message + " Iterations: ", res.nit)
        cut_committor = np.clip(res.x, 0, 1)

        full_committor = self.calculate_full_committor_from_committor_on_domain(cut_committor, state_A_index, state_B_index)

        self.committor = full_committor
        return full_committor
    
    def old_to_new_reactant_states(self, old_reactant: List[int]) -> List[int]:
        if(self.labels_value_map is not None):
            survived_labels = np.array(list(self.labels_value_map.keys()))
            old_reactant = np.intersect1d(old_reactant, survived_labels, assume_unique=True)

        old_reactant = [self.old_to_new_label(react_element) for react_element in old_reactant]
        return old_reactant
    
    def calculate_lagstop_correlation_matrix(self, stop_state_index: List[int]) -> np.ndarray:
        """
        Calculates stopped process transition matrix.
        Assign lagstop transition matrix before calculating any properties of the stopped process model.
        """

        # MSM was initiated with Tmat
        if((not hasattr(self, '_labels') or self._labels is None)
           and self.Cmat is not None):
            self.Cmat[stop_state_index, :] = 0
            self.Cmat[stop_state_index, stop_state_index] = 1
            return self.Cmat

        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        num_states = len(unique_labels)
        Cmat = np.zeros((num_states, num_states))

        if(self.labels_value_map is not None):
            survived_labels = np.array(list(self.labels_value_map.keys()))
            stop_state_index = np.intersect1d(stop_state_index, survived_labels, assume_unique=True)
        stop_state_index = [self.old_to_new_label(stop_state_element) for stop_state_element in stop_state_index]
        # Modify labels for committor calculation, so that trajectory stops at A or B state, if it is encountered within lag time.
        # Calculate transition matrix for modified self.labels in same loop.
        for i in range(len(self.labels) - self.lag_time):
            # Skip if traj_mask is False
            if self.traj_mask is not None and not self.traj_mask[i]:
                continue
            if np.any(np.isin(self.labels[i+1:i+self.lag_time+1], stop_state_index)):
                next_label = self.labels[i + 1 + np.isin(self.labels[i+1:i+self.lag_time+1], stop_state_index).nonzero()[0][0]]
            else:
                next_label = self.labels[i+self.lag_time]
            current_state = self.labels[i]
            next_state = next_label
            Cmat[current_state, next_state] += 1

        Cmat[stop_state_index, :] = 0
        Cmat[stop_state_index, stop_state_index] = 1

        self.Cmat = Cmat
        return Cmat

    def calculate_evecs(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Calculate the eigenvectors of the transition matrix

        Parameters
        ----------
        n_eigenvectors : int or None
            Number of eigenvectors to calculate. If None, calculate all eigenvectors

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray]
            Tuple of eigenvalues and eigenvectors
        """
        if self.Cmat is None:
            self.calculate_correlation_matrix()
        
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.Cmat.T)
        return self.eigenvalues, self.eigenvectors
        
    def reset_model(self):
        """
        Reset the model to its initial state
        """
        self.Cmat = None
        self.labels = MSM.custom_copy(self.untouched_labels)
        self.traj = MSM.custom_copy(self.untouched_traj)
        self.traj_mask = MSM.custom_copy(self.untouched_traj_mask)
        self.cut_reducible_states()
        self.markov_generator = None


    def calculate_stationary_distribution(self):
        if(self.markov_generator is None or self.Cmat is None):
            self.calculate_generator()

        transposed_matrix = np.transpose(self.markov_generator)
        eigenvalues, eigenvectors = np.linalg.eig(transposed_matrix)
        zero_eigenvecs = np.isclose(eigenvalues, 0, atol=1.e-7)
        if(len(zero_eigenvecs.nonzero()[0]) == 0):
            print("MSM Warning: can't find stationary distribution, no eigenvalues are close to 0")
            return None
        stationary_index = np.where(np.isclose(eigenvalues, 0, atol=1.e-7))[0][0]
        stationary_vector = np.real(eigenvectors[:, stationary_index])
        stationary_vector /= np.sum(stationary_vector)
        
        self.stationary_distribution = stationary_vector
        return stationary_vector
    
    def calculate_mfpt(self, state_A_index: List[int], stopped_process: bool=False):
        """
        Calculate mean first-passage time from any state to states from state_A_index in units of seconds.
        """
        if(self.markov_generator is None):
            self.calculate_generator()
        
        n_unknown_variables = self.n_states - len(state_A_index)
        if(n_unknown_variables == 0):
            trivial_mfpt = np.zeros(self.n_states)
            return trivial_mfpt

        state_A_index = self.old_to_new_reactant_states(state_A_index)
        unknown_variables_index = [i for i in range(self.n_states) if i not in state_A_index]
        equation_coefficients = self.markov_generator[np.ix_(unknown_variables_index, unknown_variables_index)]
        values_vector = -np.ones(equation_coefficients.shape[0])
        cut_mfpt = np.zeros_like(values_vector)

        if(stopped_process):
            for i, state_idx in enumerate(unknown_variables_index):
                cut_mfpt[i] = self.counting_mfpt(state_A_index, state_idx, self.lag_time) * self.lag_time_in_sec
        else:
            cut_mfpt = np.linalg.solve(equation_coefficients, values_vector)
        full_mfpt = np.zeros(self.n_states) # boundary condition for mfpt satisfies trivially
        full_mfpt[unknown_variables_index] = cut_mfpt
        return full_mfpt
       
    def counting_mfpt(self, reactant_states: List[int], input_state: int, lag_time: int) -> float:
        """
        Calculates the mean first passage time from every input state to the reactant state in frame units.

        Parameters
        ----------
        reactant_states : List[int]
            Labels of the reactant state
        input_state : int
            Label of the input state
        lag_time : int
            Size of the waiting time window
        """
        traj = self.labels[self.traj_mask]
        reactant_indices = np.where(np.isin(traj,  reactant_states))[0].astype(np.int8)
        input_indices = np.where(traj == input_state)[0].astype(np.int8)

        if not reactant_indices.size:
            raise ValueError("Reactant state label was never hit in the trajectory.")

        passage_times = reactant_indices[:, None] - input_indices # all passage times
        if(~np.any(passage_times < 0)):
            return np.nan
        
        passage_times[passage_times < 0] = np.max(passage_times) # Negative values are not valid, will be filtered by np.min
        passage_times[passage_times >= lag_time] = lag_time # Passage times cannot be greater than time window size
        passage_times = np.min(passage_times, axis=0) # FIRST passage time

        if not passage_times.size:
            raise ValueError("Input point label never precedes reactant state label.")

        mean_fpt = np.mean(passage_times)
        return mean_fpt
    
    def timescales(self, k=None):
        """
        The relaxation timescales corresponding to the eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be returned. By default all available
            eigenvalues, minus 1.

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales in units of the input trajectory time step,
            defined by :math: tau / ln | lambda_i |, i = 2,...,k+1.

        """
        if(self.eigenvalues is None):
            self.calculate_evecs()
        from deeptime.markov.tools.analysis import timescales_from_eigenvalues
        ts = timescales_from_eigenvalues(self.eigenvalues, tau=self.lag_time)
        if k is None:
            return ts[1:]
        else:
            return ts[1:k+1]  # exclude the stationary process
