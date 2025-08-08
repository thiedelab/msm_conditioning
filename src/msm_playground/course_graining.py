import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from msm_playground.msm import MSM
from typing import List, Tuple, Union, Callable
from sklearn.preprocessing import normalize
from tqdm import tqdm

def build_macro_Tmat_from_micro_Tmat(Tmat: sps.csr_matrix,
                                       microstates_list: npt.ArrayLike,
                                       macrostates_list: npt.ArrayLike,
                                       measure: npt.ArrayLike | None = None) -> sps.csr_matrix:
    """
    Builds a coarse Tmat matrix from "microstates" to "macrostates" by unifying the microstates into macrostates
    and counting the transitions between microstates compounding each macrostate.

    Parameters
    ----------
    Tmat : sps.csr_matrix
        The transition matrix of the MSM on a fine domain
    microstates_list : npt.ArrayLike
        A list of "microstates" coordinates (vectors) to be grouped into "macrostates"
    macrostates_list : npt.ArrayLike
        A list of "macrostates" coordinates (vectors) to be grouped into "macrostates"
    measure : npt.ArrayLike | None
        The sampling measure of each "microstate"

    Returns
    -------
    sps.csr_matrix
        The transition matrix of the MSM on the coarse macrostates domain.
    """
    if(measure is None):
        measure = np.ones(Tmat.shape[0]) / Tmat.shape[0]

    proejction_matrix = build_projection_matrix(microstates_list, macrostates_list)
    proejction_matrix_w_measure = proejction_matrix.multiply(measure[:, np.newaxis])
    Tmat_macro = proejction_matrix_w_measure.T @ Tmat @ proejction_matrix
    Tmat_macro = normalize(Tmat_macro, norm='l1', axis=1)
    
    return Tmat_macro

def build_projection_matrix(microstates_list: npt.ArrayLike,
                            macrostates_list: npt.ArrayLike) -> sps.csr_matrix:
    """
    Builds a projection matrix from "microstates" to "macrostates" by unifying the microstates into macrostates
    and counting the transitions between microstates compounding each macrostate.

    Parameters
    ----------
    microstates_list : npt.ArrayLike
        A list of "microstates" coordinates (vectors) to be grouped into "macrostates"
    macrostates_list : npt.ArrayLike
        A list of "macrostates" coordinates (vectors) to be grouped into "macrostates"
    measure : npt.ArrayLike | None
        The sampling measure of each "microstate"

    Returns
    -------
    sps.csr_matrix
        The projection (indicator) matrix of the MSM on the coarse domain.
        To get Tmat_macro, use `Tmat_macro = indicator_matrix_w_measure.T @ Tmat @ indicator_matrix` and row-normalize it.
    """
    n_microstates = len(microstates_list)
    n_macrostates = len(macrostates_list)

    # Matrix whose entry (i, j) is 1 if microstate i is in macrostate S_j
    indicator_matrix = sps.csr_matrix((n_microstates, n_macrostates))

    # plot all microstates as small points and macrostates as large squares
    import matplotlib.pyplot as plt
    plt.scatter(microstates_list[:, 0], microstates_list[:, 1], s=1, label='microstates')
    plt.scatter(macrostates_list[:, 0], macrostates_list[:, 1], s=100, c='red', marker='s', label='macrostates')
    plt.legend()
    plt.savefig("micro_macro_states")


    # Build the indicator (projection) matrix
    for i, microstate in enumerate(microstates_list):
        dists = np.linalg.norm(microstate - macrostates_list, axis=-1)
        sorted_dists = np.sort(dists)

        # microstate is equally close to 2 macrostate centers (lying on the border between the two)
        if(np.abs(sorted_dists[0] - sorted_dists[1]) < 10**(-4)):
            nearest_macrostate_idx = np.argsort(dists)[:2]
            indicator_matrix[i, nearest_macrostate_idx] = 0.5
        else:
            nearest_macrostate_idx = np.argmin(dists)
            indicator_matrix[i, nearest_macrostate_idx] = 1

    return indicator_matrix

def _build_projection_matrix(
    fine_to_course: npt.ArrayLike, n_course: int = None
) -> npt.NDArray:
    """
    Args:
        fine_to_course (npt.ArrayLike): Array of shape (n_fine_states, ) containing the course state for each fine state
        n_course (int, optional): Number of course states. Defaults to None.

    Returns:
        npt.NDArray: Projection matrix of shape (n_course, n_fine_states)
    """
    if n_course is None:
        n_course = np.max(fine_to_course) + 1

    Projection_matrix = sps.csr_matrix(
        (
            np.ones(fine_to_course.shape[0]),
            (fine_to_course, np.arange(fine_to_course.shape[0])),
        ),
        shape=(n_course, fine_to_course.shape[0]),
    )
    return Projection_matrix


def build_meshes(
    X: npt.NDArray, Y: npt.NDArray, downsampling_factor: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Evenly downsamples a grid, and then flattens both the
    high resolution and downsampled grids into a list of
    coordinates.

    Args:
        X (npt.NDArray): X coordinates of the grid
        Y (npt.NDArray): Y coordinates of the grid
        downsampling_factor (int): Downsampling factor

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: High resolution mesh, downsampled mesh
    """
    X_downsampled = X[::downsampling_factor, ::downsampling_factor]
    Y_downsampled = Y[::downsampling_factor, ::downsampling_factor]

    high_res_mesh = np.stack((X, Y), axis=-1).reshape(-1, 2)

    downsampled_mesh = np.stack((X_downsampled, Y_downsampled), axis=-1).reshape(-1, 2)
    return high_res_mesh, downsampled_mesh


def build_downsampled_matrix(
    downsampled_mesh: npt.NDArray, high_res_mesh: npt.NDArray
) -> npt.NDArray:
    downsample_matrix = np.zeros((downsampled_mesh.shape[0], high_res_mesh.shape[0]))
    for state_idx in range(high_res_mesh.shape[0]):
        # choose the top *downsample_states_per_dim_factor* closest points in the high res mesh to the downsampled mesh
        dists = np.linalg.norm(high_res_mesh[state_idx] - downsampled_mesh, axis=-1)
        closest_low_res_idx = np.argmin(dists)
        downsample_matrix[closest_low_res_idx, state_idx] += 1
    return downsample_matrix


def build_stopped_P_matrix(
    P_matrix: sps.csr_matrix, idx_outside_domain: npt.ArrayLike
) -> sps.csr_matrix:
    """
    Builds a transition matrix for a process that stops once it leaves a domain.

    Args:
        P_matrix (sps.csr_matrix): Transition matrix
        idx_outside_domain (npt.ArrayLike): Array containing the indices of the states outside the domain

    Returns:
        sps.csr_matrix: Stopped transition matrix
    """
    stopped_P_matrix = P_matrix.copy()
    stopped_P_matrix[idx_outside_domain, :] = 0
    stopped_P_matrix[idx_outside_domain, idx_outside_domain] = 1

    # Check if matrix is a sparse scipy matrix
    if isinstance(stopped_P_matrix, sps.spmatrix):
        stopped_P_matrix.eliminate_zeros()
    return stopped_P_matrix


def calculate_lagstop_mfpts_linear_lags(
    stopped_P_matrix: sps.csr_matrix,
    projection_matrix: npt.NDArray,
    tmax: int,
    state_A: npt.ArrayLike,
    measure: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates the stopped mfpt

    Parameters
    ----------
    stopped_P_matrix: Union[sps.csr_matrix, npt.NDArray]
        The transition matrix of the MSM
    projection_matrix: npt.NDArray
        The projection matrix from the high resolution mesh to the course grained mesh
    tmax: int
        The maximum lag time to calculate the committor at
    state_A: npt.ArrayLike
        A boolean array indicating which states are in state A
    measure: npt.NDArray, optional
        The measure of each state.  If None, defaults to uniform measure

    Returns
    -------
    mfpts: npt.NDArray
        The committor at each state for each time lag
    t_lags: npt.NDArray
        The time lags
    """
    # Set up the probability measure, projection matrices, and other useful quantities.
    if measure is None:
        measure = np.ones(stopped_P_matrix.shape[0])
        measure /= np.sum(measure)
    projection_mat_with_measure = projection_matrix * measure
    cg_state_A = projection_matrix @ state_A
    states_in_domain = np.where(cg_state_A == 0)[0]
    C_0_mat = projection_mat_with_measure @ projection_matrix.T
    C_0_mat = C_0_mat[states_in_domain, :][:, states_in_domain]

    # Initialize variables for the main loop
    t_lag = 1
    fg_Pmat_minus_1 = sps.eye(stopped_P_matrix.shape[0], format="csr")
    fg_Pmat = stopped_P_matrix @ fg_Pmat_minus_1

    r_vector = np.zeros(cg_state_A.shape)

    mfpts = []
    t_lags = []
    for t_lag in tqdm(range(1, tmax + 1)):
        forward_mat = fg_Pmat @ projection_matrix.T
        C_T_mat = projection_mat_with_measure @ forward_mat
        b_vector = fg_Pmat_minus_1 @ (state_A - 1)

        r_vector += projection_mat_with_measure @ b_vector

        # Subsample the matrices
        C_T_mat = C_T_mat[states_in_domain, :][:, states_in_domain]
        r_vector_ss = r_vector[states_in_domain]

        if isinstance(C_T_mat, sps.spmatrix):
            mfpt_on_domain = sps.linalg.spsolve(C_T_mat - C_0_mat, r_vector_ss)
        else:
            mfpt_on_domain = np.linalg.solve(C_T_mat - C_0_mat, r_vector_ss)

        mfpt_full = np.zeros(cg_state_A.shape)
        mfpt_full[states_in_domain] = mfpt_on_domain

        mfpts.append(mfpt_full)
        t_lags.append(t_lag)

        # Increment t_lag
        t_lag += 1
        if t_lag <= tmax:
            fg_Pmat_minus_1 = fg_Pmat
            fg_Pmat = fg_Pmat.dot(stopped_P_matrix)
    return mfpts, t_lags

def calculate_naive_mfpts_linear_lags(
    P_matrix: Union[sps.csr_matrix, npt.NDArray],
    projection_matrix: npt.NDArray,
    tmax: int,
    state_A: npt.ArrayLike,
    measure: npt.NDArray = None,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates the naive committor

    Parameters
    ----------
    P_matrix: Union[sps.csr_matrix, npt.NDArray]
        The transition matrix of the MSM
    projection_matrix: npt.NDArray
        The projection matrix from the high resolution mesh to the course grained mesh
    tmax: int
        The maximum lag time to calculate the committor at
    state_A: npt.ArrayLike
        A boolean array indicating which states are in state A
    measure: npt.NDArray, optional
        The measure of each state.  If None, defaults to uniform measure

    Returns
    -------
    mfpts: npt.NDArray
        The committor at each state for each time lag
    t_lags: npt.NDArray
        The time lags
    """
    if measure is None:
        measure = np.ones(P_matrix.shape[0]) / P_matrix.shape[0]

    projection_matrix_w_measure = projection_matrix * measure
    course_grained_A = projection_matrix @ state_A

    T_mat_at_lag_time = np.eye(P_matrix.shape[0])

    all_mfpts = []
    for lag_time in tqdm(range(1, tmax + 1)):
        T_mat_at_lag_time = P_matrix @ T_mat_at_lag_time

        course_grained_Tmat = (
            projection_matrix_w_measure @ T_mat_at_lag_time @ projection_matrix.T
        )
        rowsum = course_grained_Tmat.sum(axis=1, keepdims=True)

        course_grained_Tmat /= rowsum

        assert np.allclose(
            course_grained_Tmat.sum(axis=1), 1
        )  # Check we're row-stochastic

        mfpt = _build_naive_mfpt_from_Tmat(course_grained_Tmat, course_grained_A, lag_time)
        all_mfpts.append(mfpt)

    return all_mfpts, np.arange(1, tmax + 1)

def calculate_naive_committors_linear_lags(
    P_matrix: Union[sps.csr_matrix, npt.NDArray],
    projection_matrix: npt.NDArray,
    tmax: int,
    state_A: npt.ArrayLike,
    state_B: npt.ArrayLike,
    measure: npt.NDArray = None,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates the naive committor

    Parameters
    ----------
    P_matrix: Union[sps.csr_matrix, npt.NDArray]
        The transition matrix of the MSM
    projection_matrix: npt.NDArray
        The projection matrix from the high resolution mesh to the course grained mesh
    tmax: int
        The maximum lag time to calculate the committor at
    state_A: npt.ArrayLike
        A boolean array indicating which states are in state A
    state_B: npt.ArrayLike
        A boolean array indicating which states are in state B
    measure: npt.NDArray, optional
        The measure of each state.  If None, defaults to uniform measure

    Returns
    -------
    committors: npt.NDArray
        The committor at each state for each time lag
    t_lags: npt.NDArray
        The time lags
    """
    if measure is None:
        measure = np.ones(P_matrix.shape[0]) / P_matrix.shape[0]

    projection_matrix_w_measure = projection_matrix * measure
    course_grained_B = projection_matrix @ state_B
    course_grained_A = projection_matrix @ state_A

    T_mat_at_lag_time = np.eye(P_matrix.shape[0])

    all_committors = []
    for lag_time in tqdm(range(1, tmax + 1)):
        T_mat_at_lag_time = P_matrix @ T_mat_at_lag_time

        course_grained_Tmat = (
            projection_matrix_w_measure @ T_mat_at_lag_time @ projection_matrix.T
        )
        rowsum = course_grained_Tmat.sum(axis=1, keepdims=True)

        course_grained_Tmat /= rowsum

        assert np.allclose(
            course_grained_Tmat.sum(axis=1), 1
        )  # Check we're row-stochastic

        committor = _build_naive_committor_from_Tmat(
            course_grained_Tmat, course_grained_A, course_grained_B
        )
        all_committors.append(committor)

    return all_committors, np.arange(1, tmax + 1)


def calculate_lagstop_committors_linear_lags(
    stopped_P_matrix: sps.csr_matrix,
    projection_matrix: npt.NDArray,
    tmax: int,
    state_A: npt.ArrayLike,
    state_B: npt.ArrayLike,
    measure: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    if measure is None:
        measure = np.ones(stopped_P_matrix.shape[0]) / stopped_P_matrix.shape[0]

    # Set up the probability measure, projection matrices, and other useful quantities.
    projection_mat_with_measure = projection_matrix * measure
    cg_state_A = projection_matrix @ state_A
    cg_state_B = projection_matrix @ state_B
    states_in_domain = get_domain_state_indexes(cg_state_A, cg_state_B)
    C_0_mat = projection_mat_with_measure @ projection_matrix.T
    C_0_mat = C_0_mat[states_in_domain, :][:, states_in_domain]
    psi_boundary = np.zeros_like(cg_state_B)
    psi_boundary[cg_state_B != 0] = 1

    # Initialize variables for the main loop
    t_lag = 1
    fg_Pmat_minus_1 = sps.eye(stopped_P_matrix.shape[0], format="csr")
    fg_Pmat = stopped_P_matrix @ fg_Pmat_minus_1

    committors = []
    t_lags = []

    for t_lag in tqdm(range(1, tmax + 1)):
        forward_mat = fg_Pmat @ projection_matrix.T
        C_T_mat = projection_mat_with_measure @ forward_mat

        r_vector_ss = psi_boundary - C_T_mat @ psi_boundary
        # Subsample the matrices
        C_T_mat = C_T_mat[states_in_domain, :][:, states_in_domain]
        r_vector_ss = r_vector_ss[states_in_domain]

        if isinstance(C_T_mat, sps.spmatrix):
            committor_on_domain = sps.linalg.spsolve(C_T_mat - C_0_mat, r_vector_ss)
        else:
            committor_on_domain = np.linalg.solve(C_T_mat - C_0_mat, r_vector_ss)

        committor_full = np.zeros(cg_state_A.shape)
        committor_full[cg_state_B != 0] = 1
        committor_full[cg_state_A != 0] = 0
        committor_full[states_in_domain] = committor_on_domain

        committors.append(committor_full)
        t_lags.append(t_lag)

        # Increment t_lag
        t_lag += 1
        if t_lag <= tmax:
            fg_Pmat_minus_1 = fg_Pmat
            fg_Pmat = fg_Pmat.dot(stopped_P_matrix)
    return committors, t_lags

def p2_norm_condition_number(matrix: npt.NDArray) -> float:
    return np.linalg.cond(matrix, p=2)

def calculate_naive_cond_nums(
    P_matrix: Union[sps.csr_matrix, npt.NDArray],
    projection_matrix: npt.NDArray,
    tmax: int,
    state_A: npt.ArrayLike,
    state_B: npt.ArrayLike,
    measure: npt.NDArray = None,
    condition_function: Callable[[np.ndarray], float] = p2_norm_condition_number,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates the naive condition numbers of the LHS of the committor/mfpt equation (which is either Tmat - I on domain or C_T - C_0 on domain)

    Parameters
    ----------
    P_matrix: Union[sps.csr_matrix, npt.NDArray]
        The transition matrix of the MSM
    projection_matrix: npt.NDArray
        The projection matrix from the high resolution mesh to the course grained mesh
    tmax: int
        The maximum lag time to calculate the committor at
    state_A: npt.ArrayLike
        A boolean array indicating which states are in state A
    state_B: npt.ArrayLike
        A boolean array indicating which states are in state B
    measure: npt.NDArray, optional
        The measure of each state.  If None, defaults to uniform measure
    condition_function: Callable[[np.ndarray], float]
        The condition number function to use. Takes in a matrix and returns the condition number

    Returns
    -------
    Tmat_condition_numbers: np.array[float]
        Tmat condition number for each time lag
    Cmat_condition_numbers: np.array[float]
        The committor at each state for each time lag
    t_lags: npt.NDArray
        The time lags
    """
    if measure is None:
        measure = np.ones(P_matrix.shape[0]) / P_matrix.shape[0]

    projection_matrix_w_measure = projection_matrix * measure
    course_grained_B = projection_matrix @ state_B
    course_grained_A = projection_matrix @ state_A
    cg_domain_states = get_domain_state_indexes(course_grained_A, course_grained_B)

    T_mat_at_lag_time = np.eye(P_matrix.shape[0])

    Tmat_condition_numbers = []
    Cmat_condition_numbers = []
    t_lags = []

    for t_lag in tqdm(range(1, tmax + 1)):
        T_mat_at_lag_time = P_matrix @ T_mat_at_lag_time

        course_grained_Tmat = (
            projection_matrix_w_measure @ T_mat_at_lag_time @ projection_matrix.T
        )
        rowsum = course_grained_Tmat.sum(axis=1, keepdims=True)

        course_grained_C_T = course_grained_Tmat.copy()
        course_grained_C_0 = np.diag(course_grained_C_T.sum(axis=1))
        course_grained_Tmat /= rowsum

        assert np.allclose(
            course_grained_Tmat.sum(axis=1), 1
        )  # Check we're row-stochastic

        # Subsample the matrices
        course_grained_Tmat = course_grained_Tmat[cg_domain_states, :][:, cg_domain_states]
        course_grained_C_T = course_grained_C_T[cg_domain_states, :][:, cg_domain_states]
        course_grained_C_0 = course_grained_C_0[cg_domain_states, :][:, cg_domain_states]
        I = np.eye(course_grained_Tmat.shape[0])
        
        Tmat_condition_numbers.append(condition_function(course_grained_Tmat - I))
        Cmat_condition_numbers.append(condition_function(course_grained_C_T - course_grained_C_0))        
        t_lags.append(t_lag)

    return Tmat_condition_numbers, Cmat_condition_numbers, t_lags

def calculate_lagstop_cond_nums(
    stopped_P_matrix: sps.csr_matrix,
    projection_matrix: npt.NDArray,
    tmax: int,
    state_A: npt.ArrayLike,
    state_B: npt.ArrayLike,
    measure: npt.NDArray = None,
    condition_function: Callable[[np.ndarray], float] = p2_norm_condition_number,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Calculates the naive condition numbers of the LHS of the committor/mfpt equation (which is either Tmat - I on domain or C_T - C_0 on domain)

    Parameters
    ----------
    P_matrix: Union[sps.csr_matrix, npt.NDArray]
        The transition matrix of the MSM
    projection_matrix: npt.NDArray
        The projection matrix from the high resolution mesh to the course grained mesh
    tmax: int
        The maximum lag time to calculate the committor at
    state_A: npt.ArrayLike
        A boolean array indicating which states are in state A
    state_B: npt.ArrayLike
        A boolean array indicating which states are in state B
    measure: npt.NDArray, optional
        The measure of each state.  If None, defaults to uniform measure
    condition_function: Callable[[np.ndarray], float]
        The condition number function to use. Takes in a matrix and returns the condition number

    Returns
    -------
    Tmat_condition_numbers: np.array[float]
        Tmat condition number for each time lag
    Cmat_condition_numbers: np.array[float]
        The committor at each state for each time lag
    t_lags: npt.NDArray
        The time lags
    """
    if measure is None:
        measure = np.ones(stopped_P_matrix.shape[0]) / stopped_P_matrix.shape[0]

    # Set up the probability measure, projection matrices, and other useful quantities.
    projection_mat_with_measure = projection_matrix * measure
    cg_state_A = projection_matrix @ state_A
    cg_state_B = projection_matrix @ state_B
    states_in_domain = get_domain_state_indexes(cg_state_A, cg_state_B)
    C_0_mat = projection_mat_with_measure @ projection_matrix.T
    C_0_mat = C_0_mat[states_in_domain, :][:, states_in_domain]

    # Initialize variables for the main loop
    t_lag = 1
    fg_Pmat_minus_1 = sps.eye(stopped_P_matrix.shape[0], format="csr")
    fg_Pmat = stopped_P_matrix @ fg_Pmat_minus_1

    Tmat_condition_numbers = []
    Cmat_condition_numbers = []
    t_lags = []


    for t_lag in tqdm(range(1, tmax + 1)):
        forward_mat = fg_Pmat @ projection_matrix.T
        C_T_mat = projection_mat_with_measure @ forward_mat

        Tmat = C_T_mat.copy() / np.sum(C_T_mat, axis=1, keepdims=True)
        # Subsample the matrices
        C_T_mat = C_T_mat[states_in_domain, :][:, states_in_domain]
        Tmat = Tmat[states_in_domain, :][:, states_in_domain]
        I = np.eye(Tmat.shape[0])

        Tmat_condition_numbers.append(condition_function(Tmat - I))
        Cmat_condition_numbers.append(condition_function(C_T_mat - C_0_mat))        
        t_lags.append(t_lag)

        # Increment t_lag
        t_lag += 1
        if t_lag <= tmax:
            fg_Pmat_minus_1 = fg_Pmat
            fg_Pmat = fg_Pmat.dot(stopped_P_matrix)
    return Tmat_condition_numbers, Cmat_condition_numbers, t_lags

def get_domain_state_indexes(
    state_A: npt.ArrayLike, state_B: npt.ArrayLike
) -> npt.ArrayLike:
    states_in_domain = np.where(state_A == 0)[0]
    states_in_domain = np.intersect1d(states_in_domain, np.where(state_B == 0)[0])
    return states_in_domain
    

def get_raw_Tmat_cond_num(
    Tmat,
    state_A: npt.ArrayLike,
    state_B: npt.ArrayLike,
    condition_function: Callable[[np.ndarray], float] = p2_norm_condition_number,
    measure=None,
) -> Tuple[float, float]:
    """
    What it does is basically:
    1. Multiplies the Tmat by the measure to get the C_T
    2. Get C_0 as a diagonal matrix with the row sums of C_T
    3. Gets domain states as union of state_A and state_B
    4. Subsamples the all matrices to the domain states
    5. Get the condition number of Tmat - I and C_T - C_0

    Parameters
    ----------
    Tmat: npt.NDArray
        The transition matrix on ALL states (not just the domain states)
    state_A: npt.ArrayLike
        A boolean array indicating which states are in state A
    state_B: npt.ArrayLike
        A boolean array indicating which states are in state B
    condition_function: Callable[[np.ndarray], float]
        The condition number function to use. Takes in a matrix and returns the condition number
    measure: npt.NDArray, optional
        The measure of each state.  If None, defaults to uniform measure

    Returns
    -------
    Tmat_condition_number: float
        Tmat condition number
    Cmat_condition_number: float
        Cmat condition number
    """
    if measure is None:
        measure = np.ones(Tmat.shape[0]) / Tmat.shape[0]

    rowsum = Tmat.sum(axis=1, keepdims=True)
    Tmat /= rowsum

    C_T = Tmat.copy() * measure
    C_0 = np.diag(C_T.sum(axis=1))

    states_in_domain = get_domain_state_indexes(state_A, state_B)
    Tmat = Tmat[states_in_domain, :][:, states_in_domain]
    C_T = C_T[states_in_domain, :][:, states_in_domain]
    C_0 = C_0[states_in_domain, :][:, states_in_domain]
    I = np.eye(Tmat.shape[0])
    return condition_function(Tmat - I), condition_function(C_T - C_0)

def get_projected_Tmat_cond_num(
    Tmat,
    projection_matrix,
    state_A: npt.ArrayLike,
    state_B: npt.ArrayLike,
    condition_function: Callable[[np.ndarray], float] = p2_norm_condition_number,
    measure=None,
) -> Tuple[float, float]:
    """
    What it does is basically:
    1. Projects the Tmat to the course grained states
    2. Calls get_raw_Tmat_cond_num() on the projected Tmat

    Parameters
    ----------
    Tmat: npt.NDArray
        The transition matrix
    projection_matrix: npt.NDArray
        The projection matrix from the high resolution mesh to the course grained mesh
    state_A: npt.ArrayLike
        A boolean array indicating which states are in state A
    state_B: npt.ArrayLike
        A boolean array indicating which states are in state B
    condition_function: Callable[[np.ndarray], float]
        The condition number function to use. Takes in a matrix and returns the condition number
    measure: npt.NDArray, optional
        The measure of each state.  If None, defaults to uniform measure

    Returns
    -------
    Tmat_condition_number: float
        Tmat condition number
    Cmat_condition_number: float
        Cmat condition number
    """
    if measure is None:
        measure = np.ones(Tmat.shape[0]) / Tmat.shape[0]

    projection_matrix_w_measure = projection_matrix * measure
    course_grained_Tmat = (
            projection_matrix_w_measure @ Tmat @ projection_matrix.T
        )
    rowsum = course_grained_Tmat.sum(axis=1, keepdims=True)
    course_grained_Tmat /= rowsum
    course_grained_state_A = projection_matrix @ state_A
    course_grained_state_B = projection_matrix @ state_B
    course_grained_measure = projection_matrix @ measure

    return get_raw_Tmat_cond_num(course_grained_Tmat, course_grained_state_A, course_grained_state_B, condition_function, course_grained_measure)

def calculate_mfpts_log_lags(
    stopped_P_matrix: sps.csr_matrix,
    projection_matrix: npt.NDArray,
    tmax: int,
    state_A: npt.ArrayLike,
    measure: npt.NDArray,
) -> npt.NDArray:
    raise NotImplementedError

def _build_naive_committor_from_Tmat(reference_Tmat, state_A, state_B):
    ref_msm = MSM(
        traj=None,
        clustering_obj=None,
        lag_time=1,
        seconds_between_frames=1,
    )

    ref_msm.Cmat = reference_Tmat
    stateA_labels = np.where(state_A)[0]
    stateB_labels = np.where(state_B)[0]
    reference_committor = ref_msm.calculate_committor(
        state_A_index=stateA_labels, state_B_index=stateB_labels
    )
    return reference_committor

def _build_naive_mfpt_from_Tmat(reference_Tmat, state_A, lag_time=1):
    ref_msm = MSM(
        traj=None,
        clustering_obj=None,
        lag_time=lag_time,
        seconds_between_frames=1,
    )

    ref_msm.Cmat = reference_Tmat
    stateA_labels = np.where(state_A)[0]
    reference_mfpt = ref_msm.calculate_mfpt(state_A_index=stateA_labels)
    return reference_mfpt
