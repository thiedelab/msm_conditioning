import numba
import numpy as np
import numpy.typing as npt
from typing import Callable, Union
from numba import njit


@njit()
def set_seed(seed: int):
    np.random.seed(seed)

@numba.cfunc(numba.types.boolean(numba.types.Array(numba.types.float64, 1, 'C')))
def always_false(cfg: np.ndarray) -> bool:
    return False

@njit()
def sample_overdamped_langevin(
    cfg_0: npt.ArrayLike,
    force_function: Callable,
    nsteps: int = 1e5,
    burnin: int = 1000,
    dt: float = 0.0001,
    kT: float = 1.0,
    reject_function: Callable[[npt.ArrayLike], bool] = always_false,
) -> npt.NDArray:
    """
    Runs overdamped Langevin given a force function
    """
    cfg = cfg_0.copy()
    cfg_shape = cfg.shape
    sig = np.sqrt(dt / 2)
    rando = np.random.normal(0, 1, size=(int(nsteps + burnin + 1),) + cfg_shape)
    R_n = rando[0]
    traj = np.zeros((int(nsteps + burnin), cfg_shape[0]))
    if(reject_function is None):
        reject_function = always_accept

    for j in range(int(nsteps + burnin)):
        # Run Overdamped BAOAB algorithm
        #         force = Force_jit(cfg)
        force = force_function(cfg)
        proposed_cfg = cfg + dt * force / kT + sig * (rando[j + 1] + R_n)
        if not reject_function(proposed_cfg):
            cfg = proposed_cfg
        # if not accepted, we keep the old cfg            
        R_n = rando[j + 1]
        traj[j] = cfg
    return traj[burnin:]

@numba.cfunc(numba.types.boolean(numba.types.Array(numba.types.float64, 1, 'C')))
def always_accept(cfg: np.ndarray) -> bool:
    return False

def get_outside_of_grid_function(grid_center: np.ndarray, grid_dimensions: np.ndarray) -> Callable[[np.ndarray], bool]:
    @numba.cfunc(numba.types.boolean(numba.types.Array(numba.types.float64, 1, 'C')))
    def is_outside_of_grid(coord: np.ndarray) -> bool:
        return not np.all(np.abs(coord - grid_center) < grid_dimensions / 2)
    
    return is_outside_of_grid

@njit
def double_well_force(coord: float) -> float:
    """
    Calculate the force of the double 1D well potential V(x) = scale * (a * x^4 - b * x^2 + c * x)
    Unfortunately coefficients must be hardcoded for the sake of njit.
    Scale must be same as the one used in the potential.
    
    """
    const_a = 1.0
    const_b = 1.0
    const_c = 0.2
    scale = 5.0 * 2

    Fx = - scale * (4 * const_a * coord**3 - 2 * const_b * coord + const_c)
    return Fx

@njit
def mueller_brown_force(coord):
    x = coord[0]
    y = coord[1]

    Fx = -200 * x * np.exp(-(x**2) - 10 * (y - 0.5) ** 2)
    Fx += 200 * (2 - 2 * x) * np.exp(-10 * y**2 - (x - 1) ** 2)
    Fx += (
        170
        * (-13.0 * x + 11 * y - 23.0)
        * np.exp(
            -6.5 * (x + 0.5) ** 2
            + (11 * x + 5.5) * (y - 1.5)
            - 14.625 * (0.666666666666667 * y - 1) ** 2
        )
    )
    Fx -= (
        15
        * (1.4 * x + 0.6 * y + 0.8)
        * np.exp((0.6 * x + 0.6) * (y - 1) + 0.7 * (x + 1) ** 2 + 0.7 * (y - 1) ** 2)
    )

    Fy = -4000 * y * np.exp(-10 * y**2 - (x - 1) ** 2)
    Fy += 100 * (10.0 - 20 * y) * np.exp(-(x**2) - 10 * (y - 0.5) ** 2)
    Fy -= (
        15
        * (0.6 * x + 1.4 * y - 0.8)
        * np.exp((0.6 * x + 0.6) * (y - 1) + 0.7 * (x + 1) ** 2 + 0.7 * (y - 1) ** 2)
    )
    Fy += (
        170
        * (11 * x - 13.0 * y + 25.0)
        * np.exp(
            -6.5 * (x + 0.5) ** 2
            + (11 * x + 5.5) * (y - 1.5)
            - 14.625 * (0.666666666666667 * y - 1) ** 2
        )
    )

    return np.array([Fx, Fy])
