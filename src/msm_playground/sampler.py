import numpy as np
import numpy.typing as npt
from typing import Callable


def sample_overdamped_langevin(
    cfg_0: npt.ArrayLike,
    force_function: Callable,
    periodicity_fxn: Callable = None,
    nsteps: int = 1e5,
    burnin: int = 100,
    dt: float = 0.001,
    kT: float = 1.0,
) -> npt.NDArray:
    """
    Runs overdamped Langevin given a force function
    """
    cfg = cfg_0.copy()
    cfg_shape = cfg.shape
    R_n = np.random.normal(0, 1)
    traj = []
    sig = np.sqrt(dt / 2)
    rando = np.random.normal(0, 1, size=(int(nsteps + burnin),) + cfg_shape)
    for j in range(int(nsteps + burnin)):
        # Run Overdamped BAOAB algorithm
        force = force_function(cfg)
        cfg += dt * force / kT + sig * (rando[j] + R_n)
        R_n = rando[j]

        # Optionally enforce periodicity
        if periodicity_fxn is not None:
            cfg = periodicity_fxn(cfg)
        traj.append(np.copy(cfg))
    return np.array(traj)[int(burnin) :]
