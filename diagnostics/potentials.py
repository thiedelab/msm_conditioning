import numpy as np


class TripleWellPotential:
    o = 1.5
    a = 3.2
    b = 2.74
    k = 0.12


def triple_well_potential(x):
    # Hardcoded triple well potential function
    return (
        (TripleWellPotential.o * x) ** 6
        - TripleWellPotential.a * (TripleWellPotential.o * x) ** 4
        + TripleWellPotential.b * (TripleWellPotential.o * x) ** 2
        - TripleWellPotential.k * (TripleWellPotential.o * x)
        + 0.1
    )


def triple_well_force(x):
    # Hardcoded triple well force function
    return (
        -(
            6 * (TripleWellPotential.o * x) ** 5
            - 4 * TripleWellPotential.a * (TripleWellPotential.o * x) ** 3
            + 2 * TripleWellPotential.b * (TripleWellPotential.o * x)
            - TripleWellPotential.k
        )
        * TripleWellPotential.o
    )


def double_well_potential(
    dots_per_x_dim: int = 100,
    x_min: float = -2.5,
    x_max: float = 2.5,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 0.0,
):
    """
    Calculate the double well potential V(x) = a * x^4 - b * x^2 + c * x

    Parameters:
    dots_per_x_dim (int): Number of points in the x-dimension.
    x_min (float): Minimum x value.
    x_max (float): Maximum x value.
    a (float): Coefficient for the x^4 term. Default is 1.0.
    b (float): Coefficient for the x^2 term. Default is 1.0.
    c (float): Linear term, which makes the potential asymmetric. Default is 0.0.

    Returns:
    U (np.ndarray): The potential values.
    xgrid (np.ndarray): The x values corresponding to the potential.
    """
    xgrid = np.linspace(x_min, x_max, dots_per_x_dim, endpoint=True)
    U = a * xgrid**4 - b * xgrid**2 + c * xgrid

    return U, xgrid


def MB_potential(
    dots_per_x_dim: int = 100,
    x_min: float = -2.5,
    x_max: float = 2.5,
    dots_per_y_dim: int = 100,
    y_min: float = -2.5,
    y_max: float = 2.5,
):
    A = [-200, -100, -170, 15]
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    X0 = [1, 0, -0.5, -1]
    Y0 = [0, 0.5, 1.5, 1]

    xgrid = np.linspace(x_min, x_max, dots_per_x_dim, endpoint=True)
    ygrid = np.linspace(y_min, y_max, dots_per_y_dim, endpoint=True)
    X, Y = np.meshgrid(xgrid, ygrid)

    U = (
        A[0]
        * np.exp(
            a[0] * (X - X0[0]) ** 2
            + b[0] * (X - X0[0]) * (Y - Y0[0])
            + c[0] * (Y - Y0[0]) ** 2
        )
        + A[1]
        * np.exp(
            a[1] * (X - X0[1]) ** 2
            + b[1] * (X - X0[1]) * (Y - Y0[1])
            + c[1] * (Y - Y0[1]) ** 2
        )
        + A[2]
        * np.exp(
            a[2] * (X - X0[2]) ** 2
            + b[2] * (X - X0[2]) * (Y - Y0[2])
            + c[2] * (Y - Y0[2]) ** 2
        )
        + A[3]
        * np.exp(
            a[3] * (X - X0[3]) ** 2
            + b[3] * (X - X0[3]) * (Y - Y0[3])
            + c[3] * (Y - Y0[3]) ** 2
        )
    )

    return U, (X, Y), (xgrid, ygrid)
