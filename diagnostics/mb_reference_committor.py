from msm_playground.grid_based_reference import grid_hopping_process
from msm_playground.msm import MSM
from msm_playground.clustering.uniform_clustering import UniformClustering
from diagnostics.diagnostics_utils import *
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sps
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--kT", type=float, default=30.0, help="Temperature of the system")
parser.add_argument("--dots-per-dim", type=int, default=100, help="Grid size")
args = parser.parse_args()

plot_folder = "./"
nparr_folder = "./"
kT = args.kT
dots_per_dim = args.dots_per_dim

A = [-200, -100, -170, 15]
a = [-1, -1, -6.5, 0.7]
b = [0, 0, 11, 0.6]
c = [-10, -10, -6.5, 0.7]
X0 = [1, 0, -0.5, -1]
Y0 = [0, 0.5, 1.5, 1]

grid = np.linspace(-2.5, 2.5, dots_per_dim)
X, Y = np.meshgrid(grid, grid)
dx = grid[1] - grid[0]
dy = grid[1] - grid[0]

U = (A[0]*np.exp(a[0]*(X-X0[0])**2 + b[0]*(X-X0[0])*(Y-Y0[0]) + c[0]*(Y-Y0[0])**2) + 
     A[1]*np.exp(a[1]*(X-X0[1])**2 + b[1]*(X-X0[1])*(Y-Y0[1]) + c[1]*(Y-Y0[1])**2) +
     A[2]*np.exp(a[2]*(X-X0[2])**2 + b[2]*(X-X0[2])*(Y-Y0[2]) + c[2]*(Y-Y0[2])**2) +
     A[3]*np.exp(a[3]*(X-X0[3])**2 + b[3]*(X-X0[3])*(Y-Y0[3]) + c[3]*(Y-Y0[3])**2)
     )

L = grid_hopping_process(U, dx, kT=kT)
E = sps.identity(L.shape[0], format='csr')

def is_inside_circle(coord: np.ndarray, center: np.ndarray, rad: float) -> np.ndarray:
    return (np.linalg.norm(coord - center, axis=-1) <= rad)
def B_circle(coord: np.ndarray) -> np.ndarray:
    return is_inside_circle(coord, center=np.array([-0.558, 1.441]), rad=0.1)
def A_circle(coord: np.ndarray) -> np.ndarray:
    return is_inside_circle(coord, center=np.array([0.623, 0.028]), rad=0.1)

cluster_centers = np.stack([X.flatten(), Y.flatten()], axis=-1)

B_labels_mask = B_circle(cluster_centers)
A_labels_mask = A_circle(cluster_centers)
B_labels = np.where(B_labels_mask)[0]
A_labels = np.where(A_labels_mask)[0]

reference_MSM = MSM(traj=None, clustering_obj=None, lag_time=1)
reference_MSM.Tmat = (L + E).toarray()
reference_committor = reference_MSM.calculate_committor(state_A_index=A_labels, state_B_index=B_labels)
reference_committor_mesh = reference_committor.reshape(dots_per_dim, dots_per_dim)

plot_2D_committor(reference_committor_mesh, grid, grid, U)
plt.savefig(plot_folder + 'reference_committor_kT_' + str(kT) + '.png')

# Save numpy arrays
np.save(nparr_folder + 'reference_committor_kT_' + str(kT) + '.npy', reference_committor_mesh)
np.save(nparr_folder + 'xgrid_kT' + str(kT) + '.npy', grid)
np.save(nparr_folder + 'ygrid_kT' + str(kT) + '.npy', grid)
np.save(nparr_folder + 'potential_kT' + str(kT) + '.npy', U)
