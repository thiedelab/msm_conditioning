from msm_playground.clustering.abstract_clustering import AbstractClustering
from msm_playground.traj_utils import sample_from_grid
from diagnostics.diagnostics_utils import *
from msm_playground.msm import MSM
from msm_playground import numba_sampler as sampler
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sympy import Symbol, Derivative, lambdify, exp
import matplotlib.colors as colors
from typing import Optional
import os

class MBPipelineBlocks():
    def __init__(self):
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.plot_folder = "plots/mb/"

        self.setup_potential()
        self.setup_dummy_global_parameters()

        # Dicts, keys of which are msm names
        self.MSMs = dict()
        self.A_labels = dict()
        self.B_labels = dict()

    def setup_potential(self):
        x = Symbol('x')
        y = Symbol('y')
        A = [-200, -100, -170, 15]
        a = [-1, -1, -6.5, 0.7]
        b = [0, 0, 11, 0.6]
        c = [-10, -10, -6.5, 0.7]
        X = [1, 0, -0.5, -1]
        Y = [0, 0.5, 1.5, 1]
        U = 0
        for i in range(0, 4):
            U += A[i]*exp(a[i]*(x-X[i])**2 + b[i]*(x-X[i])*(y-Y[i]) + c[i]*(y-Y[i])**2)

        Fx = -Derivative(U, x).doit()
        Fy = -Derivative(U, y).doit()
        self.F_lambda = lambdify([x, y], [Fx, Fy])
        self.U_lambda = lambdify([x, y], U)

    def plot_function(self, potential, min_coord=[-2, -0.5], max_coord=[1, 2], label="potential"):
        x = np.linspace(min_coord[0], max_coord[0], 100)
        y = np.linspace(min_coord[1], max_coord[1], 100)
        X, Y = np.meshgrid(x, y)
        U = potential(X, Y)
        vmax = np.max(U)
        levels = np.append(np.linspace(np.min(U), 100, 10), np.array([vmax/i for i in reversed(range(1, 3))]))
        levels = np.sort(levels)
        norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)
        plt.contourf(X, Y, U, levels=levels, cmap='Blues', norm=norm)
        plt.colorbar().set_label(label)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('MB potential')

    def setup_dummy_global_parameters(self):
        self.kT = None
        self.n_simulations = None
        self.n_traj = None
        self.n_clusters_per_dim = None

        self.nsteps = None
        self.burnin = None
        self.dt = None

    def setup_global_parameters(self):
        self.kT = 30.0
        self.n_simulations = 3
        self.n_traj = 5
        self.n_clusters_per_dim = 50

        self.nsteps = 8000
        self.burnin = 1000
        self.dt = 1e-3 / 4

    def setup_bolzman_prob_on_grid(self, xgrid: npt.ArrayLike, ygrid: npt.ArrayLike):
        self.xgrid = xgrid
        self.ygrid = ygrid
        mesh = np.meshgrid(xgrid, ygrid)
        self.mesh_2d = np.vstack(list(map(np.ravel, mesh))).T
        self.ref_potential = self.U_lambda(self.mesh_2d[:, 0], self.mesh_2d[:, 1]).reshape((self.n_clusters_per_dim, self.n_clusters_per_dim))
        self.boltzmann_prob = np.exp(-self.ref_potential / self.kT)
        self.boltzmann_prob /= np.sum(self.boltzmann_prob)

    def load_references(self, do_downscale: Optional[bool] = True, plot_reference: Optional[bool] = True):
        self.ref_comm = np.load((self.abs_path + '/reference/reference_committor_kT_' + str(self.kT) + '.npy'))
        self.ref_mfpt = np.load((self.abs_path + '/reference/reference_mfpt_kT_' + str(self.kT) + '.npy'))
        self.ref_potential = np.load(self.abs_path + '/reference/potential_kT' + str(self.kT) + '.npy')
        self.xgrid = np.load(self.abs_path + '/reference/xgrid_kT' + str(self.kT) + '.npy')
        self.ygrid = np.load(self.abs_path + '/reference/ygrid_kT' + str(self.kT) + '.npy')

        if(do_downscale):
            self.ref_comm = downscale_func_on_grid(self.ref_comm, (self.n_clusters_per_dim, self.n_clusters_per_dim))
            self.ref_potential = downscale_func_on_grid(self.ref_potential, (self.n_clusters_per_dim, self.n_clusters_per_dim))
            self.xgrid = np.linspace(np.min(self.xgrid), np.max(self.xgrid), self.n_clusters_per_dim)
            self.ygrid = np.linspace(np.min(self.ygrid), np.max(self.ygrid), self.n_clusters_per_dim)

        self.setup_bolzman_prob_on_grid(self.xgrid, self.ygrid)

        if(plot_reference):
            plot_2D_committor(self.ref_comm, self.xgrid, self.ygrid, self.ref_potential)
            plt.savefig(self.plot_folder + 'reference_committor.png')
            plt.close()

    def is_inside_circle(self, coord: npt.ArrayLike, center: npt.ArrayLike, rad: float) -> npt.ArrayLike:
        return (np.linalg.norm(coord - center, axis=-1) <= rad)
    def B_circle(self, coord: npt.ArrayLike) -> npt.ArrayLike:
        return self.is_inside_circle(coord, center=np.array([-0.558, 1.441]), rad=0.2)
    def A_circle(self, coord: npt.ArrayLike) -> npt.ArrayLike:
        return self.is_inside_circle(coord, center=np.array([0.623, 0.028]), rad=0.2)

    def plot_clustering_and_potential(self, clustering_obj: AbstractClustering,
                                      traj: List[npt.ArrayLike], save_plot: Optional[bool] = False):
        clustering_obj(np.concatenate(traj))
        cluster_centers = np.array([clustering_obj.get_cluster_center(i) for i in range(clustering_obj.n_clusters)])
        B_labels_mask = self.B_circle(clustering_obj.cluster_centers)
        A_labels_mask = self.A_circle(clustering_obj.cluster_centers)
        B_labels = np.where(B_labels_mask)[0]
        A_labels = np.where(A_labels_mask)[0]

        cmap = cm.get_cmap('viridis')
        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1],
                's', markersize=4, alpha=0.15, color="black")
        plt.plot(cluster_centers[A_labels, 0],
                cluster_centers[A_labels, 1], 's', markersize=4, alpha=0.7, color=cmap(0.0))
        plt.plot(cluster_centers[B_labels, 0],
                cluster_centers[B_labels, 1], 's', markersize=4, alpha=0.7, color=cmap(1.0))
        self.plot_function(self.U_lambda, np.min(traj, axis=(0, 1)), np.max(traj, axis=(0, 1)))

        for traj_idx in range(traj.shape[0]):
            plt.scatter(traj[traj_idx, :, 0], traj[traj_idx, :, 1], color=clusters_colors[traj_idx % len(clusters_colors)], alpha=0.4, s=0.1)
        if(save_plot):
            plt.savefig(self.plot_folder + "mb_traj_and_clusters", dpi=600)
            plt.close()

    def sample_init_coord(self) -> npt.ArrayLike:
        return sample_from_grid(grid=self.mesh_2d, probabilities=self.boltzmann_prob.flatten())
    
    def simulate_simulation(self, seed: Optional[int]=None,
                            grid_mean_point: npt.ArrayLike | None = None,
                            grid_dimensions: npt.ArrayLike | None = None) -> List[npt.ArrayLike]:
        if(seed is not None):
            sampler.set_seed(seed)

        if(grid_mean_point is not None and grid_dimensions is not None):
            xgrid = np.linspace(grid_mean_point[0] - grid_dimensions[0] / 2, grid_mean_point[0] + grid_dimensions[0] / 2, self.n_clusters_per_dim)
            ygrid = np.linspace(grid_mean_point[1] - grid_dimensions[1] / 2, grid_mean_point[1] + grid_dimensions[1] / 2, self.n_clusters_per_dim)
            self.setup_bolzman_prob_on_grid(xgrid, ygrid)
        cfg_0 = self.sample_init_coord()

        if (grid_mean_point is None or grid_dimensions is None):
            reject_function = None
        else:
            reject_function = sampler.get_outside_of_grid_function(grid_mean_point, grid_dimensions)

        traj = np.array([sampler.sample_overdamped_langevin(
            cfg_0, sampler.mueller_brown_force, nsteps=self.nsteps, burnin=self.burnin, dt=self.dt, kT=self.kT
        ) for _ in range(self.n_traj)])

        return traj

    def init_msm(self, msm: MSM, msm_name: str):
        self.MSMs[msm_name] = msm

        B_labels_mask = self.B_circle(msm.cluster_centers)
        A_labels_mask = self.A_circle(msm.cluster_centers)
        self.B_labels[msm_name] = np.where(B_labels_mask)[0]
        self.A_labels[msm_name] = np.where(A_labels_mask)[0]

    def is_bad_simulation(self) -> bool:
        """
        Says whether the generated simulation is appropriate for the MSMs.
        To run properly, initialuze MSMs first.
        Function checks if the simulation visits all the states that are needed for the MSMs after truncating "bad" states.
        """

        anything_is_none = False
        for msm_name, msm in self.MSMs.items():
            anything_is_none = anything_is_none or (msm is None)
            anything_is_none = anything_is_none or (msm.Tmat is None)
            anything_is_none = anything_is_none or (msm.active_set is None)

        if(anything_is_none):
            return True
        
        bad_for_simulation = False
        for msm_name, msm in self.MSMs.items():
            bad_for_simulation = bad_for_simulation or (count_hits(msm.active_set, self.A_labels[msm_name]) == 0 or
                                  count_hits(msm.active_set, self.B_labels[msm_name]) == 0)
        
        return bad_for_simulation
    
    def calculate_boltzmann_prob_on_clusters(self, msm: MSM) -> npt.ArrayLike:
        self.boltzmann_prob_on_clusters = calc_func_values_on_grid(ref_func=self.boltzmann_prob.T,
                                                                ref_func_grid=(self.xgrid, self.ygrid),
                                                                exp_func_domain=msm.cluster_centers)
        return self.boltzmann_prob_on_clusters

    
    def reweight_error(self, pointwise_error: npt.ArrayLike, msm: MSM) -> npt.ArrayLike:
        # print("Is reversible: ", msm.is_reversible())
        # get reference committor values on msm.cluster_centers points
        ref_committor_on_clusters = calc_func_values_on_grid(ref_func=self.ref_comm.T,
                                                                ref_func_grid=(self.xgrid, self.ygrid),
                                                                exp_func_domain=msm.cluster_centers)

        self.calculate_boltzmann_prob_on_clusters(msm)
        weights = self.boltzmann_prob_on_clusters * ref_committor_on_clusters * (1 - ref_committor_on_clusters)
        return pointwise_error * weights

    def get_committor_pntwise_error(self, msm_name: str, stopped_process: bool=False) -> npt.ArrayLike:
        msm = self.MSMs[msm_name]
        msm.calculate_committor(self.A_labels[msm_name], self.B_labels[msm_name], stopped_process=stopped_process)
        
        pntwise_error = calc_func_error_on_grid(exp_func=msm.committor, ref_func=self.ref_comm.T, 
                                    exp_func_domain=msm.cluster_centers, ref_func_grid=(self.xgrid, self.ygrid))
        pntwise_error = self.reweight_error(pntwise_error, msm)
        return pntwise_error
    
    def draw_exp_function_2D_plot(self, exp_func_domain: npt.ArrayLike, exp_func: npt.ArrayLike,
                                  plot_name: str, label: str):
        plot_name = plot_name + ""

        cmap = cm.get_cmap('viridis')
        min_x, min_y = np.min(exp_func_domain, axis=0)
        max_x, max_y = np.max(exp_func_domain, axis=0)
        self.plot_function(self.U_lambda, [min_x, min_y], [max_x, max_y])
        plt.scatter(exp_func_domain[:, 0], exp_func_domain[:, 1],
                    c=exp_func, cmap=cmap, marker='s', s=400, alpha=0.7)
        plt.colorbar().set_label(label)
        plt.title(label +", " + plot_name, fontsize=8)
        plt.savefig(self.plot_folder + label + ", " + plot_name, dpi=300)
        plt.close()
    
    def get_mfpt_pntwise_error(self, msm_name: str, stopped_process=False) -> npt.ArrayLike:
        msm = self.MSMs[msm_name]
        mfpt = msm.calculate_mfpt(self.B_labels[msm_name], stopped_process=stopped_process)
        mfpt_in_seconds = mfpt * self.dt
        pntwise_error = calc_func_error_on_grid(exp_func=mfpt_in_seconds, ref_func=self.ref_mfpt.T, 
                                    exp_func_domain=msm.cluster_centers, ref_func_grid=(self.xgrid, self.ygrid),
                                    relative_error=False)
        pntwise_error = self.reweight_error(pntwise_error, msm)
        return pntwise_error[~np.isnan(pntwise_error)]
