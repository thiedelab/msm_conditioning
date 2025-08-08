# Run this script from the root directory of the repository

from diagnostics.mueller_brown.mb_diagn_blocks import MBPipelineBlocks
from msm_playground.clustering.uniform_clustering import UniformClustering
from msm_playground.msm import MSM
from diagnostics.diagnostics_utils import get_measurements_error, narrow_down_exp_func_domain
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kT", type=float, default=30.0, help="Temperature of the system")
parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters")
parser.add_argument("--n-traj", type=int, default=1, help="Number of trajectories")
parser.add_argument("--traj-length", type=int, default=10000 * 30, help="Length of each trajectory")
parser.add_argument("--n-sim", type=int, default=1, help="Number of simulations per configuration to get the error bars")
parser.add_argument("--max-lag-time", type=int, default=10, help="committor function is benchmarked on lag time from min-lat-time to max-lag-time")
parser.add_argument("--min-lag-time", type=int, default=1, help="committor function is benchmarked on lag time from min-lat-time to max-lag-time")
parser.add_argument("--stride", type=int, default=1, help="Stride for lag time")
parser.add_argument("--seed", type=int, default=2, help="Seed for random number generator")
args = parser.parse_args()

kT = args.kT
n_clusters = args.n_clusters
n_traj = args.n_traj
traj_length = args.traj_length
n_simulations = args.n_sim
max_lag_time = args.max_lag_time + 1
min_lag_time = args.min_lag_time
lag_time_stride = args.stride
seed = args.seed
burnin = 1000
dt = 1e-3 / 4
plot_folder = "plots/mb/"

mb_pb = MBPipelineBlocks()
mb_pb.setup_global_parameters() # Default values are used
mb_pb.load_references(plot_reference=False)

mb_pb.kT = kT
mb_pb.n_simulations = n_simulations
mb_pb.n_traj = n_traj
mb_pb.n_clusters_per_dim = n_clusters
mb_pb.nsteps = traj_length
mb_pb.burnin = burnin
mb_pb.dt = dt
mb_pb.plot_folder = plot_folder

lag_times_list = np.arange(min_lag_time, max_lag_time, lag_time_stride, dtype=np.float16)
vanilla_mfpt_error = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_mfpt_error =  np.full(len(lag_times_list), fill_value=np.nan)
vanilla_mfpt_error_errbar = np.full(len(lag_times_list), fill_value=np.nan)
lagstop_mfpt_error_errbar = np.full(len(lag_times_list), fill_value=np.nan)

my_clustering = UniformClustering(n_clusters_per_dimension=n_clusters)

max_tries = 10

vanilla_msm_name = "exp_vanilla_msm"
lagstop_msm_name = "exp_lagstop_msm"

keep_same_traj = True
np.random.seed(seed)
initial_state = np.random.get_state()
for tau_idx, tau in enumerate(lag_times_list.copy().astype(int)):
    print("\ntau =", tau, "\n")
    # Averaging errors over simulations
    vanilla_mfpt_errors_over_simulations = np.full(n_simulations, fill_value=np.nan)
    lagstop_mfpt_errors_over_simulations = np.full(n_simulations, fill_value=np.nan)
    for simulation_idx in range(n_simulations):
        params_str = ("lag_time={}".format(tau) + " traj_length={}".format(traj_length) +
                       " n_traj={}".format(n_traj) + " seed={}".format(seed))
        
        # Often generated trajectory does not visit A or B
        # Resimulate until we get a trajectory that visits both A and B at least once
        tries = 0
        simulation_is_bad = True
        while(tries < max_tries and (simulation_is_bad)):
            if(keep_same_traj):
                np.random.set_state(initial_state)                    
            if(tries > 0):
                print("Resimulating trajectories " + str(tries) + "/" + str(max_tries))
                if(keep_same_traj):
                    raise ValueError("same_trajectoy is True, but we have already tried to simulate the same trajectory")
            tries += 1

            if(not keep_same_traj or tau == min_lag_time):
                #print(not keep_same_traj, tau == min_lag_time)
                print("Generating new trajectories...")
                traj = mb_pb.simulate_simulation(seed=seed)

            exp_vanilla_msm = MSM(traj, my_clustering, tau)
            exp_lagstop_msm = MSM(traj, my_clustering, tau)
            mb_pb.init_msm(msm=exp_vanilla_msm, msm_name=vanilla_msm_name)
            mb_pb.init_msm(msm=exp_lagstop_msm, msm_name=lagstop_msm_name)

            exp_lagstop_msm.Tmat = np.copy(exp_vanilla_msm.Tmat) # not None, never used
            exp_vanilla_msm.calculate_correlation_matrix(stopped_process=False)

            simulation_is_bad = mb_pb.is_bad_simulation()
        if(simulation_is_bad):
            print("Simulation is still bad. Skipping this iteration...")
            continue
        print("Simulation is good. Proceeding to the next step...")

        pntwise_error = mb_pb.get_mfpt_pntwise_error(msm_name=vanilla_msm_name, stopped_process=False)
        narrowed_exp_domain, pntwise_error = narrow_down_exp_func_domain(mb_pb.boltzmann_prob_on_clusters,
                                                                         exp_vanilla_msm.cluster_centers, pntwise_error)
        mb_pb.draw_exp_function_2D_plot(narrowed_exp_domain, pntwise_error, plot_name=params_str, label="Vanilla mfpt error")
        print("Vanilla mfpt error =", np.sum(pntwise_error))
        vanilla_mfpt_errors_over_simulations[simulation_idx] = np.sum(pntwise_error)

        pntwise_error = mb_pb.get_mfpt_pntwise_error(msm_name=lagstop_msm_name, stopped_process=True)
        narrowed_exp_domain, pntwise_error = narrow_down_exp_func_domain(mb_pb.boltzmann_prob_on_clusters,
                                                                         exp_lagstop_msm.cluster_centers, pntwise_error)
        mb_pb.draw_exp_function_2D_plot(narrowed_exp_domain, pntwise_error, plot_name=params_str, label="Lagstop mfpt error")
        print("Lagstop mfpt error =", np.sum(pntwise_error))
        lagstop_mfpt_errors_over_simulations[simulation_idx] = np.sum(pntwise_error)

    vanilla_success_mask = ~np.isnan(vanilla_mfpt_errors_over_simulations)
    lagstop_success_mask = ~np.isnan(lagstop_mfpt_errors_over_simulations)
    if(np.all(~vanilla_success_mask) or np.all(~lagstop_success_mask)):
        lag_times_list[tau_idx] = np.nan
        continue
    vanilla_mfpt_error[tau_idx] = np.mean(vanilla_mfpt_errors_over_simulations[vanilla_success_mask])
    lagstop_mfpt_error[tau_idx] = np.mean(lagstop_mfpt_errors_over_simulations[lagstop_success_mask])
    
    vanilla_mfpt_error_errbar[tau_idx] = get_measurements_error(
        vanilla_mfpt_errors_over_simulations[vanilla_success_mask])
    lagstop_mfpt_error_errbar[tau_idx] = get_measurements_error(
        lagstop_mfpt_errors_over_simulations[lagstop_success_mask])

success_mask = ~np.isnan(lag_times_list)

lag_times_list = lag_times_list[success_mask]
vanilla_mfpt_error = vanilla_mfpt_error[success_mask]
lagstop_mfpt_error = lagstop_mfpt_error[success_mask]
vanilla_mfpt_error_errbar = vanilla_mfpt_error_errbar[success_mask]
lagstop_mfpt_error_errbar = lagstop_mfpt_error_errbar[success_mask]

np.save("data/mb_mfpt_error_of_lag_time_ntraj_{}_trajlength_{}_seed_{}.npy".format(n_traj, traj_length, seed), [lag_times_list, vanilla_mfpt_error, lagstop_mfpt_error, vanilla_mfpt_error_errbar, lagstop_mfpt_error_errbar])
