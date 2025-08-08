import numpy as np
import argparse
import matplotlib.pyplot as plt
from enum import Enum

class FunctionType(Enum):
    COMMITTOR = "committor"
    MFPT = "mfpt"

def validate_function_type(value):
    try:
        return FunctionType[value]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid function type: {value}")

# Create the argument parser
parser = argparse.ArgumentParser(description='Plot experimental results.')

# Add the arguments
parser.add_argument('--n_traj', type=int, default=1, help='Number of trajectories')
parser.add_argument('--traj_length', type=int, default=10000 * 30, help='Length of trajectories')
parser.add_argument('--seed', type=int, default=2, help='Random seed')
parser.add_argument('--dt', type=float, default=1e-3 / 4, help='Time step in seconds')
parser.add_argument('--exp_function', type=validate_function_type, default=FunctionType.COMMITTOR, choices=FunctionType, help='Experimental function type: "committor" or "mfpt"')

# Parse the arguments
args = parser.parse_args()

# Retrieve the values
n_traj = args.n_traj
traj_length = args.traj_length
seed = args.seed
dt = args.dt
exp_function_name = args.exp_function.value # "committor" or "mfpt"

plot_folder = "plots/mb/"

# Load the data
data = np.load("data/mb_{}_error_of_lag_time_ntraj_{}_trajlength_{}_seed_{}.npy".format(exp_function_name, n_traj, traj_length, seed))

# Retrieve the data
lag_times_list, vanilla_error, lagstop_error, vanilla_error_errbar, lagstop_error_errbar = data

if(~np.all(np.isnan(vanilla_error_errbar)) and vanilla_error_errbar.size != 0):
    plt.errorbar(lag_times_list, vanilla_error, yerr=vanilla_error_errbar,
                fmt='o', capsize=3, alpha=0.5, color="blue", label="{} error".format(exp_function_name))
if(~np.all(np.isnan(lagstop_error_errbar)) and lagstop_error_errbar.size != 0):
    plt.errorbar(lag_times_list, lagstop_error, yerr=lagstop_error_errbar,
                fmt='o', capsize=3, alpha=0.5, color="orange", label="lagstop {} error".format(exp_function_name))

plt.scatter(lag_times_list, vanilla_error, color="blue", alpha=1, label="{} error".format(exp_function_name), s=2)
plt.scatter(lag_times_list, lagstop_error, color="orange", alpha=1, label="lagstop {} error".format(exp_function_name), s=2)
plt.legend()
plt.xlabel("lag time, " + "{:.2e}".format(dt) + " s")
plt.ylabel("error")
plt.title("MB {} error of lag time {} traj length".format(exp_function_name, traj_length))
plt.savefig(plot_folder + "mb_error_of_lag_time_ntraj_{}_trajlength_{}_seed_{}".format(n_traj, traj_length, seed), dpi=300)
plt.close()