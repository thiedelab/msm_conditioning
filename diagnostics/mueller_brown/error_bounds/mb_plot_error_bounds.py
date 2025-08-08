import matplotlib.pyplot as plt
import numpy as np

n_traj = 1
traj_length = 7500000
seed = 42

(lag_times_list,
         vanilla_true_error, lagstop_true_error,
         vanilla_error_upper_bound, lagstop_error_upper_bound,
         vanilla_matrix_diff_norms, lagstop_matrix_diff_norms,
         vanilla_matrix_norms, lagstop_matrix_norms,
         vanilla_inv_matrix_norms, lagstop_inv_matrix_norms,
         vanilla_true_committor_norms, lagstop_true_committor_norms) = np.load(
    "data/mb_committor_error_of_lag_time_ntraj_{}_trajlength_{}_seed_{}.npy".format(n_traj, traj_length, seed))

plt.plot(lag_times_list, vanilla_true_error, label="Vanilla True Error",
         marker="o", markersize=4, color="orange")
plt.plot(lag_times_list, vanilla_error_upper_bound, label="Vanilla Error Upper Bound",
         marker="o", markersize=2, color="orange", linestyle="--", alpha=0.5)
plt.plot(lag_times_list, lagstop_true_error, label="Lagstop True Error",
         marker="o", markersize=4, color="blue")
plt.plot(lag_times_list, lagstop_error_upper_bound, label="Lagstop Error Upper Bound",
         linestyle="--", marker="o", markersize=2, color="blue", alpha=0.5)

plt.yscale('log')
plt.xlabel("Lag Times")
plt.ylabel("Error")
plt.legend()

plt.savefig("data/mb_committor_error_of_lag_time_ntraj_{}_trajlength_{}_seed_{}".format(n_traj, traj_length, seed))
plt.close()



plt.plot(lag_times_list, vanilla_matrix_diff_norms, label="Ref - Exp Vanilla Matrix Norm",
        marker="o", markersize=4, color="orange")
plt.plot(lag_times_list, lagstop_matrix_diff_norms, label="Ref - Exp Lagstop Matrix Norm",
        marker="o", markersize=4, color="blue")
plt.xlabel("Lag Times")
plt.ylabel("Matrix Norm")
plt.legend()

plt.savefig("data/mb_matrix_diff_norms_of_lag_time_ntraj_{}_trajlength_{}_seed_{}".format(n_traj, traj_length, seed))
plt.close()