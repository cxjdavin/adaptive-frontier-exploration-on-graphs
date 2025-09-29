# Standard library imports
import sys

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.colors import to_rgb

# Local imports
from core.io_utils import load_pickle

if __name__ == "__main__":
    print(sys.argv)
    std_idx = int(sys.argv[1])
    cc_threshold = int(sys.argv[2])
    inst_idx = int(sys.argv[3])
    num_eps_rng_seed = int(sys.argv[4])
    
    std_names = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
    assert 0 <= std_idx and std_idx < len(std_names)
    std_name = std_names[std_idx]

    plot_fname = f"noisyQ_{std_name}_threshold{cc_threshold}_{inst_idx}_combined.png"
    policy_labels = ["Random", "Greedy", "DQNPolicy", "Gittins"]
    policy_colors = ['red', 'blue', 'green', 'orange']
    policy_styles = ['dotted', 'dashed', 'dashdot', 'solid']
    policy_indices = [0, 1, 2, 3]

    all_eps = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_eps_rng_seed = 10

    fig, axes = plt.subplots(nrows=1, ncols=len(all_eps), figsize=(2 + 6*len(all_eps), 5), sharex=True, sharey=True)
    plt.rcParams['font.size'] = 22
    all_lines = []
    for idx in range(len(all_eps)):
        eps = all_eps[idx]
        eps_all_mean_vec = {policy_idx: [] for policy_idx in policy_indices}
        for policy_idx in policy_indices:
            for eps_rng_seed in range(num_eps_rng_seed):
                experiment_result_pickle_filename = f"results/noisyQ_{std_name}_threshold{cc_threshold}_{inst_idx}_{eps}_{eps_rng_seed}.pkl"
                stats, all_training_time, all_mean_vec, _, all_disc_mean_vec, _, all_time_mean_vec, all_time_std_vec = load_pickle(experiment_result_pickle_filename)
                eps_all_mean_vec[policy_idx].append(all_mean_vec[policy_idx])

        X_axis = np.array(np.arange(stats['n']))/stats['n']
        for policy_idx in policy_indices:
            max_y = np.max(np.array(eps_all_mean_vec[policy_idx]))
            mean = np.mean(eps_all_mean_vec[policy_idx], axis=0).squeeze()
            std_err = (np.std(eps_all_mean_vec[policy_idx], axis=0) / np.sqrt(num_eps_rng_seed)).squeeze()
            line_handle, = axes[idx].plot(X_axis, mean / max_y, ls=policy_styles[policy_idx], color=policy_colors[policy_idx], lw=3)
            base_color = to_rgb(line_handle.get_color())
            fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
            axes[idx].fill_between(
                X_axis,
                (mean - std_err) / max_y,
                (mean + std_err) / max_y,
                color=fill_color,
                alpha=0.5
            )
            all_lines.append(line_handle)
        axes[idx].axvline(x=X_axis[int(len(X_axis)*0.25)], color='gray', linestyle='--', linewidth=2)
        axes[idx].axvline(x=X_axis[int(len(X_axis)*0.5)], color='gray', linestyle='--', linewidth=2)
        axes[idx].axvline(x=X_axis[int(len(X_axis)*0.75)], color='gray', linestyle='--', linewidth=2)
        axes[idx].set_title(fr"$\varepsilon = {eps}$")
        if idx == 0:
            axes[idx].set_ylabel("Fraction of positive cases detected\n" + r"($\pm$ standard error over 10 runs)", fontsize=20)

    fig.text(0.5, -0.05, f"Fraction of population tested. Vertical dotted lines indicate 25%, 50%, and 75% percentage of total population being tested.", ha='center')
    fig.legend(all_lines, policy_labels[:4], loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.95), title=fr"Policies ran with $\beta =$ {stats['beta']}")
    plot_filename = f"results/plots/noisyQ_{std_name}_threshold{cc_threshold}_{std_idx}_combined.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches = 'tight')