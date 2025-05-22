# Standard library imports
import sys

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from matplotlib.colors import to_rgb

# Local imports
from core.abstract_joint_probability_class import AbstractJointProbabilityClass
from core.experiment_runner import run_experiment
from core.io_utils import load_pickle, save_pickle

def get_result_pickle_filename() -> str:
    return f"results/exp1.pkl"

def get_tree_pickle_filename(n: int, inst_idx: int) -> str:
    return f"results/exp1/graphs/{n}_{inst_idx}.pkl"

def create_all_experiment_trees(all_n: list, num_graphs: int, rng_seed: int) -> None:
    graph_rng = np.random.default_rng(rng_seed)
    for n in all_n:
        for inst_idx in range(num_graphs):
            G = nx.Graph()
            G.add_nodes_from(np.arange(n))
            G.add_edges_from((int(nodes[i]), int(nodes[graph_rng.integers(0, i)])) for nodes in [graph_rng.permutation(n).tolist()] for i in range(1, n))
            assert nx.is_tree(G)
            print(f"n = {G.number_of_nodes()}, m = {G.number_of_edges()}, diam(G) = {max(nx.diameter(G.subgraph(cc_nodes)) for cc_nodes in nx.connected_components(G))}")

            # Save graph
            save_pickle(G, get_tree_pickle_filename(n, inst_idx))

def run_experiment1(policies_to_run: dict, experiment_result_pickle_filename: str):
    multithread = True
    train_policies = True
    covariate_length = 5
    num_graphs = 10
    num_monte_carlo_runs = 200
    rng_seed = 42

    # Create all experiment trees
    create_all_experiment_trees(all_n, num_graphs, rng_seed)

    # Run experiment
    exp_training_time = dict()
    exp_rollout_means = dict()
    exp_rollout_std_errs = dict()
    for key in policies_to_run.keys():
        n, beta = key
        all_inst_configs = []
        for inst_idx in range(num_graphs):
            inst_rng = np.random.default_rng(rng_seed)

            # Setup instance configurations
            inst_configs = dict()
            inst_configs['G_pickle_filename'] = get_tree_pickle_filename(n, inst_idx)
            inst_configs['covariates'] = inst_rng.integers(0, 2, size=(n, covariate_length))
            inst_configs['theta_unary'] = inst_rng.normal(size=AbstractJointProbabilityClass.compute_theta_length(covariate_length, 1))
            inst_configs['theta_pairwise'] = inst_rng.normal(size=AbstractJointProbabilityClass.compute_theta_length(covariate_length, 2))
            inst_configs['discount_factor'] = beta
            inst_configs['num_monte_carlo_runs'] = num_monte_carlo_runs
            inst_configs['instance_hash'] = f"exp1_{n}_{inst_idx}_{beta}_{num_monte_carlo_runs}_{rng_seed}"
            inst_configs['exp_name'] = "exp1"
            inst_configs['inst_idx'] = inst_idx
            inst_configs['n'] = n
            inst_configs['eval_rng_seed'] = 42
            all_inst_configs.append(inst_configs)

        # Compute results
        print(f"========== Solving n = {n}, beta = {beta} ==========")
        policy_indices = policies_to_run[key]
        all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec = run_experiment(all_inst_configs, policy_indices, multithread, train_policies)

        # Process and store results
        exp_training_time[key] = all_training_time
        exp_rollout_means[key] = {
            policy_idx: (
                np.mean(all_mean_vec[policy_idx], axis=0),
                np.mean(all_disc_mean_vec[policy_idx], axis=0),
                np.mean(all_time_mean_vec[policy_idx], axis=0)
            )
            for policy_idx in policy_indices
        }
        exp_rollout_std_errs[key] = {
            policy_idx: (
                np.sqrt(np.sum(np.square(all_std_vec[policy_idx]), axis=0)),
                np.sqrt(np.sum(np.square(all_disc_std_vec[policy_idx]), axis=0)),
                np.sqrt(np.sum(np.square(all_time_std_vec[policy_idx]), axis=0))
            )
            for policy_idx in policy_indices
        }

    # Save output so we can plot separately
    save_pickle(tuple([exp_training_time, exp_rollout_means, exp_rollout_std_errs]), experiment_result_pickle_filename)

def plot_row(policies_to_run: dict, policy_labels: list, policy_colors: list, policy_styles: list, experiment_result_pickle_filename: str) -> None:
    _, exp_rollout_means, exp_rollout_std_errs = load_pickle(experiment_result_pickle_filename)

    all_lines = []
    key_to_axes = {
        (10, 0.5): 0,
        (10, 0.7): 1,
        (10, 0.9): 2,
    }

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    plt.rcParams['font.size'] = 20
    for key, ax_idx in key_to_axes.items():
        n, beta = key
        policy_indices = policies_to_run[key]
        X = np.arange(1, n+1)
        axes[ax_idx].set_xlim(min(X), max(X))
        axes[ax_idx].set_xticks(np.round(np.linspace(min(X), max(X), 8)))
        axes[ax_idx].set_title(fr"{n} nodes, $\beta =$ {beta}")
        for i in policy_indices:
            line_handle, = axes[ax_idx].plot(X, exp_rollout_means[key][i][1], ls=policy_styles[i], color=policy_colors[i], lw=3)
            print("Rollout means", key, policy_labels[i], exp_rollout_means[key][i][1][-1])
            base_color = to_rgb(line_handle.get_color())
            fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
            axes[ax_idx].fill_between(
                X,
                exp_rollout_means[key][i][1] - exp_rollout_std_errs[key][i][1],
                exp_rollout_means[key][i][1] + exp_rollout_std_errs[key][i][1],
                color=fill_color,
                alpha=0.5
            )
            axes[ax_idx].axvline(x=X[len(X)//2-1], color='gray', linestyle='--', linewidth=2)
            all_lines.append(line_handle)

    # Annotate
    X_label = "Number of tests"
    Y_label = fr"Expected accumulated discounted rewards"
    fig.text(0.5, -0.05, X_label, ha='center')
    fig.text(0.07, 0.53, Y_label, va='center', rotation='vertical')
    fig.legend(all_lines, policy_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.95))

    # Save plot
    plot_fname = f"results/plots/exp1-main-paper.png"
    os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname, dpi=300, bbox_inches = 'tight')

def plot_3x3(policies_to_run: dict, policy_labels: list, policy_colors: list, policy_styles: list, experiment_result_pickle_filename: str) -> None:
    _, exp_rollout_means, exp_rollout_std_errs = load_pickle(experiment_result_pickle_filename)

    key_to_axes = {
        (10, 0.5): (0, 0),
        (10, 0.7): (0, 1),
        (10, 0.9): (0, 2),
        (50, 0.5): (1, 0),
        (50, 0.7): (1, 1),
        (50, 0.9): (1, 2),
        (100, 0.5): (2, 0),
        (100, 0.7): (2, 1),
        (100, 0.9): (2, 2),
    }
    print(exp_rollout_means.keys())
    print(key_to_axes)

    # Plot discounted
    plt.rcParams['font.size'] = 20
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    all_lines = []
    for key, ax_idx in key_to_axes.items():
        n, beta = key
        policy_indices = policies_to_run[key]
        X = np.arange(1, n+1)
        axes[ax_idx].set_xlim(min(X), max(X))
        axes[ax_idx].set_xticks(np.round(np.linspace(min(X), max(X), 8)))
        axes[ax_idx].set_title(fr"{n} nodes, $\beta =$ {beta}")
        for i in policy_indices:
            line_handle, = axes[ax_idx].plot(X, exp_rollout_means[key][i][1], ls=policy_styles[i], color=policy_colors[i], lw=3)
            base_color = to_rgb(line_handle.get_color())
            fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
            axes[ax_idx].fill_between(
                X,
                exp_rollout_means[key][i][1] - exp_rollout_std_errs[key][i][1],
                exp_rollout_means[key][i][1] + exp_rollout_std_errs[key][i][1],
                color=fill_color,
                alpha=0.5
            )
            axes[ax_idx].axvline(x=X[len(X)//2-1], color='gray', linestyle='--', linewidth=2)
            all_lines.append(line_handle)

    # Annotate
    X_label = "Number of tests"
    Y_label = fr"Expected accumulated discounted rewards"
    fig.text(0.5, 0.07, X_label, ha='center')
    fig.text(0.07, 0.5, Y_label, va='center', rotation='vertical')
    fig.legend(all_lines, policy_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.9))

    # Save plot
    plot_fname = f"results/plots/exp1-appendix.png"
    os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname, dpi=300, bbox_inches = 'tight')

    # Plot undiscounted
    plt.rcParams['font.size'] = 20
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    all_lines = []
    for key, ax_idx in key_to_axes.items():
        n, beta = key
        policy_indices = policies_to_run[key]
        X = np.arange(1, n+1)
        axes[ax_idx].set_xlim(min(X), max(X))
        axes[ax_idx].set_xticks(np.round(np.linspace(min(X), max(X), 8)))
        axes[ax_idx].set_title(fr"{n} nodes, $\beta =$ {beta}")
        for i in policy_indices:
            line_handle, = axes[ax_idx].plot(X, exp_rollout_means[key][i][0], ls=policy_styles[i], color=policy_colors[i], lw=3)
            base_color = to_rgb(line_handle.get_color())
            fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
            axes[ax_idx].fill_between(
                X,
                exp_rollout_means[key][i][0] - exp_rollout_std_errs[key][i][0],
                exp_rollout_means[key][i][0] + exp_rollout_std_errs[key][i][0],
                color=fill_color,
                alpha=0.5
            )
            axes[ax_idx].axvline(x=X[len(X)//2-1], color='gray', linestyle='--', linewidth=2)
            all_lines.append(line_handle)

    # Annotate
    X_label = "Number of tests"
    Y_label = fr"Expected accumulated undiscounted rewards"
    fig.text(0.5, 0.07, X_label, ha='center')
    fig.text(0.07, 0.5, Y_label, va='center', rotation='vertical')
    fig.legend(all_lines, policy_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.9))

    # Save plot
    plot_fname = f"results/plots/exp1-undiscounted-appendix.png"
    os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname, dpi=300, bbox_inches = 'tight')

def plot(policies_to_run: dict, experiment_result_pickle_filename: str) -> None:
    # See: https://stackoverflow.com/questions/33337989/how-to-draw-more-type-of-lines-in-matplotlib
    policy_labels = ["Random", "Greedy", "DQNPolicy", "Gittins", "Optimal"]
    policy_colors = ['red', 'blue', 'green', 'orange', 'purple']
    policy_styles = ['dotted', 'dashed', 'dashdot', 'solid', (0, (3, 2, 1, 2))]

    # Plot row: n = [10, 50, 100] and beta = [0.9]
    plot_row(policies_to_run, policy_labels, policy_colors, policy_styles, experiment_result_pickle_filename)

    # Plot 3x3: n = [10, 50, 100] and beta = [0.5, 0.7, 0.9]
    plot_3x3(policies_to_run, policy_labels, policy_colors, policy_styles, experiment_result_pickle_filename)

if __name__ == "__main__":
    print(sys.argv)
    just_plot = bool(int(sys.argv[1]))
    print(just_plot)

    # policies are ["Random", "Greedy", "DQNPolicy", "Gittins", "Optimal"]
    all_n = [10, 50, 100]
    all_beta = [0.5, 0.7, 0.9]
    policies_to_run = dict()
    for n in all_n:
        for beta in all_beta:
            key = (n, beta)
            if n <= 12:
                policies_to_run[key] = [0, 1, 2, 3, 4]
            else:
                policies_to_run[key] = [0, 1, 2, 3]
    experiment_result_pickle_filename = get_result_pickle_filename()

    if not just_plot:
        run_experiment1(policies_to_run, experiment_result_pickle_filename)
    
    # Generate plots for exp1
    plot(policies_to_run, experiment_result_pickle_filename)
