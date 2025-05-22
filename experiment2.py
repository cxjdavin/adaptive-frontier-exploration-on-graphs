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

def get_result_pickle_filename(n: int, beta: float) -> str:
    return f"results/exp2_{n}_{beta}.pkl"

def get_tree_pickle_filename(n: int, inst_idx: int) -> str:
    return f"results/exp2/graphs/{n}_{inst_idx}.pkl"

def get_augmented_tree_pickle_filename(n: int, inst_idx: int, edges_to_add: int) -> str:
    return f"results/exp2/graphs/{n}_{inst_idx}_{edges_to_add}.pkl"

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

def run_experiment2(n: int, beta: float, policy_indices: list, all_edges_to_add: list, experiment_result_pickle_filename: str):
    multithread = True
    train_policies = True
    covariate_length = 5
    num_graphs = 10
    num_monte_carlo_runs = 200
    rng_seed = 42

    # Create all experiment trees
    create_all_experiment_trees(all_n, num_graphs, rng_seed)

    # Create all experiment graphs
    for inst_idx in range(num_graphs):
        for edges_to_add_idx in range(len(all_edges_to_add)):
            edges_to_add = all_edges_to_add[edges_to_add_idx]
            pair_idx = edges_to_add_idx * num_graphs + inst_idx
            shuffle_rng = np.random.default_rng(rng_seed)

            # Form new graph by adding edges to tree
            G = load_pickle(get_tree_pickle_filename(n, inst_idx))
            assert nx.is_tree(G)
            G_edges = set(G.edges())
            all_edges = set([(i,j) for i in range(n) for j in range(i+1, n)])
            missing_edges = list(all_edges.difference(G_edges))
            shuffle_rng.shuffle(missing_edges)
            H = nx.Graph()
            H.add_nodes_from(G.nodes)
            H.add_edges_from(G.edges)
            H.add_edges_from(missing_edges[:edges_to_add])
            H_pickle_filename = get_augmented_tree_pickle_filename(n, inst_idx, edges_to_add)
            save_pickle(H, H_pickle_filename)

    # Run experiment
    exp_rollout_means = dict()
    exp_rollout_std_errs = dict()
    exp_rollout_disc_means = dict()
    exp_rollout_disc_std_errs = dict()
    all_inst_configs = []
    for edges_to_add_idx in range(len(all_edges_to_add)):
        for inst_idx in range(num_graphs):
            edges_to_add = all_edges_to_add[edges_to_add_idx]
            pair_idx = edges_to_add_idx * num_graphs + inst_idx
            inst_rng = np.random.default_rng(rng_seed)

            # Setup instance configurations
            inst_configs = dict()
            inst_configs['G_pickle_filename'] = get_augmented_tree_pickle_filename(n, inst_idx, edges_to_add)
            inst_configs['covariates'] = inst_rng.integers(0, 2, size=(n, covariate_length))
            inst_configs['theta_unary'] = inst_rng.normal(size=AbstractJointProbabilityClass.compute_theta_length(covariate_length, 1))
            inst_configs['theta_pairwise'] = inst_rng.normal(size=AbstractJointProbabilityClass.compute_theta_length(covariate_length, 2))
            inst_configs['discount_factor'] = beta
            inst_configs['num_monte_carlo_runs'] = num_monte_carlo_runs
            inst_configs['instance_hash'] = f"exp2_{n}_{inst_idx}_{edges_to_add}_{beta}_{num_monte_carlo_runs}_{rng_seed}"
            inst_configs['exp_name'] = "exp2"
            inst_configs['inst_idx'] = pair_idx
            inst_configs['n'] = n
            inst_configs['eval_rng_seed'] = 42
            all_inst_configs.append(inst_configs)

    # Compute results
    print(f"========== Solving n = {n}, beta = {beta} ==========")
    _, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, _, _ = run_experiment(all_inst_configs, policy_indices, multithread, train_policies)

    # Process and store results
    for policy_idx in policy_indices:
        for edges_to_add_idx in range(len(all_edges_to_add)):
            edges_to_add = all_edges_to_add[edges_to_add_idx]
            dict_key = (policy_idx, n, beta, edges_to_add)
            exp_rollout_means[dict_key] = np.mean([all_mean_vec[policy_idx][edges_to_add_idx * num_graphs + inst_idx] for inst_idx in range(num_graphs)], axis=0)
            exp_rollout_std_errs[dict_key] = np.sqrt(np.sum(np.square([all_std_vec[policy_idx][edges_to_add_idx * num_graphs + inst_idx] for inst_idx in range(num_graphs)]), axis=0))
            exp_rollout_disc_means[dict_key] = np.mean([all_disc_mean_vec[policy_idx][edges_to_add_idx * num_graphs + inst_idx] for inst_idx in range(num_graphs)], axis=0)
            exp_rollout_disc_std_errs[dict_key] = np.sqrt(np.sum(np.square([all_disc_std_vec[policy_idx][edges_to_add_idx * num_graphs + inst_idx] for inst_idx in range(num_graphs)]), axis=0))

    # Save output so we can plot separately
    save_pickle(tuple([exp_rollout_means, exp_rollout_std_errs, exp_rollout_disc_means, exp_rollout_disc_std_errs]), experiment_result_pickle_filename)

def plot(n: int, beta: float, policy_indices: list, all_edges_to_add: list, experiment_result_pickle_filename: str) -> None:
    # See: https://stackoverflow.com/questions/33337989/how-to-draw-more-type-of-lines-in-matplotlib
    policy_labels = ["Random", "Greedy", "DQNPolicy", "Gittins", "Optimal"]
    policy_colors = ['red', 'blue', 'green', 'orange', 'purple']
    policy_styles = ['dotted', 'dashed', 'dashdot', 'solid', (0, (3, 2, 1, 2))]

    # Plot row: n = [50] and all_edges_to_add
    exp_rollout_means, exp_rollout_std_errs, exp_rollout_disc_means, exp_rollout_disc_std_errs = load_pickle(experiment_result_pickle_filename)

    # Plot
    plt.rcParams['font.size'] = 20
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(36, 5))
    all_lines = []
    X = np.arange(1, n+1)
    for ax_idx in range(len(all_edges_to_add)):
        edges_to_add = all_edges_to_add[ax_idx]
        axes[ax_idx].set_xlim(min(X), max(X))
        axes[ax_idx].set_xticks(np.round(np.linspace(min(X), max(X), 8)))
        for policy_idx in policy_indices:
            dict_key = (policy_idx, n, beta, edges_to_add)
            mean_for_policy = exp_rollout_disc_means[dict_key]
            std_err_for_policy = exp_rollout_disc_std_errs[dict_key]
            line_handle, = axes[ax_idx].plot(X, mean_for_policy, ls=policy_styles[policy_idx], color=policy_colors[policy_idx], lw=3)
            print(dict_key, policy_labels[policy_idx], mean_for_policy[-1])
            base_color = to_rgb(line_handle.get_color())
            fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
            axes[ax_idx].fill_between(
                X,
                mean_for_policy - std_err_for_policy,
                mean_for_policy + std_err_for_policy,
                color=fill_color,
                alpha=0.5
            )
            all_lines.append(line_handle)
            axes[ax_idx].set_title(fr"{edges_to_add} extra edges")

    # Annotate
    Y_label = fr"Expected accumulated discounted rewards"
    fig.text(0.5, 1, fr"({n} node graphs with discount factor $\beta =$ {beta})", ha='center')
    fig.text(0.5, -0.05, fr"Number of tests", ha='center')
    fig.text(0.1, 0.5, Y_label, va='center', rotation='vertical')
    fig.legend(all_lines, [policy_labels[idx] for idx in policy_indices], loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.05))

    # Save plot
    plot_fname = f"results/plots/exp2-{n}-{beta}.png"
    os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
    plt.savefig(plot_fname, dpi=300, bbox_inches = 'tight')

if __name__ == "__main__":
    print(sys.argv)
    just_plot = bool(int(sys.argv[1]))
    n = int(sys.argv[2])
    beta = float(sys.argv[3])
    print(just_plot, n, beta)

    # policies are ["Random", "Greedy", "DQNPolicy", "Gittins", "Optimal"]
    all_n = [n]
    all_beta = [beta]
    policies_to_run = dict()
    for n in all_n:
        for beta in all_beta:
            key = (n, beta)
            if n <= 12:
                policies_to_run[key] = [0, 1, 2, 3, 4]
            else:
                policies_to_run[key] = [0, 1, 2, 3]
    experiment_result_pickle_filename = get_result_pickle_filename(n, beta)
    all_edges_to_add = list(range(0, 11, 2))

    if not just_plot:
        run_experiment2(n, beta, policies_to_run[(n, beta)], all_edges_to_add, experiment_result_pickle_filename)
    
    # Generate plots for exp2
    plot(n, beta, policies_to_run[(n, beta)], all_edges_to_add, experiment_result_pickle_filename)
