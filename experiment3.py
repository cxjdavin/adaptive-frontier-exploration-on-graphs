# Standard library imports
import sys

# Third-party imports
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from matplotlib.colors import to_rgb

# Local imports
from core.experiment_runner import run_experiment
from core.ICPSR_22140_processor import ICPSR22140Processor
from core.io_utils import load_pickle, save_pickle

from core.binary_frontier_environment import BinaryFrontierEnv
from core.log_junction_tree import LogJunctionTree

def get_result_pickle_filename(std_name: str, cc_threshold: int, inst_idx: int) -> str:
    return f"results/exp3_{std_name}_threshold{cc_threshold}_{inst_idx}.pkl"

def get_graph_pickle_filename(std_name: str, cc_threshold: int, inst_idx: int) -> str:
    return f"results/exp3/graphs/{std_name}_threshold{cc_threshold}_{inst_idx}.pkl"

def pick_random_cc_until_cross_threshold(inst_idx: int, G: nx.Graph, covariates: dict, statuses: dict, threshold: float) -> tuple[nx.Graph, dict, dict]:
    rng = np.random.default_rng(inst_idx)
    subgraph_nodes = set()
    subgraph_covariates = dict()
    subgraph_statuses = dict()
    idx_mapping = dict()
    all_cc_nodes = np.array(list(nx.connected_components(G)))
    rng.shuffle(all_cc_nodes)
    for cc_nodes in all_cc_nodes:
        subgraph_nodes.update(cc_nodes)
        for i in cc_nodes:
            idx_mapping[i] = len(idx_mapping)
            subgraph_covariates[idx_mapping[i]] = covariates[i]
            subgraph_statuses[idx_mapping[i]] = statuses[i]
        if len(subgraph_nodes) >= threshold:
            break
    H = G.subgraph(subgraph_nodes)
    H = nx.relabel_nodes(H, idx_mapping)
    return H, subgraph_covariates, subgraph_statuses

def run_experiment3(std_name: str, cc_threshold: int, inst_idx: int, experiment_result_pickle_filename: str) -> None:
    multithread = True
    num_monte_carlo_runs = 200
    rng_seed = 42
    eps = 0
    eps_rng_seed = 42
    beta = 0.99
    policy_indices = [0, 1, 2, 3]

    tsv_file1 = "ICPSR_22140/DS0001/22140-0001-Data.tsv"
    tsv_file2 = "ICPSR_22140/DS0002/22140-0002-Data.tsv"
    tsv_file3 = "ICPSR_22140/DS0003/22140-0003-Data.tsv"
    pickle_filename = "ICPSR_22140.pkl"
    processor = ICPSR22140Processor(tsv_file1, tsv_file2, tsv_file3, pickle_filename, filter_sex_only=True)
    processor.fit_theta_parameters(std_name)

    # Extract graph, covariates, statuses based on fitted theta
    theta_unary, theta_pairwise = processor.get_theta_parameters(std_name)
    full_covariates, full_statuses, full_graph, _, _, _ = processor.merged_datasets[std_name]

    all_inst_configs = []
    G, covariates, statuses = pick_random_cc_until_cross_threshold(inst_idx, full_graph, full_covariates, full_statuses, cc_threshold)
    G_pickle_filename = get_graph_pickle_filename(std_name, cc_threshold, inst_idx)
    save_pickle(G, G_pickle_filename)

    # Setup instance configurations
    inst_configs = dict()
    inst_configs['G_pickle_filename'] = G_pickle_filename
    inst_configs['covariates'] = covariates
    inst_configs['theta_unary'] = theta_unary
    inst_configs['theta_pairwise'] = theta_pairwise
    inst_configs['discount_factor'] = beta
    inst_configs['num_monte_carlo_runs'] = num_monte_carlo_runs
    inst_configs['instance_hash'] = f"exp3_{std_name}_{cc_threshold}_{inst_idx}_{beta}_{num_monte_carlo_runs}_{rng_seed}"
    inst_configs['exp_name'] = "exp3"
    inst_configs['inst_idx'] = inst_idx
    inst_configs['n'] = G.number_of_nodes()
    inst_configs['eval_rng_seed'] = 42
    inst_configs['eps'] = eps
    inst_configs['eps_rng_seed'] = eps_rng_seed
    all_inst_configs.append(inst_configs)
    
    print(f"Solving {std_name}, cc_threshold = {cc_threshold}, instance {inst_idx}")
    all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec = run_experiment(all_inst_configs, policy_indices, multithread)
    # for policy_idx in policy_indices:
    #     print(policy_idx, all_mean_vec[policy_idx])
    
    # Save output so we can plot separately
    instance_stats = {
        'std_name': std_name,
        'n': G.number_of_nodes(),
        'm': G.number_of_edges(),
        'diameter': max(nx.diameter(G.subgraph(cc_nodes)) for cc_nodes in nx.connected_components(G)),
        'infected': sum(statuses.values()),
        'num_cc': len(list(nx.connected_components(G))),
        'approximate_tw': nx.algorithms.approximation.treewidth_min_fill_in(G)[0],
        'beta': beta,
        'cc_threshold': cc_threshold,
        'inst_idx': inst_idx,
        'instance_hash': inst_configs['instance_hash']
    }
    save_pickle(tuple([instance_stats, all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec]), experiment_result_pickle_filename)

    # Print stats
    for policy_idx in policy_indices:
        print(policy_idx, all_training_time[policy_idx])
    args = {
        'G': G,
        'covariates': inst_configs['covariates'],
        'theta_unary': inst_configs['theta_unary'],
        'theta_pairwise': inst_configs['theta_pairwise'],
        'eps': 0,
        'eps_rng_seed': 42
    }
    preprocess_P = LogJunctionTree([f"X{idx}" for idx in G.nodes()], args)
    preprocess_env = BinaryFrontierEnv(G, preprocess_P, inst_configs['discount_factor'])
    max_depth = 0
    for root in preprocess_env.cc_root:
        lengths = nx.single_source_shortest_path_length(G, root)
        max_depth = max(max_depth, max(lengths.values(), default=0))
    print(f"Disease: {instance_stats['std_name']}")
    print(f"Number of nodes: {instance_stats['n']}")
    print(f"Number of edges: {instance_stats['m']}")
    print(f"Is forest: {nx.is_forest(G)}")
    print(f"Diameter: {instance_stats['diameter']}")
    print(f"Number of CC: {instance_stats['num_cc']}")
    print(f"Max depth: {max_depth}")
    print(f"Apx treewidth: {instance_stats['approximate_tw']}")

def plot(experiment_result_pickle_filename: str) -> None:
    # See: https://stackoverflow.com/questions/33337989/how-to-draw-more-type-of-lines-in-matplotlib
    policy_labels = ["Random", "Greedy", "DQNPolicy", "Gittins", "Optimal"]
    policy_colors = ['red', 'blue', 'green', 'orange', 'purple']
    policy_styles = ['dotted', 'dashed', 'dashdot', 'solid', (0, (3, 2, 1, 2))]
    
    stats, all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec = load_pickle(experiment_result_pickle_filename)
    policy_indices = [0, 1, 2, 3]

    # Create 2 subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 + 6*2, 6))

    # Plot undiscounted
    all_lines = []
    X_axis = np.array(np.arange(stats['n']))/stats['n']
    for policy_idx in policy_indices:
        original_y_values = np.array(all_mean_vec[policy_idx]).squeeze()
        max_y = np.max(original_y_values)
        scaled_y_values = original_y_values / max_y
        line_handle, = axes[0].plot(X_axis, scaled_y_values, ls=policy_styles[policy_idx], color=policy_colors[policy_idx], lw=3)
        base_color = to_rgb(line_handle.get_color())
        fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
        axes[0].fill_between(
            X_axis,
            scaled_y_values - (np.array(all_std_vec[policy_idx]).squeeze() / max_y),
            scaled_y_values + (np.array(all_std_vec[policy_idx]).squeeze() / max_y),
            color=fill_color,
            alpha=0.5
        )
        axes[0].axvline(x=0.5, color='gray', linestyle='--', linewidth=2)
        axes[0].set_xlabel("Fraction of population tested", fontsize=20)
        axes[0].set_ylabel("Fraction of positive cases detected", fontsize=20)
        axes[0].set_title(fr"Policies ran with $\beta =$ {stats['beta']}", fontsize=20)
        all_lines.append(line_handle)
    # Plot the Greedy line again because sometimes Gittins overlaps it
    greedy_y_values = np.array(all_mean_vec[1]).squeeze()
    max_greedy_y_values = np.max(greedy_y_values)
    axes[0].plot(X_axis, greedy_y_values/max_greedy_y_values, ls=policy_styles[1], color=policy_colors[1], lw=3)

    # Plot graph
    inst_pickle_filename = f"results/exp3/instances/{stats['instance_hash']}.pkl"
    _, _, _, G, _, _, _, _, all_cc_roots = load_pickle(inst_pickle_filename)
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['red' if i in all_cc_roots else 'blue' for i in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, node_size=10, edge_color='black', with_labels=False, width=3.0, alpha=0.8, ax=axes[1])
    for root_idx in all_cc_roots:
        circle = patches.Circle(pos[root_idx], radius=0.05, facecolor='none', edgecolor='red', linewidth=2)
        axes[1].add_patch(circle)
    # Add text: after center alignment, (0,0) is lower-left and (1,1) is upper-right
    axes[1].text(0.5, 1, f"{std_name} sex interaction graph", horizontalalignment='center', verticalalignment='center', transform = axes[1].transAxes, fontsize=22)
    axes[1].text(0.5, 0, f"Frontier roots are circled in red", horizontalalignment='center', verticalalignment='center', transform = axes[1].transAxes, color='red', fontsize=22)

    # Plot legend and save
    fig.legend(all_lines, policy_labels[:4], loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.95), fontsize=22)
    plot_filename = f"results/plots/exp3_{stats['std_name']}_threshold{stats['cc_threshold']}_{stats['inst_idx']}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches = 'tight')

    # Print timing for fun
    print("=== Policy training time (mean) ===")
    for policy_idx in all_training_time.keys():
        x = np.array(all_training_time[policy_idx])
        print(policy_labels[policy_idx], np.mean(x))
    print("=== Rollout time per MC sample (mean += std err) ===")
    for policy_idx in policy_indices:
        print(policy_labels[policy_idx], all_time_mean_vec[policy_idx], all_time_std_vec[policy_idx])

if __name__ == "__main__":
    print(sys.argv)
    std_idx = int(sys.argv[1])
    cc_threshold = int(sys.argv[2])
    inst_idx = int(sys.argv[3])

    std_names = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
    assert 0 <= std_idx and std_idx < len(std_names)
    std_name = std_names[std_idx]
    experiment_result_pickle_filename = get_result_pickle_filename(std_name, cc_threshold, inst_idx)

    run_experiment3(std_name, cc_threshold, inst_idx, experiment_result_pickle_filename)
    plot(experiment_result_pickle_filename)
