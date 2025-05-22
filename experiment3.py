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

def get_result_pickle_filename(std_name: int, cc_threshold: int, inst_idx: int) -> str:
    return f"results/exp3_{std_name}_threshold{cc_threshold}_{inst_idx}.pkl"

def get_graph_pickle_filename(std_name: str, cc_threshold: int, inst_idx: int) -> str:
    return f"results/exp3/graphs/{std_name}_threshold{cc_threshold}_{inst_idx}.pkl"

def pick_random_cc_until_cross_threshold(inst_idx: int, G: nx.Graph, covariates: dict, statuses: dict, threshold: float) -> tuple[nx.Graph, list, list]:
    rng = np.random.default_rng(inst_idx)
    subgraph_nodes = set()
    subgraph_covariates = dict()
    subgraph_statuses = dict()
    idx_mapping = dict()
    all_cc_nodes = list(nx.connected_components(G))
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

def train(processor: ICPSR22140Processor, std_name: str, learning_rate: float) -> None:
    processor.fit_theta_parameters(std_name, learning_rate)

def run_experiment3(std_name: int, cc_threshold: int, inst_idx: int, experiment_result_pickle_filename: str, learning_rate: float = 0.001, fit_theta: bool = False) -> None:
    multithread = True
    train_policies = False
    num_monte_carlo_runs = 200
    rng_seed = 42
    beta = 0.99
    policy_indices = [0, 1, 2, 3]

    tsv_file1 = "ICPSR_22140/DS0001/22140-0001-Data.tsv"
    tsv_file2 = "ICPSR_22140/DS0002/22140-0002-Data.tsv"
    tsv_file3 = "ICPSR_22140/DS0003/22140-0003-Data.tsv"
    pickle_filename = "ICPSR_22140.pkl"
    processor = ICPSR22140Processor(tsv_file1, tsv_file2, tsv_file3, pickle_filename)
    
    # Extract graph, covariates, statuses based on fitted theta
    if fit_theta:
        train(processor, std_name, learning_rate)
    theta_unary, theta_pairwise = processor.get_theta_parameters(std_name, learning_rate)
    full_G, full_covariates, full_statuses = processor.merged_datasets[std_name]

    all_inst_configs = []
    G, covariates, statuses = pick_random_cc_until_cross_threshold(inst_idx, full_G, full_covariates, full_statuses, cc_threshold)
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
    inst_configs['instance_hash'] = f"exp3_{std_name}_{cc_threshold}_{inst_idx}_{learning_rate}_{beta}_{num_monte_carlo_runs}_{rng_seed}"
    inst_configs['exp_name'] = "exp3"
    inst_configs['inst_idx'] = inst_idx
    inst_configs['n'] = G.number_of_nodes()
    inst_configs['eval_rng_seed'] = 42
    all_inst_configs.append(inst_configs)
    
    print(f"Solving {std_name}, cc_threshold = {cc_threshold}, instance {inst_idx}")
    all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec = run_experiment(all_inst_configs, policy_indices, multithread, train_policies)
    for policy_idx in policy_indices:
        print(policy_idx, all_mean_vec[policy_idx])
    
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

def plot(experiment_result_pickle_filename: str) -> None:
    # See: https://stackoverflow.com/questions/33337989/how-to-draw-more-type-of-lines-in-matplotlib
    policy_labels = ["Random", "Greedy", "DQNPolicy", "Gittins", "Optimal"]
    policy_colors = ['red', 'blue', 'green', 'orange', 'purple']
    policy_styles = ['dotted', 'dashed', 'dashdot', 'solid', (0, (3, 2, 1, 2))]
    
    stats, all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec = load_pickle(experiment_result_pickle_filename)
    policy_indices = [0, 1, 2, 3]

    # Plot discounted
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 + 6*2, 6), sharex=True)
    plt.rcParams['font.size'] = 22
    X_axis = list(np.arange(1, stats['n']+1))
    all_lines = []

    for policy_idx in policy_indices:
        line_handle, = axes[0].plot(X_axis, np.array(all_disc_mean_vec[policy_idx]).squeeze(), ls=policy_styles[policy_idx], color=policy_colors[policy_idx], lw=3)
        base_color = to_rgb(line_handle.get_color())
        fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
        axes[0].fill_between(
            X_axis,
            np.array(all_disc_mean_vec[policy_idx]).squeeze() - np.array(all_disc_std_vec[policy_idx]).squeeze(),
            np.array(all_disc_mean_vec[policy_idx]).squeeze() + np.array(all_disc_std_vec[policy_idx]).squeeze(),
            color=fill_color,
            alpha=0.5
        )
        axes[0].axvline(x=X_axis[len(X_axis)//2-1], color='gray', linestyle='--', linewidth=2)
        axes[0].set_ylabel("Expected disc. accumulated rewards")
        axes[0].set_title(fr"Discount factor $\beta =$ {stats['beta']}")
        all_lines.append(line_handle)
    # Plot the Greedy line again because sometimes Gittins overlaps it
    axes[0].plot(X_axis, np.array(all_disc_mean_vec[1]).squeeze(), ls=policy_styles[1], color=policy_colors[1], lw=3)

    # Plot undiscounted
    for policy_idx in policy_indices:
        line_handle, = axes[1].plot(X_axis, np.array(all_mean_vec[policy_idx]).squeeze(), ls=policy_styles[policy_idx], color=policy_colors[policy_idx], lw=3)
        base_color = to_rgb(line_handle.get_color())
        fill_color = 0.5 * np.array(base_color) + 0.5 * np.array([1.0, 1.0, 1.0])
        axes[1].fill_between(
            X_axis,
            np.array(all_mean_vec[policy_idx]).squeeze() - np.array(all_std_vec[policy_idx]).squeeze(),
            np.array(all_mean_vec[policy_idx]).squeeze() + np.array(all_std_vec[policy_idx]).squeeze(),
            color=fill_color,
            alpha=0.5
        )
        axes[1].axvline(x=X_axis[len(X_axis)//2-1], color='gray', linestyle='--', linewidth=2)
        axes[1].set_ylabel("Expected accumulated rewards")
        axes[1].set_title(fr"Undiscounted")
        all_lines.append(line_handle)
    # Plot the Greedy line again because sometimes Gittins overlaps it
    axes[1].plot(X_axis, np.array(all_mean_vec[1]).squeeze(), ls=policy_styles[1], color=policy_colors[1], lw=3)

    fig.text(0.5, 0, "Number of tests", ha='center')
    fig.legend(all_lines, policy_labels[:4], loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.95))
    plot_filename = f"results/plots/exp3_{stats['std_name']}_threshold{stats['cc_threshold']}_{stats['inst_idx']}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches = 'tight')

    # Plot graph
    stats, _, _, _, _, _, _, _ = load_pickle(experiment_result_pickle_filename)
    inst_pickle_filename = f"results/exp3/instances/{stats['instance_hash']}.pkl"
    _, _, _, G, _, _, _, all_cc_roots = load_pickle(inst_pickle_filename)

    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams['font.size'] = 40

    pos = nx.spring_layout(G, seed=42)
    node_colors = ['red' if i in all_cc_roots else 'blue' for i in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, node_size=30, edge_color='black', with_labels=False, width=3.0, alpha=0.8)
    for root_idx in all_cc_roots:
        circle = patches.Circle(pos[root_idx], radius=0.05, facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(circle)
    
    fig.text(0.5, 0.85, f"{std_name} sex interaction graph", ha='center')
    fig.text(0.5, 0.1, f"Frontier roots are circled in red", ha='center', color='red')
    plt.savefig(f"results/plots/exp3_{std_name}_{inst_idx}_graph.png", dpi=300, bbox_inches = 'tight')

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
    mode = int(sys.argv[1])
    std_idx = int(sys.argv[2])
    cc_threshold = int(sys.argv[3])
    inst_idx = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    print(mode, std_idx, cc_threshold, inst_idx, learning_rate)

    if mode == 0:
        just_plot = True
    elif mode == 1:
        just_plot = False
        fit_theta = True
    elif mode == 2:
        just_plot = False
        fit_theta = False
    else:
        raise NotImplementedError

    std_names = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV","Hepatitis"]
    assert 0 <= std_idx and std_idx < len(std_names)
    std_name = std_names[std_idx]
    experiment_result_pickle_filename = get_result_pickle_filename(std_name, cc_threshold, inst_idx)

    if not just_plot:
        run_experiment3(std_name, cc_threshold, inst_idx, experiment_result_pickle_filename, learning_rate, fit_theta)
    
    # Generate plots for exp3
    plot(experiment_result_pickle_filename)
