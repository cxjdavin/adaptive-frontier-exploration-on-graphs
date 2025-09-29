# Standard library imports
import sys

from multiprocessing import Pool

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from matplotlib.colors import to_rgb

# Local imports
from core.experiment_runner import run_experiment
from core.ICPSR_22140_processor import ICPSR22140Processor
from core.io_utils import load_pickle, save_pickle

def get_result_pickle_filename(std_name: str, cc_threshold: int, inst_idx: int, eps: float, eps_rng_seed: int) -> str:
    return f"results/noisyQ_{std_name}_threshold{cc_threshold}_{inst_idx}_{eps}_{eps_rng_seed}.pkl"

def get_graph_pickle_filename(std_name: str, cc_threshold: int, inst_idx: int) -> str:
    return f"results/noisyQ/graphs/{std_name}_threshold{cc_threshold}_{inst_idx}.pkl"

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

def run_experiment_noisyQ_thread(args: tuple) -> None:
    (
        std_name,
        cc_threshold,
        inst_idx,
        eps,
        eps_rng_seed,
        G_pickle_filename,
        covariates,
        statuses,
        theta_unary,
        theta_pairwise,
        beta,
        num_monte_carlo_runs,
        rng_seed,
        G,
        policy_indices,
        multithread
    ) = args
    experiment_result_pickle_filename = get_result_pickle_filename(std_name, cc_threshold, inst_idx, eps, eps_rng_seed)

    all_inst_configs = []
    inst_configs = dict()
    inst_configs['G_pickle_filename'] = G_pickle_filename
    inst_configs['covariates'] = covariates
    inst_configs['theta_unary'] = theta_unary
    inst_configs['theta_pairwise'] = theta_pairwise
    inst_configs['discount_factor'] = beta
    inst_configs['num_monte_carlo_runs'] = num_monte_carlo_runs
    inst_configs['instance_hash'] = f"noisyQ_{std_name}_{cc_threshold}_{inst_idx}_{beta}_{num_monte_carlo_runs}_{rng_seed}_{eps}_{eps_rng_seed}"
    inst_configs['exp_name'] = "noisyQ"
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
        'instance_hash': inst_configs['instance_hash'],
        'eps': eps
    }
    save_pickle(tuple([instance_stats, all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec]), experiment_result_pickle_filename)


def run_experiment_noisyQ(std_name: str, cc_threshold: int, inst_idx: int, eps: float, eps_rng_seed: int) -> None:
    multithread = True
    num_monte_carlo_runs = 200
    rng_seed = 42
    beta = 0.99
    policy_indices = [0, 1, 2, 3]

    tsv_file1 = "ICPSR_22140/DS0001/22140-0001-Data.tsv"
    tsv_file2 = "ICPSR_22140/DS0002/22140-0002-Data.tsv"
    tsv_file3 = "ICPSR_22140/DS0003/22140-0003-Data.tsv"
    pickle_filename = "ICPSR_22140.pkl"
    processor = ICPSR22140Processor(tsv_file1, tsv_file2, tsv_file3, pickle_filename, filter_sex_only=True)
    
    # Extract graph, covariates, statuses based on fitted theta
    theta_unary, theta_pairwise = processor.get_theta_parameters(std_name)
    full_covariates, full_statuses, full_graph, _, _, _ = processor.merged_datasets[std_name]

    # Create graph
    G, covariates, statuses = pick_random_cc_until_cross_threshold(inst_idx, full_graph, full_covariates, full_statuses, cc_threshold)
    G_pickle_filename = get_graph_pickle_filename(std_name, cc_threshold, inst_idx)
    save_pickle(G, G_pickle_filename)

    all_inst_configs = []
    inst_configs = dict()
    inst_configs['G_pickle_filename'] = G_pickle_filename
    inst_configs['covariates'] = covariates
    inst_configs['theta_unary'] = theta_unary
    inst_configs['theta_pairwise'] = theta_pairwise
    inst_configs['discount_factor'] = beta
    inst_configs['num_monte_carlo_runs'] = num_monte_carlo_runs
    inst_configs['instance_hash'] = f"noisyQ_{std_name}_{cc_threshold}_{inst_idx}_{beta}_{num_monte_carlo_runs}_{rng_seed}_{eps}_{eps_rng_seed}"
    inst_configs['exp_name'] = "noisyQ"
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
        'instance_hash': inst_configs['instance_hash'],
        'eps': eps
    }

    experiment_result_pickle_filename = get_result_pickle_filename(std_name, cc_threshold, inst_idx, eps, eps_rng_seed)
    save_pickle(tuple([instance_stats, all_training_time, all_mean_vec, all_std_vec, all_disc_mean_vec, all_disc_std_vec, all_time_mean_vec, all_time_std_vec]), experiment_result_pickle_filename)

if __name__ == "__main__":
    print(sys.argv)
    std_idx = int(sys.argv[1])
    cc_threshold = int(sys.argv[2])
    inst_idx = int(sys.argv[3])
    eps = float(sys.argv[4])
    eps_rng_seed = int(sys.argv[5])
    
    std_names = ["Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"]
    assert 0 <= std_idx and std_idx < len(std_names)
    std_name = std_names[std_idx]

    run_experiment_noisyQ(std_name, cc_threshold, inst_idx, eps, eps_rng_seed)
