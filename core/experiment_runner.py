# Standard library imports
import os
import timeit

from multiprocessing import Pool

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local imports
from core.binary_env import BinaryEnv
from core.policy_evaluator import PolicyEvaluator
from core.log_junction_tree import LogJunctionTree
from core.io_utils import load_pickle, save_pickle
from policies.random_policy import RandomPolicy
from policies.greedy_policy import GreedyPolicy
from policies.dqn_policy import DQNPolicy
from policies.gittins_policy import GittinsPolicy
from policies.optimal_policy import OptimalPolicy

policies = [RandomPolicy, GreedyPolicy, DQNPolicy, GittinsPolicy, OptimalPolicy]
policy_labels = ["Random", "Greedy", "DQN", "Gittins", "Optimal"]

def create_one_monte_carlo_sample(args) -> np.ndarray:
    nomc_inst_pickle_filename, rng_seed = args
    _, _, _, G, factor_graph, discount_factor, cc_dict, cc_root = load_pickle(nomc_inst_pickle_filename)
    env = BinaryEnv(G, factor_graph, discount_factor, cc_dict, cc_root)
    evaluator = PolicyEvaluator(env)
    return evaluator.generate_monte_carlo_samples(1, rng_seed)

def train_policy_on_instance(args) -> tuple[int, float]:
    inst_pickle_filename, policy_idx = args
    _, _, instance_hash, G, factor_graph, discount_factor, cc_dict, cc_root = load_pickle(inst_pickle_filename)
    env = BinaryEnv(G, factor_graph, discount_factor, cc_dict, cc_root)
    start_time = timeit.default_timer()
    policies[policy_idx](env, instance_hash, train=True)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    return policy_idx, elapsed_time

def solve_instance(args) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inst_pickle_filename, policy_idx, mc_idx, mc_sample = args
    assert policy_idx in [0,1,2,3,4]
    exp_name, inst_idx, instance_hash, G, factor_graph, discount_factor, cc_dict, cc_root = load_pickle(inst_pickle_filename)
    policy_inst_pickle_filename = f"results/{exp_name}/{policy_labels[policy_idx]}/{mc_idx}_{instance_hash}.pkl"
    if not os.path.isfile(policy_inst_pickle_filename):
        env = BinaryEnv(G, factor_graph, discount_factor, cc_dict, cc_root)
        policy = policies[policy_idx](env, instance_hash, train=False)
        evaluator = PolicyEvaluator(env)
        if mc_idx is None:
            mean_vec, disc_mean_vec, elapsed_time = evaluator.exact_evaluation(policy)
        else:
            mean_vec, disc_mean_vec, elapsed_time = evaluator.evaluate_ground_truth(policy, mc_sample, discount_factor)
        save_pickle(tuple([mean_vec, disc_mean_vec, elapsed_time]), policy_inst_pickle_filename)
    else:
        mean_vec, disc_mean_vec, elapsed_time = load_pickle(policy_inst_pickle_filename)
    return policy_idx, inst_idx, mean_vec, disc_mean_vec, elapsed_time

def run_experiment(all_inst_configs: list[dict], all_policy_indices: list[int], multithread: bool, train_policies: bool = True) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    train_jobs = []
    jobs = []
    for inst_idx in tqdm(range(len(all_inst_configs)), desc=f"Pre-processing instances"):
        inst_configs = all_inst_configs[inst_idx]
        inst_pickle_filename = f"results/{inst_configs['exp_name']}/instances/{inst_configs['instance_hash']}.pkl"
        inst_mc_pickle_filename = f"results/{inst_configs['exp_name']}/monte_carlo_samples/{inst_configs['instance_hash']}.pkl"

        # Setup instance for multithreading
        if not os.path.isfile(inst_pickle_filename):
            G = load_pickle(inst_configs['G_pickle_filename'])
            args = {
                'G': G,
                'covariates': inst_configs['covariates'],
                'theta_unary': inst_configs['theta_unary'],
                'theta_pairwise': inst_configs['theta_pairwise']
            }

            # Generate factor graph either using Pgmpy or LogJunctionTree
            factor_graph = LogJunctionTree([f"X{idx}" for idx in G.nodes()], args)

            # Pre-process environment once
            env = BinaryEnv(G, factor_graph, inst_configs['discount_factor'])

            # Save instance pickle object
            save_pickle(tuple([inst_configs['exp_name'], inst_idx, inst_configs['instance_hash'], G, factor_graph, inst_configs['discount_factor'], env.cc_dict.copy(), env.cc_root.copy()]), inst_pickle_filename)
            
            if inst_configs['n'] <= 12:
                mc_samples = None
            else:
                # Pre-generate Monte Carlo samples for all instances
                rng = np.random.default_rng(inst_configs['eval_rng_seed'])
                mc_jobs = [(inst_pickle_filename, mc_seed) for mc_seed in rng.integers(0, 1e9, inst_configs['num_monte_carlo_runs'])]
                mc_samples = []
                if multithread:
                    # Multithread version
                    with Pool() as pool:
                        for mc_sample in tqdm(pool.imap_unordered(create_one_monte_carlo_sample, mc_jobs), total=len(mc_jobs), desc=f"Generating {inst_configs['num_monte_carlo_runs']} Monte Carlo samples (multithread)", leave=False):
                            mc_samples.append(mc_sample)
                else:
                    # Single thread version
                    for mc_args in tqdm(mc_jobs, desc=f"Training policies (single thread)"):
                        mc_samples.append(create_one_monte_carlo_sample(mc_args))
                mc_samples = np.array(mc_samples)

            # Save instance pickle object (with MC samples)
            save_pickle(mc_samples, inst_mc_pickle_filename)

        # Create policy training job
        if train_policies:
            for policy_idx in all_policy_indices:
                train_jobs.append([inst_pickle_filename, policy_idx])

        # Create 1 job per (policy, sample)
        if inst_configs['n'] <= 12:
            for policy_idx in all_policy_indices:
                jobs.append([inst_pickle_filename, policy_idx, None, None])
        else:
            mc_samples = load_pickle(inst_mc_pickle_filename)
            for policy_idx in all_policy_indices:
                for mc_idx in range(len(mc_samples)):
                    jobs.append([inst_pickle_filename, policy_idx, mc_idx, mc_samples[mc_idx]])

    train_time_pickle_filename = f"results/{inst_configs['exp_name']}/traintime_{inst_configs['instance_hash']}.pkl"
    if train_policies:
        # Train all policies once, then instantiate them without training in jobs
        all_training_time = {policy_idx: [] for policy_idx in all_policy_indices}
        if multithread:
            # Multithread version
            with Pool() as pool:
                for policy_idx, elapsed_time in tqdm(pool.imap_unordered(train_policy_on_instance, train_jobs), total=len(train_jobs), desc=f"Training policies (multithread)"):
                    all_training_time[policy_idx].append(elapsed_time)
        else:
            # Single thread version
            for train_job in tqdm(train_jobs, desc=f"Training policies (single thread)"):
                policy_idx, elapsed_time = train_policy_on_instance(train_job)
                all_training_time[policy_idx].append(elapsed_time)
        
        # Store training time
        save_pickle(all_training_time, train_time_pickle_filename)
    else:
        # Load training time
        all_training_time = load_pickle(train_time_pickle_filename)

    # Create dictionary to collect results
    jobs_mean_vec = {policy_idx: dict() for policy_idx in all_policy_indices}
    jobs_disc_mean_vec = {policy_idx: dict() for policy_idx in all_policy_indices}
    jobs_time_vec = {policy_idx: dict() for policy_idx in all_policy_indices}
    for policy_idx in all_policy_indices:
        for inst_idx in range(len(all_inst_configs)):
            jobs_mean_vec[policy_idx][inst_idx] = []
            jobs_disc_mean_vec[policy_idx][inst_idx] = []
            jobs_time_vec[policy_idx][inst_idx] = []

    # Run instances (multithread may not solve them in order)
    if multithread:
        # Multithread version
        with Pool() as pool:
            for policy_idx, inst_idx, mean_vec, disc_mean_vec, elapsed_time in tqdm(pool.imap_unordered(solve_instance, jobs), total=len(jobs), desc=f"Solving (multithread)"):
                jobs_mean_vec[policy_idx][inst_idx].append(mean_vec)
                jobs_disc_mean_vec[policy_idx][inst_idx].append(disc_mean_vec)
                jobs_time_vec[policy_idx][inst_idx].append(elapsed_time)
    else:
        # Single thread version
        for job in tqdm(jobs, desc=f"Solving (single thread)"):
            policy_idx, inst_idx, mean_vec, disc_mean_vec, elapsed_time = solve_instance(job)
            jobs_mean_vec[policy_idx][inst_idx].append(mean_vec)
            jobs_disc_mean_vec[policy_idx][inst_idx].append(disc_mean_vec)
            jobs_time_vec[policy_idx][inst_idx].append(elapsed_time)

    # Post-process all job results
    all_mean_vec_output = {policy_idx: [] for policy_idx in all_policy_indices}
    all_std_vec_output = {policy_idx: [] for policy_idx in all_policy_indices}
    all_disc_mean_vec_output = {policy_idx: [] for policy_idx in all_policy_indices}
    all_disc_std_vec_output = {policy_idx: [] for policy_idx in all_policy_indices}
    all_time_mean_vec_output = {policy_idx: [] for policy_idx in all_policy_indices}
    all_time_std_vec_output = {policy_idx: [] for policy_idx in all_policy_indices}
    for policy_idx in all_policy_indices:
        for inst_idx in range(len(all_inst_configs)):
            policy_instance_reward_vecs = jobs_mean_vec[policy_idx][inst_idx]
            assert len(policy_instance_reward_vecs) == 1 if all_inst_configs[inst_idx]['n'] <= 12 else len(policy_instance_reward_vecs) == all_inst_configs[inst_idx]['num_monte_carlo_runs']
            mean_vec = np.mean(np.array(policy_instance_reward_vecs), axis=0)
            std_err_vec = np.std(np.array(policy_instance_reward_vecs), axis=0) / np.sqrt(all_inst_configs[inst_idx]['num_monte_carlo_runs'])
            assert len(mean_vec) == all_inst_configs[inst_idx]['n']
            assert len(std_err_vec) == all_inst_configs[inst_idx]['n']

            policy_instance_disc_reward_vecs = jobs_disc_mean_vec[policy_idx][inst_idx]
            assert len(policy_instance_disc_reward_vecs) == 1 if all_inst_configs[inst_idx]['n'] <= 12 else len(policy_instance_disc_reward_vecs) == all_inst_configs[inst_idx]['num_monte_carlo_runs']
            disc_mean_vec = np.mean(np.array(policy_instance_disc_reward_vecs), axis=0)
            disc_std_err_vec = np.std(np.array(policy_instance_disc_reward_vecs), axis=0) / np.sqrt(all_inst_configs[inst_idx]['num_monte_carlo_runs'])
            assert len(disc_mean_vec) == all_inst_configs[inst_idx]['n']
            assert len(disc_std_err_vec) == all_inst_configs[inst_idx]['n']

            policy_instance_time_vecs = jobs_time_vec[policy_idx][inst_idx]
            assert len(policy_instance_time_vecs) == 1 if all_inst_configs[inst_idx]['n'] <= 12 else len(policy_instance_time_vecs) == all_inst_configs[inst_idx]['num_monte_carlo_runs']
            time_mean_vec = np.mean(np.array(policy_instance_time_vecs), axis=0)
            time_std_err_vec = np.std(np.array(policy_instance_time_vecs), axis=0) / np.sqrt(all_inst_configs[inst_idx]['num_monte_carlo_runs'])

            all_mean_vec_output[policy_idx].append(mean_vec)
            all_std_vec_output[policy_idx].append(std_err_vec)
            all_disc_mean_vec_output[policy_idx].append(disc_mean_vec)
            all_disc_std_vec_output[policy_idx].append(disc_std_err_vec)
            all_time_mean_vec_output[policy_idx].append(time_mean_vec)
            all_time_std_vec_output[policy_idx].append(time_std_err_vec)

    return all_training_time, all_mean_vec_output, all_std_vec_output, all_disc_mean_vec_output, all_disc_std_vec_output, all_time_mean_vec_output, all_time_std_vec_output
