# Standard library imports
import copy
import itertools
import timeit

# Third-party imports
import numpy as np
from tqdm.notebook import tqdm

# Local imports
from core.abstract_joint_probability_class import AbstractJointProbabilityClass
from core.binary_env import BinaryEnv

class PolicyEvaluator:
    def __init__(self, env: BinaryEnv) -> None:
        self.env = copy.deepcopy(env)
        self.n = self.env.n

    """
    Generate samples ahead of time
    """
    def generate_monte_carlo_samples(self, num_runs: int, rng_seed: int) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        if num_runs == 1:
            return np.array(self.draw_sample_from_P(rng))
        else:
            mc_samples = []
            for _ in tqdm(range(num_runs), desc=f"Generating Monte Carlo samples"):
                ground_truth = self.draw_sample_from_P(rng)
                mc_samples.append(ground_truth)
            return np.array(mc_samples)
        
    """
    Draws a sample realization from environment via chain rule
    """
    def draw_sample_from_P(self, rng: np.random.Generator, tol: float = 1e-12) -> np.ndarray:
        observation = dict()
        for idx in range(self.n):
            p = self.env.compute_conditional_probability({f"X{idx}": 1}, observation)
            if not (-tol <= p and p <= 1 + tol):
                print(f"P(X{idx} = 1 | {observation}) = problematic p:", p)
            assert -tol <= p and p <= 1 + tol
            if rng.random() < p:
                observation[f"X{idx}"] = 1
            else:
                observation[f"X{idx}"] = 0
        statuses = [observation[f"X{idx}"] for idx in range(self.n)]
        return np.array(statuses)

    """
    Compute reward of policy on given ground truth state
    Returns an array of accumulated (discounted) rewards until all nodes are tested
    """
    def evaluate_ground_truth(self, policy: AbstractJointProbabilityClass, ground_truth: np.ndarray, discount_factor: float) -> tuple[np.ndarray, np.ndarray]:
        start_time = timeit.default_timer()
        rewards = []
        discounted_rewards = []
        status = np.array([-1] * self.n)
        for t in range(self.n):
            action = policy.act(status)
            status[action] = ground_truth[action]
            immediate_reward = status[action]
            rewards.append(immediate_reward)
            discounted_rewards.append(pow(discount_factor, t) * immediate_reward)
        rewards = np.cumsum(np.array(rewards))
        discounted_rewards = np.cumsum(np.array(discounted_rewards))
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        return rewards, discounted_rewards, elapsed_time

    def exact_evaluation(self, policy: AbstractJointProbabilityClass) -> tuple[np.ndarray, np.ndarray]:
        rewards = []
        discounted_rewards = []
        elapsed_times = []
        total_prob = 0.0
        for ground_truth in itertools.product((0, 1), repeat=self.n):
            realization_dict = {f"X{idx}": ground_truth[idx] for idx in range(self.n)}
            prob = self.env.compute_conditional_probability(realization_dict, {})
            accumulated_reward, accumulated_discounted_reward, elapsed_time = self.evaluate_ground_truth(policy, ground_truth, self.env.discount_factor)
            assert len(accumulated_reward) == self.n
            assert len(accumulated_discounted_reward) == self.n
            rewards.append(prob * accumulated_reward)
            discounted_rewards.append(prob * accumulated_discounted_reward)
            elapsed_times.append(elapsed_time)
            total_prob += prob
        rewards = np.array(rewards)
        discounted_rewards = np.array(discounted_rewards)
        elapsed_times = np.array(elapsed_times)
        assert np.isclose(total_prob, 1.0)
        expected_rewards = np.sum(rewards, axis=0)
        expected_discounted_rewards = np.sum(discounted_rewards, axis=0)
        average_time = np.mean(elapsed_times)
        assert len(expected_rewards) == self.n
        assert len(expected_discounted_rewards) == self.n
        return expected_rewards, expected_discounted_rewards, average_time
