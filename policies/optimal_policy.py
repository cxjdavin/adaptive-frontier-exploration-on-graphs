# Standard library imports
import timeit

# Third-party imports
import numpy as np

# Local imports
from core.binary_frontier_environment import BinaryFrontierEnv
from core.io_utils import load_pickle, save_pickle
from policies.abstract_policy_class import AbstractPolicyClass

class OptimalPolicy(AbstractPolicyClass):
    def __init__(self, env: BinaryFrontierEnv, instance_hash: str) -> None:
        super().__init__(env, instance_hash)
        self.policy, self.train_time = load_pickle(self.trained_pickle_filename)

    @staticmethod
    def name() -> str:
        return "Optimal"

    def _setup_policy(self) -> None:
        self.memo = dict()
        self.policy = dict()

    def _train_policy(self) -> None:
        def dp(status: np.ndarray) -> float:
            valid_actions = self.env.get_frontier_actions(status)
            if len(valid_actions) == 0:
                return 0
            dp_key = tuple(status)
            if dp_key not in self.memo:
                best_expected_value = -float('inf')
                best_action = None
                for idx in valid_actions:
                    p1 = self.env.get_marginal_prob1(idx, status)
                    pos = status.copy()
                    pos[idx] = 1
                    neg = status.copy()
                    neg[idx] = 0
                    expected_value = p1 + self.env.discount_factor * (p1 * dp(pos) + (1-p1) * dp(neg))
                    if expected_value > best_expected_value:
                        best_expected_value = expected_value
                        best_action = idx
                self.memo[dp_key] = best_expected_value
                self.policy[dp_key] = best_action
            return self.memo[dp_key]
        
        start_time = timeit.default_timer()
        initial_status = np.array([-1] * self.n)
        dp(initial_status)
        end_time = timeit.default_timer()
        self.train_time = end_time - start_time

        # Store trained policy information
        save_pickle([self.policy, self.train_time], self.trained_pickle_filename)

    def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
        dp_key = tuple(status)
        action = self.policy[dp_key]
        return action
    