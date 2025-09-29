# Standard library imports

# Third-party imports
import numpy as np

# Local imports
from core.binary_frontier_environment import BinaryFrontierEnv
from policies.abstract_policy_class import AbstractPolicyClass

class RandomPolicy(AbstractPolicyClass):
    def __init__(self, env: BinaryFrontierEnv, instance_hash: str) -> None:
        super().__init__(env, instance_hash)

    @staticmethod
    def name() -> str:
        return "Random"

    def _setup_policy(self) -> None:
        pass

    def _train_policy(self) -> None:
        pass

    def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
        action = np.random.choice(list(valid_actions))
        return action
