# Standard library imports
import copy

from abc import ABC, abstractmethod

# Third-party imports
import numpy as np

# Local imports
from core.binary_env import BinaryEnv

class AbstractPolicyClass(ABC):
    def __init__(self, env: BinaryEnv, instance_hash: str, train: bool=True) -> None:
        self.instance_hash = instance_hash
        self.env = copy.deepcopy(env)
        self.n = self.env.n
        self.discount_factor = self.env.discount_factor
        self.frontier_testing = self.env.frontier_testing
        self.env.reset()
        self._setup_policy()
        if train:
            self._train_policy()
    
    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @abstractmethod
    def _setup_policy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_policy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
        raise NotImplementedError
    
    def act(self, status: np.ndarray) -> int:
        valid_actions = self.env.get_valid_actions(status)
        action = self._select_action(status, valid_actions)
        assert action in valid_actions
        return action
