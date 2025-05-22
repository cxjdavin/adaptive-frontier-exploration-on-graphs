# Standard library imports

# Third-party imports
import networkx as nx
import numpy as np

# Local imports
from core.binary_env import BinaryEnv
from policies.abstract_policy_class import AbstractPolicyClass

class GreedyPolicy(AbstractPolicyClass):
    def __init__(self, env: BinaryEnv, instance_hash: str, train: bool = True) -> None:
        super().__init__(env, instance_hash, train)

    @staticmethod
    def name() -> str:
        return "Greedy"
    
    def _setup_policy(self) -> None:
        # Compute marginal prob1s for every node
        initial_status = np.array([-1] * self.n)
        self.marginal_prob1s = [(self.env.get_marginal_prob1(idx, initial_status), idx) for idx in range(self.env.n)]
        
        # Keep track of which was the last tested individual
        self.last_tested = None

        # Pre-compute the vertices that belong in the same connected component of other vertices
        self.same_cc_nodes = dict()
        for cc_nodes in nx.connected_components(self.env.G):
            index_set = frozenset([int(v[1:]) for v in cc_nodes])
            for idx in index_set:
                self.same_cc_nodes[idx] = index_set
    
    def _train_policy(self) -> None:
        pass

    def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
        # Update marginal prob1s for all nodes in the same connected component as last tested individual
        # Note: The marginal prob1s of everyone else is unchanged
        if self.last_tested is not None:
            for idx in self.same_cc_nodes[self.last_tested]:
                self.marginal_prob1s[idx] = (self.env.get_marginal_prob1(idx, status), idx)
        action = sorted([self.marginal_prob1s[idx] for idx in valid_actions], reverse=True)[0][1]
        self.last_tested = action
        return action

# Direct, unoptimized implementation
# class GreedyPolicy(AbstractPolicyClass):
#     def __init__(self, env: BinaryEnv, instance_hash: str, train: bool = True) -> None:
#         super().__init__(env, instance_hash, train)

#     @staticmethod
#     def name() -> str:
#         return "Greedy"
    
#     def _setup_policy(self) -> None:
#         pass
    
#     def _train_policy(self) -> None:
#         pass

#     def _select_action(self, status: np.ndarray, valid_actions: set[int]) -> int:
#         marginal_prob1s = [(self.env.get_marginal_prob1(idx, status), idx) for idx in valid_actions]
#         action = sorted(marginal_prob1s, reverse=True)[0][1]
#         return action
