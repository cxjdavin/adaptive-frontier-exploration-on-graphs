# Standard library imports

# Third-party imports

# Local imports
from core.abstract_joint_probability_class import AbstractJointProbabilityClass

class RealizationDistribution(AbstractJointProbabilityClass):
    def __init__(self, variables: list[str], args: dict) -> None:
        super().__init__(variables, args)

    def _setup(self) -> None:
        self.realization = dict()
        for X_var in self.variables:
            var = int(X_var[1:])
            self.realization[var] = self.args['realization'][var]
    
    def compute_conditional_probability(self, query_dict: dict[str,int], evidence_dict: dict[str,int]) -> float:
        # Revealed statuses should be consistent with hidden realization
        for X_var_key, var_value in evidence_dict.items():
            assert self.realization[X_var_key] == var_value
        for X_var_key, var_value in query_dict.items():
            if self.realization[X_var_key] != var_value:
                return 0.0
        return 1.0
