# Standard library imports
import os
import shutil
import tempfile

# Third-party imports
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# Local imports
from core.abstract_joint_probability_class import AbstractJointProbabilityClass
from core.io_utils import load_pickle, save_pickle

"""
Class for processing the ICPSR_22140 dataset
"""
class ICPSR22140Processor:
    def __init__(self, tsv_file1: str, tsv_file2: str, tsv_file3: str, pickle_filename: str) -> None:
        self.pickle_filename = pickle_filename
        self.STD_to_dfkey = dict()
        self.STD_to_dfkey["Gonorrhea"] = "GONO"
        self.STD_to_dfkey["Chlamydia"] = "CHLAM"
        self.STD_to_dfkey["Syphilis"] = "SYPH"
        self.STD_to_dfkey["HIV"] = "HIV"
        self.STD_to_dfkey["Hepatitis"] = "HBV"
        self.covariate_headers = ['LOCAL', 'RACE', 'ETHN', 'SEX', 'ORIENT', 'BEHAV', 'PRO', 'PIMP', 'JOHN', 'DEALER', 'DRUGMAN', 'THIEF', 'RETIRED', 'HWIFE', 'DISABLE', 'UNEMP', 'STREETS']
        self.curated_dataset = self._extract_curated_dataset(tsv_file1, tsv_file2, tsv_file3)
        self.merged_datasets = self._merge_all_std_datasets_into_one()

    """
    Generate covariates
    """
    def _generate_covariates(self, node_df: pd.DataFrame, rid: int, studynum: int) -> np.ndarray:
        covariates = []
        mask = (node_df["RID"] == rid) & (node_df["STUDYNUM"] == studynum)
        assert len(node_df.loc[mask, self.covariate_headers]) == 1
        for col in self.covariate_headers:
            values = sorted([int(x) for x in set(node_df[col].values)])
            one_hot = [0] * len(values)
            idx = values.index(node_df.loc[mask, col].iloc[0])
            one_hot[idx] = 1
            covariates += one_hot
        return covariates

    """
    Extract dataset from ICPSR_22140
    - curated_dataset[std] is a list of datasets with keys {"Gonorrhea", "Chlamydia", "Syphilis", "HIV", "Hepatitis"}
    - Each dataset is a dictionary with keys {"studynum", "graph", "covariates", "statuses"}
    """
    def _extract_curated_dataset(self, tsv_file1: str, tsv_file2: str, tsv_file3: str) -> dict:
        if not os.path.isfile(self.pickle_filename):
            node_df = pd.read_csv(tsv_file1, sep='\t')
            df2 = pd.read_csv(tsv_file2, sep='\t')
            df3 = pd.read_csv(tsv_file3, sep='\t')

            # Reorder df3 columns to match df2    
            assert set(df2.columns) == set(df3.columns)
            df3 = df3[df2.columns]

            # Stack rows of both edge files and reset index
            assert df2.columns.equals(df3.columns)
            edge_df = pd.concat([df2, df3], ignore_index=True)

            # Filter and keep only sex edges
            sex_filter = edge_df[edge_df["TIETYPE"] == 3]
            print("Number of individuals after sex filter:", len(sex_filter))

            curated_dataset = {std: [] for std in self.STD_to_dfkey.keys()}
            for std, dfkey in self.STD_to_dfkey.items():
                std_sex_filter = sex_filter[sex_filter[f"{dfkey}1"].isin({0,1}) & sex_filter[f"{dfkey}2"].isin({0,1})]
                print(f"Number of individuals after {std} + sex filter:", len(std_sex_filter))

                graphs = {studynum: nx.Graph() for studynum in set(node_df["STUDYNUM"])}
                statuses = {studynum: dict() for studynum in set(node_df["STUDYNUM"])}
                for _, row in std_sex_filter.iterrows():
                    studynum, u, v, u_status, v_status = row["STUDYNUM"], row["ID1"], row["ID2"], row[f"{dfkey}1"], row[f"{dfkey}2"]
                    if u not in statuses[studynum].keys():
                        statuses[studynum][u] = u_status
                    else:
                        statuses[studynum][u] = max(u_status, statuses[studynum][u])
                    if v not in statuses[studynum].keys():
                        statuses[studynum][v] = v_status
                    else:
                        statuses[studynum][v] = max(v_status, statuses[studynum][v])
                    graphs[studynum].add_edge(u, v)
                for studynum, G in graphs.items():
                    if len(G) > 0:
                        num_positive = sum([1 for u in G.nodes if statuses[studynum][u] == 1])
                        tw, _ = nx.algorithms.approximation.treewidth_min_fill_in(G)
                        print(f"{std} study {studynum}: {num_positive}/{len(G.nodes)} infected. {G} has approx treewidth {tw}")

                        # Create new dataset and store into curated dataset
                        new_dataset = dict()
                        individual_mapping = dict()
                        individual_covariates = dict()
                        individual_statuses = dict()
                        for u in G.nodes:
                            individual_mapping[u] = len(individual_mapping)
                            individual_covariates[individual_mapping[u]] = self._generate_covariates(node_df, u, studynum)
                            individual_statuses[individual_mapping[u]] = statuses[studynum][u]
                        G = nx.relabel_nodes(G, individual_mapping)

                        # Store dataset
                        new_dataset["studynum"] = studynum
                        new_dataset["graph"] = G
                        new_dataset["covariates"] = individual_covariates
                        new_dataset["statuses"] = individual_statuses
                        curated_dataset[std].append(new_dataset)
            
            # Store curated_dataset to file
            save_pickle(curated_dataset, self.pickle_filename)
        
        # Load curated_dataset from file and output
        curated_dataset = load_pickle(self.pickle_filename)
        return curated_dataset

    def _merge_all_std_datasets_into_one(self) -> None:
        merged_datasets = dict()
        for std in self.STD_to_dfkey.keys():
            sz = 0
            overall_graph = nx.Graph()
            overall_covariates = dict()
            overall_statuses = dict()
            for std_dataset in self.curated_dataset[std]:
                G, covariates, statuses = std_dataset['graph'], std_dataset['covariates'], std_dataset['statuses']
                overall_graph.add_nodes_from([idx + sz for idx in G.nodes])
                for u, v in G.edges():
                    overall_graph.add_edge(u + sz, v + sz)
                n = G.number_of_nodes()
                for idx in range(n):
                    overall_covariates[idx + sz] = covariates[idx]
                    overall_statuses[idx + sz] = statuses[idx]
                sz += n
            merged_datasets[std] = [overall_graph, overall_covariates, overall_statuses]
        return merged_datasets
        
    def _load_from_checkpoint(self, checkpoint_filename: str) -> tuple[int, np.ndarray, np.ndarray]:
        assert os.path.exists(checkpoint_filename)
        checkpoint_dict = load_pickle(checkpoint_filename)
        log_pl, step_idx, theta_unary, theta_pairwise = sorted([key + value for key, value in checkpoint_dict.items()], reverse=True)[0]
        print(f"Just loaded from {checkpoint_filename}. Checkpoint iter {step_idx}, log_pseudolikelihood = {log_pl:.4f}")
        return checkpoint_dict, step_idx, theta_unary, theta_pairwise

    def get_dataset(self, dataset_key: str) -> tuple[nx.Graph, dict, dict]:
        return self.merged_datasets[dataset_key]

    def get_theta_parameters(self, dataset_key: str, learning_rate: float, max_iter: int = 1000, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        checkpoint_filename = f"ICPSR22140/checkpoints/{dataset_key}_{learning_rate}.pkl"
        if not os.path.exists(checkpoint_filename):
            self.fit_theta_parameters(dataset_key, learning_rate, max_iter, eps)
        else:
            _, _, theta_unary, theta_pairwise = self._load_from_checkpoint(checkpoint_filename)
            return theta_unary, theta_pairwise

    """
    Compute pseudo MLE estimates of theta via gradient ascent while storing the checkpoints periodically
    checkpoint_dict[(log_pseudolikelihood, step_idx)] = (theta_unary, theta_pairwise)
    """
    def fit_theta_parameters(self, std_name: str, learning_rate: float, max_iter: int = 1000, eps: float = 1e-6) -> None:
        checkpoint_filename = f"ICPSR22140/checkpoints/{std_name}_{learning_rate}.pkl"
        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)

        G, covariates, statuses = self.get_dataset(std_name)
        n = G.number_of_nodes()
        covariate_length = len(covariates[0])

        # Pre-compute unary and pairwise sum vectors for gradient computation later
        self.memo_unary_vector_sum_for_gradient = dict()
        self.memo_pairwise_vector_sum_for_gradient = dict()
        for i in range(n):
            c_i = covariates[i]
            self.memo_unary_vector_sum_for_gradient[i] = AbstractJointProbabilityClass.f_unary(1, c_i) - AbstractJointProbabilityClass.f_unary(0, c_i)
            pairwise_vector_sum = np.zeros(AbstractJointProbabilityClass.compute_theta_length(covariate_length, 2))
            for j in G.neighbors(i):
                u, v = min(i,j), max(i,j)
                c_u = covariates[u]
                c_v = covariates[v]
                if u == i:
                    x_v = statuses[v]
                    pairwise_vector_sum += AbstractJointProbabilityClass.f_pairwise(1, x_v, c_u, c_v) - AbstractJointProbabilityClass.f_pairwise(0, x_v, c_u, c_v)
                else:
                    x_u = statuses[u]
                    pairwise_vector_sum += AbstractJointProbabilityClass.f_pairwise(x_u, 1, c_u, c_v) - AbstractJointProbabilityClass.f_pairwise(x_u, 0, c_u, c_v)
            self.memo_pairwise_vector_sum_for_gradient[i] = pairwise_vector_sum

        if os.path.exists(checkpoint_filename):
            # Load theta checkpoint from file and continue fitting
            checkpoint_dict, step_idx, theta_unary, theta_pairwise = self._load_from_checkpoint(checkpoint_filename)
        else:
            # Initialize to small random numbers
            checkpoint_dict = dict()
            step_idx = 0
            theta_unary = np.random.randn(AbstractJointProbabilityClass.compute_theta_length(covariate_length, 1)) * 0.01
            theta_pairwise = np.random.randn(AbstractJointProbabilityClass.compute_theta_length(covariate_length, 2)) * 0.01

        # Run until max iter or when both gradients are near zero
        for it in tqdm(range(step_idx, step_idx + max_iter), desc=f"{checkpoint_filename}", leave=False):
            self.memo_local_log_ZProb = dict()
            unary_gradient, pairwise_gradient = self._compute_gradients(std_name, theta_unary, theta_pairwise)
            theta_unary += learning_rate * unary_gradient
            theta_pairwise += learning_rate * pairwise_gradient
            if np.linalg.norm(unary_gradient) < eps and np.linalg.norm(pairwise_gradient) < eps:
                break

            if ((it > 0 and it % 10 == 0) or it == max_iter - 1):
                self.memo_local_log_ZProb = dict()
                log_pl = self._compute_log_pseudo_likelihood(std_name, theta_unary, theta_pairwise)
                norm1 = np.linalg.norm(unary_gradient)
                norm2 = np.linalg.norm(pairwise_gradient)
                print(f"{checkpoint_filename} | Iter {it} | log_pseudolikelihood = {log_pl:.4f}, ||unary grad|| = {norm1:.4f}, ||pairwise grad|| = {norm2:.4f}")
                
                # Add checkpoint and store to file atomically (i.e. other threads will not read a half-written file)
                checkpoint_dict[(log_pl, it)] = (theta_unary.copy(), theta_pairwise.copy())
                tmp_fd, tmp_path = tempfile.mkstemp()
                os.close(tmp_fd) # Close the file descriptor returned by mkstemp
                save_pickle(checkpoint_dict, tmp_path)
                shutil.move(tmp_path, checkpoint_filename) # Atomically replace the old checkpoint file

    """
    Compute gradients for thetas
    """
    def _compute_gradients(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, statuses = self.get_dataset(dataset_key)
        unary_gradient = np.zeros_like(theta_unary)
        pairwise_gradient = np.zeros_like(theta_pairwise)
        for i in range(len(statuses)):
            # We want to avoid computing the value of the partition function Z
            # coeff = x_i - p1 = x_i - Zp1/(Zp0 + Zp1)
            # Taking exponentiation np.exp(logZp0) and np.exp(logZp1) can cause numerical issues, so we use the log-sum-exp trick
            logZp0 = self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, i, 0)
            logZp1 = self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, i, 1)
            coeff = statuses[i] - np.exp(logZp1 - AbstractJointProbabilityClass.logsumexp([logZp0, logZp1]))
            unary_gradient += coeff * self.memo_unary_vector_sum_for_gradient[i]
            pairwise_gradient += coeff * self.memo_pairwise_vector_sum_for_gradient[i]
        return unary_gradient, pairwise_gradient
    
    """
    Compute log pseudo-likelihood
    """
    def _compute_log_pseudo_likelihood(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray) -> float:
        G, _, _ = self.get_dataset(dataset_key)
        n = G.number_of_nodes()

        # Compute the common numerator term log(Z * Pr(X = x ; theta))
        log_numerator = self._compute_log_ZProb(dataset_key, theta_unary, theta_pairwise)

        # Compute each denominator term for pseudo-log-likelihood using
        # log( exp(a) + exp(b) ) = max(a,b) + log(1 + exp(min(a,b) - max(a,b))),
        # where each term is of the form Z * Pr(X_i = b, X_{-i} = x_{-i} ; theta)
        log_denominators = [0.0] * n
        for i in range(n):
            terms = [
                self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, i, b)
                for b in [0, 1]
            ]
            log_denominators[i] = max(terms) + np.log(1 + np.exp(min(terms) - max(terms)))

        log_pseudo_likelihood = n * log_numerator - sum(log_denominators)
        assert log_pseudo_likelihood <= 0
        return log_pseudo_likelihood

    """
    Computes log( Z * Pr(X = x ; theta) )
    = log( exp(sum_i <unary> + sum_{i,j} <pairwise>)
    = sum_i <unary> + sum_{i,j} <pairwise>
    """
    def _compute_log_ZProb(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray) -> float:
        # Arbitrarily use i = 0 and b = statuses[i]. Any index i works.
        _, _, statuses = self.get_dataset(dataset_key)
        return self._compute_local_log_ZProb(dataset_key, theta_unary, theta_pairwise, 0, statuses[0])

    """
    Computes log( Z * Pr(X_i = b, X_{-i} = x_{-i} ; theta) )
    = log( exp(sum_i <unary> + sum_{i,j} <pairwise>)
    = sum_i <unary> + sum_{i,j} <pairwise>
    """
    def _compute_local_log_ZProb(self, dataset_key: str, theta_unary: np.ndarray, theta_pairwise: np.ndarray, i: int, b: int) -> float:
        G, covariates, statuses = self.get_dataset(dataset_key)
        n = G.number_of_nodes()
        assert 0 <= i and i <= n
        assert b == 0 or b == 1

        key = (i, b)
        if key not in self.memo_local_log_ZProb.keys():
            val = 0.0

            # Compute contributions from theta_unary
            unary_vectors = np.stack([
                AbstractJointProbabilityClass.f_unary(b, covariates[i])
                if y == i
                else AbstractJointProbabilityClass.f_unary(statuses[y], covariates[y])
                for y in range(n)
            ])
            val += np.sum(theta_unary @ unary_vectors.T)

            # Compute contributions from theta_pairwise
            pairwise_vectors1 = np.stack([
                AbstractJointProbabilityClass.f_pairwise(statuses[y], statuses[j], covariates[y], covariates[j])
                for y in range(n)
                for j in G.neighbors(y)
                if y != i and j != i and j > y
            ])
            val += np.sum(theta_pairwise @ pairwise_vectors1.T)
            if next(G.neighbors(i), None) is not None:
                pairwise_vectors2 = np.stack([
                    AbstractJointProbabilityClass.f_pairwise(b, statuses[j], covariates[i], covariates[j])
                    if i < j
                    else AbstractJointProbabilityClass.f_pairwise(statuses[j], b, covariates[j], covariates[i])
                    for j in G.neighbors(i)
                ])
                val += np.sum(theta_pairwise @ pairwise_vectors2.T)
            
            # Store
            self.memo_local_log_ZProb[key] = val
        return self.memo_local_log_ZProb[key]
