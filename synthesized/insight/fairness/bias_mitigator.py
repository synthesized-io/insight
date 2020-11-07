from collections import Counter
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fairness_scorer import FairnessScorer, VariableType
from ...common import Synthesizer
from ...complex import ConditionalSampler, DataImputer

logger = logging.getLogger(__name__)


class BiasMitigator:
    """Find distribution biases in data and generate the inverse data to mitigate these biases.
    """

    def __init__(self, synthesizer: Synthesizer, fairness_scorer: FairnessScorer):
        """Given a FairnessScorer, build a Bias Mitigator.

        Args:
            synthesizer: An underlying synthesizer.
            fairness_scorer: The fairness score to compute distribution biases from.
        """
        if fairness_scorer.target_variable_type == VariableType.Continuous:
            raise NotImplementedError("Bias mitigator not supported for continuous target distributions.")

        self.fairness_scorer = fairness_scorer
        self.target = self.fairness_scorer.target
        self.sensitive_attrs = self.fairness_scorer.sensitive_attrs
        self.synthesizer = synthesizer
        self.update_df()

    @classmethod
    def from_dataframe(cls, synthesizer: Synthesizer, df: pd.DataFrame, target: str,
                       sensitive_attrs: List[str], n_bins: int = 5) -> 'BiasMitigator':
        """Given a DataFrame, build a Bias Mitigator.

        Args:
            synthesizer: An underlying synthesizer.
            df: Pandas DataFrame containing the data to mitigate biases.
            target: Name of the column containing the target feature.
            sensitive_attrs: Given sensitive attributes.
            n_bins: Number of bins for sensitive attributes and target to be binned.
        """

        fairness_scorer = FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target,
                                         n_bins=n_bins, target_n_bins=n_bins)
        bias_mitigator = cls(synthesizer=synthesizer, fairness_scorer=fairness_scorer)
        return bias_mitigator

    def update_df(self, df: Optional[pd.DataFrame] = None):
        """If the dataframe changes, call this function to update DF-related attributes."""

        if df is not None:
            self.fairness_scorer.set_df(df)

        self.vc_all = self.fairness_scorer.target_vc
        self.len_df = self.fairness_scorer.len_df
        # if len(self.vc_all) > 2:
        #     raise ValueError('Bias Mitigation is not available for multinomial target distributions.')

        # self.max_class = self.vc_all.idxmax()
        # self.min_class = self.vc_all.idxmin()

    def mitigate_biases(self, df: pd.DataFrame, n_biases: int = 5, marginal_softener: float = 0.25,
                        alpha: float = 0.05, get_independent: bool = False,
                        produce_nans: bool = False) -> pd.DataFrame:
        """Mitigate n biases

        Args:
            df: Pandas DataFrame containing the data to mitigate biases.
            n_biases: Number of biases to mitigate.
            marginal_softener: Whether to mitigate each bias completely (1.) or just a proportion of it
            alpha: Maximum p-value to accept a bias.
            get_independent: Whether to only mitigate independent biases.
            produce_nans:  Whether the output DF contains NaNs.
        """

        df = df.copy()

        if produce_nans is False and df.isna().any(axis=None):
            # Impute nans to original data-set
            data_imputer = DataImputer(self.synthesizer)
            data_imputer.impute_nans(df, inplace=True)

        self.update_df(df)
        dist_score, dist_biases = self.fairness_scorer.distributions_score(mode='ovr', alpha=alpha)

        if len(dist_biases) == 0:
            return df

        dist_biases_pos = dist_biases[dist_biases['distance'] > 0].copy()
        dist_biases_neg = dist_biases[dist_biases['distance'] < 0].copy()
        if get_independent:
            dist_biases_pos = self.get_top_independent_biases(dist_biases_pos, n=n_biases)
            dist_biases_neg = self.get_top_independent_biases(dist_biases_neg, n=n_biases)
        else:
            dist_biases_pos = dist_biases_pos.head(n_biases)
            dist_biases_neg = dist_biases_neg.head(n_biases)

        # Positive counts - Need to under-sample
        if len(dist_biases_pos) > 0:
            samples_to_remove: List[int] = []
            for idx, row in dist_biases_pos.iterrows():
                marginal_counts = self.get_marginal_counts(row, marginal_softener=0.2, use_colons=False)
                marginals, counts = tuple(marginal_counts.items())[0]

                groups = self.fairness_scorer.df.groupby(list(np.concatenate((row['name'], [self.target])))).groups
                samples_to_remove.extend(np.random.choice(groups[marginals], size=counts))

            df = df[~df.index.isin(samples_to_remove)]

        # Negative counts - Need to generate samples
        if len(dist_biases_neg) > 0:
            dist_biases_neg['value_w_colons'] = dist_biases_neg.apply(self.add_colon_to_bias, axis=1)

            marginal_counts = Counter()
            for idx, row in dist_biases_neg.iterrows():
                marginal_counts = self.get_marginal_counts(row, marginal_counts, marginal_softener=marginal_softener,
                                                           use_colons=True)

            marginal_keys = {col: list(self.fairness_scorer.df[col].unique())
                             for col in self.fairness_scorer.sensitive_attrs_and_target}

            cond = ConditionalSampler(self.synthesizer, min_sampled_ratio=0, synthesis_batch_size=262_144)
            df_cond = cond.synthesize_from_joined_counts(marginal_counts=marginal_counts, produce_nans=produce_nans,
                                                         marginal_keys=marginal_keys, max_trials=100)

            df = pd.concat((df, df_cond)).sample(frac=1.).reset_index(drop=True)

        return df

    def mitigate_biases_by_chunks(self, df: pd.DataFrame, chunk_size: int = 5, marginal_softener: float = 0.25,
                                  alpha: float = 0.05, n_loops: int = 100, produce_nans: bool = False,
                                  progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Mitigate biases iteratively in chunks.

        Args:
            df: Pandas DataFrame containing the data to mitigate biases.
            chunk_size: Number of biases to mitigate each iteration.
            marginal_softener: Whether to mitigate each bias completely (1.) or just a proportion of it each iteration.
            alpha: Maximum p-value to accept a bias.
            n_loops: Maximum number of loops to try to mitigate biases.
            produce_nans:  Whether the output DF contains NaNs.
            progress_callback: Progress bar callback.
        """
        if progress_callback is not None:
            progress_callback(0)

        df = df.copy()

        if produce_nans is False and df.isna().any(axis=None):
            # Impute nans to original data-set
            data_imputer = DataImputer(self.synthesizer)
            data_imputer.impute_nans(df, inplace=True)

        prev_len = len(df)

        for i in range(1, n_loops + 1):
            df = self.mitigate_biases(df, n_biases=chunk_size, marginal_softener=marginal_softener,
                                      alpha=alpha, produce_nans=produce_nans)

            if prev_len == len(df):
                logger.info("There are no more biases to remove.")
                break

            prev_len = len(df)

            if progress_callback is not None:
                progress_callback(round(i * 98 // n_loops))

        if progress_callback is not None:
            progress_callback(100)

        return df

    def add_colon_to_bias(self, bias: pd.Series, add_target: bool = False) -> List[str]:
        bias_names = bias['name']
        bias_values = bias['value']
        len_bias_names = len(bias_names)

        i = 0
        bias_out = []
        for name in self.sensitive_attrs:
            if i < len_bias_names and name == bias_names[i]:
                bias_out.append(bias_values[i])
                i += 1
            else:
                bias_out.append(':')

        if 'target' in bias.keys() and add_target and bias['target'] != 'N/A':
            bias_out.append(bias['target'])

        return bias_out

    def get_marginal_counts(self, bias, marginal_counts: Dict[Tuple[str, ...], int] = None,
                            marginal_softener: float = 0.25, use_colons: bool = True) -> Dict[Tuple[str, ...], int]:

        if not 0 < marginal_softener <= 1.:
            raise ValueError(f"Value of marginal_softener must be in the interval (0., 1.], found {marginal_softener}")

        if marginal_counts is None:
            marginal_counts = Counter()

        values_w_colons = bias['value_w_colons'] if use_colons else bias['value']

        vc_this = self.fairness_scorer.df.loc[np.all(
            [self.fairness_scorer.df[col] == v for col, v in zip(self.sensitive_attrs, values_w_colons) if v != ':'],
            axis=0), self.target].value_counts()

        vc_this = Counter(vc_this.to_dict())

        target_value = bias['target']
        target_rate = self.vc_all[target_value]
        count = sum(vc_this.values())

        value_count = int(abs((target_rate * count - vc_this[target_value]) / (1 - target_rate)) * marginal_softener)
        marginal_counts[tuple(values_w_colons + [target_value])] += value_count

        return marginal_counts

    @staticmethod
    def get_top_independent_biases(df_biases, n=10):
        affected = dict()

        top_biases = df_biases.head(0)

        for idx, bias_i in df_biases.iterrows():
            name = bias_i['name']
            value = bias_i['value']

            for name_i, value_i in zip(name, value):
                if name_i in affected.keys():
                    break
                else:
                    affected[name_i] = [value_i]
            else:
                top_biases = top_biases.append(bias_i)

            if len(top_biases) >= n:
                break

        return top_biases
