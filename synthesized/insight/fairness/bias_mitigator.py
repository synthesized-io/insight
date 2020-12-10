from collections import Counter
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
        self.fairness_scorer = fairness_scorer
        self.target = self.fairness_scorer.target
        self.sensitive_attrs = self.fairness_scorer.sensitive_attrs
        self.synthesizer = synthesizer
        self.cond_sampler = ConditionalSampler(synthesizer, min_sampled_ratio=0, synthesis_batch_size=262_144)

    @classmethod
    def from_dataframe(cls, synthesizer: Synthesizer, df: pd.DataFrame, target: str,
                       sensitive_attrs: List[str], n_bins: int = 5,
                       target_n_bins: Optional[int] = None) -> 'BiasMitigator':
        """Given a DataFrame, build a Bias Mitigator.

        Args:
            synthesizer: An underlying synthesizer.
            df: Pandas DataFrame containing the data to mitigate biases.
            target: Name of the column containing the target feature.
            sensitive_attrs: Given sensitive attributes.
            n_bins: Number of bins for sensitive attributes to be binned.
            target_n_bins: Number of bins for target to be binned, if None will use it as it is.
        """
        fairness_scorer = FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target,
                                         n_bins=n_bins, target_n_bins=(target_n_bins or n_bins),
                                         drop_dates=True)
        bias_mitigator = cls(synthesizer=synthesizer, fairness_scorer=fairness_scorer)
        return bias_mitigator

    def mitigate_biases(self, df: pd.DataFrame, n_biases: int = 5,
                        marginal_softener: Union[float, Tuple[float, float]] = 0.25,
                        alpha: float = 0.05, min_dist: Optional[float] = 0.01, get_independent: bool = False,
                        produce_nans: bool = False) -> pd.DataFrame:
        """Mitigate n biases

        Args:
            df: Pandas DataFrame containing the data to mitigate biases.
            n_biases: Number of biases to mitigate.
            marginal_softener: Whether to mitigate each bias completely (1.) or just a proportion of it. If float, both
                positive and negative biases will use the same value, if tuple will use (positive, negative).
            alpha: Maximum p-value to accept a bias.
            min_dist: If set, any bias with smaller distance than min_dist will be ignored.
            get_independent: Whether to only mitigate independent biases.
            produce_nans: Whether the output DF contains NaNs.
        """

        if self.fairness_scorer.target_variable_type == VariableType.Continuous:
            raise NotImplementedError("Bias mitigation is not supported for continuous target distributions.")

        df = df.copy()
        df_pre = self.fairness_scorer.preprocessor.transform(df)

        if produce_nans is False and df.isna().any(axis=None):
            # Impute nans to original data-set
            data_imputer = DataImputer(self.synthesizer)
            data_imputer.impute_nans(df, inplace=True)

        # Marginal softener
        if isinstance(marginal_softener, float):
            marginal_softener_pos = marginal_softener_neg = marginal_softener
        elif isinstance(marginal_softener, tuple) and \
                isinstance(marginal_softener[0], float) and isinstance(marginal_softener[1], float):
            marginal_softener_pos = marginal_softener[0]
            marginal_softener_neg = marginal_softener[1]
        else:
            raise ValueError(f"Given type '{type(marginal_softener)}' not understood.")

        if not (0 <= marginal_softener_pos <= 1) or not (0 <= marginal_softener_pos <= 1):
            raise ValueError(f"Marginal softener value must be in the interval [0., 1.], "
                             f"given ({marginal_softener_pos}, {marginal_softener_neg})")

        # Positive counts - Need to under-sample
        dist_biases_pos = self.get_biases(df, n_biases=n_biases, positive=True,
                                          get_independent=get_independent, alpha=alpha, min_dist=min_dist)
        if len(dist_biases_pos) > 0 and marginal_softener_pos > 0:
            samples_to_remove: List[int] = []
            for idx, row in dist_biases_pos.iterrows():
                marginal_counts = self.get_marginal_counts(df_pre, row, marginal_softener=marginal_softener_pos,
                                                           use_colons=False)
                marginals, counts = tuple(marginal_counts.items())[0]

                groups = df_pre.groupby(list(np.concatenate((row['name'], [self.target])))).groups
                samples_to_remove.extend(np.random.choice(groups[marginals], size=counts))

            df = df[~df.index.isin(samples_to_remove)]

        # Negative counts - Need to generate samples
        dist_biases_neg = self.get_biases(df, n_biases=n_biases, positive=True,
                                          get_independent=get_independent, alpha=alpha, min_dist=min_dist)
        if len(dist_biases_neg) > 0 and marginal_softener_neg > 0:
            dist_biases_neg['value_w_colons'] = dist_biases_neg.apply(self.add_colon_to_bias, axis=1)

            marginal_counts = Counter()
            for idx, row in dist_biases_neg.iterrows():
                marginal_counts = self.get_marginal_counts(df_pre, row, marginal_counts,
                                                           marginal_softener=marginal_softener_neg, use_colons=True)

            marginal_counts = Counter({tuple([str(k_i) for k_i in k]): v for k, v in marginal_counts.items()})
            marginal_keys = {str(col): list(map(str,
                                                filter(lambda x: x != 'nan', df_pre[col].unique()))
                                            ) for col in self.fairness_scorer.sensitive_attrs_and_target}

            df_cond = self.cond_sampler.synthesize_from_joined_counts(marginal_counts=marginal_counts,
                                                                      marginal_keys=marginal_keys,
                                                                      produce_nans=produce_nans)
            df = pd.concat((df, df_cond))
        return df

    def drop_given_biases(self, df: pd.DataFrame, biases: pd.DataFrame,
                          progress_callback: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
        """Given a dataframe and pre-computed biases, drop all rows from the dataframe affected by those biases.

        Args:
            df: Pandas DataFrame containing the data to mitigate biases.
            biases: Biases to be dropped.
            progress_callback: Progress bar callback.

        Returns:
            Unbiased DataFrame.
        """
        df = df.copy()
        df_pre = self.fairness_scorer.preprocessor.transform(df)

        idx_to_drop = []
        for i, (idx, bias) in enumerate(biases.iterrows()):
            idx_to_drop.extend(df_pre[np.all(df_pre[bias["name"]] == bias["value"], axis=1)].index)

            if progress_callback is not None:
                progress_callback(round(i * 98 / len(biases)))

        df.drop(index=idx_to_drop, inplace=True)
        return df

    def drop_biases(self, df: pd.DataFrame, mode: Optional[str] = None, alpha: float = 0.05,
                    min_dist: Optional[float] = 0.05, min_count: Optional[int] = 50,
                    progress_callback: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
        """Drop all rows affected by any bias.

        Args:
            df: Pandas DataFrame containing the data to mitigate biases.
            mode: Number of biases to mitigate each iteration.
            alpha: Maximum p-value to accept a bias.
            min_dist: If set, any bias with smaller distance than min_dist will be ignored.
            min_count: If set, any bias with less samples than min_count will be ignored.
            progress_callback: Progress bar callback.

        Returns:
            Unbiased DataFrame.
        """
        if progress_callback is not None:
            progress_callback(0)

        df = df.copy()

        dist_score, dist_biases = self.fairness_scorer.distributions_score(df, mode=mode, alpha=alpha,
                                                                           min_dist=min_dist, min_count=min_count)

        return self.drop_given_biases(df, dist_biases, progress_callback)

    def mitigate_biases_by_chunks(self, df: pd.DataFrame, chunk_size: int = 5,
                                  marginal_softener: Union[float, Tuple[float, float]] = 0.25, alpha: float = 0.05,
                                  strict: bool = False, strict_kwargs: Optional[Dict[str, Any]] = None,
                                  n_loops: int = 100, produce_nans: bool = False,
                                  progress_callback: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
        """Mitigate biases iteratively in chunks.

        Args:
            df: Pandas DataFrame containing the data to mitigate biases.
            chunk_size: Number of biases to mitigate each iteration.
            marginal_softener: Whether to mitigate each bias completely (1.) or just a proportion of it each iteration.
            alpha: Maximum p-value to accept a bias.
            strict: If true, will drop all rows affected by remaining biases.
            strict_kwargs: Keyword arguments for strict bias drop.
            n_loops: Maximum number of loops to try to mitigate biases.
            produce_nans:  Whether the output DF contains NaNs.
            progress_callback: Progress bar callback.

        Returns:
            Unbiased DataFrame.
        """
        if progress_callback is not None:
            progress_callback(0)

        df = df.copy()

        if produce_nans is False and df.isna().any(axis=None):
            # Impute nans to original data-set
            data_imputer = DataImputer(self.synthesizer)
            data_imputer.impute_nans(df, inplace=True)

        prev_idx = df.index

        for i in range(1, n_loops + 1):
            df = self.mitigate_biases(df, n_biases=chunk_size, marginal_softener=marginal_softener,
                                      alpha=alpha, produce_nans=produce_nans)

            if len(prev_idx) == len(df.index) and np.all(prev_idx == df.index):
                logger.info("There are no more biases to remove.")
                break

            prev_idx = df.index

            if progress_callback is not None:
                progress_callback(round(i * 98 / n_loops))

        if strict:
            strict_kwargs = strict_kwargs if strict_kwargs is not None else dict()
            df = self.drop_biases(df, **strict_kwargs)

        if progress_callback is not None:
            progress_callback(100)

        return df.sample(frac=1.).reset_index(drop=True)

    def resample_df(self, df: pd.DataFrame, num_rows: int, produce_nans: bool = False,
                    max_trials: int = 5) -> pd.DataFrame:
        """Given a dataframe, resample it while keeping the same sensitive groups proportion to not add/remove
         any bias.

         Args:
             df: Dataframe to resample.
             num_rows: Number of rows of output dataframe.
             produce_nans: Whether to include nans in the output.
             max_trials: Maximum number of trials to over-sample the dataframe.
         """
        df_pre = self.fairness_scorer.preprocessor.transform(df)

        for _ in range(max_trials):
            if len(df) >= num_rows:
                break

            n = 1.1 * (num_rows - len(df))
            counts = (df_pre.groupby(self.fairness_scorer.sensitive_attrs_and_target)[self.target]
                      .aggregate(count='count') / len(df_pre) * n).apply(round).astype(int)
            counts = counts[counts['count'] > 0]

            marginal_counts = Counter({tuple([str(k_i) for k_i in k]): v for k, v in counts['count'].to_dict().items()})
            marginal_keys = {str(col): list(map(str,
                                                filter(lambda x: x != 'nan', df_pre[col].unique()))
                                            ) for col in self.fairness_scorer.sensitive_attrs_and_target}

            df_synth = self.cond_sampler.synthesize_from_joined_counts(
                marginal_counts=marginal_counts, marginal_keys=marginal_keys, produce_nans=produce_nans)
            df = df.append(df_synth)

        if len(df) > num_rows:
            df = df.sample(num_rows)

        return df.sample(frac=1.).reset_index(drop=True)

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

    def get_marginal_counts(self, df_pre: pd.DataFrame, bias: pd.Series,
                            marginal_counts: Optional[Dict[Tuple[str, ...], int]] = None,
                            marginal_softener: float = 0.25, use_colons: bool = True) -> Dict[Tuple[str, ...], int]:

        if not 0 < marginal_softener <= 1.:
            raise ValueError(f"Value of marginal_softener must be in the interval (0., 1.], found {marginal_softener}")

        target_vc = df_pre[self.target].value_counts(normalize=True).to_dict()

        if marginal_counts is None:
            marginal_counts = Counter()

        values_w_colons = bias['value_w_colons'] if use_colons else bias['value']

        if use_colons:
            vc_this = df_pre.loc[np.all(
                [df_pre[col] == v for col, v in zip(self.sensitive_attrs, values_w_colons) if v != ':'],
                axis=0), self.target].value_counts()
        else:
            vc_this = df_pre.loc[np.all(
                [df_pre[col] == v for col, v in zip(bias['name'], values_w_colons) if
                 v != ':'],
                axis=0), self.target].value_counts()

        vc_this = Counter(vc_this.to_dict())

        target_value = bias['target']
        target_rate = target_vc[target_value]
        count = sum(vc_this.values())

        value_count = int(abs((target_rate * count - vc_this[target_value]) / (1 - target_rate)) * marginal_softener)
        marginal_counts[tuple(values_w_colons + [target_value])] += value_count

        return marginal_counts

    def get_biases(self, df: pd.DataFrame, n_biases: int, positive: bool, get_independent: bool = False,
                   mode: str = 'ovr', alpha: float = 0.05, min_dist: Optional[float] = 0.01) -> pd.DataFrame:
        """Compute biases and return a sample of them."""

        min_count = min(max(10, int(len(df) * 0.01)), 50)
        dist_score, dist_biases = self.fairness_scorer.distributions_score(
            df, mode=mode, alpha=alpha, min_dist=min_dist, min_count=min_count, condense_output=False)

        # Sort dist biases by distance * count
        sorted_idx = (dist_biases['distance'].abs() * dist_biases['count']).sort_values(ascending=False).index
        dist_biases = dist_biases.reindex(sorted_idx).reset_index(drop=True)

        if positive:
            dist_biases = dist_biases[dist_biases['distance'] > 0].copy()
        else:
            dist_biases = dist_biases[dist_biases['distance'] < 0].copy()

        if get_independent:
            dist_biases = self.get_top_independent_biases(dist_biases, n=n_biases)
        else:
            dist_biases = dist_biases.head(n_biases)

        return dist_biases

    @staticmethod
    def get_top_independent_biases(df_biases: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        affected: Dict[str, List[Any]] = dict()

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
