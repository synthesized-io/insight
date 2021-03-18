import logging
from itertools import combinations
from math import factorial
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union

import numpy as np
import pandas as pd
from pyemd import emd
from scipy.stats import binom, ks_2samp
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .classification_bias import ClassificationBias
from .fairness_transformer import FairnessTransformer
from .sensitive_attributes import SensitiveNamesDetector, sensitive_attr_concat_name
from ..metrics import CategoricalLogisticR2, CramersV
from ...metadata import DataFrameMeta
from ...metadata.factory import MetaExtractor
from ...metadata.value import DateTime, String
from ...model import ContinuousModel, DiscreteModel, Model
from ...model.factory import ModelBuilder, ModelFactory
from ...model.models import Histogram
from ...transformer import SequentialTransformer

logger = logging.getLogger(__name__)


def _model_is_binary(model: DiscreteModel) -> bool:
    assert isinstance(model.categories, list)
    return len(model.categories) == 2


class FairnessScorer:
    """This class analyzes a given DataFrame, looks for biases and quantifies its fairness. There are two ways to
     compute this:
        * distributions_score: Returns the biases and fairness score by analyzing the distribution difference between
            sensitive variables and the target variable.
        * classification_score: Computes few classification tasks for different classifiers and evaluates their
            performance on those sub-samples given by splitting the data-set by sensitive sub-samples.

    Example:
        >>> data = pd.read_csv('data/templates/claim_prediction.csv')
        >>> sensitive_attributes = ["age", "sex", "children", "region"]
        >>> target = "insuranceclaim"

        >>> fairness_scorer = FairnessScorer(data, sensitive_attrs=sensitive_attributes, target=target)
        >>> dist_score, dist_biases = fairness_scorer.distributions_score()
    """
    def __init__(self, df: pd.DataFrame, sensitive_attrs: Optional[Union[List[str], str]], target: str, n_bins: int = 5,
                 target_n_bins: Optional[int] = 5, detect_sensitive: bool = False, detect_hidden: bool = False,
                 positive_class: Optional[str] = None, drop_dates: bool = True):
        """FairnessScorer constructor.

        Args:
            df: Input DataFrame to be scored.
            sensitive_attrs: Given sensitive attributes.
            target: Target variable.
            n_bins: Number of bins for sensitive attributes to be binned.
            target_n_bins: Number of bins for target to be binned, if None will use it as it is.
            detect_sensitive: Whether to try to detect sensitive attributes from the column names.
            detect_hidden: Whether to try to detect sensitive attributes from hidden correlations with other sensitive
                attributes.
            positive_class: The sign of the biases depends on this class (positive biases have higher rate of this
                class). If not given, minority class will be used. Only used for binomial target variables.
            drop_dates: Whether to ignore sensitive attributes containing dates.
        """

        if isinstance(sensitive_attrs, list):
            self.sensitive_attrs = sensitive_attrs
        elif isinstance(sensitive_attrs, str):
            self.sensitive_attrs = [sensitive_attrs]
        elif sensitive_attrs is None:
            if detect_sensitive is False:
                raise ValueError("If no 'sensitive_attr' is given, 'detect_sensitive' must be set to True.")
            self.sensitive_attrs = []
        else:
            raise TypeError("Given type of 'sensitive_attrs' not valid.")

        assert isinstance(sensitive_attrs, list)

        self.drop_dates = drop_dates
        self.n_bins = n_bins
        self.target_n_bins = target_n_bins

        self.df_meta = MetaExtractor.extract(df)
        self.detect_other_sensitive(df, detect_sensitive=detect_sensitive, detect_hidden=detect_hidden)
        self.target, self.sensitive_attrs = self.validate_sensitive_attrs_and_target(target, sensitive_attrs, self.df_meta)

        if len(self.sensitive_attrs) == 0:
            logger.warning("No sensitive attributes detected. Fairness score will always be 0.")

        self.transformer = FairnessTransformer(sensitive_attrs=self.sensitive_attrs, df_meta=self.df_meta, target=self.target,
                                               n_bins=self.n_bins, target_n_bins=self.target_n_bins)

        self.transformer.fit(df)
        self.target_model = self._get_target_model(df)

        if positive_class is not None:
            if positive_class not in df[target].unique():
                raise ValueError("Positive class is not a unique value in given target.")
            else:
                self.positive_class: Optional[str] = positive_class
        else:
            self.positive_class = self.get_positive_class()

        self.values_str_to_list: Dict[str, List[str]] = dict()
        self.names_str_to_list: Dict[str, List[str]] = dict()

    def _get_target_model(self, df: pd.DataFrame) -> Model:
        target_transformer = SequentialTransformer(self.target, list(filter(lambda x: x.name == self.target, self.transformer)))
        target_transformer.fit(df)
        transformed_target = target_transformer(df[[self.target]])
        target_meta = MetaExtractor.extract(transformed_target)[self.target]
        target_model = ModelBuilder()(target_meta)
        assert isinstance(target_model, Model)
        return target_model.fit(df)

    @classmethod
    def init_detect_sensitive(cls, df: pd.DataFrame, target: str, n_bins: int = 5) -> 'FairnessScorer':
        """Create a new FairnessScorer and automatically detect sensitive attributes.
        Args:
            df: Input DataFrame to be scored.
            target: Target variable.
            n_bins: Number of bins for sensitive attributes to be binned.
        """
        sensitive_attrs = cls.detect_sensitive_attrs(list(df.columns), target=target)
        scorer = cls(df, sensitive_attrs, target, n_bins)
        return scorer

    @property
    def sensitive_attrs_and_target(self) -> List[str]:
        return list(np.concatenate((self.sensitive_attrs, [self.target])))

    def distributions_score(self, df: pd.DataFrame,
                            mode: str = 'emd', alpha: float = 0.05,
                            min_dist: Optional[float] = None, min_count: Optional[int] = 50,
                            weighted: bool = True, max_combinations: Optional[int] = 3, condense_output: bool = True,
                            progress_callback: Optional[Callable[[int], None]] = None) -> Tuple[float, pd.DataFrame]:
        """Returns the biases and fairness score by analyzing the distribution difference between
        sensitive variables and the target variable.

        Args:
            df: Dataframe to compute fairness.
            mode: Only used for multinomial target variable. Two modes are available, 'ovr' and 'emd', 'ovr'
                performs binary class with one-vs-rest, and 'emd' computes earth mover's distance.
            alpha: Maximum p-value to accept a bias
            min_dist: If set, any bias with smaller distance than min_dist will be ignored.
            min_count: If set, any bias with less samples than min_count will be ignored.
            weighted: Whether to weight the average of biases on the size of each sample.
            max_combinations: Max number of combinations of sensitive attributes to be considered.
            condense_output: Whether to return one row per group or one per group and target
            progress_callback: Progress bar callback.
        """

        df_pre = self.transformer(df)

        if len(self.sensitive_attrs) == 0 or len(df_pre) == 0 or len(df_pre.dropna()) == 0:
            if progress_callback is not None:
                progress_callback(0)
                progress_callback(100)

            return 1., pd.DataFrame([], columns=['name', 'value', 'target', 'distance', 'count'])

        biases = []
        max_combinations = min(max_combinations, len(self.sensitive_attrs)) \
            if max_combinations else len(self.sensitive_attrs)
        num_combinations = self.get_num_combinations(self.sensitive_attrs, max_combinations)

        n = 0
        if progress_callback is not None:
            progress_callback(0)

        # Compute biases for all combinations of sensitive attributes
        for k in range(1, max_combinations + 1):
            for sensitive_attr in combinations(self.sensitive_attrs, k):
                df_not_nan = df_pre[~(df_pre[list(sensitive_attr)] == 'nan').any(1)]
                if len(df_not_nan) == 0:
                    continue

                df_dist = self.calculate_distance(df_not_nan, list(sensitive_attr), mode=mode, alpha=alpha)
                biases.extend(self.format_bias(df_dist))

                if progress_callback is not None:
                    n += 1
                    progress_callback(round(n * 98.0 / num_combinations))

        df_biases = pd.DataFrame(biases, columns=['name', 'value', 'target', 'distance', 'count'])
        df_biases = df_biases[df_biases['value'] != 'Total']

        if min_dist is not None:
            df_biases = df_biases[df_biases['distance'].abs() >= min_dist]
        if min_count is not None:
            df_biases = df_biases[df_biases['count'] >= min_count]

        # Compute score
        if len(df_biases) == 0:
            return 1., pd.DataFrame([], columns=['name', 'value', 'target', 'distance', 'count'])
        elif weighted:
            score = 1 - (df_biases['distance'].abs() * df_biases['count']).sum() / df_biases['count'].sum()
        else:
            score = 1 - df_biases['distance'].abs().mean()

        if condense_output and isinstance(self.target_model, DiscreteModel):
            if _model_is_binary(self.target_model):
                df_biases = df_biases[df_biases['target'] == self.positive_class]

            elif mode == 'ovr':
                df_biases['distance'] = df_biases['distance'].abs()
                df_biases_out = df_biases.groupby(['name', 'value'], as_index=False).sum()
                df_biases_out['distance'] = df_biases.groupby(['name', 'value'], as_index=False).mean()['distance']
                df_biases_out['target'] = 'N/A'
                df_biases = df_biases_out

        # Sort values
        df_biases = df_biases.reindex(df_biases['distance'].abs().sort_values(ascending=False).index)\
            .reset_index(drop=True)

        df_biases['name'] = df_biases['name'].apply(
            lambda x: self.names_str_to_list[x] if x in self.names_str_to_list else x)
        df_biases['value'] = df_biases['value'].map(
            lambda x: self.values_str_to_list[x] if x in self.values_str_to_list else x)

        if progress_callback is not None:
            progress_callback(100)

        return score, df_biases

    def calculate_distance(self, df: pd.DataFrame, sensitive_attr: List[str], mode: str = 'emd',
                           alpha: float = 0.05) -> pd.DataFrame:
        """Check input values and decide which type of distance is computed for each case."""

        if isinstance(self.target_model, DiscreteModel):

            if _model_is_binary(self.target_model):
                df_dist = self.difference_distance(df, sensitive_attr, alpha=alpha)

            else:
                if mode == 'ovr':
                    df_dist = self.difference_distance(df, sensitive_attr, alpha=alpha)
                elif mode == 'emd':
                    df_dist = self.emd_distance(df, sensitive_attr)
                else:
                    raise ValueError(f"Given mode '{mode}' not recognized.")

        elif isinstance(self.target_model, ContinuousModel):
            df_dist = self.ks_distance(df, sensitive_attr)

        else:
            raise ValueError("Target variable type not supported")

        return df_dist

    def classification_score(self, df: Optional[pd.DataFrame] = None, threshold: float = 0.05,
                             classifiers: Optional[Dict[str, BaseEstimator]] = None,
                             min_count: Optional[int] = 50, max_combinations: Optional[int] = 3,
                             progress_callback: Optional[Callable[[int], None]] = None) -> Tuple[float, pd.DataFrame]:
        """ Computes few classification tasks for different classifiers and evaluates their performance on
        sub-samples given by splitting the data-set into sensitive sub-samples.

        Args:
            df: Dataframe to compute fairness.
            threshold: If set, any bias with smaller outcome than threshold will be ignored.
            classifiers: Dictionary of classifiers to perform tasks on.
            min_count: If set, any bias with less samples than min_count will be ignored.
            max_combinations: Max number of combinations of sensitive attributes to be considered.
            progress_callback: Progress bar callback.
        """
        df_pre = self.transformer(df)

        if len(self.sensitive_attrs) == 0:
            if progress_callback is not None:
                progress_callback(0)
                progress_callback(100)

            return 0., pd.DataFrame([], columns=['name', 'value', 'distance', 'count'])

        if not isinstance(self.target_model, DiscreteModel):
            raise NotImplementedError("Classification score only available for Binary target distributions.")

        clf_scores = []

        if classifiers is None:
            classifiers = {
                'LogisticRegression': LogisticRegression(max_iter=500),
                'RandomForest': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=1000)
            }

        biases = []
        max_combinations = min(max_combinations, len(self.sensitive_attrs)) \
            if max_combinations else len(self.sensitive_attrs)

        n = 0
        num_combinations = self.get_num_combinations(self.sensitive_attrs, max_combinations)
        if progress_callback is not None:
            progress_callback(0)

        # Compute biases for all combinations of sensitive attributes
        for k in range(1, max_combinations + 1):
            for sensitive_attr in combinations(self.sensitive_attrs, k):
                cb = ClassificationBias(df_pre, list(sensitive_attr), self.target, min_count=min_count)
                clf_score, biases_i = cb.classifier_bias(threshold=threshold, classifiers=classifiers)
                clf_scores.append(clf_score)
                biases.extend(biases_i)

                if progress_callback is not None:
                    n += 1
                    progress_callback(round(n * 98 / num_combinations))

        score = float(np.nanmean(clf_scores))
        score = 0. if np.isnan(score) else score

        df_biases = pd.DataFrame(biases, columns=['name', 'value', 'distance', 'count'])
        df_biases['name'] = df_biases['name'].apply(
            lambda x: self.names_str_to_list[x] if x in self.names_str_to_list else x)
        df_biases['value'] = df_biases['value'].map(
            lambda x: self.values_str_to_list[x] if x in self.values_str_to_list else x)

        if progress_callback is not None:
            progress_callback(100)

        return score, df_biases

    def get_rates(self, df: pd.DataFrame, sensitive_attr: List[str]) -> pd.DataFrame:
        target = self.target

        # Get group counts & rates
        sensitive_group_target_counts = df.groupby(sensitive_attr + [target])[target].aggregate(Count='count')
        sensitive_group_size = df.groupby(sensitive_attr).size()
        sensitive_group_target_counts['Rate'] = sensitive_group_target_counts['Count'] / sensitive_group_size

        # Get total counts & rates
        target_totals = df.groupby(target)[target].aggregate(Count='count')
        target_totals['Rate'] = target_totals / len(df)
        target_totals = target_totals.set_index(pd.MultiIndex.from_tuples([('Total', a) for a in target_totals.index]))

        index = sensitive_group_target_counts.index.droplevel(-1)
        name = sensitive_attr_concat_name(sensitive_attr)
        self.names_str_to_list[name] = sensitive_attr

        if len(sensitive_attr) > 1:
            index_fmt = index.map(lambda sa: f"({', '.join([str(sa_i) for sa_i in sa])})").rename(name)
            sensitive_group_target_counts = sensitive_group_target_counts.droplevel(sensitive_attr)\
                .set_index(index_fmt, append=True).swaplevel()
            self.values_str_to_list = {**self.values_str_to_list, **{k: list(v) for k, v in zip(index_fmt, index)}}
        else:
            self.values_str_to_list = {**self.values_str_to_list, **{k: [k] for k in index}}

        return pd.concat((sensitive_group_target_counts, target_totals))

    def format_bias(self, bias: pd.DataFrame) -> List[Dict[str, Any]]:
        if len(bias) == 0:
            return []

        fmt_bias = []

        nlevels = bias.index.nlevels
        name = bias.index.names[0]
        target = self.positive_class if self.positive_class else 'N/A'

        for k, v in bias.to_dict('index').items():
            bias_i = dict()

            bias_i['name'] = name
            if nlevels == 1:
                bias_i['value'], bias_i['target'] = k, target
            elif nlevels == 2:
                bias_i['value'], bias_i['target'] = k

            bias_i['distance'] = v['Distance']
            bias_i['count'] = int(v['Count'])

            fmt_bias.append(bias_i)

        return fmt_bias

    def ks_distance(self, df: pd.DataFrame, sensitive_attr: List[str], alpha: float = 0.05) -> pd.DataFrame:
        # ignore rows which have nans in any of the given sensitive attrs
        groups = df.groupby(sensitive_attr).groups
        distances = []
        for sensitive_attr_values, idxs in groups.items():
            target_group = df.loc[df.index.isin(idxs), self.target]
            target_rest = df.loc[~df.index.isin(idxs), self.target]
            if len(target_group) == 0 or len(target_rest) == 0:
                continue

            dist, pval = ks_2samp(target_group, target_rest)
            if pval < alpha:
                if np.mean(target_group) < df[self.target].mean():
                    dist = -dist

                if isinstance(sensitive_attr_values, tuple) and len(sensitive_attr_values) > 1:
                    sensitive_attr_str = "({})".format(', '.join([str(sa) for sa in sensitive_attr_values]))
                    sensitive_attr_values = list(sensitive_attr_values)
                else:
                    sensitive_attr_str = sensitive_attr_values
                    sensitive_attr_values = [sensitive_attr_values]

                self.values_str_to_list[sensitive_attr_str] = sensitive_attr_values
                distances.append([sensitive_attr_str, len(idxs), dist])

        name = sensitive_attr_concat_name(sensitive_attr)
        self.names_str_to_list[name] = sensitive_attr

        return pd.DataFrame(distances, columns=[name, 'Count', 'Distance']).set_index(name)

    def difference_distance(self, df: pd.DataFrame, sensitive_attr: List[str], alpha: float = 0.05) -> pd.DataFrame:

        df_count = self.get_rates(df, list(sensitive_attr))

        len_df = len(df)
        target_vc = df[self.target].value_counts(normalize=True).to_dict()

        df_count['Distance'] = df_count.apply(self.get_row_distance, axis=1, alpha=alpha,
                                              len_df=len_df, target_vc=target_vc)
        df_count.dropna(inplace=True)
        if 'Total' in df_count.index.get_level_values(0):
            df_count.drop('Total', inplace=True)

        return df_count

    @staticmethod
    def get_row_distance(row: pd.Series, alpha: float, len_df: int, target_vc: Dict[str, float]) -> float:
        if row.name[0] == 'Total':
            return 0.

        assert target_vc is not None

        p = target_vc[row.name[1]]
        k = p * len_df
        k_i = row['Count']
        n_i = k_i / row['Rate']
        # Get p without the current subsample
        p_rest = (k - k_i) / (len_df - n_i)

        if k_i / n_i > p_rest:
            pval = 1 - binom.cdf(k_i - 1, n_i, p_rest)
        else:
            pval = binom.cdf(k_i, n_i, p_rest)

        if pval >= alpha:
            return np.nan

        return k_i / n_i - p

    def emd_distance(self, df: pd.DataFrame, sensitive_attr: List[str]) -> pd.DataFrame:
        df_count = self.get_rates(df, list(sensitive_attr))

        emd_dist = []
        space = df_count.index.get_level_values(1).unique()

        for sensitive_value in df_count.index.get_level_values(0).unique():
            p_counts = df_count['Count'][sensitive_value].to_dict()
            # Remove counts in current subsample
            q_counts = {k: v - p_counts.get(k, 0) for k, v in df_count['Count']['Total'].to_dict().items()}

            p = np.array([float(p_counts[x]) if x in p_counts else 0.0 for x in space])
            q = np.array([float(q_counts[x]) if x in q_counts else 0.0 for x in space])

            p /= np.sum(p)
            q /= np.sum(q)

            distance_space = 1 - np.eye(len(space))

            emd_dist.append({
                df_count.index.names[0]: sensitive_value,
                'Distance': emd(p, q, distance_space),
                'Count': df_count['Count'][sensitive_value].sum()
            })
        return pd.DataFrame(emd_dist).set_index(df_count.index.names[0])

    def validate_sensitive_attrs_and_target(
            self, target: str, sensitive_attrs: List[str], df_meta: DataFrameMeta
    ) -> Tuple[str, List[str]]:

        if target not in df_meta:
            raise ValueError(f"Target variable '{target}' not found in the given DataFrame.")

        if not all(col in df_meta for col in sensitive_attrs):
            raise KeyError("Sensitive attributes not present in DataFrame.")

        if isinstance(df_meta[target], String) and len(df_meta[target].categories) > np.sqrt(df_meta.num_rows):  # type: ignore
            raise ValueError("Unable to compute fairness. Target column has too many unique non-numeric values.")

        if isinstance(df_meta[target], DateTime):
            raise TypeError("Datetime target columns not supported.")

        for attr in sensitive_attrs:
            if isinstance(df_meta[attr], DateTime) and self.drop_dates:
                self.sensitive_attrs.remove(attr)

        # If target in sensitive_attrs, drop it
        if target in sensitive_attrs:
            sensitive_attrs.remove(self.target)

        for attr in sensitive_attrs:
            if isinstance(df_meta[attr], String) and len(df_meta[attr].categories) > np.sqrt(df_meta.num_rows):  # type: ignore
                sensitive_attrs.remove(attr)
                logging.info(f"Sensitive attribute '{attr}' dropped as it is a sampled value.")

        if len(sensitive_attrs) == 0:
            logger.warning("No sensitive attributes detected. Fairness score will always be 0.")

        return target, sensitive_attrs

    def detect_other_sensitive(self, df: pd.DataFrame, detect_sensitive: bool, detect_hidden: bool) -> None:
        # Detect other hidden sensitive attrs
        if detect_sensitive:
            columns = list(filter(lambda c: c not in self.sensitive_attrs + [self.target], df.columns))
            new_sensitive_attrs = self.detect_sensitive_attrs(columns)
            for new_sensitive_attr in new_sensitive_attrs:
                logger.info(f"Adding column '{new_sensitive_attr}' to sensitive_attrs.")
                self.sensitive_attrs.append(new_sensitive_attr)

        # Detect hidden correlations
        if detect_hidden:
            corr = self.other_correlations(df)
            for new_sensitive_attr, _, _ in corr:
                logger.info(f"Adding column '{new_sensitive_attr}' to sensitive_attrs.")
                self.sensitive_attrs.append(new_sensitive_attr)

    @staticmethod
    def detect_sensitive_attrs(names: List[str], target: Optional[str] = None) -> List[str]:
        if target:
            names = list(filter(lambda c: c != target, names))

        detector = SensitiveNamesDetector()
        names_dict = detector.detect_names_dict(names)
        if len(names_dict) > 0:
            logger.info("Sensitive columns detected: "
                        "{}".format(', '.join([f"'{k}' (bias type: {v})" for k, v in names_dict.items()])))

        return [attr for attr in names_dict.keys()]

    def other_correlations(self, df: pd.DataFrame, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """

        Args:
            df: Original dataframe to compute correlations between sensitive attributes and any other column.
            threshold: Correlation threshold to be considered a sensitive attribute.
        Returns:
            List of Tuples containing (detected attr, sensitive_attr, correlation_value).

        """
        columns = list(filter(lambda c: c not in self.sensitive_attrs + [self.target], df.columns))
        df = df.dropna().copy()
        dp = MetaExtractor.extract(df[columns])
        models = ModelFactory()(dp)
        cramers_v = CramersV()
        categorical_logistic_r2 = CategoricalLogisticR2()

        correlation_pairs = []
        for sensitive_attr in self.sensitive_attrs:
            for col, model in models.items():
                corr = (cramers_v(df[col], df[sensitive_attr], dp=dp, models=models)
                        or categorical_logistic_r2(df[col], df[sensitive_attr], dp=dp, models=models))
                if corr is not None and corr > threshold:
                    correlation_pairs.append((col, sensitive_attr, corr))
        for detected_attr, sensitive_attr, corr in correlation_pairs:
            logger.warning(f"Columns '{detected_attr}' and '{sensitive_attr}' are highly correlated (corr={corr:.2f}), "
                           f"so probably '{detected_attr}' contains some sensitive information stored in "
                           f"'{sensitive_attr}'.")

        return correlation_pairs

    @staticmethod
    def get_num_combinations(iterable: Sized, max_combinations: int) -> int:
        n = len(iterable)
        num_combinations = 0

        for r in range(1, max_combinations + 1):
            num_combinations += int(factorial(n) / factorial(n - r) / factorial(r))

        return num_combinations

    def get_positive_class(self) -> Optional[str]:
        # Only set positive class for binary/multinomial, even if given.

        if isinstance(self.target_model, Histogram) and _model_is_binary(self.target_model):
            # If target class is not given, we'll use minority class as usually it is the target.
            assert self.target_model.categories is not None
            class_probabilites: Dict[str, float] = {x: self.target_model.probability(x) for x in self.target_model.categories}
            return min(class_probabilites, key=lambda x: class_probabilites.get(x, 0))

        else:
            return None
