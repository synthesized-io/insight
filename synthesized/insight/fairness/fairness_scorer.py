from enum import Enum
import logging
from itertools import combinations
from math import factorial
from typing import Any, Callable, Dict, List, Optional, Union, Sized, Tuple

from ipywidgets import widgets, HBox, VBox
import numpy as np
import pandas as pd
from pyemd import emd
from scipy.stats import binom, ks_2samp
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .classification_bias import ClassificationBias
from .sensitive_attributes import SensitiveNamesDetector, sensitive_attr_concat_name
from ..metrics import CramersV, CategoricalLogisticR2
from ..dataset import categorical_or_continuous_values

logger = logging.getLogger(__name__)


class VariableType(Enum):
    Binary = 0
    Multinomial = 1
    Continuous = 2


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

    def __init__(self, df: pd.DataFrame, sensitive_attrs: Union[List[str], str, None], target: str, n_bins: int = 5,
                 target_n_bins: Optional[int] = None, detect_sensitive: bool = False, detect_hidden: bool = False,
                 positive_class: str = None):
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
        """

        if isinstance(sensitive_attrs, list):
            self.sensitive_attrs: List[str] = sensitive_attrs
        elif isinstance(sensitive_attrs, str):
            self.sensitive_attrs = [sensitive_attrs]
        elif sensitive_attrs is None:
            if detect_sensitive is False:
                raise ValueError("If no 'sensitive_attr' is given, 'detect_sensitive' must be set to True.")
            self.sensitive_attrs = []
        else:
            raise TypeError("Given type of 'sensitive_attrs' not valid.")

        self.target = target
        self.n_bins = n_bins
        self.target_n_bins = target_n_bins

        # Check sensitive attrs
        for sensitive_attr in self.sensitive_attrs:
            if sensitive_attr not in df.columns:
                logger.warning(f"Dropping attribute '{sensitive_attr}' from sensitive attributes as it's not found in"
                               f"given DataFrame.")
                self.sensitive_attrs.remove(sensitive_attr)

        if target not in df.columns:
            raise ValueError(f"Target variable '{target}' not found in the given DataFrame.")

        # Detect any sensitive column
        if detect_sensitive:
            columns = list(filter(lambda c: c not in self.sensitive_attrs_and_target, df.columns))
            new_sensitive_attrs = self.detect_sensitive_attrs(columns)
            for new_sensitive_attr in new_sensitive_attrs:
                logger.info(f"Adding column '{new_sensitive_attr}' to sensitive_attrs.")
                self.add_sensitive_attr(new_sensitive_attr)

        # Detect hidden correlations
        if detect_hidden:
            corr = self.other_correlations(df)
            for new_sensitive_attr, _, _ in corr:
                logger.info(f"Adding column '{new_sensitive_attr}' to sensitive_attrs.")
                self.add_sensitive_attr(new_sensitive_attr)

        if len(self.sensitive_attrs) == 0:
            logger.warning("No sensitive attributes detected. Fairness score will always be 0.")

        self.set_df(df, positive_class=positive_class)

        self.values_str_to_list: Dict[str, List] = dict()
        self.names_str_to_list: Dict[str, List] = dict()

    def set_df(self, df: pd.DataFrame, positive_class: str = None):
        if not all(c in df.columns for c in self.sensitive_attrs_and_target):
            raise ValueError("Given DF must contain all sensitive attributes and the target variable.")

        self.df = df.copy()
        other_columns = list(filter(lambda c: c not in self.sensitive_attrs_and_target, self.df.columns))
        self.df.drop(other_columns, axis=1)
        self.df.dropna(inplace=True)

        self.bin_sensitive_attr(self.df)
        self.target_variable_type = self.manipulate_target_variable(self.df)

        self.len_df = len(self.df)
        # Only set positive class for binary/multinomial, even if given.
        self.positive_class = None

        if self.target_variable_type in (VariableType.Binary, VariableType.Multinomial):
            self.target_vc = self.df[self.target].value_counts(normalize=True)
            if len(self.target_vc) <= 2:
                if positive_class is None:
                    # If target class is not given, we'll use minority class as usually it is the target.
                    self.positive_class = self.target_vc.idxmin()
                elif positive_class not in self.target_vc.keys():
                    raise ValueError(f"Given positive_class '{positive_class}' is not present in dataframe.")
                else:
                    self.positive_class = positive_class
        else:
            self.target_mean = self.df[self.target].mean()

    @classmethod
    def init_detect_sensitive(cls, df: pd.DataFrame, target: str, n_bins: int = 5):
        sensitive_attrs = cls.detect_sensitive_attrs(list(df.columns))
        scorer = cls(df, sensitive_attrs, target, n_bins)
        return scorer

    @property
    def sensitive_attrs_and_target(self) -> List[str]:
        return list(np.concatenate((self.sensitive_attrs, [self.target])))

    def distributions_score(self, mode: Optional[str] = None, alpha: float = 0.05,
                            min_dist: Optional[float] = None, min_count: Optional[int] = 50,
                            weighted: bool = False, max_combinations: Optional[int] = 3,
                            progress_callback: Callable[[int], None] = None) -> Tuple[float, pd.DataFrame]:
        """Returns the biases and fairness score by analyzing the distribution difference between
        sensitive variables and the target variable.

        Args:
            alpha: Maximum p-value to accept a bias
            min_dist: If set, any bias with smaller distance than min_dist will be ignored.
            min_count: If set, any bias with less samples than min_count will be ignored.
            weighted: Whether to weight the average of biases on the size of each sample.
            max_combinations: Max number of combinations of sensitive attributes to be considered.
            mode: Only used for multinomial target variable. Two modes are available, 'ovr' and 'emd', 'ovr'
                performs binary class with one-vs-rest, and 'emd' computes earth mover's distance.
            progress_callback: Progress bar callback.
        """

        if len(self.sensitive_attrs) == 0:
            if progress_callback is not None:
                progress_callback(0)
                progress_callback(100)

            return 0., pd.DataFrame([], columns=['name', 'value', 'target', 'distance', 'count'])

        biases = []
        max_combinations = min(max_combinations, len(self.sensitive_attrs)) \
            if max_combinations else len(self.sensitive_attrs)
        num_combinations = self.get_num_combinations(self.sensitive_attrs, max_combinations)

        if progress_callback is not None:
            n = 0
            progress_callback(0)

        # Compute biases for all combinations of sensitive attributes
        for k in range(1, max_combinations + 1):
            for sensitive_attr in combinations(self.sensitive_attrs, k):

                df_dist = self.calculate_distance(list(sensitive_attr), mode=mode, alpha=alpha)
                biases.extend(self.format_bias(df_dist))

                if progress_callback is not None:
                    n += 1
                    progress_callback(round(n * 98.0 / num_combinations))

        df_biases = pd.DataFrame(biases, columns=['name', 'value', 'target', 'distance', 'count'])
        df_biases = df_biases[df_biases['value'] != 'Total']

        if len(df_biases) == 0:
            score = 1.
        elif weighted:
            score = 1 - (df_biases['distance'].abs() * df_biases['count']).sum() / df_biases['count'].sum()
        else:
            score = 1 - df_biases['distance'].abs().mean()

        if min_dist is not None:
            df_biases = df_biases[df_biases['distance'].abs() >= min_dist]
        if min_count is not None:
            df_biases = df_biases[df_biases['count'] >= min_count]

        df_biases = df_biases.reindex(df_biases['distance'].abs().sort_values(ascending=False).index)\
            .reset_index(drop=True)

        df_biases['name'] = df_biases['name'].apply(
            lambda x: self.names_str_to_list[x] if x in self.names_str_to_list else x)
        df_biases['value'] = df_biases['value'].map(
            lambda x: self.values_str_to_list[x] if x in self.values_str_to_list else x)

        if progress_callback is not None:
            progress_callback(100)

        return score, df_biases

    def calculate_distance(self, sensitive_attr: List[str], mode: Optional[str] = None,
                           alpha: float = 0.05) -> pd.DataFrame:
        """Check input values and decide which type of distance is computed for each case."""

        if self.target_variable_type == VariableType.Binary:
            df_dist = self.difference_distance(sensitive_attr, alpha=alpha)
        elif self.target_variable_type == VariableType.Multinomial:
            if mode is not None and mode == 'ovr':
                df_dist = self.difference_distance(sensitive_attr, alpha=alpha)
            elif mode is None or mode == 'emd':
                df_dist = self.emd_distance(sensitive_attr)
            else:
                raise ValueError(f"Given mode '{mode}' not recognized.")
        elif self.target_variable_type == VariableType.Continuous:
            df_dist = self.ks_distance(sensitive_attr)
        else:
            raise ValueError("Target variable type not supported")

        return df_dist

    def classification_score(self, threshold: float = 0.05, classifiers: Dict[str, BaseEstimator] = None,
                             min_count: Optional[int] = 50, max_combinations: Optional[int] = 3,
                             progress_callback: Callable[[int], None] = None) -> Tuple[float, pd.DataFrame]:
        """ Computes few classification tasks for different classifiers and evaluates their performance on
        sub-samples given by splitting the data-set into sensitive sub-samples.

        Args:
            threshold: If set, any bias with smaller outcome than threshold will be ignored.
            classifiers: Dictionary of classifiers to perform tasks on.
            min_count: If set, any bias with less samples than min_count will be ignored.
            max_combinations: Max number of combinations of sensitive attributes to be considered.
            progress_callback: Progress bar callback.
        """

        if len(self.sensitive_attrs) == 0:
            if progress_callback is not None:
                progress_callback(0)
                progress_callback(100)

            return 0., pd.DataFrame([], columns=['name', 'value', 'distance', 'count'])

        if self.target_variable_type != VariableType.Binary:
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

        if progress_callback is not None:
            n = 0
            progress_callback(0)
            num_combinations = self.get_num_combinations(self.sensitive_attrs, max_combinations)

        # Compute biases for all combinations of sensitive attributes
        for k in range(1, max_combinations + 1):
            for sensitive_attr in combinations(self.sensitive_attrs, k):
                cb = ClassificationBias(self.df, list(sensitive_attr), self.target, min_count=min_count)
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

    def get_sensitive_attrs(self) -> List[str]:
        return self.sensitive_attrs

    def set_sensitive_attrs(self, sensitive_attrs: List[str]):
        self.sensitive_attrs = sensitive_attrs

    def add_sensitive_attr(self, sensitive_attr: str):
        self.sensitive_attrs.append(sensitive_attr)

    def get_rates(self, sensitive_attr: List[str]) -> pd.DataFrame:
        df = self.df.copy()
        df['Count'] = 0
        name = sensitive_attr_concat_name(sensitive_attr)
        self.names_str_to_list[name] = sensitive_attr

        if type(sensitive_attr) == list and len(sensitive_attr) > 1:
            name_col = []
            for r in df.iterrows():
                sensitive_attr_values = [r[1][sa] for sa in sensitive_attr]
                sensitive_attr_str = "({})".format(', '.join([str(sa) for sa in sensitive_attr_values]))
                if sensitive_attr_str not in self.values_str_to_list.keys():
                    self.values_str_to_list[sensitive_attr_str] = sensitive_attr_values
                name_col.append(sensitive_attr_str)

            df[name] = name_col
            df.drop(sensitive_attr, axis=1, inplace=True)

        elif len(sensitive_attr) == 1:
            for sensitive_attr_str in df[name].unique():
                if sensitive_attr_str not in self.values_str_to_list.keys():
                    self.values_str_to_list[sensitive_attr_str] = [sensitive_attr_str]

        df_count = df.groupby([name, self.target]).count()[['Count']]

        for t in df[self.target].unique():
            df_count.loc[('Total', t), 'Count'] = sum(df_count['Count'][:, t])

        df_count['Rate'] = 0.
        rate_idx = list(df_count.columns).index('Rate')
        count_idx = list(df_count.columns).index('Count')

        attr_count_sum: Dict[str, int] = dict()
        for idx, row in df_count.iterrows():
            attr = idx[0]
            if attr not in attr_count_sum.keys():
                attr_count_sum[attr] = df_count['Count'][attr].sum()

            row[rate_idx] = row[count_idx] / attr_count_sum[attr]

        return df_count

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

    def cramers_v_score(self) -> float:
        df = self.df.copy()
        cramers_v = CramersV()
        score = 0.
        count = 0

        # Compute biases for all combinations of sensitive attributes
        for k in range(1, len(self.sensitive_attrs) + 1):
            for sensitive_attr_tuple in combinations(self.sensitive_attrs, k):
                sensitive_attr = list(sensitive_attr_tuple)

                if type(sensitive_attr) == list and len(sensitive_attr) > 1:
                    name = sensitive_attr_concat_name(sensitive_attr)
                    df[name] = df[sensitive_attr].apply(
                        lambda x: "({})".format(', '.join([str(x[attr]) for attr in sensitive_attr])),
                        axis=1)
                else:
                    name = sensitive_attr[0]

                score_i = cramers_v(df[name], df[self.target])
                if score_i is not None:
                    score += score_i
                    count += 1

        return score / count

    def other_correlations(self, df: pd.DataFrame, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """

        Args:
            df: Original dataframe to compute correlations between sensitive attributes and any other column.
            threshold: Correlation threshold to be considered a sensitive attribute.
        Returns:
            List of Tuples containing (detected attr, sensitive_attr, correlation_value).

        """
        columns = list(filter(lambda c: c not in self.sensitive_attrs_and_target, df.columns))
        df = df.dropna().copy()
        categorical, continuous = categorical_or_continuous_values(df[columns])
        cramers_v = CramersV()
        categorical_logistic_r2 = CategoricalLogisticR2()

        correlation_pairs = []
        for sensitive_attr in self.sensitive_attrs:
            for v_cat in categorical:
                corr = cramers_v(df[v_cat.name], df[sensitive_attr])
                if corr is not None and corr > threshold:
                    correlation_pairs.append((v_cat.name, sensitive_attr, corr))

            for v_cont in continuous:
                corr = categorical_logistic_r2(df[v_cont.name], df[sensitive_attr])
                if corr is not None and corr > threshold:
                    correlation_pairs.append((v_cont.name, sensitive_attr, corr))

        for detected_attr, sensitive_attr, corr in correlation_pairs:
            logger.warning(f"Columns '{detected_attr}' and '{sensitive_attr}' are highly correlated (corr={corr:.2f}), "
                           f"so probably '{detected_attr}' contains some sensitive information stored in "
                           f"'{sensitive_attr}'.")

        return correlation_pairs

    def ks_distance(self, sensitive_attr: List[str], alpha: float = 0.05) -> pd.DataFrame:
        groups = self.df.groupby(sensitive_attr).groups

        distances = []
        for group, idxs in groups.items():
            target_group = self.df.loc[self.df.index.isin(idxs), self.target]
            target_rest = self.df.loc[~self.df.index.isin(idxs), self.target]
            dist, pval = ks_2samp(target_group, target_rest)
            if pval < alpha:
                if np.mean(target_group) < self.target_mean:
                    dist = -dist

                distances.append([group, len(idxs), dist])

        name = sensitive_attr_concat_name(sensitive_attr)
        return pd.DataFrame(distances, columns=[name, 'Count', 'Distance']).set_index(name)

    def difference_distance(self, sensitive_attr: List[str], alpha: float = 0.05) -> pd.DataFrame:
        df_count = self.get_rates(list(sensitive_attr))

        if len(self.target_vc) <= 2:
            df_count = df_count[df_count.index.get_level_values(1) == self.positive_class]

        df_count['Distance'] = df_count.apply(self.get_row_distance, axis=1, alpha=alpha)
        df_count.dropna(inplace=True)
        if 'Total' in df_count.index.get_level_values(0):
            df_count.drop('Total', inplace=True)

        return df_count

    def get_row_distance(self, row: pd.Series, alpha: float = 0.05) -> float:
        if row.name[0] == 'Total':
            return 0.

        p = self.target_vc[row.name[1]]
        k = p * self.len_df
        k_i = row['Count']
        n_i = k_i / row['Rate']
        # Get p without the current subsample
        p_rest = (k - k_i) / (self.len_df - n_i)

        if k_i / n_i > p_rest:
            pval = 1 - binom.cdf(k_i - 1, n_i, p_rest)
        else:
            pval = binom.cdf(k_i, n_i, p_rest)

        if pval >= alpha:
            return np.nan

        return k_i / n_i - p

    def emd_distance(self, sensitive_attr: List[str]) -> pd.DataFrame:
        df_count = self.get_rates(list(sensitive_attr))

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

    @staticmethod
    def detect_sensitive_attrs(names: List[str]) -> List[str]:
        detector = SensitiveNamesDetector()
        names_dict = detector.detect_names_dict(names)
        if len(names_dict) > 0:
            logger.info("Sensitive columns detected: "
                        "{}".format(', '.join([f"'{k}' (bias type: {v})" for k, v in names_dict.items()])))

        return [attr for attr in names_dict.keys()]

    def check_box(self, columns: List[str], box_cols: int = 3) -> Tuple[Dict[str, Any], Any]:
        columns = list(filter(lambda c: c != self.target, columns))
        checks = dict()
        box = []
        for i in range(int(np.ceil(len(columns) / box_cols))):
            row = []
            for j in range(box_cols):
                idx = i * box_cols + j
                if idx >= len(columns):
                    break
                column = columns[idx]
                check = widgets.Checkbox(
                    value=True if column in self.sensitive_attrs else False,
                    description=column,
                    disabled=False
                )
                checks[column] = check
                row.append(check)
            box.append(HBox(row))
        VBox(box)

        return checks, VBox(box)

    def manipulate_target_variable(self, df: pd.DataFrame, inplace: bool = True) -> VariableType:
        """Check the target variable column, binned it if needed, and return target variable type"""
        if not inplace:
            df = df.copy()

        # Convert to numeric
        n_nans = df[self.target].isna().sum()
        col_num = pd.to_numeric(df[self.target], errors='coerce')
        if col_num.isna().sum() == n_nans:
            df[self.target] = col_num

        num_unique = df[self.target].nunique()

        # If it's numeric
        if df[self.target].dtype.kind in ('i', 'u', 'f'):
            if self.target_n_bins is None:
                # Even if it's numerical, well considered binary/multinomial if it has few unique values
                if num_unique <= 2:
                    self.target_n_bins = num_unique
                    return VariableType.Binary
                elif num_unique <= 5:
                    self.target_n_bins = num_unique
                    return VariableType.Multinomial
                else:
                    return VariableType.Continuous

            if num_unique > self.target_n_bins:
                df[self.target] = pd.cut(df[self.target], bins=self.target_n_bins, duplicates='drop').astype(str)

        else:
            df[self.target] = df[self.target].astype(str).fillna('nan')
            self.target_n_bins = num_unique

        return VariableType.Binary if num_unique == 2 else VariableType.Multinomial

    def bin_sensitive_attr(self, df: pd.DataFrame, inplace: bool = True):
        if not inplace:
            df = df.copy()

        for col in self.sensitive_attrs:
            # Try to convert it to numeric
            n_nans = df[col].isna().sum()
            col_num = pd.to_numeric(df[col], errors='coerce')
            if col_num.isna().sum() == n_nans:
                df[col] = col_num

            # If it's numeric, binned it
            if df[col].dtype.kind in ('i', 'u', 'f') and df[col].nunique() > self.n_bins:
                df[col] = pd.cut(df[col], bins=self.n_bins, duplicates='drop').astype(str)
            else:
                df[col] = df[col].astype(str).fillna('nan')

    @staticmethod
    def get_num_combinations(iterable: Sized, max_combinations: int) -> int:
        n = len(iterable)
        num_combinations = 0

        for r in range(1, max_combinations + 1):
            num_combinations += int(factorial(n) / factorial(n - r) / factorial(r))

        return num_combinations
