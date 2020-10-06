import logging
from itertools import combinations
from math import factorial
from typing import Any, Callable, Dict, List, Optional, Union, Sized, Tuple

from ipywidgets import widgets, HBox, VBox
import numpy as np
import pandas as pd
from pyemd import emd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .classification_bias import ClassificationBias
from .sensitive_attributes import SensitiveNamesDetector, sensitive_attr_concat_name
from ..metrics import CramersV, CategoricalLogisticR2
from ..dataset import categorical_or_continuous_values

logger = logging.getLogger(__name__)


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
                 detect_sensitive: bool = False, detect_hidden: bool = False):
        """FairnessScorer constructor.

        Args:
            df: Input DataFrame to be scored.
            sensitive_attrs: Given sensitive attributes.
            target: Target variable.
            n_bins: Number of bins for sensitive attributes/target to be binarized.
            detect_sensitive: Whether to try to detect sensitive attributes from the column names.
            detect_hidden: Whether to try to detect sensitive attributes from hidden correlations with other sensitive
                attributes.
        """
        self.df = df.copy()
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
            corr = self.other_correlations()
            for new_sensitive_attr, _, _ in corr:
                logger.info(f"Adding column '{new_sensitive_attr}' to sensitive_attrs.")
                self.add_sensitive_attr(new_sensitive_attr)

        if len(self.sensitive_attrs) == 0:
            logger.warning("No sensitive attributes detected. Fairness score will always be 0.")

        self.df.dropna(inplace=True)
        self.binarize_columns(self.df)
        self.values_str_to_list: Dict[str, List] = dict()
        self.names_str_to_list: Dict[str, List] = dict()

    @classmethod
    def init_detect_sensitive(cls, df: pd.DataFrame, target: str, n_bins: int = 5):
        sensitive_attrs = cls.detect_sensitive_attrs(list(df.columns))
        scorer = cls(df, sensitive_attrs, target, n_bins)
        return scorer

    @property
    def sensitive_attrs_and_target(self) -> List[str]:
        return np.concatenate((self.sensitive_attrs, [self.target]))

    def distributions_score(self, min_dist: float = 0.1, min_count: float = 50, weighted: bool = False,
                            mode: str = 'emd', max_combinations: Optional[int] = 3,
                            progress_callback: Callable[[int], None] = None) -> Tuple[float, pd.DataFrame]:
        """Returns the biases and fairness score by analyzing the distribution difference between
        sensitive variables and the target variable."""

        if len(self.sensitive_attrs) == 0:
            if progress_callback is not None:
                progress_callback(0)
                progress_callback(100)

            return 0., pd.DataFrame([], columns=['name', 'value', 'distance', 'count'])

        biases = []
        score = 0.
        count = 0

        max_combinations = min(max_combinations, len(self.sensitive_attrs)) \
            if max_combinations else len(self.sensitive_attrs)

        if progress_callback is not None:
            n = 0
            progress_callback(0)
            num_combinations = self.get_num_combinations(self.sensitive_attrs, max_combinations)

        # Compute biases for all combinations of sensitive attributes
        for k in range(1, max_combinations + 1):
            for sensitive_attr in combinations(self.sensitive_attrs, k):
                df_count = self.get_rates(list(sensitive_attr))

                if mode == 'diff':
                    df_dist = self.difference_distance(df_count)
                elif mode == 'emd':
                    df_dist = self.emd_distance(df_count)
                else:
                    raise ValueError(f"Given mode='{mode}' not supported")

                if weighted:
                    score += np.sum(df_dist['Distance'].abs() * df_dist['Count']) / df_dist['Count'].sum()
                else:
                    score += np.average(df_dist['Distance'].abs())

                biases.extend(self.format_bias(df_dist))
                count += 1

                if progress_callback is not None:
                    n += 1
                    progress_callback(round(n * 98.0 // num_combinations))

        df_biases = pd.DataFrame(biases, columns=['name', 'value', 'distance', 'count'])
        df_biases = df_biases[(df_biases['distance'] >= min_dist) & (df_biases['count'] >= min_count)].sort_values(
            'distance', ascending=False).reset_index(drop=True)
        df_biases = df_biases[df_biases['value'] != 'Total']
        df_biases['name'] = df_biases['name'].map(self.names_str_to_list)
        df_biases['value'] = df_biases['value'].map(self.values_str_to_list)

        score /= count

        if progress_callback is not None:
            progress_callback(100)

        return score, df_biases

    def classification_score(self, threshold: float = 0.05, classifiers: Dict[str, BaseEstimator] = None,
                             min_count: int = 100, max_combinations: Optional[int] = 3,
                             progress_callback: Callable[[int], None] = None) -> Tuple[float, pd.DataFrame]:
        """ Computes few classification tasks for different classifiers and evaluates their performance on
        sub-samples given by splitting the data-set into sensitive sub-samples."""

        if len(self.sensitive_attrs) == 0:
            if progress_callback is not None:
                progress_callback(0)
                progress_callback(100)

            return 0., pd.DataFrame([], columns=['name', 'value', 'distance', 'count'])

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
                    progress_callback(n * 100 // num_combinations)

        score = float(np.nanmean(clf_scores))
        score = 0. if np.isnan(score) else score

        df_biases = pd.DataFrame(biases, columns=['name', 'value', 'distance', 'count'])
        df_biases['name'] = df_biases['name'].map(self.names_str_to_list)
        df_biases['value'] = df_biases['value'].map(self.values_str_to_list)

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
                sensitive_attr_name = "({})".format(', '.join([str(sa) for sa in sensitive_attr_values]))
                if sensitive_attr_name not in self.values_str_to_list.keys():
                    self.values_str_to_list[sensitive_attr_name] = sensitive_attr_values
                name_col.append(sensitive_attr_name)

            df[name] = name_col
            df.drop(sensitive_attr, axis=1, inplace=True)

        df_count = df.groupby([name, self.target]).count()[['Count']]

        for t in df[self.target].unique():
            df_count.loc[('Total', t), 'Count'] = sum(df_count['Count'][:, t])

        sensitive_unique = df_count.index.get_level_values(0).unique()

        df_count['Rate'] = 0.
        for attr in sensitive_unique:
            df_count['Rate'][attr] = df_count['Count'][attr] / df_count['Count'][attr].sum()

        return df_count

    def format_bias(self, bias: pd.DataFrame) -> List[Dict[str, Any]]:
        fmt_bias = []

        for item in bias.iterrows():
            bias_i = dict()

            bias_i['name'] = bias.index.names[0]
            if bias.index.nlevels == 1:
                bias_i['value'] = item[0]
            elif bias.index.nlevels == 2:
                bias_i['value'] = item[0][0]
                bias_i[self.target] = item[0][1]
            else:
                raise NotImplementedError

            bias_i['distance'] = round(item[1]['Distance'], 3)
            bias_i['count'] = int(item[1]['Count'])

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

    def other_correlations(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """

        Args:
            threshold:
        Returns:
            List of Tuples containing (detected attr, sensitive_attr, correlation_value).

        """
        columns = list(filter(lambda c: c not in self.sensitive_attrs_and_target, self.df.columns))
        df = self.df.dropna()
        categorical, continuous = categorical_or_continuous_values(self.df[columns])
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

    @staticmethod
    def difference_distance(df_count: pd.DataFrame) -> pd.DataFrame:
        df_count = df_count.copy()
        df_count['Distance'] = 0.
        for idx in df_count.index:
            l0, l1 = idx
            df_count['Distance'][idx] = df_count['Rate'][idx] - df_count['Rate'][('Total', l1)]
        df_count.drop('Total', inplace=True)
        return df_count

    @staticmethod
    def emd_distance(df_count: pd.DataFrame) -> pd.DataFrame:
        emd_dist = []
        space = df_count.index.get_level_values(1).unique()

        for sensitive_value in df_count.index.get_level_values(0).unique():
            p_counts = df_count['Count'][sensitive_value].to_dict()
            q_counts = df_count['Count']['Total'].to_dict()

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

    def binarize_columns(self, df: pd.DataFrame):
        for col in self.sensitive_attrs_and_target:
            if df[col].dtype.kind in ('i', 'u', 'f') and df[col].nunique() > self.n_bins:
                df[col] = pd.qcut(df[col], q=self.n_bins, duplicates='drop').astype(str)
            else:
                df[col] = df[col].astype(str).fillna('NaN')

    @staticmethod
    def get_num_combinations(iterable: Sized, max_combinations: int) -> int:
        n = len(iterable)
        num_combinations = 0

        for r in range(1, max_combinations + 1):
            num_combinations += int(factorial(n) / factorial(n - r) / factorial(r))

        return num_combinations
