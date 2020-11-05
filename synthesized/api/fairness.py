from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator

from ..insight.fairness import FairnessScorer as _FairnessScorer


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
                 target_n_bins: int = 2, detect_sensitive: bool = False, detect_hidden: bool = False,
                 positive_class: str = None):
        """FairnessScorer constructor.

        Args:
            df: Input DataFrame to be scored.
            sensitive_attrs: Given sensitive attributes.
            target: Target variable.
            n_bins: Number of bins for sensitive attributes to be binarized.
            target_n_bins: Number of bins for target to be binarized.
            detect_sensitive: Whether to try to detect sensitive attributes from the column names.
            detect_hidden: Whether to try to detect sensitive attributes from hidden correlations with other sensitive
                attributes.
            positive_class: The sign of the biases depends on this class (positive biases have higher rate of this
                class). If not given, minority class will be used. Only used for binomial target variables.

        """
        self._fairness_scorer = _FairnessScorer(df=df, sensitive_attrs=sensitive_attrs, target=target, n_bins=n_bins,
                                                target_n_bins=target_n_bins, detect_sensitive=detect_sensitive,
                                                detect_hidden=detect_hidden, positive_class=positive_class)

    def distributions_score(self, max_pval: float = 0.05, min_dist: Optional[float] = 0.1,
                            min_count: Optional[int] = 50, weighted: bool = False, max_combinations: Optional[int] = 3,
                            progress_callback: Callable[[int], None] = None) -> Tuple[float, pd.DataFrame]:
        """ Returns the biases and fairness score by analyzing the distribution difference between
        sensitive variables and the target variable."""

        return self._fairness_scorer.distributions_score(max_pval=max_pval, min_dist=min_dist, min_count=min_count,
                                                         weighted=weighted, max_combinations=max_combinations,
                                                         progress_callback=progress_callback)

    def classification_score(self, threshold: float = 0.05, classifiers: Dict[str, BaseEstimator] = None,
                             min_count: int = 100, max_combinations: Optional[int] = 3,
                             progress_callback: Callable[[int], None] = None) -> Tuple[float, pd.DataFrame]:
        """ Computes few classification tasks for different classifiers and evaluates their performance on
        sub-samples given by splitting the data-set into sensitive sub-samples."""

        return self._fairness_scorer.classification_score(threshold=threshold, classifiers=classifiers,
                                                          min_count=min_count, max_combinations=max_combinations,
                                                          progress_callback=progress_callback)
