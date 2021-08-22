from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator

from ..check import ColumnCheck
from ..modelling import check_model_type
from .base import TwoDataFrameMetric
from .modelling_metrics import split_and_preprocess
from .privacy_cap_scorer import DistanceCAPScorer, GeneralizedCAPScorer


class AttributeInferenceAttackML(TwoDataFrameMetric):
    """Computes the privacy score between two dataframes using ML model

    The privacy score is calculated by fitting a ML model on the synthetic dataset. The fitted
    model is then used to calculate the privacy score of the sensitive column in the original dataset.

    Score lies in the range of 0 to 1, 0 means negligible privacy and 1 means absolute privacy.

    Args:
        model: Str or Base estimator
        sensitive_col: Sensitive column to be predicted
        predictors: List of predictor column names. If not provided, then all columns
                    except the sensitive column is treated as predictors

    Returns:
        Privacy score between 0 and 1
    """
    def __init__(self,
                 model: Union[BaseEstimator, str],
                 sensitive_col: str,
                 predictors: Optional[List[str]] = None) -> None:
        self.model = model
        self.sensitive_col = sensitive_col
        self.predictors = predictors

    def _privacy_score_categorical(self, true: pd.Series, pred: pd.Series) -> float:
        """Computes privacy score as the ratio of total true and pred values that do not match

        Args:
            true: True values of the sensitive attribute
            pred: Predicted values of the sensitive attribute

        Returns:
            The privacy score between 0 and 1
            0 means negligible privacy and 1 means absolute privacy.
        """
        num_rows = len(true)
        matches = 0
        for idx in range(num_rows):
            if true[idx] == pred[idx]:
                matches += 1

        return 1 - (matches / num_rows)

    def _privacy_score_numerical(self, true: pd.Series, pred: pd.Series, lp: float = 0.5) -> float:
        """Computes privacy score by measuring the distance between each set of true and pred values of
        the sensitive column using CDF

        Firstly, the norm is fit on the true data. Then, the distance for each pair of true and pred record
        is the difference in the percentile of these values according to cdf raised by pth power. The distance is
        averaged over all the records to give a final score.

        Args:
            true: True values of the sensitive column in the original dataset
            pred: Predicted values of the sensitive column of the original dataset

        Returns:
            Privacy score between 0 and 1
        """
        num_rows = len(true)
        norm.fit(true)
        dist = 0
        for idx in range(num_rows):
            percentiles = norm.cdf(np.array([pred[idx], true[idx]]))
            dist += abs(percentiles[0] - percentiles[1])**lp

        return dist / num_rows

    def _calculate_privacy_score(self,
                                 is_discrete_target: bool,
                                 estimator: BaseEstimator,
                                 x_test: pd.DataFrame,
                                 y_test: pd.Series) -> float:
        """Calculate the privacy score using fitted ML model

        Args:
            is_discrete_target: If target is discrete or not
            estimator: ML model
            x_test: Preprocessed predictors in the original dataset
            y_test: Sensitive target in the original dataset

        Returns:
            Privacy score between 0 and 1
        """
        if is_discrete_target:
            if hasattr(estimator, 'predict_proba'):
                f_proba_test = estimator.predict_proba(x_test)
                y_pred_test = np.argmax(f_proba_test, axis=1)
            else:
                y_pred_test = estimator.predict(x_test)
            privacy_score = self._privacy_score_categorical(y_test, y_pred_test)
        else:
            y_pred_test = estimator.predict(x_test)
            privacy_score = self._privacy_score_numerical(y_test, y_pred_test)

        return privacy_score

    def __call__(self, orig_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Computes the privacy score of the sensitive column of the original dataset based on the synthetic data

        Fits the ML model on the synthetic data. The trained model is used to predict the sensitive column
        of the original data. Finally, numerical or categorical privacy scorer is called depending on
        whether the sensitive column is a numerical or a categorical column.

        Args:
            orig_df: The original dataset
            synth_df: The synthetic dataset
            df_model: (Optional) The DataFrame model corresponding to synthetic dataset

        Returns:
            Privacy score between 0 and 1
        """
        if self.predictors is not None and not all(col in orig_df and col in synth_df for col in self.predictors):
            raise ValueError('All the predictors are not in original data and/or synthetic data')

        if not (self.sensitive_col in orig_df and self.sensitive_col in synth_df):
            raise ValueError('Sensitive column is not present in original data and/or synthetic data')

        if self.predictors is not None:
            orig_df = orig_df[self.predictors + [self.sensitive_col]]
            synth_df = synth_df[self.predictors + [self.sensitive_col]]
        else:
            self.predictors = synth_df.columns.intersection(orig_df.columns).tolist()

        estimator = None

        # Remove rows with response column value as NaN
        orig_df.dropna(subset=[self.sensitive_col], inplace=True)
        synth_df.dropna(subset=[self.sensitive_col], inplace=True)

        # Get the estimator
        check = ColumnCheck()
        is_discrete_target = True
        if check.categorical(orig_df[self.sensitive_col]) is True:
            estimator = check_model_type(model=self.model, copy_model=True, task='clf')
        elif check.continuous(orig_df[self.sensitive_col]) is True:
            is_discrete_target = False
            estimator = check_model_type(model=self.model, copy_model=True, task='rgr')
        else:
            raise ValueError(f"Can't understand y_label '{self.sensitive_col}' type.")

        x_train, y_train, x_test, y_test = split_and_preprocess(df=orig_df,
                                                                y_label=self.sensitive_col,
                                                                x_labels=self.predictors,
                                                                df_synth=synth_df)
        estimator.fit(x_train, y_train)
        privacy_score = self._calculate_privacy_score(is_discrete_target,
                                                      estimator, x_test, y_test)
        return privacy_score


class AttributeInferenceAttackCAP(TwoDataFrameMetric):
    """Computes the privacy score between two dataframes using CAP (Correct Attribution Probability) method

    Args:
        model: 'GeneralizedCAP' or 'DistanceCAP'
        sensitive_col: Name of the sensitive column
        predictors: List of predictor column names. If not provided, then all columns
                    except the sensitive column is treated as predictors

    Returns:
        Privacy score between 0 and 1
    """
    def __init__(self,
                 model: str,
                 sensitive_col: str,
                 predictors: Optional[List[str]] = None) -> None:
        self.model: Any = None
        if model == 'GeneralizedCAP':
            self.model = GeneralizedCAPScorer()
        elif model == 'DistanceCAP':
            self.model = DistanceCAPScorer()
        else:
            raise ValueError('Provided model string is not valid')
        self.predictors = predictors
        self.sensitive_col = sensitive_col

    def __call__(self, orig_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
        """Computes the privacy score of the sensitive column of the original dataset based on the synthetic data

        Fits the model on the synthetic data by mapping predictor key to sensitive attribute values of the synthetic dataset.
        Once fitted, the score method calculates the privacy of the sensitive attribute of the original dataset

        Args:
            orig_df: The original dataset
            synth_df: The synthetic dataset
            df_model: (Optional) The DataFrame model corresponding to synthetic dataset

        Return:
            Privacy score between 0 and 1
        """
        if self.predictors is not None and not all(col in orig_df and col in synth_df for col in self.predictors):
            raise ValueError('All the predictors are not in original data and/or synthetic data')

        if not (self.sensitive_col in orig_df and self.sensitive_col in synth_df):
            raise ValueError('Sensitive column is not present in original data and/or synthetic data')

        if self.predictors is None:
            predictors = [col for col in orig_df.columns if col != self.sensitive_col]
        else:
            predictors = self.predictors

        self.model.fit(synth_df[predictors], synth_df[self.sensitive_col])
        return self.model.score(orig_df[predictors], orig_df[self.sensitive_col])
