import pandas as pd

from typing import Sequence, Union, Dict, Optional, Any, Tuple
from sklearn.base import BaseEstimator

from synthesized.metadata.factory import MetaExtractor
from synthesized.model.factory import ModelFactory
from synthesized.model import DiscreteModel
from synthesized.insight.metrics.modelling_metrics import predictive_modelling_comparison


class UnifierModellingAssessor:
    """Class that provides the modelling scores of predicting the given target using given predictors
       Used to compare how model performs when trained on the unified dataset and evaluated on the original dataset
            as compared to when trained and tested on the original dataset
    Attributes:
        unified_df: The unified dataframe created out of the original dfs
        sub_dfs: Sequence of original dataframes
        orig_df: The original dataframe from which the unified data was procured

    """
    def __init__(self,
                 unified_df: pd.DataFrame,
                 sub_dfs: Optional[Union[Dict[str, pd.DataFrame], Sequence[pd.DataFrame]]] = None,
                 orig_df: Optional[pd.DataFrame] = None):
        """Create an instance of UnifierColumnsModellingMetrics.
        Either sub_dfs or orig_df should be provided
        """
        if sub_dfs is None and orig_df is None:
            raise ValueError("Both sub_dfs and orig_df can't be None")
        elif sub_dfs is not None and orig_df is not None:
            raise ValueError("Both sub_dfs and orig_df can't be provided.\
                              Expecting only one of them")

        self.unified_df = unified_df
        self.orig_df: pd.DataFrame = None
        self.sub_dfs: Union[Dict[str, pd.DataFrame], None] = None

        if orig_df is not None:
            self.orig_df = orig_df
        else:
            if isinstance(sub_dfs, dict):
                self.sub_dfs = sub_dfs
            elif sub_dfs is not None:
                self.sub_dfs = {f"df{i}": df for i, df in enumerate(sub_dfs)}

    def _remove_target_categories_not_in_unified(self,
                                                 original_df: pd.DataFrame,
                                                 target: str) -> pd.DataFrame:
        """For classification task, remove those target categories which are in original dataset(test)
        but not in unified_df(train)"""
        original_unique_cats = original_df[target].unique()
        unified_unique_cats = self.unified_df[target].unique()
        to_remove_cats = [cat for cat in original_unique_cats if cat not in unified_unique_cats]
        to_remove_idxs = original_df.index[original_df[target].isin(to_remove_cats)].tolist()
        original_df = original_df.drop(labels=to_remove_idxs, axis=0)
        return original_df

    def _get_predictive_modelling_scores(self,
                                         orig_df: pd.DataFrame,
                                         predictors: Sequence[str],
                                         target: str,
                                         model: Union[str, BaseEstimator],
                                         is_regression_task: bool):
        """Helper method to get the predictive modelling scores"""
        if target not in orig_df.columns:
            return None

        common_columns = orig_df.columns.intersection(self.unified_df.columns).tolist()
        available_predictors = [predictor for predictor in predictors if predictor in common_columns]
        if not available_predictors:
            return None

        original_df = orig_df.copy()
        if not is_regression_task:
            original_df = self._remove_target_categories_not_in_unified(original_df=original_df, target=target)

        original_df_score, unified_df_score, metric, _ = predictive_modelling_comparison(data=original_df,
                                                                                         synth_data=self.unified_df,
                                                                                         y_label=target,
                                                                                         x_labels=available_predictors,
                                                                                         model=model)

        return original_df_score, unified_df_score, metric

    def get_metric_score_for_unified_df(self,
                                        target: str,
                                        predictors: Sequence[str],
                                        model: Union[str, BaseEstimator]):

        """Computes test metric scores
        First score corresponds to the model trained on a part of original data and
            evaluated on the other part of original data,
        Second score corresponds to the model trained on the unified data and evaluated
            on a part of original data.
        This is to compare how close the unified data is to the original data in this regard.

        Args:
            target: The column to be predicted
            predictors: List of columns used to predict the target
            model: The regressor or classifier estimator

        Returns:
            A tuple of:-
                metric: Metric used for evaluation
                result: If sub dfs are provided then, return a dictionary mapping
                        the name or id of the dataset to the tumple if modelling scores,
                        else return the tuple of modelling scores on the original dataset
        """
        metric = None
        result: Union[Optional[Tuple[Any, Any]], Dict[Any, Optional[Tuple[Any, Any]]]] = None

        unified_df_meta = MetaExtractor.extract(df=self.unified_df)
        unified_df_model = ModelFactory()(unified_df_meta)
        target_col_model = unified_df_model[target]

        is_regression_task = True
        # Check predictor and decide if regression or classification task
        if isinstance(target_col_model, DiscreteModel):
            is_regression_task = False

        if self.orig_df is not None:
            out = self._get_predictive_modelling_scores(orig_df=self.orig_df,
                                                        predictors=predictors,
                                                        target=target,
                                                        model=model,
                                                        is_regression_task=is_regression_task)

            if out is not None:
                original_df_score, unified_df_score, metric = out[0], out[1], out[2]
                result = (original_df_score, unified_df_score)

        elif self.sub_dfs is not None:
            result = {}
            for name, df in self.sub_dfs.items():
                out = self._get_predictive_modelling_scores(orig_df=df,
                                                            predictors=predictors,
                                                            target=target,
                                                            model=model,
                                                            is_regression_task=is_regression_task)

                if out is not None:
                    original_df_score, unified_df_score, metric = out[0], out[1], out[2]
                    result[name] = (original_df_score, unified_df_score)
        return (metric, result)
