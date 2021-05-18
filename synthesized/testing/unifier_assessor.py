import pandas as pd

from typing import Sequence, Tuple
from synthesized.insight.metrics import (ColumnComparisonVector, TwoColumnMetric, TwoColumnMetricMatrix)


class UnifierAssessor:
    """Class that lets compare the quality of unified data to the original data sources"""

    def __init__(self, orig_dfs: Sequence[pd.DataFrame], unified_df: pd.DataFrame):
        """Create an instance of UtilityTesting.

        Args:
            orig_dfs: A Sequence of original dataframes
            unified_df: A unified dataframe created out of the original dfs
        """
        self.orig_dfs = orig_dfs
        self.unified_df = unified_df

    def get_first_order_metric_distances(self, metric: TwoColumnMetric) -> Sequence[pd.Series]:
        """Returns the comparison metrics of each column in the unified dataframe as compared
            to the distribution of that column in original dataframes

        Args:
            metric: The metric to be applied to compute distance

        Returns:
            results: A sequence of the series of the distance/metric
                     between the column in the unified dataframe and that column
                     in the original dataframe.
        """
        if metric.name is None:
            raise ValueError("Metric has no name.")
        metric_vector = ColumnComparisonVector(metric)
        results = []
        unified_df_len = len(self.unified_df)
        for df in self.orig_dfs:
            df_len = len(df)
            if df_len > unified_df_len:
                result = metric_vector(df_old=df.sample(unified_df_len), df_new=self.unified_df)
            else:
                result = metric_vector(df_old=df, df_new=self.unified_df.sample(df_len))
            if result is None or len(result.dropna()) == 0:
                result = (0., 0.)
            results.append(result)

        return results

    def get_filtered_metric_matrix(self, df, metric):
        metric_matrix = TwoColumnMetricMatrix(metric)
        matrix = metric_matrix(df)

        # remove those columns which are not used for computing correlation
        # using the given metrics
        for c in matrix.columns:
            if matrix.loc[:, c].isna().all() and matrix.loc[c, :].isna().all():
                matrix.drop(c, axis=1, inplace=True)
                matrix.drop(c, axis=0, inplace=True)
        return matrix

    def get_second_order_metric_matrices(self, metric: TwoColumnMetric) -> Tuple[pd.DataFrame, Sequence[pd.DataFrame]]:
        """Gets the correlation matrices for the unified data and the original dataframes

        Args:
            metric: The metric to be applied to compute correlation

        Returns:
            results: A tuple of correlation matrix of the unified dataframe columns and the sequence of
                     the correlation matrices of the original dataframes.
        """
        if metric.name is None:
            raise ValueError("Metric has no name.")

        matrix_orig_dfs_list = []
        matrix_unified_df = self.get_filtered_metric_matrix(self.unified_df, metric)
        for df in self.orig_dfs:
            matrix_orig_dfs_list.append(self.get_filtered_metric_matrix(df, metric))

        return matrix_unified_df, matrix_orig_dfs_list
